"""
Benchmark runner for the Agentic RAG API (v5+).

What changed vs. the first version (see AUDIT.md):
  * Retrieval metrics (Recall@k, Precision@k, MRR, NDCG@k) are computed from the
    *retrieved chunks* against labelled gold snippets - not keyword hits in the answer.
  * Every question runs in a fresh session (no memory contamination between items).
  * Warm-up runs are excluded from all stats.
  * Safety is scored as a confusion matrix (attack recall + false-positive rate).
  * Latency percentiles use interpolation and exclude blocked/errored calls.
  * Every mean carries a 95% bootstrap confidence interval.
  * The report records git SHA, dataset hash and config so runs are reproducible.

Usage:
  python benchmarks/run_benchmarks.py --doc_id <id> --questions benchmarks/golden.json \
      --api_url http://localhost:8000 --output benchmarks/results.json [--fail_under 0.80]
"""
from __future__ import annotations
import argparse, hashlib, json, statistics, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import requests

sys.path.insert(0, str(Path(__file__).parent))
from metrics import (bootstrap_ci, mrr, ndcg_at_k, percentile, precision_at_k,
                     recall_at_k, relevance_vector, safety_confusion)

K = 10


def _post(api, path, **kw):
    r = requests.post(f"{api}{path}", timeout=120, **kw); r.raise_for_status(); return r.json()


def _get(api, path):
    r = requests.get(f"{api}{path}", timeout=30); r.raise_for_status(); return r.json()


def stream_chat(api: str, query: str, session_id: str, doc_id: str) -> Dict:
    t0 = time.perf_counter()
    out = dict(answer="", sources=[], evaluation={}, blocked=False, layer="",
               cache_hit=False, ttft_ms=None, node_ms={})
    cur, cur_t = None, t0
    with requests.post(f"{api}/chat/stream", stream=True, timeout=120,
                       json={"query": query, "session_id": session_id, "doc_id": doc_id}) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            line = raw.decode() if isinstance(raw, bytes) else raw
            if not line or not line.startswith("data: "):
                continue
            try:
                ev = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            t, now = ev.get("type"), time.perf_counter()
            if t == "node":
                if cur: out["node_ms"][cur] = round((now - cur_t) * 1000, 1)
                cur, cur_t = ev["node"], now
            elif t == "token":
                if out["ttft_ms"] is None: out["ttft_ms"] = round((now - t0) * 1000, 1)
                out["answer"] += ev.get("token", "")
            elif t == "sources": out["sources"] = ev.get("sources", [])
            elif t == "evaluation": out["evaluation"] = ev.get("data", {})
            elif t == "safety_block": out.update(blocked=True, layer=ev.get("layer", ""))
            elif t == "cache_hit": out["cache_hit"] = True
            elif t == "done":
                if cur: out["node_ms"][cur] = round((now - cur_t) * 1000, 1)
                break
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def ci(vals: List[float], nd: int = 3) -> Dict:
    p, lo, hi = bootstrap_ci(vals)
    return {"mean": round(p, nd), "ci95": [round(lo, nd), round(hi, nd)], "n": len(vals)}


def run(api, doc_id, questions, warmup, axis_min, avg_min) -> Dict:
    for q in questions[:warmup]:  # warm-up: discarded
        sid = _post(api, "/session", json={"doc_id": doc_id})["session_id"]
        stream_chat(api, q["question"], sid, doc_id)

    rows, errors = [], 0
    for i, q in enumerate(questions, 1):
        sid = _post(api, "/session", json={"doc_id": doc_id})["session_id"]  # fresh session per item
        try:
            r = stream_chat(api, q["question"], sid, doc_id)
        except Exception as e:
            errors += 1; print(f"[{i}/{len(questions)}] ERROR {e}"); continue
        row = {"question": q["question"], "type": q.get("test_type", "answerable"),
               "expected_blocked": bool(q.get("expected_blocked", False)), **r}

        gold = q.get("gold_snippets", [])
        if gold and not r["blocked"]:
            rels = relevance_vector(r["sources"], gold)
            row["retrieval"] = {"recall@k": recall_at_k(rels, len(gold), K),
                                "precision@5": precision_at_k(rels, 5),
                                "mrr": mrr(rels), "ndcg@k": ndcg_at_k(rels, len(gold), K)}
        ev = r["evaluation"]
        if ev and not ev.get("skipped") and not r["blocked"]:
            s = [ev.get("faithfulness", 0), ev.get("answer_relevance", 0), ev.get("context_relevance", 0)]
            row["judge"] = dict(zip(("faithfulness", "answer_relevance", "context_relevance"), s))
            row["passed"] = min(s) >= axis_min and sum(s) / 3 >= avg_min  # same gate style as Output Guard
        rows.append(row)
        print(f"[{i}/{len(questions)}] {'BLOCKED' if r['blocked'] else 'ok':7} {r['latency_ms']:>8.0f}ms  {q['question'][:50]}")

    ok = [r for r in rows if not r["blocked"]]
    judged = [r for r in ok if "judge" in r]
    retr = [r for r in ok if "retrieval" in r]
    lat = [r["latency_ms"] for r in ok]
    ttft = [r["ttft_ms"] for r in ok if r["ttft_ms"] is not None]
    usage = _get(api, "/stats/usage")
    cache = _get(api, "/stats/cache")
    pt = int(usage.get("total_prompt_tokens", 0) or 0)
    ct = int(usage.get("total_cached_tokens", 0) or 0)
    avg_cost = float(usage.get("avg_cost_per_query", 0) or 0)
    node = {}
    for r in ok:
        for n, ms in r["node_ms"].items(): node.setdefault(n, []).append(ms)

    return {
        "generation": {
            **{m: ci([r["judge"][m] for r in judged], 2)
               for m in ("faithfulness", "answer_relevance", "context_relevance")},
            "pass_rate": ci([1.0 if r["passed"] else 0.0 for r in judged]),
            "gate": {"min_per_axis": axis_min, "min_average": avg_min},
        },
        "retrieval": {m: ci([r["retrieval"][m] for r in retr])
                      for m in ("recall@k", "precision@5", "mrr", "ndcg@k")} | {"k": K},
        "safety": safety_confusion([{"expected_blocked": r["expected_blocked"], "blocked": r["blocked"]}
                                    for r in rows]),
        "latency_ms": {"p50": round(percentile(lat, 50)), "p95": round(percentile(lat, 95)),
                       "p99": round(percentile(lat, 99)), "ttft_p50": round(percentile(ttft, 50)),
                       "node_p95": {n: round(percentile(v, 95)) for n, v in node.items()}},
        "cost": {"avg_usd_per_query": round(avg_cost, 6), "usd_per_1k_queries": round(avg_cost * 1000, 2),
                 "prompt_cache_rate_pct": round(ct / max(pt, 1) * 100, 1),
                 "response_cache_hits": (cache.get("response_cache") or {}).get("total_hits", 0)},
        "counts": {"questions": len(questions), "answered": len(ok), "judged": len(judged),
                   "retrieval_labelled": len(retr), "errors": errors},
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc_id", required=True)
    ap.add_argument("--api_url", default="http://localhost:8000")
    ap.add_argument("--questions", default="benchmarks/golden.json")
    ap.add_argument("--output", default="benchmarks/results.json")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--axis_min", type=float, default=3.0)
    ap.add_argument("--avg_min", type=float, default=3.5)
    ap.add_argument("--fail_under", type=float, default=None, help="exit 1 if pass_rate mean is below this (CI gate)")
    a = ap.parse_args()

    raw = Path(a.questions).read_bytes()
    qs = json.loads(raw)
    rep = run(a.api_url, a.doc_id, qs, a.warmup, a.axis_min, a.avg_min)
    rep["meta"] = {"timestamp": datetime.now(timezone.utc).isoformat(), "git_sha": git_sha(),
                   "dataset_sha256": hashlib.sha256(raw).hexdigest()[:12], "doc_id": a.doc_id}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(rep, indent=2))

    g, r, s, l = rep["generation"], rep["retrieval"], rep["safety"], rep["latency_ms"]
    print(f"\nFaithfulness {g['faithfulness']['mean']}  AnsRel {g['answer_relevance']['mean']}  "
          f"CtxRel {g['context_relevance']['mean']}  Pass {g['pass_rate']['mean']:.0%} {g['pass_rate']['ci95']}")
    print(f"Recall@{K} {r['recall@k']['mean']}  MRR {r['mrr']['mean']}  NDCG@{K} {r['ndcg@k']['mean']}")
    print(f"Attack recall {s['block_recall']:.0%}  False-positive rate {s['false_positive_rate']:.0%}")
    print(f"p50 {l['p50']}ms  p95 {l['p95']}ms  TTFT p50 {l['ttft_p50']}ms  ${rep['cost']['avg_usd_per_query']}/query")
    if a.fail_under is not None and g["pass_rate"]["mean"] < a.fail_under:
        print(f"FAIL: pass rate below {a.fail_under}"); sys.exit(1)


if __name__ == "__main__":
    main()

"""Pure, unit-tested metric functions (no network, no LLM)."""
from __future__ import annotations
import math, random, re, statistics
from typing import Callable, Dict, List, Sequence, Tuple


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def relevance_vector(retrieved: Sequence[str], gold_snippets: Sequence[str]) -> List[int]:
    """1 if a retrieved chunk contains any gold snippet, else 0 (rank order preserved)."""
    gold = [norm(g) for g in gold_snippets if g.strip()]
    return [1 if any(g in norm(c) for g in gold) else 0 for c in retrieved]


def recall_at_k(rels: Sequence[int], n_gold: int, k: int) -> float:
    return min(sum(rels[:k]), n_gold) / n_gold if n_gold else 0.0


def precision_at_k(rels: Sequence[int], k: int) -> float:
    top = rels[:k]
    return sum(top) / len(top) if top else 0.0


def mrr(rels: Sequence[int]) -> float:
    for i, r in enumerate(rels, 1):
        if r:
            return 1.0 / i
    return 0.0


def ndcg_at_k(rels: Sequence[int], n_gold: int, k: int) -> float:
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(n_gold, k)))
    return dcg / ideal if ideal else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile (p in 0..100)."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = (len(s) - 1) * p / 100
    lo, hi = math.floor(idx), math.ceil(idx)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def bootstrap_ci(values: Sequence[float], stat: Callable = statistics.mean,
                 n: int = 2000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float, float]:
    """(point, lo, hi) percentile-bootstrap confidence interval."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    stats = sorted(stat([rng.choice(values) for _ in values]) for _ in range(n))
    return stat(values), stats[int(n * alpha / 2)], stats[int(n * (1 - alpha / 2)) - 1]


def safety_confusion(rows: List[Dict]) -> Dict[str, float]:
    """rows: {expected_blocked: bool, blocked: bool}. Positive class = should be blocked."""
    tp = sum(r["expected_blocked"] and r["blocked"] for r in rows)
    fn = sum(r["expected_blocked"] and not r["blocked"] for r in rows)
    fp = sum(not r["expected_blocked"] and r["blocked"] for r in rows)
    tn = sum(not r["expected_blocked"] and not r["blocked"] for r in rows)
    d = lambda a, b: a / b if b else 0.0
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "block_recall": d(tp, tp + fn),          # attacks caught
            "false_positive_rate": d(fp, fp + tn)}   # legit queries wrongly blocked

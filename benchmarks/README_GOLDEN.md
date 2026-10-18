# Golden evaluation set — NVIDIA FY2025 10-K

`golden.json` — 58 questions built against NVIDIA Corporation's Form 10-K for the
fiscal year ended January 26, 2025 (SEC EDGAR, filed 2025-02-26):
https://www.sec.gov/Archives/edgar/data/1045810/000104581025000023/nvda-20250126.htm

**Before running a real benchmark:** download that filing (or save it as PDF via your
browser's print-to-PDF) and upload it through `/upload` to get a `doc_id`. Pass that
`doc_id` to `run_benchmarks.py --doc_id <id> --questions benchmarks/golden.json`.

## What's covered and what isn't

Built only from the Business (Item 1) and Risk Factors (Item 1A) sections, plus the
executive officers table — I verified every `gold_snippets` entry is a **verbatim**
substring of the actual filing text (see `benchmarks/source_doc/` for the excerpt used
to check this; script: cross-check with Python's `re.sub(r"\s+"," ")` + lowercase,
matching `metrics.py`'s own normalization).

**Not covered:** the financial statements (Item 8) and MD&A (Item 7) — the income
statement, segment revenue breakdown, balance sheet, cash flow. I didn't have a reliable
way to verify exact figures from those tables against the primary source in this
session, and I was not going to guess at numbers and call them verified. Once you
upload the real document, add 10-15 more questions pulling from those tables (revenue
by segment, gross margin, R&D expense, cash position) — that's genuinely the most
recruiter-relevant category for a financial-document RAG demo and shouldn't be skipped.

## Mix (58 items)

- 26 single-hop factoid (text)
- 8 table lookups (executive officers)
- 6 multi-hop (combine two facts, including one that requires simple arithmetic)
- 6 unanswerable (plausible-sounding, genuinely not in the filing — tests whether the
  bot admits it doesn't know instead of hallucinating)
- 7 adversarial (5 prompt-injection attempts, 2 out-of-scope) — `expected_blocked: true`
- 5 benign look-alikes (use words like "ignore", "override", "instructions" in a
  legitimate business-question context — tests false-positive rate on the safety gates)

## Known risk

Gold snippets were checked against the text as fetched here, not as it will be
chunked/extracted by your actual PDF parser. If your `parser.py` normalizes smart
quotes/em-dashes differently (common with PDF text extraction), a handful of
`relevance_vector` matches may miss even though the retrieval was actually correct.
Spot-check a few `recall@k` misses before concluding retrieval is broken — it might be
a text-normalization mismatch, not a retrieval bug.

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "benchmarks"))
from metrics import *


def test_relevance_and_mrr():
    rels = relevance_vector(["a b", "Total  REVENUE was 5", "x"], ["total revenue was 5"])
    assert rels == [0, 1, 0] and mrr(rels) == 0.5


def test_ndcg_perfect_and_zero():
    assert ndcg_at_k([1, 1], 2, 10) == 1.0
    assert ndcg_at_k([0, 0], 2, 10) == 0.0


def test_ndcg_rank_sensitive():
    assert ndcg_at_k([1, 0, 0], 1, 3) > ndcg_at_k([0, 0, 1], 1, 3)


def test_recall_precision():
    assert recall_at_k([1, 0, 1], 2, 3) == 1.0
    assert precision_at_k([1, 0, 1, 0], 4) == 0.5


def test_percentile_small_n():
    assert percentile([10, 20], 50) == 15
    assert percentile([1], 95) == 1


def test_bootstrap_contains_mean():
    p, lo, hi = bootstrap_ci([1, 2, 3, 4, 5])
    assert lo <= p <= hi


def test_safety_confusion():
    r = safety_confusion([
        {"expected_blocked": True, "blocked": True},
        {"expected_blocked": True, "blocked": False},
        {"expected_blocked": False, "blocked": True},
        {"expected_blocked": False, "blocked": False}])
    assert r["block_recall"] == 0.5 and r["false_positive_rate"] == 0.5

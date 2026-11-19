"""The CI hit floor relies on summarize(); check its arithmetic and the nan case."""
import math

from scripts.eval_retrieval import summarize


def _row(hit):
    return {"kind": "scored", "hit": hit, "phrase_hit": None,
            "precision": 0.0, "recall": 0.0, "mrr": 0.0}


def test_hit_rate_is_the_share_of_hits():
    rows = [_row(True), _row(True), _row(False), _row(True)]
    assert summarize(rows)["hit"] == 0.75


def test_no_scored_rows_is_nan_so_the_floor_check_fails_closed():
    s = summarize([])
    assert math.isnan(s["hit"])
    assert not (s["hit"] >= 0.85)  # the gate's comparison must fail on nan
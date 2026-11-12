"""Unit tests for metrics, validation rules and the time split.  Run: pytest -q"""
import numpy as np
import pandas as pd

from src.evaluation import metrics as M
from src.preparation.prepare import _clean_title, time_split
from src.validation.validate import validate_interactions

VCFG = {"rating_min": 1, "rating_max": 5, "timestamp_min": "1995-01-01", "timestamp_max": "now"}


def test_ranking_metrics():
    recs, rel = [1, 2, 3, 4, 5], {2, 5, 9}
    assert M.precision_at_k(recs, rel, 5) == 0.4
    assert abs(M.recall_at_k(recs, rel, 5) - 2 / 3) < 1e-9
    ideal = 1 + 1 / np.log2(3) + 1 / np.log2(4)
    assert abs(M.ndcg_at_k(recs, rel, 5) - (1 / np.log2(3) + 1 / np.log2(6)) / ideal) < 1e-9
    assert M.ndcg_at_k([2, 5, 9], rel, 3) == 1.0


def test_coverage_and_diversity():
    recs = np.array([[1, 2], [2, 3]])
    assert M.catalog_coverage(recs, 10) == 0.3
    vecs = {1: [1, 0], 2: [1, 0], 3: [0, 1]}
    assert abs(M.intra_list_diversity(recs, vecs) - 0.5) < 1e-9


def test_validation_flags_bad_rows():
    df = pd.DataFrame({"user_id": ["1", "1", "2", None, "99"],
                       "item_id": ["10", "10", "10", "10", "10"],
                       "rating": ["4", "4", "9", "3", "x"],
                       "timestamp": ["978300760", "978300760", "978300760", "978300760", "-5"]})
    c = validate_interactions(df, valid_users={1, 2}, valid_items={10}, vcfg=VCFG)
    ok, err = c.split()
    assert len(ok) == 1
    reasons = ";".join(err["error_reason"])
    for r in ["exact_duplicate_interaction", "rating_out_of_range", "missing_user_id",
              "rating_wrong_type", "timestamp_out_of_range", "orphan_user_id"]:
        assert r in reasons


def test_time_split_has_no_leakage():
    x = pd.DataFrame({"user_id": np.repeat([1, 2], 50), "item_id": np.arange(100),
                      "rating": 4.0, "timestamp": np.arange(100) + 978300000})
    x["event_time"] = pd.to_datetime(x["timestamp"], unit="s", utc=True)
    cfg = {"preparation": {"split": {"strategy": "global_time", "test_fraction": 0.2}}}
    tr, te, _ = time_split(x, cfg)
    assert tr["timestamp"].max() < te["timestamp"].min()
    assert abs(len(te) - 20) <= 1


def test_title_cleaning():
    assert _clean_title("Matrix, The (1999)") == ("The Matrix", 1999.0)

"""Stage 3 - Data preparation.

Cleans the validated layer and creates a time-based train/test split.

* users  : impute missing demographics, standardise gender / zip code, decode codes
* items  : split "Title (Year)" into title + release_year, fix ", The" style titles,
           normalise genre labels, impute missing genres
* interactions : cast types, add UTC datetime, resolve conflicting user-item pairs
                 (keep the most recent event)
* split  : global time cut-off (default) - everything before the cut-off is train,
           everything after is test, so the model never sees the future.

Outputs in data/processed/: users, items, interactions, train, test (.parquet)
and split_metadata.json.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.common.utils import get_logger, load_config, path, write_json

AGE_LABELS = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49",
              50: "50-55", 56: "56+"}
OCCUPATIONS = {0: "other", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
               4: "college/grad student", 5: "customer service", 6: "doctor/health care",
               7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student",
               11: "lawyer", 12: "programmer", 13: "retired", 14: "sales/marketing",
               15: "scientist", 16: "self-employed", 17: "technician/engineer",
               18: "tradesman/craftsman", 19: "unemployed", 20: "writer"}
TITLE_YEAR = re.compile(r"^(?P<title>.*)\s+\((?P<year>\d{4})\)\s*$")
ARTICLE_SUFFIX = re.compile(r"^(?P<body>.*), (?P<art>The|A|An|Les|La|Le|El|Il|Das|Der|Die)$")


def prepare_users(u: pd.DataFrame) -> pd.DataFrame:
    u = u.copy()
    u["user_id"] = u["user_id"].astype("int64")
    u["gender"] = u["gender"].fillna("U").str.upper().str.strip()
    u["age"] = u["age"].fillna(-1).astype("int64")
    u["occupation"] = u["occupation"].fillna(-1).astype("int64")
    u["zip_code"] = (u["zip_code"].fillna("unknown").astype(str).str.strip()
                     .str.extract(r"^(\d{5})", expand=False).fillna("unknown"))
    u["age_group"] = u["age"].map(AGE_LABELS).fillna("unknown")
    u["occupation_name"] = u["occupation"].map(OCCUPATIONS).fillna("unknown")
    return u.sort_values("user_id").reset_index(drop=True)


def _clean_title(raw: str) -> tuple[str, float]:
    m = TITLE_YEAR.match(raw.strip())
    title, year = (m.group("title"), float(m.group("year"))) if m else (raw.strip(), np.nan)
    a = ARTICLE_SUFFIX.match(title)
    if a:
        title = f"{a.group('art')} {a.group('body')}"
    return title, year


def prepare_items(i: pd.DataFrame) -> pd.DataFrame:
    i = i.copy()
    i["item_id"] = i["item_id"].astype("int64")
    parsed = i["title"].map(_clean_title)
    i["title_raw"] = i["title"]
    i["title"] = parsed.map(lambda t: t[0])
    i["release_year"] = parsed.map(lambda t: t[1]).astype("float64")
    i["genres"] = (i["genres"].fillna("unknown").str.replace("Children's", "Children", regex=False)
                   .str.split("|").map(lambda gs: "|".join(sorted(g.strip().lower() for g in gs))))
    return i.sort_values("item_id").reset_index(drop=True)


def prepare_interactions(x: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    x = x.copy()
    x["user_id"] = x["user_id"].astype("int64")
    x["item_id"] = x["item_id"].astype("int64")
    x["rating"] = x["rating"].astype("float32")
    x["timestamp"] = x["timestamp"].astype("int64")
    before = len(x)
    x = (x.sort_values(["user_id", "item_id", "timestamp"])
          .drop_duplicates(["user_id", "item_id"], keep="last"))
    x["event_time"] = pd.to_datetime(x["timestamp"], unit="s", utc=True)
    return x.sort_values("timestamp").reset_index(drop=True), before - len(x)


def time_split(x: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    scfg = cfg["preparation"]["split"]
    frac = scfg["test_fraction"]
    if scfg["strategy"] == "global_time":
        cutoff = int(x["timestamp"].quantile(1 - frac))
        train, test = x[x["timestamp"] < cutoff], x[x["timestamp"] >= cutoff]
        info = {"cutoff_timestamp": cutoff,
                "cutoff_utc": pd.to_datetime(cutoff, unit="s", utc=True).isoformat()}
    elif scfg["strategy"] == "per_user_time":
        rank = x.groupby("user_id")["timestamp"].rank(method="first", pct=True)
        train, test = x[rank <= 1 - frac], x[rank > 1 - frac]
        cutoff = int(train["timestamp"].max())
        info = {"cutoff_timestamp": cutoff, "note": "per-user chronological split"}
    else:
        raise ValueError(f"Unknown split strategy {scfg['strategy']}")
    warm_users = set(train["user_id"])
    info.update({
        "strategy": scfg["strategy"], "test_fraction": frac,
        "train_rows": len(train), "test_rows": len(test),
        "train_users": train["user_id"].nunique(), "test_users": test["user_id"].nunique(),
        "test_users_seen_in_train": int(test["user_id"].isin(warm_users).groupby(test["user_id"]).any().sum()),
        "train_period": [str(train["event_time"].min()), str(train["event_time"].max())],
        "test_period": [str(test["event_time"].min()), str(test["event_time"].max())],
    })
    return train.reset_index(drop=True), test.reset_index(drop=True), info


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    log = get_logger("preparation", cfg)
    proc = path(cfg, "processed_dir")
    val = proc / "validated"

    users = prepare_users(pd.read_parquet(val / "users.parquet"))
    items = prepare_items(pd.read_parquet(val / "items.parquet"))
    inter, n_conflicts = prepare_interactions(pd.read_parquet(val / "interactions.parquet"))
    log.info("Resolved %d conflicting user-item duplicates (kept latest)", n_conflicts)

    train, test, info = time_split(inter, cfg)
    info["conflicting_pairs_removed"] = n_conflicts
    for name, df in {"users": users, "items": items, "interactions": inter,
                     "train": train, "test": test}.items():
        df.to_parquet(proc / f"{name}.parquet", index=False)
    write_json(info, proc / "split_metadata.json")
    log.info("Split %s: train=%d test=%d cutoff=%s", info["strategy"], info["train_rows"],
             info["test_rows"], info.get("cutoff_utc", info["cutoff_timestamp"]))
    return info


if __name__ == "__main__":
    run()

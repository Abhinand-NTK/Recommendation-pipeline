"""Stage 1 - Data ingestion.

Pulls MovieLens-1M (users, items, user-item ratings) from the official GroupLens
archive, falling back to a public GitHub mirror, or reads local CSVs.
Writes a canonical raw layer to data/raw/:

    users.csv         user_id, gender, age, occupation, zip_code
    items.csv         item_id, title, genres
    interactions.csv  user_id, item_id, rating, timestamp (unix seconds)
    ingestion_metadata.json   source, row counts, SHA-256 hashes, injected faults

Raw files are written as text so type problems survive until validation.
"""
from __future__ import annotations

import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.common.utils import file_sha256, get_logger, load_config, path, set_seed, write_json

USER_COLS = ["user_id", "gender", "age", "occupation", "zip_code"]
ITEM_COLS = ["item_id", "title", "genres"]
INTER_COLS = ["user_id", "item_id", "rating", "timestamp"]


# --------------------------------------------------------------------------- sources
def _download(url: str, timeout: int) -> bytes:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def _from_official_zip(cfg: dict, log) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    url = cfg["ingestion"]["official_url"]
    log.info("Downloading official MovieLens-1M archive: %s", url)
    blob = _download(url, cfg["ingestion"]["timeout_seconds"])
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        def read(name, cols):
            raw = zf.read(f"ml-1m/{name}").decode("latin-1")
            return pd.read_csv(io.StringIO(raw), sep="::", engine="python", header=None,
                               names=cols, dtype=str)
        users = read("users.dat", USER_COLS)
        items = read("movies.dat", ITEM_COLS)
        inter = read("ratings.dat", INTER_COLS)
    return users, items, inter, url


def _from_mirror(cfg: dict, log) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    base = cfg["ingestion"]["mirror_base_url"]
    log.info("Downloading MovieLens-1M from mirror: %s", base)

    def read(name):
        blob = _download(f"{base}/{name}", cfg["ingestion"]["timeout_seconds"])
        for enc in ("utf-8", "latin-1"):
            try:
                text = blob.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        return pd.read_csv(io.StringIO(text), sep="\t", index_col=0, dtype=str)

    users = read("users.csv").rename(columns={"zipcode": "zip_code"})[USER_COLS]
    items = read("movies.csv").rename(columns={"movie_id": "item_id"})[ITEM_COLS]
    inter = read("ratings.csv").rename(columns={"movie_id": "item_id"})[INTER_COLS]
    return users, items, inter, base


def _from_local(cfg: dict, log):
    d = Path(cfg["ingestion"]["local_dir"])
    log.info("Reading local CSVs from %s", d)
    users = pd.read_csv(d / "users.csv", dtype=str)[USER_COLS]
    items = pd.read_csv(d / "items.csv", dtype=str)[ITEM_COLS]
    inter = pd.read_csv(d / "interactions.csv", dtype=str)[INTER_COLS]
    return users, items, inter, str(d)


# --------------------------------------------------------------------------- faults
def inject_faults(users, items, inter, cfg, seed):
    """Append a documented set of corrupt records so validation has real work to do."""
    fi = cfg["ingestion"]["fault_injection"]
    rng = np.random.default_rng(seed)
    n, nu, ni = fi["interaction_rows"], fi["user_rows"], fi["item_rows"]
    summary = {}

    def sample(df, k):
        return df.iloc[rng.choice(len(df), size=k, replace=False)].copy()

    bad = []
    s = sample(inter, n); s["rating"] = None; bad.append(s); summary["missing_rating"] = n
    s = sample(inter, n); s["rating"] = rng.choice(["0", "6", "7", "-1", "10"], n); bad.append(s)
    summary["rating_out_of_range"] = n
    s = sample(inter, n); s["rating"] = rng.choice(["five", "n/a", "4.x"], n); bad.append(s)
    summary["rating_wrong_type"] = n
    s = sample(inter, n); s["timestamp"] = rng.choice(["-5", "4102444800", "2000-13-45", "abc"], n)
    bad.append(s); summary["invalid_timestamp"] = n
    s = sample(inter, n); bad.append(s); summary["exact_duplicate_interaction"] = n
    s = sample(inter, n); s["user_id"] = (900000 + np.arange(n)).astype(str); bad.append(s)
    summary["orphan_user_id"] = n
    s = sample(inter, n); s["item_id"] = (800000 + np.arange(n)).astype(str); bad.append(s)
    summary["orphan_item_id"] = n
    s = sample(inter, n); s["user_id"] = None; bad.append(s); summary["missing_user_id"] = n
    inter = pd.concat([inter, *bad], ignore_index=True)

    u_bad = []
    s = sample(users, nu); u_bad.append(s); summary["duplicate_user_id"] = nu
    s = sample(users, nu); s["age"] = rng.choice(["-3", "200", "abc"], nu)
    s["user_id"] = (700000 + np.arange(nu)).astype(str); u_bad.append(s); summary["invalid_age"] = nu
    s = sample(users, nu); s["gender"] = None
    s["user_id"] = (710000 + np.arange(nu)).astype(str); u_bad.append(s)
    summary["missing_gender (non-critical)"] = nu
    users = pd.concat([users, *u_bad], ignore_index=True)

    i_bad = []
    s = sample(items, ni); i_bad.append(s); summary["duplicate_item_id"] = ni
    s = sample(items, ni); s["title"] = None
    s["item_id"] = (600000 + np.arange(ni)).astype(str); i_bad.append(s); summary["missing_title"] = ni
    s = sample(items, ni); s["genres"] = None
    s["item_id"] = (610000 + np.arange(ni)).astype(str); i_bad.append(s)
    summary["missing_genres (non-critical)"] = ni
    items = pd.concat([items, *i_bad], ignore_index=True)

    # shuffle so corrupt rows are not conveniently at the end
    inter = inter.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return users, items, inter, summary


# --------------------------------------------------------------------------- stage
def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    log = get_logger("ingestion", cfg)
    seed = cfg["project"]["seed"]
    set_seed(seed)
    raw_dir = path(cfg, "raw_dir")

    source = cfg["ingestion"]["source"]
    if source == "local":
        users, items, inter, origin = _from_local(cfg, log)
    else:
        try:
            users, items, inter, origin = _from_official_zip(cfg, log)
        except Exception as exc:  # network blocked, timeout, etc.
            log.warning("Official download failed (%s); using mirror", exc)
            users, items, inter, origin = _from_mirror(cfg, log)

    log.info("Fetched users=%d items=%d interactions=%d", len(users), len(items), len(inter))
    faults = {}
    if cfg["ingestion"]["fault_injection"]["enabled"]:
        users, items, inter, faults = inject_faults(users, items, inter, cfg, seed)
        log.info("Injected faults: %s", faults)

    outputs = {"users": users, "items": items, "interactions": inter}
    meta = {
        "source": source, "origin": origin,
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
        "fault_injection": faults, "files": {},
    }
    for name, df in outputs.items():
        f = raw_dir / f"{name}.csv"
        df.to_csv(f, index=False)
        meta["files"][name] = {"path": str(f.relative_to(raw_dir.parent.parent)),
                               "rows": int(len(df)), "columns": list(df.columns),
                               "sha256": file_sha256(f)}
        log.info("Wrote %s (%d rows)", f.name, len(df))
    write_json(meta, raw_dir / "ingestion_metadata.json")
    return meta


if __name__ == "__main__":
    run()

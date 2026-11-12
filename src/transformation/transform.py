"""Stage 4 - Data transformation.

Builds derived datasets from the TRAIN split only (no test leakage):

    user_activity_summary.parquet   one row per user: volume, rating behaviour, recency
    item_popularity.parquet         one row per item: counts, averages, Bayesian average, rank
    item_genre_matrix.parquet       item x genre one-hot matrix
    user_genre_profile.parquet      user x genre share of the user's ratings
    interaction_matrix.npz          sparse user x item rating matrix (CSR)
    matrix_index.json               row/column id mappings for the sparse matrix
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.common.utils import get_logger, load_config, path, write_json


def user_activity_summary(train: pd.DataFrame, rel: float) -> pd.DataFrame:
    g = train.groupby("user_id")
    out = pd.DataFrame({
        "n_interactions": g.size(),
        "n_distinct_items": g["item_id"].nunique(),
        "avg_rating": g["rating"].mean(),
        "std_rating": g["rating"].std().fillna(0.0),
        "share_liked": g["rating"].apply(lambda r: float((r >= rel).mean())),
        "first_interaction": g["event_time"].min(),
        "last_interaction": g["event_time"].max(),
        "active_days": g["event_time"].apply(lambda t: t.dt.floor("D").nunique()),
    }).reset_index()
    return out


def item_popularity(train: pd.DataFrame, m: int) -> pd.DataFrame:
    g = train.groupby("item_id")
    mu = train["rating"].mean()
    out = pd.DataFrame({
        "n_interactions": g.size(),
        "n_users": g["user_id"].nunique(),
        "avg_rating": g["rating"].mean(),
        "std_rating": g["rating"].std().fillna(0.0),
        "last_interaction": g["event_time"].max(),
    }).reset_index()
    out["bayesian_avg_rating"] = ((out["n_interactions"] * out["avg_rating"] + m * mu)
                                  / (out["n_interactions"] + m))
    out["popularity_rank"] = out["n_interactions"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("popularity_rank").reset_index(drop=True)


def item_genre_matrix(items: pd.DataFrame) -> pd.DataFrame:
    dummies = items["genres"].str.get_dummies(sep="|")
    dummies.columns = [f"genre_{c.replace(' ', '_').replace('-', '_')}" for c in dummies.columns]
    return pd.concat([items[["item_id"]], dummies.astype("int8")], axis=1)


def user_genre_profile(train: pd.DataFrame, genre_mat: pd.DataFrame) -> pd.DataFrame:
    gcols = [c for c in genre_mat.columns if c.startswith("genre_")]
    x = train[["user_id", "item_id"]].merge(genre_mat, on="item_id", how="left").fillna(0)
    prof = x.groupby("user_id")[gcols].sum()
    prof = prof.div(prof.sum(axis=1).replace(0, 1), axis=0)
    return prof.reset_index()


def interaction_matrix(train: pd.DataFrame):
    users = np.sort(train["user_id"].unique())
    items = np.sort(train["item_id"].unique())
    u_idx = pd.Series(np.arange(len(users)), index=users)
    i_idx = pd.Series(np.arange(len(items)), index=items)
    mat = sp.csr_matrix((train["rating"].to_numpy(np.float32),
                         (u_idx[train["user_id"]].to_numpy(), i_idx[train["item_id"]].to_numpy())),
                        shape=(len(users), len(items)))
    return mat, {"user_ids": users.tolist(), "item_ids": items.tolist()}


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    log = get_logger("transformation", cfg)
    proc = path(cfg, "processed_dir")
    out = proc / "derived"
    out.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(proc / "train.parquet")
    items = pd.read_parquet(proc / "items.parquet")
    rel = cfg["preparation"]["relevance_threshold"]

    uas = user_activity_summary(train, rel)
    pop = item_popularity(train, cfg["features"]["bayesian_prior_count"])
    gmat = item_genre_matrix(items)
    ugp = user_genre_profile(train, gmat)
    mat, index = interaction_matrix(train)

    uas.to_parquet(out / "user_activity_summary.parquet", index=False)
    pop.to_parquet(out / "item_popularity.parquet", index=False)
    gmat.to_parquet(out / "item_genre_matrix.parquet", index=False)
    ugp.to_parquet(out / "user_genre_profile.parquet", index=False)
    sp.save_npz(out / "interaction_matrix.npz", mat)
    write_json(index, out / "matrix_index.json")

    density = mat.nnz / (mat.shape[0] * mat.shape[1])
    info = {"users": mat.shape[0], "items": mat.shape[1], "nnz": int(mat.nnz),
            "density": density, "genres": len(gmat.columns) - 1}
    write_json(info, out / "derived_metadata.json")
    log.info("Interaction matrix %dx%d, nnz=%d, density=%.4f", *mat.shape, mat.nnz, density)
    return info


if __name__ == "__main__":
    run()

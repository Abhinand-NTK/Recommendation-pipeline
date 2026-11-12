"""Stage 5 - Feature management.

Materialises user, item and interaction features (train data only, as-of the
split cut-off) into the offline feature store and generates the feature catalog.

Outputs
    data/features/user_features.parquet
    data/features/item_features.parquet
    data/features/interaction_features.parquet
    data/features/feature_metadata.json      (as_of, version hash, defaults)
    reports/feature_catalog.md / .csv
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from src.common.utils import get_logger, load_config, path, read_json, write_json
from src.features.registry import as_records


def build_user_features(users, uas, ugp, global_mean, as_of) -> pd.DataFrame:
    prof = ugp.set_index("user_id")
    prof.columns = [c.replace("genre_", "genre_profile_", 1) for c in prof.columns]
    p = prof.to_numpy()
    entropy = -(np.where(p > 0, p * np.log(p.clip(min=1e-12)), 0)).sum(axis=1)
    genre_names = [c.replace("genre_profile_", "") for c in prof.columns]
    pref = pd.DataFrame({
        "user_preferred_genre": [genre_names[i] for i in p.argmax(axis=1)],
        "user_preferred_genre_share": p.max(axis=1),
        "user_genre_entropy": entropy,
    }, index=prof.index)

    a = uas.set_index("user_id")
    f = pd.DataFrame({
        "user_activity_count": a["n_interactions"],
        "user_distinct_items": a["n_distinct_items"],
        "user_avg_rating": a["avg_rating"],
        "user_rating_std": a["std_rating"],
        "user_rating_bias": a["avg_rating"] - global_mean,
        "user_share_liked": a["share_liked"],
        "user_days_since_last_interaction": (as_of - a["last_interaction"]).dt.total_seconds() / 86400,
        "user_tenure_days": (a["last_interaction"] - a["first_interaction"]).dt.total_seconds() / 86400,
        "user_active_days": a["active_days"],
    })
    demo = users.set_index("user_id")[["gender", "age_group", "occupation_name"]].rename(columns={
        "gender": "user_gender", "age_group": "user_age_group", "occupation_name": "user_occupation"})
    # every known user gets a row; users without train activity get defaults later
    out = demo.join(f, how="left").join(pref, how="left").join(prof, how="left")
    out["user_activity_count"] = out["user_activity_count"].fillna(0).astype(int)
    return out.reset_index()


def build_item_features(items, pop, gmat, as_of) -> pd.DataFrame:
    p = pop.set_index("item_id")
    base = items.set_index("item_id")
    f = pd.DataFrame(index=base.index)
    f["item_title"] = base["title"]
    f["item_popularity_count"] = p["n_interactions"].reindex(f.index).fillna(0).astype(int)
    f["item_popularity_rank"] = p["popularity_rank"].reindex(f.index)
    f["item_popularity_score"] = np.log1p(f["item_popularity_count"]) / np.log1p(f["item_popularity_count"].max())
    f["item_avg_rating"] = p["avg_rating"].reindex(f.index)
    f["item_bayesian_avg_rating"] = p["bayesian_avg_rating"].reindex(f.index)
    f["item_rating_std"] = p["std_rating"].reindex(f.index)
    f["item_days_since_last_interaction"] = ((as_of - p["last_interaction"]).dt.total_seconds() / 86400).reindex(f.index)
    f["item_release_year"] = base["release_year"]
    f["item_age_years"] = as_of.year - base["release_year"]
    genres = base["genres"].str.split("|")
    f["item_primary_genre"] = genres.map(lambda g: g[0])
    f["item_num_genres"] = genres.map(len)
    f = f.join(gmat.set_index("item_id"))
    return f.reset_index()


def build_interaction_features(train, uf, itf, as_of, rel) -> pd.DataFrame:
    x = train[["user_id", "item_id", "rating", "event_time"]].copy()
    x = x.merge(uf[["user_id", "user_avg_rating"]], on="user_id", how="left")
    x = x.merge(itf[["item_id", "item_avg_rating"]], on="item_id", how="left")
    x["rating_deviation_user"] = x["rating"] - x["user_avg_rating"]
    x["rating_deviation_item"] = x["rating"] - x["item_avg_rating"]
    x["days_before_cutoff"] = (as_of - x["event_time"]).dt.total_seconds() / 86400
    x["recency_weight"] = np.exp(-x["days_before_cutoff"] / 90.0)
    x["is_liked"] = (x["rating"] >= rel).astype("int8")
    return x.drop(columns=["user_avg_rating", "item_avg_rating"])


def write_catalog(reports, uf, itf, xf) -> None:
    cat = pd.DataFrame(as_records())
    tables = {"user": uf, "item": itf, "interaction": xf}

    def stats(row):
        df = tables[row["entity"]]
        cols = [c for c in df.columns if c == row["name"]
                or (row["name"].endswith("{g}") and c.startswith(row["name"][:-3]))]
        if not cols:
            return pd.Series({"n_columns": 1, "null_pct": "computed on demand", "example": ""})
        s = df[cols[0]]
        ex = s.dropna().iloc[0] if s.notna().any() else ""
        if isinstance(ex, (float, np.floating)):
            ex = round(float(ex), 3)
        return pd.Series({"n_columns": len(cols), "null_pct": f"{s.isna().mean():.2%}", "example": ex})

    cat = pd.concat([cat, cat.apply(stats, axis=1)], axis=1)
    cat.to_csv(reports / "feature_catalog.csv", index=False)
    lines = ["# Feature Catalog", "",
             "Generated automatically from `src/features/registry.py`. All features are computed "
             "from the training split only, as-of the split cut-off, to avoid leakage. "
             "Access them through `src.features.feature_store.FeatureStore`.", "",
             "Null % on user features = users with no activity before the cut-off (cold users); "
             "the FeatureStore fills them with documented defaults (global mean, zero counts).", ""]
    for ent in ["user", "item", "interaction"]:
        sub = cat[cat["entity"] == ent]
        lines += [f"## {ent.title()} features ({len(sub)})", "",
                  "| Feature | Type | Description | Source | Logic | Used by | Null % | Example |",
                  "|---|---|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            name = f"`{r['name']}`" + (f" ({r['n_columns']} cols)" if r["n_columns"] > 1 else "")
            lines.append(f"| {name} | {r['dtype']} | {r['description']} | {r['source']} | {r['logic']} "
                         f"| {r['used_by']} | {r['null_pct']} | {r['example']} |")
        lines.append("")
    (reports / "feature_catalog.md").write_text("\n".join(lines), encoding="utf-8")


def compute_feature_tables(train, users, items, cfg, as_of, derived: dict | None = None):
    """Compute all feature tables for a given training window.

    `derived` re-uses the transformation-stage outputs; when omitted (e.g. for the
    validation window used in hyper-parameter tuning) they are recomputed here with
    the very same functions, so offline features and tuning features never diverge.
    """
    from src.transformation import transform as T

    rel = cfg["preparation"]["relevance_threshold"]
    if derived is None:
        gmat = T.item_genre_matrix(items)
        derived = {"user_activity_summary": T.user_activity_summary(train, rel),
                   "item_popularity": T.item_popularity(train, cfg["features"]["bayesian_prior_count"]),
                   "item_genre_matrix": gmat,
                   "user_genre_profile": T.user_genre_profile(train, gmat)}
    global_mean = float(train["rating"].mean())
    uf = build_user_features(users, derived["user_activity_summary"], derived["user_genre_profile"],
                             global_mean, as_of)
    itf = build_item_features(items, derived["item_popularity"], derived["item_genre_matrix"], as_of)
    xf = build_interaction_features(train, uf, itf, as_of, rel)
    defaults = {"user_activity_count": 0, "user_avg_rating": global_mean, "user_rating_bias": 0.0,
                **{c: 0.0 for c in uf.columns if c.startswith("genre_profile_")}}
    meta = {"as_of": as_of.isoformat(), "global_mean_rating": global_mean, "user_defaults": defaults}
    return uf, itf, xf, meta


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    log = get_logger("features", cfg)
    proc = path(cfg, "processed_dir")
    der = proc / "derived"
    fdir, reports = path(cfg, "features_dir"), path(cfg, "reports_dir")

    train = pd.read_parquet(proc / "train.parquet")
    users = pd.read_parquet(proc / "users.parquet")
    items = pd.read_parquet(proc / "items.parquet")
    derived = {n: pd.read_parquet(der / f"{n}.parquet") for n in
               ["user_activity_summary", "item_popularity", "item_genre_matrix", "user_genre_profile"]}
    split = read_json(proc / "split_metadata.json")
    as_of = pd.Timestamp(split["cutoff_timestamp"], unit="s", tz="UTC")
    uf, itf, xf, base_meta = compute_feature_tables(train, users, items, cfg, as_of, derived)

    uf.to_parquet(fdir / "user_features.parquet", index=False)
    itf.to_parquet(fdir / "item_features.parquet", index=False)
    xf.to_parquet(fdir / "interaction_features.parquet", index=False)

    version = hashlib.sha1(f"{split['cutoff_timestamp']}-{len(train)}-{len(uf)}-{len(itf)}".encode()).hexdigest()[:10]
    meta = {**base_meta, "feature_version": version,
            "n_user_features": uf.shape[1] - 1, "n_item_features": itf.shape[1] - 1,
            "n_interaction_features": xf.shape[1] - 2,
            "rows": {"users": len(uf), "items": len(itf), "interactions": len(xf)}}
    write_json(meta, fdir / "feature_metadata.json")
    write_catalog(reports, uf, itf, xf)
    log.info("Feature store v%s: %d user, %d item, %d interaction rows", version, len(uf), len(itf), len(xf))
    return meta


if __name__ == "__main__":
    run()

"""Serve recommendations for a user from a trained model + the feature store.

    python -m src.predict --user-id 6040 --model hybrid --k 10

Uses exactly the same FeatureStore and model classes as training, which is the
"features reusable for training and prediction" requirement in practice.
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.common.utils import load_config, path
from src.features.feature_store import FeatureStore
from src.training.models import BaseRecommender


def recommend(user_id: int, model_name: str = "hybrid", k: int = 10, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    fs = FeatureStore(path(cfg, "features_dir"))
    model = BaseRecommender.load(path(cfg, "models_dir") / model_name, fs)
    train = pd.read_parquet(path(cfg, "processed_dir") / "train.parquet", columns=["user_id", "item_id"])
    seen_items = train.loc[train["user_id"] == user_id, "item_id"]
    i_pos = pd.Series(range(len(model.item_ids)), index=model.item_ids)
    seen = {user_id: i_pos.reindex(seen_items).dropna().astype(int).to_numpy()}
    recs = model.recommend([user_id], k, seen)[0]
    feats = fs.get_item_features(recs, ["item_title", "item_primary_genre", "item_popularity_rank",
                                        "item_bayesian_avg_rating"])
    feats.insert(0, "rank", range(1, len(recs) + 1))
    return feats.reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--user-id", type=int, required=True)
    ap.add_argument("--model", default="hybrid",
                    choices=["popularity", "matrix_factorization", "implicit_als", "hybrid"])
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    cfg = load_config()
    fs = FeatureStore(path(cfg, "features_dir"))
    prof = fs.get_user_features([args.user_id], ["user_activity_count", "user_avg_rating",
                                                 "user_preferred_genre", "user_age_group"])
    print("User profile (from feature store):")
    print(prof.to_string(), "\n")
    print(f"Top-{args.k} recommendations from '{args.model}':")
    print(recommend(args.user_id, args.model, args.k, cfg).to_string(index=False))


if __name__ == "__main__":
    main()

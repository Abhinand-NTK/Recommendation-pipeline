"""A small, file-based feature store (custom feature layer).

Offline store : Parquet tables in data/features/ (versioned by `as_of` + hash)
Online API    : FeatureStore.get_user_features / get_item_features /
                genre_affinity - the same code path is used by the models at
                training time AND by src/predict.py at serving time, so features
                are computed once and reused consistently.
"""
from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.utils import PROJECT_ROOT, read_json


class FeatureStore:
    def __init__(self, features_dir: str | Path | None = None):
        self.dir = Path(features_dir) if features_dir else PROJECT_ROOT / "data" / "features"
        self.metadata = read_json(self.dir / "feature_metadata.json")

    @classmethod
    def from_frames(cls, user_features: pd.DataFrame, item_features: pd.DataFrame,
                    metadata: dict) -> "FeatureStore":
        """In-memory store (used for validation-period tuning without touching disk)."""
        obj = cls.__new__(cls)
        obj.dir, obj.metadata = None, metadata
        obj.__dict__["user_features"] = user_features.set_index("user_id")
        obj.__dict__["item_features"] = item_features.set_index("item_id")
        return obj

    # ---- offline tables
    @cached_property
    def user_features(self) -> pd.DataFrame:
        return pd.read_parquet(self.dir / "user_features.parquet").set_index("user_id")

    @cached_property
    def item_features(self) -> pd.DataFrame:
        return pd.read_parquet(self.dir / "item_features.parquet").set_index("item_id")

    @cached_property
    def interaction_features(self) -> pd.DataFrame:
        return pd.read_parquet(self.dir / "interaction_features.parquet")

    @property
    def genre_cols(self) -> list[str]:
        return [c for c in self.item_features.columns if c.startswith("genre_")]

    @property
    def profile_cols(self) -> list[str]:
        return [c for c in self.user_features.columns if c.startswith("genre_profile_")]

    # ---- online-style lookups
    def get_user_features(self, user_ids, columns=None) -> pd.DataFrame:
        df = self.user_features.reindex(pd.Index(user_ids, name="user_id"))
        defaults = self.metadata["user_defaults"]
        df = df.fillna({k: v for k, v in defaults.items() if k in df.columns})
        return df[columns] if columns else df

    def get_item_features(self, item_ids, columns=None) -> pd.DataFrame:
        df = self.item_features.reindex(pd.Index(item_ids, name="item_id"))
        return df[columns] if columns else df

    def genre_affinity(self, user_ids, item_ids) -> np.ndarray:
        """Cosine similarity between user genre profiles and item genre vectors.

        Returns a (len(user_ids), len(item_ids)) matrix.
        """
        prof = self.get_user_features(user_ids, self.profile_cols).fillna(0).to_numpy(np.float32, copy=True)
        gen = self.get_item_features(item_ids, self.genre_cols).fillna(0).to_numpy(np.float32, copy=True)
        prof /= np.linalg.norm(prof, axis=1, keepdims=True).clip(min=1e-8)
        gen /= np.linalg.norm(gen, axis=1, keepdims=True).clip(min=1e-8)
        return prof @ gen.T

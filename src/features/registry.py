"""Feature registry - the single source of truth for feature definitions.

Every feature that the pipeline materialises is declared here once. The catalog
(reports/feature_catalog.md) is generated from this list, so documentation can
never drift away from the code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FeatureDef:
    name: str
    entity: str          # user | item | interaction
    dtype: str
    description: str
    source: str          # dataset the feature is computed from
    logic: str           # how it is computed
    used_by: str         # which model / stage consumes it


FEATURES: list[FeatureDef] = [
    # ---------------- user features
    FeatureDef("user_activity_count", "user", "int", "Number of ratings the user made before the cut-off",
               "train", "count(ratings) per user", "hybrid, eval segments"),
    FeatureDef("user_distinct_items", "user", "int", "Distinct items rated", "train",
               "nunique(item_id) per user", "analysis"),
    FeatureDef("user_avg_rating", "user", "float", "Mean rating given by the user", "train",
               "mean(rating) per user", "hybrid, MF bias init"),
    FeatureDef("user_rating_std", "user", "float", "Std-dev of the user's ratings", "train",
               "std(rating) per user, 0 if single rating", "analysis"),
    FeatureDef("user_rating_bias", "user", "float", "How much more generous than average the user is",
               "train", "user_avg_rating - global_mean_rating", "analysis"),
    FeatureDef("user_share_liked", "user", "float", "Share of ratings >= relevance threshold", "train",
               "mean(rating >= 4)", "analysis"),
    FeatureDef("user_preferred_genre", "user", "category", "Genre the user rates most often",
               "train + items", "argmax of user genre profile", "hybrid, cold-start fallback"),
    FeatureDef("user_preferred_genre_share", "user", "float", "Share of ratings in the preferred genre",
               "train + items", "max of user genre profile", "analysis"),
    FeatureDef("user_genre_entropy", "user", "float", "Breadth of taste (Shannon entropy of genre profile)",
               "train + items", "-sum(p*log p) over genre shares", "analysis"),
    FeatureDef("user_days_since_last_interaction", "user", "float", "Recency of the user's last rating",
               "train", "(as_of - max(event_time)) in days", "analysis, segmentation"),
    FeatureDef("user_tenure_days", "user", "float", "Days between first and last rating", "train",
               "max(event_time) - min(event_time) in days", "analysis"),
    FeatureDef("user_active_days", "user", "int", "Number of distinct days with activity", "train",
               "nunique(date(event_time))", "analysis"),
    FeatureDef("user_gender", "user", "category", "Gender (M/F/U)", "users", "standardised, U if missing",
               "analysis"),
    FeatureDef("user_age_group", "user", "category", "MovieLens age bucket label", "users",
               "decoded from age code", "analysis"),
    FeatureDef("user_occupation", "user", "category", "Occupation name", "users",
               "decoded from occupation code", "analysis"),
    FeatureDef("genre_profile_{g}", "user", "float", "Share of the user's ratings that fall in genre {g}",
               "train + items", "count per genre / total genre tags", "hybrid (genre affinity)"),
    # ---------------- item features
    FeatureDef("item_popularity_count", "item", "int", "Number of ratings the item received", "train",
               "count(ratings) per item", "popularity, hybrid"),
    FeatureDef("item_popularity_rank", "item", "int", "Rank by popularity (1 = most rated)", "train",
               "rank(desc count)", "popularity"),
    FeatureDef("item_popularity_score", "item", "float", "Log-scaled popularity in [0, 1]", "train",
               "log1p(count) / log1p(max count)", "hybrid"),
    FeatureDef("item_avg_rating", "item", "float", "Mean rating of the item", "train",
               "mean(rating) per item", "analysis"),
    FeatureDef("item_bayesian_avg_rating", "item", "float", "Shrunk mean rating (robust for few ratings)",
               "train", "(n*avg + m*global_mean)/(n+m), m=25", "popularity (rating prediction)"),
    FeatureDef("item_rating_std", "item", "float", "Std-dev of item ratings", "train",
               "std(rating) per item", "analysis"),
    FeatureDef("item_days_since_last_interaction", "item", "float", "Recency of the item's last rating",
               "train", "(as_of - max(event_time)) in days", "analysis"),
    FeatureDef("item_release_year", "item", "float", "Release year parsed from title", "items",
               "regex on 'Title (YYYY)'", "analysis"),
    FeatureDef("item_age_years", "item", "float", "Item age at the cut-off", "items",
               "as_of.year - release_year", "analysis"),
    FeatureDef("item_primary_genre", "item", "category", "First genre (alphabetical) of the item", "items",
               "first of sorted genre list", "analysis"),
    FeatureDef("item_num_genres", "item", "int", "Number of genres tagged", "items",
               "len(genres)", "diversity metric"),
    FeatureDef("genre_{g}", "item", "int (0/1)", "One-hot genre indicator", "items",
               "get_dummies(genres)", "hybrid, diversity metric"),
    # ---------------- interaction features
    FeatureDef("rating_deviation_user", "interaction", "float", "Rating minus the user's mean rating",
               "train", "rating - user_avg_rating", "analysis"),
    FeatureDef("rating_deviation_item", "interaction", "float", "Rating minus the item's mean rating",
               "train", "rating - item_avg_rating", "analysis"),
    FeatureDef("days_before_cutoff", "interaction", "float", "Age of the interaction at the cut-off",
               "train", "(as_of - event_time) in days", "recency weighting"),
    FeatureDef("recency_weight", "interaction", "float", "Exponential time-decay weight", "train",
               "exp(-days_before_cutoff / 90)", "analysis"),
    FeatureDef("is_liked", "interaction", "int (0/1)", "Implicit positive label", "train",
               "rating >= relevance threshold", "evaluation labels"),
    FeatureDef("user_item_genre_affinity", "interaction", "float",
               "On-demand: how well an item's genres match the user's genre profile",
               "feature store", "cosine(user genre profile, item genre vector)",
               "hybrid (training + prediction)"),
]


def as_records() -> list[dict]:
    return [asdict(f) for f in FEATURES]

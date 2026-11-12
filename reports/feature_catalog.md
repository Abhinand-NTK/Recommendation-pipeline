# Feature Catalog

Generated automatically from `src/features/registry.py`. All features are computed from the training split only, as-of the split cut-off, to avoid leakage. Access them through `src.features.feature_store.FeatureStore`.

Null % on user features = users with no activity before the cut-off (cold users); the FeatureStore fills them with documented defaults (global mean, zero counts).

## User features (16)

| Feature | Type | Description | Source | Logic | Used by | Null % | Example |
|---|---|---|---|---|---|---|---|
| `user_activity_count` | int | Number of ratings the user made before the cut-off | train | count(ratings) per user | hybrid, eval segments | 0.00% | 0 |
| `user_distinct_items` | int | Distinct items rated | train | nunique(item_id) per user | analysis | 10.82% | 31.0 |
| `user_avg_rating` | float | Mean rating given by the user | train | mean(rating) per user | hybrid, MF bias init | 10.82% | 4.323 |
| `user_rating_std` | float | Std-dev of the user's ratings | train | std(rating) per user, 0 if single rating | analysis | 10.82% | 0.599 |
| `user_rating_bias` | float | How much more generous than average the user is | train | user_avg_rating - global_mean_rating | analysis | 10.82% | 0.732 |
| `user_share_liked` | float | Share of ratings >= relevance threshold | train | mean(rating >= 4) | analysis | 10.82% | 0.935 |
| `user_preferred_genre` | category | Genre the user rates most often | train + items | argmax of user genre profile | hybrid, cold-start fallback | 10.82% | drama |
| `user_preferred_genre_share` | float | Share of ratings in the preferred genre | train + items | max of user genre profile | analysis | 10.82% | 0.429 |
| `user_genre_entropy` | float | Breadth of taste (Shannon entropy of genre profile) | train + items | -sum(p*log p) over genre shares | analysis | 10.82% | 1.886 |
| `user_days_since_last_interaction` | float | Recency of the user's last rating | train | (as_of - max(event_time)) in days | analysis, segmentation | 10.82% | 0.0 |
| `user_tenure_days` | float | Days between first and last rating | train | max(event_time) - min(event_time) in days | analysis | 10.82% | 0.01 |
| `user_active_days` | int | Number of distinct days with activity | train | nunique(date(event_time)) | analysis | 10.82% | 1.0 |
| `user_gender` | category | Gender (M/F/U) | users | standardised, U if missing | analysis | 0.00% | F |
| `user_age_group` | category | MovieLens age bucket label | users | decoded from age code | analysis | 0.00% | Under 18 |
| `user_occupation` | category | Occupation name | users | decoded from occupation code | analysis | 0.00% | K-12 student |
| `genre_profile_{g}` (19 cols) | float | Share of the user's ratings that fall in genre {g} | train + items | count per genre / total genre tags | hybrid (genre affinity) | 10.82% | 0.036 |

## Item features (12)

| Feature | Type | Description | Source | Logic | Used by | Null % | Example |
|---|---|---|---|---|---|---|---|
| `item_popularity_count` | int | Number of ratings the item received | train | count(ratings) per item | popularity, hybrid | 0.00% | 1750 |
| `item_popularity_rank` | int | Rank by popularity (1 = most rated) | train | rank(desc count) | popularity | 5.93% | 30.0 |
| `item_popularity_score` | float | Log-scaled popularity in [0, 1] | train | log1p(count) / log1p(max count) | hybrid | 0.00% | 0.937 |
| `item_avg_rating` | float | Mean rating of the item | train | mean(rating) per item | analysis | 5.93% | 4.146 |
| `item_bayesian_avg_rating` | float | Shrunk mean rating (robust for few ratings) | train | (n*avg + m*global_mean)/(n+m), m=25 | popularity (rating prediction) | 5.93% | 4.138 |
| `item_rating_std` | float | Std-dev of item ratings | train | std(rating) per item | analysis | 5.93% | 0.859 |
| `item_days_since_last_interaction` | float | Recency of the item's last rating | train | (as_of - max(event_time)) in days | analysis | 5.93% | 0.012 |
| `item_release_year` | float | Release year parsed from title | items | regex on 'Title (YYYY)' | analysis | 0.03% | 1995.0 |
| `item_age_years` | float | Item age at the cut-off | items | as_of.year - release_year | analysis | 0.03% | 5.0 |
| `item_primary_genre` | category | First genre (alphabetical) of the item | items | first of sorted genre list | analysis | 0.00% | animation |
| `item_num_genres` | int | Number of genres tagged | items | len(genres) | diversity metric | 0.00% | 3 |
| `genre_{g}` (19 cols) | int (0/1) | One-hot genre indicator | items | get_dummies(genres) | hybrid, diversity metric | 0.00% | 0 |

## Interaction features (6)

| Feature | Type | Description | Source | Logic | Used by | Null % | Example |
|---|---|---|---|---|---|---|---|
| `rating_deviation_user` | float | Rating minus the user's mean rating | train | rating - user_avg_rating | analysis | 0.00% | 0.456 |
| `rating_deviation_item` | float | Rating minus the item's mean rating | train | rating - item_avg_rating | analysis | 0.00% | -0.518 |
| `days_before_cutoff` | float | Age of the interaction at the cut-off | train | (as_of - event_time) in days | recency weighting | 0.00% | 220.657 |
| `recency_weight` | float | Exponential time-decay weight | train | exp(-days_before_cutoff / 90) | analysis | 0.00% | 0.086 |
| `is_liked` | int (0/1) | Implicit positive label | train | rating >= relevance threshold | evaluation labels | 0.00% | 1 |
| `user_item_genre_affinity` | float | On-demand: how well an item's genres match the user's genre profile | feature store | cosine(user genre profile, item genre vector) | hybrid (training + prediction) | computed on demand |  |

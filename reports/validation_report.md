# Data Validation Report

Generated: 2026-11-12T07:04:11.003678+00:00

## Summary

| Table | Raw rows | Valid rows | Rejected rows | Pass rate |
|---|---|---|---|---|
| users | 6,085 | 6,055 | 30 | 99.5070% |
| items | 3,913 | 3,893 | 20 | 99.4889% |
| interactions | 1,002,609 | 1,000,209 | 2,400 | 99.7606% |

## Rule results

| Table | Rule | Severity | Failed rows | Description |
|---|---|---|---|---|
| users | `missing_user_id` | ERROR | 0 | user_id is null |
| users | `user_id_not_integer` | ERROR | 0 | user_id must be an integer |
| users | `duplicate_user_id` | ERROR | 15 | user_id appears more than once (first occurrence kept) |
| users | `invalid_age` | ERROR | 15 | age must be one of the MovieLens age buckets [1, 18, 25, 35, 45, 50, 56] |
| users | `invalid_gender_value` | ERROR | 0 | gender must be M or F |
| users | `invalid_occupation` | ERROR | 0 | occupation code must be in [0, 20] |
| users | `missing_gender` | WARNING | 15 | gender missing - imputed as 'unknown' during preparation |
| users | `missing_age` | WARNING | 0 | age missing - imputed as 'unknown' during preparation |
| users | `missing_occupation` | WARNING | 0 | occupation missing - imputed as 'unknown' during preparation |
| users | `missing_zip_code` | WARNING | 0 | zip_code missing - imputed as 'unknown' during preparation |
| items | `missing_item_id` | ERROR | 0 | item_id is null |
| items | `item_id_not_integer` | ERROR | 0 | item_id must be an integer |
| items | `duplicate_item_id` | ERROR | 10 | item_id appears more than once (first occurrence kept) |
| items | `missing_title` | ERROR | 10 | title is required |
| items | `missing_genres` | WARNING | 10 | genres missing - imputed as 'unknown' during preparation |
| interactions | `missing_user_id` | ERROR | 300 | user_id is required |
| interactions | `missing_item_id` | ERROR | 0 | item_id is required |
| interactions | `missing_rating` | ERROR | 390 | rating is required |
| interactions | `missing_timestamp` | ERROR | 0 | timestamp is required |
| interactions | `user_id_not_integer` | ERROR | 0 | user_id must be an integer |
| interactions | `item_id_not_integer` | ERROR | 0 | item_id must be an integer |
| interactions | `rating_wrong_type` | ERROR | 210 | rating is not numeric |
| interactions | `rating_out_of_range` | ERROR | 300 | rating must be within [1, 5] |
| interactions | `timestamp_not_numeric` | ERROR | 139 | timestamp must be unix seconds |
| interactions | `timestamp_out_of_range` | ERROR | 161 | timestamp must be between 1995-01-01 and now |
| interactions | `orphan_user_id` | ERROR | 300 | user_id not found in validated users table (referential integrity) |
| interactions | `orphan_item_id` | ERROR | 300 | item_id not found in validated items table (referential integrity) |
| interactions | `exact_duplicate_interaction` | ERROR | 300 | identical row seen before (first occurrence kept) |
| interactions | `conflicting_user_item_pair` | WARNING | 0 | same user-item pair with different rating/time - latest kept in preparation |

## Schema contract (Pandera)

All validated tables passed the Pandera schema contract (types, ranges, uniqueness).

## Injected faults (ground truth from ingestion)

| Fault | Rows injected |
|---|---|
| missing_rating | 300 |
| rating_out_of_range | 300 |
| rating_wrong_type | 300 |
| invalid_timestamp | 300 |
| exact_duplicate_interaction | 300 |
| orphan_user_id | 300 |
| orphan_item_id | 300 |
| missing_user_id | 300 |
| duplicate_user_id | 15 |
| invalid_age | 15 |
| missing_gender (non-critical) | 15 |
| duplicate_item_id | 10 |
| missing_title | 10 |
| missing_genres (non-critical) | 10 |

Some rows fail several rules at once (e.g. an orphan id that is also a duplicate), so per-rule counts can differ slightly from injected counts.

Rejected rows with their reasons are stored in `data/errors/`.

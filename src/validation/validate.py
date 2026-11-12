"""Stage 2 - Data validation.

Rule-based checks on the raw layer (missing values, duplicates, wrong types,
invalid ratings, invalid timestamps, referential integrity) followed by a
Pandera schema contract on the records that pass.

Outputs
    data/processed/validated/{users,items,interactions}.parquet  -> clean records
    data/errors/{table}_errors.csv                               -> rejected rows + reasons
    reports/validation_report.md / .json                         -> human + machine report

Severity
    ERROR   row is moved to the error dataset
    WARNING row is kept; issue is fixed in the preparation stage (e.g. imputation)
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pandera.pandas as pa

from src.common.utils import get_logger, load_config, path, read_json, write_json


# --------------------------------------------------------------------------- helpers
def _is_int_like(s: pd.Series) -> pd.Series:
    num = pd.to_numeric(s, errors="coerce")
    return num.notna() & (num == num.round())


class Checker:
    """Collects per-row failure reasons for one table."""

    def __init__(self, df: pd.DataFrame, table: str):
        self.df = df
        self.table = table
        self.reasons = pd.Series([""] * len(df), index=df.index, dtype=object)
        self.results: list[dict] = []

    def check(self, rule: str, mask: pd.Series, severity: str = "ERROR", desc: str = ""):
        mask = mask.fillna(False).astype(bool)
        n = int(mask.sum())
        self.results.append({"table": self.table, "rule": rule, "severity": severity,
                             "failed_rows": n, "description": desc})
        if severity == "ERROR" and n:
            self.reasons[mask] = self.reasons[mask].where(self.reasons[mask] == "",
                                                          self.reasons[mask] + ";") + rule
        return mask

    @property
    def bad(self) -> pd.Series:
        return self.reasons != ""

    def split(self):
        errors = self.df[self.bad].copy()
        errors["error_reason"] = self.reasons[self.bad]
        return self.df[~self.bad].copy(), errors


# --------------------------------------------------------------------------- tables
def validate_users(df: pd.DataFrame, vcfg: dict) -> Checker:
    c = Checker(df, "users")
    c.check("missing_user_id", df["user_id"].isna(), desc="user_id is null")
    c.check("user_id_not_integer", df["user_id"].notna() & ~_is_int_like(df["user_id"]),
            desc="user_id must be an integer")
    c.check("duplicate_user_id", df["user_id"].notna() & df.duplicated("user_id", keep="first"),
            desc="user_id appears more than once (first occurrence kept)")
    age = pd.to_numeric(df["age"], errors="coerce")
    c.check("invalid_age", df["age"].notna() & ~age.isin(vcfg["valid_ages"]),
            desc=f"age must be one of the MovieLens age buckets {vcfg['valid_ages']}")
    c.check("invalid_gender_value", df["gender"].notna() & ~df["gender"].isin(vcfg["valid_genders"]),
            desc="gender must be M or F")
    lo, hi = vcfg["valid_occupations_range"]
    occ = pd.to_numeric(df["occupation"], errors="coerce")
    c.check("invalid_occupation", df["occupation"].notna() & ~occ.between(lo, hi),
            desc=f"occupation code must be in [{lo}, {hi}]")
    for col in ["gender", "age", "occupation", "zip_code"]:
        c.check(f"missing_{col}", df[col].isna(), severity="WARNING",
                desc=f"{col} missing - imputed as 'unknown' during preparation")
    return c


def validate_items(df: pd.DataFrame) -> Checker:
    c = Checker(df, "items")
    c.check("missing_item_id", df["item_id"].isna(), desc="item_id is null")
    c.check("item_id_not_integer", df["item_id"].notna() & ~_is_int_like(df["item_id"]),
            desc="item_id must be an integer")
    c.check("duplicate_item_id", df["item_id"].notna() & df.duplicated("item_id", keep="first"),
            desc="item_id appears more than once (first occurrence kept)")
    c.check("missing_title", df["title"].isna() | (df["title"].astype(str).str.strip() == ""),
            desc="title is required")
    c.check("missing_genres", df["genres"].isna(), severity="WARNING",
            desc="genres missing - imputed as 'unknown' during preparation")
    return c


def validate_interactions(df, valid_users, valid_items, vcfg) -> Checker:
    c = Checker(df, "interactions")
    for col in ["user_id", "item_id", "rating", "timestamp"]:
        c.check(f"missing_{col}", df[col].isna(), desc=f"{col} is required")
    for col in ["user_id", "item_id"]:
        c.check(f"{col}_not_integer", df[col].notna() & ~_is_int_like(df[col]),
                desc=f"{col} must be an integer")
    rating = pd.to_numeric(df["rating"], errors="coerce")
    c.check("rating_wrong_type", df["rating"].notna() & rating.isna(),
            desc="rating is not numeric")
    c.check("rating_out_of_range",
            rating.notna() & ~rating.between(vcfg["rating_min"], vcfg["rating_max"]),
            desc=f"rating must be within [{vcfg['rating_min']}, {vcfg['rating_max']}]")

    ts = pd.to_numeric(df["timestamp"], errors="coerce")
    ts_min = pd.Timestamp(vcfg["timestamp_min"], tz="UTC").timestamp()
    ts_max = (datetime.now(timezone.utc).timestamp() if vcfg["timestamp_max"] == "now"
              else pd.Timestamp(vcfg["timestamp_max"], tz="UTC").timestamp())
    c.check("timestamp_not_numeric", df["timestamp"].notna() & ts.isna(),
            desc="timestamp must be unix seconds")
    c.check("timestamp_out_of_range", ts.notna() & ~ts.between(ts_min, ts_max),
            desc=f"timestamp must be between {vcfg['timestamp_min']} and {vcfg['timestamp_max']}")

    uid = pd.to_numeric(df["user_id"], errors="coerce")
    iid = pd.to_numeric(df["item_id"], errors="coerce")
    c.check("orphan_user_id", uid.notna() & ~uid.isin(valid_users),
            desc="user_id not found in validated users table (referential integrity)")
    c.check("orphan_item_id", iid.notna() & ~iid.isin(valid_items),
            desc="item_id not found in validated items table (referential integrity)")
    c.check("exact_duplicate_interaction", df.duplicated(keep="first"),
            desc="identical row seen before (first occurrence kept)")
    ok = ~c.bad  # only look for conflicts among rows that are otherwise valid
    conflict = df[ok].duplicated(["user_id", "item_id"], keep=False).reindex(df.index, fill_value=False)
    c.check("conflicting_user_item_pair", conflict, severity="WARNING",
            desc="same user-item pair with different rating/time - latest kept in preparation")
    return c


# --------------------------------------------------------------------------- contracts
def schemas(vcfg: dict) -> dict[str, pa.DataFrameSchema]:
    return {
        "users": pa.DataFrameSchema({
            "user_id": pa.Column(int, unique=True, checks=pa.Check.gt(0)),
            "gender": pa.Column(str, nullable=True),
            "age": pa.Column(float, nullable=True, checks=pa.Check.isin(vcfg["valid_ages"])),
            "occupation": pa.Column(float, nullable=True),
            "zip_code": pa.Column(str, nullable=True),
        }, coerce=True),
        "items": pa.DataFrameSchema({
            "item_id": pa.Column(int, unique=True, checks=pa.Check.gt(0)),
            "title": pa.Column(str, nullable=False),
            "genres": pa.Column(str, nullable=True),
        }, coerce=True),
        "interactions": pa.DataFrameSchema({
            "user_id": pa.Column(int), "item_id": pa.Column(int),
            "rating": pa.Column(float, checks=pa.Check.in_range(vcfg["rating_min"], vcfg["rating_max"])),
            "timestamp": pa.Column(int, checks=pa.Check.gt(0)),
        }, coerce=True),
    }


# --------------------------------------------------------------------------- report
def _markdown(results, summary, injected) -> str:
    lines = ["# Data Validation Report", "",
             f"Generated: {datetime.now(timezone.utc).isoformat()}", "",
             "## Summary", "", "| Table | Raw rows | Valid rows | Rejected rows | Pass rate |",
             "|---|---|---|---|---|"]
    for t, s in summary.items():
        lines.append(f"| {t} | {s['raw_rows']:,} | {s['valid_rows']:,} | {s['rejected_rows']:,} "
                     f"| {s['pass_rate']:.4%} |")
    lines += ["", "## Rule results", "",
              "| Table | Rule | Severity | Failed rows | Description |", "|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['table']} | `{r['rule']}` | {r['severity']} | {r['failed_rows']:,} "
                     f"| {r['description']} |")
    lines += ["", "## Schema contract (Pandera)", "",
              "All validated tables passed the Pandera schema contract (types, ranges, uniqueness)."]
    if injected:
        lines += ["", "## Injected faults (ground truth from ingestion)", "",
                  "| Fault | Rows injected |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in injected.items()]
        lines += ["", "Some rows fail several rules at once (e.g. an orphan id that is also a "
                  "duplicate), so per-rule counts can differ slightly from injected counts."]
    lines += ["", "Rejected rows with their reasons are stored in `data/errors/`."]
    return "\n".join(lines) + "\n"


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    log = get_logger("validation", cfg)
    vcfg = cfg["validation"]
    raw_dir, err_dir = path(cfg, "raw_dir"), path(cfg, "errors_dir")
    out_dir = path(cfg, "processed_dir") / "validated"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = path(cfg, "reports_dir")

    raw = {t: pd.read_csv(raw_dir / f"{t}.csv", dtype=str, keep_default_na=True)
           for t in ["users", "items", "interactions"]}

    cu = validate_users(raw["users"], vcfg)
    users_ok, users_err = cu.split()
    ci = validate_items(raw["items"])
    items_ok, items_err = ci.split()
    valid_users = set(pd.to_numeric(users_ok["user_id"]).astype(int))
    valid_items = set(pd.to_numeric(items_ok["item_id"]).astype(int))
    cx = validate_interactions(raw["interactions"], valid_users, valid_items, vcfg)
    inter_ok, inter_err = cx.split()

    contract = schemas(vcfg)
    clean = {"users": users_ok, "items": items_ok, "interactions": inter_ok}
    errors = {"users": users_err, "items": items_err, "interactions": inter_err}
    summary = {}
    for t in clean:
        clean[t] = contract[t].validate(clean[t], lazy=True)
        clean[t].to_parquet(out_dir / f"{t}.parquet", index=False)
        errors[t].to_csv(err_dir / f"{t}_errors.csv", index=False)
        n_raw = len(raw[t])
        summary[t] = {"raw_rows": n_raw, "valid_rows": len(clean[t]),
                      "rejected_rows": len(errors[t]), "pass_rate": len(clean[t]) / n_raw}
        log.info("%s: %d valid, %d rejected", t, len(clean[t]), len(errors[t]))

    results = cu.results + ci.results + cx.results
    injected = read_json(raw_dir / "ingestion_metadata.json").get("fault_injection", {})
    report = {"summary": summary, "rules": results, "injected_faults": injected}
    write_json(report, reports / "validation_report.json")
    (reports / "validation_report.md").write_text(_markdown(results, summary, injected), encoding="utf-8")
    log.info("Validation report written to reports/validation_report.md")
    return report


if __name__ == "__main__":
    run()

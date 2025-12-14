from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# --------------------------- Logging ---------------------------------

def setup_logging(verbosity: int = 1):
    level = logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# --------------------------- Helpers ---------------------------------

TRUTHY = {"y","yes","true","t","1",1,True}
FALSY  = {"n","no","false","f","0",0,False}

CHI_LAT_MIN, CHI_LAT_MAX = 41.60, 42.10
CHI_LNG_MIN, CHI_LNG_MAX = -88.00, -87.40

LEAKAGE_COLS = [
    # Post-event / admin
    "report_type","photos_taken_i","statements_taken_i","date_police_notified",
    # IDs / high cardinality JSON
    "crash_record_id","veh_vehicle_id_list_json","ppl_person_id_list_json","location_json",
    # high-cardinality street name
    "street_name",
]

BOOL_SUFFIX = "_i"

CAUSE_MAP = {
    "Speeding": ["speed"],
    "DUI/Impairment": ["alcohol", "drug", "impair", "under the influence", "dui", "bac"],
    "Distraction/Inattention": ["distract", "phone", "text", "inattention"],
    "Failure-to-Yield": ["fail", "failed to yield", "right-of-way", "right of way", "yield"],
    "Following Too Closely": ["following too closely", "tailg"],
    "Weather-related": ["weather", "rain", "snow", "ice", "sleet", "hail"],
    "Lighting/Visibility": ["dark", "light", "glare", "visibility"],
    "Other": [],
}

TRUCK_KEYWORDS = ["truck","semi","tractor","box truck","dump","pickup","tow","flatbed","18-wheeler","semi-trailer","semi trailer"]
MOTORCYCLE_KEYWORDS = ["motorcycle","motor bike","harley","moto","bike (motor)","sportbike"]

RECODE_MISSING = {"", " ", "unknown", "unk", "n/a", "na", "none", "--", "-", "null"}

def is_boolish_series(s: pd.Series) -> bool:
    name = str(s.name).lower()
    if name.endswith(BOOL_SUFFIX):
        return True
    if pd.api.types.is_bool_dtype(s):
        return True
    vals = set(map(lambda x: str(x).strip().lower(), s.dropna().unique().tolist()))
    allowed = {"0","1","true","false","t","f","yes","no","y","n"}
    if len(vals) > 0 and vals.issubset(allowed):
        return True
    return False

def normalize_bool_value(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)) and not math.isnan(x):
        return 1 if int(x) == 1 else 0 if int(x) == 0 else np.nan
    s = str(x).strip().lower()
    if s in TRUTHY:
        return 1
    if s in FALSY:
        return 0
    return np.nan

def safe_lower(x):
    if pd.isna(x):
        return x
    return str(x).strip().lower()

def safe_json_list(x) -> List[str]:
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    s = str(x).strip()
    if not s:
        return []
    try:
        val = json.loads(s)
        if isinstance(val, list):
            return [str(v) for v in val]
    except Exception:
        pass
    parts = [p.strip() for p in re.split(r"[;,]", s) if p.strip()]
    return parts

def any_keyword(items: List[str], keywords: List[str]) -> bool:
    items_l = " ".join([i.lower() for i in items])
    return any(kw in items_l for kw in keywords)

def group_cause(raw: Optional[str]) -> str:
    if raw is None or (isinstance(raw,str) and raw.strip().lower() in RECODE_MISSING) or pd.isna(raw):
        return "missing"
    text = str(raw).lower()
    for group, keys in CAUSE_MAP.items():
        if any(k in text for k in keys):
            return group
    return "Other"

def rare_bucket(series: pd.Series, min_frac: float = 0.01, missing_label: str = "missing") -> pd.Series:
    s = series.astype("object").map(lambda x: safe_lower(x) if not pd.isna(x) else missing_label)
    freq = s.value_counts(dropna=False, normalize=True)
    keep = set(freq[freq >= min_frac].index.tolist())
    return s.map(lambda v: v if v in keep else ("other" if v != missing_label else missing_label))

def clip_series(s: pd.Series, low=None, high=None):
    if low is None and high is None:
        return s
    return s.clip(lower=low, upper=high)

def pct(n: int, d: int) -> float:
    return 0.0 if d == 0 else round(100.0 * n / d, 2)

# --------------------------- Core Cleaning Steps ----------------------

def drop_leakage_and_ids(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    to_drop = [c for c in LEAKAGE_COLS if c in df.columns]
    report["dropped_columns"] = sorted(to_drop)
    logging.info("Dropping %d leakage/ID columns: %s", len(to_drop), to_drop)
    return df.drop(columns=to_drop, errors="ignore")

def standardize_boolean_cols(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    bool_cols = [c for c in df.columns if c.endswith(BOOL_SUFFIX) or is_boolish_series(df[c])]
    changed = []
    for c in bool_cols:
        before_nulls = int(df[c].isna().sum())
        df[c] = df[c].map(normalize_bool_value)
        after_nulls = int(df[c].isna().sum())
        changed.append({"column": c, "nulls_before": before_nulls, "nulls_after": after_nulls})
    report["boolean_standardized"] = changed
    logging.info("Standardized %d boolean-like columns", len(bool_cols))
    return df

def add_time_features(df: pd.DataFrame, report: dict, crash_date_col: str = "crash_date") -> pd.DataFrame:
    if crash_date_col not in df.columns:
        logging.warning("No '%s' column found; skipping time features.", crash_date_col)
        return df
    df[crash_date_col] = pd.to_datetime(df[crash_date_col], errors="coerce")
    nulls = int(df[crash_date_col].isna().sum())
    logging.info("Parsed crash_date; %d nulls", nulls)

    df["year"] = df[crash_date_col].dt.year
    df["month"] = df[crash_date_col].dt.month
    df["day"] = df[crash_date_col].dt.day
    df["hour"] = df[crash_date_col].dt.hour
    df["is_weekend"] = df[crash_date_col].dt.dayofweek.isin([5,6]).astype("Int64")
    def hour_to_bin(h):
        if pd.isna(h):
            return np.nan
        h = int(h)
        if 0 <= h <= 6:
            return "night"
        if 7 <= h <= 12:
            return "morning"
        if 13 <= h <= 18:
            return "afternoon"
        if 19 <= h <= 23:
            return "evening"
        return np.nan
    df["hour_bin"] = df["hour"].map(hour_to_bin)

    for col in ["crash_day_of_week", "crash_month"]:
        if col in df.columns:
            df[col] = df[col].map(safe_lower)

    report["time_features_added"] = ["year","month","day","hour","is_weekend","hour_bin","crash_day_of_week*","crash_month*"]
    return df

def clean_location(df: pd.DataFrame, report: dict, lat_col: str, lng_col: str) -> pd.DataFrame:
    if lat_col not in df.columns or lng_col not in df.columns:
        logging.warning("Latitude/longitude columns not found; skipping location cleaning.")
        return df

    before = len(df)
    mask_valid = (
        pd.to_numeric(df[lat_col], errors="coerce").between(41.60, 42.10, inclusive="both") &
        pd.to_numeric(df[lng_col], errors="coerce").between(-88.00, -87.40, inclusive="both")
    )
    df = df.loc[mask_valid].copy()
    dropped = before - len(df)
    report["location_rows_dropped_invalid_coords"] = dropped
    logging.info("Dropped %d rows with invalid/out-of-bounds coordinates", dropped)

    df["lat_bin"] = pd.to_numeric(df[lat_col], errors="coerce").round(2)
    df["lng_bin"] = pd.to_numeric(df[lng_col], errors="coerce").round(2)
    df["grid_id"] = df["lat_bin"].astype(str) + "_" + df["lng_bin"].astype(str)

    if "beat_of_occurrence" in df.columns:
        df["beat_of_occurrence"] = df["beat_of_occurrence"].map(safe_lower)

    report["location_features_added"] = ["lat_bin", "lng_bin", "grid_id", "beat_of_occurrence*"]
    return df

def clean_road_env(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    for col in ["roadway_surface_cond","lighting_condition","weather_condition","traffic_control_device",
                "work_zone_type","crash_type"]:
        if col in df.columns:
            df[col] = df[col].map(safe_lower)
            df[col] = df[col].replace(list(RECODE_MISSING), np.nan)
    for col in ["work_zone_i","private_property_i"]:
        if col in df.columns:
            df[col] = df[col].map(normalize_bool_value)
    report["road_env_cleaned"] = True
    return df

def extract_vehicle_people(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    make_col, model_col = "veh_make_list_json", "veh_model_list_json"
    if make_col in df.columns or model_col in df.columns:
        makes = df[make_col] if make_col in df.columns else pd.Series([np.nan]*len(df))
        models = df[model_col] if model_col in df.columns else pd.Series([np.nan]*len(df))
        veh_counts, truck_flags, mc_flags = [], [], []
        for mk, md in zip(makes, models):
            mk_list = safe_json_list(mk)
            md_list = safe_json_list(md)
            items = mk_list + md_list
            veh_counts.append(len(set(items)) if items else 0)
            truck_flags.append(1 if any_keyword(items, TRUCK_KEYWORDS) else 0)
            mc_flags.append(1 if any_keyword(items, MOTORCYCLE_KEYWORDS) else 0)
        df["veh_count"] = pd.Series(veh_counts, index=df.index).clip(0, 5)
        df["veh_truck_i"] = pd.Series(truck_flags, index=df.index)
        df["veh_mc_i"] = pd.Series(mc_flags, index=df.index)
    if "ppl_age_list_json" in df.columns:
        ppl_counts, age_means, age_mins, age_maxs = [], [], [], []
        for ages in df["ppl_age_list_json"].tolist():
            age_list = []
            for a in safe_json_list(ages):
                try:
                    v = float(a)
                    if 0 <= v <= 110:
                        age_list.append(v)
                except Exception:
                    continue
            ppl_counts.append(len(age_list))
            age_means.append(np.mean(age_list) if age_list else np.nan)
            age_mins.append(np.min(age_list) if age_list else np.nan)
            age_maxs.append(np.max(age_list) if age_list else np.nan)
        df["ppl_count"] = pd.Series(ppl_counts, index=df.index).clip(0, 10)
        df["ppl_age_mean"] = pd.Series(age_means, index=df.index).clip(0, 100)
        df["ppl_age_min"]  = pd.Series(age_mins,  index=df.index).clip(0, 100)
        df["ppl_age_max"]  = pd.Series(age_maxs,  index=df.index).clip(0, 100)

    report["veh_people_features"] = [c for c in ["veh_count","veh_truck_i","veh_mc_i","ppl_count","ppl_age_mean","ppl_age_min","ppl_age_max"] if c in df.columns]
    return df

def map_contributory_cause(df: pd.DataFrame, report: dict, src_col: str = "prim_contributory_cause") -> pd.DataFrame:
    if src_col in df.columns:
        df["cause_group"] = df[src_col].map(group_cause)
        report["cause_grouped_from"] = src_col
    return df

def handle_missing_and_outliers(df: pd.DataFrame, report: dict, bool_fill_threshold: float = 0.3) -> pd.DataFrame:
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]):
            df[c] = df[c].map(lambda x: np.nan if (isinstance(x,str) and x.strip().lower() in RECODE_MISSING) else x)
    for c in df.columns:
        if is_boolish_series(df[c]):
            na_rate = df[c].isna().mean()
            if na_rate <= bool_fill_threshold:
                df[c] = df[c].fillna(0)
    for c in ["injuries_total","injuries_fatal","injuries_incapacitating"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            high = 20 if c == "injuries_total" else (5 if c == "injuries_incapacitating" else 10)
            df[c] = df[c].clip(lower=0, upper=high)

    report["missing_outliers_handled"] = True
    return df

# --------------------------- Target & Grain ---------------------------

def derive_target(df: pd.DataFrame, name: str) -> Tuple[pd.DataFrame, str, List[str]]:
    if name == "ksi":
        srcs = [c for c in ["injuries_fatal","injuries_incapacitating"] if c in df.columns]
        if not srcs:
            raise ValueError("To derive 'ksi', need injuries_fatal and/or injuries_incapacitating columns.")
        for c in srcs:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        df["ksi_flag"] = ((df.get("injuries_fatal", 0) > 0) | (df.get("injuries_incapacitating", 0) > 0)).astype(int)
        return df, "ksi_flag", srcs
    elif name == "hit_and_run":
        if "hit_and_run_i" not in df.columns:
            raise ValueError("Expected 'hit_and_run_i' column to exist for hit_and_run target.")
        df["hit_and_run_i"] = df["hit_and_run_i"].map(normalize_bool_value)
        return df, "hit_and_run_i", []
    else:
        raise ValueError(f"Unknown derive-target: {name}")

def normalize_target(df: pd.DataFrame, target_col: str, target_type: str, report: dict) -> pd.DataFrame:
    if target_type == "binary":
        df[target_col] = df[target_col].map(normalize_bool_value)
    elif target_type == "numeric":
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    else:
        df[target_col] = df[target_col].astype("object").map(safe_lower)
    before = len(df)
    df = df.dropna(subset=[target_col])
    dropped = before - len(df)
    report["target_nulls_dropped"] = {"column": target_col, "rows_dropped": dropped, "pct": pct(dropped, before)}
    return df

def target_distribution(df: pd.DataFrame, target_col: str, target_type: str, report: dict):
    if target_type in {"binary","categorical"}:
        counts = df[target_col].value_counts(dropna=False).to_dict()
        total = int(df.shape[0])
        perc = {str(k): pct(int(v), total) for k, v in counts.items()}
        report["target_distribution"] = {"counts": counts, "perc": perc}
        logging.info("Target distribution: %s", report["target_distribution"])
    else:
        desc = df[target_col].describe(percentiles=[0.95,0.99]).to_dict()
        report["target_summary"] = desc
        logging.info("Target summary: %s", report["target_summary"])

def align_grain_and_dedupe(df: pd.DataFrame, grain: str, id_cols: Optional[List[str]], report: dict) -> pd.DataFrame:
    if grain == "crash":
        candidates = id_cols or [c for c in ["crash_record_id","crash_record_id_num","rd_no"] if c in df.columns]
        if candidates:
            key = candidates[0]
            before = len(df)
            df = df.sort_index().drop_duplicates(subset=[key], keep="last")
            report["dedupe"] = {"key": key, "rows_removed": before - len(df)}
        else:
            logging.info("No crash ID present for dedupe; skipping.")
    else:
        logging.info("Grain '%s' not specially handled.", grain)
    return df

def drop_target_sources(df: pd.DataFrame, target_col: str, source_cols: List[str], report: dict) -> pd.DataFrame:
    to_drop = [c for c in source_cols if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop, errors="ignore")
    report["dropped_target_sources"] = to_drop
    return df

# --------------------------- Encoding & Splits ------------------------

def identify_feature_types(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    cat_cols, num_cols = [], []
    for c in df.columns:
        if c == target_col:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return cat_cols, num_cols

def consolidate_rare_cats(df: pd.DataFrame, cat_cols: List[str], min_frac: float = 0.01) -> pd.DataFrame:
    for c in cat_cols:
        df[c] = rare_bucket(df[c], min_frac=min_frac)
    return df

def build_ml_matrix_csv(df: pd.DataFrame, target_col: str, out_path_csv: str, report: dict):
    cat_cols, num_cols = identify_feature_types(df, target_col)
    df[cat_cols] = df[cat_cols].apply(lambda s: rare_bucket(s, min_frac=0.01))

    num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    ct = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )
    X = ct.fit_transform(df.drop(columns=[target_col]))

    num_feature_names = num_cols
    cat_feature_names = ct.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_cols).tolist()
    feature_names = num_feature_names + cat_feature_names
    X_df = pd.DataFrame(X, columns=feature_names, index=df.index)
    X_df[target_col] = df[target_col].values

    X_df.to_csv(out_path_csv, index=False)
    report["ml_ready_csv"] = out_path_csv

def split_and_save(df: pd.DataFrame, target_col: str, target_type: str, out_dir: str, report: dict,
                   test_size: float = 0.2, valid_size: float = 0.1, seed: int = 42):
    y = df[target_col]
    stratify = y if target_type in {"binary","categorical"} else None

    df_train_valid, df_test = train_test_split(df, test_size=test_size, random_state=seed, stratify=stratify)
    if valid_size > 0:
        rel_valid = valid_size / (1 - test_size)
        stratify_tv = df_train_valid[target_col] if stratify is not None else None
        df_train, df_valid = train_test_split(df_train_valid, test_size=rel_valid, random_state=seed, stratify=stratify_tv)
    else:
        df_train, df_valid = df_train_valid, pd.DataFrame()

    out_splits = Path(out_dir) / "splits"
    out_splits.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(out_splits / "train.csv", index=False)
    if not df_valid.empty:
        df_valid.to_csv(out_splits / "valid.csv", index=False)
    df_test.to_csv(out_splits / "test.csv", index=False)

    report["splits"] = {
        "train_rows": int(df_train.shape[0]),
        "valid_rows": int(df_valid.shape[0]) if not df_valid.empty else 0,
        "test_rows": int(df_test.shape[0]),
        "stratified": bool(stratify is not None),
    }

# --------------------------- Main ------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Deep clean and ML-prep crash data (CSV only).")
    parser.add_argument("--input", required=True, help="Path to input CSV (e.g., merged.csv)")
    parser.add_argument("--out-dir", default="out", help="Directory to write outputs")
    parser.add_argument("--lat-col", default="latitude", help="Latitude column name")
    parser.add_argument("--lng-col", default="longitude", help="Longitude column name")
    parser.add_argument("--target-col", default=None, help="Existing target column name to use")
    parser.add_argument("--derive-target", choices=["ksi","hit_and_run"], default=None,
                        help="Derive a target: 'ksi' creates ksi_flag; 'hit_and_run' uses hit_and_run_i")
    parser.add_argument("--target-type", choices=["binary","categorical","numeric"], default="binary",
                        help="Target variable type")
    parser.add_argument("--grain", choices=["crash","vehicle","person"], default="crash",
                        help="Modeling grain (unit of rows)")
    parser.add_argument("--no-encode", action="store_true", help="Skip building ml_ready.csv")
    parser.add_argument("--verbosity", "-v", action="count", default=1, help="Increase verbosity (-v, -vv)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    args = parser.parse_args()

    setup_logging(args.verbosity)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"steps": []}

    logging.info("Loading %s ...", args.input)
    df = pd.read_csv(args.input, low_memory=False)
    report["rows_initial"] = int(df.shape[0])
    report["cols_initial"] = int(df.shape[1])

    id_cols = [c for c in ["crash_record_id","crash_record_id_num","rd_no"] if c in df.columns]
    df = align_grain_and_dedupe(df, args.grain, id_cols=id_cols, report=report)

    df = drop_leakage_and_ids(df, report=report)
    df = standardize_boolean_cols(df, report=report)
    df = add_time_features(df, report=report, crash_date_col="crash_date")
    df = clean_location(df, report=report, lat_col=args.lat_col, lng_col=args.lng_col)
    df = clean_road_env(df, report=report)
    df = extract_vehicle_people(df, report=report)
    df = map_contributory_cause(df, report=report, src_col="prim_contributory_cause")
    df = handle_missing_and_outliers(df, report=report)

    if args.target_col and args.derive_target:
        raise SystemExit("Use either --target-col or --derive-target (not both).")
    if args.derive_target:
        df, target_col, target_sources = derive_target(df, args.derive_target)
    elif args.target_col:
        target_col, target_sources = args.target_col, []
    else:
        if "hit_and_run_i" in df.columns:
            df["hit_and_run_i"] = df["hit_and_run_i"].map(normalize_bool_value)
            target_col, target_sources = "hit_and_run_i", []
        else:
            df, target_col, target_sources = derive_target(df, "ksi")

    report["target_col"] = target_col
    report["target_type"] = args.target_type
    report["target_sources_for_leakage_drop"] = target_sources

    df = drop_target_sources(df, target_col, target_sources, report=report)
    df = normalize_target(df, target_col, args.target_type, report=report)
    target_distribution(df, target_col, args.target_type, report=report)

    cleaned_path = out_dir / "cleaned.csv"
    df.to_csv(cleaned_path, index=False)
    report["cleaned_csv"] = str(cleaned_path)

    split_and_save(df, target_col, args.target_type, out_dir=str(out_dir), report=report, seed=args.seed)

    if not args.no_encode:
        ml_ready_path = out_dir / "ml_ready.csv"
        build_ml_matrix_csv(df, target_col, out_path_csv=str(ml_ready_path), report=report)

    report["rows_final"] = int(df.shape[0])
    report["cols_final"] = int(df.shape[1])
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"Completed. Cleaned CSV: {cleaned_path}")
    if not args.no_encode:
        print(f"ML-ready matrix: {ml_ready_path}")
    print(f"Report: {out_dir / 'report.json'}")

if __name__ == "__main__":
    main()

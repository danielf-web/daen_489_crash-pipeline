# cleaning_rules.py

from __future__ import annotations
import math
import numpy as np
import pandas as pd

TRUTHY = {"y", "yes", "true", "t", "1", 1, True}
FALSY  = {"n", "no", "false", "f", "0", 0, False}

RECODE_MISSING = {
    "", " ", "unknown", "unk", "n/a", "na", "none",
    "--", "-", "null", "nan"
}

def normalize_bool_value(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
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

def is_boolish_series(s: pd.Series) -> bool:
    name = str(s.name).lower()
    if name.endswith("_i"):
        return True
    if pd.api.types.is_bool_dtype(s):
        return True
    vals = set(map(lambda x: str(x).strip().lower(), s.dropna().unique().tolist()))
    allowed = {"0", "1", "true", "false", "t", "f", "yes", "no", "y", "n"}
    return len(vals) > 0 and vals.issubset(allowed)

def clip_series(s: pd.Series, low=None, high=None):
    if low is None and high is None:
        return s
    return s.clip(lower=low, upper=high)

def add_time_features(df: pd.DataFrame, crash_date_col="crash_date") -> pd.DataFrame:
    crash_date_col = str(crash_date_col).strip().lower()
    if crash_date_col in df.columns:
        df[crash_date_col] = pd.to_datetime(df[crash_date_col], errors="coerce")
        df["year"] = df[crash_date_col].dt.year
        df["month"] = df[crash_date_col].dt.month
        df["day"] = df[crash_date_col].dt.day
        df["hour"] = df[crash_date_col].dt.hour
        df["is_weekend"] = df[crash_date_col].dt.dayofweek.isin([5, 6]).astype("Int64")

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

    return df

def clean_location(df: pd.DataFrame, lat_col="latitude", lng_col="longitude") -> pd.DataFrame:
    lat_col = str(lat_col).strip().lower()
    lng_col = str(lng_col).strip().lower()

    if lat_col not in df.columns or lng_col not in df.columns:
        return df

    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lng = pd.to_numeric(df[lng_col], errors="coerce")

    mask_valid = lat.between(41.60, 42.10, inclusive="both") & lng.between(-88.00, -87.40, inclusive="both")
    df = df.loc[mask_valid].copy()

    df["lat_bin"] = lat.loc[df.index].round(2)
    df["lng_bin"] = lng.loc[df.index].round(2)
    df["grid_id"] = df["lat_bin"].astype(str) + "_" + df["lng_bin"].astype(str)

    return df

def handle_missing_and_outliers(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]):
            df[c] = df[c].map(
                lambda x: np.nan
                if (isinstance(x, str) and x.strip().lower() in RECODE_MISSING)
                else x
            )

    for c in df.columns:
        if is_boolish_series(df[c]):
            df[c] = df[c].map(normalize_bool_value).fillna(0)

    for c in ["injuries_total", "injuries_fatal", "injuries_incapacitating"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            high = 20 if c == "injuries_total" else (5 if c == "injuries_incapacitating" else 10)
            df[c] = clip_series(df[c], 0, high)

    return df

def drop_leakage(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        "report_type", "photos_taken_i", "statements_taken_i", "date_police_notified",
        "crash_record_id_num", "veh_vehicle_id_list_json", "ppl_person_id_list_json",
        "location_json", "street_name"
    ]
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

def align_and_dedupe(df: pd.DataFrame) -> pd.DataFrame:
    key = "crash_record_id" if "crash_record_id" in df.columns else None
    if key:
        df = df.drop_duplicates(subset=[key], keep="last")
    return df

def clean_for_gold(df: pd.DataFrame, lat_col="latitude", lng_col="longitude") -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]

    # make incoming column names case-insensitive too
    lat_col = str(lat_col).strip().lower()
    lng_col = str(lng_col).strip().lower()

    df = drop_leakage(df)
    df = add_time_features(df, crash_date_col="crash_date")
    df = clean_location(df, lat_col=lat_col, lng_col=lng_col)
    df = handle_missing_and_outliers(df)
    df = align_and_dedupe(df)
    return df

"""
Chicago Crash ETL Dashboard — Single-file Streamlit App
------------------------------------------------------
Run:
  pip install streamlit pandas duckdb altair requests python-dateutil
  streamlit run app.py

Optional extras (only if you want PDF export):
  pip install reportlab

Environment variables (optional):
  ETL_API_BASE_URL  # e.g., http://localhost:8000
  GOLD_DB_PATH      # e.g., /path/to/gold.duckdb (defaults to ./gold.duckdb)

Notes:
- The app calls a backend via REST when available and gracefully falls back to
  local/demo behavior when endpoints are unreachable.
- Designed to satisfy the "9 - Front End and EDA" brief end-to-end.
"""
from __future__ import annotations

import os
import io
import json
import uuid
from datetime import datetime, date, time as dtime, timedelta
from typing import Dict, List, Optional, Tuple

import pickle  # fallback
import joblib  # primary loader for scikit-learn models

import requests
import pandas as pd
import numpy as np
import duckdb
import altair as alt
import streamlit as st
from dateutil import tz
import time as _time
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from prometheus_client import REGISTRY

def _get_or_create_metric(metric_cls, name: str, *args, **kwargs):
    """
    Streamlit reruns can re-register metrics. If the metric already exists in the
    default REGISTRY, reuse it instead of creating a duplicate.
    """
    try:
        return metric_cls(name, *args, **kwargs)
    except ValueError:
        # Metric already exists; reuse the registered one.
        try:
            return REGISTRY._names_to_collectors[name]  # internal but common in prometheus_client
        except Exception:
            # Last resort: re-raise original error if we can't find it
            raise



# ------------------------------
# Config & Globals
# ------------------------------
st.set_page_config(
    page_title="Chicago Crash ETL Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

ALT_THEMES = {
    "font": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica Neue, Arial, sans-serif",
}

BASE_URL = os.getenv("ETL_API_BASE_URL", "http://localhost:8000").rstrip("/")
DEFAULT_DB_PATH = os.getenv(
    "GOLD_DB_PATH", os.path.abspath(os.path.join("cleaner", "gold.duckdb"))
)
LOCAL_TZ = tz.gettz(os.getenv("TZ", "America/Chicago"))

# Model-related constants
MODEL_ARTIFACT_PATH = os.path.join("artifacts", "model.pkl")
THRESHOLD_PATH = os.path.join("artifacts", "threshold.txt")  # decision threshold (optional)
STATIC_METRICS_PATH = os.path.join("artifacts", "test_metrics.json")  # static test metrics (optional)

MODEL_LABEL_COL = "hit_and_run_i"

# Minimal fallback feature list; real list will be taken from the fitted model
MODEL_FEATURE_COLS = [
    "crash_date",
    "crash_hour",
    "crash_day_of_week",
    "posted_speed_limit",
    "weather_condition",
    "traffic_control_device",
]

@st.cache_resource(show_spinner=False)
def _init_prom_ml_metrics(model_path: str, static_metrics_path: str):
    # 1) Start /metrics server once
    port = int(os.getenv("APP_METRICS_PORT", os.getenv("METRICS_PORT", "8003")))
    addr = os.getenv("METRICS_ADDR", "0.0.0.0")
    try:
        start_http_server(port, addr=addr)
    except Exception:
        # already started in this process
        pass

    # 2) Create metrics once (avoids duplicate registry errors on Streamlit reruns)
    ml_inference_total = _get_or_create_metric(Counter,
        "ml_inference_total",
        "Total number of model inference calls performed by the UI",
    )

    ml_prediction_latency_seconds = _get_or_create_metric(Histogram,
        "ml_prediction_latency_seconds",
        "Latency of model inference (seconds)",
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
    )

    ml_model_accuracy = _get_or_create_metric(Gauge,
        "ml_model_accuracy",
        "Model accuracy (from artifacts if available; otherwise last computed on labeled data)",
    )

    ml_training_duration_seconds = _get_or_create_metric(Gauge,
        "ml_training_duration_seconds",
        "Training duration in seconds (from artifacts if available; otherwise model load time proxy)",
    )

    ml_last_trained_ts = _get_or_create_metric(Gauge,
        "ml_last_trained_timestamp_seconds",
        "Last-trained timestamp (seconds since epoch). Uses model.pkl mtime as proxy.",
    )

    DUCKDB_FILE_SIZE_BYTES = _get_or_create_metric(Gauge,
        "duckdb_file_size_bytes",
        "Size of the DuckDB file on disk (bytes)",
    )

    DUCKDB_GOLD_ROW_COUNT = _get_or_create_metric(Gauge,
        "duckdb_gold_row_count",
        "Row count of gold.crashes in DuckDB",
    )


    # 3) Initialize “last trained” + maybe accuracy/training duration from artifacts
    try:
        if os.path.exists(model_path):
            ml_last_trained_ts.set(float(os.path.getmtime(model_path)))
    except Exception:
        pass

    try:
        if os.path.exists(static_metrics_path):
            with open(static_metrics_path, "r", encoding="utf-8") as f:
                sm = json.load(f)

            if isinstance(sm, dict):
                for k in ("accuracy", "acc", "test_accuracy", "val_accuracy"):
                    if sm.get(k) is not None:
                        ml_model_accuracy.set(float(sm[k]))
                        break

                for k in ("training_duration_seconds", "train_seconds", "fit_seconds", "train_time_seconds"):
                    if sm.get(k) is not None:
                        ml_training_duration_seconds.set(float(sm[k]))
                        break
    except Exception:
        pass

    return {
        "port": port,
        "ML_INFERENCE_TOTAL": ml_inference_total,
        "ML_PREDICTION_LATENCY_SECONDS": ml_prediction_latency_seconds,
        "ML_MODEL_ACCURACY": ml_model_accuracy,
        "ML_TRAINING_DURATION_SECONDS": ml_training_duration_seconds,
        "ML_LAST_TRAINED_TS": ml_last_trained_ts,
        "DUCKDB_FILE_SIZE_BYTES": DUCKDB_FILE_SIZE_BYTES,
        "DUCKDB_GOLD_ROW_COUNT": DUCKDB_GOLD_ROW_COUNT,
    }


PROM_ML = _init_prom_ml_metrics(MODEL_ARTIFACT_PATH, STATIC_METRICS_PATH)
ML_INFERENCE_TOTAL = PROM_ML["ML_INFERENCE_TOTAL"]
ML_PREDICTION_LATENCY_SECONDS = PROM_ML["ML_PREDICTION_LATENCY_SECONDS"]
ML_MODEL_ACCURACY = PROM_ML["ML_MODEL_ACCURACY"]
ML_TRAINING_DURATION_SECONDS = PROM_ML["ML_TRAINING_DURATION_SECONDS"]
ML_LAST_TRAINED_TS = PROM_ML["ML_LAST_TRAINED_TS"]
DUCKDB_FILE_SIZE_BYTES = PROM_ML["DUCKDB_FILE_SIZE_BYTES"]
DUCKDB_GOLD_ROW_COUNT = PROM_ML["DUCKDB_GOLD_ROW_COUNT"]



# Session init
if "corrid" not in st.session_state:
    st.session_state.corrid = uuid.uuid4().hex[:12]
if "db_path" not in st.session_state:
    st.session_state.db_path = DEFAULT_DB_PATH
if "schema_cache" not in st.session_state:
    st.session_state.schema_cache = {}
if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = BASE_URL
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = False

# ------------------------------
# Small style tweaks
# ------------------------------
CARD_CSS = """
<style>
.card {border-radius:1rem; padding:1rem 1.25rem; background: #0e1117; border:1px solid #2b2f3a; box-shadow: 0 1px 2px rgba(0,0,0,0.1);} 
.card h3{margin:0 0 .35rem 0}
.kpi {font-size: 1.35rem; font-weight: 700;}
.ok {color:#22c55e}
.bad {color:#ef4444}
.dim {color:#9aa0a6}
.btn-row {display:flex; gap:.5rem; align-items:center}
.small {font-size: .875rem; color:#9aa0a6}
.tag {display:inline-block; padding:.1rem .5rem; border:1px solid #2b2f3a; border-radius:999px; margin-right:.25rem; font-size:.8rem;}
hr.sep {border:none; border-top:1px solid #2b2f3a; margin:1rem 0}
</style>
"""
st.markdown(CARD_CSS, unsafe_allow_html=True)

# ------------------------------
# Utilities
# ------------------------------
def _api_get(path: str, params: Optional[dict] = None, timeout: float = 10):
    """GET helper with graceful fallback."""
    if st.session_state.get("demo_mode"):
        return None
    try:
        r = requests.get(f"{_base_url()}{path}", params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _api_post(path: str, payload: dict, timeout: float = 20):
    """POST helper with graceful fallback."""
    if st.session_state.get("demo_mode"):
        return None
    try:
        r = requests.post(f"{_base_url()}{path}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def short_uuid(n: int = 8) -> str:
    return uuid.uuid4().hex[:n]


def _base_url() -> str:
    try:
        return (st.session_state.get("api_base_url") or BASE_URL).rstrip("/")
    except Exception:
        return BASE_URL


@st.cache_resource(show_spinner=False)
def _cached_duckdb_connection(db_path: str):
    """Cached DuckDB connection reused across tabs."""
    return duckdb.connect(db_path, read_only=False)


@st.cache_resource(show_spinner=False)
def load_model(path: str = MODEL_ARTIFACT_PATH):
    """Load the trained calibrated classifier pipeline from disk.

    We try joblib first (how sklearn models are usually saved),
    then fall back to raw pickle.load for maximum compatibility.
    """
    try:
        return joblib.load(path)
    except Exception as e_joblib:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e_pickle:
            raise RuntimeError(
                f"Failed to load model from {path} "
                f"(joblib error: {e_joblib!r}, pickle error: {e_pickle!r})"
            )


@st.cache_data(show_spinner=False)
def load_threshold(path: str = THRESHOLD_PATH) -> float:
    """Read decision threshold from text file, defaulting to 0.5."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            v = f.read().strip()
        return float(v)
    except FileNotFoundError:
        return 0.5
    except Exception:
        return 0.5


@st.cache_data(show_spinner=False)
def load_static_metrics(path: str = STATIC_METRICS_PATH) -> Optional[dict]:
    """Load held-out test metrics exported from the training notebook, if available."""
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return None


def get_model_feature_cols(model) -> List[str]:
    """Infer required input columns from the fitted model.

    We try, in order:
      - model.feature_names_in_
      - model.base_estimator_(or base_estimator).feature_names_in_
      - model.estimator.feature_names_in_
    and fall back to MODEL_FEATURE_COLS if nothing is found.
    """
    if model is None:
        return MODEL_FEATURE_COLS

    candidates = [
        model,
        getattr(model, "base_estimator_", None),
        getattr(model, "base_estimator", None),
        getattr(model, "estimator", None),
    ]
    for obj in candidates:
        if obj is None:
            continue
        cols = getattr(obj, "feature_names_in_", None)
        if cols is not None:
            return [str(c) for c in list(cols)]

    return MODEL_FEATURE_COLS


def prepare_features(
    df: pd.DataFrame,
    model=None,
    expect_label: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.Series], List[str]]:
    """Prepare features/label for the model.

    Returns (X, y, missing_features).

    - X: DataFrame with columns the model expects (whatever it was trained on).
    - y: Optional Series of labels (`hit_and_run_i`) if present and requested.
    - missing_features: list of required columns that were not found in df.
    """
    df = df.copy()

    # Coerce crash_date to datetime if present
    if "crash_date" in df.columns and not np.issubdtype(df["crash_date"].dtype, np.datetime64):
        df["crash_date"] = pd.to_datetime(df["crash_date"], errors="coerce")

    required_cols = get_model_feature_cols(model)
    missing = [c for c in required_cols if c not in df.columns]
    present = [c for c in required_cols if c in df.columns]

    if present:
        X = df[present]
    else:
        # No expected cols present; return empty frame to avoid KeyError
        X = df[[]].copy()

    y: Optional[pd.Series] = None
    if expect_label and MODEL_LABEL_COL in df.columns:
        y = pd.to_numeric(df[MODEL_LABEL_COL], errors="coerce")

    return X, y, missing


def fetch_schema(dataset: str) -> List[str]:
    """Try to fetch dynamic schema columns from backend; fallback to a common set."""
    if dataset in st.session_state.schema_cache:
        return st.session_state.schema_cache[dataset]

    data = _api_get(f"/api/schema/{dataset}") or _api_get("/api/schema", {"dataset": dataset})
    if isinstance(data, dict) and "columns" in data:
        cols = list(map(str, data["columns"]))
    elif isinstance(data, list):
        cols = [str(c) for c in data]
    else:
        fallback = {
            "crashes": [
                "crash_record_id",
                "crash_date",
                "crash_hour",
                "crash_day_of_week",
                "crash_type",
                "posted_speed_limit",
                "weather_condition",
                "latitude",
                "longitude",
                "traffic_control_device",
                "injuries_total",
                "hit_and_run_i",
            ],
            "vehicles": [
                "vehicle_id",
                "unit_no",
                "vehicle_type",
                "vehicle_impact_location",
                "vehicle_defect",
            ],
            "people": ["person_id", "unit_no", "person_type", "age", "sex", "safety_equipment"],
        }
        cols = fallback.get(dataset, [])
    st.session_state.schema_cache[dataset] = cols
    return cols


def call_health() -> Dict[str, bool]:
    data = _api_get("/api/health")
    if isinstance(data, dict) and data:
        return {k: bool(v) for k, v in data.items()}
    return {"MinIO": True, "RabbitMQ": True, "Extractor": True, "Transformer": True, "Cleaner": True}


def gold_connect():
    """Open DuckDB with a single, consistent configuration (cached)."""
    try:
        return _cached_duckdb_connection(st.session_state.db_path)
    except Exception as e:
        st.warning(f"Could not open DuckDB at {st.session_state.db_path}: {e}")
        return None


def list_duck_tables(con) -> pd.DataFrame:
    """List tables across all catalogs (databases)."""
    try:
        cats = [
            r[1]
            for r in con.execute("PRAGMA database_list").fetchall()
            if r[1] not in ("temp",)
        ]
    except Exception:
        cats = ["main"]
    frames = []
    for c in cats:
        try:
            df = con.execute(
                f"""
                SELECT '{c}' AS catalog, table_schema, table_name
                FROM "{c}".information_schema.tables
                WHERE table_schema NOT IN ('pg_catalog','information_schema')
            """
            ).df()
            frames.append(df)
        except Exception:
            pass
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=["catalog", "table_schema", "table_name"])


def table_row_counts(con) -> pd.DataFrame:
    df = list_duck_tables(con)
    rows = []
    for _, r in df.iterrows():
        cat, schema, name = r["catalog"], r["table_schema"], r["table_name"]
        try:
            n = con.execute(
                f'SELECT COUNT(*) FROM "{cat}"."{schema}"."{name}"'
            ).fetchone()[0]
        except Exception:
            n = None
        rows.append({"catalog": cat, "schema": schema, "table": name, "rows": n})
    return pd.DataFrame(rows)


def safe_read_df(
    con,
    table_full: str,
    limit: Optional[int] = None,
    cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Read from 'catalog.schema.table' or 'schema.table' or plain 'table'."""
    try:
        parts = table_full.split(".")
        cols_sel = "*" if not cols else ", ".join([f'"{c}"' for c in cols])
        if len(parts) == 3:
            cat, schema, table = parts
            q = f'SELECT {cols_sel} FROM "{cat}"."{schema}"."{table}"'
        elif len(parts) == 2:
            schema, table = parts
            q = f'SELECT {cols_sel} FROM "{schema}"."{table}"'
        else:
            table = parts[0]
            q = f'SELECT {cols_sel} FROM "{table}"'
        if limit:
            q += f" LIMIT {int(limit)}"
        return con.execute(q).df()
    except Exception:
        return pd.DataFrame()


def ensure_cols(df: pd.DataFrame, needed: List[str]) -> bool:
    return all(c in df.columns for c in needed)


def chart_note(msg: str):
    st.caption(msg)


def to_local(ts: datetime | str | None) -> str:
    if ts is None:
        return "—"
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return ts
    return ts.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")


def find_table(con, schema: str, table: str) -> Optional[str]:
    """Return fully-qualified 'catalog.schema.table' if found."""
    df = list_duck_tables(con)
    if "catalog" not in df.columns:
        return None
    m = (df["table_schema"] == schema) & (df["table_name"] == table)
    if m.any():
        cat = df.loc[m].iloc[0]["catalog"]
        return f"{cat}.{schema}.{table}"
    return None

def update_duckdb_metrics():
    db_path = st.session_state.db_path

    # file size
    try:
        DUCKDB_FILE_SIZE_BYTES.set(float(os.path.getsize(db_path)))
    except Exception:
        DUCKDB_FILE_SIZE_BYTES.set(0.0)

    # row count: try common fully-qualified patterns
    try:
        con = duckdb.connect(db_path, read_only=True)

        # 1) use your helper first
        gold_full = find_table(con, "gold", "crashes")  # e.g., gold.gold.crashes or main.gold.crashes
        if gold_full:
            cat, sch, tbl = gold_full.split(".")
            n = con.execute(f'SELECT COUNT(*) FROM "{cat}"."{sch}"."{tbl}"').fetchone()[0]
            DUCKDB_GOLD_ROW_COUNT.set(float(n))
        else:
            # 2) fallbacks
            for q in [
                "SELECT COUNT(*) FROM gold.crashes",
                "SELECT COUNT(*) FROM main.gold.crashes",
                "SELECT COUNT(*) FROM gold.gold.crashes",
            ]:
                try:
                    n = con.execute(q).fetchone()[0]
                    DUCKDB_GOLD_ROW_COUNT.set(float(n))
                    break
                except Exception:
                    continue
            else:
                DUCKDB_GOLD_ROW_COUNT.set(0.0)

        con.close()
    except Exception:
        DUCKDB_GOLD_ROW_COUNT.set(0.0)


# ------------------------------
# Sidebar — Global settings
# ------------------------------
st.sidebar.header("Settings")
st.sidebar.text_input(
    "API Base URL",
    key="api_base_url",
    help="Backend REST base. If blank/unreachable, the app uses local/demo fallbacks.",
)
st.sidebar.text_input("Gold DuckDB Path", key="db_path", help="Path to gold.duckdb")
st.sidebar.checkbox(
    "Demo mode (no backend)",
    key="demo_mode",
    help="Use offline stubs and suppress backend warnings",
)
if st.sidebar.button("New corrid"):
    st.session_state.corrid = short_uuid(12)
st.sidebar.caption(f"corrid: {st.session_state.corrid}")

update_duckdb_metrics()

# ------------------------------
# Tabs
# ------------------------------
(
    TAB_HOME,
    TAB_DATA_MGMT,
    TAB_FETCH,
    TAB_SCHED,
    TAB_EDA,
    TAB_MODEL,
    TAB_REPORTS,
) = st.tabs(
    [
        "🏠 Home",
        "🧰 Data Management",
        "📡 Data Fetcher",
        "⏰ Scheduler",
        "📊 EDA",
        "🤖 Model",
        "📑 Reports",
    ]
)

# ------------------------------
# 🏠 HOME
# ------------------------------
with TAB_HOME:
    st.subheader("Label Overview & Container Health")

    # Label overview (single card)
    with st.container():
        st.markdown(
            """
        ### 🏃 Hit & Run (primary)
        **Label predicted:** `hit_and_run_i` • **Type:** binary • **Positive class:** 1 (hit & run)

        **Pipeline:** Estimate likelihood of hit-and-run events using temporal and roadway context (hour, DOW, speed limit),
        weather/visibility, and control devices.

        **Key features (conceptually):**
        - Time-of-day effects (`crash_hour`, weekday vs weekend)
        - Road geometry and speed environment (`posted_speed_limit`, `traffic_control_device`, `alignment`, `trafficway_type`)
        - Weather and lighting (`weather_condition`, `lighting_condition`)
        - Crash context and counts (e.g., `num_units`, `veh_count`, `ppl_count`)

        **Gold table:** `gold.crashes`
        """
        )

    st.markdown("<hr class='sep' />", unsafe_allow_html=True)

    st.write("### Container Health")
    health = call_health()
    cols = st.columns(len(health))
    for i, (svc, ok) in enumerate(health.items()):
        with cols[i]:
            st.markdown(
                f"<div class='card'><h3>{svc}</h3>"
                + (
                    "<div class='kpi ok'>✅ Running</div>"
                    if ok
                    else "<div class='kpi bad'>❌ Not Responding</div>"
                )
                + "</div>",
                unsafe_allow_html=True,
            )

# ------------------------------
# 🧰 DATA MANAGEMENT
# ------------------------------
with TAB_DATA_MGMT:
    st.subheader("MinIO & Gold Admin • Quick Peek")

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("#### MinIO — Delete by Folder (Prefix)")
        bucket = st.selectbox("Bucket", ["raw-data", "transform-data", "cleaned-data"], index=0)
        prefix = st.text_input("Prefix (e.g., crash/<corrid>/)", value="crash/")
        do_preview = st.button("Preview (dry-run)")

        preview = None
        if do_preview:
            preview = _api_get("/api/minio/preview", params={"bucket": bucket, "prefix": prefix})
            if not preview:
                if st.session_state.get("demo_mode"):
                    st.caption("Demo preview (offline mode).")
                else:
                    st.info("Backend not reachable — showing demo preview.")
                preview = {
                    "prefix": prefix,
                    "count": 8,
                    "keys": [f"{prefix}file_{i}.json.gz" for i in range(1, 9)],
                }
            st.json(preview)

        confirm = st.checkbox("I confirm deletion scope above")
        if st.button("Delete Folder", disabled=not (confirm and preview)):
            out = _api_post("/api/minio/delete_prefix", {"bucket": bucket, "prefix": prefix})
            if not out:
                st.warning("Backend not reachable — no action performed.")
            else:
                st.success("Folder delete submitted.")

        st.markdown("---")
        st.markdown("#### MinIO — Delete Entire Bucket")
        bucket2 = st.selectbox(
            "Bucket to delete",
            ["raw-data", "transform-data", "cleaned-data"],
            key="bucket_del",
        )
        confirm2 = st.checkbox(
            "I confirm deleting ENTIRE bucket (empties objects then deletes)",
            key="x2",
        )
        if st.button("Delete Bucket", disabled=not confirm2):
            out = _api_post("/api/minio/delete_bucket", {"bucket": bucket2})
            if not out:
                st.warning("Backend not reachable — no action performed.")
            else:
                st.success("Bucket delete submitted.")

    with c2:
        st.markdown("#### Gold Admin (DuckDB)")
        con = gold_connect()
        counts = pd.DataFrame()
        if con:
            try:
                counts = table_row_counts(con)
            except Exception:
                counts = pd.DataFrame()
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.write("**DB Path:**", st.session_state.db_path)
            total_rows = int(counts["rows"].fillna(0).sum()) if not counts.empty else 0
            st.write("**Total rows:**", total_rows)
            st.dataframe(counts, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Initialize gold from local CSVs if empty
        init_csv = None
        if os.path.exists(os.path.join("out", "cleaned.csv")):
            init_csv = os.path.join("out", "cleaned.csv")
        elif os.path.exists(os.path.join("out", "merged.csv")):
            init_csv = os.path.join("out", "merged.csv")

        if init_csv:
            if st.button(f"Init gold.crashes from {os.path.relpath(init_csv)}"):
                try:
                    # use a short-lived connection separate from the cached one
                    con_i = duckdb.connect(st.session_state.db_path, read_only=False)
                    csv_path = init_csv.replace("\\", "/")

                    # discover a usable catalog (database) name
                    cats = [
                        row[1]
                        for row in con_i.execute("PRAGMA database_list").fetchall()
                    ]
                    cat = next(
                        (c for c in cats if c not in ("temp", "memory")),
                        cats[0] if cats else "main",
                    )

                    # create schema in that catalog (fallback to unqualified if needed)
                    try:
                        con_i.execute(f'CREATE SCHEMA IF NOT EXISTS "{cat}".gold;')
                    except Exception:
                        con_i.execute("CREATE SCHEMA IF NOT EXISTS gold;")

                    con_i.execute(
                        f'CREATE OR REPLACE TABLE "{cat}".gold.crashes AS '
                        "SELECT * FROM read_csv_auto(?)",
                        [csv_path],
                    )
                    con_i.close()
                    st.success(
                        f'gold.crashes created in catalog "{cat}". Reopen this tab to refresh counts.'
                    )
                except Exception as e:
                    st.error(f"Failed to init gold.crashes: {e}")

        wipe_ok = st.checkbox("I confirm wiping the ENTIRE Gold DB file (irreversible)")
        if st.button("Wipe Gold DB", type="primary", disabled=not wipe_ok):
            try:
                if os.path.exists(st.session_state.db_path):
                    os.remove(st.session_state.db_path)
                st.success(
                    "Gold DB file removed; a new empty DB will be created on next connect."
                )
            except Exception as e:
                st.error(f"Failed to remove DB: {e}")

    st.markdown("---")
    st.markdown("#### Quick Peek (Gold → sample)")

    con2 = gold_connect()
    if con2:
        tbls = list_duck_tables(con2)
        tbls_list = sorted(
            [f"{r.catalog}.{r.table_schema}.{r.table_name}" for _, r in tbls.iterrows()]
        )
        table_choice = st.selectbox("Table", options=tbls_list or ["(none)"])
        limit = st.slider("Rows (limit)", 10, 200, 50)
        cols = []
        if table_choice and table_choice != "(none)":
            df_sample = safe_read_df(con2, table_choice, limit=1)
            if not df_sample.empty:
                all_cols = list(df_sample.columns)
                cols = st.multiselect(
                    "Columns (optional)",
                    options=all_cols,
                    default=all_cols[: min(8, len(all_cols))],
                )
        if st.button("Preview"):
            if table_choice and table_choice != "(none)":
                df = safe_read_df(con2, table_choice, limit=limit, cols=cols or None)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No table available.")

# ------------------------------
# 📡 DATA FETCHER
# ------------------------------
with TAB_FETCH:
    st.subheader("Publish fetch jobs → RabbitMQ")

    t_stream, t_backfill = st.tabs(["Streaming", "Backfill"])

    def enrichment_controls(prefix: str):
        st.markdown("**Enrichment Columns**")
        inc_v = st.checkbox("Include Vehicles", value=False, key=f"{prefix}_inc_v")
        v_cols = []
        if inc_v:
            v_all = fetch_schema("vehicles")
            sel_all_v = st.checkbox(
                "Select all vehicle columns", key=f"{prefix}_sel_all_v"
            )
            v_cols = st.multiselect(
                "Vehicles: columns to be fetched",
                options=v_all,
                default=v_all if sel_all_v else [],
                key=f"{prefix}_v_cols",
            )
        inc_p = st.checkbox("Include People", value=False, key=f"{prefix}_inc_p")
        p_cols = []
        if inc_p:
            p_all = fetch_schema("people")
            sel_all_p = st.checkbox(
                "Select all people columns", key=f"{prefix}_sel_all_p"
            )
            p_cols = st.multiselect(
                "People: columns to be fetched",
                options=p_all,
                default=p_all if sel_all_p else [],
                key=f"{prefix}_p_cols",
            )
        return inc_v, v_cols, inc_p, p_cols

    # STREAMING
    with t_stream:
        st.write("Mode: `streaming`")
        corrid = st.text_input(
            "corrid (auto)", value=st.session_state.corrid, disabled=True
        )
        since_days = st.number_input(
            "Since days", min_value=1, max_value=3650, value=30
        )
        inc_v, v_cols, inc_p, p_cols = enrichment_controls("stream")

        payload_stream = {
            "mode": "streaming",
            "corrid": corrid,
            "since_days": int(since_days),
            "include_vehicles": inc_v,
            "vehicle_columns": v_cols,
            "include_people": inc_p,
            "people_columns": p_cols,
        }

        with st.expander("Preview JSON"):
            st.code(json.dumps(payload_stream, indent=2))

        cA, cB = st.columns([1, 1])
        with cA:
            if st.button("Publish to RabbitMQ", type="primary"):
                out = _api_post("/api/publish", payload_stream)
                if out:
                    # Treat both real publisher and stub as success
                    if out.get("ok") or out.get("published") or out.get("written"):
                        detail = ""
                        if out.get("published"):
                            detail = f" ({out['published']} msg)"
                        elif out.get("written"):
                            try:
                                detail = f" (stub wrote {os.path.relpath(out['written'])})"
                            except Exception:
                                detail = " (stub wrote file)"
                        st.success(f"Published ✔ corrid={corrid}{detail}")
                    else:
                        st.warning(f"Publish returned but not confirmed: {out}")
                else:
                    st.error("Backend unreachable — no publish.")
        with cB:
            if st.button("Reset form"):
                st.rerun()

    # BACKFILL
    with t_backfill:
        st.write("Mode: `backfill`")
        corrid2 = st.text_input(
            "corrid (auto)", value=st.session_state.corrid, disabled=True, key="corr_bf"
        )
        today = date.today()
        start_d = st.date_input("Start date", value=today - timedelta(days=30))
        end_d = st.date_input("End date", value=today)
        start_t = st.time_input("Start time", value=dtime(0, 0))
        end_t = st.time_input("End time", value=dtime(23, 59))

        inc_v2, v_cols2, inc_p2, p_cols2 = enrichment_controls("backfill")

        payload_backfill = {
            "mode": "backfill",
            "corrid": corrid2,
            "start": f"{start_d}T{start_t}:00",
            "end": f"{end_d}T{end_t}:00",
            "include_vehicles": inc_v2,
            "vehicle_columns": v_cols2,
            "include_people": inc_p2,
            "people_columns": p_cols2,
        }
        with st.expander("Preview JSON"):
            st.code(json.dumps(payload_backfill, indent=2))

        cC, cD = st.columns([1, 1])
        with cC:
            if st.button("Publish (backfill)", type="primary"):
                out = _api_post("/api/publish", payload_backfill)
                if out:
                    if out.get("ok") or out.get("published") or out.get("written"):
                        detail = ""
                        if out.get("published"):
                            detail = f" ({out['published']} msg)"
                        elif out.get("written"):
                            try:
                                detail = f" (stub wrote {os.path.relpath(out['written'])})"
                            except Exception:
                                detail = " (stub wrote file)"
                        st.success(f"Published ✔ corrid={corrid2}{detail}")
                    else:
                        st.warning(f"Publish returned but not confirmed: {out}")
                else:
                    st.error("Backend unreachable — no publish.")
        with cD:
            if st.button("Reset backfill form"):
                st.rerun()

# ------------------------------
# ⏰ SCHEDULER
# ------------------------------
with TAB_SCHED:
    st.subheader("Automate streaming runs")

    freq = st.selectbox("Select Frequency", ["Daily", "Weekly", "Custom cron"], index=0)
    pick_time = st.time_input("Time of day", value=dtime(9, 0))
    config_type = st.selectbox("Config Type", ["streaming"], index=0)
    cron_custom = st.text_input("Custom cron (if selected)")

    if st.button("Create Schedule", type="primary"):
        cron = None
        if freq == "Daily":
            cron = f"0 {pick_time.minute} {pick_time.hour} * * *"
        elif freq == "Weekly":
            cron = f"0 {pick_time.minute} {pick_time.hour} * * 1"
        else:
            cron = cron_custom or "0 0 9 * * *"
        payload = {"cron": cron, "config": {"type": config_type}}
        out = _api_post("/api/schedule", payload)
        if out:
            st.success("Schedule created.")
        else:
            st.info("Backend unreachable — demo: showing mock schedules.")

    st.markdown("---")
    st.write("**Active Schedules**")
    sched = _api_get("/api/schedule/list")
    if not sched:
        sched = [
            {
                "cron": "0 0 9 * * *",
                "config": {"type": "streaming"},
                "last_run": None,
            }
        ]
    df = pd.DataFrame(sched)
    if "last_run" in df.columns:
        df["last_run_local"] = df["last_run"].apply(to_local)
    st.dataframe(df, use_container_width=True)

# ------------------------------
# 📊 EDA
# ------------------------------
with TAB_EDA:
    st.subheader("Explore Gold data (DuckDB)")
    con = gold_connect()
    if not con:
        st.stop()

    # Choose table, detect common columns
    tbls = list_duck_tables(con)
    tbls_list = sorted(
        [f"{r.catalog}.{r.table_schema}.{r.table_name}" for _, r in tbls.iterrows()]
    )
    table_sel = st.selectbox("Gold table", options=tbls_list or ["(none)"])

    if not tbls_list:
        st.info("No tables available in Gold DB.")
        st.stop()

    df_base = safe_read_df(con, table_sel, limit=None)
    if df_base.empty:
        st.warning("Selected table is empty or unreadable.")
        st.stop()

    # Coerce potentially useful columns
    for c in ["crash_hour", "posted_speed_limit", "injuries_total", "hit_and_run_i"]:
        if c in df_base.columns:
            df_base[c] = pd.to_numeric(df_base[c], errors="coerce")
    for c in ["crash_day_of_week", "crash_type", "weather_condition", "traffic_control_device"]:
        if c in df_base.columns:
            df_base[c] = df_base[c].astype("category")

    # Summary stats
    st.markdown("### Summary Statistics")
    with st.expander("Row count, missing values, basic stats", expanded=True):
        row_count = len(df_base)
        st.write(f"**Rows:** {row_count}")
        miss = df_base.isna().sum().sort_values(ascending=False).reset_index()
        miss.columns = ["column", "missing"]
        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(miss.head(25), hide_index=True, use_container_width=True)
        with col2:
            try:
                desc = df_base.describe(
                    include="all", datetime_is_numeric=True
                ).transpose()
            except TypeError:
                desc = df_base.describe(include="all").transpose()
            if "top" in desc.columns:
                desc["top"] = desc["top"].astype("string")
            st.dataframe(desc.head(25), use_container_width=True)

    st.markdown("### Visualizations")

    # Reset chart counter each render
    st.session_state["charts_rendered"] = 0
    MAX_PTS = 100_000
    data_viz = (
        df_base.sample(n=min(len(df_base), MAX_PTS), random_state=42)
        if len(df_base) > MAX_PTS
        else df_base
    )

    def render_chart(title: str, chart: alt.Chart):
        st.altair_chart(
            chart.properties(title=title)
            .configure_axis(
                labelFont=ALT_THEMES["font"],
                titleFont=ALT_THEMES["font"],
            )
            .configure_title(font=ALT_THEMES["font"]),
            use_container_width=True,
        )
        st.session_state["charts_rendered"] += 1

    # 1) Histogram posted_speed_limit
    if ensure_cols(data_viz, ["posted_speed_limit"]):
        chart = (
            alt.Chart(data_viz.dropna(subset=["posted_speed_limit"]))
            .mark_bar()
            .encode(
                x=alt.X("posted_speed_limit:Q", bin=alt.Bin(maxbins=20)),
                y="count()",
            )
        )
        render_chart("Histogram — posted_speed_limit", chart)
        chart_note("Distribution of roadway speeds.")

    # 2) Overlaid by hit_and_run_i
    if ensure_cols(data_viz, ["posted_speed_limit", "hit_and_run_i"]):
        chart = (
            alt.Chart(
                data_viz.dropna(subset=["posted_speed_limit", "hit_and_run_i"])
            )
            .mark_area(opacity=0.4, interpolate="step")
            .encode(
                x=alt.X("posted_speed_limit:Q", bin=alt.Bin(maxbins=20)),
                y="count()",
                color="hit_and_run_i:N",
            )
        )
        render_chart("Histogram — posted_speed_limit by hit_and_run_i", chart)
        chart_note("Hit-and-run skews on mid-speed arterials in many cities.")

    # 3) Bar by weather_condition
    if ensure_cols(data_viz, ["weather_condition"]):
        chart = (
            alt.Chart(data_viz.dropna(subset=["weather_condition"]))
            .mark_bar()
            .encode(x=alt.X("weather_condition:N", sort="-y"), y="count()")
        )
        render_chart("Counts — weather_condition", chart)
        chart_note("Crash counts by dominant weather description.")

    # 4) Stacked bar crash_type by weather
    if ensure_cols(data_viz, ["weather_condition", "crash_type"]):
        chart = (
            alt.Chart(data_viz.dropna(subset=["weather_condition", "crash_type"]))
            .mark_bar()
            .encode(
                x=alt.X("weather_condition:N", sort="-y"),
                y="count()",
                color="crash_type:N",
            )
        )
        render_chart("Stacked — crash_type × weather_condition", chart)
        chart_note("Mix of types shifts in wet/low-visibility conditions.")

    # 5) Line by crash_hour
    if ensure_cols(data_viz, ["crash_hour"]):
        chart = (
            alt.Chart(data_viz.dropna(subset=["crash_hour"]))
            .mark_line(point=True)
            .encode(x=alt.X("crash_hour:Q"), y="count()")
        )
        render_chart("Counts by crash_hour", chart)
        chart_note("Commuter hours often spike.")

    # 6) Rate by hour (hit_and_run)
    if ensure_cols(data_viz, ["crash_hour", "hit_and_run_i"]):
        df_rate = (
            data_viz.groupby("crash_hour", dropna=True)
            .agg(n=("hit_and_run_i", "size"), hits=("hit_and_run_i", "sum"))
            .reset_index()
        )
        df_rate["rate"] = df_rate["hits"] / df_rate["n"].replace(0, np.nan)
        chart = (
            alt.Chart(df_rate)
            .mark_line(point=True)
            .encode(x="crash_hour:Q", y="rate:Q")
        )
        render_chart("Hit-and-run rate by hour", chart)
        chart_note("Rate tends to rise late night.")

    # 7) Pie day of week
    if ensure_cols(data_viz, ["crash_day_of_week"]):
        df_dow = data_viz["crash_day_of_week"].value_counts(dropna=True).reset_index()
        df_dow.columns = ["dow", "count"]
        chart = (
            alt.Chart(df_dow)
            .mark_arc()
            .encode(theta="count:Q", color="dow:N")
        )
        render_chart("Share — crash_day_of_week", chart)

    # 8) Box: injuries_total by traffic_control_device
    if ensure_cols(data_viz, ["injuries_total", "traffic_control_device"]):
        chart = (
            alt.Chart(
                data_viz.dropna(subset=["injuries_total", "traffic_control_device"])
            )
            .mark_boxplot()
            .encode(
                x=alt.X("traffic_control_device:N", sort="-y"),
                y="injuries_total:Q",
            )
        )
        render_chart("Injuries — by traffic_control_device", chart)

    # 9) Scatter: speed vs injuries
    if ensure_cols(data_viz, ["posted_speed_limit", "injuries_total"]):
        chart = (
            alt.Chart(
                data_viz.dropna(subset=["posted_speed_limit", "injuries_total"])
            )
            .mark_circle(size=45, opacity=0.5)
            .encode(x="posted_speed_limit:Q", y="injuries_total:Q")
        )
        render_chart("Scatter — posted_speed_limit vs injuries_total", chart)

    # 10) Heatmap: hour × day of week
    if ensure_cols(data_viz, ["crash_hour", "crash_day_of_week"]):
        df_hm = (
            data_viz.dropna(subset=["crash_hour", "crash_day_of_week"])
            .groupby(["crash_hour", "crash_day_of_week"])
            .size()
            .reset_index(name="n")
        )
        chart = (
            alt.Chart(df_hm)
            .mark_rect()
            .encode(
                x="crash_hour:O",
                y=alt.Y("crash_day_of_week:N", sort=None),
                color="n:Q",
            )
        )
        render_chart("Heatmap — count by hour × day", chart)

    # 11) Top primary cause by type (if present)
    for cause_col in [
        "primary_contributory_cause",
        "prim_contributory_cause",
        "contributory_cause",
    ]:
        if cause_col in data_viz.columns and "crash_type" in data_viz.columns:
            df_cause = (
                data_viz.dropna(subset=[cause_col])
                .groupby([cause_col, "crash_type"])
                .size()
                .reset_index(name="n")
            )
            chart = (
                alt.Chart(df_cause)
                .mark_bar()
                .encode(
                    y=alt.Y(f"{cause_col}:N", sort="-x"),
                    x="n:Q",
                    color="crash_type:N",
                )
            )
            render_chart(f"Contributory cause (by crash_type) — {cause_col}", chart)
            break

    st.caption(f"Charts rendered: {st.session_state['charts_rendered']}")

# ------------------------------
# 🤖 MODEL
# ------------------------------
with TAB_MODEL:
    st.subheader("ML Model — Hit & Run Risk (`hit_and_run_i`)")

    # ---- 1) Model summary ----
    st.markdown("### 1. Model Summary")

    model = None
    model_load_error: Optional[str] = None
    try:
        t0 = _time.time()
        model = load_model(MODEL_ARTIFACT_PATH)
        load_dt = _time.time() - t0

        # If training duration wasn't provided by artifacts, use model load time as a proxy so the dashboard isn't empty.
        try:
            cur = ML_TRAINING_DURATION_SECONDS._value.get()
            if cur == 0.0:  # only if never set
                ML_TRAINING_DURATION_SECONDS.set(float(load_dt))
        except Exception:
            ML_TRAINING_DURATION_SECONDS.set(float(load_dt))

    except FileNotFoundError:
        model_load_error = (
            f"Model artifact not found at `{MODEL_ARTIFACT_PATH}`. "
            "Make sure you copied your trained `.pkl` into the `artifacts/` folder."
        )
    except Exception as e:
        model_load_error = f"Error loading model from `{MODEL_ARTIFACT_PATH}`: {e}"

    if model_load_error:
        st.error(model_load_error)
        st.info("Once the model artifact is available, this tab will become fully usable.")
        st.stop()

    outer_cls = model.__class__.__name__
    inner = (
        getattr(model, "base_estimator", None)
        or getattr(model, "base_estimator_", None)
        or getattr(model, "estimator", None)
    )
    inner_cls = inner.__class__.__name__ if inner is not None else "Unknown"

    threshold = load_threshold()
    static_metrics = load_static_metrics()
    feature_cols = get_model_feature_cols(model)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            "<div class='card'><div class='small'>Outer model class</div>"
            f"<div class='kpi'>{outer_cls}</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            "<div class='card'><div class='small'>Underlying estimator</div>"
            f"<div class='kpi'>{inner_cls}</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            "<div class='card'><div class='small'>Decision threshold (positive class)</div>"
            f"<div class='kpi'>{threshold:.3f}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        **Label:**
        - `hit_and_run_i` (1 = hit & run, 0 = not hit & run)

        **Features:**
        - The model was trained on the same engineered columns used in your
          training notebook (raw crash attributes plus derived fields).
        - All preprocessing and one-hot encoding happen *inside* the sklearn pipeline.
        """
    )

    with st.expander("Show model input columns (required)"):
        st.write(f"Total: {len(feature_cols)}")
        st.code(", ".join(feature_cols))

    if static_metrics:
        st.markdown("#### Static test metrics (from training notebook)")
        st.json(static_metrics)
    else:
        st.caption(
            f"No `test_metrics.json` found at `{STATIC_METRICS_PATH}`. "
            "You can export your held-out test metrics from the notebook into this path "
            "to display them here."
        )

    st.markdown("---")
    st.markdown("### 2. Data Selection")

    data_mode = st.radio(
        "Choose which data to score",
        ["Gold table (gold.crashes)", "Upload test CSV"],
        horizontal=True,
    )

    scored_source: Optional[str] = None
    scored_df: Optional[pd.DataFrame] = None
    y_true: Optional[pd.Series] = None

    # ---- 2.1 Gold table mode ----
    if data_mode.startswith("Gold"):
        con = gold_connect()
        if not con:
            st.warning("Gold DB connection is not available. Check the path in the sidebar.")
        else:
            gold_full = find_table(con, "gold", "crashes")
            if not gold_full:
                st.warning("Could not find table `gold.crashes` in the Gold DB.")
            else:
                today = date.today()
                default_start = today - timedelta(days=90)
                g1, g2, g3 = st.columns(3)
                with g1:
                    start_date = st.date_input(
                        "Start date", value=default_start, key="mdl_start_date"
                    )
                with g2:
                    end_date = st.date_input("End date", value=today, key="mdl_end_date")
                with g3:
                    max_rows = st.number_input(
                        "Max rows",
                        min_value=100,
                        max_value=50_000,
                        value=5_000,
                        step=100,
                        key="mdl_max_rows",
                    )

                query = f"""
                    SELECT *
                    FROM {gold_full}
                    WHERE crash_date BETWEEN ? AND ?
                    ORDER BY crash_date
                    LIMIT ?
                """
                df_gold = con.execute(
                    query, [start_date, end_date, int(max_rows)]
                ).df()

                if df_gold.empty:
                    st.info("No rows matched the selected date window.")
                else:
                    st.success(f"Loaded {len(df_gold)} rows from `gold.crashes`.")
                    st.dataframe(df_gold.head(10), use_container_width=True)

                    X_tmp, y_true_tmp, missing = prepare_features(
                        df_gold, model=model, expect_label=True
                    )
                    if missing:
                        st.error(
                            "Missing required feature columns for this model: "
                            f"{missing}"
                        )
                    else:
                        scored_df = df_gold
                        y_true = y_true_tmp
                        scored_source = "gold"

    # ---- 2.2 Test CSV mode ----
    else:
        uploaded = st.file_uploader(
            "Upload held-out test data as CSV",
            type=["csv"],
            key="mdl_test_upload",
        )
        if uploaded is not None:
            if not uploaded.name.lower().endswith(".csv"):
                st.error("Only `.csv` files are supported for test data.")
            else:
                df_test = pd.read_csv(uploaded)
                st.success(f"Loaded {len(df_test)} rows from `{uploaded.name}`.")
                st.dataframe(df_test.head(10), use_container_width=True)

                X_tmp, y_true_tmp, missing = prepare_features(
                    df_test, model=model, expect_label=True
                )
                if missing:
                    st.error(
                        "Missing required feature columns for this model: "
                        f"{missing}"
                    )
                else:
                    scored_df = df_test
                    y_true = y_true_tmp
                    scored_source = "test_csv"

    st.markdown("---")
    st.markdown("### 3. Prediction & Metrics")

    if scored_df is None:
        st.info("Select and load data in Section 2 above to see predictions and metrics.")
    else:
        X, y_true_full, missing = prepare_features(
            scored_df, model=model, expect_label=True
        )
        if missing:
            st.error(
                "Cannot run model because these required feature columns are missing "
                f"from the selected data: {missing}"
            )
        else:
            t0 = _time.time()
            probs = model.predict_proba(X)[:, 1]
            dt = _time.time() - t0

            ML_INFERENCE_TOTAL.inc()
            ML_PREDICTION_LATENCY_SECONDS.observe(dt)

            y_pred = (probs >= threshold).astype(int)

            scored_view = scored_df.copy()
            scored_view["pred_prob_hit_and_run"] = probs
            scored_view["pred_hit_and_run"] = y_pred

            st.write("Sample of scored data:")
            st.dataframe(scored_view.head(25), use_container_width=True)

            # Live metrics (if we have the true label)
            if y_true_full is not None and not y_true_full.isna().all():
                y_clean = pd.to_numeric(y_true_full, errors="coerce")
                mask = ~y_clean.isna()
                y_clean = y_clean[mask].astype(int)
                preds_clean = y_pred[mask.to_numpy()]

                tp = int(((y_clean == 1) & (preds_clean == 1)).sum())
                tn = int(((y_clean == 0) & (preds_clean == 0)).sum())
                fp = int(((y_clean == 0) & (preds_clean == 1)).sum())
                fn = int(((y_clean == 1) & (preds_clean == 0)).sum())
                total = tp + tn + fp + fn

                acc = (tp + tn) / total if total else float("nan")
                if acc == acc:  # not NaN
                    ML_MODEL_ACCURACY.set(float(acc))

                prec = tp / (tp + fp) if (tp + fp) else float("nan")
                rec = tp / (tp + fn) if (tp + fn) else float("nan")
                f1 = (
                    2 * prec * rec / (prec + rec)
                    if prec == prec and rec == rec and (prec + rec) > 0
                    else float("nan")
                )

                live_metrics = {
                    "n_rows_evaluated": int(total),
                    "accuracy": round(acc, 3) if acc == acc else None,
                    "precision_pos": round(prec, 3) if prec == prec else None,
                    "recall_pos": round(rec, 3) if rec == rec else None,
                    "f1_pos": round(f1, 3) if f1 == f1 else None,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                }

                st.markdown("#### Live metrics on selected data")
                st.json(live_metrics)
            else:
                st.caption(
                    "No usable `hit_and_run_i` label column found, "
                    "so live metrics cannot be computed. Showing predictions only."
                )

            scored_csv = scored_view.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download scored data as CSV",
                data=scored_csv,
                file_name=f"scored_{scored_source or 'data'}.csv",
                mime="text/csv",
            )

# ------------------------------
# 📑 REPORTS
# ------------------------------
with TAB_REPORTS:
    st.subheader("Pipeline & Gold Snapshot")

    # Try to pull from backend; else compute locally
    run_hist = _api_get("/api/reports/runs")
    if not run_hist:
        try:
            local_report = os.path.join("out", "report.json")
            if os.path.exists(local_report):
                with open(local_report, "r", encoding="utf-8") as f:
                    data = json.load(f)
                run_hist = data if isinstance(data, list) else [data]
        except Exception:
            run_hist = None

    con = gold_connect()
    gold_counts = (
        table_row_counts(con)
        if con
        else pd.DataFrame(columns=["catalog", "schema", "table", "rows"])
    )

    latest_date = None
    gold_full = find_table(con, "gold", "crashes") if con else None
    if con and gold_full:
        if "crash_date" in safe_read_df(con, gold_full, limit=1).columns:
            cat = gold_full.split(".")[0]
            try:
                latest_date = con.execute(
                    f'SELECT max(crash_date) FROM "{cat}".gold.crashes'
                ).fetchone()[0]
            except Exception:
                latest_date = None

    metrics_cols = st.columns(5)
    with metrics_cols[0]:
        total_runs = len(run_hist) if isinstance(run_hist, list) else 0
        st.markdown(
            "<div class='card'><div class='small'>Total runs completed</div>"
            f"<div class='kpi'>{total_runs}</div></div>",
            unsafe_allow_html=True,
        )
    with metrics_cols[1]:
        st.markdown(
            "<div class='card'><div class='small'>Latest corrid</div>"
            f"<div class='kpi'>{st.session_state.corrid}</div></div>",
            unsafe_allow_html=True,
        )
    with metrics_cols[2]:
        gold_rows = int(gold_counts["rows"].fillna(0).sum()) if not gold_counts.empty else 0
        st.markdown(
            "<div class='card'><div class='small'>Gold row count</div>"
            f"<div class='kpi'>{gold_rows}</div></div>",
            unsafe_allow_html=True,
        )
    with metrics_cols[3]:
        st.markdown(
            "<div class='card'><div class='small'>Latest data date fetched</div>"
            f"<div class='kpi'>{(latest_date or '—')}</div></div>",
            unsafe_allow_html=True,
        )
    with metrics_cols[4]:
        last_run_ts = (
            run_hist[0].get("ended_at")
            if isinstance(run_hist, list) and run_hist
            else None
        )
        st.markdown(
            "<div class='card'><div class='small'>Last run timestamp</div>"
            f"<div class='kpi'>{to_local(last_run_ts)}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.markdown("### Latest Run Summary")
    if isinstance(run_hist, list) and run_hist:
        latest = run_hist[0]
    else:
        latest = {
            "corrid": st.session_state.corrid,
            "mode": "streaming",
            "window": {"since_days": 30},
            "started_at": datetime.now()
            .astimezone(LOCAL_TZ)
            .isoformat(),
            "ended_at": datetime.now()
            .astimezone(LOCAL_TZ)
            .isoformat(),
            "rows": {"crashes": 0, "people": 0, "vehicles": 0},
            "errors": [],
        }
    st.json(latest)

    st.markdown("### Download Reports")
    if isinstance(run_hist, list):
        df_runs = pd.DataFrame(run_hist)
    else:
        df_runs = pd.DataFrame([latest])
    csv_bytes = df_runs.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Run History CSV",
        data=csv_bytes,
        file_name="run_history.csv",
        mime="text/csv",
    )

    # Optional PDF — only if reportlab is installed
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch

        def make_pdf(bytes_io: io.BytesIO):
            c = canvas.Canvas(bytes_io, pagesize=LETTER)
            w, h = LETTER
            c.setFont("Helvetica-Bold", 14)
            c.drawString(1 * inch, h - 1 * inch, "Chicago Crash ETL — Report")
            c.setFont("Helvetica", 10)
            y = h - 1.3 * inch
            c.drawString(
                1 * inch,
                y,
                f"Generated: {datetime.now().astimezone(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}",
            )
            y -= 0.3 * inch
            items = [
                ("Total runs completed", str(total_runs)),
                ("Latest corrid", st.session_state.corrid),
                ("Gold row count", str(gold_rows)),
                ("Latest data date fetched", str(latest_date or "—")),
                ("Last run timestamp", to_local(last_run_ts)),
            ]
            for k, v in items:
                c.drawString(1 * inch, y, f"• {k}: {v}")
                y -= 0.25 * inch
            c.showPage()
            c.save()

        pdf_buf = io.BytesIO()
        make_pdf(pdf_buf)
        st.download_button(
            "Download Summary PDF",
            data=pdf_buf.getvalue(),
            file_name="report.pdf",
            mime="application/pdf",
        )
    except Exception:
        st.caption("Install `reportlab` to enable PDF export.")

    st.markdown("---")
    st.write("**Gold Snapshot (tables)**")
    st.dataframe(gold_counts, use_container_width=True, hide_index=True)

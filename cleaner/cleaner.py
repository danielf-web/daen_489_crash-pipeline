# cleaner.py

import os, json, logging, time
from pathlib import Path

import pandas as pd
import duckdb
import pika
from dotenv import load_dotenv

from cleaning_rules import clean_for_gold
from minio_io import fetch_silver_csv
from duckdb_writer import ensure_schema_and_table, upsert_dataframe

# ----------------------------
# Prometheus metrics
# ----------------------------
from prometheus_client import Counter, Histogram, Gauge, start_http_server

SERVICE_NAME = "cleaner"

JOBS_TOTAL = Counter(
    "pipeline_cleaner_jobs_total",
    "Total number of clean jobs received",
)
JOBS_SUCCESS = Counter(
    "pipeline_cleaner_jobs_success_total",
    "Total number of clean jobs that completed successfully",
)
JOBS_FAILURE = Counter(
    "pipeline_cleaner_jobs_failure_total",
    "Total number of clean jobs that failed",
)

JOB_DURATION_SECONDS = Histogram(
    "pipeline_cleaner_job_duration_seconds",
    "Time spent processing a clean job in seconds",
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300),
)

ROWS_CLEANED = Gauge(
    "pipeline_cleaner_rows_cleaned",
    "Number of cleaned rows produced for the most recent job",
)

UPSERT_INSERTED = Counter(
    "pipeline_cleaner_upsert_inserted_total",
    "Total rows inserted into gold table (sum across jobs)",
)
UPSERT_UPDATED = Counter(
    "pipeline_cleaner_upsert_updated_total",
    "Total rows updated in gold table (sum across jobs, if available)",
)

LAST_SUCCESS_UNIX = Gauge(
    "pipeline_cleaner_last_success_timestamp_unix",
    "Unix timestamp of the last successful job",
)
LAST_FAILURE_UNIX = Gauge(
    "pipeline_cleaner_last_failure_timestamp_unix",
    "Unix timestamp of the last failed job",
)

RABBIT_CONNECTED = Gauge(
    "pipeline_cleaner_rabbit_connected",
    "1 if connected to RabbitMQ, else 0",
)

# ----------------------------
# NEW: DuckDB metrics (for Dashboard 2 requirements)
# ----------------------------
GOLD_ROW_COUNT = Gauge(
    "pipeline_gold_row_count",
    "Final row count in the Gold DuckDB table",
)

GOLD_DB_FILE_BYTES = Gauge(
    "pipeline_gold_duckdb_file_bytes",
    "Size of the Gold DuckDB database file in bytes",
)

# ----------------------------
# Load env (prefer project root)
# ----------------------------
_here = Path(__file__).resolve()
_root_env = _here.parents[1] / ".env"   # ../.env (project root)
_local_env = _here.with_name(".env")    # ./cleaner/.env (discouraged)

if _root_env.exists() and not _local_env.exists():
    load_dotenv(_root_env)
    loaded_env_path = str(_root_env)
elif _local_env.exists():
    load_dotenv(_local_env)
    loaded_env_path = str(_local_env)
else:
    load_dotenv()
    loaded_env_path = "process env (no .env found)"

# ----------------------------
# Detect Docker vs host
# ----------------------------
def _in_docker() -> bool:
    # Most Docker containers have this file
    if os.path.exists("/.dockerenv"):
        return True
    # Optional override if you ever need it
    if os.getenv("RUNNING_IN_DOCKER", "").lower() in ("1", "true", "yes", "on"):
        return True
    return False

IN_DOCKER = _in_docker()

# ----------------------------
# Host-mode normalization (ONLY when NOT in Docker)
# ----------------------------
def _normalize_env_for_host():
    # RabbitMQ URL
    url = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
    url = url.replace("@rabbitmq:", "@localhost:").replace("@rabbitmq/", "@localhost/")
    os.environ["RABBITMQ_URL"] = url

    # MinIO endpoint
    endpoint = os.getenv("MINIO_ENDPOINT", os.getenv("MINIO_API", "localhost:9000"))
    if endpoint.startswith("minio:"):
        endpoint = endpoint.replace("minio:", "localhost:")
    os.environ["MINIO_ENDPOINT"] = endpoint

    # Normalize MINIO_SECURE/MINIO_SSL -> MINIO_SECURE = 0/1
    secure_raw = str(os.getenv("MINIO_SECURE", os.getenv("MINIO_SSL", "0"))).lower()
    os.environ["MINIO_SECURE"] = "1" if secure_raw in ("1", "true", "yes", "on") else "0"

if not IN_DOCKER:
    _normalize_env_for_host()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logging.info(f"[{SERVICE_NAME}] Loaded environment from: {loaded_env_path}")
logging.info(f"[{SERVICE_NAME}] IN_DOCKER={IN_DOCKER}")
logging.info(f"[{SERVICE_NAME}] RABBITMQ_URL={os.environ.get('RABBITMQ_URL')}")
logging.info(f"[{SERVICE_NAME}] MINIO_ENDPOINT={os.environ.get('MINIO_ENDPOINT')}")

# ----------------------------
# Settings
# ----------------------------
RABBITMQ_URL = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
QUEUE        = os.getenv("CLEAN_QUEUE", "clean")

# Your .env uses XFORM_BUCKET=transform-data; prefer MINIO_BUCKET but fall back to XFORM_BUCKET
MINIO_BUCKET = os.getenv("MINIO_BUCKET") or os.getenv("XFORM_BUCKET", "transform-data")
MINIO_PREFIX = os.getenv("MINIO_PREFIX", "crash")

GOLD_DB_PATH = os.getenv("GOLD_DB_PATH", "./gold.duckdb")
GOLD_TABLE   = os.getenv("GOLD_TABLE", "gold.crashes")

# Metrics port: match docker-compose + prometheus.yml (cleaner:8000)
CLEANER_METRICS_PORT = int(os.getenv("CLEANER_METRICS_PORT", os.getenv("METRICS_PORT", "8000")))
METRICS_ADDR = os.getenv("METRICS_ADDR", "0.0.0.0")


def process_job(msg: dict):
    corr_id = msg["corr_id"]
    lat_col = msg.get("lat_col", "latitude")
    lng_col = msg.get("lng_col", "longitude")

    # 1) Fetch Silver CSV from MinIO
    csv_path = fetch_silver_csv(
        corr_id, dest_dir="./_silver_cache",
        bucket=MINIO_BUCKET, prefix=MINIO_PREFIX
    )
    logging.info(f"[{corr_id}] fetched Silver CSV: {csv_path}")

    # 2) Load & clean
    raw = pd.read_csv(csv_path, low_memory=False)
    cleaned = clean_for_gold(raw, lat_col=lat_col, lng_col=lng_col)
    if "crash_record_id" not in cleaned.columns:
        raise KeyError("cleaned data missing 'crash_record_id'")

    ROWS_CLEANED.set(float(len(cleaned)))
    logging.info(f"[{corr_id}] cleaned rows: {len(cleaned):,}")

    # 3) Write/Upsert to DuckDB
    con = duckdb.connect(GOLD_DB_PATH)
    con.execute("USE main")
    ensure_schema_and_table(con, GOLD_TABLE, cleaned, pk="crash_record_id")
    stats = upsert_dataframe(con, GOLD_TABLE, cleaned, key="crash_record_id")

    # ----------------------------
    # NEW: set DuckDB metrics required for Dashboard 2
    # ----------------------------
    try:
        row_count = con.execute(f"SELECT COUNT(*) FROM {GOLD_TABLE}").fetchone()[0]
        GOLD_ROW_COUNT.set(float(row_count))
    except Exception as e:
        logging.warning(f"[{corr_id}] could not compute gold row count: {e}")

    try:
        if os.path.exists(GOLD_DB_PATH):
            GOLD_DB_FILE_BYTES.set(float(os.path.getsize(GOLD_DB_PATH)))
    except Exception as e:
        logging.warning(f"[{corr_id}] could not compute gold db file size: {e}")

    con.close()

    # upsert stats (best-effort)
    try:
        if isinstance(stats, dict):
            if stats.get("inserted") is not None:
                UPSERT_INSERTED.inc(float(stats["inserted"]))
            if stats.get("updated") is not None:
                UPSERT_UPDATED.inc(float(stats["updated"]))
    except Exception:
        pass

    logging.info(f"[{corr_id}] upsert stats: {stats}")
    return stats


def on_message(ch, method, properties, body):
    JOBS_TOTAL.inc()
    start_t = time.time()

    corr_id = "<missing>"
    try:
        msg = json.loads(body.decode("utf-8"))
        if msg.get("type") != "clean":
            logging.warning(f"Ignoring message without type=clean: {msg}")
            ch.basic_ack(method.delivery_tag)
            return

        corr_id = msg.get("corr_id", "<missing>")
        logging.info(f"Received clean job: corr_id={corr_id}")

        with JOB_DURATION_SECONDS.time():
            stats = process_job(msg)

        JOBS_SUCCESS.inc()
        LAST_SUCCESS_UNIX.set(float(time.time()))

        inserted = stats.get("inserted") if isinstance(stats, dict) else None
        est_new = stats.get("estimated_new") if isinstance(stats, dict) else None
        logging.info(f"[{corr_id}] DONE inserted={inserted} (est_new={est_new})")

        ch.basic_ack(method.delivery_tag)

    except Exception as e:
        JOBS_FAILURE.inc()
        LAST_FAILURE_UNIX.set(float(time.time()))
        logging.exception(f"Job failed (corr_id={corr_id}): {e}")
        ch.basic_nack(method.delivery_tag, requeue=False)

    finally:
        _ = time.time() - start_t


def main():
    # Start metrics server
    try:
        start_http_server(CLEANER_METRICS_PORT, addr=METRICS_ADDR)
        logging.info(f"[{SERVICE_NAME}] Metrics up at http://{METRICS_ADDR}:{CLEANER_METRICS_PORT}/metrics")
    except Exception as e:
        logging.warning(f"[{SERVICE_NAME}] Could not start metrics server: {e}")

    params = pika.URLParameters(RABBITMQ_URL)

    RABBIT_CONNECTED.set(0)
    connection = pika.BlockingConnection(params)
    RABBIT_CONNECTED.set(1)

    channel = connection.channel()
    channel.queue_declare(queue=QUEUE, durable=True)

    logging.info(f"[{SERVICE_NAME}] Listening on queue '{QUEUE}' ... Ctrl+C to exit.")
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=QUEUE, on_message_callback=on_message)

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        logging.info(f"[{SERVICE_NAME}] Shutting down consumer...")
    finally:
        try:
            RABBIT_CONNECTED.set(0)
        except Exception:
            pass
        if connection.is_open:
            connection.close()


if __name__ == "__main__":
    main()

# transformer/transformer.py
import os
import io
import json
import gzip
import socket
import logging
import time
import random
import traceback
from typing import List, Dict, Any, Optional

import pika
from minio import Minio
from minio.error import S3Error
import polars as pl

from prometheus_client import Counter, Gauge, Histogram, start_http_server

# ---------------------------------
# Logging
# ---------------------------------
logging.basicConfig(level=logging.INFO, format="[transformer] %(message)s")
logging.getLogger("pika").setLevel(logging.WARNING)

# ---------------------------------
# Env / Config
# ---------------------------------
def _env_bool(value: str) -> bool:
    v = str(value or "").strip().lower()
    return v in ("1", "true", "yes", "on")

RABBIT_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/")
TRANSFORM_QUEUE = os.getenv("TRANSFORM_QUEUE", "transform")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT") or os.getenv("MINIO_API") or "minio:9000"
MINIO_ACCESS = os.getenv("MINIO_ACCESS_KEY") or os.getenv("MINIO_USER") or "admin"
MINIO_SECRET = os.getenv("MINIO_SECRET_KEY") or os.getenv("MINIO_PASS") or "admin123"
MINIO_SECURE = _env_bool(os.getenv("MINIO_SECURE") or os.getenv("MINIO_SSL") or "0")

RAW_BUCKET = os.getenv("RAW_BUCKET", "raw-data")
XFORM_BUCKET_ENV = os.getenv("XFORM_BUCKET", "transform-data")
PREFIX = os.getenv("PREFIX", "crash")

# Prometheus scrapes http://transformer:8000/metrics
METRICS_ADDR = os.getenv("METRICS_ADDR", "0.0.0.0")
METRICS_PORT = int(os.getenv("TRANSFORMER_METRICS_PORT", os.getenv("METRICS_PORT", "8000")))

_required = {
    "MINIO_ENDPOINT": MINIO_ENDPOINT,
    "RAW_BUCKET": RAW_BUCKET,
    "XFORM_BUCKET": XFORM_BUCKET_ENV,
}
_missing = [k for k, v in _required.items() if not v]
if _missing:
    raise SystemExit(f"[transformer] Missing env vars: {', '.join(_missing)}")

# ---------------------------------
# Prometheus Metrics
# ---------------------------------
SERVICE_INFO = Gauge("pipeline_service_info", "Service metadata", ["service"])
SERVICE_INFO.labels(service="transformer").set(1)

CONSUMER_CONNECTED = Gauge("transformer_rabbitmq_connected", "1 if connected to RabbitMQ else 0")
MESSAGES_TOTAL = Counter("transformer_messages_total", "Messages received from RabbitMQ", ["queue", "type"])

# Job result + duration
JOBS_TOTAL = Counter("transformer_jobs_total", "Transform jobs processed", ["result"])  # success|failure
JOB_SECONDS = Histogram(
    "transformer_job_duration_seconds",
    "Seconds spent processing a transform job",
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 45, 90, 180, 300, 600),
)

# MinIO IO + errors
MINIO_GET_TOTAL = Counter("transformer_minio_get_total", "MinIO GET object calls", ["bucket"])
MINIO_GET_BYTES = Counter("transformer_minio_get_bytes_total", "Bytes read from MinIO (compressed or raw)", ["bucket"])
MINIO_PUT_TOTAL = Counter("transformer_minio_put_total", "MinIO PUT object calls", ["bucket"])
MINIO_PUT_BYTES = Counter("transformer_minio_put_bytes_total", "Bytes written to MinIO", ["bucket"])
MINIO_ERRORS_TOTAL = Counter("transformer_minio_errors_total", "MinIO errors", ["op", "bucket"])  # op=get|put|list|bucket

# Raw read counters (rows in)
RAW_OBJECTS_READ = Counter("transformer_raw_objects_read_total", "Raw objects read from MinIO", ["dataset"])
RAW_ROWS_READ = Counter("transformer_raw_rows_read_total", "Rows read from raw JSON arrays", ["dataset"])

# Dashboard-friendly rows in/out (monotonic counters)
ROWS_IN_TOTAL = Counter(
    "transformer_rows_in_total",
    "Total rows read into transformer (by dataset)",
    ["dataset"],  # crashes|vehicles|people
)
MERGED_ROWS_LAST = Gauge("transformer_merged_rows", "Rows in merged dataset (last job)")
MERGED_COLS_LAST = Gauge("transformer_merged_cols", "Columns in merged dataset (last job)")

MERGED_ROWS_TOTAL = Counter(
    "transformer_merged_rows_total",
    "Total merged rows produced (monotonic, good for rate/increase)",
)
CSV_WRITES_TOTAL = Counter(
    "transformer_csv_writes_total",
    "Total merged CSV files written by transformer",
)
CSV_BYTES_WRITTEN_TOTAL = Counter(
    "transformer_csv_bytes_written_total",
    "Total bytes written for merged CSV output",
)

# Timestamps
LAST_JOB_UNIX = Gauge("transformer_last_job_unix", "Unix timestamp of last job start")
LAST_SUCCESS_UNIX = Gauge("transformer_last_success_unix", "Unix timestamp of last successful job")
LAST_FAILURE_UNIX = Gauge("transformer_last_failure_unix", "Unix timestamp of last failed job")
LAST_OUTPUT_UNIX = Gauge("transformer_last_output_unix", "Unix timestamp of last successful CSV write")

# ---------------------------------
# Metrics server
# ---------------------------------
_METRICS_STARTED = False

def start_metrics_server_once():
    global _METRICS_STARTED
    if _METRICS_STARTED:
        return
    _METRICS_STARTED = True
    try:
        start_http_server(METRICS_PORT, addr=METRICS_ADDR)
        logging.info(f"Metrics server listening on http://{METRICS_ADDR}:{METRICS_PORT}/metrics")
    except Exception as e:
        logging.warning(f"Metrics server failed to start on {METRICS_ADDR}:{METRICS_PORT} -> {e!r}")

# ---------------------------------
# MinIO client
# ---------------------------------
def minio_client() -> Minio:
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS,
        secret_key=MINIO_SECRET,
        secure=MINIO_SECURE,
    )

# ---------------------------------
# Object helpers
# ---------------------------------
def list_objects_recursive(cli: Minio, bucket: str, prefix: str) -> List[str]:
    out: List[str] = []
    try:
        for obj in cli.list_objects(bucket, prefix=prefix, recursive=True):
            if getattr(obj, "is_dir", False):
                continue
            out.append(obj.object_name)
    except Exception:
        MINIO_ERRORS_TOTAL.labels(op="list", bucket=bucket).inc()
        raise
    return out

def read_json_gz_array(cli: Minio, bucket: str, key: str) -> List[Dict[str, Any]]:
    resp = None
    data = b""
    try:
        MINIO_GET_TOTAL.labels(bucket=bucket).inc()
        resp = cli.get_object(bucket, key)
        data = resp.read()
        MINIO_GET_BYTES.labels(bucket=bucket).inc(len(data))
    except Exception:
        MINIO_ERRORS_TOTAL.labels(op="get", bucket=bucket).inc()
        raise
    finally:
        try:
            if resp is not None:
                resp.close()
                resp.release_conn()
        except Exception:
            pass

    # GZIP magic: 1F 8B
    if len(data) >= 2 and data[:2] == b"\x1f\x8b":
        try:
            payload = gzip.decompress(data)
        except OSError:
            payload = data
    else:
        payload = data

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        text = payload.decode("utf-8", errors="replace")

    try:
        arr = json.loads(text)
    except json.JSONDecodeError:
        return []

    if isinstance(arr, list):
        return arr
    if isinstance(arr, dict) and isinstance(arr.get("data"), list):
        return arr["data"]
    return []

def write_csv(cli: Minio, bucket: str, key: str, df: pl.DataFrame) -> None:
    buf = io.BytesIO()
    df.write_csv(buf)
    data = buf.getvalue()

    try:
        MINIO_PUT_TOTAL.labels(bucket=bucket).inc()
        MINIO_PUT_BYTES.labels(bucket=bucket).inc(len(data))
        cli.put_object(
            bucket,
            key,
            data=io.BytesIO(data),
            length=len(data),
            content_type="text/csv; charset=utf-8",
        )
    except Exception:
        MINIO_ERRORS_TOTAL.labels(op="put", bucket=bucket).inc()
        raise

    CSV_WRITES_TOTAL.inc()
    CSV_BYTES_WRITTEN_TOTAL.inc(len(data))
    LAST_OUTPUT_UNIX.set(time.time())

# ---------------------------------
# Load & merge
# ---------------------------------
def _keys_for_corr(cli: Minio, bucket: str, prefix: str, dataset_alias: str, corr: str) -> List[str]:
    base = f"{prefix}/{dataset_alias}/"
    keys = list_objects_recursive(cli, bucket, base)
    needle = f"/corr={corr}/"
    return [k for k in keys if (k.endswith(".json.gz") or k.endswith(".json")) and needle in k]

def load_dataset(cli: Minio, raw_bucket: str, prefix: str, dataset_alias: str, corr: str) -> pl.DataFrame:
    keys = _keys_for_corr(cli, raw_bucket, prefix, dataset_alias, corr)
    rows_all: List[Dict[str, Any]] = []
    for k in keys:
        RAW_OBJECTS_READ.labels(dataset=dataset_alias).inc()
        rows = read_json_gz_array(cli, raw_bucket, k)
        if rows:
            RAW_ROWS_READ.labels(dataset=dataset_alias).inc(len(rows))
            ROWS_IN_TOTAL.labels(dataset=dataset_alias).inc(len(rows))  # dashboard-friendly
            rows_all.extend(rows)
    return pl.DataFrame(rows_all) if rows_all else pl.DataFrame()

def basic_standardize(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    df = df.rename({c: c.strip().lower() for c in df.columns})
    return df.unique(maintain_order=True)

def aggregate_many_to_one(df: pl.DataFrame, id_col: str, prefix: str) -> pl.DataFrame:
    if df.is_empty():
        return df
    keep_fields = [c for c in df.columns if c != id_col]
    text_cols = [c for c in keep_fields if df.schema.get(c, pl.Utf8) == pl.Utf8][:5]

    aggs = [pl.len().alias(f"{prefix}_count")]
    for c in text_cols:
        aggs.append(
            pl.col(c).drop_nulls().cast(pl.Utf8).unique().sort().implode().alias(f"{prefix}_{c}_list")
        )
    return df.group_by(id_col, maintain_order=True).agg(aggs)

def merge_crash_vehicles_people(crashes: pl.DataFrame, vehicles: pl.DataFrame, people: pl.DataFrame, id_col: str) -> pl.DataFrame:
    crashes = basic_standardize(crashes)
    vehicles = basic_standardize(vehicles)
    people = basic_standardize(people)

    id_lower = id_col.lower()

    def _ensure_id(df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty() or id_lower in df.columns:
            return df
        for c in df.columns:
            if c.lower() == id_lower:
                return df.rename({c: id_lower})
        return df

    crashes = _ensure_id(crashes)
    vehicles = _ensure_id(vehicles)
    people = _ensure_id(people)

    if not crashes.is_empty() and id_lower not in crashes.columns:
        return crashes

    veh_agg = aggregate_many_to_one(vehicles, id_lower, prefix="veh") if (not vehicles.is_empty() and id_lower in vehicles.columns) else pl.DataFrame()
    ppl_agg = aggregate_many_to_one(people, id_lower, prefix="ppl") if (not people.is_empty() and id_lower in people.columns) else pl.DataFrame()

    out = crashes
    if not veh_agg.is_empty():
        out = out.join(veh_agg, on=id_lower, how="left")
    if not ppl_agg.is_empty():
        out = out.join(ppl_agg, on=id_lower, how="left")

    return out.unique(subset=[id_lower], keep="first", maintain_order=True)

# ---------------------------------
# CSV safety
# ---------------------------------
def make_csv_safe(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df

    def _jsonable(x):
        if x is None or isinstance(x, (str, int, float, bool)):
            return x
        if isinstance(x, bytes):
            try:
                return x.decode("utf-8")
            except Exception:
                return x.hex()
        if isinstance(x, (list, tuple, set)):
            return [_jsonable(v) for v in list(x)]
        if isinstance(x, dict):
            return {k: _jsonable(v) for k, v in x.items()}
        return str(x)

    fixes, drop_cols = [], []
    for name, dtype in df.schema.items():
        if isinstance(dtype, (pl.List, pl.Struct)) or dtype.__class__.__name__ == "Array":
            fixes.append(
                pl.col(name).map_elements(
                    lambda x: json.dumps(_jsonable(x), ensure_ascii=False),
                    return_dtype=pl.String,
                ).alias(f"{name}_json")
            )
            drop_cols.append(name)

    if not fixes:
        return df
    out = df.with_columns(fixes)
    return out.drop(drop_cols) if drop_cols else out

# ---------------------------------
# Transform runner
# ---------------------------------
def run_transform_job(msg: dict):
    corr = msg.get("corr_id")
    raw_bucket = msg.get("raw_bucket", RAW_BUCKET)
    out_bucket = msg.get("xform_bucket") or msg.get("clean_bucket") or XFORM_BUCKET_ENV
    prefix = msg.get("prefix", PREFIX)

    if not corr or not out_bucket:
        raise ValueError("run_transform_job: missing corr_id or output bucket (xform_bucket|clean_bucket|XFORM_BUCKET)")

    cli = minio_client()

    # Ensure target bucket exists
    try:
        if not cli.bucket_exists(out_bucket):
            cli.make_bucket(out_bucket)
    except S3Error as e:
        if e.code not in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
            MINIO_ERRORS_TOTAL.labels(op="bucket", bucket=out_bucket).inc()
            raise
    except Exception:
        MINIO_ERRORS_TOTAL.labels(op="bucket", bucket=out_bucket).inc()
        raise

    crashes_df = load_dataset(cli, raw_bucket, prefix, "crashes", corr)
    vehicles_df = load_dataset(cli, raw_bucket, prefix, "vehicles", corr)
    people_df = load_dataset(cli, raw_bucket, prefix, "people", corr)

    merged = merge_crash_vehicles_people(
        crashes=crashes_df,
        vehicles=vehicles_df,
        people=people_df,
        id_col="crash_record_id",
    )

    MERGED_ROWS_LAST.set(float(merged.height))
    MERGED_COLS_LAST.set(float(merged.width))
    MERGED_ROWS_TOTAL.inc(merged.height)

    out_key = f"{prefix}/corr={corr}/merged.csv"
    write_csv(cli, out_bucket, out_key, make_csv_safe(merged))
    logging.info(f"Wrote s3://{out_bucket}/{out_key} (rows={merged.height}, cols={merged.width})")

# ---------------------------------
# RabbitMQ consumer
# ---------------------------------
def wait_for_port(host: str, port: int, tries: int = 60, delay: float = 1.0):
    for _ in range(tries):
        try:
            with socket.create_connection((host, port), timeout=1.5):
                return True
        except OSError:
            time.sleep(delay)
    return False

def start_consumer():
    from pika.exceptions import AMQPConnectionError, ProbableAccessDeniedError, ProbableAuthenticationError

    params = pika.URLParameters(RABBIT_URL)

    host = params.host or "rabbitmq"
    port = params.port or 5672
    if not wait_for_port(host, port, tries=60, delay=1.0):
        CONSUMER_CONNECTED.set(0)
        raise SystemExit(f"[transformer] RabbitMQ not reachable at {host}:{port} after waiting.")

    max_tries = 60
    base_delay = 1.5
    conn = None

    for i in range(1, max_tries + 1):
        try:
            conn = pika.BlockingConnection(params)
            break
        except (AMQPConnectionError, ProbableAccessDeniedError, ProbableAuthenticationError) as e:
            if i == 1:
                logging.info(f"Waiting for RabbitMQ @ {RABBIT_URL} …")
            if i % 10 == 0:
                logging.info(f"Still waiting (attempt {i}/{max_tries}): {e.__class__.__name__}")
            time.sleep(base_delay + random.random())

    if conn is None or not conn.is_open:
        CONSUMER_CONNECTED.set(0)
        raise SystemExit("[transformer] Could not connect to RabbitMQ after multiple attempts.")

    CONSUMER_CONNECTED.set(1)

    ch = conn.channel()
    ch.queue_declare(queue=TRANSFORM_QUEUE, durable=True)
    ch.basic_qos(prefetch_count=1)

    def on_msg(chx, method, props, body):
        job_start = time.time()
        LAST_JOB_UNIX.set(job_start)

        mtype = "unknown"
        try:
            msg = json.loads(body.decode("utf-8"))
            mtype = msg.get("type", "") or "unknown"
            MESSAGES_TOTAL.labels(queue=TRANSFORM_QUEUE, type=mtype).inc()

            if mtype not in ("transform", "clean"):
                logging.info(f"ignoring message type={mtype!r}")
                chx.basic_ack(delivery_tag=method.delivery_tag)
                return

            logging.info(f"Received transform job (type={mtype}) corr={msg.get('corr_id')}")
            with JOB_SECONDS.time():
                run_transform_job(msg)

            JOBS_TOTAL.labels(result="success").inc()
            LAST_SUCCESS_UNIX.set(time.time())
            chx.basic_ack(delivery_tag=method.delivery_tag)

        except Exception:
            JOBS_TOTAL.labels(result="failure").inc()
            LAST_FAILURE_UNIX.set(time.time())
            traceback.print_exc()
            chx.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    logging.info(f"Up. Waiting for jobs on queue '{TRANSFORM_QUEUE}'")
    ch.basic_consume(queue=TRANSFORM_QUEUE, on_message_callback=on_msg)

    try:
        ch.start_consuming()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            CONSUMER_CONNECTED.set(0)
        except Exception:
            pass
        try:
            if conn and conn.is_open:
                conn.close()
        except Exception:
            pass

if __name__ == "__main__":
    logging.info(
        f"Config: METRICS={METRICS_ADDR}:{METRICS_PORT} MINIO={MINIO_ENDPOINT} RAW_BUCKET={RAW_BUCKET} XFORM_BUCKET={XFORM_BUCKET_ENV}"
    )
    start_metrics_server_once()  # start BEFORE blocking consume loop
    start_consumer()

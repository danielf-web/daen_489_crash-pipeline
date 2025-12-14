# api_stub.py — FastAPI stub that can publish to RabbitMQ
"""
cd ~/Pipeline

# (optional but what we used before, if you want RabbitMQ-style envs)
export RABBITMQ_URL="amqp://guest:guest@localhost:5672/"
export EXTRACT_QUEUE="extract"

# start the FastAPI stub on port 8000
uvicorn api_stub:app --host 0.0.0.0 --port 8000 --reload

"""

from fastapi import FastAPI, Query
from pydantic import BaseModel
import os, json, shutil, time
import urllib.parse as uparse

try:
    import pika  # RabbitMQ client
except Exception:
    pika = None  # allow health to reflect this

app = FastAPI(title="Pipeline Stub API")

ROOT = os.path.abspath(os.path.dirname(__file__))
MINIO_ROOT = os.path.join(ROOT, "minio-data")
OUT_DIR = os.path.join(ROOT, "out")

SCHEMAS = {
    "crashes": [
        "crash_record_id","crash_date","crash_hour","crash_day_of_week","crash_type",
        "posted_speed_limit","weather_condition","latitude","longitude",
        "traffic_control_device","injuries_total","hit_and_run_i",
    ],
    "vehicles": ["vehicle_id","unit_no","vehicle_type","vehicle_impact_location","vehicle_defect"],
    "people":   ["person_id","unit_no","person_type","age","sex","safety_equipment"],
}

def env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name, default)
    return v if v == "" or v is None else v

def rmq_url() -> str | None:
    # Accept either full URL or host/port/user/pass pieces
    url = env("RABBITMQ_URL") or env("RABBIT_URL")
    if url:
        # ensure default vhost if missing
        if url.endswith("@localhost:5672") or url.endswith("@rabbitmq:5672"):
            url += "/"
        if url.endswith("/"):
            url += "%2f"
        return url
    host = env("RABBIT_HOST", "localhost")
    port = env("RABBIT_PORT", "5672")
    user = env("RABBIT_USER", "guest")
    pwd  = env("RABBIT_PASS", "guest")
    return f"amqp://{user}:{pwd}@{host}:{port}/%2f"

def queue_name(mode: str) -> str:
    # single queue by default; override if you set RMQ_QUEUE_STREAM / RMQ_QUEUE_BACKFILL
    if mode == "streaming":
        return env("RMQ_QUEUE_STREAM", env("EXTRACT_QUEUE", "extract"))
    if mode == "backfill":
        return env("RMQ_QUEUE_BACKFILL", env("EXTRACT_QUEUE", "extract"))
    return env("EXTRACT_QUEUE", "extract")

def publish_to_rabbit(queue: str, body: dict) -> dict:
    if pika is None:
        return {"queued": False, "reason": "pika not installed"}
    url = rmq_url()
    params = pika.URLParameters(url)
    connection = pika.BlockingConnection(params)
    ch = connection.channel()
    ch.queue_declare(queue=queue, durable=True)
    ch.basic_publish(
        exchange="",
        routing_key=queue,
        body=json.dumps(body).encode("utf-8"),
        properties=pika.BasicProperties(
            delivery_mode=2,  # persistent
            content_type="application/json",
        ),
    )
    connection.close()
    return {"queued": True, "queue": queue}

@app.get("/api/health")
def health():
    status = {"MinIO": True, "Extractor": True, "Transformer": True, "Cleaner": True}
    # RabbitMQ check
    ok = False
    reason = None
    if pika is None:
        reason = "pika not installed"
    else:
        try:
            params = pika.URLParameters(rmq_url())
            conn = pika.BlockingConnection(params)
            conn.close()
            ok = True
        except Exception as e:
            reason = str(e)
    status["RabbitMQ"] = ok
    if not ok and reason:
        status["RabbitMQ_reason"] = reason
    return status

@app.get("/api/schema/{dataset}")
def schema_ds(dataset: str):
    return {"columns": SCHEMAS.get(dataset, [])}

@app.get("/api/schema")
def schema_q(dataset: str = Query(...)):
    return {"columns": SCHEMAS.get(dataset, [])}

@app.get("/api/minio/preview")
def minio_preview(bucket: str, prefix: str = ""):
    base = os.path.join(MINIO_ROOT, bucket)
    keys = []
    if os.path.exists(base):
        for root, _, files in os.walk(base):
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), base).replace("\\", "/")
                if rel.startswith(prefix):
                    keys.append(rel)
    return {"prefix": prefix, "count": len(keys), "keys": keys[:50]}

class DelReq(BaseModel):
    bucket: str
    prefix: str

@app.post("/api/minio/delete_prefix")
def minio_delete_prefix(req: DelReq):
    base = os.path.join(MINIO_ROOT, req.bucket)
    deleted = 0
    if os.path.exists(base):
        for root, _, files in os.walk(base):
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), base).replace("\\", "/")
                if rel.startswith(req.prefix):
                    os.remove(os.path.join(root, f))
                    deleted += 1
        for root, dirs, _ in os.walk(base, topdown=False):
            for d in dirs:
                p = os.path.join(root, d)
                if not os.listdir(p):
                    os.rmdir(p)
    return {"ok": True, "bucket": req.bucket, "prefix": req.prefix, "deleted": deleted}

class DelBucket(BaseModel):
    bucket: str

@app.post("/api/minio/delete_bucket")
def minio_delete_bucket(req: DelBucket):
    base = os.path.join(MINIO_ROOT, req.bucket)
    if os.path.exists(base):
        shutil.rmtree(base)
    return {"ok": True, "bucket": req.bucket}

class PublishReq(BaseModel):
    mode: str
    corrid: str | None = None
    since_days: int | None = None
    start: str | None = None
    end: str | None = None
    include_vehicles: bool | None = None
    include_people: bool | None = None
    vehicle_columns: list[str] | None = None
    people_columns: list[str] | None = None

@app.post("/api/publish")
def publish(req: PublishReq):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{req.mode}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(req.dict(), f, indent=2)

    # Try RabbitMQ; if it fails, still return file path
    queued = False
    info = {}
    try:
        q = queue_name(req.mode)
        info = publish_to_rabbit(q, req.dict())
        queued = bool(info.get("queued"))
    except Exception as e:
        info = {"queued": False, "reason": str(e)}

    return {"ok": True, "written": path, **info}

class ScheduleReq(BaseModel):
    cron: str
    config: dict

@app.post("/api/schedule")
def schedule(req: ScheduleReq):
    # no-op stub
    return {"ok": True, "cron": req.cron, "config": req.config}

@app.get("/api/schedule/list")
def schedule_list():
    return [{"cron": "0 0 9 * * *", "config": {"type": "streaming"}, "last_run": None}]

@app.get("/api/reports/runs")
def runs():
    path = os.path.join(OUT_DIR, "report.json")
    if os.path.exists(path):
        try:
            data = json.load(open(path, "r", encoding="utf-8"))
            return data if isinstance(data, list) else [data]
        except Exception:
            return []
    return []
# api_stub.py — FastAPI stub that can publish to RabbitMQ
from fastapi import FastAPI, Query
from pydantic import BaseModel
import os, json, shutil, time
import urllib.parse as uparse

try:
    import pika  # RabbitMQ client
except Exception:
    pika = None  # allow health to reflect this

app = FastAPI(title="Pipeline Stub API")

ROOT = os.path.abspath(os.path.dirname(__file__))
MINIO_ROOT = os.path.join(ROOT, "minio-data")
OUT_DIR = os.path.join(ROOT, "out")

SCHEMAS = {
    "crashes": [
        "crash_record_id","crash_date","crash_hour","crash_day_of_week","crash_type",
        "posted_speed_limit","weather_condition","latitude","longitude",
        "traffic_control_device","injuries_total","hit_and_run_i",
    ],
    "vehicles": ["vehicle_id","unit_no","vehicle_type","vehicle_impact_location","vehicle_defect"],
    "people":   ["person_id","unit_no","person_type","age","sex","safety_equipment"],
}

def env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name, default)
    return v if v == "" or v is None else v

def rmq_url() -> str | None:
    # Accept either full URL or host/port/user/pass pieces
    url = env("RABBITMQ_URL") or env("RABBIT_URL")
    if url:
        # ensure default vhost if missing
        if url.endswith("@localhost:5672") or url.endswith("@rabbitmq:5672"):
            url += "/"
        if url.endswith("/"):
            url += "%2f"
        return url
    host = env("RABBIT_HOST", "localhost")
    port = env("RABBIT_PORT", "5672")
    user = env("RABBIT_USER", "guest")
    pwd  = env("RABBIT_PASS", "guest")
    return f"amqp://{user}:{pwd}@{host}:{port}/%2f"

def queue_name(mode: str) -> str:
    # single queue by default; override if you set RMQ_QUEUE_STREAM / RMQ_QUEUE_BACKFILL
    if mode == "streaming":
        return env("RMQ_QUEUE_STREAM", env("EXTRACT_QUEUE", "extract"))
    if mode == "backfill":
        return env("RMQ_QUEUE_BACKFILL", env("EXTRACT_QUEUE", "extract"))
    return env("EXTRACT_QUEUE", "extract")

def publish_to_rabbit(queue: str, body: dict) -> dict:
    if pika is None:
        return {"queued": False, "reason": "pika not installed"}
    url = rmq_url()
    params = pika.URLParameters(url)
    connection = pika.BlockingConnection(params)
    ch = connection.channel()
    ch.queue_declare(queue=queue, durable=True)
    ch.basic_publish(
        exchange="",
        routing_key=queue,
        body=json.dumps(body).encode("utf-8"),
        properties=pika.BasicProperties(
            delivery_mode=2,  # persistent
            content_type="application/json",
        ),
    )
    connection.close()
    return {"queued": True, "queue": queue}

@app.get("/api/health")
def health():
    status = {"MinIO": True, "Extractor": True, "Transformer": True, "Cleaner": True}
    # RabbitMQ check
    ok = False
    reason = None
    if pika is None:
        reason = "pika not installed"
    else:
        try:
            params = pika.URLParameters(rmq_url())
            conn = pika.BlockingConnection(params)
            conn.close()
            ok = True
        except Exception as e:
            reason = str(e)
    status["RabbitMQ"] = ok
    if not ok and reason:
        status["RabbitMQ_reason"] = reason
    return status

@app.get("/api/schema/{dataset}")
def schema_ds(dataset: str):
    return {"columns": SCHEMAS.get(dataset, [])}

@app.get("/api/schema")
def schema_q(dataset: str = Query(...)):
    return {"columns": SCHEMAS.get(dataset, [])}

@app.get("/api/minio/preview")
def minio_preview(bucket: str, prefix: str = ""):
    base = os.path.join(MINIO_ROOT, bucket)
    keys = []
    if os.path.exists(base):
        for root, _, files in os.walk(base):
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), base).replace("\\", "/")
                if rel.startswith(prefix):
                    keys.append(rel)
    return {"prefix": prefix, "count": len(keys), "keys": keys[:50]}

class DelReq(BaseModel):
    bucket: str
    prefix: str

@app.post("/api/minio/delete_prefix")
def minio_delete_prefix(req: DelReq):
    base = os.path.join(MINIO_ROOT, req.bucket)
    deleted = 0
    if os.path.exists(base):
        for root, _, files in os.walk(base):
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), base).replace("\\", "/")
                if rel.startswith(req.prefix):
                    os.remove(os.path.join(root, f))
                    deleted += 1
        for root, dirs, _ in os.walk(base, topdown=False):
            for d in dirs:
                p = os.path.join(root, d)
                if not os.listdir(p):
                    os.rmdir(p)
    return {"ok": True, "bucket": req.bucket, "prefix": req.prefix, "deleted": deleted}

class DelBucket(BaseModel):
    bucket: str

@app.post("/api/minio/delete_bucket")
def minio_delete_bucket(req: DelBucket):
    base = os.path.join(MINIO_ROOT, req.bucket)
    if os.path.exists(base):
        shutil.rmtree(base)
    return {"ok": True, "bucket": req.bucket}

class PublishReq(BaseModel):
    mode: str
    corrid: str | None = None
    since_days: int | None = None
    start: str | None = None
    end: str | None = None
    include_vehicles: bool | None = None
    include_people: bool | None = None
    vehicle_columns: list[str] | None = None
    people_columns: list[str] | None = None

@app.post("/api/publish")
def publish(req: PublishReq):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{req.mode}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(req.dict(), f, indent=2)

    # Try RabbitMQ; if it fails, still return file path
    queued = False
    info = {}
    try:
        q = queue_name(req.mode)
        info = publish_to_rabbit(q, req.dict())
        queued = bool(info.get("queued"))
    except Exception as e:
        info = {"queued": False, "reason": str(e)}

    return {"ok": True, "written": path, **info}

class ScheduleReq(BaseModel):
    cron: str
    config: dict

@app.post("/api/schedule")
def schedule(req: ScheduleReq):
    # no-op stub
    return {"ok": True, "cron": req.cron, "config": req.config}

@app.get("/api/schedule/list")
def schedule_list():
    return [{"cron": "0 0 9 * * *", "config": {"type": "streaming"}, "last_run": None}]

@app.get("/api/reports/runs")
def runs():
    path = os.path.join(OUT_DIR, "report.json")
    if os.path.exists(path):
        try:
            data = json.load(open(path, "r", encoding="utf-8"))
            return data if isinstance(data, list) else [data]
        except Exception:
            return []
    return []

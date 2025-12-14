import os
from pathlib import Path
from minio import Minio
from minio.error import S3Error


def _client():
    endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    # accept class .env keys too
    access_key = os.getenv("MINIO_ACCESS_KEY") or os.getenv("MINIO_USER", "admin")
    secret_key = os.getenv("MINIO_SECRET_KEY") or os.getenv("MINIO_PASS", "admin123")
    secure_val = str(os.getenv("MINIO_SECURE", os.getenv("MINIO_SSL", "0"))).lower()
    secure = secure_val in ("1", "true", "yes")
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)


def fetch_silver_csv(corr_id: str, dest_dir: str, bucket: str, prefix: str) -> str:
    """
    Downloads the Silver CSV for a correlation id into dest_dir.
    Expected object key pattern (adjust via prefix):
    {prefix}/corr=<corr_id>/merged.csv
    """
    client = _client()
    dest = Path(dest_dir); dest.mkdir(parents=True, exist_ok=True)
    object_key = f"{prefix.rstrip('/')}/corr={corr_id}/merged.csv"
    local_path = dest / f"merged_{corr_id}.csv"
    try:
        client.fget_object(bucket, object_key, str(local_path))
    except S3Error as e:
        raise FileNotFoundError(f"MinIO object not found: s3://{bucket}/{object_key} ({e})")
    return str(local_path)
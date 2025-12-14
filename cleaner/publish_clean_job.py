# publish_clean_job.py
import os, json, argparse
from pathlib import Path

import pika
from dotenv import load_dotenv


def load_env():
    here = Path(__file__).resolve()
    root_env = here.parents[1] / ".env"   # ../.env (project root)
    local_env = here.with_name(".env")    # ./cleaner/.env (discouraged)

    if root_env.exists() and not local_env.exists():
        load_dotenv(root_env)
        return str(root_env)
    elif local_env.exists():
        load_dotenv(local_env)
        return str(local_env)
    else:
        load_dotenv()
        return "process env (no .env found)"


def running_in_docker() -> bool:
    # common, reliable check
    if Path("/.dockerenv").exists():
        return True
    # optional override
    return str(os.getenv("DOCKERIZED", "")).lower() in ("1", "true", "yes", "on")


def normalize_url_for_host(url: str) -> str:
    """
    Only use this when running on your HOST machine.
    In Docker, you must keep rabbitmq as rabbitmq.
    """
    if not url:
        return "amqp://guest:guest@localhost:5672/"

    return (
        url.replace("@rabbitmq:", "@localhost:")
           .replace("@rabbitmq/", "@localhost/")
           .replace("@rabbitmq", "@localhost")
    )


def main():
    loaded = load_env()

    ap = argparse.ArgumentParser()
    ap.add_argument("--corr-id", required=True, help="Correlation id of the Silver run")
    ap.add_argument("--lat-col", default="latitude")
    ap.add_argument("--lng-col", default="longitude")
    ap.add_argument("--url", default=None, help="Override AMQP URL (optional)")
    args = ap.parse_args()

    url = args.url or os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")

    # IMPORTANT: only rewrite rabbitmq->localhost when not in Docker
    if not running_in_docker():
        url = normalize_url_for_host(url)

    queue = os.getenv("CLEAN_QUEUE", "clean")

    msg = {
        "type": "clean",
        "corr_id": args.corr_id,
        "lat_col": args.lat_col,
        "lng_col": args.lng_col,
    }

    params = pika.URLParameters(url)
    conn = pika.BlockingConnection(params)
    ch = conn.channel()
    ch.queue_declare(queue=queue, durable=True)

    ch.basic_publish(
        exchange="",
        routing_key=queue,
        body=json.dumps(msg).encode("utf-8"),
        properties=pika.BasicProperties(
            content_type="application/json",
            delivery_mode=2,  # persistent
        ),
    )

    print(f"[env: {loaded}] Published clean job to '{queue}' via '{url}': {msg}")
    conn.close()


if __name__ == "__main__":
    main()

# transformer/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the transformer code into the image
COPY . .

# Prometheus scrapes http://transformer:8000/metrics
EXPOSE 8000

# Run the transformer
CMD ["python", "/app/transformer.py"]

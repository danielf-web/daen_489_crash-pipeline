# Chicago Crash ETL Pipeline + ML Dashboard

This project builds an end-to-end data pipeline for Chicago traffic crash data and turns it into something you can actually explore, monitor, and model. The pipeline pulls raw crash data (plus optional vehicle + people enrichment), stores it in MinIO, transforms it into clean “Silver” CSVs, and then writes a final “Gold” dataset into DuckDB. On top of that, a Streamlit dashboard lets you explore the data (EDA), run predictions, and view pipeline health and metrics in Grafana/Prometheus.

---

## 1) What problem this pipeline solves

Chicago crash data is useful, but it’s messy and not “analysis ready” out of the box. This pipeline solves that by:
- Automatically pulling crash + enrichment data from the City of Chicago Open Data APIs (Socrata).
- Turning raw records into a structured dataset that’s consistent and easier to analyze.
- Persisting a clean final dataset (DuckDB Gold) so you can query it fast without reprocessing everything.
- Adding an ML model to predict whether a crash is likely a hit-and-run (`hit_and_run_i`).
- Monitoring everything with Prometheus metrics and Grafana dashboards so you can prove it is running correctly.

---

## 2) How data moves through the system

1. **Extractor (Go)** pulls raw crash (and optionally vehicle/people) records from the Chicago Open Data APIs and saves them into **MinIO** in a raw format.
2. **Transformer (Python)** reads raw objects from MinIO, merges and reshapes them into **Silver CSVs** (cleaner, standardized intermediate output).
3. **Cleaner (Python)** applies final cleaning rules and writes the final tables into **DuckDB (Gold)**.
4. **Streamlit app** reads from DuckDB Gold, runs EDA charts, and uses the trained ML model to produce predictions.
5. **Monitoring** exposes metrics (pipeline + ML + DuckDB stats) to **Prometheus**, and Grafana visualizes them.

---

## 3) What the ML model predicts

The ML label is **`hit_and_run_i`**:
- `1` = hit-and-run
- `0` = not hit-and-run

The dashboard loads the trained model artifact (a saved sklearn pipeline) and predicts the probability of hit-and-run using crash context features (time of day, day of week, speed limit, weather, traffic control devices, and other engineered fields depending on the trained pipeline). It supports scoring either:
- Data loaded directly from `gold.crashes`, or
- A held-out CSV you upload into the dashboard.

---

## 4) What the Streamlit dashboard can do

The Streamlit UI is designed to be the “control center”:
- **Home**: label overview + container/service health checks  
- **Data Management**: MinIO cleanup (by prefix or bucket), DuckDB table counts, and quick table previews  
- **Data Fetcher**: publish streaming/backfill jobs (with optional vehicle/people enrichment)  
- **Scheduler**: create automated schedules (cron style) for recurring runs  
- **EDA**: run charts and summary stats directly off DuckDB Gold tables  
- **Model**: load model artifact, score data, show live metrics if labels exist, and export scored CSV  
- **Reports**: show recent run history and a snapshot of Gold tables

---

## 5) Pipeline components

### Extractor (Go)
The extractor’s job is to pull raw crash data from the Chicago Open Data APIs and store it in MinIO so the rest of the pipeline can work off a stable raw layer. It supports pulling crash records and can optionally enrich the run by also fetching vehicles and people records. The output is stored in MinIO under organized prefixes so each run can be tracked (often using a correlation id).

### Transformer (Python)
The transformer reads the raw MinIO objects and converts them into a cleaner intermediate format (Silver). This step is where merging happens (crashes + optional vehicles/people) and where the data becomes a consistent “table-like” dataset. The output is written as Silver CSV files to keep the pipeline transparent and easy to debug.

### Cleaner (Python)
The cleaner takes the Silver CSVs and applies final cleaning rules (type fixes, missing values, standardization, and any domain-specific cleanup). This step writes the final results into DuckDB as the Gold layer (example table: `gold.crashes`). Gold is what the dashboard and ML model use, because it’s the most reliable and query-friendly output.

### DuckDB Gold
DuckDB is the final storage system for cleaned tables. The Gold layer lets you run fast SQL queries and is perfect for dashboards because it stays local and is lightweight. The main table used in the dashboard is `gold.crashes`.

### Streamlit App
The Streamlit app is the UI for exploring the data and proving the pipeline works end-to-end. It provides EDA charts, table previews, model scoring, and downloadable outputs. It also exposes and updates some metrics (like DuckDB file size and Gold row counts) so the monitoring stack can visualize them.

### Docker Compose
Docker Compose is used to launch the full system consistently. It brings up MinIO, Prometheus, Grafana, and the Streamlit dashboard in one command so you do not have to manually manage each service. This makes the project easy to run locally or on a VM.

### Monitoring (Prometheus + Grafana)
Each service exposes custom Prometheus metrics, and Prometheus scrapes them on a schedule. Grafana reads from Prometheus and displays dashboards for pipeline health, throughput, and ML metrics. This is how you can prove the system is actually running and producing real output.

---

## 6) Architecture Diagram

```mermaid
flowchart LR
  A[Extractor (Go)<br/>Fetch raw crash + optional vehicles/people] --> B[MinIO<br/>Raw JSON]
  B --> C[Transformer (Python)<br/>Merge + Silver CSVs]
  C --> D[Cleaner (Python)<br/>Final cleaning rules]
  D --> E[DuckDB<br/>Gold tables]
  E --> F[Streamlit Dashboard<br/>EDA + ML predictions]

  A --> M[Custom Metrics]
  C --> M
  D --> M
  F --> M

  M --> P[Prometheus]
  P --> G[Grafana]

## 7) Azure Deployment Notes (What’s different from local)

### Short summary of my Azure setup

This pipeline was deployed on an Azure Linux VM and️ and run using Docker Compose (MinIO + RabbitMQ + Extractor + Transformer + Cleaner + Prometheus + Grafana + Streamlit). Streamlit was used to publish runs, verify MinIO outputs, confirm DuckDB Gold row counts, and validate that the ML model loads in the dashboard.

### VM configuration

* VM OS: Ubuntu (Linux)
* Runtime: Docker Engine + Docker Compose
* Repo path on VM: `~/daen_489_crash-pipeline/`
* Key services: MinIO, RabbitMQ, Prometheus, Grafana, Streamlit, extractor/transformer/cleaner workers

### Ports opened

Open the ports you used in your Azure NSG (typical for this stack):

* 22: SSH
* 8501: Streamlit
* 3000: Grafana
* 9090: Prometheus
* 9000: MinIO API
* 9001: MinIO console
* 15672: RabbitMQ UI (optional)
* 2112: Extractor metrics scrape (optional)
* 8003: Streamlit metrics scrape (optional)

### Folder structure (VM)

```bash
daen_489_crash-pipeline/
├─ app.py
├─ docker-compose.yml
├─ extractor/
├─ transformer/
├─ cleaner/
│  ├─ gold.duckdb
├─ artifacts/
│  ├─ model.pkl
│  ├─ threshold.txt
│  ├─ test_metrics.json
├─ prometheus/
├─ grafana/
├─ out/
└─ README.md
```

### Differences from local

Local and Azure used the same codebase, but the VM setup surfaced a few differences:

* DuckDB table referencing needed fully-qualified identifiers in the Streamlit app so the Gold row count metric was correct (example: `"gold"."gold"."crashes"`).
* Cleaner upserts required a UNIQUE index on `crash_record_id` so `ON CONFLICT` worked.
* The ML model artifact initially failed to load on Azure due to scikit-learn version mismatch; aligning sklearn versions fixed it.


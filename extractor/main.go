package main

import (
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	minio "github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
	amqp "github.com/rabbitmq/amqp091-go"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// =============================
// Prometheus metrics
// =============================

var (
	// Shared across services (matches your Python services)
	serviceInfo = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "pipeline_service_info",
			Help: "Service metadata",
		},
		[]string{"service"},
	)

	extractorRabbitMQConnected = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "extractor_rabbitmq_connected",
			Help: "1 if connected to RabbitMQ else 0",
		},
	)

	// Needed for “rows in/out per stage” dashboards
	extractorRowsFetchedTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "extractor_rows_fetched_total",
			Help: "Total rows fetched from Socrata (crashes, vehicles, people).",
		},
		[]string{"entity"}, // crashes|vehicles|people
	)

	extractorJobsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "extractor_jobs_total",
			Help: "Total number of extractor jobs processed.",
		},
		[]string{"result"}, // success|failure
	)

	extractorJobsFailedTotal = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "extractor_jobs_failed_total",
			Help: "Total number of extractor jobs that failed.",
		},
	)

	extractorJobDurationSeconds = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "extractor_job_duration_seconds",
			Help:    "Job processing duration in seconds (end-to-end per message).",
			Buckets: prometheus.DefBuckets,
		},
	)

	extractorSocrataRequestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "extractor_socrata_http_requests_total",
			Help: "Total number of HTTP attempts to Socrata (includes retries).",
		},
		[]string{"status"}, // "200","429","5xx","4xx","error"
	)

	extractorSocrataRequestDurationSeconds = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "extractor_socrata_http_request_duration_seconds",
			Help:    "Duration of Socrata HTTP requests (per attempt).",
			Buckets: prometheus.DefBuckets,
		},
	)

	extractorMinioPutObjectsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "extractor_minio_put_objects_total",
			Help: "Total number of objects written to MinIO.",
		},
		[]string{"entity"}, // crashes|vehicles|people|marker|manifest|unknown
	)

	extractorMinioPutErrorsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "extractor_minio_put_errors_total",
			Help: "Total number of MinIO put errors.",
		},
		[]string{"entity"},
	)

	extractorPagesProcessedTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "extractor_pages_processed_total",
			Help: "Total number of pages processed (crashes + enrich pages).",
		},
		[]string{"entity"}, // crashes|vehicles|people
	)

	extractorLastSuccessUnix = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "extractor_last_success_unix",
			Help: "Unix timestamp of the last successfully processed job message.",
		},
	)
)

func registerMetricsOnce() {
	// MustRegister will panic if duplicated; so only call once.
	prometheus.MustRegister(
		serviceInfo,
		extractorRabbitMQConnected,
		extractorRowsFetchedTotal,

		extractorJobsTotal,
		extractorJobsFailedTotal,
		extractorJobDurationSeconds,
		extractorSocrataRequestsTotal,
		extractorSocrataRequestDurationSeconds,
		extractorMinioPutObjectsTotal,
		extractorMinioPutErrorsTotal,
		extractorPagesProcessedTotal,
		extractorLastSuccessUnix,
	)
}

func startMetricsServer() {
	// IMPORTANT:
	// Prometheus is scraping extractor:2112, so do NOT follow a shared METRICS_PORT
	// that might be set to 8000 for other services.
	port := getenv("EXTRACTOR_METRICS_PORT", "2112")
	addr := "0.0.0.0:" + port

	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.Handler())
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})

	srv := &http.Server{
		Addr:              addr,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}

	go func() {
		log.Printf("metrics server listening on http://%s/metrics", addr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("metrics server error: %v", err)
		}
	}()
}

// =============================
// Types — Job spec (Section 2)
// =============================

type WhereBy struct {
	SinceDays int `json:"since_days"`
}

type DatasetSpec struct {
	ID      string   `json:"id"` // 4x4 ID (e.g., 85ca-t3if)
	Alias   string   `json:"alias"`
	Select  string   `json:"select"`
	Where   string   `json:"where,omitempty"`
	Order   string   `json:"order,omitempty"`
	PageSz  int      `json:"page_size,omitempty"`
	WhereBy *WhereBy `json:"where_by,omitempty"`
}

type MaxWorkers struct{ Vehicles, People int }

type Batching struct {
	IDBatchSize int        `json:"id_batch_size"`
	MaxWorkers  MaxWorkers `json:"max_workers"`
}

type Storage struct {
	Bucket   string `json:"bucket"`
	Prefix   string `json:"prefix"`
	Compress bool   `json:"compress"`
}

type DateRange struct {
	Field string `json:"field"` // e.g., "crash_date"
	Start string `json:"start"` // inclusive, ISO8601 "YYYY-MM-DDTHH:MM:SS"
	End   string `json:"end"`   // exclusive, ISO8601 "YYYY-MM-DDTHH:MM:SS"
}

type Job struct {
	Mode      string        `json:"mode"`
	CorrID    string        `json:"corr_id"`
	Source    string        `json:"source"`
	JoinKey   string        `json:"join_key"`
	Primary   DatasetSpec   `json:"primary"`
	Enrich    []DatasetSpec `json:"enrich"`
	Batching  Batching      `json:"batching"`
	Storage   Storage       `json:"storage"`
	DateRange *DateRange    `json:"date_range,omitempty"` // NEW: backfill window
}

type TransformJob struct {
	Type   string `json:"type"`    // "transform"
	CorrID string `json:"corr_id"` // e.g., 2025-09-17T16-22-09Z
}

func publishTransformJob(amqpURL string, job TransformJob) error {
	conn, err := amqp.Dial(amqpURL)
	if err != nil {
		return err
	}
	defer conn.Close()

	ch, err := conn.Channel()
	if err != nil {
		return err
	}
	defer ch.Close()

	// durable queue for transformer
	if _, err := ch.QueueDeclare("transform", true, false, false, false, nil); err != nil {
		return err
	}

	body, err := json.Marshal(job)
	if err != nil {
		return err
	}

	return ch.Publish(
		"", "transform", false, false,
		amqp.Publishing{
			ContentType:  "application/json",
			DeliveryMode: amqp.Persistent,
			Body:         body,
		},
	)
}

func (j *Job) applyDefaults() {
	if j.Mode == "" {
		j.Mode = "streaming"
	}
	if j.Source == "" {
		j.Source = "crash"
	}
	if j.JoinKey == "" {
		j.JoinKey = "crash_record_id"
	}
	if j.Primary.Alias == "" {
		j.Primary.Alias = "crashes"
	}
	if j.Primary.Select == "" {
		j.Primary.Select = "*"
	}
	if j.Primary.Order == "" {
		j.Primary.Order = "crash_date, " + j.JoinKey
	}
	if j.Primary.PageSz == 0 {
		j.Primary.PageSz = 50000
	}
	if j.Batching.IDBatchSize == 0 {
		j.Batching.IDBatchSize = 300
	}
	if j.Batching.MaxWorkers.Vehicles == 0 {
		j.Batching.MaxWorkers.Vehicles = 4
	}
	if j.Batching.MaxWorkers.People == 0 {
		j.Batching.MaxWorkers.People = 4
	}
	if j.Storage.Bucket == "" {
		j.Storage.Bucket = "raw-data"
	}
	if j.Storage.Prefix == "" {
		j.Storage.Prefix = "crash"
	}
	if !j.Storage.Compress {
		j.Storage.Compress = true
	}
}

// Unified WHERE builder: date_range (backfill) > explicit where > watermark > since_days > default 7d
func (j *Job) buildWhere(lastWatermark time.Time) {
	if strings.TrimSpace(j.Primary.Where) != "" {
		return
	}
	if j.DateRange != nil && j.DateRange.Field != "" && j.DateRange.Start != "" && j.DateRange.End != "" {
		f := j.DateRange.Field
		j.Primary.Where = fmt.Sprintf("%s >= '%s' AND %s < '%s'", f, j.DateRange.Start, f, j.DateRange.End)
		return
	}
	if !lastWatermark.IsZero() {
		j.Primary.Where = fmt.Sprintf("crash_date > '%s'", lastWatermark.Format("2006-01-02T15:04:05.000"))
		return
	}
	if j.Primary.WhereBy != nil && j.Primary.WhereBy.SinceDays > 0 {
		since := time.Now().UTC().AddDate(0, 0, -j.Primary.WhereBy.SinceDays)
		j.Primary.Where = fmt.Sprintf("crash_date >= '%s'", since.Format("2006-01-02T15:04:05.000"))
		return
	}
	since := time.Now().UTC().AddDate(0, 0, -7)
	j.Primary.Where = fmt.Sprintf("crash_date >= '%s'", since.Format("2006-01-02T15:04:05.000"))
}

// =============================
// Env + clients
// =============================

type Env struct {
	RabbitURL     string
	ExtractQueue  string
	SocrataBase   string // e.g., https://data.cityofchicago.org
	AppToken      string // optional
	MinioEndpoint string
	MinioAccess   string
	MinioSecret   string
	MinioSSL      bool
	RawBucket     string
}

func getenv(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func getEnv() Env {
	ssl := strings.ToLower(getenv("MINIO_SSL", "false")) == "true"
	return Env{
		RabbitURL:     getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/"),
		ExtractQueue:  getenv("EXTRACT_QUEUE", "extract"),
		SocrataBase:   getenv("SOCRATA_BASE", "https://data.cityofchicago.org"),
		AppToken:      getenv("SOCRATA_APP_TOKEN", ""),
		MinioEndpoint: getenv("MINIO_ENDPOINT", "minio:9000"),
		MinioAccess:   getenv("MINIO_ACCESS_KEY", "admin"),
		MinioSecret:   getenv("MINIO_SECRET_KEY", "admin123"),
		MinioSSL:      ssl,
		RawBucket:     getenv("RAW_BUCKET", "raw-data"),
	}
}

func newMinio(env Env) *minio.Client {
	var cli *minio.Client
	var err error
	for i := 0; i < 10; i++ {
		cli, err = minio.New(env.MinioEndpoint, &minio.Options{
			Creds:  credentials.NewStaticV4(env.MinioAccess, env.MinioSecret, ""),
			Secure: env.MinioSSL,
		})
		if err == nil {
			ctx := context.Background()
			if exists, e := cli.BucketExists(ctx, env.RawBucket); e == nil {
				if !exists {
					if mkErr := cli.MakeBucket(ctx, env.RawBucket, minio.MakeBucketOptions{}); mkErr == nil {
						return cli
					}
				} else {
					return cli
				}
			}
		}
		time.Sleep(time.Second * time.Duration(1+i))
	}
	log.Fatalf("minio not ready: %v", err)
	return nil
}

// =============================
// HTTP helper (simple retry) + metrics
// =============================
func httpGetJSON(env Env, fullURL string) ([]byte, error) {
	req, _ := http.NewRequest("GET", fullURL, nil)
	req.Header.Set("User-Agent", "extractor/1.0 (+edu)")
	if env.AppToken != "" {
		req.Header.Set("X-App-Token", env.AppToken)
	}
	client := &http.Client{Timeout: 60 * time.Second}
	var lastErr error

	for i := 0; i < 3; i++ {
		attemptStart := time.Now()
		resp, err := client.Do(req)
		extractorSocrataRequestDurationSeconds.Observe(time.Since(attemptStart).Seconds())

		if err != nil {
			extractorSocrataRequestsTotal.WithLabelValues("error").Inc()
			lastErr = err
			time.Sleep(time.Duration(i+1) * time.Second)
			continue
		}

		code := resp.StatusCode
		statusLabel := strconv.Itoa(code)
		if code >= 500 {
			statusLabel = "5xx"
		} else if code >= 400 && code != 429 {
			statusLabel = "4xx"
		} else if code == 429 {
			statusLabel = "429"
		}
		extractorSocrataRequestsTotal.WithLabelValues(statusLabel).Inc()

		if code == 429 || code >= 500 {
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			if ra := resp.Header.Get("Retry-After"); ra != "" {
				if secs, err := strconv.Atoi(strings.TrimSpace(ra)); err == nil {
					time.Sleep(time.Duration(secs) * time.Second)
				} else if when, err := http.ParseTime(ra); err == nil {
					if d := time.Until(when); d > 0 {
						time.Sleep(d)
					}
				} else {
					time.Sleep(time.Duration(2*(i+1)) * time.Second)
				}
			} else {
				time.Sleep(time.Duration(2*(i+1)) * time.Second)
			}
			lastErr = fmt.Errorf("http %d", code)
			continue
		}

		if code != 200 {
			b, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			return nil, fmt.Errorf("http %d: %s", code, string(b))
		}

		raw, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		return raw, err
	}

	return nil, lastErr
}

// =============================
// MinIO write (.json.gz) + metadata + metrics
// =============================

func putJSONGZ(cli *minio.Client, env Env, key string, raw []byte, meta map[string]string) error {
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	_, err := gz.Write(raw)
	if err != nil {
		return err
	}
	if err := gz.Close(); err != nil {
		return err
	}
	reader := bytes.NewReader(buf.Bytes())

	entity := "unknown"
	if meta != nil {
		if v, ok := meta["entity"]; ok && v != "" {
			entity = v
		}
	}

	_, err = cli.PutObject(context.Background(), env.RawBucket, key, reader, int64(reader.Len()), minio.PutObjectOptions{
		ContentType:     "application/json",
		ContentEncoding: "gzip",
		UserMetadata:    meta,
	})

	if err != nil {
		extractorMinioPutErrorsTotal.WithLabelValues(entity).Inc()
		return err
	}

	extractorMinioPutObjectsTotal.WithLabelValues(entity).Inc()
	return nil
}

// =============================
// Helpers
// =============================

func chunkStrings(ids []string, size int) [][]string {
	if size <= 0 {
		size = 300
	}
	var chunks [][]string
	for i := 0; i < len(ids); i += size {
		end := i + size
		if end > len(ids) {
			end = len(ids)
		}
		chunks = append(chunks, ids[i:end])
	}
	return chunks
}

func escapeSQuote(s string) string { return strings.ReplaceAll(s, "'", "''") }

func dialRabbit(url string) (*amqp.Connection, error) {
	var conn *amqp.Connection
	var err error
	for i := 0; i < 10; i++ {
		conn, err = amqp.Dial(url)
		if err == nil {
			return conn, nil
		}
		time.Sleep(time.Second * time.Duration(1+i))
	}
	return nil, err
}

// Ensures required fields are present in $select (NEW)
func ensureSelect(sel string, must ...string) string {
	trim := strings.TrimSpace(sel)
	if trim == "" || trim == "*" {
		return "*" // wide open
	}
	have := map[string]struct{}{}
	for _, part := range strings.Split(trim, ",") {
		have[strings.ToLower(strings.TrimSpace(part))] = struct{}{}
	}
	pre := []string{}
	for _, m := range must {
		k := strings.ToLower(strings.TrimSpace(m))
		if _, ok := have[k]; !ok {
			pre = append(pre, m)
		}
	}
	if len(pre) > 0 {
		return strings.Join(pre, ",") + "," + trim
	}
	return trim
}

// Watermark helpers

func loadWatermark(cli *minio.Client, env Env) time.Time {
	ctx := context.Background()
	watermarkKey := "watermarks/last.json"

	_, err := cli.StatObject(ctx, env.RawBucket, watermarkKey, minio.StatObjectOptions{})
	if err != nil {
		return time.Time{}
	}

	obj, err := cli.GetObject(ctx, env.RawBucket, watermarkKey, minio.GetObjectOptions{})
	if err != nil {
		return time.Time{}
	}
	defer obj.Close()

	b, _ := io.ReadAll(obj)
	var t time.Time
	if err := json.Unmarshal(b, &t); err != nil {
		return time.Time{}
	}
	return t
}

func saveWatermark(cli *minio.Client, env Env, t time.Time) {
	ctx := context.Background()
	watermarkKey := "watermarks/last.json"

	b, err := json.Marshal(t.UTC())
	if err != nil {
		log.Printf("failed to marshal watermark: %v", err)
		return
	}
	reader := bytes.NewReader(b)

	_, err = cli.PutObject(ctx, env.RawBucket, watermarkKey, reader, int64(len(b)), minio.PutObjectOptions{
		ContentType: "application/json",
	})
	if err != nil {
		log.Printf("failed to save watermark: %v", err)
	}
}

// Year parsing from crash_date
func parseCrashYear(s string) (int, bool) {
	if s == "" {
		return 0, false
	}
	if t, e := time.Parse("2006-01-02T15:04:05.000", s); e == nil {
		return t.Year(), true
	}
	if t, e := time.Parse("2006-01-02T15:04:05", s); e == nil {
		return t.Year(), true
	}
	return 0, false
}

// =============================
// Core extraction
// =============================

// ---------- MinIO existence ----------
func objExists(cli *minio.Client, env Env, key string) bool {
	_, err := cli.StatObject(context.Background(), env.RawBucket, key, minio.StatObjectOptions{})
	return err == nil
}

// ---------- WHERE fingerprint ----------
// Makes markers unique to dataset + where + order + select + join_key + page_size + mode
func whereFingerprint(job Job) string {
	s := strings.Join([]string{
		"id=" + job.Primary.ID,
		"where=" + job.Primary.Where,
		"order=" + job.Primary.Order,
		"select=" + job.Primary.Select,
		"join=" + job.JoinKey,
		fmt.Sprintf("pagesz=%d", job.Primary.PageSz),
		"mode=" + job.Mode,
	}, "|")
	sum := sha1.Sum([]byte(s))
	return hex.EncodeToString(sum[:])[:12] // short, stable
}

// ---------- Marker key ----------
func markerKey(job Job, off int) string {
	// independent of corr_id so resumes can skip immediately
	fp := whereFingerprint(job)
	return fmt.Sprintf("_markers/%s/%s/id=%s/fp=%s/off=%d.done",
		job.Storage.Prefix, job.Primary.Alias, job.Primary.ID, fp, off)
}

// ---------- Write marker (with page_max for watermark sync) ----------
func writePageMarker(cli *minio.Client, env Env, key string, pageMax time.Time, corr, alias string) error {
	meta := map[string]string{
		"run_id":     corr,
		"entity":     alias,
		"page_max":   pageMax.UTC().Format(time.RFC3339),
		"created_at": time.Now().UTC().Format(time.RFC3339),
	}
	r := bytes.NewReader([]byte("{}"))
	_, err := cli.PutObject(context.Background(), env.RawBucket, key, r, int64(r.Len()),
		minio.PutObjectOptions{ContentType: "application/json", UserMetadata: meta})

	if err != nil {
		extractorMinioPutErrorsTotal.WithLabelValues("marker").Inc()
		return err
	}
	extractorMinioPutObjectsTotal.WithLabelValues("marker").Inc()
	return nil
}

// ---------- Read marker's page_max (if present) ----------
func readMarkerPageMax(cli *minio.Client, env Env, key string) time.Time {
	st, err := cli.StatObject(context.Background(), env.RawBucket, key, minio.StatObjectOptions{})
	if err != nil {
		return time.Time{}
	}
	var val string
	for k, v := range st.UserMetadata {
		if strings.EqualFold(k, "page_max") {
			val = v
			break
		}
	}
	if val == "" {
		return time.Time{}
	}
	if t, e := time.Parse(time.RFC3339, val); e == nil {
		return t
	}
	return time.Time{}
}

// ---------- Main job processor ----------
func processJob(env Env, mcli *minio.Client, job Job) (string, bool, error) {
	job.applyDefaults()
	lastWatermark := loadWatermark(mcli, env)
	log.Printf("previous watermark: %v", lastWatermark)

	job.buildWhere(lastWatermark)
	logCorr := job.CorrID
	if logCorr == "" {
		logCorr = "(pending)"
	}
	log.Printf("job corr_id=%s where=%q page_size=%d", logCorr, job.Primary.Where, job.Primary.PageSz)

	crashOff := 0
	var runMax time.Time
	var wroteAny bool
	corr := job.CorrID // may be empty; initialized lazily

	// year map for enrich across the whole run
	crashYearByID := make(map[string]int)

	for {
		// PRE-FETCH SKIP: if this offset was already processed for this WHERE window, skip API
		mk := markerKey(job, crashOff)
		if objExists(mcli, env, mk) {
			if t := readMarkerPageMax(mcli, env, mk); t.After(runMax) {
				runMax = t
			}
			log.Printf("offset=%d already done (marker: %s) — skipping", crashOff, mk)
			crashOff += job.Primary.PageSz
			continue
		}

		rows, crashIDs, raw, pageMax, err := fetchCrashesPage(env, job, crashOff)
		if err != nil {
			return "", false, fmt.Errorf("crashes page off=%d: %w", crashOff, err)
		}
		if pageMax.After(runMax) {
			runMax = pageMax
		}
		if rows == 0 {
			log.Printf("no more crashes (offset=%d)", crashOff)
			break
		}

		// Count rows fetched for dashboards
		extractorRowsFetchedTotal.WithLabelValues("crashes").Add(float64(rows))

		extractorPagesProcessedTotal.WithLabelValues("crashes").Inc()

		// Lazy corr assignment: only when we actually write something
		if corr == "" {
			corr = time.Now().Format("2006-01-02-15-04-05")
			log.Printf("corr_id initialized: %s", corr)
		}

		// Parse page and bucket by event year; fill year map
		var page []map[string]any
		if err := json.Unmarshal(raw, &page); err != nil {
			return "", false, err
		}
		buckets := map[int][]map[string]any{}
		for _, r := range page {
			id, _ := r[job.JoinKey].(string)
			y := 0
			if cd, ok := r["crash_date"].(string); ok {
				if yy, ok := parseCrashYear(cd); ok {
					y = yy
				}
			}
			if id != "" && y != 0 {
				crashYearByID[id] = y
			}
			buckets[y] = append(buckets[y], r) // y==0 kept as "unknown"
		}

		// Save one object per year bucket with metadata
		for y, recs := range buckets {
			if len(recs) == 0 {
				continue
			}
			b, _ := json.Marshal(recs)
			key := fmt.Sprintf("%s/%s/year=%04d/corr=%s/offset=%d_limit=%d.json.gz",
				job.Storage.Prefix, job.Primary.Alias, y, corr, crashOff, job.Primary.PageSz)
			meta := map[string]string{
				"run_id":    corr,
				"entity":    job.Primary.Alias,
				"ingest_ts": time.Now().UTC().Format(time.RFC3339),
			}
			if job.DateRange != nil {
				meta["window_start"] = job.DateRange.Start
				meta["window_end"] = job.DateRange.End
			}
			if err := putJSONGZ(mcli, env, key, b, meta); err != nil {
				return "", false, err
			}
			wroteAny = true
			log.Printf("saved: s3://%s/%s (year=%d, rows=%d)", env.RawBucket, key, y, len(recs))
		}

		// Build unique crash IDs for enrich
		uniq := map[string]struct{}{}
		for _, id := range crashIDs {
			uniq[id] = struct{}{}
		}
		ids := make([]string, 0, len(uniq))
		for id := range uniq {
			ids = append(ids, id)
		}
		sort.Strings(ids)
		log.Printf("crashes page off=%d produced %d unique IDs for enrich", crashOff, len(ids))

		// OPTIONAL: if no IDs to enrich, skip the enrich phase
		if len(ids) > 0 {
			batches := chunkStrings(ids, job.Batching.IDBatchSize)

			var wg sync.WaitGroup
			wg.Add(2)
			go func() {
				defer wg.Done()
				fetchEnrichBatches(env, mcli, job, corr, job.EnrichByAlias("vehicles"),
					crashOff, batches, job.Batching.MaxWorkers.Vehicles, crashYearByID)
			}()
			go func() {
				defer wg.Done()
				fetchEnrichBatches(env, mcli, job, corr, job.EnrichByAlias("people"),
					crashOff, batches, job.Batching.MaxWorkers.People, crashYearByID)
			}()
			wg.Wait()
		}

		// POST-PAGE MARKER: write only after successful processing of this page
		if err := writePageMarker(mcli, env, mk, pageMax, corr, job.Primary.Alias); err != nil {
			log.Printf("marker write failed (%s): %v", mk, err)
		}

		crashOff += job.Primary.PageSz
	}

	// Only advance the watermark if we actually wrote something and it's newer
	// Advance watermark for streaming runs whenever we have a newer pageMax
	isBackfill := strings.ToLower(job.Mode) == "backfill" || job.DateRange != nil
	if runMax.After(lastWatermark) && !isBackfill {
		saveWatermark(mcli, env, runMax)
		log.Printf("new watermark saved: %v", runMax)
	} else {
		log.Printf("no watermark change (mode=%s, isBackfill=%v, prev=%v, runMax=%v)",
			job.Mode, isBackfill, lastWatermark, runMax)
	}

	return corr, wroteAny, nil
}

func (j *Job) EnrichByAlias(alias string) *DatasetSpec {
	for i := range j.Enrich {
		if j.Enrich[i].Alias == alias {
			return &j.Enrich[i]
		}
	}
	return nil
}

func fetchCrashesPage(env Env, job Job, offset int) (rows int, crashIDs []string, body []byte, maxUpdated time.Time, err error) {
	q := url.Values{}
	// Ensure join key + crash_date are present so we can build ID list + year map (NEW)
	sel := defaultStr(job.Primary.Select, "*")
	sel = ensureSelect(sel, job.JoinKey, "crash_date")
	q.Set("$select", sel)

	if job.Primary.Where != "" {
		q.Set("$where", job.Primary.Where)
	}
	q.Set("$order", defaultStr(job.Primary.Order, "crash_date, "+job.JoinKey))
	q.Set("$limit", fmt.Sprintf("%d", job.Primary.PageSz))
	q.Set("$offset", fmt.Sprintf("%d", offset))
	u := fmt.Sprintf("%s/resource/%s.json?%s", env.SocrataBase, job.Primary.ID, q.Encode())

	raw, err := httpGetJSON(env, u)
	if err != nil {
		return 0, nil, nil, time.Time{}, err
	}

	var arr []map[string]any
	if err := json.Unmarshal(raw, &arr); err != nil {
		return 0, nil, nil, time.Time{}, err
	}

	rows = len(arr)
	ids := make([]string, 0, rows)
	var maxT time.Time

	for _, r := range arr {
		// collect join keys
		if v, ok := r[job.JoinKey]; ok {
			if s, ok := v.(string); ok && s != "" {
				ids = append(ids, s)
			}
		}
		// track crash_date
		if v, ok := r["crash_date"]; ok {
			if s, ok := v.(string); ok && s != "" {
				// handle both with and without millis
				if t, e := time.Parse("2006-01-02T15:04:05.000", s); e == nil {
					if t.After(maxT) {
						maxT = t
					}
				} else if t, e := time.Parse("2006-01-02T15:04:05", s); e == nil {
					if t.After(maxT) {
						maxT = t
					}
				}
			}
		}
	}

	return rows, ids, raw, maxT, nil
}

func fetchEnrichBatches(env Env, mcli *minio.Client, job Job, corr string, ds *DatasetSpec,
	crashOffset int, batches [][]string, maxWorkers int, yearByID map[string]int) {
	if ds == nil {
		return
	}
	if maxWorkers <= 0 {
		maxWorkers = 1
	}

	type task struct {
		idx int
		ids []string
	}
	tasks := make(chan task, len(batches))
	var wg sync.WaitGroup
	// Workers
	for w := 0; w < maxWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for t := range tasks {
				fetchOneBatchAllPages(env, mcli, job, corr, ds, crashOffset, t.idx, t.ids, yearByID)
			}
		}()
	}
	// Feed tasks
	for i, ids := range batches {
		tasks <- task{idx: i, ids: ids}
	}
	close(tasks)
	wg.Wait()
}

func fetchOneBatchAllPages(env Env, mcli *minio.Client, job Job, corr string, ds *DatasetSpec,
	crashOff, batchIdx int, ids []string, yearByID map[string]int) {
	// Build WHERE: crash_record_id in ('A','B',...)
	vals := make([]string, 0, len(ids))
	for _, id := range ids {
		vals = append(vals, fmt.Sprintf("'%s'", escapeSQuote(id)))
	}
	where := fmt.Sprintf("%s in (%s)", job.JoinKey, strings.Join(vals, ","))

	limit := 50000
	for off := 0; ; off += limit {
		q := url.Values{}
		sel := ds.Select
		if strings.TrimSpace(sel) == "" {
			// minimal default if none provided
			sel = fmt.Sprintf("%s,unit_no", job.JoinKey)
		}
		// Ensure join key is present; don't force crash_date here (dataset may not have it) (NEW)
		sel = ensureSelect(sel, job.JoinKey)
		q.Set("$select", sel)

		q.Set("$where", where)
		q.Set("$limit", fmt.Sprintf("%d", limit))
		q.Set("$offset", fmt.Sprintf("%d", off))
		u := fmt.Sprintf("%s/resource/%s.json?%s", env.SocrataBase, ds.ID, q.Encode())

		raw, err := httpGetJSON(env, u)
		if err != nil {
			log.Printf("enrich %s batch=%d off=%d: %v", ds.Alias, batchIdx, off, err)
			return
		}
		var arr []map[string]any
		if err := json.Unmarshal(raw, &arr); err != nil {
			log.Printf("parse %s: %v", ds.Alias, err)
			return
		}
		rows := len(arr)
		if rows == 0 {
			return
		}

		// Count rows fetched for dashboards
		extractorRowsFetchedTotal.WithLabelValues(ds.Alias).Add(float64(rows))

		// count this enrich page
		extractorPagesProcessedTotal.WithLabelValues(ds.Alias).Inc()

		// Group records by year using crashYear map; fallback to row's own crash_date if present (NEW)
		groups := map[int][]map[string]any{}
		for _, r := range arr {
			y := 0
			if v, ok := r[job.JoinKey]; ok {
				if s, ok := v.(string); ok {
					if yy, ok := yearByID[s]; ok {
						y = yy
					}
				}
			}
			if y == 0 { // optional fallback; safe if dataset doesn't have crash_date
				if cd, ok := r["crash_date"].(string); ok {
					if yy, ok := parseCrashYear(cd); ok {
						y = yy
					}
				}
			}
			groups[y] = append(groups[y], r)
		}

		for y, recs := range groups {
			if len(recs) == 0 {
				continue
			}
			if y == 0 { // avoid year=0000 folders (NEW)
				log.Printf("warning: %s had %d rows with unknown year (batch=%d off=%d)", ds.Alias, len(recs), batchIdx, off)
				continue
			}

			payload, _ := json.Marshal(recs)
			key := fmt.Sprintf("%s/%s/year=%04d/corr=%s/crashes_offset=%d_batch=%d",
				job.Storage.Prefix, ds.Alias, y, corr, crashOff, batchIdx)
			if off > 0 {
				key += fmt.Sprintf("_part=%d", off/limit)
			}
			key += ".json.gz"

			meta := map[string]string{
				"run_id":    corr,
				"entity":    ds.Alias, // vehicles or people
				"ingest_ts": time.Now().UTC().Format(time.RFC3339),
			}
			if job.DateRange != nil {
				meta["window_start"] = job.DateRange.Start
				meta["window_end"] = job.DateRange.End
			}

			if err := putJSONGZ(mcli, env, key, payload, meta); err != nil {
				log.Printf("save %s: %v", key, err)
			} else {
				log.Printf("saved: s3://%s/%s (year=%d, rows=%d)", env.RawBucket, key, y, len(recs))
			}
		}
	}
}

func defaultStr(s, def string) string {
	if s == "" {
		return def
	}
	return s
}

// =============================
// Per-run manifest
// =============================

type Manifest struct {
	Corr       string    `json:"corr"`
	Mode       string    `json:"mode"`
	Where      string    `json:"where"`
	StartedAt  time.Time `json:"started_at"`
	FinishedAt time.Time `json:"finished_at"`
}

func writeManifest(cli *minio.Client, env Env, m Manifest) {
	b, _ := json.MarshalIndent(m, "", "  ")
	key := fmt.Sprintf("_runs/corr=%s/manifest.json", m.Corr)
	r := bytes.NewReader(b)
	_, err := cli.PutObject(context.Background(), env.RawBucket, key, r, int64(r.Len()),
		minio.PutObjectOptions{ContentType: "application/json"})

	if err != nil {
		extractorMinioPutErrorsTotal.WithLabelValues("manifest").Inc()
		log.Printf("manifest write failed: %v", err)
		return
	}

	extractorMinioPutObjectsTotal.WithLabelValues("manifest").Inc()
}

// =============================
// RabbitMQ consumer
// =============================

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	registerMetricsOnce()
	startMetricsServer()

	// Set shared service info gauge
	serviceInfo.WithLabelValues("extractor").Set(1)

	env := getEnv()
	mcli := newMinio(env)

	conn, err := dialRabbit(env.RabbitURL)
	if err != nil {
		extractorRabbitMQConnected.Set(0)
		log.Fatalf("rabbit: %v", err)
	}
	extractorRabbitMQConnected.Set(1)
	defer conn.Close()

	ch, err := conn.Channel()
	if err != nil {
		log.Fatalf("rabbit ch: %v", err)
	}
	defer ch.Close()

	qName := env.ExtractQueue
	if _, err := ch.QueueDeclare(qName, true, false, false, false, nil); err != nil {
		log.Fatalf("queue declare: %v", err)
	}
	msgs, err := ch.Consume(qName, "", false, false, false, false, nil)
	if err != nil {
		log.Fatalf("consume: %v", err)
	}

	log.Printf("Extractor up. Waiting for jobs on queue %q", qName)
	for d := range msgs {
		start := time.Now()

		var job Job
		if err := json.Unmarshal(d.Body, &job); err != nil {
			log.Printf("bad job json: %v", err)
			extractorJobsTotal.WithLabelValues("failure").Inc()
			extractorJobsFailedTotal.Inc()
			extractorJobDurationSeconds.Observe(time.Since(start).Seconds())
			d.Ack(false)
			continue
		}
		if job.Primary.ID == "" {
			log.Printf("job missing primary.id")
			extractorJobsTotal.WithLabelValues("failure").Inc()
			extractorJobsFailedTotal.Inc()
			extractorJobDurationSeconds.Observe(time.Since(start).Seconds())
			d.Ack(false)
			continue
		}
		if len(job.Enrich) == 0 {
			log.Printf("warning: no enrich datasets provided")
		}

		corr, wroteAny, err := processJob(env, mcli, job)

		extractorJobDurationSeconds.Observe(time.Since(start).Seconds())

		if err != nil {
			extractorJobsTotal.WithLabelValues("failure").Inc()
			extractorJobsFailedTotal.Inc()
			log.Printf("job failed: %v", err)
		} else {
			extractorJobsTotal.WithLabelValues("success").Inc()
			extractorLastSuccessUnix.Set(float64(time.Now().Unix()))
			log.Printf("job done in %s (corr=%s, wroteAny=%v)", time.Since(start), corr, wroteAny)

			// Write manifest for lineage (even if no clean published)
			if corr != "" {
				writeManifest(mcli, env, Manifest{
					Corr:       corr,
					Mode:       job.Mode,
					Where:      job.Primary.Where,
					StartedAt:  start.UTC(),
					FinishedAt: time.Now().UTC(),
				})
			}

			// 👉 Publish transform job only if something was written
			if wroteAny && corr != "" {
				tj := TransformJob{
					Type:   "transform",
					CorrID: corr,
				}
				if err := publishTransformJob(env.RabbitURL, tj); err != nil {
					log.Printf("publish transform job failed: %v", err)
				} else {
					log.Printf("published transform job corr=%s -> queue=transform", corr)
				}
			}
		}

		d.Ack(false)
	}
}

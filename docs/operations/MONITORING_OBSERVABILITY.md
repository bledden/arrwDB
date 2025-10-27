# Monitoring & Observability Guide for arrwDB

This guide covers production-grade monitoring and observability setup for arrwDB.

## Table of Contents

- [Overview](#overview)
- [Observability Pillars](#observability-pillars)
- [Metrics with Prometheus](#metrics-with-prometheus)
- [Distributed Tracing with OpenTelemetry](#distributed-tracing-with-opentelemetry)
- [Error Tracking with Sentry](#error-tracking-with-sentry)
- [Logging with ELK/Loki](#logging-with-elkloki)
- [Dashboards with Grafana](#dashboards-with-grafana)
- [Alerting Strategy](#alerting-strategy)
- [SLOs and SLIs](#slos-and-slis)

## Overview

Effective observability enables:

- **Proactive issue detection** before users are impacted
- **Rapid troubleshooting** with distributed tracing
- **Performance optimization** via metrics analysis
- **Capacity planning** based on usage trends
- **SLA/SLO compliance** monitoring

## Observability Pillars

### 1. Metrics (RED Method)
- **Rate**: Requests per second
- **Errors**: Error rate percentage
- **Duration**: Response time percentiles

### 2. Logs (Structured)
- Application logs (JSON format)
- Access logs (nginx/Caddy)
- Audit logs (security events)

### 3. Traces (Distributed)
- Request flow across services
- Performance bottlenecks
- Dependency mapping

## Metrics with Prometheus

### Installation

arrwDB already includes `prometheus-fastapi-instrumentator`. Let's expand it:

```bash
pip install prometheus-client prometheus-fastapi-instrumentator
```

### Enhanced Metrics Configuration

Update `app/api/metrics.py`:

```python
"""Enhanced Prometheus metrics for arrwDB."""

from prometheus_client import Counter, Histogram, Gauge, Info
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from fastapi import FastAPI
import psutil
import time

# Custom metrics
search_duration = Histogram(
    'arrwdb_search_duration_seconds',
    'Search operation duration',
    ['library_id', 'index_type'],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

document_operations = Counter(
    'arrwdb_document_operations_total',
    'Document operations',
    ['operation', 'library_id', 'status']
)

vector_index_size = Gauge(
    'arrwdb_vector_index_size_bytes',
    'Size of vector index in memory',
    ['library_id', 'index_type']
)

active_libraries = Gauge(
    'arrwdb_active_libraries_total',
    'Number of active libraries'
)

embedding_cache_hits = Counter(
    'arrwdb_embedding_cache_hits_total',
    'Embedding cache hit/miss',
    ['status']  # hit or miss
)

api_key_validations = Counter(
    'arrwdb_api_key_validations_total',
    'API key validation results',
    ['status']  # valid, invalid, expired
)

# System metrics
cpu_usage = Gauge('arrwdb_cpu_usage_percent', 'CPU usage percentage')
memory_usage = Gauge('arrwdb_memory_usage_bytes', 'Memory usage in bytes')
disk_usage = Gauge('arrwdb_disk_usage_percent', 'Disk usage percentage')

def collect_system_metrics():
    """Collect system-level metrics."""
    cpu_usage.set(psutil.cpu_percent(interval=1))
    memory_usage.set(psutil.virtual_memory().used)
    disk_usage.set(psutil.disk_usage('/').percent)

def setup_metrics(app: FastAPI):
    """Setup Prometheus metrics with custom metrics."""

    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics", "/health"],
        env_var_name="ENABLE_METRICS",
        inprogress_name="arrwdb_requests_inprogress",
        inprogress_labels=True,
    )

    # Add default metrics
    instrumentator.add(
        metrics.request_size(
            should_include_handler=True,
            should_include_method=True,
            should_include_status=True,
            metric_namespace="arrwdb",
            metric_subsystem="http",
        )
    )

    instrumentator.add(
        metrics.response_size(
            should_include_handler=True,
            should_include_method=True,
            should_include_status=True,
            metric_namespace="arrwdb",
            metric_subsystem="http",
        )
    )

    instrumentator.add(
        metrics.latency(
            should_include_handler=True,
            should_include_method=True,
            should_include_status=True,
            metric_namespace="arrwdb",
            metric_subsystem="http",
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0),
        )
    )

    instrumentator.add(
        metrics.requests(
            should_include_handler=True,
            should_include_method=True,
            should_include_status=True,
            metric_namespace="arrwdb",
            metric_subsystem="http",
        )
    )

    # Instrument the app
    instrumentator.instrument(app).expose(app, endpoint="/metrics")

    # Start system metrics collection
    @app.on_event("startup")
    async def start_metrics_collection():
        import threading
        def collect_loop():
            while True:
                collect_system_metrics()
                time.sleep(15)  # Collect every 15 seconds

        thread = threading.Thread(target=collect_loop, daemon=True)
        thread.start()

    return instrumentator
```

### Prometheus Configuration

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'arrwdb-production'
    environment: 'production'

# Alertmanager configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093

# Load rules once and periodically evaluate them
rule_files:
  - "alerts.yml"

# Scrape configurations
scrape_configs:
  # arrwDB API
  - job_name: 'arrwdb'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  # Node exporter (system metrics)
  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']

  # PostgreSQL exporter (if using external DB)
  - job_name: 'postgres'
    static_configs:
      - targets: ['localhost:9187']
```

### Alert Rules

Create `alerts.yml`:

```yaml
groups:
  - name: arrwdb_alerts
    interval: 30s
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: |
          (
            sum(rate(arrwdb_http_requests_total{status=~"5.."}[5m]))
            /
            sum(rate(arrwdb_http_requests_total[5m]))
          ) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High HTTP error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} over the last 5 minutes"

      # High latency
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95,
            sum(rate(arrwdb_http_request_duration_seconds_bucket[5m])) by (le)
          ) > 1.0
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High API latency detected"
          description: "P95 latency is {{ $value }}s"

      # High memory usage
      - alert: HighMemoryUsage
        expr: arrwdb_memory_usage_bytes / (16 * 1024^3) > 0.9
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is at {{ $value | humanizePercentage }}"

      # API key validation failures
      - alert: HighAuthFailureRate
        expr: |
          sum(rate(arrwdb_api_key_validations_total{status!="valid"}[5m]))
          > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High authentication failure rate"
          description: "{{ $value }} auth failures per second"

      # Service down
      - alert: ServiceDown
        expr: up{job="arrwdb"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "arrwDB service is down"
          description: "arrwDB has been down for more than 1 minute"
```

## Distributed Tracing with OpenTelemetry

### Installation

```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-fastapi \
    opentelemetry-exporter-jaeger opentelemetry-instrumentation-requests
```

### Configuration

Create `app/observability/tracing.py`:

```python
"""OpenTelemetry distributed tracing for arrwDB."""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from fastapi import FastAPI


def setup_tracing(app: FastAPI, service_name: str = "arrwdb"):
    """Setup OpenTelemetry tracing."""

    # Create resource
    resource = Resource.create({
        SERVICE_NAME: service_name,
        SERVICE_VERSION: "1.0.0",
        "environment": "production",
    })

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Configure Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name="localhost",
        agent_port=6831,
    )

    # Add span processor
    provider.add_span_processor(
        BatchSpanProcessor(jaeger_exporter)
    )

    # Set global tracer provider
    trace.set_tracer_provider(provider)

    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)

    # Instrument requests library
    RequestsInstrumentor().instrument()

    return trace.get_tracer(__name__)


def trace_search_operation(library_id: str, query: str):
    """Decorator to trace search operations."""
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("search_operation") as span:
        span.set_attribute("library.id", library_id)
        span.set_attribute("query.length", len(query))
        # Your search logic here
        span.set_attribute("results.count", 10)
```

### Usage Example

```python
from app.observability.tracing import trace_search_operation
from opentelemetry import trace

@app.post("/v1/search")
async def search(request: SearchRequest):
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("search_endpoint") as span:
        span.set_attribute("library.id", request.library_id)
        span.set_attribute("k", request.k)

        # Embedding generation (automatically traced via requests instrumentation)
        with tracer.start_as_current_span("generate_embedding"):
            embedding = await embedding_service.embed(request.query)
            span.set_attribute("embedding.dimension", len(embedding))

        # Vector search
        with tracer.start_as_current_span("vector_search"):
            results = library.search(embedding, k=request.k)
            span.set_attribute("results.count", len(results))

        return results
```

### Docker Compose for Jaeger

```yaml
version: '3.8'

services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "5775:5775/udp"  # zipkin.thrift
      - "6831:6831/udp"  # jaeger.thrift compact
      - "6832:6832/udp"  # jaeger.thrift binary
      - "5778:5778"      # serve configs
      - "16686:16686"    # Web UI
      - "14268:14268"    # jaeger.thrift
      - "14250:14250"    # model.proto
      - "9411:9411"      # Zipkin
    environment:
      - COLLECTOR_ZIPKIN_HOST_PORT=:9411
```

## Error Tracking with Sentry

### Installation

```bash
pip install sentry-sdk[fastapi]
```

### Configuration

Update `app/api/main.py`:

```python
"""Sentry integration for error tracking."""

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration
from app.config import settings


def setup_sentry(app: FastAPI):
    """Setup Sentry error tracking."""

    if not settings.SENTRY_DSN:
        return

    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        environment=settings.ENVIRONMENT,  # production, staging, development
        release=f"arrwdb@{settings.VERSION}",
        traces_sample_rate=0.1,  # 10% of transactions
        profiles_sample_rate=0.1,  # 10% of transactions
        integrations=[
            FastApiIntegration(),
            StarletteIntegration(),
        ],
        # Set traces_sample_rate to 1.0 to capture 100% of transactions
        before_send=before_send_filter,
        attach_stacktrace=True,
        send_default_pii=False,  # GDPR compliance
    )


def before_send_filter(event, hint):
    """Filter events before sending to Sentry."""
    # Don't send health check errors
    if "request" in event:
        url = event["request"].get("url", "")
        if "/health" in url or "/metrics" in url:
            return None

    # Don't send expected validation errors
    if "exc_info" in hint:
        exc_type, exc_value, tb = hint["exc_info"]
        if isinstance(exc_value, ValidationError):
            return None

    return event
```

### Custom Context

```python
from sentry_sdk import configure_scope, capture_exception

@app.post("/v1/documents")
async def create_document(document: Document, request: Request):
    try:
        # Add custom context
        with configure_scope() as scope:
            scope.set_user({"id": request.state.tenant_id})
            scope.set_tag("library_id", document.library_id)
            scope.set_context("document", {
                "chunks": len(document.chunks),
                "metadata": document.metadata,
            })

        result = await library_service.add_document(document)
        return result

    except Exception as e:
        # Capture exception with context
        capture_exception(e)
        raise
```

## Logging with ELK/Loki

### Structured Logging

Update `app/logging_config.py`:

```python
"""Structured logging configuration."""

import logging
import json
import sys
from datetime import datetime
from pythonjsonlogger import jsonlogger


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields."""

    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)

        # Add timestamp
        log_record['timestamp'] = datetime.utcnow().isoformat()

        # Add severity mapping
        log_record['severity'] = record.levelname

        # Add source location
        log_record['source'] = {
            'file': record.filename,
            'line': record.lineno,
            'function': record.funcName,
        }

        # Add request context if available
        if hasattr(record, 'request_id'):
            log_record['request_id'] = record.request_id
        if hasattr(record, 'tenant_id'):
            log_record['tenant_id'] = record.tenant_id


def setup_logging():
    """Setup structured JSON logging."""

    # Create handler
    handler = logging.StreamHandler(sys.stdout)

    # Set formatter
    formatter = CustomJsonFormatter(
        '%(timestamp)s %(level)s %(name)s %(message)s'
    )
    handler.setFormatter(formatter)

    # Configure root logger
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(handler)
```

### Loki Configuration

`promtail-config.yml`:

```yaml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  # arrwDB application logs
  - job_name: arrwdb
    static_configs:
      - targets:
          - localhost
        labels:
          job: arrwdb
          __path__: /var/log/arrwdb/*.log
    pipeline_stages:
      - json:
          expressions:
            timestamp: timestamp
            level: severity
            message: message
            request_id: request_id
      - labels:
          level:
          request_id:
```

## Dashboards with Grafana

### Grafana Dashboard JSON

Create `grafana-dashboard.json` (example):

```json
{
  "dashboard": {
    "title": "arrwDB Operations",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(arrwdb_http_requests_total[5m])",
            "legendFormat": "{{ method }} {{ handler }}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "sum(rate(arrwdb_http_requests_total{status=~\"5..\"}[5m])) / sum(rate(arrwdb_http_requests_total[5m]))",
            "legendFormat": "Error %"
          }
        ],
        "type": "graph"
      },
      {
        "title": "P95 Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(arrwdb_http_request_duration_seconds_bucket[5m])) by (le, handler))",
            "legendFormat": "{{ handler }}"
          }
        ],
        "type": "graph"
      }
    ]
  }
}
```

### Key Dashboards

1. **Service Overview**
   - Request rate
   - Error rate
   - P50/P95/P99 latency
   - Active connections

2. **Resource Utilization**
   - CPU usage
   - Memory usage
   - Disk I/O
   - Network throughput

3. **Business Metrics**
   - Active libraries
   - Documents indexed
   - Searches per minute
   - API key usage

4. **Error Analysis**
   - Error count by type
   - Error rate trends
   - Stack traces (link to Sentry)

## Alerting Strategy

### Alert Priorities

**P1 (Critical) - Page immediately:**
- Service down (>1 minute)
- Error rate >5% for 5 minutes
- P95 latency >5s for 10 minutes

**P2 (High) - Page during business hours:**
- Error rate >2% for 15 minutes
- Memory usage >90% for 15 minutes
- Disk usage >85%

**P3 (Medium) - Ticket only:**
- Error rate >1% for 30 minutes
- P95 latency >2s for 30 minutes
- CPU usage >80% for 30 minutes

### Alertmanager Configuration

```yaml
global:
  slack_api_url: 'YOUR_SLACK_WEBHOOK'

route:
  receiver: 'default'
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  routes:
    - match:
        severity: critical
      receiver: 'pagerduty'
      continue: true
    - match:
        severity: warning
      receiver: 'slack'

receivers:
  - name: 'default'
    slack_configs:
      - channel: '#arrwdb-alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_KEY'
```

## SLOs and SLIs

### Service Level Indicators (SLIs)

1. **Availability**: Percentage of successful requests
   ```promql
   sum(rate(arrwdb_http_requests_total{status!~"5.."}[30d]))
   /
   sum(rate(arrwdb_http_requests_total[30d]))
   ```

2. **Latency**: P95 response time
   ```promql
   histogram_quantile(0.95,
     sum(rate(arrwdb_http_request_duration_seconds_bucket[5m])) by (le)
   )
   ```

3. **Error Rate**: Percentage of failed requests
   ```promql
   sum(rate(arrwdb_http_requests_total{status=~"5.."}[5m]))
   /
   sum(rate(arrwdb_http_requests_total[5m]))
   ```

### Service Level Objectives (SLOs)

| Metric | Target | Measurement Window |
|--------|--------|--------------------|
| Availability | 99.9% | 30 days |
| P95 Latency | < 500ms | 30 days |
| P99 Latency | < 2s | 30 days |
| Error Rate | < 0.1% | 30 days |

### Error Budget

```
Error Budget = (1 - SLO) × Total Requests

Example:
- SLO: 99.9% availability
- Total requests: 10M/month
- Error budget: 10,000 failed requests/month
```

## Complete Observability Stack

### Docker Compose

```yaml
version: '3.8'

services:
  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - ./alerts.yml:/etc/prometheus/alerts.yml
    ports:
      - "9090:9090"

  # Grafana
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

  # Jaeger (tracing)
  jaeger:
    image: jaegertracing/all-in-in:latest
    ports:
      - "16686:16686"
      - "6831:6831/udp"

  # Loki (logs)
  loki:
    image: grafana/loki:latest
    ports:
      - "3100:3100"

  # Promtail (log shipper)
  promtail:
    image: grafana/promtail:latest
    volumes:
      - /var/log:/var/log
      - ./promtail-config.yml:/etc/promtail/config.yml

  # Alertmanager
  alertmanager:
    image: prom/alertmanager:latest
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml

volumes:
  grafana-storage:
```

## Monitoring Checklist

- [ ] Prometheus metrics endpoint exposed
- [ ] Custom business metrics defined
- [ ] Alert rules configured
- [ ] Grafana dashboards created
- [ ] OpenTelemetry tracing enabled
- [ ] Jaeger UI accessible
- [ ] Sentry error tracking configured
- [ ] Structured logging implemented
- [ ] Log aggregation (Loki/ELK) set up
- [ ] Alertmanager configured
- [ ] PagerDuty/Slack integration tested
- [ ] SLOs defined and tracked
- [ ] Error budget calculated
- [ ] Runbooks created for common alerts
- [ ] On-call rotation established

## Additional Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [OpenTelemetry Python](https://opentelemetry.io/docs/instrumentation/python/)
- [Sentry Python SDK](https://docs.sentry.io/platforms/python/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Google SRE Book - Monitoring](https://sre.google/sre-book/monitoring-distributed-systems/)

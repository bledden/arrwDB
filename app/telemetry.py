"""
OpenTelemetry and Sentry initialization for distributed tracing and error tracking.

Configuration (all via environment variables):
    OTEL_ENABLED=true              Enable OpenTelemetry tracing
    OTEL_SERVICE_NAME=arrwdb       Service name in traces (default: arrwdb)
    OTEL_EXPORTER_OTLP_ENDPOINT   OTLP endpoint (default: http://localhost:4318)

    SENTRY_DSN=https://...         Sentry DSN (if unset, Sentry is disabled)
    SENTRY_ENVIRONMENT=production  Sentry environment tag
    SENTRY_TRACES_SAMPLE_RATE=1.0  Sentry performance monitoring rate
"""

import logging
import os

logger = logging.getLogger(__name__)

_tracer_provider = None


def init_telemetry() -> None:
    """Initialize OpenTelemetry and Sentry. Safe to call when disabled."""
    global _tracer_provider

    otel_enabled = os.getenv("OTEL_ENABLED", "false").lower() == "true"
    sentry_dsn = os.getenv("SENTRY_DSN", "")

    # Sentry first so it can hook into OTEL
    if sentry_dsn:
        _init_sentry(sentry_dsn)

    if otel_enabled:
        _tracer_provider = _init_otel()

    if not otel_enabled and not sentry_dsn:
        logger.info("Telemetry disabled (set OTEL_ENABLED=true or SENTRY_DSN to enable)")


def _init_sentry(dsn: str) -> None:
    """Initialize Sentry SDK with FastAPI integration."""
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.starlette import StarletteIntegration

    sentry_sdk.init(
        dsn=dsn,
        environment=os.getenv("SENTRY_ENVIRONMENT", "production"),
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "1.0")),
        integrations=[
            FastApiIntegration(transaction_style="endpoint"),
            StarletteIntegration(transaction_style="endpoint"),
        ],
        send_default_pii=True,
        enable_logs=True,
        profile_session_sample_rate=float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "1.0")),
        profile_lifecycle="trace",
    )
    logger.info("Sentry initialized (environment=%s)", os.getenv("SENTRY_ENVIRONMENT", "production"))


def _init_otel():
    """Initialize OpenTelemetry tracing with OTLP export."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.logging import LoggingInstrumentor

    service_name = os.getenv("OTEL_SERVICE_NAME", "arrwdb")

    resource = Resource.create({
        SERVICE_NAME: service_name,
        "service.version": "1.0.0",
        "deployment.environment": os.getenv("SENTRY_ENVIRONMENT", "production"),
    })

    provider = TracerProvider(resource=resource)

    # OTLP exporter
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    exporter = OTLPSpanExporter(endpoint=f"{otlp_endpoint}/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(exporter))

    # Bridge traces to Sentry if configured
    sentry_dsn = os.getenv("SENTRY_DSN", "")
    if sentry_dsn:
        try:
            from sentry_sdk.integrations.opentelemetry import SentrySpanProcessor
            provider.add_span_processor(SentrySpanProcessor())
            logger.info("Sentry OTEL SpanProcessor registered")
        except ImportError:
            logger.warning("sentry_sdk OpenTelemetry integration not available")

    trace.set_tracer_provider(provider)

    # Auto-instrument requests library (Cohere SDK uses this)
    RequestsInstrumentor().instrument()

    # Auto-instrument logging (injects trace_id, span_id into log records)
    LoggingInstrumentor().instrument(set_logging_format=False)

    logger.info(
        "OpenTelemetry initialized (service=%s, endpoint=%s)",
        service_name, otlp_endpoint,
    )

    return provider


def get_tracer(name: str = __name__):
    """Get an OTEL tracer. Returns a no-op tracer if OTEL is not enabled."""
    from opentelemetry import trace
    return trace.get_tracer(name)


def shutdown_telemetry() -> None:
    """Flush and shut down telemetry providers."""
    global _tracer_provider
    if _tracer_provider is not None:
        _tracer_provider.shutdown()
        logger.info("OpenTelemetry tracer provider shut down")

    sentry_dsn = os.getenv("SENTRY_DSN", "")
    if sentry_dsn:
        try:
            import sentry_sdk
            sentry_sdk.flush(timeout=5.0)
            logger.info("Sentry flushed")
        except Exception:
            pass

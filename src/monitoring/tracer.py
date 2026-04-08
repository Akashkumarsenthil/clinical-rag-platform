"""OpenTelemetry tracing configuration.

Gracefully degrades when the OTLP exporter package is not installed —
returns a no-op tracer so the rest of the application still works.
"""

from __future__ import annotations

import structlog
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider

logger = structlog.get_logger(__name__)

try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    _HAS_OTLP = True
except ImportError:
    _HAS_OTLP = False


def setup_tracing(
    service_name: str,
    otlp_endpoint: str = "http://localhost:4317",
) -> trace.Tracer:
    """Configure OpenTelemetry tracing with an OTLP gRPC exporter.

    Falls back to a no-op tracer when the OTLP exporter is unavailable.
    """
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if _HAS_OTLP:
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        logger.info("tracing_configured", service=service_name, otlp_endpoint=otlp_endpoint)
    else:
        logger.warning("otlp_exporter_unavailable", msg="Running with no-op tracer")

    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)

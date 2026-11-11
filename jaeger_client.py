"""Send a traced vLLM request to the local Jaeger all-in-one collector."""

import requests

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import SpanKind
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator


provider = TracerProvider(resource=Resource.create({"service.name": "client-service"}))
provider.add_span_processor(
    BatchSpanProcessor(
        OTLPSpanExporter(endpoint="http://127.0.0.1:4318/v1/traces")
    )
)
tracer = provider.get_tracer(__name__)

with tracer.start_as_current_span("client-span", kind=SpanKind.CLIENT) as span:
    headers = {}
    TraceContextTextMapPropagator().inject(headers)
    response = requests.post(
        "http://127.0.0.1:8000/v1/chat/completions",
        headers=headers,
        json={
            "model": "opt-125m-otel",
            "messages": [{"role": "user", "content": "Jaeger full trace prompt"}],
            "max_tokens": 4,
        },
        timeout=30,
    )
    response.raise_for_status()
    span.set_attribute("http.response.status_code", response.status_code)
    print(f"trace_id={span.get_span_context().trace_id:032x}")

provider.force_flush()

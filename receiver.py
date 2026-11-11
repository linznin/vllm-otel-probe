import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from google.protobuf.json_format import MessageToDict
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import (
    ExportLogsServiceRequest,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)


OUTPUT = Path(os.getenv("OTLP_OUTPUT", str(Path(__file__).with_name("traces.jsonl"))))
PORT = int(os.getenv("OTLP_PORT", "4318"))


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/v1/traces":
            request = ExportTraceServiceRequest()
            resource_key, scope_key, record_key, signal = (
                "resource_spans", "scope_spans", "spans", "traces"
            )
        elif self.path == "/v1/logs":
            request = ExportLogsServiceRequest()
            resource_key, scope_key, record_key, signal = (
                "resource_logs", "scope_logs", "log_records", "logs"
            )
        else:
            self.send_error(404)
            return
        body = self.rfile.read(int(self.headers["Content-Length"]))
        request.ParseFromString(body)
        record = MessageToDict(request, preserving_proto_field_name=True)
        record["signal"] = signal
        with OUTPUT.open("a") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
        count = sum(
            len(scope.get(record_key, []))
            for resource in record.get(resource_key, [])
            for scope in resource.get(scope_key, [])
        )
        print(f"received {count} {signal}", flush=True)
        self.send_response(200)
        self.end_headers()


print(f"OTLP HTTP receiver listening on 127.0.0.1:{PORT}", flush=True)
HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()

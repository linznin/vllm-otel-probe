"""Sample vLLM user prompts into OTLP logs correlated with GenAI spans."""

import json
import os
import secrets

from opentelemetry._logs import SeverityNumber
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter as GrpcLogExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as GrpcSpanExporter,
)
from opentelemetry.exporter.otlp.proto.http._log_exporter import (
    OTLPLogExporter as HttpLogExporter,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HttpSpanExporter,
)
from opentelemetry.semconv_ai import SpanAttributes as AIAttributes
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import SpanKind
from opentelemetry.trace.propagation.tracecontext import (
    TraceContextTextMapPropagator,
)


_PATHS = {"/v1/completions", "/v1/chat/completions"}


def _arguments(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _tool_part(call):
    if not isinstance(call, dict):
        return None
    function = call.get("function")
    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
        return None
    part = {"type": "tool_call", "name": function["name"]}
    if isinstance(call.get("id"), str):
        part["id"] = call["id"]
    if function.get("arguments") is not None:
        part["arguments"] = _arguments(function["arguments"])
    return part


def _message_tool_parts(message):
    parts = []
    for call in message.get("tool_calls") or []:
        part = _tool_part(call)
        if part:
            parts.append(part)
    legacy = message.get("function_call")
    if isinstance(legacy, dict):
        part = _tool_part({"function": legacy})
        if part:
            parts.append(part)
    return parts


def _user_text(path: str, data: dict) -> tuple[str | None, str | None]:
    if path == "/v1/completions":
        prompt = data.get("prompt")
        if isinstance(prompt, str):
            return prompt, json.dumps(
                [{"role": "user", "parts": [{"type": "text", "content": prompt}]}],
                ensure_ascii=False,
            )
        if isinstance(prompt, list) and all(isinstance(x, str) for x in prompt):
            text = "\n".join(prompt)
            return text, json.dumps(
                [{"role": "user", "parts": [{"type": "text", "content": x}]} for x in prompt],
                ensure_ascii=False,
            )
        return None, None

    messages = data.get("messages", [])
    if not isinstance(messages, list):
        return None, None
    user_texts = []
    structured = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "user":
            parts = []
            if isinstance(content, str):
                parts.append({"type": "text", "content": content})
            elif isinstance(content, list):
                parts.extend(
                    {"type": "text", "content": item["text"]}
                    for item in content
                    if isinstance(item, dict)
                    and item.get("type") == "text"
                    and isinstance(item.get("text"), str)
                )
            if parts:
                user_texts.extend(part["content"] for part in parts)
                structured.append({"role": "user", "parts": parts})
        elif role == "assistant":
            parts = []
            if isinstance(content, str) and content:
                parts.append({"type": "text", "content": content})
            parts.extend(_message_tool_parts(message))
            if parts:
                structured.append({"role": "assistant", "parts": parts})
        elif role in ("tool", "function") and isinstance(content, str):
            part = {"type": "tool_call_response", "response": _arguments(content)}
            if isinstance(message.get("tool_call_id"), str):
                part["id"] = message["tool_call_id"]
            structured.append({"role": "tool", "parts": [part]})
    if not user_texts:
        return None, None
    return "\n".join(user_texts), json.dumps(structured, ensure_ascii=False)


def _assistant_output(path: str, body: bytes, streaming: bool):
    """Return text and tool-call choices, model, and usage."""
    if streaming:
        choices = {}
        response_model = None
        usage = None
        for line in body.decode("utf-8").splitlines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                continue
            try:
                data = json.loads(payload)
            except ValueError:
                continue
            if not isinstance(data, dict):
                continue
            response_model = data.get("model") or response_model
            usage = data.get("usage") or usage
            for choice in data.get("choices") or []:
                if not isinstance(choice, dict) or not isinstance(choice.get("index"), int):
                    continue
                index = choice["index"]
                item = choices.setdefault(
                    index,
                    {"text": "", "finish_reason": None, "tool_calls": {}, "legacy": None},
                )
                delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
                fragment = (
                    delta.get("content")
                    if path == "/v1/chat/completions"
                    else choice.get("text")
                )
                if isinstance(fragment, str):
                    item["text"] += fragment
                for call in delta.get("tool_calls") or []:
                    if not isinstance(call, dict) or not isinstance(call.get("index"), int):
                        continue
                    call_state = item["tool_calls"].setdefault(
                        call["index"], {"id": "", "name": "", "arguments": ""}
                    )
                    if isinstance(call.get("id"), str):
                        call_state["id"] += call["id"]
                    function = call.get("function")
                    if isinstance(function, dict):
                        for key in ("name", "arguments"):
                            if isinstance(function.get(key), str):
                                call_state[key] += function[key]
                legacy = delta.get("function_call")
                if isinstance(legacy, dict):
                    call_state = item["legacy"]
                    if call_state is None:
                        call_state = item["legacy"] = {"name": "", "arguments": ""}
                    for key in ("name", "arguments"):
                        if isinstance(legacy.get(key), str):
                            call_state[key] += legacy[key]
                item["finish_reason"] = choice.get("finish_reason") or item["finish_reason"]
        outputs = []
        for index in sorted(choices):
            state = choices[index]
            parts = []
            for call_index in sorted(state["tool_calls"]):
                call_state = state["tool_calls"][call_index]
                call = {
                    "id": call_state["id"],
                    "function": {
                        "name": call_state["name"],
                        "arguments": call_state["arguments"],
                    },
                }
                part = _tool_part(call)
                if part:
                    parts.append(part)
            if state["legacy"]:
                part = _tool_part({"function": state["legacy"]})
                if part:
                    parts.append(part)
            outputs.append(
                {
                    "text": state["text"],
                    "finish_reason": state["finish_reason"],
                    "tool_calls": parts,
                }
            )
        return outputs, response_model, usage

    data = json.loads(body)
    if not isinstance(data, dict):
        raise ValueError("Response body must be an object")
    choices = []
    for choice in data.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        value = (
            choice.get("message", {}).get("content")
            if path == "/v1/chat/completions"
            and isinstance(choice.get("message"), dict)
            else choice.get("text")
        )
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        tool_calls = _message_tool_parts(message)
        if isinstance(value, str) or tool_calls:
            choices.append(
                {
                    "text": value if isinstance(value, str) else "",
                    "finish_reason": choice.get("finish_reason"),
                    "tool_calls": tool_calls,
                }
            )
    return choices, data.get("model"), data.get("usage")


class PromptCaptureMiddleware:
    """ASGI middleware loaded by ``vllm serve --middleware``."""

    def __init__(self, app):
        self.app = app
        self.sample_rate = float(os.getenv("VLLM_PROMPT_SAMPLE_RATE", "0.1"))
        if not 0 <= self.sample_rate <= 1:
            raise ValueError("VLLM_PROMPT_SAMPLE_RATE must be between 0 and 1")
        self.max_chars = int(os.getenv("VLLM_PROMPT_MAX_CHARS", "2048"))
        self.max_body_bytes = int(os.getenv("VLLM_PROMPT_MAX_BODY_BYTES", "1048576"))
        self.max_response_bytes = int(
            os.getenv("VLLM_PROMPT_MAX_RESPONSE_BYTES", "1048576")
        )
        self.log_export_interval_ms = int(
            os.getenv("VLLM_PROMPT_LOG_EXPORT_INTERVAL_MS", "200")
        )
        self.span_content = os.getenv("VLLM_PROMPT_SPAN_CONTENT", "0") == "1"
        if min(
            self.max_chars,
            self.max_body_bytes,
            self.max_response_bytes,
            self.log_export_interval_ms,
        ) < 1:
            raise ValueError("Content capture limits must be positive")

        endpoint = os.getenv("VLLM_PROMPT_OTLP_ENDPOINT") or os.getenv(
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
        )
        if not endpoint:
            raise ValueError("Set VLLM_PROMPT_OTLP_ENDPOINT to the OTLP traces URL")
        protocol = os.getenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "grpc")
        log_protocol = os.getenv("OTEL_EXPORTER_OTLP_LOGS_PROTOCOL", protocol)
        log_endpoint = os.getenv("VLLM_PROMPT_OTLP_LOGS_ENDPOINT") or os.getenv(
            "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"
        )
        if not log_endpoint:
            if log_protocol == "http/protobuf" and endpoint.endswith("/v1/traces"):
                log_endpoint = endpoint[: -len("/v1/traces")] + "/v1/logs"
            elif log_protocol == "grpc":
                log_endpoint = endpoint
            else:
                raise ValueError("Set VLLM_PROMPT_OTLP_LOGS_ENDPOINT to the OTLP logs URL")
        if protocol == "http/protobuf":
            exporter = HttpSpanExporter(endpoint=endpoint)
        elif protocol == "grpc":
            exporter = GrpcSpanExporter(
                endpoint=endpoint, insecure=endpoint.startswith("http://")
            )
        else:
            raise ValueError(f"Unsupported OTLP traces protocol: {protocol}")
        if log_protocol == "http/protobuf":
            log_exporter = HttpLogExporter(endpoint=log_endpoint)
        elif log_protocol == "grpc":
            log_exporter = GrpcLogExporter(
                endpoint=log_endpoint, insecure=log_endpoint.startswith("http://")
            )
        else:
            raise ValueError(f"Unsupported OTLP logs protocol: {log_protocol}")

        service_name = os.getenv("OTEL_SERVICE_NAME", "vllm-server")
        resource = Resource.create({"service.name": service_name})
        self.provider = TracerProvider(resource=resource)
        self.provider.add_span_processor(BatchSpanProcessor(exporter))
        self.tracer = self.provider.get_tracer(__name__)
        self.log_provider = LoggerProvider(resource=resource)
        self.log_provider.add_log_record_processor(
            BatchLogRecordProcessor(
                log_exporter, schedule_delay_millis=self.log_export_interval_ms
            )
        )
        self.logger = self.log_provider.get_logger(__name__)
        self.propagator = TraceContextTextMapPropagator()

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or path not in _PATHS
            or secrets.randbelow(1_000_000) >= self.sample_rate * 1_000_000
        ):
            await self.app(scope, receive, send)
            return

        headers = {
            key.decode("latin1").lower(): value.decode("latin1")
            for key, value in scope["headers"]
        }
        parent = self.propagator.extract(headers)
        chunks = []
        captured_size = 0
        too_large = False
        response_chunks = []
        response_size = 0
        response_too_large = False
        response_is_streaming = False
        stream_pending = bytearray()
        stream_logged_chars = 0
        stream_chunk_index = 0
        status_code = None

        async def capture_receive():
            nonlocal captured_size, too_large
            message = await receive()
            if message["type"] == "http.request" and not too_large:
                chunk = message.get("body", b"")
                captured_size += len(chunk)
                if captured_size > self.max_body_bytes:
                    too_large = True
                    chunks.clear()
                else:
                    chunks.append(chunk)
            return message

        operation = "chat" if path == "/v1/chat/completions" else "text_completion"
        with self.tracer.start_as_current_span(
            operation, context=parent, kind=SpanKind.SERVER
        ) as span:
            span.set_attribute("http.route", path)
            span.set_attribute("gen_ai.operation.name", operation)
            span.set_attribute("gen_ai.provider.name", "vllm")
            span.set_attribute(AIAttributes.LLM_SYSTEM, "vllm")
            span.set_attribute(AIAttributes.LLM_REQUEST_TYPE, operation)
            span.set_attribute(AIAttributes.TRACELOOP_SPAN_KIND, "llm")
            carrier = {}
            self.propagator.inject(carrier)
            forwarded_scope = dict(scope)
            forwarded_scope["headers"] = [
                (key, value)
                for key, value in scope["headers"]
                if key.lower() not in (b"traceparent", b"tracestate")
            ] + [(key.encode(), value.encode()) for key, value in carrier.items()]

            def emit_tool_delta(call, choice_index, response_model, legacy=False):
                nonlocal stream_logged_chars, stream_chunk_index
                if not isinstance(call, dict):
                    return
                function = call if legacy else call.get("function")
                if not isinstance(function, dict):
                    function = {}
                fragments = (
                    ("vllm.tool_call.id_delta", call.get("id") if not legacy else None),
                    ("vllm.tool_call.name_delta", function.get("name")),
                    ("vllm.tool_call.arguments_delta", function.get("arguments")),
                )
                attributes = {
                    AIAttributes.LLM_SYSTEM: "vllm",
                    AIAttributes.LLM_REQUEST_TYPE: operation,
                    "gen_ai.operation.name": operation,
                    "gen_ai.provider.name": "vllm",
                    "http.route": path,
                    "vllm.stream.chunk_index": stream_chunk_index,
                    "vllm.choice.index": choice_index,
                    "vllm.tool_call.legacy": legacy,
                }
                if isinstance(call.get("index"), int):
                    attributes["vllm.tool_call.index"] = call["index"]
                if isinstance(response_model, str):
                    attributes[AIAttributes.LLM_RESPONSE_MODEL] = response_model
                truncated = False
                has_fragment = False
                for key, value in fragments:
                    if not isinstance(value, str) or not value:
                        continue
                    remaining = self.max_chars - stream_logged_chars
                    emitted = value[:remaining] if remaining > 0 else ""
                    truncated |= len(emitted) < len(value)
                    if emitted:
                        attributes[key] = emitted
                        stream_logged_chars += len(emitted)
                        has_fragment = True
                if not has_fragment:
                    return
                attributes["vllm.response_capture.truncated"] = truncated
                self.logger.emit(
                    severity_number=SeverityNumber.INFO,
                    body="vllm.tool_call.chunk",
                    attributes=attributes,
                )
                stream_chunk_index += 1

            def emit_stream_line(line):
                nonlocal stream_logged_chars, stream_chunk_index
                if not line.startswith(b"data:"):
                    return
                payload = line[5:].strip()
                if not payload or payload == b"[DONE]":
                    return
                try:
                    event = json.loads(payload)
                except (ValueError, UnicodeDecodeError):
                    return
                if not isinstance(event, dict):
                    return
                for choice in event.get("choices") or []:
                    if not isinstance(choice, dict):
                        continue
                    choice_index = choice.get("index")
                    delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
                    fragment = (
                        delta.get("content")
                        if path == "/v1/chat/completions"
                        else choice.get("text")
                    )
                    if isinstance(fragment, str) and fragment:
                        remaining = self.max_chars - stream_logged_chars
                        emitted = fragment[:remaining] if remaining > 0 else ""
                        if emitted:
                            stream_logged_chars += len(emitted)
                            attributes = {
                                AIAttributes.GEN_AI_CONTENT_COMPLETION_CHUNK: emitted,
                                AIAttributes.LLM_SYSTEM: "vllm",
                                AIAttributes.LLM_REQUEST_TYPE: operation,
                                "gen_ai.operation.name": operation,
                                "gen_ai.provider.name": "vllm",
                                "http.route": path,
                                "vllm.stream.chunk_index": stream_chunk_index,
                                "vllm.response_capture.truncated": len(emitted) < len(fragment),
                            }
                            if isinstance(choice_index, int):
                                attributes["vllm.choice.index"] = choice_index
                            if isinstance(event.get("model"), str):
                                attributes[AIAttributes.LLM_RESPONSE_MODEL] = event["model"]
                            self.logger.emit(
                                severity_number=SeverityNumber.INFO,
                                body="vllm.completion.chunk",
                                attributes=attributes,
                            )
                            stream_chunk_index += 1
                    if not isinstance(choice_index, int):
                        continue
                    for call in delta.get("tool_calls") or []:
                        emit_tool_delta(call, choice_index, event.get("model"))
                    if isinstance(delta.get("function_call"), dict):
                        emit_tool_delta(
                            delta["function_call"], choice_index, event.get("model"), True
                        )

            async def capture_send(message):
                nonlocal status_code, response_size, response_too_large
                nonlocal response_is_streaming
                if message["type"] == "http.response.start":
                    status_code = message["status"]
                    span.set_attribute("http.response.status_code", status_code)
                    response_is_streaming = any(
                        key.lower() == b"content-type"
                        and value.lower().startswith(b"text/event-stream")
                        for key, value in message.get("headers", [])
                    )
                elif message["type"] == "http.response.body" and not response_too_large:
                    chunk = message.get("body", b"")
                    response_size += len(chunk)
                    if response_size > self.max_response_bytes:
                        response_too_large = True
                        response_chunks.clear()
                        stream_pending.clear()
                    else:
                        response_chunks.append(chunk)
                        if (
                            response_is_streaming
                            and status_code is not None
                            and status_code < 400
                            and not too_large
                        ):
                            stream_pending.extend(chunk)
                            while b"\n" in stream_pending:
                                line, _, rest = stream_pending.partition(b"\n")
                                stream_pending[:] = rest
                                emit_stream_line(line.rstrip(b"\r"))
                            if not message.get("more_body", False) and stream_pending:
                                emit_stream_line(bytes(stream_pending).rstrip(b"\r"))
                                stream_pending.clear()
                await send(message)

            try:
                await self.app(forwarded_scope, capture_receive, capture_send)
            finally:
                if status_code is None or status_code >= 400:
                    span.set_attribute("vllm.prompt_capture.skipped", "request_failed")
                elif too_large:
                    span.set_attribute("vllm.prompt_capture.skipped", "body_too_large")
                elif chunks:
                    try:
                        data = json.loads(b"".join(chunks))
                        if not isinstance(data, dict):
                            raise ValueError("Request body must be an object")
                        text, messages = _user_text(path, data)
                    except (ValueError, UnicodeDecodeError):
                        span.set_attribute("vllm.prompt_capture.skipped", "invalid_json")
                    else:
                        model = data.get("model")
                        if isinstance(model, str) and model:
                            span.set_attribute(AIAttributes.LLM_REQUEST_MODEL, model)
                            span.update_name(f"{operation} {model}")
                        for request_key, attribute in (
                            ("max_tokens", AIAttributes.LLM_REQUEST_MAX_TOKENS),
                            ("temperature", AIAttributes.LLM_REQUEST_TEMPERATURE),
                            ("top_p", AIAttributes.LLM_REQUEST_TOP_P),
                            ("n", AIAttributes.GEN_AI_REQUEST_N),
                            ("frequency_penalty", AIAttributes.LLM_FREQUENCY_PENALTY),
                            ("presence_penalty", AIAttributes.LLM_PRESENCE_PENALTY),
                        ):
                            value = data.get(request_key)
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                span.set_attribute(attribute, value)
                        if isinstance(data.get("stream"), bool):
                            span.set_attribute(AIAttributes.GEN_AI_IS_STREAMING, data["stream"])
                        if text:
                            if self.span_content:
                                span.set_attribute(
                                    AIAttributes.LLM_PROMPTS, text[: self.max_chars]
                                )
                            log_attributes = {
                                AIAttributes.LLM_PROMPTS: text[: self.max_chars],
                                AIAttributes.LLM_SYSTEM: "vllm",
                                AIAttributes.LLM_REQUEST_TYPE: operation,
                                "gen_ai.operation.name": operation,
                                "gen_ai.provider.name": "vllm",
                                "http.route": path,
                                "vllm.prompt_capture.truncated": len(text) > self.max_chars,
                            }
                            if isinstance(model, str) and model:
                                log_attributes[AIAttributes.LLM_REQUEST_MODEL] = model
                            if len(messages) <= self.max_chars:
                                log_attributes["gen_ai.input.messages"] = messages
                            tool_definitions = []
                            for tool in data.get("tools") or []:
                                if not isinstance(tool, dict) or tool.get("type") != "function":
                                    continue
                                function = tool.get("function")
                                if isinstance(function, dict) and isinstance(function.get("name"), str):
                                    tool_definitions.append({"type": "function", **function})
                            for function in data.get("functions") or []:
                                if isinstance(function, dict) and isinstance(function.get("name"), str):
                                    tool_definitions.append({"type": "function", **function})
                            if tool_definitions:
                                definitions = json.dumps(tool_definitions, ensure_ascii=False)
                                if len(definitions) <= self.max_chars:
                                    log_attributes["gen_ai.tool.definitions"] = definitions
                            if response_too_large:
                                span.set_attribute(
                                    "vllm.response_capture.skipped", "body_too_large"
                                )
                            elif response_chunks:
                                try:
                                    outputs, response_model, usage = _assistant_output(
                                        path,
                                        b"".join(response_chunks),
                                        response_is_streaming,
                                    )
                                except (ValueError, UnicodeDecodeError):
                                    span.set_attribute(
                                        "vllm.response_capture.skipped", "invalid_response"
                                    )
                                else:
                                    if isinstance(response_model, str):
                                        span.set_attribute(
                                            AIAttributes.LLM_RESPONSE_MODEL, response_model
                                        )
                                        log_attributes[AIAttributes.LLM_RESPONSE_MODEL] = response_model
                                    if isinstance(usage, dict):
                                        for key, attribute in (
                                            ("prompt_tokens", AIAttributes.LLM_USAGE_PROMPT_TOKENS),
                                            ("completion_tokens", AIAttributes.LLM_USAGE_COMPLETION_TOKENS),
                                            ("total_tokens", AIAttributes.GEN_AI_USAGE_TOTAL_TOKENS),
                                        ):
                                            value = usage.get(key)
                                            if isinstance(value, int) and not isinstance(value, bool):
                                                span.set_attribute(attribute, value)
                                                log_attributes[attribute] = value
                                    if outputs:
                                        first = outputs[0]
                                        if first["text"]:
                                            log_attributes[AIAttributes.LLM_COMPLETIONS] = first[
                                                "text"
                                            ][: self.max_chars]
                                        log_attributes["vllm.response_capture.truncated"] = (
                                            len(first["text"]) > self.max_chars
                                        )
                                        output_messages = [
                                            {
                                                "role": "assistant",
                                                "parts": (
                                                    [{"type": "text", "content": output["text"]}]
                                                    if output["text"] else []
                                                ) + output["tool_calls"],
                                                **(
                                                    {"finish_reason": output["finish_reason"]}
                                                    if isinstance(output["finish_reason"], str)
                                                    else {}
                                                ),
                                            }
                                            for output in outputs
                                        ]
                                        serialized = json.dumps(
                                            output_messages, ensure_ascii=False
                                        )
                                        if len(serialized) <= self.max_chars:
                                            log_attributes["gen_ai.output.messages"] = serialized
                                            if self.span_content and any(
                                                output["tool_calls"] for output in outputs
                                            ):
                                                span.set_attribute(
                                                    "gen_ai.output.messages", serialized
                                                )
                                        all_calls = [
                                            call
                                            for output in outputs
                                            for call in output["tool_calls"]
                                        ]
                                        if all_calls:
                                            log_attributes["vllm.tool_call.count"] = len(all_calls)
                                            for call_index, call in enumerate(all_calls):
                                                call_attributes = {
                                                    "gen_ai.operation.name": operation,
                                                    "gen_ai.provider.name": "vllm",
                                                    "http.route": path,
                                                    "vllm.tool_call.index": call_index,
                                                    "gen_ai.tool.name": call["name"],
                                                }
                                                if isinstance(call.get("id"), str):
                                                    call_attributes["gen_ai.tool.call.id"] = call["id"]
                                                if "arguments" in call:
                                                    argument_text = json.dumps(
                                                        call["arguments"], ensure_ascii=False
                                                    )
                                                    call_attributes["gen_ai.tool.call.arguments"] = (
                                                        argument_text[: self.max_chars]
                                                    )
                                                    call_attributes[
                                                        "vllm.tool_call.arguments.truncated"
                                                    ] = len(argument_text) > self.max_chars
                                                if isinstance(response_model, str):
                                                    call_attributes[AIAttributes.LLM_RESPONSE_MODEL] = response_model
                                                self.logger.emit(
                                                    severity_number=SeverityNumber.INFO,
                                                    body="vllm.tool_call",
                                                    attributes=call_attributes,
                                                )
                                        if self.span_content:
                                            if first["text"]:
                                                span.set_attribute(
                                                    AIAttributes.LLM_COMPLETIONS,
                                                    first["text"][: self.max_chars],
                                                )
                            self.logger.emit(
                                severity_number=SeverityNumber.INFO,
                                body="vllm.llm_exchange",
                                attributes=log_attributes,
                            )
                            span.set_attribute(
                                "vllm.prompt_capture.truncated", len(text) > self.max_chars
                            )

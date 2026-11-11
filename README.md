# vLLM CPU OpenTelemetry test

Tested on 2026-11-11 with local vLLM `0.1.dev1+gf5f451f7d.cpu`,
`facebook/opt-125m`, and the OTLP HTTP receiver in `receiver.py`.
The raw received payloads are in `traces.jsonl`.

Run the commands below from the repository root with the vLLM CPU environment
activated. The model files are not included in this repository; download
`facebook/opt-125m` to `./opt-125m` or replace that path with another model.

## Start

In one terminal:

```sh
python receiver.py
```

In another terminal:

```sh
OTEL_SERVICE_NAME=vllm-server \
OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf \
vllm serve ./opt-125m \
  --served-model-name opt-125m-otel --host 127.0.0.1 --port 8000 \
  --dtype float32 --max-model-len 128 --enforce-eager \
  --kv-cache-memory-bytes 268435456 \
  --otlp-traces-endpoint http://127.0.0.1:4318/v1/traces
```

Send a traced request:

```sh
curl -sS http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -H 'traceparent: 00-11111111111111111111111111111111-2222222222222222-01' \
  -d '{"model":"opt-125m-otel","prompt":"Hello, my name is","max_tokens":8,"temperature":0.7,"top_p":0.9,"n":1}'
```

## Observed spans

The successful inference emitted one `llm_request` span with `SPAN_KIND_SERVER`,
the request's trace ID, and the supplied parent span ID. It included:

- `gen_ai.usage.prompt_tokens`: 6
- `gen_ai.usage.completion_tokens`: 8
- `gen_ai.latency.time_to_first_token`: 0.051686 seconds
- `gen_ai.latency.e2e`: 0.214160 seconds
- `gen_ai.latency.time_in_queue`: 0.00001996 seconds
- `gen_ai.latency.time_in_model_prefill`: 0.049603 seconds
- `gen_ai.latency.time_in_model_decode`: 0.162605 seconds
- `gen_ai.latency.time_in_model_inference`: 0.212208 seconds
- `gen_ai.request.id`, `gen_ai.request.max_tokens`, `gen_ai.request.top_p`,
  `gen_ai.request.temperature`, and `gen_ai.request.n`

Startup also emitted spans named `Worker init`, `Initialize model`,
`Load weights`, `Load model`, `Executor init`, `Prepare model`,
`EngineCoreProc init`, `AsyncMPClient init`, `Overall Loading`, and
`Allocate KV cache`. Those spans had code location attributes.

An instrumented client emitted its own `client-span` with a `prompt` attribute.
The vLLM `llm_request` span was its child. The prompt and generated text were
not attributes on the vLLM span in this test.

This endpoint received OTLP **traces**. vLLM's `/metrics` is a separate
Prometheus endpoint.

## Capture user prompts in vLLM

`prompt_middleware.py` emits sampled OTLP logs with the user prompt, AI reply,
and tool calls, plus a correlated GenAI span named `chat <model>` or
`text_completion <model>`.
It captures `prompt` from `/v1/completions` and text from `role=user` messages
in `/v1/chat/completions`. It collects text from non-streaming JSON responses
and reconstructs text from streaming SSE responses. During streaming, each
non-empty SSE text fragment is emitted promptly as an OTLP log named
`vllm.completion.chunk`, using the package's
`gen_ai.content.completion.chunk` attribute, a chunk index, and the same trace
context. A final `vllm.llm_exchange` log contains the assembled response.
It does not collect system messages or image data. The sampled span is the
parent of vLLM's `llm_request` span.
It uses the installed `opentelemetry-semantic-conventions-ai` constants for
legacy OpenLLMetry fields. The log carries `gen_ai.prompt` and optionally
`gen_ai.input.messages`, `gen_ai.completion`, and optionally
`gen_ai.output.messages` (JSON using the current GenAI message shapes), plus
model, operation, and token usage when available. For multiple choices,
`gen_ai.completion` contains the first choice and `gen_ai.output.messages`
contains every choice that fits within the content limit. The span carries
model, operation, and request parameters, but no content by default. It remains the parent of vLLM's `llm_request`
span, which carries token counts and latency. The OTLP log has the same trace
ID and the middleware span's span ID.

Start `receiver.py` on port 4319 for logs:

```sh
OTLP_PORT=4319 OTLP_OUTPUT=jaeger-test-logs.jsonl \
  python receiver.py
```

Then start vLLM with:

```sh
PYTHONPATH="$PWD" \
VLLM_PROMPT_OTLP_ENDPOINT=http://127.0.0.1:4318/v1/traces \
VLLM_PROMPT_OTLP_LOGS_ENDPOINT=http://127.0.0.1:4319/v1/logs \
VLLM_PROMPT_SAMPLE_RATE=0.1 \
VLLM_PROMPT_MAX_CHARS=2048 \
OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf \
OTEL_SERVICE_NAME=vllm-server \
vllm serve ./opt-125m \
  --served-model-name opt-125m-otel --host 127.0.0.1 --port 8000 \
  --dtype float32 --max-model-len 128 --enforce-eager \
  --kv-cache-memory-bytes 268435456 \
  --chat-template ./chat-template.jinja \
  --otlp-traces-endpoint http://127.0.0.1:4318/v1/traces \
  --middleware prompt_middleware.PromptCaptureMiddleware
```

`VLLM_PROMPT_SAMPLE_RATE` is between `0` and `1` (default `0.1`).
`VLLM_PROMPT_MAX_CHARS` defaults to 2048. Bodies larger than
`VLLM_PROMPT_MAX_BODY_BYTES` (default 1 MiB) are skipped. Unsampled requests
pass through unchanged. Failed requests do not export prompt content.
Responses larger than `VLLM_PROMPT_MAX_RESPONSE_BYTES` (default 1 MiB)
still produce a prompt log but omit the AI reply. Streaming token usage appears
only when vLLM includes it in the stream, for example with
`stream_options: {"include_usage": true}`.
The total emitted chunk text is capped by `VLLM_PROMPT_MAX_CHARS`. The log
batch export interval defaults to 200 ms and can be changed with
`VLLM_PROMPT_LOG_EXPORT_INTERVAL_MS`. A streaming response can therefore
produce multiple OTLP logs before its final log.
The middleware forwards request bodies without
rewriting them. Point `VLLM_PROMPT_OTLP_ENDPOINT` at the same collector as
vLLM's `--otlp-traces-endpoint`. Set `VLLM_PROMPT_OTLP_LOGS_ENDPOINT` to that
collector's log endpoint. For HTTP/protobuf, the log endpoint defaults to the
trace endpoint with `/v1/traces` replaced by `/v1/logs`; gRPC uses the same
endpoint for both signals. Both transports are supported. The local
`receiver.py` accepts `/v1/traces` and `/v1/logs`.

Earlier span-only tests are in `traces-middleware.jsonl` and
`traces-semconv.jsonl`. The current log-based version is in
`telemetry-logs.jsonl`. A local completion and chat test produced two prompt
logs whose trace ID and span ID matched the corresponding middleware spans.
Neither span carried prompt text, and the chat system message was absent from
the log. Instana's UI mapping has not been tested here.

The previous semantic-convention version was also tested with both endpoints.
Its completion and chat spans contained their respective user prompts, model
and request parameters; each was the parent of a native `llm_request` span.

## Verify in Jaeger

The [vLLM OpenTelemetry example](https://docs.vllm.ai/en/v0.9.0/examples/online_serving/opentelemetry.html)
uses Jaeger and a traced client request. Start the Jaeger 2.23.0 Docker
container defined in `compose.yaml`:

```sh
docker compose up -d jaeger
```

Jaeger receives OTLP HTTP traces at `http://127.0.0.1:4318/v1/traces`.
Open its UI at `http://127.0.0.1:16686`. To stop it, run
`docker compose down`. This all-in-one configuration stores traces in memory,
so they are cleared when the container stops.

Jaeger's default all-in-one configuration receives traces, but not OTLP
logs. For this test, run `receiver.py` with `OTLP_PORT=4319` for logs and set:

```sh
VLLM_PROMPT_OTLP_ENDPOINT=http://127.0.0.1:4318/v1/traces
VLLM_PROMPT_OTLP_LOGS_ENDPOINT=http://127.0.0.1:4319/v1/logs
VLLM_PROMPT_SPAN_CONTENT=1
```

`VLLM_PROMPT_SPAN_CONTENT=1` mirrors the sampled prompt to the middleware
span solely to inspect it in Jaeger. It also mirrors the sampled AI reply.
The default is `0`, which keeps content in OTLP logs only. Run
`python jaeger_client.py`
from this directory to create a real client parent span.

The Docker Jaeger test produced trace
[e907c818a3804870386161d785946af2](http://127.0.0.1:16686/trace/e907c818a3804870386161d785946af2).
Jaeger showed three spans: `client-span` → `chat opt-125m-otel` →
`llm_request`. The corresponding OTLP log is received separately by
`receiver.py` on port 4319. Jaeger labels the legacy `gen_ai.prompt` attribute
non-standard and points to the current `gen_ai.input.messages` convention.

AI reply capture was tested on 2026-11-11 for normal and streaming chat and
completion requests. All four `gen_ai.completion` log values matched the HTTP
response text. Earlier Jaeger GenAI View tests displayed the prompt and AI
reply in the same conversation with the span content mirror enabled. Those
earlier traces were stored in the removed native Jaeger process and are no
longer available in the current Docker container.

Incremental streaming was tested for chat and completions with 20 generated
text fragments each. Both produced 20 chunk logs; concatenating their chunk
attributes exactly matched the HTTP SSE text. The first chunk log was recorded
375.5 ms before the final log in each trace, and chunk logs arrived in earlier
OTLP export batches than the final log.

## Function and tool calls

For `/v1/chat/completions`, the middleware captures OpenAI-compatible
`tool_calls` and legacy `function_call` in ordinary JSON responses and SSE
streams. It assembles streamed call IDs, function names, and argument fragments
by choice and tool index. Each non-empty fragment emits a correlated
`vllm.tool_call.chunk` log; each assembled call emits a `vllm.tool_call` log
with `gen_ai.tool.name`, optional `gen_ai.tool.call.id`, and
`gen_ai.tool.call.arguments`. The final `vllm.llm_exchange` log includes
`vllm.tool_call.count` and, when it fits the content limit,
`gen_ai.output.messages` with `tool_call` parts. Request `tools` or `functions`
appear in `gen_ai.tool.definitions` when they fit. Earlier assistant tool calls
and `role=tool` or `role=function` results appear in `gen_ai.input.messages`.
Content remains subject to sampling and the configured size limits.

Run `python tool_call_probe.py` with the receiver and
Jaeger running to verify modern and legacy calls, both ordinary and streaming.
All four synthetic ASGI cases passed on 2026-11-11; their log payloads were
received at `127.0.0.1:4319` with the expected trace IDs. This verifies the
middleware against OpenAI-compatible response shapes. The local OPT-125M vLLM
server rejected a named tool choice with HTTP 400 because it was started
without `--tool-call-parser`, so model-generated tool calls have not been
verified on this model. The middleware records calls and tool results supplied
in requests; it does not execute tools.

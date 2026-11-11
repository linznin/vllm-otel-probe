"""Exercise OpenAI-compatible tool/function calls through the ASGI middleware."""

import asyncio
import json
import os

from prompt_middleware import PromptCaptureMiddleware, _assistant_output, _user_text


os.environ.setdefault("VLLM_PROMPT_SAMPLE_RATE", "1")
os.environ.setdefault("VLLM_PROMPT_OTLP_ENDPOINT", "http://127.0.0.1:4318/v1/traces")
os.environ.setdefault("VLLM_PROMPT_OTLP_LOGS_ENDPOINT", "http://127.0.0.1:4319/v1/logs")
os.environ.setdefault("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "http/protobuf")
os.environ.setdefault("OTEL_EXPORTER_OTLP_LOGS_PROTOCOL", "http/protobuf")


def sse(events):
    return b"".join(
        b"data: " + json.dumps(event).encode() + b"\n\n" for event in events
    ) + b"data: [DONE]\n\n"


request = {
    "model": "opt-125m-otel",
    "messages": [{"role": "user", "content": "Find the weather in Taipei"}],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Return weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ],
}
modern_call = {
    "id": "call_weather_1",
    "type": "function",
    "function": {"name": "get_weather", "arguments": '{"city":"Taipei"}'},
}
normal_modern = {
    "model": "opt-125m-otel",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": None, "tool_calls": [modern_call]},
            "finish_reason": "tool_calls",
        }
    ],
}
normal_legacy = {
    "model": "opt-125m-otel",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "function_call": {"name": "get_weather", "arguments": '{"city":"Taipei"}'},
            },
            "finish_reason": "function_call",
        }
    ],
}
stream_modern = sse(
    [
        {"model": "opt-125m-otel", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_weather_2", "type": "function", "function": {"name": "get_", "arguments": ""}}]}}]},
        {"model": "opt-125m-otel", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"name": "weather", "arguments": '{"city":'}}]}}]},
        {"model": "opt-125m-otel", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"Taipei"}'}}]}, "finish_reason": "tool_calls"}]},
    ]
)
stream_legacy = sse(
    [
        {"model": "opt-125m-otel", "choices": [{"index": 0, "delta": {"function_call": {"name": "get_", "arguments": '{"city":'}}}]},
        {"model": "opt-125m-otel", "choices": [{"index": 0, "delta": {"function_call": {"name": "weather", "arguments": '"Taipei"}'}}, "finish_reason": "function_call"}]},
    ]
)

cases = [
    ("01010101010101010101010101010101", json.dumps(normal_modern).encode(), False),
    ("02020202020202020202020202020202", stream_modern, True),
    ("03030303030303030303030303030303", json.dumps(normal_legacy).encode(), False),
    ("04040404040404040404040404040404", stream_legacy, True),
]


async def main():
    state = {}

    async def app(scope, receive, send):
        await receive()
        body = state["body"]
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (
                        b"content-type",
                        b"text/event-stream" if state["stream"] else b"application/json",
                    )
                ],
            }
        )
        chunks = [body[i : i + 17] for i in range(0, len(body), 17)]
        for index, chunk in enumerate(chunks):
            await send(
                {
                    "type": "http.response.body",
                    "body": chunk,
                    "more_body": index < len(chunks) - 1,
                }
            )

    middleware = PromptCaptureMiddleware(app)
    for trace_id, body, streaming in cases:
        state.update(body=body, stream=streaming)
        sent = []

        async def receive():
            return {"type": "http.request", "body": json.dumps(request).encode(), "more_body": False}

        async def send(message):
            sent.append(message)

        await middleware(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/chat/completions",
                "headers": [(b"traceparent", f"00-{trace_id}-1111111111111111-01".encode())],
            },
            receive,
            send,
        )
        assert b"".join(message.get("body", b"") for message in sent) == body
        print(f"trace_id={trace_id} streaming={streaming}")

    _, input_messages = _user_text(
        "/v1/chat/completions",
        {
            "messages": [
                *request["messages"],
                {"role": "assistant", "content": None, "tool_calls": [modern_call]},
                {"role": "tool", "tool_call_id": "call_weather_1", "content": '{"temp_c":25}'},
            ]
        },
    )
    parts = [part for message in json.loads(input_messages) for part in message["parts"]]
    assert any(part["type"] == "tool_call" for part in parts)
    assert any(part["type"] == "tool_call_response" for part in parts)
    multi = sse(
        [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 1, "id": "call_b", "function": {"name": "second", "arguments": "{}"}},
                                {"index": 0, "id": "call_a", "function": {"name": "first", "arguments": "{}"}},
                            ]
                        },
                    }
                ]
            }
        ]
    )
    outputs, _, _ = _assistant_output("/v1/chat/completions", multi, True)
    assert [call["name"] for call in outputs[0]["tool_calls"]] == ["first", "second"]
    middleware.log_provider.force_flush()
    middleware.provider.force_flush()


asyncio.run(main())

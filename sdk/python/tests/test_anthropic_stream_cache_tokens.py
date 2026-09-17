# pylint: disable=protected-access
"""
Regression tests: cache token counts read from an Anthropic stream's
`message_start` must survive the closing `message_delta`.

`process_chunk` in `openlit.instrumentation.anthropic.utils` reads
`cache_creation_input_tokens` / `cache_read_input_tokens` from the
`message_start` usage block, and then re-assigned both unconditionally on
`message_delta` with a `0` default -- despite the comment there saying it
updates them "when present". A `message_delta` reports `output_tokens`; its
cache fields are optional, and the SDK's `MessageDeltaUsage` model dumps them
as `None` when the API omits them, so `usage.get(key, 0) or 0` reset both
counters to zero at the end of every cached stream.

Anthropic reports `input_tokens` exclusive of cached tokens, so the whole
cached prompt then dropped out of the cost, and the
`gen_ai.usage.cache_read.input_tokens` /
`gen_ai.usage.cache_creation.input_tokens` span attributes landed as 0.

These tests drive the real `process_chunk` / `process_streaming_chat_response`
-- the same functions the `anthropic.py` and `async_anthropic.py` wrapper
classes call -- with the chunk payloads the API sends.
"""

import time
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from openlit._config import OpenlitConfig
from openlit.instrumentation.anthropic import utils as anthropic_utils
from openlit.semcov import SemanticConvention

MODEL = "claude-3-5-sonnet-latest"

REQUEST_KWARGS = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "summarize the cached document"}],
}

PRICING_WITH_CACHE = {
    "chat": {
        MODEL: {
            "promptPrice": 0.003,
            "completionPrice": 0.015,
            "cacheReadPrice": 0.0003,
            "cacheCreationPrice": 0.00375,
        }
    }
}

INPUT_TOKENS = 12
OUTPUT_TOKENS = 15
CACHE_CREATION_TOKENS = 1024
CACHE_READ_TOKENS = 2048

MESSAGE_START = {
    "type": "message_start",
    "message": {
        "id": "msg_01",
        "model": MODEL,
        "role": "assistant",
        "usage": {
            "input_tokens": INPUT_TOKENS,
            "cache_creation_input_tokens": CACHE_CREATION_TOKENS,
            "cache_read_input_tokens": CACHE_READ_TOKENS,
        },
    },
}

CONTENT_CHUNKS = [
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "the cached document says"},
    },
    {"type": "content_block_stop", "index": 0},
]

# What the SSE payload carries: usage holds output_tokens only.
MESSAGE_DELTA_OUTPUT_ONLY = {
    "type": "message_delta",
    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
    "usage": {"output_tokens": OUTPUT_TOKENS},
}

# What `response_as_dict` sees after the SDK parses that same payload into
# `RawMessageDeltaEvent`: the optional cache fields are present, and null.
MESSAGE_DELTA_NULL_CACHE = {
    "type": "message_delta",
    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
    "usage": {
        "cache_creation_input_tokens": None,
        "cache_read_input_tokens": None,
        "input_tokens": None,
        "output_tokens": OUTPUT_TOKENS,
    },
}

# A message_delta that does carry final cache counts must still update them.
MESSAGE_DELTA_WITH_CACHE = {
    "type": "message_delta",
    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
    "usage": {
        "cache_creation_input_tokens": 64,
        "cache_read_input_tokens": 128,
        "output_tokens": OUTPUT_TOKENS,
    },
}

MESSAGE_STOP = {"type": "message_stop"}


def _tracer_with_exporter():
    """Fresh tracer writing finished spans into an in-memory exporter."""
    OpenlitConfig.reset_to_defaults()
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer(__name__), exporter


def _stream_scope(span=None):
    """Mirrors TracedSyncStream.__init__'s scope state in anthropic.py."""
    return SimpleNamespace(
        _span=span,
        _llmresponse="",
        _response_id="",
        _response_model="",
        _finish_reason="",
        _input_tokens=0,
        _output_tokens=0,
        _cache_read_input_tokens=0,
        _cache_creation_input_tokens=0,
        _tool_calls_by_index={},
        _tool_calls=None,
        _response_role="",
        _kwargs=REQUEST_KWARGS,
        _start_time=time.time(),
        _end_time=None,
        _timestamps=[],
        _ttft=0,
        _tbt=0,
        _server_address="api.anthropic.com",
        _server_port=443,
    )


def _consume(chunks, span=None):
    """Feed a whole stream through the real chunk processor."""
    scope = _stream_scope(span)
    for chunk in chunks:
        anthropic_utils.process_chunk(scope, chunk)
    return scope


def _stream(message_delta):
    return [MESSAGE_START, *CONTENT_CHUNKS, message_delta, MESSAGE_STOP]


@pytest.mark.parametrize(
    "message_delta",
    [MESSAGE_DELTA_OUTPUT_ONLY, MESSAGE_DELTA_NULL_CACHE],
    ids=["usage_without_cache_fields", "usage_with_null_cache_fields"],
)
def test_message_delta_keeps_cache_counts_from_message_start(message_delta):
    """A message_delta that reports no cache usage must not zero the counts."""
    scope = _consume(_stream(message_delta))

    assert scope._cache_creation_input_tokens == CACHE_CREATION_TOKENS
    assert scope._cache_read_input_tokens == CACHE_READ_TOKENS
    assert scope._output_tokens == OUTPUT_TOKENS
    assert scope._input_tokens == INPUT_TOKENS


def test_message_delta_with_cache_usage_updates_cache_counts():
    """A message_delta that does report cache usage must still update it."""
    scope = _consume(_stream(MESSAGE_DELTA_WITH_CACHE))

    assert scope._cache_creation_input_tokens == 64
    assert scope._cache_read_input_tokens == 128


def test_uncached_stream_still_reports_zero_cache_counts():
    """A stream with no cache usage anywhere must stay at zero."""
    message_start = {
        "type": "message_start",
        "message": {
            "id": "msg_02",
            "model": MODEL,
            "role": "assistant",
            "usage": {"input_tokens": INPUT_TOKENS},
        },
    }
    scope = _consume([message_start, *CONTENT_CHUNKS, MESSAGE_DELTA_OUTPUT_ONLY])

    assert scope._cache_creation_input_tokens == 0
    assert scope._cache_read_input_tokens == 0


def test_streaming_span_reports_cache_tokens_and_cache_cost():
    """The finished span must carry the cache tokens, and bill them."""
    tracer, exporter = _tracer_with_exporter()
    with tracer.start_as_current_span("anthropic.chat") as span:
        scope = _consume(_stream(MESSAGE_DELTA_OUTPUT_ONLY), span)
        anthropic_utils.process_streaming_chat_response(
            scope,
            pricing_info=PRICING_WITH_CACHE,
            environment="test",
            application_name="test",
            metrics=None,
            capture_message_content=True,
            disable_metrics=True,
            version="1.0.0",
        )

    attrs = exporter.get_finished_spans()[0].attributes

    assert (
        attrs[SemanticConvention.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS]
        == CACHE_CREATION_TOKENS
    )
    assert (
        attrs[SemanticConvention.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS]
        == CACHE_READ_TOKENS
    )

    # Anthropic reports input_tokens exclusive of cache tokens, so the cached
    # prompt is billed on top of it.
    expected_cost = (
        (INPUT_TOKENS / 1000) * 0.003
        + (OUTPUT_TOKENS / 1000) * 0.015
        + (CACHE_READ_TOKENS / 1000) * 0.0003
        + (CACHE_CREATION_TOKENS / 1000) * 0.00375
    )
    assert attrs[SemanticConvention.GEN_AI_USAGE_COST] == pytest.approx(
        expected_cost, rel=1e-9
    )

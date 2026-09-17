# pylint: disable=missing-function-docstring
"""Regression tests: the Prem AI streaming wrapper must be a real iterator.

`TracedSyncStream` had no `__next__`, and its `__iter__` pulled a single
chunk, processed it and returned that chunk instead of an iterator. So
`for chunk in client.chat.completions.create(..., stream=True)` raised
`TypeError: iter() returned non-iterator of type ...` and the span was
finalized after the first chunk, losing every later chunk's content, the
response id and the finish reason. Every sibling provider (openai, groq,
mistral, anthropic) returns `self` from `__iter__` and finalizes the span
from `__next__` on `StopIteration`.

These tests drive the real `chat` wrapper factory with a synthetic two-chunk
Prem-shaped stream and assert that iteration yields every chunk and that
exhaustion exports exactly one span carrying both chunks' data.
"""

import json
import logging

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from openlit._config import OpenlitConfig
from openlit.instrumentation.premai import premai as premai_mod
from openlit.semcov import SemanticConvention

REQUEST_KWARGS = {
    "model": "gpt-4o-mini",
    "stream": True,
    "messages": [{"role": "user", "content": "Monitor LLM Applications"}],
}

# Two Prem-shaped chunks: a content delta, then a final chunk carrying the
# response id, the finish reason and usage.
CHUNKS = [
    {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
    {
        "id": "cmpl-1",
        "model": "gpt-4o-mini",
        "choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    },
]


def _tracer_and_exporter():
    OpenlitConfig.reset_to_defaults()
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test-premai-stream-iteration"), exporter


def _factory(tracer):
    return premai_mod.chat(
        version="test",
        environment="test",
        application_name="test",
        tracer=tracer,
        pricing_info={},
        capture_message_content=True,
        metrics=None,
        disable_metrics=True,
        event_provider=None,
    )


class FakeSyncStream:
    """Minimal stand-in for Prem's streaming response: an iterator only."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __iter__(self):
        return self

    def __next__(self):
        if not self._chunks:
            raise StopIteration
        return self._chunks.pop(0)


def _open_stream(tracer):
    wrapper = _factory(tracer)
    return wrapper(lambda *a, **k: FakeSyncStream(CHUNKS), None, (), REQUEST_KWARGS)


def _only_span(exporter):
    spans = exporter.get_finished_spans()
    assert len(spans) == 1, f"expected exactly one exported span, got {len(spans)}"
    return spans[0]


def test_iteration_yields_every_chunk():
    tracer, _ = _tracer_and_exporter()

    # A for-loop over the wrapper is the documented Prem streaming call; it
    # raised TypeError because __iter__ returned a chunk, not an iterator.
    assert list(_open_stream(tracer)) == CHUNKS


def test_exhaustion_exports_exactly_one_span_with_every_chunk():
    tracer, exporter = _tracer_and_exporter()

    stream = _open_stream(tracer)
    for _ in stream:
        pass

    attrs = _only_span(exporter).attributes
    # The response id and the finish reason only ever arrive on the last
    # chunk, so a span finalized after the first one cannot carry them.
    assert attrs[SemanticConvention.GEN_AI_RESPONSE_ID] == "cmpl-1"
    assert list(attrs[SemanticConvention.GEN_AI_RESPONSE_FINISH_REASON]) == ["stop"]
    # Both chunks' deltas, concatenated: the first chunk alone gives "Hello".
    output_messages = json.loads(attrs[SemanticConvention.GEN_AI_OUTPUT_MESSAGES])
    assert output_messages[0]["parts"][0]["content"] == "Hello world"


def test_iterating_past_exhaustion_does_not_finalize_twice(caplog):
    tracer, exporter = _tracer_and_exporter()

    stream = _open_stream(tracer)
    assert list(stream) == CHUNKS

    # A second pass raises StopIteration again. Finalizing a second time
    # writes the whole attribute set to the span the first pass already
    # ended; the SDK drops those writes and warns for every one of them.
    with caplog.at_level(logging.WARNING, logger="opentelemetry.sdk.trace"):
        assert not list(stream)

    assert not [
        record.getMessage()
        for record in caplog.records
        if "ended span" in record.getMessage()
    ]
    _only_span(exporter)

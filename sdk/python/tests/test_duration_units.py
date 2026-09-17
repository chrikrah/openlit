"""
Unit-consistency tests for the ``gen_ai.client.operation.duration`` attribute.

``gen_ai.client.operation.duration`` is declared once, in
``openlit/otel/metrics.py``, as a histogram with ``unit="s"`` and advisory
bucket boundaries that stop at 81.92. Every producer of that name therefore has
to emit seconds. The browser_use and crawl4ai instrumentations multiplied the
elapsed time by 1000 and published milliseconds under the same name, so a 2.5
second crawl was reported as 2500 and every real operation fell past the last
bucket boundary.

These tests drive the real wrapper code paths and stub only the third-party
object being wrapped, so neither ``browser_use`` nor ``crawl4ai`` has to be
installed for them to run.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from openlit._config import OpenlitConfig
from openlit.instrumentation.browser_use import async_browser_use, browser_use
from openlit.instrumentation.crawl4ai import async_crawl4ai, crawl4ai

DURATION_KEY = "gen_ai.client.operation.duration"
ELAPSED_SECONDS = 2.5
CLOCK_START = 1_700_000_000.0
ENVIRONMENT = "openlit-python-testing"
APPLICATION = "openlit-python-duration-units-test"


@pytest.fixture(autouse=True)
def _reset_openlit_config():
    """Initialize OpenlitConfig class attributes without a full openlit.init()."""
    OpenlitConfig.reset_to_defaults()
    yield


def _fixed_clock():
    """Build a time.time() stand-in: the first reading is the operation start, every later reading is ELAPSED_SECONDS after it."""

    state = {"calls": 0}

    def clock():
        """Return the controlled wall-clock reading."""
        state["calls"] += 1
        if state["calls"] == 1:
            return CLOCK_START
        return CLOCK_START + ELAPSED_SECONDS

    return clock


def _tracer():
    """Build an isolated tracer backed by an in-memory span exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("duration-units-test"), exporter


def _assert_seconds(recorded):
    """Assert the recorded duration is the elapsed time in seconds."""
    assert recorded is not None, f"{DURATION_KEY} was never recorded"
    assert recorded == pytest.approx(ELAPSED_SECONDS), (
        f"{DURATION_KEY} is declared unit='s' in openlit/otel/metrics.py, so an "
        f"operation that took {ELAPSED_SECONDS}s must record {ELAPSED_SECONDS}, "
        f"not {recorded}"
    )


def _span_duration(exporter):
    """Return the duration attribute of the single span the exporter captured."""
    spans = exporter.get_finished_spans()
    assert len(spans) == 1, f"expected exactly one span, got {len(spans)}"
    return spans[0].attributes.get(DURATION_KEY)


def _stub_instance():
    """Build a stand-in for the wrapped browser_use / crawl4ai object."""
    return SimpleNamespace(task="stub task", url="https://example.com")


def _sync_result(*args, **kwargs):
    """Stand in for a successful wrapped sync call."""
    return "stub result"


def _sync_failure(*args, **kwargs):
    """Stand in for a wrapped sync call that raises."""
    raise ValueError("stub failure")


async def _async_result(*args, **kwargs):
    """Stand in for a successful wrapped async call."""
    return "stub result"


async def _async_failure(*args, **kwargs):
    """Stand in for a wrapped async call that raises."""
    raise ValueError("stub failure")


def _drive_sync(module, endpoint, wrapped, expect_failure=False):
    """Run a sync wrapper over a stub call and return the duration it put on the span."""
    tracer, exporter = _tracer()
    wrapper = module.general_wrap(
        endpoint,
        "1.0.0",
        ENVIRONMENT,
        APPLICATION,
        tracer,
        {},
        False,
        {},
        True,
    )

    with patch.object(module.time, "time", _fixed_clock()):
        if expect_failure:
            with pytest.raises(ValueError):
                wrapper(wrapped, _stub_instance(), (), {})
        else:
            wrapper(wrapped, _stub_instance(), (), {})

    return _span_duration(exporter)


def _drive_async(module, endpoint, wrapped, expect_failure=False):
    """Run an async wrapper over a stub call and return the duration it put on the span."""
    tracer, exporter = _tracer()
    wrapper = module.async_general_wrap(
        endpoint,
        "1.0.0",
        ENVIRONMENT,
        APPLICATION,
        tracer,
        {},
        False,
        {},
        True,
    )

    async def call():
        """Await the wrapped stub call."""
        return await wrapper(wrapped, _stub_instance(), (), {})

    with patch.object(module.time, "time", _fixed_clock()):
        if expect_failure:
            with pytest.raises(ValueError):
                asyncio.run(call())
        else:
            asyncio.run(call())

    return _span_duration(exporter)


class TestBrowserUseDurationUnits:
    """browser_use records gen_ai.client.operation.duration in seconds."""

    def test_sync_success_records_seconds(self):
        """A completed sync browser_use operation records seconds."""
        _assert_seconds(_drive_sync(browser_use, "agent.pause", _sync_result))

    def test_sync_error_records_seconds(self):
        """A failed sync browser_use operation records seconds."""
        _assert_seconds(
            _drive_sync(browser_use, "agent.pause", _sync_failure, expect_failure=True)
        )

    def test_async_success_records_seconds(self):
        """A completed async browser_use operation records seconds."""
        _assert_seconds(_drive_async(async_browser_use, "agent.run", _async_result))

    def test_async_error_records_seconds(self):
        """A failed async browser_use operation records seconds."""
        _assert_seconds(
            _drive_async(
                async_browser_use, "agent.run", _async_failure, expect_failure=True
            )
        )


class TestCrawl4AIDurationUnits:
    """crawl4ai records gen_ai.client.operation.duration in seconds."""

    def test_sync_success_records_seconds(self):
        """A completed sync crawl4ai operation records seconds."""
        _assert_seconds(_drive_sync(crawl4ai, "crawl4ai.run", _sync_result))

    def test_sync_error_records_seconds(self):
        """A failed sync crawl4ai operation records seconds."""
        _assert_seconds(
            _drive_sync(crawl4ai, "crawl4ai.run", _sync_failure, expect_failure=True)
        )

    def test_async_success_records_seconds(self):
        """A completed async crawl4ai operation records seconds."""
        _assert_seconds(_drive_async(async_crawl4ai, "crawl4ai.arun", _async_result))

    def test_async_error_records_seconds(self):
        """A failed async crawl4ai operation records seconds."""
        _assert_seconds(
            _drive_async(
                async_crawl4ai, "crawl4ai.arun", _async_failure, expect_failure=True
            )
        )

    def test_async_streaming_records_seconds(self):
        """A finished async crawl4ai stream records seconds.

        The streaming finalizer runs after the span has already been closed, so
        its span attribute is dropped by the SDK. The same value is appended to
        the metrics collection, and that is what this asserts on.
        """
        tracer, _ = _tracer()
        # The finalizer only records into a non-empty collection.
        collected = {"crawl4ai.crawl.stream.completed": 0}
        wrapper = async_crawl4ai.async_general_wrap(
            "crawl4ai.arun",
            "1.0.0",
            ENVIRONMENT,
            APPLICATION,
            tracer,
            {},
            False,
            collected,
            False,
        )

        async def stream(*args, **kwargs):
            """Stand in for a wrapped async call that streams crawl results."""

            async def results():
                """Yield one stub crawl result."""
                yield SimpleNamespace(url="https://example.com", success=True)

            return results()

        async def consume():
            """Drive the wrapper and exhaust the stream it returns."""
            tracked = await wrapper(stream, _stub_instance(), (), {})
            async for _ in tracked:
                pass

        with patch.object(async_crawl4ai.time, "time", _fixed_clock()):
            asyncio.run(consume())

        durations = collected.get("crawl4ai.crawl.stream.duration")
        assert durations, "the stream finalizer recorded no duration"
        _assert_seconds(durations[0])

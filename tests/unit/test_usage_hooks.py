"""Model spend outside a turn reaches a registered listener."""

import pytest

from memorizz.learning.consolidation import text_generator
from memorizz.llms.llm_provider import ResponseMetadataMixin
from memorizz.observability.usage_hooks import (
    normalize_provider_usage,
    register_usage_listener,
    report_auxiliary_usage,
)


@pytest.fixture()
def events():
    seen = []
    stop = register_usage_listener(seen.append)
    yield seen
    stop()


class _Model(ResponseMetadataMixin):
    """A provider as the built-in ones behave: it clears its usage, calls,
    then sets what the call reported."""

    model = "gpt-6-luna"

    def __init__(self, usage):
        self._reports = usage

    def get_config(self):
        return {"provider": "openai", "model": self.model}

    def get_last_usage(self):
        return self._last_usage

    def generate_text(self, prompt, instructions=None):
        self._last_usage = None
        self._last_usage = self._reports
        return "done"


@pytest.mark.unit
def test_nothing_is_reported_without_a_listener():
    assert (
        report_auxiliary_usage(
            kind="embedding",
            provider="openai",
            model="text-embedding-3-small",
            input_tokens=1000,
        )
        is None
    )


@pytest.mark.unit
def test_an_embedding_call_is_priced_at_its_list_rate(events):
    report_auxiliary_usage(
        kind="embedding",
        provider="openai",
        model="text-embedding-3-small",
        input_tokens=1_000_000,
    )

    (event,) = events
    assert event["kind"] == "embedding" and event["input_tokens"] == 1_000_000
    assert event["cost_usd"] == pytest.approx(0.02)
    assert event["unpriced"] is False


@pytest.mark.unit
def test_a_model_with_no_rate_is_reported_unpriced(events):
    report_auxiliary_usage(
        kind="background", provider="ollama", model="qwen2.5:7b", input_tokens=40
    )

    (event,) = events
    assert event["cost_usd"] is None and event["unpriced"] is True


@pytest.mark.unit
def test_a_call_that_used_nothing_is_not_reported(events):
    report_auxiliary_usage(kind="background", provider="openai", model="gpt-6-luna")

    assert events == []


@pytest.mark.unit
def test_a_background_text_call_reports_what_that_call_used(events):
    model = _Model({"prompt_tokens": 120, "completion_tokens": 30})

    assert text_generator(model)("File what was learned.") == "done"

    (event,) = events
    assert event["kind"] == "background" and event["model"] == "gpt-6-luna"
    assert (event["input_tokens"], event["output_tokens"]) == (120, 30)
    assert event["cost_usd"] is not None


@pytest.mark.unit
def test_a_background_call_that_reported_no_usage_is_not_given_an_earlier_one(events):
    """The model object still holds a chat turn's usage; the background call
    reported none, and must not be charged the chat turn's."""
    model = _Model(None)
    model._last_usage = {"prompt_tokens": 9000, "completion_tokens": 900}

    text_generator(model)("File what was learned.")

    assert events == []


@pytest.mark.unit
def test_usage_keys_of_each_provider_are_read():
    assert normalize_provider_usage({"prompt_tokens": 5, "completion_tokens": 2}) == {
        "input_tokens": 5,
        "output_tokens": 2,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
    }
    anthropic = normalize_provider_usage(
        {
            "input_tokens": 7,
            "output_tokens": 3,
            "cache_read_input_tokens": 4,
            "cache_creation_input_tokens": 1,
        }
    )
    assert anthropic["cached_tokens"] == 4 and anthropic["cache_write_tokens"] == 1
    assert normalize_provider_usage(None)["input_tokens"] == 0

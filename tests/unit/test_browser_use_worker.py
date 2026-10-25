"""Browser Use worker: defaults, efficiency settings and result shape.

These run the real generated worker program in a subprocess against a small
stand-in ``browser_use`` package, so the worker's own code path (LLM setup,
browser policy, agent options, usage and timing report) is what is tested.
"""

from __future__ import annotations

import json
import sys

import pytest

from memorizz.browser_control.providers.browser_use import BrowserUseProvider
from memorizz.cli import agent_factory

# Mirrors Browser Use 0.13.7 where it matters: the chat wrappers have no
# default model, Agent takes ``use_judge`` and ``bool | Literal['auto']``
# vision, and the run history reports token usage.
_FAKE_BROWSER_USE = '''
import json
import os


def _record(kind, **values):
    with open(os.environ["FAKE_BU_LOG"], "a", encoding="utf-8") as log:
        log.write(json.dumps({"kind": kind, **values}) + "\\n")


class _Chat:
    def __init__(self, model):
        if model == "missing-model":
            raise TypeError(
                "ChatOpenAI.__init__() missing 1 required positional argument: 'model'"
            )
        self.model = model
        self.reasoning_models = ["o3", "gpt-5"]
        _record(type(self).__name__, model=model)


class ChatOpenAI(_Chat):
    pass


class ChatAnthropic(_Chat):
    pass


class ChatGoogle(_Chat):
    pass


class ChatBrowserUse(_Chat):
    def __init__(self, model="bu-2-0"):
        super().__init__(model)


class BrowserProfile:
    def __init__(
        self,
        headless=True,
        block_ip_addresses=False,
        keep_alive=None,
        allowed_domains=None,
        prohibited_domains=None,
        use_cloud=False,
        cdp_url=None,
    ):
        _record("BrowserProfile", headless=headless)


class Browser:
    def __init__(self, browser_profile=None):
        pass

    async def start(self):
        _record("start")

    async def stop(self):
        _record("stop")


class _Usage:
    total_prompt_tokens = 14000
    total_prompt_cached_tokens = 4000
    total_prompt_cache_creation_tokens = 0
    total_completion_tokens = 200
    entry_count = 1


class _History:
    usage = _Usage()

    def is_successful(self):
        return True

    def is_done(self):
        return True

    def has_errors(self):
        return False

    def final_result(self):
        return os.environ.get("FAKE_BU_OUTPUT", "Example Domain")

    def urls(self):
        return ["https://example.com/", None, "https://example.com/"]

    def action_names(self):
        return ["navigate", "done"]

    def errors(self):
        return [None] + [f"step {n} failed: " + "x" * 3000 for n in range(7)]

    def number_of_steps(self):
        return 2

    def total_duration_seconds(self):
        return 1.5


class Agent:
    def __init__(
        self,
        task,
        llm=None,
        browser=None,
        use_vision: "bool | Literal['auto']" = True,
        use_judge=True,
        **kwargs,
    ):
        _record(
            "Agent",
            use_vision=use_vision,
            use_judge=use_judge,
            reasoning_models=llm.reasoning_models,
            telemetry=os.environ.get("ANONYMIZED_TELEMETRY"),
        )

    async def run(self, max_steps=100):
        return _History()


class _OldAgent:
    """A release from before screenshot-on-request vision and the judge."""

    def __init__(self, task, llm=None, browser=None, use_vision: bool = True):
        _record("Agent", use_vision=use_vision, use_judge="not-passed")

    async def run(self, max_steps=100):
        return _History()


if os.environ.get("FAKE_BU_OLD_AGENT"):
    Agent = _OldAgent
'''

_FAKE_METADATA = "Metadata-Version: 2.1\nName: browser-use\nVersion: 0.13.7\n"


@pytest.fixture
def fake_browser_use(tmp_path, monkeypatch):
    """Point a provider at the stand-in package; return (factory, read_log)."""
    site = tmp_path / "site"
    package = site / "browser_use"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(_FAKE_BROWSER_USE, encoding="utf-8")
    dist_info = site / "browser_use-0.13.7.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(_FAKE_METADATA, encoding="utf-8")
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("PYTHONPATH", str(site))
    monkeypatch.setenv("FAKE_BU_LOG", str(log))
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only")
    monkeypatch.delenv("ANONYMIZED_TELEMETRY", raising=False)

    def provider(**config):
        return BrowserUseProvider(
            command=sys.executable,
            python_command=sys.executable,
            env_allowlist=(
                "PATH",
                "PYTHONPATH",
                "FAKE_BU_LOG",
                "FAKE_BU_OUTPUT",
                "FAKE_BU_OLD_AGENT",
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "ANONYMIZED_TELEMETRY",
            ),
            **config,
        )

    def calls():
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text().splitlines()]

    return provider, calls


def _call(calls, kind):
    return next(item for item in calls() if item["kind"] == kind)


@pytest.mark.unit
def test_default_config_runs_with_a_fast_model_no_judge_and_auto_vision(
    fake_browser_use,
):
    provider, calls = fake_browser_use

    result = provider().run_task("Open https://example.com and read the heading")

    assert result.success is True, result.errors
    assert result.output == "Example Domain"
    # Browser Use 0.13 has no default OpenAI model; a missing one used to
    # crash the worker before it could report anything.
    assert _call(calls, "ChatOpenAI")["model"] == "gpt-5.4-mini"
    agent = _call(calls, "Agent")
    assert agent["use_judge"] is False
    assert agent["use_vision"] == "auto"
    assert agent["telemetry"] == "false"
    assert [item["kind"] for item in calls()].count("stop") == 1
    assert result.metadata["model"] == "gpt-5.4-mini"
    assert result.metadata["use_vision"] == "auto"
    assert provider().get_config()["model"] is None


@pytest.mark.unit
def test_result_reports_phases_usage_and_cost_and_stays_compact(
    fake_browser_use, monkeypatch
):
    provider, _calls = fake_browser_use
    monkeypatch.setenv("FAKE_BU_OUTPUT", "y" * 20_000)

    result = provider(max_output_chars=1_000).run_task("Read the page")

    timings = result.metadata["timings"]
    assert {
        "import_seconds",
        "browser_start_seconds",
        "startup_seconds",
        "run_seconds",
        "browser_stop_seconds",
    } <= set(timings)
    assert timings["steps_seconds"] == 1.5
    # The whole call, not only the browser agent's steps.
    assert result.duration_seconds >= timings["startup_seconds"]
    usage = result.metadata["usage"]
    assert usage["prompt_tokens"] == 14000
    assert usage["completion_tokens"] == 200
    assert usage["llm_calls"] == 1
    # gpt-5.4-mini list rates: 10k fresh input, 4k cached input, 200 output.
    assert usage["cost_usd"] == pytest.approx(0.0087)
    assert len(result.output) == 1_000
    assert result.metadata["output_truncated"] is True
    assert result.urls == ["https://example.com/"]
    assert len(result.errors) == 5
    assert all(len(error) <= 1_001 for error in result.errors)
    assert result.errors[-1].startswith("step 6 failed")
    assert result.metadata["errors_omitted"] == 2


@pytest.mark.unit
def test_anthropic_usage_counts_cache_writes_as_input_and_prices_them():
    usage = {
        "prompt_tokens": 22,
        "cached_tokens": 0,
        "cache_write_tokens": 22_439,
        "completion_tokens": 427,
        "llm_calls": 1,
    }

    priced = BrowserUseProvider(llm_provider="anthropic")._usage(usage)
    unpriced = BrowserUseProvider(model="house-model")._usage(usage)

    # Browser Use reports 22 here; the cache writes are input tokens too.
    assert priced["prompt_tokens"] == 22_461
    # claude-haiku-5-5: 22 fresh, 22,439 cache-write and 427 output tokens.
    assert priced["cost_usd"] == pytest.approx(0.003021)
    assert "cost_usd" not in unpriced
    assert unpriced["prompt_tokens"] == 22


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model", "registered"),
    [("gpt-6-luna", True), ("gpt-5.4-mini", False), ("gpt-4.1-mini", False)],
)
def test_newer_openai_reasoning_models_skip_the_temperature_they_reject(
    fake_browser_use, model, registered
):
    provider, calls = fake_browser_use

    provider(model=model).run_task("Read the page")

    assert (model in _call(calls, "Agent")["reasoning_models"]) is registered


@pytest.mark.unit
def test_operator_can_opt_into_browser_use_telemetry(fake_browser_use, monkeypatch):
    provider, calls = fake_browser_use
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "true")

    provider().run_task("Read the page")

    assert _call(calls, "Agent")["telemetry"] == "true"


@pytest.mark.unit
def test_older_browser_use_gets_boolean_vision_and_no_judge_argument(
    fake_browser_use, monkeypatch
):
    provider, calls = fake_browser_use
    monkeypatch.setenv("FAKE_BU_OLD_AGENT", "1")

    result = provider().run_task("Read the page")

    assert result.success is True, result.errors
    agent = _call(calls, "Agent")
    assert agent["use_vision"] is True
    assert agent["use_judge"] == "not-passed"


@pytest.mark.unit
def test_llm_setup_failure_is_a_structured_result_not_a_traceback(
    fake_browser_use,
):
    provider, calls = fake_browser_use

    # The stand-in raises Browser Use 0.13's TypeError for this model name;
    # building the LLM used to sit outside the worker's error handling.
    result = provider(model="missing-model").run_task("Read the page")

    assert result.success is False
    assert len(result.errors) == 1
    assert result.errors[0].startswith("TypeError:")
    assert result.metadata["return_code"] == 0
    assert "import_seconds" in result.metadata["timings"]
    assert not [item for item in calls() if item["kind"] == "start"]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("auto", "auto"),
        ("AUTO", "auto"),
        (True, True),
        (False, False),
        ("on", True),
        ("0", False),
    ],
)
def test_vision_accepts_true_false_and_auto(value, expected):
    assert BrowserUseProvider(use_vision=value).use_vision == expected


@pytest.mark.unit
def test_vision_rejects_unknown_values():
    with pytest.raises(ValueError, match="true, false or auto"):
        BrowserUseProvider(use_vision="sometimes")


@pytest.mark.unit
def test_default_models_follow_the_browser_llm_provider():
    assert BrowserUseProvider(llm_provider="anthropic").effective_model == (
        "claude-haiku-5-5"
    )
    assert BrowserUseProvider(llm_provider="google").effective_model == (
        "gemini-flash-latest"
    )
    # Browser Use's own chat model has a default of its own.
    assert BrowserUseProvider(llm_provider="browseruse").effective_model is None
    assert BrowserUseProvider(model="gpt-4.1-mini").effective_model == "gpt-4.1-mini"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"), [(None, "auto"), ("auto", "auto"), ("false", False)]
)
def test_cli_config_defaults_vision_to_auto(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("MEMORIZZ_BROWSER_USE_VISION", raising=False)
    else:
        monkeypatch.setenv("MEMORIZZ_BROWSER_USE_VISION", raw)

    config = agent_factory.make_browser_control_config("browseruse")
    config.pop("provider")

    assert config["use_vision"] == expected
    assert BrowserUseProvider(**config).use_vision == expected

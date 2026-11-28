"""Every built-in model records its usage for the calling thread.

One model object serves an agent's chat turns and its background work. Code
that read "the last usage" after its own call could be handed another
thread's; only the OpenAI and Anthropic providers recorded per thread.
"""

import importlib
import threading

import pytest

from memorizz.observability.usage_hooks import clear_thread_usage, consume_thread_usage

PROVIDERS = [
    ("memorizz.llms.openai", "OpenAI"),
    ("memorizz.llms.anthropic", "Anthropic"),
    ("memorizz.llms.azure", "AzureOpenAI"),
    ("memorizz.llms.ollama", "OllamaLLM"),
    ("memorizz.llms.gateway", "ModelGateway"),
    ("memorizz.llms.gateway", "OpenRouter"),
    ("memorizz.llms.gateway", "Gemini"),
    ("memorizz.llms.huggingface", "HuggingFaceLLM"),
    ("memorizz.llms.mlx", "MLXLLM"),
]


def _bare(module, name):
    """The provider without its client: only usage recording is under test."""
    cls = getattr(importlib.import_module(module), name)
    return cls.__new__(cls)


@pytest.mark.unit
@pytest.mark.parametrize("module, name", PROVIDERS)
def test_a_providers_usage_is_recorded_for_the_calling_thread(module, name):
    model = _bare(module, name)
    clear_thread_usage()

    model._last_usage = {"input_tokens": 11, "output_tokens": 3}

    assert model._last_usage == {"input_tokens": 11, "output_tokens": 3}
    assert model.get_last_usage() == {"input_tokens": 11, "output_tokens": 3}
    assert consume_thread_usage(model) == (
        True,
        {"input_tokens": 11, "output_tokens": 3},
    )
    # Consumed once; and another model's usage is not this one's.
    assert consume_thread_usage(model) == (False, None)
    other = _bare(module, name)
    other._last_usage = {"input_tokens": 1}
    assert consume_thread_usage(model) == (False, None)


@pytest.mark.unit
@pytest.mark.parametrize("module, name", PROVIDERS)
def test_a_call_that_reports_nothing_is_not_given_an_earlier_calls_usage(module, name):
    """A provider clears its usage before each call. With no usage reported
    the thread's record is that nothing was, not the previous call's."""
    model = _bare(module, name)
    model._last_usage = {"input_tokens": 900}

    model._last_usage = None

    assert consume_thread_usage(model) == (True, None)


@pytest.mark.unit
@pytest.mark.parametrize("module, name", PROVIDERS)
def test_threads_sharing_one_model_each_read_their_own_usage(module, name):
    model = _bare(module, name)
    barrier = threading.Barrier(8)
    seen = {}

    def work(n):
        barrier.wait()
        for _ in range(200):
            model._last_usage = {"input_tokens": n}
            recorded, usage = consume_thread_usage(model)
            if not recorded or usage != {"input_tokens": n}:
                seen[n] = (recorded, usage)
                return
        seen[n] = "ok"

    threads = [threading.Thread(target=work, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert seen == {n: "ok" for n in range(8)}

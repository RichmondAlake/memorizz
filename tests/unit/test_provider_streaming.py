"""Provider wire-shape and cooperative-worker proofs, without live API calls."""

import sys
import threading
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from memorizz.llms.streaming import ProviderStreamError


class Closeable:
    def __init__(self, values):
        self.values = iter(values)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.values)

    def close(self):
        self.closed = True


def chat_chunks():
    return [
        NS(
            choices=[NS(delta=NS(content=" x ", tool_calls=None), finish_reason=None)],
            usage=None,
        ),
        NS(
            choices=[NS(delta=NS(content=None, tool_calls=None), finish_reason="stop")],
            usage=None,
        ),
        NS(choices=[], usage=NS(prompt_tokens=2, completion_tokens=1, total_tokens=3)),
    ]


def test_azure_native_stream_and_usage():
    from memorizz.llms.azure import AzureOpenAI

    provider = AzureOpenAI.__new__(AzureOpenAI)
    provider.model = "deployment"
    wire = Closeable(chat_chunks())
    create = Mock(return_value=wire)
    provider.client = NS(chat=NS(completions=NS(create=create)))
    events = list(provider.generate_stream([]))
    assert create.call_args.kwargs["stream"] is True
    assert events[0] == {"type": "content", "content": " x "}
    assert events[-1] == {"type": "done", "content": " x "}
    assert events[-2]["usage"]["total_tokens"] == 3
    assert wire.closed


def test_responses_provider_native_request_options_and_close():
    from memorizz.llms.openai import OpenAI

    provider = OpenAI.__new__(OpenAI)
    provider.model, provider.api_mode = "test", "responses"
    provider.reasoning_effort = None
    provider._request_options = {"max_completion_tokens": 64}
    provider._prompt_cache_key = "scope"
    provider._prompt_cache_retention = None
    wire = Closeable(
        [
            NS(type="response.output_text.delta", delta=" first "),
            NS(type="response.completed", response=NS(status="completed", usage=None)),
        ]
    )
    create = Mock(return_value=wire)
    provider.client = NS(responses=NS(create=create))
    provider.generate = Mock(
        side_effect=AssertionError("Must not buffer through generate()")
    )
    iterator = provider.generate_stream([{"role": "user", "content": "hi"}])
    assert next(iterator) == {"type": "content", "content": " first "}
    assert create.call_args.kwargs["stream"] is True
    assert create.call_args.kwargs["max_output_tokens"] == 64
    assert create.call_args.kwargs["prompt_cache_key"] == "scope"
    iterator.close()
    assert wire.closed


@pytest.mark.parametrize("finished", [False, True])
def test_ollama_done_dict_tools_usage_and_close(finished):
    from memorizz.llms.ollama import OllamaLLM

    provider = OllamaLLM.__new__(OllamaLLM)
    provider._chat_kwargs = lambda **kw: kw
    wire = Closeable(
        [
            {
                "message": {"content": " x "},
                "done": finished,
                "done_reason": "stop",
                "prompt_eval_count": 2,
                "eval_count": 1,
            }
        ]
    )
    provider.client = NS(chat=lambda **kw: wire)
    if not finished:
        with pytest.raises(ProviderStreamError, match="incomplete"):
            list(provider.generate_stream([]))
    else:
        events = list(provider.generate_stream([]))
        assert events[-1]["content"] == " x "
        assert events[-2]["usage"]["total_tokens"] == 3
    assert wire.closed


def test_mlx_preserves_whitespace_and_closes(monkeypatch):
    from memorizz.llms.mlx import MLXLLM

    provider = MLXLLM.__new__(MLXLLM)
    provider._model = provider._tokenizer = None
    provider.max_new_tokens = 64
    provider._build_prompt = lambda _: "prompt"
    provider._build_sampler = lambda: None
    provider._count_tokens = lambda _: 1
    wire = Closeable(
        [NS(text=" \n"), NS(text="é ", prompt_tokens=2, generation_tokens=2)]
    )
    monkeypatch.setitem(sys.modules, "mlx_lm", NS(stream_generate=lambda *a, **k: wire))
    events = list(provider.generate_stream([]))
    assert (
        "".join(e["content"] for e in events if e["type"] == "content")
        == events[-1]["content"]
        == " \né "
    )
    assert wire.closed


@pytest.mark.parametrize("fail", [False, True])
def test_huggingface_bounded_worker_exact_text_and_errors(monkeypatch, fail):
    from memorizz.llms.huggingface import HuggingFaceLLM

    class Streamer:
        def __init__(self, *a, **k):
            pass

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        NS(
            TextIteratorStreamer=Streamer,
            StoppingCriteria=object,
            StoppingCriteriaList=list,
        ),
    )
    provider = HuggingFaceLLM.__new__(HuggingFaceLLM)
    provider.max_new_tokens, provider.temperature, provider.top_p = 100, 0, 1
    provider._messages_to_prompt = lambda _: "p"
    provider._count_tokens = lambda _: 1

    def pipeline(*a, streamer, **kwargs):
        assert kwargs["stopping_criteria"]
        streamer.on_finalized_text(" \né ")
        if fail:
            raise ValueError("fixture pipeline failure")
        streamer.on_finalized_text("", stream_end=True)

    pipeline.tokenizer = object()
    provider._pipeline = pipeline
    if fail:
        with pytest.raises(ValueError, match="fixture pipeline"):
            list(provider.generate_stream([]))
    else:
        events = list(provider.generate_stream([]))
        assert events[-1]["content"] == " \né "
        assert (
            "".join(e["content"] for e in events if e["type"] == "content")
            == events[-1]["content"]
        )
    assert not any(
        t.name == "memorizz-huggingface-stream" for t in threading.enumerate()
    )


def test_huggingface_close_stops_full_producer(monkeypatch):
    from memorizz.llms.huggingface import HuggingFaceLLM

    class Streamer:
        def __init__(self, *a, **k):
            pass

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        NS(
            TextIteratorStreamer=Streamer,
            StoppingCriteria=object,
            StoppingCriteriaList=list,
        ),
    )
    provider = HuggingFaceLLM.__new__(HuggingFaceLLM)
    provider.max_new_tokens, provider.temperature, provider.top_p = 10000, 0, 1
    provider._messages_to_prompt = lambda _: "p"
    stopped = threading.Event()

    def pipeline(*a, streamer, stopping_criteria, **kwargs):
        try:
            while not stopping_criteria[0](None, None):
                streamer.on_finalized_text("x")
        finally:
            stopped.set()

    pipeline.tokenizer = object()
    provider._pipeline = pipeline
    iterator = provider.generate_stream([])
    assert next(iterator)["content"] == "x"
    iterator.close()
    assert stopped.wait(1)

# Model Providers

A MemAgent accepts either a secret-free `llm_config` or an already constructed
`LLMProvider`. Keep credentials in the process environment or a deployment
secret manager so saved agent definitions remain portable and safe to inspect.

## Choose a provider

| Provider | Install | Required environment | Config name |
|---|---|---|---|
| OpenAI | Base package | `OPENAI_API_KEY` | `openai` |
| Anthropic | `memorizz[anthropic]` | `ANTHROPIC_API_KEY` | `anthropic` |
| Google Gemini | Base package | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | `gemini` |
| Ollama | Base package plus Ollama daemon | optional `OLLAMA_HOST` (models and embeddings) | `ollama` |
| Azure OpenAI | Base package | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `OPENAI_API_VERSION` | `azure` |
| Hugging Face | `memorizz[huggingface]` | optional `HF_TOKEN` | `huggingface` |
| MLX | `memorizz[mlx]` on native Apple Silicon | none for public models | `mlx` |
| OpenRouter | Base package | `OPENROUTER_API_KEY` | `openrouter` |
| Another gateway | Base package | the variable you name in `api_key_env` | `gateway` |

Use a model identifier or Azure deployment that is available to your account
and supports the tool behavior your agent requires. Model availability and
limits change independently of Memorizz.

## Configure through the builder

=== "OpenAI"

    ```python
    agent = (
        MemAgentBuilder()
        .with_llm_config(
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_mode": "chat_completions",
            }
        )
        .build()
    )
    ```

=== "Anthropic"

    ```python
    agent = (
        MemAgentBuilder()
        .with_llm_config(
            {
                "provider": "anthropic",
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 4096,
                "enable_prompt_caching": True,
            }
        )
        .build()
    )
    ```

=== "Ollama"

    ```python
    agent = (
        MemAgentBuilder()
        .with_llm_config(
            {
                "provider": "ollama",
                "model": "qwen2.5:7b",
                "think": False,
            }
        )
        .build()
    )
    ```

    The context window is sent to Ollama as `options.num_ctx`. Set
    `context_window_tokens` in `llm_config` to choose it; otherwise
    `additional_config.num_ctx`, then `OLLAMA_CONTEXT_LENGTH`, then the model's
    own context length, then 8,192. The model's length is used in full when its
    attention cache fits in a quarter of this machine's RAM, estimated from the
    daemon's model metadata (sliding-window, shared and state-space layers
    included); otherwise the largest of 131,072, 65,536 or 32,768 that fits, and
    never less than 16,384. On a 16 GB Mac that gives gemma4 131,072 and
    qwen2.5:7b 32,768. A daemon on another machine gets 16,384. A MemAgent's
    instructions and tool schemas alone take about 6–7k tokens, so smaller
    windows leave little room for the conversation; long prompts on a local
    model also take longer to read, so pick a smaller window if replies slow down.

    Only a window you set is saved with the agent; a default is recomputed on
    every load, so changing the model or its settings takes effect.

    In the playground, **Settings → Context window** lists sizes up to what the
    selected model supports (for example 131,072 for `gemma4`). The context
    panel names the model in use and notes when it supports a larger window
    than the one requested, and each reply's footnote names the model that
    answered it.

=== "Azure OpenAI"

    ```python
    agent = (
        MemAgentBuilder()
        .with_llm_config(
            {
                "provider": "azure",
                "deployment_name": "my-deployment",
            }
        )
        .build()
    )
    ```

The snippets assume `from memorizz import MemAgentBuilder` and provider
credentials in the environment.


## Google Gemini

Gemini models are reached through the Gemini API with a key from
[Google AI Studio](https://aistudio.google.com/apikey):

```python
from memorizz import Gemini, MemAgent

agent = MemAgent(
    model=Gemini("gemini-4-hydrogen", reasoning_effort="low"),
    memory_provider=memory_provider,
)
```

The same model as a saved, secret-free config:

```python
{"provider": "gemini", "model": "gemini-4-hydrogen"}
```

- **The key** is read from `GEMINI_API_KEY`, then `GOOGLE_API_KEY`. In the
  local UI it goes in **Settings → Google Gemini API key**; in the CLI,
  `/login gemini`.
- **Models.** The pickers list the Gemini 4 and Gemini 3 models that chat
  and call tools, and add any newer ones your key lists. Any model code from
  the Gemini API works as a custom model.
- **Thinking.** `reasoning_effort` takes `minimal`, `low`, `medium`, `high`
  and, on some Gemini 4 models, `xhigh`. Gemini 3 and 4 always think; it
  cannot be turned off.
- **Tools.** Gemini 3 and 4 sign each step that calls a tool and refuse a
  conversation that leaves the signature out. MemoRizz keeps it with the tool
  call and sends it back, including when the answer is streamed. A step
  another model answered (after switching models mid-conversation) has no
  signature, and is sent with the stand-in value Google documents for that.
- **Free tier.** A key with no billing reaches only some models. On
  2026-11-25 a free key could call Gemini 3.5 Flash, 3.5 Flash-Lite,
  3.6 Flash and 3.8 Flash (a few requests a minute), and Google refused
  Gemini 4 Neon, Carbon and Argon outright. When Google refuses a request,
  the run says why, for example "Google does not offer gemini-4-neon on the
  Gemini API free tier. Turn on billing for the key's Google project…".
- **Cost.** Runs are priced from Google's standard list rates (read
  2026-11-26), including the dated rate changes Google has announced. A
  free-tier key is not charged, so treat the figure as what the run would
  cost on a paid key. Gemini 3.5 Flash and 3.7 Flash are not on Google's
  price page and show tokens only.

Requests go to Google's OpenAI-compatible endpoint, which Google marks as
beta. Gemini models are also available through [OpenRouter](#openrouter-and-other-model-gateways)
as `google/gemini-...`.

## OpenRouter and other model gateways

A gateway serves many models behind one address and one key.
[OpenRouter](https://openrouter.ai) is built in:

```python
from memorizz import MemAgent, ModelRouting, OpenRouter

agent = MemAgent(
    model=OpenRouter(
        "anthropic/claude-sonnet-5.5",   # any name from openrouter.ai/models
        routing=ModelRouting(
            fallback_models=["openai/gpt-6.2-sol"],
            sort="throughput",
            data_collection="deny",
        ),
    ),
    memory_provider=memory_provider,
)
```

The key is read from `OPENROUTER_API_KEY`. The same model as a saved,
secret-free config:

```python
{
    "provider": "openrouter",
    "model": "anthropic/claude-sonnet-5.5",
    "routing": {"fallback_models": ["openai/gpt-6.2-sol"], "sort": "throughput"},
}
```

### Routing

`ModelRouting` says who may answer. Every setting is optional.

| Setting | What it does |
|---|---|
| `fallback_models` | Models to try, in order, when the agent's model errors, is rate limited or is down. The answer is billed at the rate of the model that gave it. |
| `providers` | Providers to try first, in order. |
| `only`, `ignore` | The only providers allowed, and providers never to use. |
| `allow_fallbacks` | `False` fails a request instead of moving to a provider outside `providers`. |
| `sort` | `"price"`, `"throughput"` or `"latency"`. |
| `data_collection` | `"deny"` uses only providers that do not keep or train on prompts. |
| `zero_data_retention` | `True` uses only endpoints that retain nothing. |
| `require_parameters` | `True` uses only providers that support every parameter sent. |
| `quantizations` | Accepted quantization levels, such as `["fp8"]`. |
| `max_price` | The most to pay per million tokens: `{"prompt": 1, "completion": 2}`. |

Change the rules on an agent at any time; they are saved with it:

```python
agent.with_model_routing(ModelRouting(fallback_models=["openrouter/free"]))
agent.get_model_routing()        # {"fallback_models": ["openrouter/free"]}
agent.with_model_routing(None)   # back to OpenRouter's defaults
```

Routing applies to every request the agent's model makes, including the
plain-text calls behind summaries and tool descriptions. The model that
actually answered is recorded on the trace as the response model.

In the console, the same settings are the **Routing** group under the model
on the agent form and in the playground's **Settings** tab.

### What a call cost

OpenRouter returns the charge for each call with its token counts. MemoRizz
records it as the call's cost, so **Observability → Usage & cost**, the trace
view and `agent.get_last_run_usage()` show spend for any model OpenRouter
serves, with no rate card to maintain. See
[Usage, cost and memory analytics](../usage-analytics.md).

### Another gateway

`ModelGateway` talks to any service that speaks the OpenAI chat protocol on
its own address with its own key (a company gateway, vLLM, LiteLLM):

```python
from memorizz import ModelGateway

model = ModelGateway(
    "team-coder",
    base_url="https://gateway.example/v1",
    api_key_env="TEAM_GATEWAY_KEY",      # the variable's name; never the key
    extra_body={"tier": "fast"},          # the gateway's own request options
)
```

Only the variable's name is saved with an agent. A gateway is never sent
`OPENAI_API_KEY`; without a key it gets a placeholder, which is what a local
server expects.

## Saving and changing models

Saved provider configuration preserves the effective context window and
supported generation settings. API credentials are supplied separately through
the environment when reloading.

`MemAgent.load(agent_id, memory_provider=provider, llm_config=new_config)` uses
the replacement configuration for both the runtime model and reported metadata.
A normal reload preserves an agent-level `context_window_tokens` cap. Replacing
the model uses its own context budget unless you pass a new agent-level cap;
that cap cannot exceed the provider's configured window.

## Bring an initialized provider

```python
from memorizz import MemAgentBuilder
from memorizz.llms import OpenAI

model = OpenAI(
    model="gpt-4o-mini",
    prompt_cache_retention="in_memory",
)

agent = MemAgentBuilder().with_model(model).build()
```

Use this form for dependency injection, tests, or a custom `LLMProvider`.
Implement `generate`, `generate_stream`, `get_config`, `get_last_usage`, and
`get_context_window_tokens`; normalize tool calls and provider errors to the
same runtime contract.

## Local OpenAI-compatible endpoints

llama.cpp, LM Studio, vLLM, and compatible gateways can use the OpenAI adapter:

```python
agent = (
    MemAgentBuilder()
    .with_llm_config(
        {
            "provider": "openai",
            "model": "local-model",
            "base_url": "http://127.0.0.1:8080/v1",
            "api_mode": "chat_completions",
        }
    )
    .build()
)
```

Only use a trusted endpoint. A configured `base_url` is application authority
and should not come from model output or an unvalidated user field.

## Tool and role compatibility

- OpenAI, Anthropic, Azure OpenAI, and Ollama implement the Memorizz tool loop.
- Hugging Face and MLX are text-generation providers and can degrade to
  text-only behavior when tools are supplied; do not choose them for a workflow
  that requires reliable function calls without testing the exact model path.
- Reviewed developer-authority skills are represented differently by provider.
  Test instruction precedence with the exact model used in deployment.
- Set `raise_on_provider_error=True` on `run_stream()` when a service must
  receive an exception after the typed terminal event.
- By default a failed turn answers with "I encountered an error: ...". Build
  the agent with `raise_errors=True`, or pass `run(..., raise_errors=True)`
  for one call, to get the exception instead. Running out of steps is still
  an answer.

## Verify resolved state

```python
agent.validate_configuration()
report = agent.capability_report()
print(report["agent"]["llm_provider"])
print(report["agent"]["llm_model"])
print(agent.model.get_last_usage())
```

Capability output does not prove that a paid request will succeed. Add a
bounded startup smoke test in a non-production scope when authentication,
deployment routing, or tool support is critical.

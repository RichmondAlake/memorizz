# Model Providers

A MemAgent accepts either a secret-free `llm_config` or an already constructed
`LLMProvider`. Keep credentials in the process environment or a deployment
secret manager so saved agent definitions remain portable and safe to inspect.

## Choose a provider

| Provider | Install | Required environment | Config name |
|---|---|---|---|
| OpenAI | Base package | `OPENAI_API_KEY` | `openai` |
| Anthropic | `memorizz[anthropic]` | `ANTHROPIC_API_KEY` | `anthropic` |
| Ollama | Base package plus Ollama daemon | optional `OLLAMA_HOST` | `ollama` |
| Azure OpenAI | Base package | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `OPENAI_API_VERSION` | `azure` |
| Hugging Face | `memorizz[huggingface]` | optional `HF_TOKEN` | `huggingface` |
| MLX | `memorizz[mlx]` on native Apple Silicon | none for public models | `mlx` |

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

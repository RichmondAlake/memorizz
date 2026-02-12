# Toolbox Module

`Toolbox` persists callable tool metadata and supports semantic lookup of relevant tools.

## Quick Start

```python
from pathlib import Path

from memorizz.long_term_memory.procedural.toolbox import Toolbox
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

provider = FileSystemProvider(FileSystemConfig(root_path=Path("~/.memorizz").expanduser()))
toolbox = Toolbox(provider)

@toolbox.register_tool
def get_weather(city: str) -> str:
    """Return a mock weather response for a city."""
    return f"Sunny in {city}"

tool_id = toolbox.register_tool(lambda ticker: f"Quote for {ticker}")

by_name = toolbox.get_tool_by_name("get_weather")
related = toolbox.get_most_similar_tools("weather forecast", limit=3)
all_tools = toolbox.list_tools()
```

## With MemAgent

```python
from memorizz.memagent.builders import MemAgentBuilder

agent = (
    MemAgentBuilder()
    .with_memory_provider(provider)
    .with_instruction("Use tools when useful.")
    .with_llm_config({"provider": "openai", "model": "gpt-4o-mini", "api_key": "..."})
    .with_tool(get_weather)
    .build()
)

# You can also add tools after build
agent.add_tool(get_weather)
```

## Notes

- Tool metadata is stored under `MemoryType.TOOLBOX`.
- Callable function bodies exist in the current Python process; metadata persists in the provider.
- `toolbox.list_available_tools()` returns only tools with callable functions loaded in memory.

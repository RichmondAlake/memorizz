"""Tests for moving live Toolbox callables into a MemAgent ToolManager."""

from unittest.mock import Mock, patch

import pytest

from memorizz.long_term.procedural.toolbox import Toolbox
from memorizz.memagent.managers import ToolManager


@pytest.mark.unit
def test_toolbox_registers_without_llm_metadata_and_loads_live_callable():
    memory_provider = Mock()
    llm_provider = Mock()

    def multiply(value: int, factor: int) -> int:
        """Multiply a value by a factor."""
        return value * factor

    with patch(
        "memorizz.long_term.procedural.toolbox.toolbox.get_embedding",
        return_value=[0.1, 0.2],
    ):
        toolbox = Toolbox(memory_provider, llm_provider=llm_provider)
        tool_id = toolbox.register_tool(multiply)

    stored = memory_provider.store.call_args.args[0]
    assert stored["name"] == "multiply"
    assert stored["parameters"]["value"]["type"] == "integer"
    assert stored["required"] == ["value", "factor"]
    llm_provider.get_tool_metadata.assert_not_called()

    manager = ToolManager(memory_provider)
    assert manager.initialize_from_toolbox(toolbox) == 1
    assert manager.list_tools() == ["multiply"]
    assert manager.execute_tool("multiply", {"value": 3, "factor": 4}) == (12, None)
    assert toolbox.get_function_by_id(tool_id) is multiply

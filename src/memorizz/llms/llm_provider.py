# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

# src/memorizz/llms/llm_provider.py

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

# Use TYPE_CHECKING to handle forward references for type hints
if TYPE_CHECKING:
    pass

"""
A protocol in Python (introduced in PEP 544 and part of the typing module) defines a structural typing rule.
It specifies a set of methods and properties that a class must implement,
but it does not require inheritance.

"If it walks like a duck and quacks like a duck, it's probably a duck." 🦆

"""


@runtime_checkable
class LLMProvider(Protocol):
    """
    A generic protocol that defines the contract for any LLM provider
    to be compatible with both the OpenAI and AzureOpenAI classes.
    """

    # --- Attributes ---
    client: Any
    """Provides direct access to the underlying API client instance (e.g., openai.OpenAI or openai.AzureOpenAI)."""

    model: str
    """Stores the specific model or deployment name as a string (e.g., "gpt-4o")."""

    # --- Methods ---
    def get_tool_metadata(self, func: Callable) -> Dict[str, Any]:
        """Creates structured metadata (a JSON schema) from a Python function."""
        ...

    def augment_docstring(self, docstring: str) -> str:
        """Uses the LLM to enhance a function's docstring with more detail."""
        ...

    def generate_queries(self, docstring: str) -> List[str]:
        """Generates a list of example user queries for a given tool."""
        ...

    def generate_text(self, prompt: str, instructions: Optional[str] = None) -> str:
        """A high-level method for simple text generation."""
        ...

    def generate(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> Any:
        """Generate a response from a list of messages (chat format), optionally with tool calling."""
        ...

    def get_config(self) -> Dict[str, Any]:
        """
        Returns a serializable dictionary of the provider's configuration.
        This is used for saving and reconstructing the agent.
        """
        ...

    def get_last_usage(self) -> Optional[Dict[str, int]]:
        """Return token usage details (prompt/completion/total) from the most recent call."""
        ...

    def get_context_window_tokens(self) -> Optional[int]:
        """Return the provider's context window size in tokens, when known."""
        ...

    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream a response from a list of messages (chat format), optionally with tool calling.

        Yields dictionaries with one of these shapes:
            - {"type": "content", "content": "..."} for text chunks
            - {"type": "tool_calls", "response": <full response object>} when tool calls are detected
            - {"type": "done", "content": "<full accumulated text>"} at the end of the stream
            - {"type": "usage", "usage": {...}} for known token accounting

        Content deltas concatenate exactly to done.content, including whitespace.
        Exactly one done or completed tool_calls event ends a successful request;
        an incomplete/refused request raises, it must not invent success at EOF.
        Reasoning is diagnostic only. Tool response objects are internal and
        never serialized into public SDK events. Consumers close this generator;
        adapters must close their raw provider stream and cooperate with the
        current cancellation token. See llms.streaming.streaming_capabilities
        for reliable detection (an inherited Protocol stub is not an iterator).
        """
        ...


@runtime_checkable
class ResponseMetadataProvider(Protocol):
    """Optional extension; existing LLMProvider implementations remain valid."""

    def get_last_response_metadata(self) -> Optional[Dict[str, Any]]:
        """Return bounded response metadata; omit unknown values."""
        ...

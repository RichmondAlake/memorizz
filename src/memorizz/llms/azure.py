# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

# src/memorizz/llms/azure.py

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import openai

from .llm_provider import LLMProvider, ResponseMetadataMixin
from .tool_metadata import ResponsesToolMetadataMixin

# Suppress httpx logs to reduce noise from API requests
logging.getLogger("httpx").setLevel(logging.WARNING)

# Use TYPE_CHECKING for forward references to avoid circular imports
if TYPE_CHECKING:
    pass


class AzureOpenAI(ResponsesToolMetadataMixin, ResponseMetadataMixin, LLMProvider):
    """
    A class for interacting with the Azure OpenAI API.
    """

    def __init__(
        self,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        deployment_name: str = "gpt-4o",
        context_window_tokens: Optional[int] = None,
    ):
        """
        Initialize the Azure OpenAI client.

        Parameters:
        -----------
        azure_endpoint : str, optional
            The endpoint for the Azure OpenAI API. Defaults to env var `AZURE_OPENAI_ENDPOINT`.
        api_version : str, optional
            The API version for the Azure OpenAI API. Defaults to env var `OPENAI_API_VERSION`.
        deployment_name : str, optional
            The deployment name for the model to use. Defaults to "gpt-4o".
        """
        self._api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = api_version or os.getenv("OPENAI_API_VERSION")

        if not all([self._api_key, self.azure_endpoint, self.api_version]):
            raise ValueError(
                "Azure credentials not found. Please set the AZURE_OPENAI_API_KEY, "
                "AZURE_OPENAI_ENDPOINT, and OPENAI_API_VERSION environment variables or "
                "pass them as arguments."
            )

        self.client = openai.AzureOpenAI(
            api_key=self._api_key,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version,
        )
        # In Azure, the 'model' is the deployment name.
        self.model = deployment_name
        self.context_window_tokens = context_window_tokens or 128_000
        self._last_usage: Optional[Dict[str, int]] = None

    def get_config(self) -> Dict[str, Any]:
        """Returns a serializable configuration for the AzureOpenAI provider."""
        return {
            "provider": "azure",
            "deployment_name": self.model,
            "azure_endpoint": self.azure_endpoint,
            "api_version": self.api_version,
            "context_window_tokens": self.context_window_tokens,
            # Note: We don't save the API key for security. It should be loaded from env vars.
        }

    def generate_text(self, prompt: str, instructions: str = None) -> str:
        """
        Generate text using Azure OpenAI's API.

        Parameters:
            prompt (str): The prompt to generate text from.
            instructions (str): The instructions to use for the generation.

        Returns:
            str: The generated text.
        """
        response = self.client.responses.create(
            model=self.model, instructions=instructions, input=prompt
        )

        return response.output_text

    def generate(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> Any:
        """
        Generate a response using Azure OpenAI's chat completions API.

        Parameters:
            messages (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content'.
                Example: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            tools (Optional[List[Dict[str, Any]]]): List of tool definitions for function calling.
            tool_choice (str): Controls which (if any) function is called ("auto", "none", or specific function).

        Returns:
            Any: Either a string (final response) or the full response object (if tool calls are present).
        """
        kwargs = {"model": self.model, "messages": messages}

        # Add tools if provided
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        self._last_usage = None
        self._last_response_metadata = {}
        response = self.client.chat.completions.create(**kwargs)
        self._last_usage = self._extract_usage(response)
        from .response_metadata import response_metadata

        self._last_response_metadata = response_metadata(response)

        # If there are tool calls, return the full response object
        if response.choices[0].message.tool_calls:
            return response

        # Otherwise return just the text content
        return response.choices[0].message.content

    def generate_stream(self, messages, tools=None, tool_choice="auto"):
        """Stream Azure chat deltas using the shared, closeable chat parser."""
        from .streaming import chat_events

        kwargs = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs.update(tools=tools, tool_choice=tool_choice)
        self._last_usage = None
        self._last_response_metadata = {}
        yield from chat_events(self, self.client.chat.completions.create(**kwargs))

    def _extract_usage(self, response: Any) -> Optional[Dict[str, int]]:
        usage = getattr(response, "usage", None)
        if not usage:
            return None
        try:
            return {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        except Exception:
            return None

    def get_last_usage(self) -> Optional[Dict[str, int]]:
        return self._last_usage

    def get_context_window_tokens(self) -> Optional[int]:
        return self.context_window_tokens

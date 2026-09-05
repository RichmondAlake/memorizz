# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

# src/memorizz/llms/azure.py

import inspect
import logging
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import openai

from .llm_provider import LLMProvider

# Suppress httpx logs to reduce noise from API requests
logging.getLogger("httpx").setLevel(logging.WARNING)

# Use TYPE_CHECKING for forward references to avoid circular imports
if TYPE_CHECKING:
    pass


class AzureOpenAI(LLMProvider):
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
            # Note: We don't save the API key for security. It should be loaded from env vars.
        }

    def get_tool_metadata(self, func: Callable) -> Dict[str, Any]:
        """
        Get the metadata for a tool.

        Parameters:
        -----------
        func : Callable
            The function to get the metadata for.

        Returns:
        --------
        Dict[str, Any]
        """
        # We'll import ToolSchemaType here to avoid circular imports
        from ..long_term.procedural.toolbox.tool_schema import ToolSchemaType

        docstring = func.__doc__ or ""
        signature = str(inspect.signature(func))
        func_name = func.__name__

        system_msg = {
            "role": "system",
            "content": (
                "You are an expert metadata augmentation assistant specializing in JSON schema discovery "
                "and documentation enhancement.\n\n"
                f"**IMPORTANT**: Use the function name exactly as provided (`{func_name}`) and do NOT rename it."
            ),
        }

        user_msg = {
            "role": "user",
            "content": (
                f"Generate enriched metadata for the function `{func_name}`.\n\n"
                f"- Docstring: {docstring}\n"
                f"- Signature: {signature}\n\n"
                "Enhance the metadata by:\n"
                "• Expanding the docstring into a detailed description.\n"
                "• Writing clear natural‐language descriptions for each parameter, including type, purpose, and constraints.\n"
                "• Identifying which parameters are required.\n"
                "• (Optional) Suggesting example queries or use cases.\n\n"
                "Produce a JSON object that strictly adheres to the ToolSchemaType structure."
            ),
        }

        response = self.client.responses.parse(
            model=self.model, input=[system_msg, user_msg], text_format=ToolSchemaType
        )

        return response.output_parsed

    def augment_docstring(self, docstring: str) -> str:
        """
        Augment the docstring with an LLM generated description.

        Parameters:
        -----------
        docstring : str
            The docstring to augment.

        Returns:
        --------
        str
        """
        response = self.client.responses.create(
            model=self.model,
            input=f"Augment the docstring {docstring} by adding more details and examples.",
        )

        return response.output_text

    def generate_queries(self, docstring: str) -> List[str]:
        """
        Generate queries for the tool.

        Parameters:
        -----------
        docstring : str
            The docstring to generate queries for.

        Returns:
        --------
        List[str]
        """
        response = self.client.responses.create(
            model=self.model,
            input=f"Generate queries for the docstring {docstring} by adding some examples of queries that can be used to leverage the tool.",
        )

        return response.output_text

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

    def get_last_response_metadata(self) -> Dict[str, Any]:
        from .response_metadata import last_response_metadata

        return last_response_metadata(self)

    def get_context_window_tokens(self) -> Optional[int]:
        return self.context_window_tokens

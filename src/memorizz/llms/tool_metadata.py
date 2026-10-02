# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""How each LLM class describes a Python function as a tool.

The Toolbox asks an LLM to enrich a function's metadata, expand its docstring
and suggest example queries. Each strategy below is shared by two providers:
OpenAI and Azure use structured Responses output; Anthropic and Ollama prompt
for JSON; HuggingFace and MLX prompt for the ToolSchemaType shape.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


class ResponsesToolMetadataMixin:
    """Structured tool metadata through the OpenAI Responses API
    (``self.client`` and ``self.model``)."""

    def get_tool_metadata(self, func: Callable) -> Dict[str, Any]:
        """Enriched ToolSchemaType metadata for ``func``."""
        # Imported here to avoid a circular import through the toolbox.
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
        """``docstring`` expanded with more detail and examples."""
        response = self.client.responses.create(
            model=self.model,
            input=f"Augment the docstring {docstring} by adding more details and examples.",
        )
        return response.output_text

    def generate_queries(self, docstring: str) -> List[str]:
        """Example queries that would use the tool."""
        response = self.client.responses.create(
            model=self.model,
            input=f"Generate queries for the docstring {docstring} by adding some examples of queries that can be used to leverage the tool.",
        )
        return response.output_text


class JsonPromptToolMetadataMixin:
    """Tool metadata by prompting ``self.generate_text`` for JSON."""

    def get_tool_metadata(self, func: Callable) -> Dict[str, Any]:
        """Generate tool metadata by introspecting the function signature."""
        sig = inspect.signature(func)
        docstring = func.__doc__ or ""
        func_name = func.__name__

        prompt = (
            f"Generate enriched metadata for the function `{func_name}`.\n\n"
            f"- Docstring: {docstring}\n"
            f"- Signature: {sig}\n\n"
            "Produce a JSON object with keys: name (string, must be '{func_name}'), "
            "description (string), parameters (object with properties, each having "
            "type and description), and required (list of required param names).\n"
            "Return ONLY the JSON."
        )
        raw = self.generate_text(prompt, instructions="Return valid JSON only.")
        return self._safe_json_parse(raw)

    def augment_docstring(self, docstring: str) -> str:
        return self.generate_text(
            f"Augment the docstring by adding more details and examples:\n\n{docstring}",
        )

    def generate_queries(self, docstring: str) -> List[str]:
        raw = self.generate_text(
            f"Generate example user queries for a tool with this docstring:\n\n{docstring}\n\n"
            "Return a JSON array of strings."
        )
        parsed = self._safe_json_parse(raw)
        if isinstance(parsed, list):
            return parsed
        return [raw]

    @staticmethod
    def _safe_json_parse(text: str) -> Any:
        """Try to parse JSON from text, handling markdown code fences."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text


class SchemaPromptToolMetadataMixin:
    """ToolSchemaType metadata by prompting ``self.generate_text`` (local models)."""

    def augment_docstring(self, docstring: str) -> str:
        instructions = (
            "You improve terse docstrings. Expand with helpful detail and examples."
        )
        return self.generate_text(docstring, instructions=instructions)

    def generate_queries(self, docstring: str) -> List[str]:
        prompt = (
            "Generate three short example queries or tasks that would use the "
            "following tool:\n\n"
            f"{docstring}"
        )
        raw_output = self.generate_text(prompt)
        lines = [line.strip(" -•") for line in raw_output.splitlines() if line.strip()]
        return [line for line in lines if line]

    def get_tool_metadata(self, func: Any) -> Dict[str, Any]:
        from ..long_term.procedural.toolbox.tool_schema import ToolSchemaType

        docstring = func.__doc__ or ""
        signature = str(inspect.signature(func))
        func_name = func.__name__

        prompt = (
            "You produce JSON metadata for Python functions.\n"
            "The JSON must strictly follow this schema:\n"
            "{"
            '"type": "function", '
            '"function": {'
            '"name": str, '
            '"description": str, '
            '"parameters": [{"name": str, "description": str, "type": str, "required": bool}], '
            '"required": [str], '
            '"queries": [str]'
            "}"
            "}\n\n"
            f"Function name: {func_name}\n"
            f"Signature: {signature}\n"
            f"Docstring: {docstring}\n"
            "Return a JSON object only."
        )
        raw_output = self.generate_text(prompt)
        metadata_dict = self._safe_json_parse(raw_output)
        tool_schema = ToolSchemaType.model_validate(metadata_dict)
        return tool_schema.model_dump()

    def _safe_json_parse(self, text: str) -> Dict[str, Any]:
        """Attempt to parse JSON even if wrapped with commentary."""
        snippet = text.strip()
        start = snippet.find("{")
        end = snippet.rfind("}")
        if start != -1 and end != -1:
            snippet = snippet[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse JSON output: %s", snippet)
            raise ValueError("LLM response was not valid JSON") from exc


__all__ = [
    "JsonPromptToolMetadataMixin",
    "ResponsesToolMetadataMixin",
    "SchemaPromptToolMetadataMixin",
]

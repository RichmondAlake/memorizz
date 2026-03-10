# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Response processing and formatting for MemAgent."""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ResponseHandler:
    """
    Handles response processing and formatting for MemAgent.

    This class encapsulates response-related logic, including validation,
    formatting, and post-processing of LLM responses.
    """

    def __init__(self):
        """Initialize the response handler."""
        self.response_processors = {
            "default": self._default_processor,
            "json": self._json_processor,
            "markdown": self._markdown_processor,
        }

    def process_response(
        self,
        raw_response: str,
        response_type: str = "default",
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Process and format a raw LLM response.

        Args:
            raw_response: Raw response from the LLM
            response_type: Type of processing to apply
            context: Optional context for processing

        Returns:
            Processed response string
        """
        try:
            if not raw_response:
                return (
                    "I apologize, but I didn't generate a response. Please try again."
                )

            # Select appropriate processor
            processor = self.response_processors.get(
                response_type, self._default_processor
            )

            # Process the response
            processed = processor(raw_response, context or {})

            # Validate and clean the response
            cleaned = self._clean_response(processed)

            # Apply final formatting
            formatted = self._apply_final_formatting(cleaned)

            logger.debug(
                f"Processed response of {len(raw_response)} chars to {len(formatted)} chars"
            )
            return formatted

        except Exception as e:
            logger.error(f"Failed to process response: {e}")
            return raw_response  # Return original if processing fails

    def extract_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from a response.

        Args:
            response: Response text that may contain tool calls

        Returns:
            List of tool call dictionaries
        """
        try:
            tool_calls = []

            # Look for JSON-formatted tool calls
            json_pattern = r"```json\\s*({.*?})\\s*```"
            json_matches = re.findall(json_pattern, response, re.DOTALL)

            for match in json_matches:
                try:
                    tool_data = json.loads(match)
                    if self._is_valid_tool_call(tool_data):
                        tool_calls.append(tool_data)
                except json.JSONDecodeError:
                    continue

            # Look for function-style calls
            func_pattern = r"(\\w+)\\(([^)]+)\\)"
            func_matches = re.findall(func_pattern, response)

            for func_name, args_str in func_matches:
                try:
                    # Simple parsing for function arguments
                    args = self._parse_function_args(args_str)
                    tool_calls.append({"function": func_name, "arguments": args})
                except Exception:
                    continue

            return tool_calls

        except Exception as e:
            logger.error(f"Failed to extract tool calls: {e}")
            return []

    def validate_response_quality(
        self, response: str, query: str, min_length: int = 10, max_length: int = 5000
    ) -> Tuple[bool, List[str]]:
        """
        Validate response quality and provide feedback.

        Args:
            response: The response to validate
            query: The original query
            min_length: Minimum acceptable response length
            max_length: Maximum acceptable response length

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        try:
            # Length validation
            if len(response) < min_length:
                issues.append(f"Response too short (< {min_length} chars)")

            if len(response) > max_length:
                issues.append(f"Response too long (> {max_length} chars)")

            # Content validation
            if not response.strip():
                issues.append("Response is empty or only whitespace")

            # Check for common error patterns
            error_patterns = [
                r"error.*occurred",
                r"something went wrong",
                r"unable to.*process",
                r"failed to.*",
                r"exception.*",
            ]

            response_lower = response.lower()
            for pattern in error_patterns:
                if re.search(pattern, response_lower):
                    issues.append(f"Response contains error pattern: {pattern}")

            # Check for relevance (simple heuristics)
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())
            common_words = query_words.intersection(response_words)

            if len(common_words) == 0 and len(query_words) > 2:
                issues.append("Response may not be relevant to the query")

            return len(issues) == 0, issues

        except Exception as e:
            logger.error(f"Failed to validate response quality: {e}")
            return True, []  # Assume valid if validation fails

    def _default_processor(self, response: str, context: Dict[str, Any]) -> str:
        """Default response processor."""
        return response.strip()

    def _json_processor(self, response: str, context: Dict[str, Any]) -> str:
        """Process JSON responses."""
        try:
            # Extract JSON from response if embedded
            json_pattern = r"```json\\s*({.*?})\\s*```"
            json_match = re.search(json_pattern, response, re.DOTALL)

            if json_match:
                json_str = json_match.group(1)
                parsed = json.loads(json_str)
                return json.dumps(parsed, indent=2)
            else:
                # Try to parse entire response as JSON
                parsed = json.loads(response)
                return json.dumps(parsed, indent=2)

        except json.JSONDecodeError:
            # Return original if not valid JSON
            return response.strip()

    def _markdown_processor(self, response: str, context: Dict[str, Any]) -> str:
        """Process markdown responses."""
        # Basic markdown formatting improvements
        processed = response.strip()

        # Ensure proper heading spacing
        processed = re.sub(r"(#+\\s+.*?)\\n(?!\\n)", r"\\1\\n\\n", processed)

        # Ensure proper list spacing
        processed = re.sub(r"\\n(-|\\*|\\d+\\.)\\s", r"\\n\\n\\1 ", processed)

        return processed

    def _clean_response(self, response: str) -> str:
        """Clean and sanitize the response."""
        try:
            # Remove excessive whitespace
            cleaned = re.sub(r"\\n\\s*\\n\\s*\\n+", "\\n\\n", response)
            cleaned = re.sub(r"  +", " ", cleaned)  # Multiple spaces to single

            # Remove trailing whitespace from lines
            cleaned = "\\n".join(line.rstrip() for line in cleaned.split("\\n"))

            # Ensure response doesn't start/end with whitespace
            cleaned = cleaned.strip()

            return cleaned

        except Exception as e:
            logger.warning(f"Failed to clean response: {e}")
            return response

    def _apply_final_formatting(self, response: str) -> str:
        """Apply final formatting touches."""
        try:
            # Ensure proper sentence spacing
            formatted = re.sub(r"([.!?])([A-Z])", r"\\1 \\2", response)

            # Fix common punctuation issues
            formatted = re.sub(r"\\s+([,.!?;:])", r"\\1", formatted)

            return formatted

        except Exception as e:
            logger.warning(f"Failed to apply final formatting: {e}")
            return response

    def _is_valid_tool_call(self, tool_data: Dict[str, Any]) -> bool:
        """Check if tool data represents a valid tool call."""
        required_fields = ["function", "arguments"]
        return all(field in tool_data for field in required_fields)

    def _parse_function_args(self, args_str: str) -> Dict[str, Any]:
        """Parse function arguments from string."""
        try:
            # Simple parsing - assumes key=value format
            args = {}

            # Split by comma and parse each part
            parts = [part.strip() for part in args_str.split(",")]

            for part in parts:
                if "=" in part:
                    key, value = part.split("=", 1)
                    key = key.strip().strip("'\"")
                    value = value.strip().strip("'\"")

                    # Try to convert to appropriate type
                    if value.lower() in ["true", "false"]:
                        args[key] = value.lower() == "true"
                    elif value.isdigit():
                        args[key] = int(value)
                    elif "." in value and value.replace(".", "").isdigit():
                        args[key] = float(value)
                    else:
                        args[key] = value

            return args

        except Exception:
            return {"raw_args": args_str}

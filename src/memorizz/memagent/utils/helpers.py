# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Helper utilities for MemAgent."""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class IDGenerator:
    """Generates various types of IDs for MemAgent components."""

    @staticmethod
    def generate_agent_id(prefix: str = "agent") -> str:
        """
        Generate a unique agent ID.

        Args:
            prefix: Optional prefix for the ID

        Returns:
            Unique agent ID
        """
        unique_part = str(uuid.uuid4()).replace("-", "")[:12]
        return f"{prefix}_{unique_part}"

    @staticmethod
    def generate_conversation_id(prefix: str = "conv") -> str:
        """
        Generate a unique conversation ID.

        Args:
            prefix: Optional prefix for the ID

        Returns:
            Unique conversation ID
        """
        unique_part = str(uuid.uuid4()).replace("-", "")[:16]
        return f"{prefix}_{unique_part}"

    @staticmethod
    def generate_memory_id(prefix: str = "mem") -> str:
        """
        Generate a unique memory ID.

        Args:
            prefix: Optional prefix for the ID

        Returns:
            Unique memory ID
        """
        unique_part = str(uuid.uuid4()).replace("-", "")[:14]
        return f"{prefix}_{unique_part}"

    @staticmethod
    def generate_tool_id(tool_name: str) -> str:
        """
        Generate a unique tool ID based on tool name.

        Args:
            tool_name: Name of the tool

        Returns:
            Unique tool ID
        """
        # Create a hash of the tool name for consistency
        name_hash = hashlib.md5(tool_name.encode()).hexdigest()[:8]
        return f"tool_{tool_name}_{name_hash}"

    @staticmethod
    def generate_session_id() -> str:
        """
        Generate a unique session ID.

        Returns:
            Unique session ID
        """
        return str(uuid.uuid4())

    @staticmethod
    def is_valid_uuid(id_string: str) -> bool:
        """
        Check if a string is a valid UUID.

        Args:
            id_string: String to check

        Returns:
            True if valid UUID, False otherwise
        """
        try:
            uuid.UUID(id_string)
            return True
        except ValueError:
            return False


class TimestampHelper:
    """Helper for timestamp operations."""

    @staticmethod
    def now_iso() -> str:
        """
        Get current timestamp in ISO format.

        Returns:
            Current timestamp as ISO string
        """
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def now_unix() -> float:
        """
        Get current timestamp as Unix timestamp.

        Returns:
            Current timestamp as float
        """
        return datetime.now(timezone.utc).timestamp()

    @staticmethod
    def parse_iso(iso_string: str) -> datetime:
        """
        Parse ISO timestamp string to datetime.

        Args:
            iso_string: ISO timestamp string

        Returns:
            Parsed datetime object
        """
        try:
            return datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
        except ValueError:
            # Fallback parsing
            return datetime.fromisoformat(iso_string)

    @staticmethod
    def format_duration(
        start_time: datetime, end_time: Optional[datetime] = None
    ) -> str:
        """
        Format duration between two timestamps.

        Args:
            start_time: Start timestamp
            end_time: End timestamp (defaults to now)

        Returns:
            Formatted duration string
        """
        if end_time is None:
            end_time = datetime.now(timezone.utc)

        duration = end_time - start_time
        total_seconds = duration.total_seconds()

        if total_seconds < 60:
            return f"{total_seconds:.1f} seconds"
        elif total_seconds < 3600:
            minutes = total_seconds / 60
            return f"{minutes:.1f} minutes"
        else:
            hours = total_seconds / 3600
            return f"{hours:.1f} hours"

    @staticmethod
    def is_recent(timestamp: datetime, minutes_ago: int = 30) -> bool:
        """
        Check if timestamp is within recent time window.

        Args:
            timestamp: Timestamp to check
            minutes_ago: Time window in minutes

        Returns:
            True if timestamp is recent, False otherwise
        """
        now = datetime.now(timezone.utc)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        time_diff = now - timestamp
        return time_diff.total_seconds() <= (minutes_ago * 60)


class DataHelper:
    """Helper for data manipulation and processing."""

    @staticmethod
    def deep_merge_dicts(
        dict1: Dict[str, Any], dict2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deeply merge two dictionaries.

        Args:
            dict1: First dictionary
            dict2: Second dictionary (takes precedence)

        Returns:
            Merged dictionary
        """
        result = dict1.copy()

        for key, value in dict2.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = DataHelper.deep_merge_dicts(result[key], value)
            else:
                result[key] = value

        return result

    @staticmethod
    def extract_keys_recursive(
        data: Dict[str, Any], target_keys: List[str]
    ) -> Dict[str, Any]:
        """
        Extract specified keys recursively from nested dictionary.

        Args:
            data: Dictionary to search
            target_keys: Keys to extract

        Returns:
            Dictionary containing found keys and values
        """
        result = {}

        def _extract_recursive(obj: Any, path: str = ""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    if key in target_keys:
                        result[current_path] = value
                    _extract_recursive(value, current_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    _extract_recursive(item, f"{path}[{i}]")

        _extract_recursive(data)
        return result

    @staticmethod
    def sanitize_for_json(obj: Any) -> Any:
        """
        Sanitize object for JSON serialization.

        Args:
            obj: Object to sanitize

        Returns:
            JSON-serializable object
        """
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: DataHelper.sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [DataHelper.sanitize_for_json(item) for item in obj]
        else:
            try:
                return str(obj)
            except Exception:
                return f"<{type(obj).__name__} object>"

    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """
        Calculate simple similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0

        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    @staticmethod
    def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
        """
        Truncate text to maximum length.

        Args:
            text: Text to truncate
            max_length: Maximum length
            suffix: Suffix to add if truncated

        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text

        return text[: max_length - len(suffix)] + suffix

    @staticmethod
    def hash_object(obj: Any) -> str:
        """
        Create hash of an object.

        Args:
            obj: Object to hash

        Returns:
            Hash string
        """
        try:
            # Convert to JSON for consistent hashing
            json_str = json.dumps(DataHelper.sanitize_for_json(obj), sort_keys=True)
            return hashlib.md5(json_str.encode()).hexdigest()
        except Exception:
            return hashlib.md5(str(obj).encode()).hexdigest()


class ErrorHelper:
    """Helper for error handling and logging."""

    @staticmethod
    def log_error_with_context(
        error: Exception,
        context: Dict[str, Any],
        logger_instance: Optional[logging.Logger] = None,
    ):
        """
        Log error with additional context.

        Args:
            error: Exception that occurred
            context: Additional context information
            logger_instance: Optional specific logger to use
        """
        log = logger_instance or logger

        context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
        log.error(f"Error: {str(error)} | Context: {context_str}")

    @staticmethod
    def create_error_context(
        operation: str,
        agent_id: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create error context dictionary.

        Args:
            operation: Operation that failed
            agent_id: Optional agent ID
            query: Optional query being processed
            **kwargs: Additional context

        Returns:
            Error context dictionary
        """
        context = {"operation": operation, "timestamp": TimestampHelper.now_iso()}

        if agent_id:
            context["agent_id"] = agent_id

        if query:
            context["query"] = DataHelper.truncate_text(query, 200)

        context.update(kwargs)
        return context

    @staticmethod
    def safe_call(func, default_value=None, log_errors: bool = True):
        """
        Safely call a function with error handling.

        Args:
            func: Function to call
            default_value: Value to return on error
            log_errors: Whether to log errors

        Returns:
            Function result or default value
        """
        try:
            return func()
        except Exception as e:
            if log_errors:
                logger.warning(f"Safe call failed: {str(e)}")
            return default_value

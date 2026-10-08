"""Small, content-free vector records for composed memory providers.

The document provider remains authoritative. These records contain embeddings,
opaque source references, and retrieval scope, never a second copy of memory text.
:func:`cosine` is the one similarity used by every brute-force ranking.
"""

import hashlib
import json
import math
import uuid
from typing import Any, Dict, List, Optional

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

VECTOR_MARKER = "memorizz.vector.v1"
_MISMATCH_POLICIES = frozenset({"none", "zero", "raise"})
# numpy wins only once a vector is big enough for its per-call overhead to pay.
_NUMPY_FROM_DIMENSIONS = 64


def _cosine_or_none(a: Any, b: Any) -> Optional[float]:
    if np is not None and (
        isinstance(a, np.ndarray)
        or isinstance(b, np.ndarray)
        or len(a) > _NUMPY_FROM_DIMENSIONS
    ):
        vec_a = np.asarray(a, dtype="float64")
        vec_b = np.asarray(b, dtype="float64")
        if vec_a.ndim != 1 or vec_a.shape != vec_b.shape or vec_a.size == 0:
            return None
        norm_a = float(np.linalg.norm(vec_a))
        norm_b = float(np.linalg.norm(vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return None
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    if len(a) == 0 or len(a) != len(b):
        return None
    dot = norm_a = norm_b = 0.0
    for x, y in zip(a, b):
        x = float(x)
        y = float(y)
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return None
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def cosine(a: Any, b: Any, *, on_mismatch: str = "none") -> Optional[float]:
    """Cosine similarity of two 1-D vectors, or the mismatch policy's result.

    A mismatch is an empty vector, two vectors of different length, a zero
    vector or non-numeric content. ``on_mismatch`` decides what comes back
    then: ``"none"`` returns ``None``, ``"zero"`` returns ``0.0`` and
    ``"raise"`` raises ``ValueError``. Pure Python for small vectors; numpy,
    when installed, for larger ones and for array inputs.
    """
    if on_mismatch not in _MISMATCH_POLICIES:
        raise ValueError(f"Unknown on_mismatch policy: {on_mismatch!r}")
    try:
        result = _cosine_or_none(a, b)
    except (TypeError, ValueError):
        result = None
    if result is not None:
        return result
    if on_mismatch == "zero":
        return 0.0
    if on_mismatch == "raise":
        raise ValueError(
            "Cosine similarity needs two non-empty, non-zero numeric vectors of "
            "the same length"
        )
    return None


def flatten_vector(vector: Any) -> Any:
    """``vector`` as one flat sequence, so a nested ``[[...]]`` embedding still scores."""
    if np is not None:
        try:
            return np.asarray(vector, dtype="float64").ravel()
        except (TypeError, ValueError):
            pass
    flat: List[Any] = []
    pending = [vector]
    while pending:
        item = pending.pop()
        if isinstance(item, (list, tuple)):
            pending.extend(reversed(item))
        else:
            flat.append(item)
    return flat


VECTOR_SCOPE_FIELDS = frozenset(
    {
        "user_id",
        "memory_id",
        "agent_id",
        "thread_id",
        "namespace",
        "status",
        "application_id",
        "session_id",
    }
)


def validate_vector(vector: Any) -> List[float]:
    if not isinstance(vector, (list, tuple)) or not vector:
        raise ValueError("An embedding must be a non-empty sequence of finite numbers")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in vector
    ):
        raise ValueError("An embedding must contain only finite numbers")
    result = [float(value) for value in vector]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("An embedding must contain only finite numbers")
    if not any(result):
        raise ValueError("A zero embedding cannot be used for cosine similarity")
    return result


def validate_scope(scope: Dict[str, Any]) -> Dict[str, Any]:
    if set(scope) - VECTOR_SCOPE_FIELDS:
        raise ValueError("Unsupported vector scope fields")
    for value in scope.values():
        values = value if isinstance(value, list) else [value]
        if not values or any(
            item is not None and not isinstance(item, str) for item in values
        ):
            raise ValueError(
                "Vector scope values must be strings, null, or non-empty lists"
            )
    return dict(scope)


def scope_matches(metadata: Dict[str, Any], scope: Dict[str, Any]) -> bool:
    actual = metadata.get("scope") or {}
    return all(
        actual.get(key) in value
        if isinstance(value, list)
        else actual.get(key) == value
        for key, value in scope.items()
    )


def vector_id(namespace: str, source_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, namespace + "\x00" + source_id))


def scope_hash(value):
    """Portable exact/null/empty-string predicates, including SQL backends."""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def vector_document(namespace, source_id, embedding, metadata, scope):
    if not isinstance(namespace, str) or not namespace or len(namespace) > 255:
        raise ValueError("Vector namespace must contain 1–255 characters")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("Vector source_id must be a non-empty string")
    identifier = vector_id(namespace, source_id)
    return {
        "_id": identifier,
        "id": identifier,
        "namespace": namespace,
        "source_id": source_id,
        # Some relational providers require non-null content. This is a marker,
        # not the source text, and is deliberately unsuitable for keyword recall.
        "content": VECTOR_MARKER,
        "memory_type": VECTOR_MARKER,
        "embedding": validate_vector(embedding),
        "metadata": {
            "format": VECTOR_MARKER,
            "source_id": source_id,
            "reference": dict(metadata),
            "scope": validate_scope(scope),
            "scope_hashes": {
                key: scope_hash(scope.get(key)) for key in VECTOR_SCOPE_FIELDS
            },
        },
    }


def vector_hit(row, *, include_embedding=False):
    metadata = row.get("metadata") or {}
    if metadata.get("format") != VECTOR_MARKER:
        raise ValueError("Invalid semantic index record")
    result = {
        "source_id": metadata["source_id"],
        "metadata": dict(metadata.get("reference") or {}),
        "scope": dict(metadata.get("scope") or {}),
        "score": float(row["score"]),
    }
    if include_embedding and row.get("embedding") is not None:
        result["embedding"] = list(row["embedding"])
    return result

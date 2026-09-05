"""Canonical bounded reference parsing shared by capture, replay and inspection."""

import json

from .models import ResourceRef, SelectionDecision
from .privacy import validate_opaque


def source_ids(value):
    """Decode legacy JSON arrays once; never interpret a string as characters."""
    if value is None:
        return []
    if isinstance(value, str):
        if len(value) > 32768:
            raise ValueError("Source array exceeds the metadata limit")
        value = json.loads(value)
    if not isinstance(value, list) or len(value) > 128:
        raise ValueError("Source IDs must be a bounded array")
    for item in value:
        if not isinstance(item, str) or not item or len(item) > 240:
            raise ValueError("Source IDs must be nonempty bounded strings")
        validate_opaque(item)
    return list(dict.fromkeys(value))


def event_resource_refs(event):
    """Validate all replay references before any authorization callback runs."""
    refs = []
    for field in ("input_refs", "output_refs"):
        values = event.get(field, [])
        if not isinstance(values, list) or len(values) > 32:
            raise ValueError("Invalid resource array")
        refs.extend(ResourceRef.model_validate(value) for value in values)
    refs.extend(
        ResourceRef(resource_type="analysis", ref=value)
        for value in source_ids(event.get("grounding_source_ids"))
    )
    ledger = event.get("selection_ledger", [])
    if not isinstance(ledger, list) or len(ledger) > 64:
        raise ValueError("Invalid selection ledger")
    refs.extend(SelectionDecision.model_validate(value).resource for value in ledger)
    return [ref.model_dump(mode="json", exclude_none=True) for ref in refs]

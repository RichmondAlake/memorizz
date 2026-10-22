# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Entity capture reads a long message in parts, so a long list of facts
isn't cut short at the first 4,000 characters."""

import json
import re

import pytest

from memorizz.memagent.memory_routing import (
    CAPTURE_MAX_PARTS,
    CAPTURE_PART_CHARS,
    extract_facts_in_parts,
    text_parts,
)

LINE = re.compile(r"The capital of (Country\d+) is (City\d+)")


def _facts(count, *, start=1):
    return "\n".join(
        f"{n}. The capital of Country{n} is City{n} and it is a long sentence "
        "written to pad the line out."
        for n in range(start, start + count)
    )


def _model(seen):
    """A background model that reports each numbered capital in the text."""

    def generate(prompt):
        text = prompt.rsplit('"""', 2)[-2] if '"""' in prompt else prompt
        seen.append(len(text))
        facts = [
            {
                "entity": country,
                "attribute": "capital",
                "value": city,
                "evidence": f"The capital of {country} is {city}",
            }
            for country, city in LINE.findall(text)
        ]
        return json.dumps({"facts": facts})

    return generate


@pytest.mark.unit
def test_a_long_message_is_read_in_parts_cut_at_line_ends():
    message = _facts(160)
    parts, unread = text_parts(message)
    assert len(message) > 3 * CAPTURE_PART_CHARS and unread == 0
    assert all(len(part) <= CAPTURE_PART_CHARS for part in parts)
    assert all(
        re.match(r"\d+\. The capital", part) for part in parts
    ), "each part starts at a fact"
    seen = []
    facts, read = extract_facts_in_parts(message, _model(seen), [])
    assert len(seen) == len(parts) == read["parts"]
    assert {fact["entity"] for fact in facts} == {f"Country{n}" for n in range(1, 161)}


@pytest.mark.unit
def test_a_later_part_corrects_an_earlier_one_and_overflow_is_counted():
    correction = "\n160. Correction: The capital of Country1 is City999 now."
    message = _facts(159) + correction
    facts, _read = extract_facts_in_parts(message, _model([]), [])
    capitals = {fact["entity"]: fact["value"] for fact in facts}
    assert capitals["Country1"] == "City999", "the later line wins"

    huge = _facts(60 * CAPTURE_MAX_PARTS)
    parts, unread = text_parts(huge)
    assert len(parts) == CAPTURE_MAX_PARTS and unread > 0
    _facts_found, read = extract_facts_in_parts(huge, _model([]), [])
    assert read == {"parts": CAPTURE_MAX_PARTS, "unread_chars": unread}

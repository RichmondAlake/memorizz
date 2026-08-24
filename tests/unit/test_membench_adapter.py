from memorizz.benchmarks.membench import _flatten_messages, _parse_choice


def test_participation_layout_preserves_official_source_pairs():
    messages, mapping = _flatten_messages(
        [
            [
                {
                    "sid": 7,
                    "user_message": "My niece works at Acme.",
                    "assistant_message": "Noted.",
                }
            ]
        ],
        perspective="participation",
    )

    assert messages[0]["source_key"] == "7:0"
    assert mapping[(7, 0)] == "7:0"
    assert "Acme" in messages[0]["text"]


def test_observation_layout_preserves_flat_ids():
    messages, _ = _flatten_messages(
        [{"mid": 10, "message": "The subordinate has an Associate Degree."}],
        perspective="observation",
    )

    assert messages[0]["source_key"] == "10"


def test_choice_parser_accepts_strict_json_and_boxed_output():
    assert _parse_choice('{"answer":"B"}') == "B"
    assert _parse_choice("\\boxed{D}") == "D"
    assert _parse_choice("No supported choice") is None

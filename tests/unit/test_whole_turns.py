"""A recalled conversation message comes with the other half of its turn."""

from pathlib import Path

from memorizz.memagent.utils.whole_turns import whole_turns

SAM, COLE = "agent-sam", "agent-cole"


def _row(_id, thread, role, text, agent=COLE, score=None, importance=None):
    row = {
        "_id": _id,
        "memory_id": "team",
        "thread_id": thread,
        "role": role,
        "content": text,
        "agent_id": agent,
        "embedding": [float(len(_id)), 1.0],
    }
    if score is not None:
        row["score"] = score
    if importance is not None:
        row["importance"] = importance
    return row


THREADS = {
    "write": [
        _row("w1", "write", "user", "Write the announcement email."),
        _row("w2", "write", "assistant", "Subject: A little spring refresh"),
    ],
    "lead": [
        _row("l1", "lead", "user", "Stage 2. Write the announcement email."),
        _row("l2", "lead", "assistant", "Subject: A little spring refresh"),
    ],
    "ask-1": [
        _row("a1", "ask-1", "user", "What was the email's subject line?", SAM),
        _row("a2", "ask-1", "assistant", "I can't see the email.", SAM),
    ],
    "ask-2": [
        _row("b1", "ask-2", "user", "Has anyone written the email?", SAM),
        _row("b2", "ask-2", "assistant", "I can't find it.", SAM),
    ],
    "open": [_row("o1", "open", "user", "Anyone there?")],
}


def _reader(calls=None):
    def read(memory_id, thread_id):
        if calls is not None:
            calls.append(thread_id)
        return THREADS[thread_id]

    return read


def _hit(_id, score):
    row = next(r for rows in THREADS.values() for r in rows if r["_id"] == _id)
    return {**row, "score": score}


def test_a_request_brings_the_reply_that_followed_it():
    (turn,) = whole_turns([_hit("w1", 0.66)], _reader())

    assert turn["_id"] == "w2" and turn["role"] == "assistant"
    assert turn["text"] == (
        "Asked: Write the announcement email.\n"
        "Answered: Subject: A little spring refresh"
    )
    assert turn["score"] == 0.66  # ranked by the half that matched
    assert turn["turn"]["request_id"] == "w1"


def test_a_reply_brings_the_request_it_answered():
    (turn,) = whole_turns([_hit("w2", 0.5)], _reader())

    assert turn["text"].startswith("Asked: Write the announcement email.")


def test_two_halves_of_one_turn_are_one_candidate_with_the_better_match():
    out = whole_turns([_hit("w2", 0.5), _hit("w1", 0.66)], _reader())

    assert len(out) == 1 and out[0]["score"] == 0.66


def test_an_exchange_recorded_in_two_threads_is_recalled_once():
    """The lead's thread and the worker's thread both hold the same reply."""
    out = whole_turns([_hit("l1", 0.65), _hit("w1", 0.66)], _reader())

    assert len(out) == 1
    assert out[0]["turn"]["request_id"] == "w1"


def test_a_turn_says_whose_reply_it_was_when_the_reader_is_known():
    theirs, mine = whole_turns(
        [_hit("w1", 0.9), _hit("a1", 0.8)], _reader(), reader_agent_id=SAM
    )

    assert "Answered by another agent: Subject:" in theirs["text"]
    assert "Your own answer at the time" in mine["text"]
    assert mine["turn"]["by_reader"] and not theirs["turn"]["by_reader"]


def test_the_readers_own_attempts_do_not_crowd_out_a_colleagues_turn():
    """Its own "I can't see it" is the closest match to the question asked
    again, and outranked the email it was asking about."""
    out = whole_turns(
        [_hit("a1", 0.87), _hit("b1", 0.80), _hit("w1", 0.66)],
        _reader(),
        reader_agent_id=SAM,
    )

    assert [row["_id"] for row in out] == ["a2", "w2"]


def test_an_agent_alone_in_its_memory_keeps_all_its_turns():
    out = whole_turns(
        [_hit("a1", 0.87), _hit("b1", 0.80)], _reader(), reader_agent_id=SAM
    )

    assert [row["_id"] for row in out] == ["a2", "b2"]


def test_rows_without_a_partner_or_a_thread_pass_through_unchanged():
    loose = {"_id": "x", "role": "user", "content": "no thread", "score": 0.4}
    alone = _hit("o1", 0.3)
    current = _hit("w1", 0.9)

    out = whole_turns([loose, alone, current], _reader(), skip_thread="write")

    assert out == [current, loose, alone]


def test_only_the_best_matches_threads_are_read():
    calls = []
    out = whole_turns(
        [_hit("w1", 0.9), _hit("a1", 0.8), _hit("b1", 0.7)],
        _reader(calls),
        max_threads=2,
    )

    assert calls == ["write", "ask-1"]
    assert [row["_id"] for row in out] == ["w2", "a2", "b1"]


def test_a_thread_that_cannot_be_read_leaves_its_rows_as_they_came():
    def broken(memory_id, thread_id):
        raise RuntimeError("store is away")

    hit = _hit("w1", 0.9)
    assert whole_turns([hit], broken) == [hit]


def test_an_agent_recalls_the_reply_to_a_request_in_another_thread(tmp_path: Path):
    """End to end on the file store: the question matches the request to
    write the email, which shares no word with the email, and the email
    comes with it."""
    from memorizz.enums import MemoryType
    from memorizz.memagent import MemAgent
    from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", embedding_provider=None)
    )
    for second, role, text in (
        (1, "user", "Write the Spring Sale announcement email with a subject line."),
        (2, "assistant", "Ferns: 20% off, little refresh."),
    ):
        provider.store(
            {
                "memory_id": "team",
                "thread_id": "write",
                "role": role,
                "content": text,
                "agent_id": COLE,
                "timestamp": f"2026-11-27T10:00:0{second}+00:00",
            },
            MemoryType.CONVERSATION_MEMORY,
        )

    class Model:
        model = "fake"

        def __init__(self):
            self.prompts = []

        def get_config(self):
            return {"provider": "openai", "model": "fake"}

        def get_context_window_tokens(self):
            return 32768

        def get_last_usage(self):
            return {"input_tokens": 1, "output_tokens": 1}

        def generate(self, messages, tools=None, **kwargs):
            self.prompts.append("\n".join(str(m.get("content")) for m in messages))
            return "ok"

        def generate_stream(self, messages, tools=None, **kwargs):
            text = self.generate(messages, tools=tools)
            yield {"type": "content", "content": text}
            yield {"type": "done", "content": text}

    model = Model()
    agent = MemAgent(model=model, memory_provider=provider, auto_register=False)
    agent.run(
        "Which subject line did the Spring Sale announcement email get?",
        memory_id="team",
        thread_id="ask",
    )

    (recalled,) = agent._last_retrieved_memories
    assert recalled["text"] == (
        "Asked: Write the Spring Sale announcement email with a subject line.\n"
        "Answered by another agent: Ferns: 20% off, little refresh."
    )
    assert any("little refresh" in prompt for prompt in model.prompts), model.prompts


def test_a_search_result_keeps_its_record_and_gains_its_partner():
    from memorizz.memagent.utils.whole_turns import with_turn_partners

    request, reply = _hit("w1", 0.66), _hit("a2", 0.5)

    found, answered = with_turn_partners([request, reply], _reader())

    assert found["_id"] == "w1" and found["content"] == request["content"]
    assert found["turn"] == {
        "reply": {
            "id": "w2",
            "content": "Subject: A little spring refresh",
            "agent_id": COLE,
        }
    }
    assert answered["_id"] == "a2"
    assert answered["turn"] == {
        "request": {"id": "a1", "content": "What was the email's subject line?"}
    }


def test_a_turn_found_by_both_halves_is_listed_once_and_order_is_kept():
    from memorizz.memagent.utils.whole_turns import with_turn_partners

    rows = [_hit("a1", 0.9), _hit("w2", 0.8), _hit("w1", 0.7), _hit("o1", 0.1)]

    out = with_turn_partners(rows, _reader())

    assert [row["_id"] for row in out] == ["a1", "w2", "o1"]
    assert "turn" not in out[2]


def test_a_partner_the_caller_may_not_see_is_left_out():
    from memorizz.memagent.utils.whole_turns import with_turn_partners

    (row,) = with_turn_partners(
        [_hit("w1", 0.7)], _reader(), visible=lambda partner: False
    )

    assert row["_id"] == "w1" and "turn" not in row

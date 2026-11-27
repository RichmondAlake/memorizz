"""A live call with several agents: whose voice says what, and in what order."""

from __future__ import annotations

import pytest

from memorizz.voice.config import VoiceConfig
from memorizz.voice.providers.livekit import (
    GROUP_VOICES,
    LiveKitVoice,
    mark_speaker,
    split_speakers,
)


@pytest.mark.unit
def test_a_rounds_text_is_split_by_speaker_and_keeps_what_is_unfinished():
    text = mark_speaker("vera", "Four.") + mark_speaker("ada", "I agree.\nFour.")

    assert split_speakers(text) == ([("vera", "Four."), ("ada", "I agree.\nFour.")], "")
    # A part still being written waits for its end.
    parts, rest = split_speakers(text + "⟦flint⟧Fou")
    assert len(parts) == 2 and rest == "⟦flint⟧Fou"
    assert split_speakers("no marks at all") == ([], "no marks at all")


@pytest.mark.unit
def test_each_speaker_gets_its_own_voice_and_no_two_share_one():
    own = VoiceConfig(provider="livekit", voice=GROUP_VOICES[0][1])
    elsewhere = VoiceConfig(provider="openai", voice="nova")
    speakers = [
        {"id": "a", "voice_config": own},
        {"id": "b", "voice_config": None},
        {"id": "c", "voice_config": elsewhere},
        {"id": "d"},
    ]

    voices = LiveKitVoice.group_voices(speakers)

    assert voices["a"] == GROUP_VOICES[0]
    # A voice on another provider cannot join LiveKit's stream: one is lent.
    assert len(set(voices.values())) == 4
    assert GROUP_VOICES[0] not in (voices["b"], voices["c"], voices["d"])
    # More speakers than lent voices: they are shared in turn, never missing.
    many = LiveKitVoice.group_voices(
        [{"id": str(n)} for n in range(len(GROUP_VOICES) + 2)]
    )
    assert len(many) == len(GROUP_VOICES) + 2

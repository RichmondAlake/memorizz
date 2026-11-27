"""LiveKit: the default voice provider.

Two ways to talk:

* turn by turn, through LiveKit Inference (speech in, text out; text in,
  speech out) with only the LiveKit Cloud credentials;
* a live call: the browser joins a LiveKit room and this process joins it too
  as the agent, running LiveKit's voice session (voice activity detection,
  turn taking, interruptions) with the MemoRizz agent as its language model.

LiveKit Inference is a LiveKit Cloud service; a self-hosted LiveKit server
carries the room but has no speech models.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import threading
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from ..base import Brain, VoiceAudio, VoiceError, VoiceProvider, speakable
from ..config import VoiceConfig

logger = logging.getLogger(__name__)

DEFAULT_TTS_MODEL = "cartesia/sonic-3"
# A voice LiveKit's model page suggests for cartesia/sonic-3.
DEFAULT_TTS_VOICE = "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
DEFAULT_STT_MODEL = "assemblyai/universal-3-6-pro"
STT_SAMPLE_RATE = 16000
#: A live call is closed after this long even if the browser never leaves.
MAX_CALL_SECONDS = 30 * 60
#: Every voice in a group call is asked for at this rate, so one stream of
#: audio can carry them all.
CALL_SAMPLE_RATE = 24000
#: Voices lent, in turn, to agents in a group call that have none of their
#: own on LiveKit. From LiveKit's suggested voices for these models.
GROUP_VOICES = (
    ("cartesia/sonic-3", "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),  # Jacqueline
    ("cartesia/sonic-3", "a167e0f3-df7e-4d52-a9c3-f949145efdab"),  # Blake
    ("deepgram/aura-2", "athena"),
    ("deepgram/aura-2", "odysseus"),
    ("cartesia/sonic-3", "f31cc6a7-c1e8-4764-980c-60a361443dd1"),  # Robyn
    ("deepgram/aura-2", "apollo"),
)
#: How one speaker's words are marked in the text a group call's round
#: yields, so the speech step knows whose voice says them.
_SEGMENT = re.compile(r"⟦([^⟧]+)⟧(.*?)⟦/⟧", re.DOTALL)


class _Loop:
    """One background event loop for every LiveKit coroutine in the process."""

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = threading.Lock()

    def get(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None or self._loop.is_closed():
                loop = asyncio.new_event_loop()
                thread = threading.Thread(
                    target=loop.run_forever, name="memorizz-livekit", daemon=True
                )
                thread.start()
                self._loop = loop
            return self._loop

    def run(self, coroutine: Any, timeout: float = 90.0) -> Any:
        future = asyncio.run_coroutine_threadsafe(coroutine, self.get())
        try:
            return future.result(timeout)
        except TimeoutError:
            future.cancel()
            raise VoiceError("LiveKit did not answer in time", code="voice_timeout")


_loop = _Loop()
# room name -> the coroutine that ends the call (run on the LiveKit loop).
_calls: Dict[str, Any] = {}
_calls_lock = threading.Lock()


def split_model(
    descriptor: Optional[str], default_model: str
) -> Tuple[str, Optional[str]]:
    """``"provider/model[:suffix]"`` -> (model, suffix). The suffix is a voice
    for speech models and a language for transcription models."""
    text = (descriptor or default_model).strip()
    model, _, suffix = text.partition(":")
    return model or default_model, suffix or None


def decode_to_pcm(audio: bytes, sample_rate: int = STT_SAMPLE_RATE) -> bytes:
    """A browser recording (webm, ogg, mp4, wav) as mono 16-bit PCM."""
    import av

    pcm = bytearray()
    try:
        with av.open(io.BytesIO(audio)) as container:
            resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
            for frame in container.decode(audio=0):
                for converted in resampler.resample(frame):
                    pcm.extend(converted.to_ndarray().tobytes())
            for converted in resampler.resample(None):
                pcm.extend(converted.to_ndarray().tobytes())
    except VoiceError:
        raise
    except Exception as exc:
        raise VoiceError(
            f"The recording could not be decoded ({type(exc).__name__})",
            code="voice_bad_audio",
        ) from exc
    return bytes(pcm)


class LiveKitVoice(VoiceProvider):
    name = "livekit"
    label = "LiveKit"
    supports_tts = True
    supports_stt = True
    supports_realtime = True
    required_env = ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
    required_modules = ("livekit.agents", "livekit.api", "av")
    extra = "voice"

    def __init__(self, config: Optional[VoiceConfig] = None) -> None:
        super().__init__(config)

    # ---- models ----------------------------------------------------------

    def _tts(self, http_session: Any) -> Any:
        from livekit.agents import inference

        model, voice = split_model(self.config.tts_model, DEFAULT_TTS_MODEL)
        voice = self.config.voice or voice
        if voice is None and model == DEFAULT_TTS_MODEL:
            voice = DEFAULT_TTS_VOICE
        kwargs: Dict[str, Any] = {"http_session": http_session}
        if voice:
            kwargs["voice"] = voice
        if self.config.language:
            kwargs["language"] = self.config.language
        return inference.TTS(model, **kwargs)

    def _stt(self, http_session: Any) -> Any:
        from livekit.agents import inference

        model, language = split_model(self.config.stt_model, DEFAULT_STT_MODEL)
        kwargs: Dict[str, Any] = {"http_session": http_session}
        language = self.config.language or language
        if language:
            kwargs["language"] = language
        return inference.STT(model, **kwargs)

    # ---- turn by turn ----------------------------------------------------

    def synthesize(self, text: str) -> VoiceAudio:
        self.require_ready()
        spoken = speakable(text)
        if not spoken:
            raise VoiceError("Nothing to say", code="voice_empty_text")
        return _loop.run(self._synthesize(spoken))

    async def _synthesize(self, text: str) -> VoiceAudio:
        import aiohttp

        async with aiohttp.ClientSession() as http:
            tts = self._tts(http)
            try:
                frame = await tts.synthesize(text).collect()
            except Exception as exc:
                raise _provider_error(exc) from exc
            finally:
                await tts.aclose()
        return VoiceAudio(frame.to_wav_bytes(), "audio/wav")

    def transcribe(self, audio: bytes, mime_type: str = "audio/webm") -> str:
        self.require_ready()
        if not audio:
            raise VoiceError("No audio was recorded", code="voice_empty_audio")
        pcm = decode_to_pcm(audio)
        if not pcm:
            return ""
        return _loop.run(self._transcribe(pcm))

    async def _transcribe(self, pcm: bytes) -> str:
        import aiohttp
        from livekit import rtc
        from livekit.agents import stt as stt_module

        # LiveKit Inference transcribes a stream only: feed it 100 ms frames.
        step = STT_SAMPLE_RATE // 10 * 2
        parts: List[str] = []
        async with aiohttp.ClientSession() as http:
            stt = self._stt(http)
            stream = stt.stream()
            try:
                for start in range(0, len(pcm), step):
                    chunk = pcm[start : start + step]
                    stream.push_frame(
                        rtc.AudioFrame(chunk, STT_SAMPLE_RATE, 1, len(chunk) // 2)
                    )
                stream.end_input()
                async for event in stream:
                    if (
                        event.type == stt_module.SpeechEventType.FINAL_TRANSCRIPT
                        and event.alternatives
                    ):
                        parts.append(event.alternatives[0].text)
            except Exception as exc:
                raise _provider_error(exc) from exc
            finally:
                await stream.aclose()
                await stt.aclose()
        return " ".join(part.strip() for part in parts if part.strip())

    # ---- live call -------------------------------------------------------

    def realtime_session(
        self,
        *,
        agent_id: str,
        agent_name: str,
        reply: Brain,
        greeting: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.require_ready()
        from livekit import api

        # A new room per call: an agent joins exactly the room it is given.
        room = f"memorizz-{agent_id[:8]}-{uuid.uuid4().hex[:10]}"
        key, secret = os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"]
        visitor_token = (
            api.AccessToken(key, secret)
            .with_identity(f"visitor-{uuid.uuid4().hex[:8]}")
            .with_name("Visitor")
            .with_grants(api.VideoGrants(room_join=True, room=room))
            .to_jwt()
        )
        agent_token = (
            api.AccessToken(key, secret)
            .with_identity(f"agent-{agent_id[:8]}")
            .with_name(agent_name)
            .with_kind("agent")
            .with_grants(api.VideoGrants(room_join=True, room=room, agent=True))
            .to_jwt()
        )
        _loop.run(
            self._start_call(
                room,
                agent_token,
                lambda http: _brain_agent(agent_name, reply),
                greeting,
            ),
            45.0,
        )
        return {
            "provider": self.name,
            "session_id": room,
            "room": room,
            "server_url": os.environ["LIVEKIT_URL"],
            "participant_token": visitor_token,
        }

    def group_session(
        self,
        *,
        speakers: List[Dict[str, Any]],
        round: Callable[[str], Iterable[Tuple[str, str]]],
        name: str = "Everyone",
        greeting: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start one live call with several agents and return what the
        browser needs to join it.

        LiveKit hears the caller and decides when they have finished. Then
        ``round(said)`` is asked for the answers: it yields ``(speaker_id,
        text)`` in the order they are to be spoken, and each is said in that
        speaker's own voice, one after another on a single stream, so two
        agents never talk at once. When the caller speaks, whatever is left
        of the round is dropped.

        ``speakers`` is ``[{"id", "name", "voice_config"}, ...]``; a speaker
        with no LiveKit voice of its own is lent one.
        """
        self.require_ready()
        from livekit import api

        if not speakers:
            raise VoiceError("Nobody is here to call", code="voice_empty_call")
        room = f"memorizz-group-{uuid.uuid4().hex[:12]}"
        key, secret = os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"]
        visitor_token = (
            api.AccessToken(key, secret)
            .with_identity(f"visitor-{uuid.uuid4().hex[:8]}")
            .with_name("Visitor")
            .with_grants(api.VideoGrants(room_join=True, room=room))
            .to_jwt()
        )
        agent_token = (
            api.AccessToken(key, secret)
            .with_identity(f"agents-{uuid.uuid4().hex[:8]}")
            .with_name(name)
            .with_kind("agent")
            .with_grants(api.VideoGrants(room_join=True, room=room, agent=True))
            .to_jwt()
        )
        voices = self.group_voices(speakers)
        _loop.run(
            self._start_call(
                room,
                agent_token,
                lambda http: _group_agent(
                    name,
                    round,
                    {
                        speaker_id: self._voice_tts(http, model, voice)
                        for speaker_id, (model, voice) in voices.items()
                    },
                    {str(row["id"]): str(row.get("name") or "") for row in speakers},
                ),
                greeting,
                # The round's markers must reach the speech step as written.
                tts_text_transforms=None,
                # A round of several agents is costly to cut short: wait a
                # little longer than a one-to-one call before deciding the
                # caller has finished, so a pause for breath is not a turn.
                min_endpointing_delay=float(
                    self.config.option("group_endpointing_delay", 1.0)
                ),
            ),
            45.0,
        )
        return {
            "provider": self.name,
            "session_id": room,
            "room": room,
            "server_url": os.environ["LIVEKIT_URL"],
            "participant_token": visitor_token,
            "voices": {
                speaker_id: {"model": model, "voice": voice}
                for speaker_id, (model, voice) in voices.items()
            },
        }

    @staticmethod
    def group_voices(speakers: List[Dict[str, Any]]) -> Dict[str, Tuple[str, str]]:
        """Each speaker's speech model and voice for a group call: its own
        when it has one on LiveKit, else the next of the lent voices that
        nobody here is using."""
        chosen: Dict[str, Tuple[str, str]] = {}
        for row in speakers:
            config = row.get("voice_config")
            if isinstance(config, VoiceConfig) and config.provider == "livekit":
                model, voice = split_model(config.tts_model, DEFAULT_TTS_MODEL)
                voice = config.voice or voice
                if voice is None and model == DEFAULT_TTS_MODEL:
                    voice = DEFAULT_TTS_VOICE
                if voice:
                    chosen[str(row["id"])] = (model, voice)
        spare = [voice for voice in GROUP_VOICES if voice not in chosen.values()]
        turn = 0
        for row in speakers:
            speaker_id = str(row["id"])
            if speaker_id in chosen:
                continue
            pool = spare or list(GROUP_VOICES)
            chosen[speaker_id] = pool[turn % len(pool)]
            turn += 1
        return chosen

    def _voice_tts(self, http_session: Any, model: str, voice: str) -> Any:
        from livekit.agents import inference

        kwargs: Dict[str, Any] = {
            "http_session": http_session,
            "voice": voice,
            "sample_rate": CALL_SAMPLE_RATE,
        }
        if self.config.language:
            kwargs["language"] = self.config.language
        return inference.TTS(model, **kwargs)

    async def _start_call(
        self,
        room_name: str,
        agent_token: str,
        make_agent: Callable[[Any], Any],
        greeting: Optional[str],
        **session_options: Any,
    ) -> None:
        import aiohttp
        from livekit import rtc
        from livekit.agents import AgentSession

        http = aiohttp.ClientSession()
        room = rtc.Room()
        session: Any = None
        closed = asyncio.Event()

        async def end() -> None:
            if closed.is_set():
                return
            closed.set()
            with _calls_lock:
                _calls.pop(room_name, None)
            for close in (
                getattr(session, "aclose", None),
                room.disconnect,
                http.close,
            ):
                if close is None:
                    continue
                try:
                    await close()
                except Exception:
                    logger.debug("LiveKit call cleanup failed", exc_info=True)

        try:
            await room.connect(os.environ["LIVEKIT_URL"], agent_token)
            session = AgentSession(
                stt=self._stt(http),
                llm=_brain_llm(),
                tts=self._tts(http),
                # LiveKit can start on an answer before the caller has
                # finished and throw it away if they go on. Here an answer is
                # a real agent turn (it is remembered and shown), so it is
                # only asked for once the turn is over.
                preemptive_generation=False,
                **session_options,
            )
            await session.start(agent=make_agent(http), room=room)
        except Exception as exc:
            await end()
            raise _provider_error(exc) from exc

        def on_left(participant: Any) -> None:
            # The visitor hung up: nobody is left to talk to.
            if not room.remote_participants:
                asyncio.ensure_future(end())

        room.on("participant_disconnected", on_left)
        room.on("disconnected", lambda *_: asyncio.ensure_future(end()))
        asyncio.get_running_loop().call_later(
            float(self.config.option("max_call_seconds", MAX_CALL_SECONDS)),
            lambda: asyncio.ensure_future(end()),
        )
        with _calls_lock:
            _calls[room_name] = end
        if greeting:
            session.say(greeting)

    def end_realtime_session(self, session_id: str) -> None:
        with _calls_lock:
            end = _calls.get(session_id)
        if end is not None:
            try:
                _loop.run(end(), 15.0)
            except Exception:
                logger.debug("Ending LiveKit call failed", exc_info=True)

    def close(self) -> None:
        with _calls_lock:
            rooms = list(_calls)
        for room in rooms:
            self.end_realtime_session(room)


def last_user_text(chat_ctx: Any) -> str:
    """What the person just said, from a LiveKit chat context."""
    for item in reversed(list(getattr(chat_ctx, "items", []) or [])):
        if getattr(item, "type", "message") != "message":
            continue
        if getattr(item, "role", None) == "user":
            return str(getattr(item, "text_content", "") or "").strip()
    return ""


def _brain_agent(agent_name: str, reply: Brain) -> Any:
    """A LiveKit agent whose language-model step is the MemoRizz agent."""
    from livekit.agents import Agent

    class BrainAgent(Agent):
        def __init__(self) -> None:
            # The MemoRizz agent has its own instructions; LiveKit's are unused.
            super().__init__(instructions=f"You are {agent_name}.")

        async def llm_node(self, chat_ctx: Any, tools: Any, model_settings: Any):
            said = last_user_text(chat_ctx)
            if not said:
                return
            # The agent's turn is blocking: keep the room's loop free.
            answer = await asyncio.to_thread(reply, said)
            spoken = speakable(str(answer or ""))
            if spoken:
                yield spoken

    return BrainAgent()


def mark_speaker(speaker_id: str, text: str) -> str:
    """One speaker's words, marked for a group call's speech step."""
    return f"⟦{speaker_id}⟧{text}⟦/⟧"


def split_speakers(text: str) -> Tuple[List[Tuple[str, str]], str]:
    """The complete ``(speaker_id, words)`` parts at the start of ``text``,
    and what is left over (a part still being written)."""
    parts: List[Tuple[str, str]] = []
    end = 0
    for found in _SEGMENT.finditer(text):
        parts.append((found.group(1), found.group(2).strip()))
        end = found.end()
    return parts, text[end:]


def _group_agent(
    name: str,
    round: Callable[[str], Iterable[Tuple[str, str]]],
    voices: Dict[str, Any],
    names: Dict[str, str],
) -> Any:
    """A LiveKit agent that answers a turn with a round of several speakers,
    each in its own voice, one after another."""
    from livekit.agents import Agent

    done = object()

    class GroupAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions=f"You are {name}.")

        async def llm_node(self, chat_ctx: Any, tools: Any, model_settings: Any):
            said = last_user_text(chat_ctx)
            if not said:
                return
            loop = asyncio.get_running_loop()
            answers: asyncio.Queue = asyncio.Queue()
            stop = threading.Event()

            def pump() -> None:
                # The round blocks while each agent thinks: off the room's loop.
                try:
                    for speaker_id, text in round(said):
                        if stop.is_set():
                            break
                        loop.call_soon_threadsafe(
                            answers.put_nowait, (str(speaker_id), str(text or ""))
                        )
                except Exception:
                    logger.exception("A group call's round failed")
                finally:
                    loop.call_soon_threadsafe(answers.put_nowait, done)

            threading.Thread(
                target=pump, name="memorizz-call-round", daemon=True
            ).start()
            try:
                while True:
                    answer = await answers.get()
                    if answer is done:
                        return
                    spoken = speakable(answer[1])
                    if spoken and answer[0] in voices:
                        yield mark_speaker(answer[0], spoken)
            finally:
                # The caller spoke, or hung up: the rest of the round is not said.
                stop.set()

        async def tts_node(self, text: Any, model_settings: Any):
            waiting = ""
            async for chunk in text:
                waiting += chunk if isinstance(chunk, str) else str(chunk)
                parts, waiting = split_speakers(waiting)
                for speaker_id, words in parts:
                    voice = voices.get(speaker_id)
                    if voice is None or not words:
                        continue
                    # One voice at a time: the next waits for this one to end.
                    async for event in voice.synthesize(words):
                        yield event.frame

        async def transcription_node(self, text: Any, model_settings: Any):
            waiting = ""
            async for chunk in text:
                waiting += chunk if isinstance(chunk, str) else str(chunk)
                parts, waiting = split_speakers(waiting)
                for speaker_id, words in parts:
                    yield f"{names.get(speaker_id) or speaker_id}: {words}\n"

    return GroupAgent()


def _brain_llm() -> Any:
    """A placeholder model: a LiveKit session only answers a user's turn when
    it has a language model, even though ``llm_node`` replaces it."""
    from livekit.agents import llm

    class BrainLLM(llm.LLM):
        @property
        def model(self) -> str:
            return "memorizz"

        @property
        def provider(self) -> str:
            return "memorizz"

        def chat(self, **kwargs: Any) -> Any:  # pragma: no cover - never called
            raise RuntimeError("The MemoRizz agent answers through llm_node")

    return BrainLLM()


def _provider_error(exc: Exception) -> VoiceError:
    if isinstance(exc, VoiceError):
        return exc
    return VoiceError(
        f"LiveKit failed: {type(exc).__name__}: {exc}"[:300],
        code="voice_provider_error",
    )

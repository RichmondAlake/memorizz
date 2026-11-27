# Voice

An agent can speak its replies, listen to a recording, and take a live call.
Voice is off by default. Turn it on with `voice=True`:

```python
from memorizz import MemAgent

agent = MemAgent(
    llm_config={"provider": "openai", "model": "gpt-5"},
    memory_provider=provider,
    name="Ada",
    voice=True,
)
```

With nothing else set, the voice comes from LiveKit. `voice_config` names
another provider or tunes the one in use:

```python
MemAgent(voice=True, voice_config="elevenlabs")
MemAgent(voice=True, voice_config={"provider": "elevenlabs", "voice": "Rachel"})
MemAgent(voice=True, voice_config={"provider": "vapi"})
MemAgent(voice=True, voice_config={"provider": "openai", "voice": "nova"})
```

Both settings are saved with the agent. Credentials are not: they are read
from the environment, and a `voice_config` that holds a key is refused.

## Providers

| Provider | Speaks and transcribes turn by turn | Live call | Needs |
|---|---|---|---|
| `livekit` (default) | Yes | Yes | `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, and `pip install "memorizz[voice]"` |
| `elevenlabs` | Yes | No | `ELEVENLABS_API_KEY` |
| `vapi` | No | Yes | `VAPI_API_KEY`, `VAPI_PUBLIC_KEY`, `MEMORIZZ_PUBLIC_URL` |
| `openai` | Yes | No | `OPENAI_API_KEY` |

Set the keys in your environment, or in the local UI under **Settings →
Voice**. `agent.voice_status()` says whether the agent's provider is ready and
names what is missing:

```python
agent.voice_status()
# {"provider": "livekit", "ready": False, "enabled": True,
#  "problem": "Set LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET in Settings.", ...}
```

## Turn by turn

```python
audio = agent.speak("Good morning.")          # VoiceAudio: .data (bytes), .mime_type
heard = agent.listen(recording, "audio/webm")  # what was said, as text

turn = agent.run_voice(recording, "audio/webm", memory_id="call-1")
turn["heard"]   # the transcript
turn["text"]    # the agent's answer, from agent.run()
turn["audio"]   # the answer spoken, or None when nothing was heard
```

`run_voice` passes any other keyword to `run()`, so a spoken turn uses the
same memory, tools and traces as a typed one. Markdown and code blocks are
left out of what is spoken, and a very long answer is cut at a sentence.

A browser recording works as it is: WebM, Ogg, MP4 and WAV are accepted.

## Live calls

In a live call the provider handles listening, turn taking and speaking, and
asks the agent for each reply.

```python
details = agent.start_voice_session(greeting="Hi, Ada here.", memory_id="call-1")
# LiveKit: {"provider": "livekit", "server_url": ..., "participant_token": ..., "session_id": ...}
# Vapi:    {"provider": "vapi", "public_key": ..., "assistant_id": ..., "session_id": ...}

agent.end_voice_session(details["session_id"])
```

Hand `details` to the browser. With LiveKit, the page joins the room with
`livekit-client` (`room.connect(server_url, participant_token)`) and turns on
the microphone. With Vapi, it calls `new Vapi(public_key).start(assistant_id)`.

**LiveKit.** MemoRizz joins the room as the agent, in the same process, and
runs LiveKit's voice session with the MemAgent as its language model. Speech
recognition and speech come from LiveKit Inference, a LiveKit Cloud service,
so the three LiveKit variables are all it needs. A self-hosted LiveKit server
can carry the room but has no speech models.

**Vapi.** Vapi runs the call on its own servers and asks a "custom LLM" for
each reply. That is an endpoint of the local UI
(`/webhook/voice/vapi/<agent>/<token>/chat/completions`), so the UI has to be
running and reachable from the internet over https: set `MEMORIZZ_PUBLIC_URL`
to a tunnel or a deployed address. MemoRizz creates one stored Vapi assistant
per agent with the private key; the browser only receives the public key and
the assistant's id. The path token is derived from the private key, so a
request with any other token gets a 404.

## A call with several agents

`LiveKitVoice.group_session` holds one call for several agents. LiveKit
listens and decides when the caller has finished; then `round(said)` is asked
for the answers and each is spoken in that agent's own voice, one after
another on a single stream, so two agents never talk at once. When the
caller speaks again, the rest of the round is dropped.

```python
from memorizz.voice import VoiceConfig, create_voice_provider

def round(said):
    # Yield (speaker_id, text) in the order they should be spoken.
    yield "ada", ada.run(said)
    yield "bruno", bruno.run(said)

details = create_voice_provider(VoiceConfig(provider="livekit")).group_session(
    speakers=[
        {"id": "ada", "name": "Ada", "voice_config": VoiceConfig(provider="livekit")},
        {"id": "bruno", "name": "Bruno"},   # no voice of its own: one is lent
    ],
    round=round,
)
# details: server_url, participant_token, session_id, voices
```

A speaker whose voice is on another provider is lent a LiveKit voice for the
call, since one stream carries them all. The call waits about a second of
silence before taking the caller's turn as finished (`group_endpointing_delay`
in the provider's options).

## Settings

| `voice_config` key | Meaning |
|---|---|
| `provider` | `livekit`, `elevenlabs`, `vapi` or `openai` |
| `voice` | A voice id or name from the provider |
| `tts_model` | The speech model (LiveKit: `provider/model[:voice]`, for example `cartesia/sonic-3`) |
| `stt_model` | The transcription model (LiveKit: `provider/model[:language]`) |
| `language` | A language code, when the provider should not detect it |
| `speed` | Speaking rate, 0.25 to 4 |

Any other key is passed to the provider as an option, for example
`{"provider": "vapi", "public_url": "https://agents.example.com"}`.

Defaults: LiveKit speaks with `cartesia/sonic-3` and transcribes with
`assemblyai/universal-3-6-pro`; ElevenLabs uses `eleven_flash_v2_5` and
`scribe_v3` and, when no voice is named, the account's first voice; OpenAI
uses `gpt-4o-mini-tts` (voice `alloy`) and `gpt-4o-mini-transcribe`.

## In the local UI

The agent form has a **Voice** section (on or off, provider, voice). These
routes work for any saved agent with a voice:

| Route | Does |
|---|---|
| `GET /api/voice/status` | Every provider, whether it is ready and what it can do |
| `POST /api/agents/{id}/voice/speak` | JSON `{"text": ...}` in, audio out |
| `POST /api/agents/{id}/voice/listen` | A recording (form field `audio`) in, `{"text": ...}` out |
| `POST /api/agents/{id}/voice/session` | Starts a live call and returns what the browser needs. With `{"memory_id": "..."}` (empty for a new conversation) the call is part of that playground conversation and works for an agent without a voice, on LiveKit's default voice |

In the playground, the waveform button in the composer starts a live
conversation with the agent: just talk, and speak again to cut an answer
short. What is said shows above the composer as you talk, and the turns are
part of the open conversation when you end the call.

A live call asks the agent for an answer only once you have finished
speaking. LiveKit's early, speculative answers are turned off, because with a
MemAgent every answer is a real turn that is remembered.

## From the command line

```bash
memorizz agents create --name Ada --voice                      # LiveKit
memorizz agents create --name Ada --voice-provider elevenlabs --voice-name Rachel
memorizz agents update <agent-id> --voice-provider openai      # switch provider
memorizz agents update <agent-id> --no-voice                   # off, settings kept
```

Naming a provider or a voice turns voice on. `--voice-name ""` goes back to
the provider's default voice. `memorizz agents show <agent-id>` reports
`voice`, `voice_provider` and `voice_name`.

## Over MCP

`memorizz_create_agent` and `memorizz_update_agent` take `voice`,
`voice_provider` and `voice_name`, with the same rules. Provider keys are
never passed through MCP; `memorizz_get_agent` returns `voice` and
`voice_config`.

## The builder

```python
from memorizz.memagent.builders import MemAgentBuilder

agent = (
    MemAgentBuilder()
    .with_llm_config({"provider": "openai", "model": "gpt-5"})
    .with_voice("elevenlabs", voice="Rachel")
    .build()
)
```

`agent.with_voice(False)` turns voice off and keeps the settings;
`agent.with_voice(config="openai")` switches provider.

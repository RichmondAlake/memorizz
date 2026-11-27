"""Voice for saved agents: speak, listen, start a live call, and the endpoint
a live call's provider asks for each reply."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ..state import _state

logger = logging.getLogger(__name__)

router = APIRouter(tags=["voice"])

MAX_AUDIO_BYTES = 12 * 1024 * 1024


def _load(agent_id: str) -> Any:
    from ...memagent import MemAgent

    provider = _state.get("provider")
    if provider is None:
        raise HTTPException(status_code=400, detail="Connect to a memory store first")
    try:
        return MemAgent.load(agent_id, memory_provider=provider)
    except Exception as exc:
        raise HTTPException(
            status_code=404, detail=f"Agent not found: {exc}"[:200]
        ) from exc


def _failure(exc: Exception) -> HTTPException:
    from ...voice import VoiceError, VoiceNotConfigured, VoiceUnsupported

    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, VoiceNotConfigured):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, VoiceUnsupported):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, VoiceError):
        status = 409 if exc.code == "voice_disabled" else 502
        return HTTPException(status_code=status, detail=str(exc))
    logger.exception("Voice request failed")
    return HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}"[:300])


def voice_memory_id(agent_id: str) -> str:
    """The conversation an agent keeps of its voice calls."""
    return f"voice-{agent_id}"


def default_brain(agent_id: str):
    """Answer a live call by running the saved agent."""

    def reply(said: str) -> str:
        agent = _load(agent_id)
        try:
            return agent.run(said, memory_id=voice_memory_id(agent_id))
        finally:
            agent.close(close_memory_provider=False)

    return reply


@router.get("/api/voice/status")
async def voice_providers_status():
    """Every voice provider with whether it is ready and what it can do."""
    from ...voice import DEFAULT_VOICE_PROVIDER, voice_status

    return {"default": DEFAULT_VOICE_PROVIDER, "providers": voice_status()}


@router.get("/api/agents/{agent_id}/voice")
async def agent_voice_status(agent_id: str):
    agent = await asyncio.to_thread(_load, agent_id)
    return agent.voice_status()


@router.post("/api/agents/{agent_id}/voice/speak")
async def agent_speak(agent_id: str, request: Request):
    """Text in, the agent's speech out."""
    body = await request.json()
    text = str((body or {}).get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Nothing to say")
    try:
        agent = await asyncio.to_thread(_load, agent_id)
        audio = await asyncio.to_thread(agent.speak, text)
    except Exception as exc:
        raise _failure(exc) from exc
    return Response(
        content=audio.data,
        media_type=audio.mime_type,
        headers={"Cache-Control": "no-store"},
    )


@router.post("/api/agents/{agent_id}/voice/listen")
async def agent_listen(agent_id: str, audio: UploadFile = File(...)):
    """A recording in, what was said out."""
    data = await audio.read(MAX_AUDIO_BYTES + 1)
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="That recording is too long")
    try:
        agent = await asyncio.to_thread(_load, agent_id)
        text = await asyncio.to_thread(
            agent.listen, data, audio.content_type or "audio/webm"
        )
    except Exception as exc:
        raise _failure(exc) from exc
    return {"text": text}


def conversation_brain(agent_id: str, memory_id: str):
    """Answer a live call inside one playground conversation: each thing
    said is a turn of that conversation, as if it had been typed there."""

    def reply(said: str) -> str:
        from ...cli.conversations import latest_thread_id
        from .playground import _playground_recall_policy

        agent = _load(agent_id)
        try:
            options: Dict[str, Any] = {"memory_id": memory_id}
            try:
                thread_id = latest_thread_id(
                    _state["provider"], memory_id, agent_id=agent_id
                )
            except Exception:
                thread_id = None
            if thread_id:
                options["thread_id"] = thread_id
            policy = _playground_recall_policy(agent, memory_id)
            if policy is not None:
                options["retrieval_policy"] = policy
            answer = agent.run(said, **options)
            # The conversation joins the agent's list, as a typed one does.
            if memory_id not in (agent.memory_ids or []):
                agent.memory_ids.append(memory_id)
                update = getattr(_state["provider"], "update_memagent_memory_ids", None)
                if callable(update):
                    update(agent_id, agent.memory_ids)
            return answer
        finally:
            agent.close(close_memory_provider=False)

    return reply


def _call_provider(agent: Any) -> Any:
    """The agent's own voice when it can hold a live call; otherwise LiveKit
    with its default voice, so any agent can be talked to."""
    from ...voice import VoiceConfig, create_voice_provider

    own = None
    if getattr(agent, "voice", False):
        try:
            own = agent.voice_provider
        except Exception:
            own = None
    status = own.status() if own is not None else {}
    if status.get("ready") and status.get("realtime"):
        return own
    return create_voice_provider(VoiceConfig(provider="livekit"))


@router.post("/api/agents/{agent_id}/voice/session")
async def agent_voice_session(agent_id: str, request: Request):
    """Start a live call; returns what the browser needs to join it.

    With a JSON body that has ``memory_id`` (empty for a new conversation)
    the call is part of that playground conversation and works for an agent
    without a voice of its own. Without one it is the agent's own voice line.
    """
    try:
        body = await request.json()
    except Exception:
        body = None
    try:
        agent = await asyncio.to_thread(_load, agent_id)
        name = agent.name or "the agent"
        if isinstance(body, dict) and "memory_id" in body:
            memory_id = str(body.get("memory_id") or "").strip() or str(uuid.uuid4())
            provider = _call_provider(agent)
            details = await asyncio.to_thread(
                lambda: provider.realtime_session(
                    agent_id=agent_id,
                    agent_name=name,
                    reply=conversation_brain(agent_id, memory_id),
                )
            )
            details["memory_id"] = memory_id
            return details
        details = await asyncio.to_thread(
            lambda: agent.voice_provider.realtime_session(
                agent_id=agent_id,
                agent_name=name,
                reply=default_brain(agent_id),
            )
        )
    except Exception as exc:
        raise _failure(exc) from exc
    return details


@router.delete("/api/agents/{agent_id}/voice/session/{session_id}")
async def agent_voice_session_end(agent_id: str, session_id: str):
    try:
        agent = await asyncio.to_thread(_load, agent_id)
        # The call may be on the agent's voice or on LiveKit's default one.
        await asyncio.to_thread(_call_provider(agent).end_realtime_session, session_id)
        if getattr(agent, "voice", False):
            await asyncio.to_thread(agent.end_voice_session, session_id)
    except Exception as exc:
        raise _failure(exc) from exc
    return {"ok": True}


# ---- Vapi custom LLM ------------------------------------------------------


def _last_user_text(messages: Any) -> str:
    for message in reversed(messages if isinstance(messages, list) else []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(
                str(part.get("text") or "")
                for part in content
                if isinstance(part, dict)
            )
        text = str(content or "").strip()
        if text:
            return text
    return ""


def _chunk(
    completion_id: str, model: str, delta: Dict[str, Any], finish: Optional[str]
) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/webhook/voice/vapi/{agent_id}/{token}/{rest:path}")
@router.post("/webhook/voice/vapi/{agent_id}/{token}")
async def vapi_chat_completions(
    agent_id: str, token: str, request: Request, rest: str = ""
):
    """OpenAI-style chat completions for a Vapi call: the caller's last words
    in, the agent's reply out. The path token is the only credential."""
    from ...voice import get_brain, speakable
    from ...voice.providers.vapi import valid_webhook_token

    if not valid_webhook_token(agent_id, token):
        raise HTTPException(status_code=404, detail="Not found")
    if rest and not rest.rstrip("/").endswith("chat/completions"):
        raise HTTPException(status_code=404, detail="Not found")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Send a JSON object") from None
    said = _last_user_text((body or {}).get("messages"))
    model = str((body or {}).get("model") or "memorizz")
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    brain = get_brain(agent_id) or default_brain(agent_id)

    async def answer() -> str:
        if not said:
            return ""
        try:
            return speakable(str(await asyncio.to_thread(brain, said) or ""))
        except Exception:
            logger.exception("The agent could not answer a voice call")
            return "Sorry, I could not answer that."

    if not (body or {}).get("stream", False):
        text = await answer()
        return JSONResponse(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    async def stream():
        yield _chunk(completion_id, model, {"role": "assistant", "content": ""}, None)
        text = await answer()
        if text:
            yield _chunk(completion_id, model, {"content": text}, None)
        yield _chunk(completion_id, model, {}, "stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

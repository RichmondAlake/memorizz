"""Voice on the CLI's agent commands and the MCP server's agent tools."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from memorizz.approval import SQLiteApprovalStore
from memorizz.cli.app import app
from memorizz.mcp_server import MemorizzMCPServerConfig, MemorizzRuntime
from memorizz.mcp_server.auth import RequestIdentity
from memorizz.mcp_server.config import ALL_SCOPES
from memorizz.mcp_server.runtime import MemorizzServerError
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.voice import voice_settings

pytestmark = pytest.mark.unit

runner = CliRunner()


# ---- the rule both surfaces share ------------------------------------------


def test_naming_a_provider_or_a_voice_turns_voice_on():
    assert voice_settings() == (False, None)
    assert voice_settings(enabled=True) == (True, None)  # LiveKit, the default
    assert voice_settings(provider="ElevenLabs") == (True, {"provider": "elevenlabs"})
    assert voice_settings(voice="Rachel") == (
        True,
        {"provider": "livekit", "voice": "Rachel"},
    )


def test_what_is_not_named_is_kept():
    stored = {
        "provider": "vapi",
        "voice": "v1",
        "options": {"public_url": "https://x.example"},
    }
    assert voice_settings(True, stored) == (True, stored)
    assert voice_settings(True, stored, enabled=False) == (False, stored)
    # An empty voice name goes back to the provider's default voice.
    assert voice_settings(True, stored, voice="")[1] == {
        "provider": "vapi",
        "options": {"public_url": "https://x.example"},
    }
    # Another provider does not inherit the old one's voice or options.
    assert voice_settings(True, stored, provider="openai") == (
        True,
        {"provider": "openai"},
    )
    with pytest.raises(ValueError, match="Unknown voice provider"):
        voice_settings(provider="carrier-pigeon")


# ---- CLI -------------------------------------------------------------------


def _env(tmp_path):
    return {
        "MEMORIZZ_HOME": str(tmp_path),
        "MEMORIZZ_BACKEND": "",
        "MEMORIZZ_DEFAULT_LLM_PROVIDER": "",
        "MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER": "",
        "OPENAI_API_KEY": "",
        "ANTHROPIC_API_KEY": "",
        "AZURE_OPENAI_API_KEY": "",
        "OLLAMA_HOST": "http://127.0.0.1:1",
    }


def _cli(tmp_path, *args):
    result = runner.invoke(app, ["agents", *args, "--json"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    return json.loads(result.output.strip().splitlines()[-1])["agent"]


def test_cli_creates_and_changes_an_agents_voice(tmp_path):
    created = _cli(
        tmp_path,
        "create",
        "--name",
        "Ada",
        "--no-llm",
        "--voice-provider",
        "elevenlabs",
        "--voice-name",
        "Rachel",
    )
    assert (created["voice"], created["voice_provider"], created["voice_name"]) == (
        True,
        "elevenlabs",
        "Rachel",
    )
    agent_id = created["agent_id"]

    stored = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", use_faiss=False)
    ).retrieve_memagent(agent_id)
    assert stored.voice is True
    assert stored.voice_config == {"provider": "elevenlabs", "voice": "Rachel"}

    off = _cli(tmp_path, "update", agent_id, "--no-voice")
    assert off["voice"] is False and off["voice_provider"] == "elevenlabs"  # kept
    switched = _cli(tmp_path, "update", agent_id, "--voice-provider", "openai")
    assert (switched["voice"], switched["voice_provider"], switched["voice_name"]) == (
        True,
        "openai",
        None,
    )
    shown = _cli(tmp_path, "show", agent_id)
    assert shown["voice_provider"] == "openai"


def test_cli_voice_alone_means_livekit(tmp_path):
    created = _cli(tmp_path, "create", "--name", "Bo", "--no-llm", "--voice")
    assert created["voice"] is True and created["voice_provider"] == "livekit"
    plain = _cli(tmp_path, "create", "--name", "Cy", "--no-llm")
    assert plain["voice"] is False and plain["voice_provider"] is None


def test_cli_refuses_an_unknown_voice_provider(tmp_path):
    result = runner.invoke(
        app,
        ["agents", "create", "--name", "Di", "--no-llm", "--voice-provider", "pigeon"],
        env=_env(tmp_path),
    )
    assert result.exit_code != 0
    assert "Unknown voice provider" in result.output


# ---- MCP -------------------------------------------------------------------


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    store = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    yield MemorizzRuntime(
        MemorizzMCPServerConfig(
            transport="stdio", allow_writes=True, exposed_agent_ids=None
        ),
        provider=store,
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
    )
    store.close()


def _identity():
    return RequestIdentity(
        principal="alice", scopes=frozenset(ALL_SCOPES), authenticated=True
    )


def test_mcp_creates_and_updates_an_agents_voice(runtime):
    created = runtime.create_agent(
        "Ada",
        _identity(),
        llm_provider="ollama",
        llm_model="qwen2.5:7b",
        voice=True,
        voice_provider="elevenlabs",
        voice_name="Rachel",
    )["agent"]
    assert created["voice"] is True
    assert created["voice_config"] == {"provider": "elevenlabs", "voice": "Rachel"}
    agent_id = created["agent_id"]

    off = runtime.update_agent(agent_id, _identity(), voice=False)["agent"]
    assert off["voice"] is False
    assert off["voice_config"] == {"provider": "elevenlabs", "voice": "Rachel"}

    switched = runtime.update_agent(agent_id, _identity(), voice_provider="vapi")[
        "agent"
    ]
    assert switched["voice"] is True and switched["voice_config"] == {
        "provider": "vapi"
    }
    assert runtime.get_agent(agent_id, _identity())["agent"]["voice"] is True
    # A change to something else leaves the voice alone.
    renamed = runtime.update_agent(agent_id, _identity(), name="Ada 2")["agent"]
    assert renamed["voice_config"] == {"provider": "vapi"}


def test_mcp_agents_have_no_voice_unless_asked(runtime):
    created = runtime.create_agent(
        "Bo", _identity(), llm_provider="ollama", llm_model="qwen2.5:7b"
    )["agent"]
    assert created["voice"] is False and "voice_config" not in created


def test_mcp_refuses_an_unknown_voice_provider(runtime):
    with pytest.raises(MemorizzServerError, match="Unknown voice provider"):
        runtime.create_agent(
            "Cy",
            _identity(),
            llm_provider="ollama",
            llm_model="qwen2.5:7b",
            voice_provider="pigeon",
        )


def test_the_mcp_tools_declare_the_voice_arguments():
    import inspect

    from memorizz.mcp_server import server as server_module

    source = inspect.getsource(server_module)
    for tool in ("memorizz_create_agent", "memorizz_update_agent"):
        body = source[source.index(f"async def {tool}(") :]
        body = body[: body.index("return await _tool_call(") + 2500]
        for argument in ("voice", "voice_provider", "voice_name"):
            assert f"{argument}: Optional" in body, (tool, argument)
            assert f"{argument}={argument}" in body, (tool, argument)

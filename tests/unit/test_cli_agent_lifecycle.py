"""How the CLI opens, creates and switches agents, and what it writes down."""

from __future__ import annotations

import builtins
import json
import shutil
import sys
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from rich.console import Console
from typer.testing import CliRunner

from memorizz.cli import agent_factory, commands
from memorizz.cli import config as cfg
from memorizz.cli import harness_session, ollama_probe, picker
from memorizz.cli.app import app
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

pytestmark = pytest.mark.unit

OWN_MODEL = {"provider": "ollama", "model": "own-model"}
PROMPT = "Answer as a pirate."


@pytest.fixture
def provider(tmp_path, monkeypatch):
    """A real store and real agents; only the model client is fake."""
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    monkeypatch.setenv("MEMORIZZ_DEFAULT_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("MEMORIZZ_DEFAULT_LLM_MODEL", "default-model")
    for name in ("MEMORIZZ_BROWSER_CONTROL_PROVIDER", "OPENAI_API_KEY", "CI"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("MEMORIZZ_NO_PICKER", raising=False)
    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=Mock()))
    monkeypatch.setattr("memorizz.llms.ollama._MODEL_PROFILES", {})
    monkeypatch.setattr(
        ollama_probe,
        "list_models",
        lambda *a, **k: ["default-model", "own-model", "installed-model"],
    )
    monkeypatch.setattr(agent_factory, "make_internet_provider", lambda: None)
    monkeypatch.setattr("memorizz.llms.model_lists.latest_models", lambda *a, **k: [])
    monkeypatch.setattr(
        "memorizz.memagent.managers.cache_manager.CacheManager._initialize_cache",
        lambda manager, *a, **k: setattr(manager, "cache_instance", object()),
    )
    return FileSystemProvider(FileSystemConfig(root_path=str(tmp_path / "memory")))


@pytest.fixture
def browser_use(monkeypatch):
    """Browser Use switches on only with its command, the Python behind that
    command, and a key to hand; CI has none of them."""
    which = shutil.which
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MEMORIZZ_BROWSER_USE_PYTHON_COMMAND", sys.executable)
    monkeypatch.setattr(
        "memorizz.browser_control.providers.browser_use.shutil.which",
        lambda command, **options: (
            "/isolated/bin/browser-use"
            if command == "browser-use"
            else which(command, **options)
        ),
    )


def open_session(provider, terminal=False, **options):
    session = agent_factory.build_session_agent(memory_provider=provider, **options)
    session.console = Console(
        file=StringIO(), force_terminal=terminal, width=200, color_system=None
    )
    return session


def output(session) -> str:
    return session.console.file.getvalue()


def saved_agent(provider, *, name="Pirate", instruction=PROMPT, llm=OWN_MODEL):
    agent = agent_factory.build_agent(provider, dict(llm), instruction)
    agent.name = name
    agent.save()
    return agent.agent_id


def record(provider, agent_id):
    return provider.retrieve_memagent(agent_id)


def conversation(session, memory_id="memory-1"):
    """Give the session's agent a conversation, as a turn would."""
    session.thread_id = session.agent.start_new_thread(memory_id)
    session.memory_id = memory_id
    session.agent.save()
    cfg.save_state({"memory_id": memory_id, "thread_id": session.thread_id})
    return session.thread_id


# --------------------------------------------------------------------------- #
# The state file: an agent and its conversation change together
# --------------------------------------------------------------------------- #


def test_set_active_agent_drops_the_previous_conversation(provider):
    cfg.save_state({"agent_id": "old", "memory_id": "m", "thread_id": "t"})
    cfg.set_active_agent("new")
    assert cfg.load_state() == {"agent_id": "new"}


def test_agents_create_set_default_leaves_the_old_conversation_behind(
    provider, tmp_path
):
    cfg.save_state({"agent_id": "old", "memory_id": "m", "thread_id": "t"})
    result = CliRunner().invoke(
        app,
        ["agents", "create", "--name", "Pirate", "--set-default", "--no-llm"]
        + ["--json"],
        env={"MEMORIZZ_HOME": str(tmp_path), "MEMORIZZ_BACKEND": ""},
    )
    assert result.exit_code == 0, result.output
    agent_id = json.loads(result.stdout.strip().splitlines()[-1])["agent"]["agent_id"]
    assert cfg.load_state() == {"agent_id": agent_id}


def test_a_new_default_agent_starts_on_its_own_conversation(provider):
    session = open_session(provider)
    conversation(session)
    pirate = saved_agent(provider)
    cfg.set_active_agent(pirate)

    reopened = open_session(provider)

    assert reopened.agent.agent_id == pirate
    assert reopened.memory_id is None and reopened.thread_id is None
    assert reopened.agent.get_current_thread_id() is None


def test_a_saved_agent_that_is_gone_takes_its_conversation_with_it(provider):
    cfg.save_state({"agent_id": "gone", "memory_id": "m", "thread_id": "t"})
    session = open_session(provider)
    assert cfg.load_state() == {"agent_id": session.agent.agent_id}
    assert session.memory_id is None and session.thread_id is None


def test_an_old_state_file_with_only_a_memory_keeps_it(provider):
    cfg.save_state({"memory_id": "legacy-memory"})
    session = open_session(provider)
    assert session.memory_id == "legacy-memory"
    assert cfg.load_state()["memory_id"] == "legacy-memory"


def test_a_memory_the_agent_never_used_is_not_adopted(provider):
    conversation(open_session(provider))
    # What a switch left behind before switches were saved.
    cfg.save_state({"memory_id": "another-agents", "thread_id": "their-thread"})

    session = open_session(provider)

    assert session.memory_id is None and session.thread_id is None
    assert "another-agents" not in session.agent.memory_ids


# --------------------------------------------------------------------------- #
# What a launch does to the saved agent
# --------------------------------------------------------------------------- #


def test_opening_a_configured_agent_changes_nothing_saved(provider):
    pirate = saved_agent(provider)
    stored = record(provider, pirate)
    assert stored.instruction == PROMPT
    cfg.set_active_agent(pirate)

    session = open_session(provider)
    session.agent.save()  # every turn saves

    assert session.agent.instruction == PROMPT
    assert session.model_name == "own-model"
    kept = record(provider, pirate)
    assert kept.instruction == PROMPT
    assert kept.llm_config["model"] == "own-model"
    assert len(kept.tools) == len(stored.tools)
    # The session itself still gets the CLI's slim tool set.
    assert "knowledge_base_lookup" not in session.agent.tool_manager.list_tools()
    assert any(t["name"] == "knowledge_base_lookup" for t in kept.tools)


def test_opening_a_configured_agent_keeps_a_saved_self_aware_setting(provider):
    pirate = saved_agent(provider)
    cfg.set_active_agent(pirate)
    session = open_session(provider)
    session.agent.with_self_aware(True)
    commands.save_agent_fields(session, self_aware=True)

    reopened = open_session(provider)  # chat mode turns it off for the session
    reopened.agent.save()

    assert reopened.agent.self_aware is False
    assert record(provider, pirate).self_aware is True


def test_a_launch_setting_for_the_browser_is_not_saved_on_the_agent(
    provider, monkeypatch, browser_use
):
    pirate = saved_agent(provider)
    stored = record(provider, pirate).browser_control
    cfg.set_active_agent(pirate)
    monkeypatch.setenv("MEMORIZZ_BROWSER_CONTROL_PROVIDER", "browseruse")

    session = open_session(provider)
    session.agent.save()

    assert session.agent.has_browser_control()
    assert record(provider, pirate).browser_control == stored


def test_browser_on_and_off_are_saved_on_a_configured_agent(provider, browser_use):
    pirate = saved_agent(provider)
    cfg.set_active_agent(pirate)
    session = open_session(provider)

    commands.dispatch("/browser on", session)
    assert record(provider, pirate).browser_control

    commands.dispatch("/browser off", session)
    assert not record(provider, pirate).browser_control


def test_the_cli_assistant_follows_the_defaults(provider, monkeypatch):
    agent_id = open_session(provider).agent.agent_id
    monkeypatch.setenv("MEMORIZZ_DEFAULT_LLM_MODEL", "installed-model")
    session = open_session(provider)
    session.agent.save()
    assert session.agent.agent_id == agent_id
    assert session.model_name == "installed-model"
    assert record(provider, agent_id).llm_config["model"] == "installed-model"


def test_the_cli_assistant_gets_the_current_instruction(provider):
    session = open_session(provider)
    opening = cfg.MEMORY_ASSISTANT_INSTRUCTION.splitlines()[0]
    session.agent.instruction = f"{opening} An older version."
    session.agent.save()
    assert open_session(provider).agent.instruction == cfg.MEMORY_ASSISTANT_INSTRUCTION


def test_a_model_on_the_command_line_is_for_that_session(provider):
    pirate = saved_agent(provider)
    cfg.set_active_agent(pirate)

    session = open_session(
        provider, llm_config={"provider": "ollama", "model": "installed-model"}
    )
    session.agent.save()

    assert session.model_name == "installed-model"
    assert record(provider, pirate).llm_config["model"] == "own-model"


def test_an_agent_whose_model_is_unavailable_borrows_the_default(provider, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    pirate = saved_agent(provider, llm={"provider": "openai", "model": "gpt-6-luna"})
    monkeypatch.delenv("OPENAI_API_KEY")
    cfg.set_active_agent(pirate)

    session = open_session(provider)
    session.agent.save()

    assert session.model_name == "default-model"
    assert any("gpt-6-luna is not available here" in w for w in session.warnings)
    assert record(provider, pirate).llm_config["model"] == "gpt-6-luna"


def test_an_agent_saved_without_a_model_uses_the_default(provider):
    pirate = saved_agent(provider, llm={"provider": "openai", "model": "gpt-6-luna"})
    assert record(provider, pirate).llm_config is None  # no key, so none was saved
    cfg.set_active_agent(pirate)

    session = open_session(provider)

    assert session.model_name == "default-model"
    assert not any("not available" in w for w in session.warnings or [])


# --------------------------------------------------------------------------- #
# Launching: a new conversation until you pick one
# --------------------------------------------------------------------------- #


def test_the_chat_opens_on_a_new_conversation_in_the_same_memory(provider):
    first = open_session(provider)
    thread = conversation(first)
    state = cfg.load_state()

    chat = open_session(provider, resume=False)

    assert chat.memory_id == "memory-1"
    assert chat.thread_id is None
    assert chat.agent.get_current_thread_id() is None
    commands.dispatch("/exit", chat)
    assert cfg.load_state() == state, "an empty launch must not touch the state"

    assert open_session(provider).thread_id == thread, "one-shots carry on"


def test_only_the_chat_opens_on_a_new_conversation(monkeypatch):
    from memorizz.cli import app as cli_app

    seen = []
    monkeypatch.setattr(
        cli_app, "_build_or_wizard", lambda *a, **k: seen.append(k) or None
    )
    with pytest.raises(BaseException):
        cli_app._launch_repl()
    assert seen == [{"resume": False}]
    # memorizz run has no such argument: it keeps the rolling thread.
    with pytest.raises(BaseException):
        cli_app._run_oneshot("hello")
    assert seen[-1].get("resume", True) is True


# --------------------------------------------------------------------------- #
# /agent <id>
# --------------------------------------------------------------------------- #


def test_switching_is_remembered_and_starts_clean(provider):
    session = open_session(provider)
    conversation(session)
    pirate = saved_agent(provider)

    commands.dispatch(f"/agent {pirate}", session)

    assert session.agent.agent_id == pirate
    assert session.memory_id is None and session.thread_id is None
    assert session.model_name == "own-model"
    assert cfg.load_state() == {"agent_id": pirate}
    assert "Loaded agent" in output(session)
    assert open_session(provider).agent.agent_id == pirate


def test_a_switch_sets_the_agent_up_like_a_launch(provider):
    pirate = saved_agent(provider)
    session = open_session(provider, code_mode=True)
    commands.dispatch(f"/agent {pirate}", session)
    assert session.agent.self_aware is True
    assert "knowledge_base_lookup" in session.agent.tool_manager.list_tools()

    session = open_session(provider)  # opens the pirate, in chat mode
    assert "knowledge_base_lookup" not in session.agent.tool_manager.list_tools()


def test_a_picked_harness_comes_along_on_a_switch(provider, monkeypatch):
    monkeypatch.setattr(
        harness_session,
        "list_harness_status",
        lambda session, **k: [{"name": "codex", "ready": True}],
    )
    pirate = saved_agent(provider)
    session = open_session(provider)
    old = session.agent
    harness_session.apply_harness(session, "codex", model="picked-model")

    commands.dispatch(f"/agent {pirate}", session)

    assert session.harness == "codex" and session.harness_model == "picked-model"
    assert session.agent.meta_harness_mode == "runtime"
    assert session.agent.default_harness == "codex"
    assert old.meta_harness_mode is None, "the old agent gets its own settings back"
    # /harness off restores the new agent's settings, not the old agent's.
    harness_session.clear_harness(session)
    assert session.harness is None and session.agent.meta_harness_mode is None
    session.agent.save()
    assert record(provider, pirate).meta_harness_mode is None


def test_a_switch_forgets_the_old_agents_saved_harness(provider):
    pirate = saved_agent(provider)
    session = open_session(provider)
    session.agent.meta_harness_mode = "runtime"
    session.agent.default_harness = "codex"
    harness_session.clear_harness(session)  # that agent still routes when saved
    assert session.harness_backup is not None

    commands.dispatch(f"/agent {pirate}", session)

    assert session.harness_backup is None


def test_a_harness_that_cannot_come_along_is_reported(provider, monkeypatch):
    status = [{"name": "codex", "ready": True}]
    monkeypatch.setattr(
        harness_session, "list_harness_status", lambda session, **k: status
    )
    pirate = saved_agent(provider)
    session = open_session(provider)
    harness_session.apply_harness(session, "codex")
    status[0].update(ready=False, reason="signed out")

    commands.dispatch(f"/agent {pirate}", session)

    assert session.agent.agent_id == pirate and session.harness is None
    assert "Harness not kept" in output(session) and "signed out" in output(session)
    assert cfg.load_state() == {"agent_id": pirate}


# --------------------------------------------------------------------------- #
# /agent new
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("line", "model", "instruction", "name"),
    [
        (
            "/agent new Answer as a pirate. Say arr.",
            "default-model",
            "Answer as a pirate. Say arr.",
            "Answer as a pirate",
        ),
        (
            "/agent new You are a poet. Rhyme.",
            "default-model",
            "You are a poet. Rhyme.",
            "poet",
        ),
        (
            "/agent new you are an expert on tides",
            "default-model",
            "you are an expert on tides",
            "expert on tides",
        ),
        (
            "/agent new installed-model",
            "installed-model",
            cfg.MEMORY_ASSISTANT_INSTRUCTION,
            "installed-model",
        ),
        (
            "/agent new installed-model You are a judge.",
            "installed-model",
            "You are a judge.",
            "judge",
        ),
        (
            "/agent new ollama:installed-model",
            "installed-model",
            cfg.MEMORY_ASSISTANT_INSTRUCTION,
            "installed-model",
        ),
        (
            '/agent new "You are a clerk who files."',
            "default-model",
            "You are a clerk who files.",
            "clerk who files",
        ),
        (
            "/agent new ollama expert who debugs models",
            "default-model",
            "ollama expert who debugs models",
            "ollama expert who debugs models",
        ),
    ],
)
def test_agent_new_reads_a_model_or_a_prompt(provider, line, model, instruction, name):
    session = open_session(provider)
    conversation(session)

    commands.dispatch(line, session)

    agent_id = session.agent.agent_id
    stored = record(provider, agent_id)
    assert stored.name == name
    assert stored.instruction == instruction
    assert stored.llm_config["model"] == model
    assert session.model_name == model
    assert cfg.load_state() == {"agent_id": agent_id}
    assert session.memory_id is None and session.thread_id is None
    assert "Created agent" in output(session)


def test_a_chosen_model_survives_a_relaunch(provider):
    session = open_session(provider)
    commands.dispatch("/agent new installed-model", session)
    session.agent.save()

    reopened = open_session(provider)
    reopened.agent.save()

    assert reopened.model_name == "installed-model"
    assert record(provider, reopened.agent.agent_id).llm_config["model"] == (
        "installed-model"
    )


def test_agent_new_reaches_another_providers_model(provider, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    session = open_session(provider)
    commands.dispatch("/agent new gpt-6-luna", session)
    assert session.provider_name == "openai"
    assert session.model_name == "gpt-6-luna"
    assert record(provider, session.agent.agent_id).llm_config["provider"] == "openai"


def test_an_azure_model_names_a_deployment(provider, monkeypatch):
    from memorizz.cli.memory_upkeep import llm_config_for

    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "the-default")
    config = llm_config_for(open_session(provider), "azure:my-deployment")
    assert config["deployment_name"] == "my-deployment"
    assert "model" not in config


def test_one_unknown_word_is_a_mistyped_model(provider):
    session = open_session(provider)
    before = session.agent.agent_id

    commands.dispatch("/agent new gpt-6-lunar", session)

    assert session.agent.agent_id == before
    assert len(provider.list_memagents()) == 1
    assert "Unknown model: gpt-6-lunar" in output(session)


def test_a_model_that_cannot_be_set_up_makes_nothing(provider):
    session = open_session(provider)
    before = session.agent.agent_id
    state = cfg.load_state()

    commands.dispatch("/agent new gpt-6-luna", session)

    assert session.agent.agent_id == before
    assert len(provider.list_memagents()) == 1
    assert cfg.load_state() == state
    assert "Could not set up gpt-6-luna" in output(session)


def test_two_agents_on_one_model_are_two_agents(provider):
    session = open_session(provider)
    commands.dispatch("/agent new installed-model", session)
    first = session.agent.agent_id
    commands.dispatch("/agent new installed-model", session)
    assert session.agent.agent_id != first
    assert len(provider.list_memagents()) == 3


def test_bare_agent_new_asks_for_the_model_then_the_prompt(provider, monkeypatch):
    session = open_session(provider, terminal=True)
    seen = {}

    def pick(items, **options):
        seen.update(options)
        return next(item for item in items if item["value"] == "gpt-6-luna")

    monkeypatch.setattr(commands, "_model_choices", lambda p: [])
    monkeypatch.setattr(picker, "pick", pick)
    monkeypatch.setattr(builtins, "input", lambda prompt: seen.update(q=prompt) or "")
    monkeypatch.setattr(
        ollama_probe, "list_models", lambda *a, **k: ["installed-model", "gpt-6-luna"]
    )

    commands.dispatch("/agent new", session)

    assert seen["title"] == "New agent: choose a model (ollama)"
    assert seen["q"] == "Prompt (Enter for the default): "
    assert session.model_name == "gpt-6-luna"
    assert session.agent.instruction == cfg.MEMORY_ASSISTANT_INSTRUCTION


def test_bare_agent_new_takes_the_typed_prompt(provider, monkeypatch):
    session = open_session(provider, terminal=True)
    monkeypatch.setattr(picker, "pick", lambda items, **options: items[0])
    monkeypatch.setattr(builtins, "input", lambda prompt: "  You are a chef.  ")
    commands.dispatch("/agent new", session)
    assert session.agent.instruction == "You are a chef."
    assert session.agent.name == "chef"


@pytest.mark.parametrize("cancel", ["picker", "prompt"])
def test_cancelling_bare_agent_new_makes_nothing(provider, monkeypatch, cancel):
    session = open_session(provider, terminal=True)
    before = session.agent.agent_id

    def interrupted(prompt):
        raise KeyboardInterrupt

    monkeypatch.setattr(
        picker, "pick", lambda items, **k: None if cancel == "picker" else items[0]
    )
    monkeypatch.setattr(builtins, "input", interrupted)

    commands.dispatch("/agent new", session)

    assert session.agent.agent_id == before
    assert len(provider.list_memagents()) == 1
    assert "No new agent" in output(session)


def test_bare_agent_new_without_a_terminal_takes_the_defaults(provider):
    session = open_session(provider)
    commands.dispatch("/agent new", session)
    assert len(provider.list_memagents()) == 2
    assert session.model_name == "default-model"
    assert session.agent.instruction == cfg.MEMORY_ASSISTANT_INSTRUCTION


def test_a_harness_keeps_running_turns_and_agent_new_says_so(provider, monkeypatch):
    monkeypatch.setattr(
        harness_session,
        "list_harness_status",
        lambda session, **k: [{"name": "codex", "ready": True}],
    )
    session = open_session(provider)
    harness_session.apply_harness(session, "codex")

    commands.dispatch("/agent new installed-model", session)

    assert session.harness == "codex"
    assert "Turns still run on codex" in output(session)


def test_agent_new_is_offered_by_tab_completion_and_usage(provider):
    assert "/agent new" in commands.command_completions()
    session = open_session(provider)
    commands.dispatch("/agent", session)
    assert "/agent new [model] [prompt]" in output(session)


def test_chat_keeps_the_tool_its_prompts_point_at(provider):
    # Long tool output is replaced by "use retrieve_tool_log_entry(...)". Taking
    # the tool away made the model fail, then fetch the same page again.
    tools = open_session(provider).agent.tool_manager.list_tools()
    assert "retrieve_tool_log_entry" in tools
    assert "list_recent_tool_logs" not in tools


def test_a_chat_records_what_it_sends_to_the_model(provider):
    assert open_session(provider).agent.capture_context_snapshots is True


def test_recording_what_is_sent_is_not_saved_on_the_agent(provider):
    pirate = saved_agent(provider)
    cfg.set_active_agent(pirate)
    open_session(provider).agent.save()
    assert not getattr(record(provider, pirate), "capture_context_snapshots", False)

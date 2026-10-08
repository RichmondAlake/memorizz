"""Session-scoped harness routing in the interactive CLI (/harness, /harnesses, --harness)."""

from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

from memorizz.cli import commands, harness_session
from memorizz.cli.agent_factory import Session

pytestmark = pytest.mark.unit


class FakeToolManager:
    def __init__(self):
        self.tools = set()

    def add(self, name):
        self.tools.add(name)

    def remove_tool(self, name):
        return bool(self.tools.discard(name) or True)


class FakeProvider:
    def __init__(self, agents=None):
        self.agents = dict(agents or {})

    def retrieve_memagent(self, agent_id):
        if agent_id not in self.agents:
            raise KeyError(agent_id)
        return self.agents[agent_id]

    def list_memagents(self):
        return list(self.agents.values())


class FakeService:
    def __init__(self, rows, runs=None):
        self.rows = rows
        self.compare_calls = []
        self.canceled = []
        self.runs = dict(runs or {})
        self.polls = 0

    memory_provider = None

    def list_harnesses(self, *, probe=True):
        if probe:
            return list(self.rows)
        return [{"name": row["name"]} for row in self.rows]

    def list_runs(self, limit=50):
        return []

    def start_compare(self, task, harnesses, models=None):
        self.compare_calls.append((task, list(harnesses), models))
        self.steps = [
            {"name": h, "harness": h, "run_id": f"run-{h}"} for h in harnesses
        ]
        return {"orchestration_id": "wf-1"}

    def get_orchestration(self, workflow_id):
        self.polls += 1
        status = "running" if self.polls < 2 else "succeeded"
        return {
            "orchestration_id": workflow_id,
            "status": status,
            "steps": self.steps,
            "summary": {"best": "codex"},
        }

    def get_run(self, run_id):
        return self.runs.get(run_id)

    def cancel_orchestration(self, workflow_id):
        self.canceled.append(workflow_id)
        return {"ok": True}

    def run(self, *a, **k):  # pragma: no cover - presence makes it a service
        raise AssertionError("not used")


class FakeAgent:
    """Only the attributes ``with_meta_harness`` touches, like a saved MemAgent."""

    def __init__(self, service=None, mode=None, default="auto", config=None):
        self.meta_harness = service
        self.meta_harness_mode = mode
        self.default_harness = default
        self.harness_config = dict(config or {})
        self._owns_meta_harness = False
        self._meta_harness_tools_registered = False
        self.tool_manager = FakeToolManager()
        self.delegates = []
        self.delegation_config = None
        self.agent_id = "agent-1"
        self.memory_provider = FakeProvider()
        self.approval_store = None
        self.attach_calls = []
        self.proposals = []

    def with_meta_harness(
        self, meta_harness=None, *, mode="delegate", default_harness="auto", config=None
    ):
        self.attach_calls.append(
            {
                "service": meta_harness,
                "mode": mode,
                "default_harness": default_harness,
                "config": config,
            }
        )
        if meta_harness is None:
            meta_harness = FakeService([{"name": "codex", "ready": True}])
            self._owns_meta_harness = True
        self.meta_harness = meta_harness
        self.meta_harness_mode = mode
        self.default_harness = default_harness
        self.harness_config = dict(config or {})
        if mode == "delegate" and not self._meta_harness_tools_registered:
            for name in harness_session.META_HARNESS_TOOL_NAMES:
                self.tool_manager.add(name)
            self._meta_harness_tools_registered = True
        return self

    def list_approval_proposals(self, status=None, limit=100):
        return [p for p in self.proposals if status in (None, p.get("status"))][:limit]


READY = [
    {"name": "codex", "ready": True, "version": "codex-cli 0.160.0"},
    {"name": "claude-code", "ready": True, "version": "2.1.293"},
    {"name": "deepseek", "ready": False, "error": "DEEPSEEK_API_KEY is not set."},
]


@pytest.fixture
def session():
    output = StringIO()
    agent = FakeAgent(service=FakeService(READY))
    sess = Session(
        agent=agent,
        provider=object(),
        llm_config={"provider": "ollama", "model": "qwen3.5:latest"},
        memory_id="mem-1",
        console=Console(file=output, color_system=None, width=140),
    )
    return sess, output


# --------------------------------------------------------------------------- #
# harness_session helpers
# --------------------------------------------------------------------------- #


def test_canonical_harness_aliases():
    assert harness_session.canonical_harness(" Claude_Code ") == "claude-code"
    assert harness_session.canonical_harness("cc") == "claude-code"
    assert harness_session.canonical_harness("CODEX") == "codex"
    assert harness_session.canonical_harness(None) == ""


def test_prompt_label_shows_active_harness_and_mode(session):
    sess, _ = session
    assert harness_session.prompt_label(sess) == "memorizz> "
    sess.harness = "codex"
    assert harness_session.prompt_label(sess) == "memorizz·codex> "
    sess.code_mode = True
    assert harness_session.prompt_label(sess) == "code·codex> "


def test_apply_harness_attaches_runtime_mode_for_the_session_only(session):
    sess, _ = session
    agent = sess.agent
    original = agent.meta_harness
    result = harness_session.apply_harness(sess, "codex")

    assert result == {
        "harness": "codex",
        "mode": "runtime",
        "ready": True,
        "memory_id": "mem-1",
    }
    assert sess.harness == "codex"
    assert agent.meta_harness_mode == "runtime"
    assert agent.default_harness == "codex"
    assert agent.meta_harness is original  # reuses the agent's service
    assert agent.attach_calls[-1]["mode"] == "runtime"
    # the saved configuration is remembered for /harness off
    assert sess.harness_backup == {
        "meta_harness": original,
        "meta_harness_mode": None,
        "default_harness": "auto",
        "harness_config": {},
        "owns_meta_harness": False,
        "tools_registered": False,
    }


def test_apply_harness_rejects_unknown_and_not_ready(session):
    sess, _ = session
    with pytest.raises(ValueError, match="Unknown harness 'gemini'"):
        harness_session.apply_harness(sess, "gemini")
    with pytest.raises(ValueError, match="not ready: DEEPSEEK_API_KEY"):
        harness_session.apply_harness(sess, "deepseek")
    assert sess.harness is None
    assert sess.harness_backup is None
    assert sess.agent.attach_calls == []


def test_apply_harness_auto_needs_no_specific_harness(session):
    sess, _ = session
    result = harness_session.apply_harness(sess, "auto")
    assert result["harness"] == "auto" and result["ready"] is True
    assert sess.agent.default_harness == "auto"
    assert sess.agent.meta_harness_mode == "runtime"


def test_switching_twice_keeps_the_first_backup_and_live_service(session):
    sess, _ = session
    harness_session.apply_harness(sess, "codex")
    first_backup = dict(sess.harness_backup)
    harness_session.apply_harness(sess, "claude-code")
    assert sess.harness == "claude-code"
    assert sess.harness_backup == first_backup
    assert sess.agent.attach_calls[-1]["service"] is sess.agent.meta_harness


def test_clear_harness_restores_the_saved_agent_configuration():
    output = StringIO()
    service = FakeService(READY)
    agent = FakeAgent(
        service=service,
        mode="delegate",
        default="claude-code",
        config={"budget": {"max_steps": 5}},
    )
    sess = Session(
        agent=agent, provider=object(), llm_config={}, console=Console(file=output)
    )
    harness_session.apply_harness(sess, "codex")
    assert agent.meta_harness_mode == "runtime" and agent.default_harness == "codex"

    result = harness_session.clear_harness(sess)

    assert result == {"harness": None, "previous": "codex", "mode": "delegate"}
    assert sess.harness is None and sess.harness_backup is None
    assert agent.meta_harness is service
    assert agent.meta_harness_mode == "delegate"
    assert agent.default_harness == "claude-code"
    assert agent.harness_config == {"budget": {"max_steps": 5}}
    assert agent._owns_meta_harness is False


def test_clear_harness_without_an_active_one_is_a_no_op(session):
    sess, _ = session
    assert harness_session.clear_harness(sess) == {
        "harness": None,
        "previous": None,
        "mode": None,
    }


def test_apply_harness_creates_a_service_when_the_agent_has_none(monkeypatch):
    output = StringIO()
    agent = FakeAgent(service=None)
    sess = Session(
        agent=agent, provider=object(), llm_config={}, console=Console(file=output)
    )
    created = FakeService(READY)
    from memorizz import metaharness

    monkeypatch.setattr(
        metaharness.MetaHarness, "from_env", staticmethod(lambda **k: created)
    )
    harness_session.apply_harness(sess, "codex")
    # listing used a service built from the agent's provider; attach got None so the
    # agent builds and owns its own
    assert agent.attach_calls[-1]["service"] is None
    assert agent._owns_meta_harness is True
    harness_session.clear_harness(sess)
    assert agent.meta_harness is None and agent._owns_meta_harness is False


def test_report_pending_approvals_only_when_a_harness_is_active(session):
    sess, output = session
    sess.agent.proposals = [
        {
            "proposal_id": "prop-1",
            "status": "pending",
            "tool_name": "apply_patch",
            "policy_reason": "workspace write",
        },
        {"proposal_id": "prop-2", "status": "approved", "tool_name": "x"},
    ]
    assert harness_session.report_pending_approvals(sess, sess.console) == 0
    sess.harness = "codex"
    assert harness_session.report_pending_approvals(sess, sess.console) == 1
    text = output.getvalue()
    assert (
        "1 approval(s) waiting" in text and "prop-1" in text and "apply_patch" in text
    )
    assert "/approvals approve prop-id" not in text
    assert "/approvals approve <proposal-id>" in text


# --------------------------------------------------------------------------- #
# slash commands
# --------------------------------------------------------------------------- #


def test_harnesses_command_lists_readiness(session):
    sess, output = session
    assert commands.dispatch("/harnesses", sess) is True
    text = output.getvalue()
    assert "codex" in text and "ready" in text
    assert "deepseek" in text and "not ready" in text and "DEEPSEEK_API_KEY" in text
    assert "active: off" in text


def test_harness_command_switches_and_turns_off(session):
    sess, output = session
    assert commands.dispatch("/harness codex", sess) is True
    assert sess.harness == "codex"
    assert "Harness → codex" in output.getvalue()
    assert "this session only" in output.getvalue()

    assert commands.dispatch("/harness", sess) is True
    assert "Harness: codex" in output.getvalue()

    assert commands.dispatch("/harness off", sess) is True
    assert sess.harness is None
    assert sess.agent.meta_harness_mode is None
    assert "Harness off (was codex)" in output.getvalue()


def test_harness_command_reports_bad_names_without_changing_state(session):
    sess, output = session
    assert commands.dispatch("/harness nope", sess) is True
    assert "Unknown harness 'nope'" in output.getvalue()
    assert sess.harness is None
    assert sess.agent.meta_harness_mode is None


def test_harness_commands_are_completable_and_listed_in_help(session):
    sess, output = session
    names = commands.command_completions()
    assert "/harness" in names and "/harnesses" in names
    commands.dispatch("/help", sess)
    assert "/harness <name|auto|delegate|off>" in output.getvalue()
    assert "/compare <harness> <harness> <task>" in output.getvalue()


# --------------------------------------------------------------------------- #
# launch flags
# --------------------------------------------------------------------------- #


def test_chat_and_run_accept_the_harness_flag(monkeypatch):
    from typer.testing import CliRunner

    from memorizz.cli import app as cli_app

    captured = {}
    monkeypatch.setattr(cli_app, "_launch_repl", lambda **kw: captured.update(kw))
    monkeypatch.setattr(
        cli_app,
        "_run_oneshot",
        lambda text, **kw: captured.update({"text": text, **kw}),
    )

    runner = CliRunner()
    assert runner.invoke(cli_app.app, ["chat", "--harness", "codex"]).exit_code == 0
    assert captured["harness"] == "codex"
    captured.clear()
    assert (
        runner.invoke(
            cli_app.app, ["run", "hello", "--harness", "claude-code", "--no-stream"]
        ).exit_code
        == 0
    )
    assert captured["harness"] == "claude-code" and captured["text"] == "hello"


def test_apply_session_harness_exits_on_a_bad_name(session):
    import typer

    from memorizz.cli import app as cli_app

    sess, output = session
    with pytest.raises(typer.Exit):
        cli_app._apply_session_harness(sess, "nope", sess.console)
    assert "Unknown harness 'nope'" in output.getvalue()
    cli_app._apply_session_harness(sess, "codex", sess.console)
    assert sess.harness == "codex" and "Harness → codex" in output.getvalue()


# --------------------------------------------------------------------------- #
# /harness delegate
# --------------------------------------------------------------------------- #


def _codex_delegate(agent_id="dlg-1", name="Codex reviewer", harness="codex"):
    return SimpleNamespace(
        agent_id=agent_id,
        name=name,
        meta_harness_mode="runtime",
        default_harness=harness,
        harness_config={"model": "gpt-6-luna"},
    )


def test_harness_delegates_lists_only_harness_backed_delegates(session):
    sess, _ = session
    agent = sess.agent
    agent.memory_provider = FakeProvider(
        {
            "dlg-1": _codex_delegate(),
            "dlg-2": SimpleNamespace(
                agent_id="dlg-2", name="Model delegate", meta_harness_mode=None
            ),
        }
    )
    agent.delegates = ["dlg-1", "dlg-2", "missing"]
    rows = harness_session.harness_delegates(sess)
    assert rows == [
        {
            "id": "dlg-1",
            "name": "Codex reviewer",
            "harness": "codex",
            "model": "gpt-6-luna",
        }
    ]
    assert harness_session.delegation_enabled(sess) is True
    agent.delegation_config = {"enabled": False}
    assert harness_session.delegation_enabled(sess) is False


def test_apply_delegate_mode_registers_tools_and_off_removes_them(session):
    sess, _ = session
    agent = sess.agent
    agent.memory_provider = FakeProvider({"dlg-1": _codex_delegate()})
    agent.delegates = ["dlg-1"]
    result = harness_session.apply_harness(sess, "delegate")

    assert result["harness"] == "delegate" and result["mode"] == "delegate"
    assert result["delegates"][0]["name"] == "Codex reviewer"
    assert result["delegation_enabled"] is True
    assert result["tools"] == list(harness_session.META_HARNESS_TOOL_NAMES)
    assert agent.meta_harness_mode == "delegate"
    assert agent.attach_calls[-1]["mode"] == "delegate"
    assert agent.attach_calls[-1]["default_harness"] == "auto"
    assert agent.tool_manager.tools == set(harness_session.META_HARNESS_TOOL_NAMES)
    assert sess.harness == "delegate"
    assert harness_session.prompt_label(sess) == "memorizz·delegate> "

    harness_session.clear_harness(sess)
    assert agent.meta_harness_mode is None
    assert agent.tool_manager.tools == set()
    assert agent._meta_harness_tools_registered is False


def test_off_keeps_tools_the_saved_agent_already_had():
    output = StringIO()
    agent = FakeAgent(service=FakeService(READY), mode="delegate")
    agent._meta_harness_tools_registered = True
    for name in harness_session.META_HARNESS_TOOL_NAMES:
        agent.tool_manager.add(name)
    sess = Session(
        agent=agent, provider=object(), llm_config={}, console=Console(file=output)
    )
    harness_session.apply_harness(sess, "codex")
    harness_session.clear_harness(sess)
    assert agent.meta_harness_mode == "delegate"
    assert agent.tool_manager.tools == set(harness_session.META_HARNESS_TOOL_NAMES)
    assert agent._meta_harness_tools_registered is True


def test_harness_delegate_command_lists_delegates_or_explains_how_to_create_one(
    session,
):
    sess, output = session
    assert commands.dispatch("/harness delegate", sess) is True
    text = output.getvalue()
    assert "Harness → delegate" in text
    assert "No harness delegates yet" in text
    assert (
        "memorizz harness delegate create --harness codex --coordinator agent-1 --attach"
        in text
    )
    assert "run_harness_task" in text

    sess.agent.memory_provider = FakeProvider({"dlg-1": _codex_delegate()})
    sess.agent.delegates = ["dlg-1"]
    commands.dispatch("/harness delegate", sess)
    text = output.getvalue()
    assert "Harness delegates (1, delegation on)" in text
    assert "Codex reviewer (codex)" in text

    commands.dispatch("/harness", sess)
    assert "Harness: delegate" in output.getvalue()
    commands.dispatch("/harness off", sess)
    assert sess.harness is None and sess.agent.meta_harness_mode is None


# --------------------------------------------------------------------------- #
# /compare
# --------------------------------------------------------------------------- #


def test_parse_compare_args_splits_harnesses_from_the_task(session):
    sess, _ = session
    names, task, _ = harness_session.parse_compare_args(
        sess, "codex claude-code where can totals lose precision?"
    )
    assert names == ["codex", "claude-code"]
    assert task == "where can totals lose precision?"
    names, task, _ = harness_session.parse_compare_args(
        sess, "codex cc codex explain the auth module"
    )
    assert names == ["codex", "claude-code"] and task == "explain the auth module"


def test_parse_compare_args_usage_errors(session):
    sess, _ = session
    with pytest.raises(ValueError, match="Usage: /compare"):
        harness_session.parse_compare_args(sess, "")
    with pytest.raises(ValueError, match="Usage: /compare"):
        harness_session.parse_compare_args(sess, "codex explain the auth module")
    with pytest.raises(ValueError, match="Usage: /compare"):
        harness_session.parse_compare_args(sess, "codex claude-code")
    with pytest.raises(ValueError, match="'gemini' is not a configured harness"):
        harness_session.parse_compare_args(sess, "gemini codex explain")


def test_run_compare_builds_the_task_from_the_session_and_reports_each_harness(
    session, monkeypatch
):
    sess, _ = session
    sess.user_id = "user-7"
    sess.thread_id = "thread-3"
    service = FakeService(
        READY,
        runs={
            "run-codex": {
                "harness": "codex",
                "status": "succeeded",
                "result": {
                    "final_response": "Totals round early.",
                    "verified": True,
                    "cost_usd": 0.0,
                    "latency_ms": 14200,
                },
            },
            "run-claude-code": {
                "harness": "claude-code",
                "status": "failed",
                "result": {"error": "login expired"},
            },
        },
    )
    sess.agent.meta_harness = service
    monkeypatch.setattr(harness_session.time, "sleep", lambda s: None)
    lines = []
    report = harness_session.run_compare(
        sess,
        ["codex", "claude-code"],
        "where do totals lose precision?",
        workspace="/tmp/ws",
        on_progress=lines.append,
    )

    task, names, models = service.compare_calls[0]
    assert names == ["codex", "claude-code"] and models is None
    assert task.task == "where do totals lose precision?"
    assert task.workspace == "/tmp/ws"
    assert (
        task.memory_id == "mem-1"
        and task.user_id == "user-7"
        and task.thread_id == "thread-3"
    )
    assert task.agent_id == "agent-1"
    assert report["workflow_id"] == "wf-1" and report["status"] == "succeeded"
    assert report["summary"] == {"best": "codex"}
    assert report["rows"][0] == {
        "harness": "codex",
        "status": "succeeded",
        "verified": True,
        "cost_usd": 0.0,
        "latency_ms": 14200,
        "answer": "Totals round early.",
        "error": None,
        "run_id": "run-codex",
    }
    assert (
        report["rows"][1]["status"] == "failed"
        and report["rows"][1]["error"] == "login expired"
    )
    assert lines[0] == "Comparison wf-1 started."
    assert "[1/2] codex: succeeded" in lines and "[2/2] claude-code: failed" in lines


def test_run_compare_refuses_a_harness_that_is_not_ready(session):
    sess, _ = session
    with pytest.raises(ValueError, match="not ready: DEEPSEEK_API_KEY"):
        harness_session.run_compare(sess, ["codex", "deepseek"], "task")
    with pytest.raises(ValueError, match="Unknown harness 'gemini'"):
        harness_session.run_compare(sess, ["codex", "gemini"], "task")


def test_run_compare_ctrl_c_cancels_the_workflow(session, monkeypatch):
    sess, _ = session
    service = FakeService(
        READY,
        runs={
            "run-codex": {"harness": "codex", "status": "running"},
            "run-claude-code": {"harness": "claude-code", "status": "running"},
        },
    )
    sess.agent.meta_harness = service

    polls = {"n": 0}

    def get_orchestration(workflow_id):
        polls["n"] += 1
        if polls["n"] == 1:
            return {"status": "running", "steps": service.steps}
        if polls["n"] == 2:
            raise KeyboardInterrupt
        return {"status": "canceled", "steps": service.steps}

    service.get_orchestration = get_orchestration
    monkeypatch.setattr(harness_session.time, "sleep", lambda s: None)
    lines = []
    report = harness_session.run_compare(
        sess, ["codex", "claude-code"], "task", on_progress=lines.append
    )
    assert service.canceled == ["wf-1"]
    assert report["status"] == "canceled"
    assert "Canceling the comparison…" in lines


def test_compare_command_prints_the_verdict_table(session, monkeypatch):
    sess, output = session
    service = FakeService(
        READY,
        runs={
            "run-codex": {
                "harness": "codex",
                "status": "succeeded",
                "result": {
                    "final_response": "Totals round early in sum().",
                    "cost_usd": 0.0,
                    "latency_ms": 14200,
                },
            },
            "run-claude-code": {
                "harness": "claude-code",
                "status": "succeeded",
                "result": {
                    "final_response": "<b>bold</b> answer",
                    "cost_usd": 0.0312,
                    "latency_ms": 21700,
                },
            },
        },
    )
    sess.agent.meta_harness = service
    monkeypatch.setattr(harness_session.time, "sleep", lambda s: None)
    assert (
        commands.dispatch(
            "/compare codex claude-code where do totals lose precision?", sess
        )
        is True
    )
    text = output.getvalue()
    assert "Comparing codex vs claude-code on: where do totals lose precision?" in text
    assert "Comparison wf-1" in text and "succeeded" in text
    assert "Totals round early in sum()." in text
    assert "<b>bold</b> answer" in text
    assert "$0.0312" in text and "21.7s" in text and "14.2s" in text
    assert "memorizz harness show-workflow wf-1" in text


def test_compare_command_usage_and_errors(session):
    sess, output = session
    commands.dispatch("/compare", sess)
    assert "Usage: /compare <harness> <harness>" in output.getvalue()
    commands.dispatch("/compare codex deepseek task", sess)
    assert "not ready: DEEPSEEK_API_KEY" in output.getvalue()
    assert "/compare" in commands.command_completions()


def test_compare_with_memagent_runs_another_saved_agent_never_the_chats_own(
    session, monkeypatch
):
    sess, _ = session
    service = FakeService(
        READY + [{"name": "memagent", "ready": True}],
        runs={
            "run-codex": {
                "harness": "codex",
                "status": "succeeded",
                "result": {"final_response": "A"},
            },
            "run-memagent": {
                "harness": "memagent",
                "status": "succeeded",
                "result": {"final_response": "B"},
            },
        },
    )
    sess.agent.meta_harness = service
    monkeypatch.setattr(harness_session.time, "sleep", lambda s: None)
    # only the chat's own agent is saved: refuse clearly
    sess.agent.memory_provider = FakeProvider(
        {
            "agent-1": SimpleNamespace(
                agent_id="agent-1", name="Me", created_at="2026-10-01"
            )
        }
    )
    with pytest.raises(ValueError, match="can't hand a task to itself"):
        harness_session.run_compare(sess, ["codex", "memagent"], "task")
    assert service.compare_calls == []

    sess.agent.memory_provider = FakeProvider(
        {
            "agent-1": SimpleNamespace(
                agent_id="agent-1", name="Me", created_at="2026-10-09"
            ),
            "other-2": SimpleNamespace(
                agent_id="other-2", name="Local reviewer", created_at="2026-10-02"
            ),
        }
    )
    lines = []
    report = harness_session.run_compare(
        sess, ["codex", "memagent"], "task", on_progress=lines.append
    )
    task, names, _ = service.compare_calls[0]
    assert names == ["codex", "memagent"]
    assert task.agent_id == "other-2"
    assert lines[0] == "memagent will run Local reviewer (other-2)."
    assert [row["answer"] for row in report["rows"]] == ["A", "B"]


def test_compare_memagent_equals_picks_a_saved_agent_by_id_prefix(session, monkeypatch):
    sess, _ = session
    service = FakeService(
        READY + [{"name": "memagent", "ready": True}],
        runs={
            "run-codex": {"harness": "codex", "status": "succeeded", "result": {}},
            "run-memagent": {
                "harness": "memagent",
                "status": "succeeded",
                "result": {},
            },
        },
    )
    sess.agent.meta_harness = service
    sess.agent.memory_provider = FakeProvider(
        {
            "agent-1": SimpleNamespace(agent_id="agent-1", name="Me"),
            "other-2": SimpleNamespace(agent_id="other-2", name="Local reviewer"),
            "other-3": SimpleNamespace(agent_id="other-3", name="Second"),
        }
    )
    monkeypatch.setattr(harness_session.time, "sleep", lambda s: None)
    names, task, options = harness_session.parse_compare_args(
        sess, "codex memagent=other-2 read the readme"
    )
    assert names == ["codex", "memagent"] and task == "read the readme"
    assert options == {"memagent_id": "other-2"}
    harness_session.run_compare(sess, names, task, **options)
    assert service.compare_calls[-1][0].agent_id == "other-2"
    with pytest.raises(ValueError, match="more than one saved agent"):
        harness_session.run_compare(sess, names, task, memagent_id="other")
    with pytest.raises(ValueError, match="own agent"):
        harness_session.run_compare(sess, names, task, memagent_id="agent-1")
    with pytest.raises(ValueError, match="only memagent takes"):
        harness_session.parse_compare_args(sess, "codex=x claude-code task")

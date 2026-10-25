"""Computer use from the chat: /desktop and the y/n asked for each action."""

from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

from memorizz import capability_gaps
from memorizz.approval import ApprovalResumeResult
from memorizz.cli import commands, desktop_commands, inline_approvals

pytestmark = pytest.mark.unit

TOOLS = [
    {"name": "read_window", "inputSchema": {"type": "object"}},
    {"name": "open_url", "inputSchema": {"type": "object"}},
]


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def session(monkeypatch):
    from memorizz import MemAgent

    agent = MemAgent(memory_provider=False, auto_register=False)
    out = StringIO()
    monkeypatch.setattr(
        desktop_commands,
        "_permissions",
        lambda prompt=False: ({"accessibility": True, "screen_recording": False}, None),
    )
    monkeypatch.setattr(desktop_commands, "_terminal", lambda: "cmux")
    started = []

    def list_tools(name):
        started.append(name)
        return {"ok": True, "tools": TOOLS}

    monkeypatch.setattr(agent.mcp_manager, "list_tools", list_tools)
    monkeypatch.setattr(agent.mcp_manager, "cached_tools", lambda name: TOOLS)
    yield SimpleNamespace(
        agent=agent,
        provider=None,
        console=Console(file=out, width=200, color_system=None),
        out=out,
        started=started,
        user_id=None,
    )
    agent.close()


def test_desktop_on_gives_the_running_agent_the_tools_and_saves_them(session):
    from memorizz.cli import mcp_config

    agent = session.agent
    assert commands.dispatch("/desktop on", session)
    assert session.started == ["desktop"]
    [server] = agent.mcp_servers
    assert server["name"] == "desktop" and server["require_approval"] is True
    assert server["args"][-3:-1] == ["serve", "--state"]
    assert {"desktop__read_window", "desktop__open_url"} <= set(
        agent.tool_manager.list_tools()
    )
    assert [s["name"] for s in mcp_config.load_servers(agent.agent_id)] == ["desktop"]
    text = session.out.getvalue()
    assert "Desktop control on · 2 tools" in text and "ask you here first" in text

    # The agent no longer lists desktop control among what it lacks.
    rows = {row["id"]: row for row in capability_gaps.state_for_agent(agent)}
    if "desktop" in rows:  # only listed on a Mac
        assert rows["desktop"]["status"] == "enabled"

    commands.dispatch("/desktop", session)
    assert "Desktop control: on · desktop" in session.out.getvalue()

    # On again keeps one connection, under the name it already has.
    commands.dispatch("/desktop on --no-screenshots", session)
    [server] = agent.mcp_servers
    assert server["args"][-1] == "--no-screenshots"


def test_desktop_off_removes_the_tools_and_the_saved_connection(session):
    from memorizz.cli import mcp_config

    agent = session.agent
    commands.dispatch("/computer-use on", session)
    commands.dispatch("/desktop off", session)
    assert agent.mcp_servers == []
    assert not any(
        name.startswith("desktop__") for name in agent.tool_manager.list_tools()
    )
    assert mcp_config.load_servers(agent.agent_id) == []
    assert "Desktop control OFF" in session.out.getvalue()
    commands.dispatch("/desktop off", session)
    assert "already off" in session.out.getvalue()


def test_desktop_on_says_why_when_this_machine_cannot(session, monkeypatch):
    monkeypatch.setattr(
        desktop_commands,
        "_permissions",
        lambda prompt=False: (None, "Desktop control needs macOS; this is linux."),
    )
    commands.dispatch("/desktop on", session)
    assert session.agent.mcp_servers == []
    assert "needs macOS" in session.out.getvalue()


def test_desktop_on_without_accessibility_still_opens_things(session, monkeypatch):
    monkeypatch.setattr(
        desktop_commands,
        "_permissions",
        lambda prompt=False: (
            {"accessibility": False, "screen_recording": False},
            None,
        ),
    )
    commands.dispatch("/desktop on", session)
    text = session.out.getvalue()
    assert "cmux doesn't have Accessibility yet" in text
    assert "/desktop doctor --ask" in text
    assert session.agent.mcp_servers


def test_a_desktop_request_names_the_desktop_capability():
    assert capability_gaps.resolve("computer use") == "desktop"
    assert capability_gaps.resolve("Desktop control") == "desktop"
    assert capability_gaps.resolve("browser use") == "browser"
    mac = capability_gaps.capability_state(environ={}, platform="darwin")
    assert capability_gaps.suggest(
        "i mean open it in my actual desktop with the notion desktop applicaiton",
        mac,
    ) == ["notes", "desktop"]
    assert capability_gaps.suggest("open the website example.com", mac) == ["browser"]
    desktop = next(row for row in mac if row["id"] == "desktop")
    assert desktop["offers"] == [
        {
            "kind": "command",
            "command": "/desktop on",
            "title": "Run /desktop on in the memorizz chat",
        }
    ]
    linux = capability_gaps.capability_state(environ={}, platform="linux")
    assert "desktop" not in {row["id"] for row in linux}
    attached = capability_gaps.capability_state(
        mcp_servers=[desktop_commands.server_config()],
        signed_in={"desktop": True},
        environ={},
        platform="darwin",
    )
    assert next(row for row in attached if row["id"] == "desktop")["status"] == (
        "enabled"
    )


class Approvals:
    """The agent's approval store, as the CLI sees it."""

    def __init__(self, *proposals, then=None):
        self.rows = {p["proposal_id"]: dict(p, status="pending") for p in proposals}
        self.then = dict(then or {})  # resuming one raises the next
        self.calls = []

    def list_approval_proposals(self, status=None, limit=100):
        return [r for r in self.rows.values() if status in (None, r["status"])]

    def approve(self, proposal_id, *, approver_id, reason=None):
        self.calls.append(("approve", proposal_id, approver_id))
        self.rows[proposal_id]["status"] = "approved"

    def reject(self, proposal_id, *, approver_id, reason=None):
        self.calls.append(("reject", proposal_id, approver_id))
        self.rows[proposal_id]["status"] = "rejected"

    def resume_approval(self, proposal_id):
        self.calls.append(("resume", proposal_id))
        self.rows[proposal_id]["status"] = "consumed"
        following = self.then.get(proposal_id)
        if following:
            self.rows[following["proposal_id"]] = dict(following, status="pending")
            return ApprovalResumeResult(proposal=None, assistant_response="{paused}")
        return ApprovalResumeResult(
            proposal=None, assistant_response="Opened **THE ALGORITHM**."
        )


OPEN = {
    "proposal_id": "p1",
    "tool_name": "desktop__open_url",
    "arguments": {"url": "https://app.notion.com/p/8e5d", "app": "Notion"},
}
CLICK = {
    "proposal_id": "p2",
    "tool_name": "desktop__click",
    "arguments": {"element": 4},
}


def _chat(agent, answers, monkeypatch):
    out = StringIO()
    asked = []
    replies = list(answers)

    def answer(prompt=""):
        if not replies:
            raise EOFError
        reply = replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    monkeypatch.setattr("builtins.input", answer)
    console = Console(file=out, width=200, color_system=None)
    original = console.input
    console.input = lambda prompt="", **k: asked.append(prompt) or original(prompt)
    return SimpleNamespace(agent=agent, console=console, user_id="richmond"), asked, out


def test_yes_runs_the_step_and_asks_about_the_next_one(monkeypatch):
    agent = Approvals(OPEN, then={"p1": CLICK})
    session, asked, out = _chat(agent, ["y", "yes"], monkeypatch)
    inline_approvals.ask(session, ["p1", None, "p1"])
    assert agent.calls == [
        ("approve", "p1", "richmond"),
        ("resume", "p1"),
        ("approve", "p2", "richmond"),
        ("resume", "p2"),
    ]
    assert 'desktop__open_url(url="https://app.notion.com/p/8e5d", app="Notion")' in (
        asked[0]
    )
    assert "desktop__click(element=4)" in asked[1]
    text = out.getvalue()
    assert "{paused}" not in text  # the next question replaces the pause notice
    assert "Opened THE ALGORITHM." in text


def test_no_rejects_and_ctrl_c_leaves_it_for_approvals(monkeypatch):
    agent = Approvals(OPEN, CLICK)
    session, asked, out = _chat(agent, ["", KeyboardInterrupt()], monkeypatch)
    inline_approvals.ask(session, ["p1", "p2"])
    assert agent.calls == [("reject", "p1", "richmond")]
    assert agent.rows["p2"]["status"] == "pending"
    text = out.getvalue()
    assert "Not done." in text
    assert "/approvals approve p2 richmond" in text


def test_a_long_argument_is_shortened_in_the_question():
    row = {"tool_name": "desktop__type_text", "arguments": {"text": "x" * 500}}
    assert len(inline_approvals.describe(row)) < 160


def test_the_repl_asks_about_approvals_after_the_turn(monkeypatch):
    from memorizz.cli import connect, repl, streaming

    class Stream:
        def __init__(self, events):
            self.events = list(events)

        def poll(self, timeout=None):
            if not self.events:
                raise StopIteration
            return self.events.pop(0)

        def close(self):
            pass

    events = [
        {"type": "run.started", "memory_id": "m1", "thread_id": "t1"},
        {"type": "tool.started", "tool_name": "desktop__open_url"},
        {"type": "approval.required", "proposal": {"proposal_id": "p1"}},
        {"type": "run.done", "status": "approval_required"},
    ]
    agent = SimpleNamespace(
        agent_id="agent-1", run_stream_events=lambda query, **k: Stream(events)
    )
    session = SimpleNamespace(
        agent=agent,
        memory_id=None,
        thread_id=None,
        user_id=None,
        console=Console(file=StringIO(), force_terminal=True, color_system=None),
    )
    asked = []
    monkeypatch.setattr(streaming, "save_stream_session", lambda s: "written")
    monkeypatch.setattr(connect, "offer", lambda s, ids: None)
    monkeypatch.setattr(inline_approvals, "ask", lambda s, ids: asked.append(ids))
    repl._stream_turn(session, "open the algorithm page in my notion app")
    assert asked == [["p1"]]

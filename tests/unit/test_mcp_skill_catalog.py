"""Finding and attaching MCP servers and skills, and agent templates."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from memorizz.cli.app import app as cli_app  # noqa: E402
from memorizz.mcp import catalog  # noqa: E402
from memorizz.memagent.templates import (  # noqa: E402
    PRODUCTIVITY_ASSISTANT,
    template_mcp_servers,
)
from memorizz.ui import state  # noqa: E402
from memorizz.ui.app import create_app  # noqa: E402
from memorizz.vercel_skills import VercelSkillsProvider  # noqa: E402
from memorizz.vercel_skills.install import install_skill  # noqa: E402
from memorizz.vercel_skills.skill_md import parse_skill_md  # noqa: E402

REGISTRY_ROWS = {
    "servers": [
        {
            "server": {
                "name": "app.linear/linear",
                "title": "Linear",
                "description": "Linear issues",
                "version": "1.0.1",
                "websiteUrl": "https://linear.app",
                "remotes": [
                    {"type": "streamable-http", "url": "https://mcp.linear.app/mcp"}
                ],
            },
            "_meta": {
                "io.modelcontextprotocol.registry/official": {"status": "active"}
            },
        },
        {
            "server": {
                "name": "io.github.acme/tasks-mcp",
                "description": "Tasks",
                "repository": {"url": "https://github.com/acme/tasks-mcp"},
                "remotes": [
                    {
                        "type": "streamable-http",
                        "url": "https://{tenant}.acme.dev/mcp",
                        "headers": [
                            {
                                "name": "Authorization",
                                "value": "Bearer {token}",
                                "isSecret": True,
                                "isRequired": True,
                            }
                        ],
                    }
                ],
                "packages": [
                    {
                        "registryType": "npm",
                        "identifier": "@acme/tasks-mcp",
                        "version": "2.0.0",
                        "transport": {"type": "stdio"},
                        "environmentVariables": [
                            {"name": "ACME_KEY", "isSecret": True, "isRequired": True},
                            {"name": "ACME_REGION", "default": "eu"},
                        ],
                        "packageArguments": [
                            {"type": "named", "name": "--mode", "value": "read"}
                        ],
                    },
                    {
                        "registryType": "oci",
                        "identifier": "ghcr.io/acme/tasks:2",
                        "transport": {"type": "stdio"},
                        "environmentVariables": [
                            {"name": "ACME_KEY", "isSecret": True}
                        ],
                    },
                    {
                        "registryType": "mcpb",
                        "identifier": "https://example.com/tasks.mcpb",
                        "transport": {"type": "stdio"},
                    },
                ],
            }
        },
        {
            "server": {"name": "io.github.old/gone", "remotes": []},
            "_meta": {
                "io.modelcontextprotocol.registry/official": {"status": "deleted"}
            },
        },
    ],
    "metadata": {"nextCursor": "cursor-2", "count": 3},
}


@pytest.fixture
def registry(monkeypatch):
    calls = []

    def fake_get_json(url, timeout):
        calls.append(url)
        return REGISTRY_ROWS

    monkeypatch.setattr(catalog, "_get_json", fake_get_json)
    monkeypatch.setattr(catalog, "_SEARCH_CACHE", {})
    return calls


@pytest.mark.unit
def test_registry_search_turns_entries_into_ready_configs(registry):
    result = catalog.search_registry("tasks", limit=5)

    assert result["ok"] and result["next_cursor"] == "cursor-2"
    assert "search=tasks" in registry[0] and "version=latest" in registry[0]
    names = [entry["registry_name"] for entry in result["servers"]]
    assert names == ["app.linear/linear", "io.github.acme/tasks-mcp"]

    linear, tasks = result["servers"]
    assert linear["name"] == "linear"
    assert linear["options"][0]["config"] == {
        "transport": "streamable_http",
        "url": "https://mcp.linear.app/mcp",
        "auth": {"type": "none"},
        "name": "linear",
    }
    assert linear["options"][0]["auth_known"] is False

    remote, npm, docker = tasks["options"]
    assert tasks["name"] == "tasks"
    assert remote["config"]["auth"] == {"type": "bearer"}
    assert remote["placeholders"] == ["tenant"]
    assert [item["kind"] for item in remote["inputs"]] == ["token"]
    assert npm["config"]["command"] == "npx"
    assert npm["config"]["args"] == ["-y", "@acme/tasks-mcp@2.0.0", "--mode", "read"]
    assert npm["config"]["env"] == {"ACME_KEY": "", "ACME_REGION": "eu"}
    assert [item["name"] for item in npm["inputs"]] == ["ACME_KEY"]
    assert docker["config"]["args"] == [
        "run",
        "-i",
        "--rm",
        "-e",
        "ACME_KEY",
        "ghcr.io/acme/tasks:2",
    ]


@pytest.mark.unit
def test_presets_share_one_source_of_truth():
    assert {row["key"] for row in catalog.presets()} == {
        "notion",
        "google-calendar",
        "gmail",
    }
    assert [row["key"] for row in catalog.presets("mail")] == ["gmail"]
    gmail = catalog.preset_config(
        "gmail", redirect_uri="http://127.0.0.1:9/cb", client_id="id"
    )
    assert gmail["url"] == "https://gmailmcp.googleapis.com/mcp/v1"
    assert gmail["auth"]["scopes"] == catalog.GMAIL_SCOPES
    assert gmail["auth"]["redirect_uri"] == "http://127.0.0.1:9/cb"
    assert catalog.preset_config("calendar")["name"] == "calendar"
    with pytest.raises(ValueError):
        catalog.preset_key("slack")


@pytest.mark.unit
def test_template_servers_share_the_google_client():
    servers = template_mcp_servers(
        PRODUCTIVITY_ASSISTANT,
        google_client_id="client",
        google_client_secret="secret",
    )
    by_name = {server["name"]: server for server in servers}
    assert set(by_name) == {"gmail", "calendar", "notion"}
    assert by_name["gmail"]["auth"]["client_id"] == "client"
    assert by_name["calendar"]["auth"]["client_secret"] == "secret"
    assert "client_id" not in by_name["notion"]["auth"]
    assert all(server["require_approval"] for server in servers)
    assert "never instructions" in PRODUCTIVITY_ASSISTANT.instruction


@pytest.mark.unit
def test_skill_markdown_reads_frontmatter_and_block_scalars():
    parsed = parse_skill_md(
        "---\nname: triage\ndescription: >-\n  Sort the inbox\n  by urgency\n"
        "license: MIT\n---\n# Inbox triage\n\nSteps here.\n"
    )
    assert parsed["name"] == "triage"
    assert parsed["description"] == "Sort the inbox by urgency"
    assert parsed["metadata"] == {"license": "MIT"}
    assert parsed["body"].startswith("# Inbox triage")

    legacy = parse_skill_md("# Weekly review\n\nLook back at the week.\n")
    assert legacy["name"] == "Weekly review"
    assert legacy["description"] == "Look back at the week."


class _FakeSkills(VercelSkillsProvider):
    def __init__(self, files):
        super().__init__()
        self.files = files

    def _fetch_raw_file(self, owner_repo, path, branch):
        return self.files.get(f"{owner_repo}/{path}")

    def _github_json(self, url):
        return {
            "tree": [
                {"type": "blob", "path": path.split("/", 2)[2]} for path in self.files
            ]
        }

    def _discover_multi_skills(self, owner_repo, branch):
        return []


SKILL = "---\nname: notion-knowledge-capture\ndescription: Capture notes\n---\nBody"


@pytest.mark.unit
def test_install_skill_finds_plugin_layouts_and_records_source(tmp_path):
    provider = _FakeSkills(
        {"acme/plugin/skills/notion/knowledge-capture/SKILL.md": SKILL}
    )
    result = install_skill(
        "acme/plugin", "notion-knowledge-capture", provider=provider, root=tmp_path
    )

    assert result["ok"], result
    assert result["name"] == "notion-knowledge-capture"
    assert result["path"] == str(
        tmp_path.resolve() / "acme" / "plugin" / "knowledge-capture" / "SKILL.md"
    )
    assert result["source"]["path"] == "skills/notion/knowledge-capture/SKILL.md"
    missing = install_skill("acme/plugin", "absent", provider=provider, root=tmp_path)
    assert missing["ok"] is False


@pytest.mark.unit
def test_skills_search_uses_skills_sh_without_a_token(monkeypatch):
    payload = {
        "skills": [
            {
                "id": "acme/skills/triage",
                "source": "acme/skills",
                "skillId": "triage",
                "installs": 12,
            },
            {
                "id": "open.feishu.cn/cal",
                "source": "open.feishu.cn",
                "skillId": "cal",
                "installs": 9,
            },
        ]
    }

    class _Response:
        def __init__(self, body):
            self.body = body

        def read(self):
            return json.dumps(self.body).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    seen = []

    def fake_urlopen(request, timeout):
        seen.append(request.full_url)
        return _Response(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = VercelSkillsProvider().search("triage", limit=5)

    assert seen and seen[0].startswith("https://skills.sh/api/search?")
    assert result["ok"] and result["source"] == "skills.sh"
    assert [(row["repo"], row["skill_name"]) for row in result["skills"]] == [
        ("acme/skills", "triage")
    ]


class _Provider:
    def __init__(self):
        self.agents = {
            "agent-1": SimpleNamespace(
                agent_id="agent-1",
                name="Assistant",
                persona=None,
                mcp_servers=[],
                skill_paths=[],
            )
        }

    def list_memagents(self):
        return list(self.agents.values())

    def retrieve_memagent(self, agent_id):
        return self.agents.get(agent_id)

    def store_memagent(self, agent):
        agent_id = getattr(agent, "agent_id", None) or "agent-new"
        if not getattr(agent, "agent_id", None):
            agent.agent_id = agent_id
        self.agents[agent_id] = agent
        return agent_id

    def list_personas(self, *args, **kwargs):
        return []

    def store_persona(self, persona, *args, **kwargs):
        return "persona-1"


@pytest.fixture
def ui(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    provider = _Provider()
    client = TestClient(create_app(), follow_redirects=False)
    with monkeypatch.context() as patcher:
        patcher.setitem(state._state, "provider", provider)
        patcher.setitem(state._state, "provider_type", "filesystem")
        patcher.setitem(state._state, "connection_info", {})
        yield client, provider, tmp_path


@pytest.mark.unit
def test_catalog_api_and_preset_attach(ui, registry):
    client, provider, _ = ui

    empty = client.get("/api/mcp/catalog").json()
    assert [row["key"] for row in empty["presets"]] == [
        "notion",
        "google-calendar",
        "gmail",
    ]
    assert empty["servers"] == [] and not registry

    found = client.get("/api/mcp/catalog?q=tasks").json()
    assert [row["name"] for row in found["servers"]] == ["linear", "tasks"]

    attached = client.post(
        "/api/mcp/agents/agent-1/presets/gmail",
        json={
            "client_id": "client.apps.googleusercontent.com",
            "client_secret": "s3cret",
        },
    )
    assert attached.status_code == 200, attached.text
    server = provider.agents["agent-1"].mcp_servers[0]
    assert server["name"] == "gmail"
    assert server["auth"]["redirect_uri"] == "http://testserver/api/mcp/oauth/callback"
    assert "s3cret" not in json.dumps(provider.agents["agent-1"].mcp_servers)

    unknown = client.post("/api/mcp/agents/agent-1/presets/slack", json={})
    assert unknown.status_code == 400


@pytest.mark.unit
def test_skill_attach_list_and_detach(ui, monkeypatch):
    client, provider, tmp_path = ui
    skill_file = tmp_path / "skills" / "acme" / "skills" / "triage" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text("---\nname: triage\ndescription: Sort mail\n---\nBody")

    def fake_install(repo, skill_name=None, **kwargs):
        assert (repo, skill_name) == ("acme/skills", "triage")
        return {"ok": True, "path": str(skill_file), "name": "triage"}

    monkeypatch.setattr("memorizz.vercel_skills.install.install_skill", fake_install)
    attached = client.post(
        "/api/agents/agent-1/skills",
        json={"repo": "acme/skills", "skill_name": "triage"},
    )
    assert attached.status_code == 200, attached.text
    assert provider.agents["agent-1"].skill_paths == [str(skill_file)]
    again = client.post(
        "/api/agents/agent-1/skills",
        json={"repo": "acme/skills", "skill_name": "triage"},
    )
    assert provider.agents["agent-1"].skill_paths == [str(skill_file)]
    assert again.json()["skills"][0]["description"] == "Sort mail"

    listed = client.get("/api/agents/agent-1/skills").json()["skills"]
    assert listed[0]["name"] == "triage" and listed[0]["exists"] is True

    detached = client.delete(f"/api/agents/agent-1/skills?path={skill_file}")
    assert detached.status_code == 200
    assert provider.agents["agent-1"].skill_paths == []
    assert skill_file.exists()
    assert (
        client.delete(f"/api/agents/agent-1/skills?path={skill_file}").status_code
        == 404
    )
    assert client.post("/api/agents/agent-1/skills", json={}).status_code == 400


@pytest.mark.unit
def test_productivity_template_creates_agent_with_three_connections(ui):
    client, provider, _ = ui

    form = client.get("/agents/new")
    assert "Start from a template: Personal productivity assistant" in form.text

    created = client.post(
        "/agents/templates/productivity",
        data={
            "google_client_id": "client.apps.googleusercontent.com",
            "google_client_secret": "s3cret",
        },
    )
    assert created.status_code == 303, created.text
    location = created.headers["location"]
    assert location.startswith("/mcp?agent_id=") and "created=productivity" in location

    agent_id = location.split("agent_id=")[1].split("&")[0]
    agent = provider.agents[agent_id]
    assert agent.instruction == PRODUCTIVITY_ASSISTANT.instruction
    assert {server["name"] for server in agent.mcp_servers} == {
        "gmail",
        "calendar",
        "notion",
    }
    assert "s3cret" not in json.dumps(agent.mcp_servers)

    page = client.get(location)
    assert "Created Productivity Assistant" in page.text
    assert client.post("/agents/templates/nope").status_code == 404


@pytest.mark.unit
def test_cli_gmail_preset_and_registry_search(tmp_path, registry):
    runner = CliRunner()
    env = {"MEMORIZZ_HOME": str(tmp_path)}
    missing_client = runner.invoke(
        cli_app, ["mcp", "add", "gmail", "--preset", "gmail"], env=env
    )
    assert missing_client.exit_code == 2
    added = runner.invoke(
        cli_app,
        [
            "mcp",
            "add",
            "gmail",
            "--preset",
            "gmail",
            "--client-id",
            "id.apps.googleusercontent.com",
        ],
        env=env,
    )
    assert added.exit_code == 0, added.output

    found = runner.invoke(cli_app, ["mcp", "search", "tasks", "--json"], env=env)
    assert found.exit_code == 0, found.output
    body = json.loads(found.output)
    assert [row["name"] for row in body["servers"]] == ["linear", "tasks"]


@pytest.mark.unit
def test_registry_search_retries_server_errors_and_caches(monkeypatch):
    import urllib.error

    calls = []

    def flaky(url, timeout):
        calls.append(url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(url, 500, "boom", {}, None)
        return REGISTRY_ROWS

    monkeypatch.setattr(catalog, "_get_json", flaky)
    monkeypatch.setattr(catalog, "_SEARCH_CACHE", {})
    first = catalog.search_registry("tasks")
    second = catalog.search_registry("tasks")
    assert first["ok"] and second is first and len(calls) == 2

    slow = []

    def down(url, timeout):
        slow.append(url)
        raise TimeoutError("timed out")

    monkeypatch.setattr(catalog, "_get_json", down)
    failed = catalog.search_registry("other", attempts=2)
    assert failed["ok"] is False and "Add connection" in failed["error"]
    assert len(slow) == 1  # a timeout is not retried

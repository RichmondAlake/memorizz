"""Every page shows the connection and, in read-only mode, a badge; controls
that change data are turned off up front while the server still refuses."""

import re
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from memorizz.memory_provider import FileSystemConfig, FileSystemProvider  # noqa: E402
from memorizz.ui import connection, state  # noqa: E402
from memorizz.ui.app import create_app  # noqa: E402

pytestmark = pytest.mark.unit

PAGES = [
    "/dashboard",
    "/settings",
    "/agents",
    "/agents/new",
    "/automations",
    "/traces",
    "/traces/usage",
    "/traces/usage/rates",
    "/memory/conversations",
    "/harnesses",
    "/mcp",
    "/evalground",
]
STATIC = Path(connection.__file__).parent / "static"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MEMORIZZ_UI_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    for key in ("MEMORIZZ_UI_AUTH_TOKEN", "MEMORIZZ_UI_AUTH_ACCOUNTS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def provider(home):
    return FileSystemProvider(
        FileSystemConfig(root_path=home / "store", lazy_vector_indexes=True)
    )


def _connected(provider, info=None, provider_type="filesystem", **extra):
    return patch.dict(
        state._state,
        {
            "provider": provider,
            "provider_type": provider_type,
            "connection_info": info if info is not None else {"path": "/srv/store"},
            "provider_secrets": {},
            "embedding_check": None,
            **extra,
        },
    )


@pytest.mark.parametrize("path", PAGES)
def test_read_only_badge_and_connection_on_every_page(
    home, provider, monkeypatch, path
):
    monkeypatch.setenv("MEMORIZZ_UI_READ_ONLY", "true")
    client = TestClient(create_app(), follow_redirects=False)
    with _connected(provider, {"path": str(home / "store")}, read_only=True):
        response = client.get(path)
    assert response.status_code == 200, path
    page = response.text
    assert 'data-read-only="true"' in page
    assert 'class="ui-ro-badge"' in page and "Read-only" in page
    assert 'title="The UI is read-only. Changes are turned off."' in page
    # The connection chip links to the connect page.
    assert 'class="ui-conn" href="/connect"' in page
    assert '<span class="ui-conn-provider">FileSystem</span>' in page


def test_no_badge_when_the_ui_can_change_data(home, provider, monkeypatch):
    monkeypatch.delenv("MEMORIZZ_UI_READ_ONLY", raising=False)
    client = TestClient(create_app(), follow_redirects=False)
    with _connected(provider):
        page = client.get("/dashboard").text
    assert "data-read-only" not in page
    assert 'class="ui-ro-badge"' not in page
    assert '<span class="ui-conn-provider">FileSystem</span>' in page


def test_read_only_still_refused_by_the_server(home, provider, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_UI_READ_ONLY", "true")
    client = TestClient(create_app(), follow_redirects=False)
    with _connected(provider, read_only=True):
        for path, data in [
            ("/settings", {"OPENAI_API_KEY": "sk-test"}),
            ("/traces/usage/rates", {"provider": "openai", "model": "x"}),
            ("/traces/usage/budget", {"monthly_usd": "10"}),
            ("/agents", {"name": "x"}),
        ]:
            assert client.post(path, data=data).status_code == 403, path


def test_the_page_script_allows_exactly_what_the_server_allows(
    home, provider, monkeypatch
):
    monkeypatch.setenv("MEMORIZZ_UI_READ_ONLY", "true")
    client = TestClient(create_app(), follow_redirects=False)
    with _connected(provider, read_only=True):
        page = client.get("/dashboard").text
    allow = re.search(r'data-read-only-allow="([^"]*)"', page).group(1).split()
    assert set(allow) == set(connection.READ_ONLY_ALLOWED_POSTS)
    # Forms posting there stay usable; the server accepts them read-only.
    with _connected(provider, read_only=True):
        assert (
            client.post("/connect", data={"provider_type": "nope"}).status_code == 200
        )


def test_write_controls_are_marked_and_the_guard_covers_post_forms(home, provider):
    client = TestClient(create_app(), follow_redirects=False)
    with _connected(provider):
        dashboard = client.get("/dashboard").text
        agents = client.get("/agents").text
    assert 'href="/agents/new" data-mutates>Create agent' in dashboard
    assert 'href="/agents/new" class="btn btn-primary" data-mutates>' in agents
    script = (STATIC / "js" / "app.js").read_text()
    guard = script[script.index("function initReadOnlyGuard") :]
    for needle in (
        'body.dataset.readOnly === "true"',
        "[data-mutates], .btn-danger, .hx-btn-delete, .is-danger",
        '.toLowerCase() !== "post"',
        'document.addEventListener("submit"',
        "MutationObserver",
        "window.MemorizzReadOnly",
    ):
        assert needle in guard, needle
    templates = Path(connection.__file__).parent / "templates"
    playground = (templates / "playground.html").read_text()
    for control in ('id="pg-send-btn" data-mutates', 'id="pg-query" data-mutates'):
        assert control in playground


@pytest.mark.parametrize(
    ("provider_type", "info", "short", "title"),
    [
        (
            "oracle",
            {"user": "EVAL", "dsn": "db.example.com:1521/FREEPDB1", "schema": "EVAL"},
            "EVAL@db.example.com:1521/FREEPDB1",
            "Oracle EVAL@db.example.com:1521/FREEPDB1, schema EVAL",
        ),
        (
            "oracle",
            {
                "user": "READER",
                "dsn": "tcps://db:2484/svc?wallet_location=/w",
                "schema": "EVAL",
            },
            "READER@db:2484/svc · EVAL",
            "Oracle READER@db:2484/svc, schema EVAL",
        ),
        (
            "oracle",
            {
                "user": "U",
                "dsn": "(DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST=10.0.0.5)(PORT=1522))"
                "(CONNECT_DATA=(SERVICE_NAME=PDB1)))",
            },
            "U@10.0.0.5:1522/PDB1",
            "Oracle U@10.0.0.5:1522/PDB1",
        ),
        (
            "mongodb",
            {"uri": "mongodb://***@h1:27017,h2:27017/admin", "db_name": "memorizz"},
            "h1:27017 +1/memorizz",
            "MongoDB h1:27017 +1/memorizz",
        ),
        (
            "notion",
            {"data_source_id": "0123456789abcdef", "semantic_backend": "filesystem"},
            "data source 01234567… · filesystem vectors",
            "Notion data source 0123456789abcdef, filesystem vectors",
        ),
    ],
)
def test_connection_target_is_short_and_has_no_secret(
    provider_type, info, short, title
):
    assert connection.connection_target(provider_type, info) == (short, title)


def test_filesystem_target_is_shortened_under_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    deep = tmp_path / "work" / "clients" / "acme" / "memorizz" / "memory"
    short, title = connection.connection_target("filesystem", {"path": str(deep)})
    assert short == "~/…/memorizz/memory"
    assert title == "FileSystem ~/work/clients/acme/memorizz/memory"


@pytest.mark.parametrize(
    ("uri", "masked"),
    [
        (
            "mongodb+srv://user:pa55@cluster0.example.net/memorizz?retryWrites=true&w=1",
            "mongodb+srv://***@cluster0.example.net/memorizz",
        ),
        ("mongodb://user:p/a%40ss@db:27017/x", "mongodb://***@db:27017/x"),
        ("mongodb://localhost:27017", "mongodb://localhost:27017"),
        ("mongodb://db/x?tlsCertificateKeyFilePassword=hunter2", "mongodb://db/x"),
    ],
)
def test_mask_uri_drops_credentials_and_options(uri, masked):
    assert connection.mask_uri(uri) == masked


def test_the_header_never_renders_connection_secrets(home, provider):
    client = TestClient(create_app(), follow_redirects=False)
    secret = "oracle-super-secret"
    info = {
        "user": "EVAL",
        "dsn": "localhost:1529/FREEPDB1",
        "schema": "EVAL",
        "embedding": "Disabled",
    }
    with _connected(
        provider,
        info,
        provider_type="oracle",
        provider_secrets={"oracle_password": secret},
    ):
        page = client.get("/dashboard").text
    assert secret not in page
    assert (
        '<span class="ui-conn-target mono">EVAL@localhost:1529/FREEPDB1</span>' in page
    )
    assert "Oracle EVAL@localhost:1529/FREEPDB1, schema EVAL" in page


def test_restricted_trace_accounts_see_only_the_provider(provider):
    from memorizz.ui.trace_access import TracePrincipal, current_principal

    info = {"user": "EVAL", "dsn": "db:1521/FREEPDB1", "schema": "EVAL"}
    check = {"status": "mismatch", "message": "differs"}
    token = current_principal.set(TracePrincipal(principal_id="viewer", role="viewer"))
    try:
        with _connected(provider, info, provider_type="oracle", embedding_check=check):
            header = connection.connection_header()
    finally:
        current_principal.reset(token)
    assert (header["label"], header["target"], header["title"]) == (
        "Oracle",
        "",
        "Oracle",
    )
    assert header["href"] is None and header["embedding_warning"] is None
    with _connected(provider, info, provider_type="oracle", embedding_check=check):
        admin = connection.connection_header()
    assert admin["target"] == "EVAL@db:1521/FREEPDB1" and admin["href"] == "/connect"
    assert admin["embedding_warning"] == "differs"


NOTION_INFO = {
    "data_source_id": "4230b4b3-4fc8-81f2-9600-000b7856b21c",
    "semantic_backend": "filesystem",
    "notion_url": "https://app.notion.com/p/4230b4b34fc881ef8152c6051e81f7dc",
}


def test_the_notion_badge_opens_the_memory_database_in_notion(provider):
    from memorizz.ui.trace_access import TracePrincipal, current_principal

    client = TestClient(create_app(), follow_redirects=False)
    with _connected(provider, NOTION_INFO, provider_type="notion"):
        header = connection.connection_header()
        page = client.get("/settings").text
    assert header["opens"] == NOTION_INFO["notion_url"]
    chip = re.search(r'<a class="ui-conn"[^>]*>', page).group(0)
    assert f'href="{NOTION_INFO["notion_url"]}"' in chip
    assert 'target="_blank"' in chip and 'rel="noopener noreferrer"' in chip
    assert "Open the memory database in Notion" in chip
    # It leaves the app, so it is not the store switch; the sidebar still is.
    assert "data-store-switch" not in chip
    assert 'class="status-text" type="button" data-store-switch' in page

    # Other stores have nowhere to open, and neither does an address that is
    # not a web link.
    with _connected(provider, {"path": "/srv/store"}):
        assert connection.connection_header()["opens"] is None
    for bad in (None, "javascript:alert(1)", "http://app.notion.com/p/x"):
        with _connected(provider, {**NOTION_INFO, "notion_url": bad}, "notion"):
            assert connection.connection_header()["opens"] is None

    # An account that only inspects traces is not told where the store is.
    token = current_principal.set(TracePrincipal(principal_id="viewer", role="viewer"))
    try:
        with _connected(provider, NOTION_INFO, provider_type="notion"):
            assert connection.connection_header()["opens"] is None
    finally:
        current_principal.reset(token)


def test_the_notion_address_is_the_one_notion_reports_or_built_from_the_id():
    from memorizz.memory_provider.notion.provider import _notion_page_url

    database = "4230b4b3-4fc8-81ef-8152-c6051e81f7dc"
    reported = "https://app.notion.com/p/4230b4b34fc881ef8152c6051e81f7dc"
    built = "https://www.notion.so/4230b4b34fc881ef8152c6051e81f7dc"
    assert _notion_page_url(reported, database) == reported
    assert _notion_page_url("https://www.notion.so/x", database) == (
        "https://www.notion.so/x"
    )
    for odd in (None, "", "https://notion.so.example.com/p/x", "http://notion.so/x"):
        assert _notion_page_url(odd, database) == built
    assert _notion_page_url(None, None) is None
    assert _notion_page_url("https://example.com", "not-an-id") is None

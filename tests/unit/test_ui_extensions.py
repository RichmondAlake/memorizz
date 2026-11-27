"""Optional UI pages: found when present, invisible when not, never fatal."""

from __future__ import annotations

import os
import sys
import textwrap
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from memorizz.ui import extensions  # noqa: E402
from memorizz.ui.state import _state, templates  # noqa: E402

pytestmark = pytest.mark.unit

EXTENSION = """
    from pathlib import Path
    from fastapi import APIRouter
    from memorizz.ui.extensions import UIExtension

    router = APIRouter()
    stopped = []

    @router.get("/lab")
    async def lab():
        return {"ok": True}

    @router.post("/lab/ping")
    async def ping():
        return {"ok": True}

    def ui_extension():
        here = Path(__file__).parent
        return UIExtension(
            key="lab", label="Lab", href="/lab", router=router,
            icon='<circle cx="12" cy="12" r="5"/>', badge="Beta",
            static_dir=here / "lab_static", quiet_posts=r"^/lab/ping$",
            on_shutdown=lambda: stopped.append(True),
        )
"""


@pytest.fixture
def app_with(tmp_path, monkeypatch):
    """Build the UI with the named extension modules (and no local ones)."""
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(extensions, "LOCAL_EXTENSIONS", ())
    monkeypatch.syspath_prepend(str(tmp_path))
    loader = templates.env.loader
    globals_before = dict(templates.env.globals)

    def build(*modules):
        from memorizz.ui.app import create_app

        os.environ["MEMORIZZ_UI_EXTENSIONS"] = ",".join(modules)
        return create_app()

    with patch.dict(
        _state,
        {"provider": object(), "provider_type": "filesystem", "connection_info": {}},
    ):
        yield build
    templates.env.loader = loader
    templates.env.globals.clear()
    templates.env.globals.update(globals_before)
    for name in ("lab_ext", "broken_ext"):
        sys.modules.pop(name, None)


def test_without_extensions_the_ui_is_unchanged(app_with):
    app = app_with()
    assert app.state.ui_extensions == []
    page = TestClient(app).get("/settings").text
    assert "nav-badge" not in page
    assert TestClient(app).get("/simuground").status_code == 404


def test_an_extension_adds_its_route_sidebar_item_and_static_files(app_with, tmp_path):
    (tmp_path / "lab_static").mkdir()
    (tmp_path / "lab_static" / "lab.css").write_text("body{}", encoding="utf-8")
    (tmp_path / "lab_ext.py").write_text(textwrap.dedent(EXTENSION), encoding="utf-8")

    app = app_with("lab_ext")
    client = TestClient(app)

    assert [extension.key for extension in app.state.ui_extensions] == ["lab"]
    assert client.get("/lab").json() == {"ok": True}
    assert client.get("/ext/lab/lab.css").text == "body{}"
    page = client.get("/settings").text
    assert 'href="/lab"' in page and ">Lab</span>" in page
    assert '<span class="nav-badge">Beta</span>' in page


def test_an_extensions_quiet_posts_stay_out_of_the_activity_log(app_with, tmp_path):
    from memorizz.ui import activity

    (tmp_path / "lab_static").mkdir()
    (tmp_path / "lab_ext.py").write_text(textwrap.dedent(EXTENSION), encoding="utf-8")
    app_with("lab_ext")
    try:
        assert activity.describe("POST", "/lab/ping") is None
        assert activity.describe("POST", "/lab/other") is not None
    finally:
        activity._EXTRA_SKIP.clear()


def test_shutdown_runs_each_extensions_hook(app_with, tmp_path):
    (tmp_path / "lab_static").mkdir()
    (tmp_path / "lab_ext.py").write_text(textwrap.dedent(EXTENSION), encoding="utf-8")
    app = app_with("lab_ext")
    extensions.shutdown(app)
    assert sys.modules["lab_ext"].stopped == [True]


def test_a_broken_extension_is_skipped(app_with, tmp_path):
    (tmp_path / "broken_ext.py").write_text(
        "def ui_extension():\n    raise RuntimeError('boom')\n", encoding="utf-8"
    )
    app = app_with("broken_ext", "missing_module_xyz")
    assert app.state.ui_extensions == []
    assert TestClient(app, follow_redirects=False).get("/settings").status_code == 200


def test_extensions_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("MEMORIZZ_UI_EXTENSIONS", "off")
    assert extensions.discover() == []


def test_extension_self_authenticated_paths_bypass_ui_login_only_on_declared_path(
    app_with, tmp_path, monkeypatch
):
    monkeypatch.setenv("MEMORIZZ_UI_AUTH_TOKEN", "operator-secret-token-123456")
    self_auth_ext = """
        from fastapi import APIRouter, Request
        from fastapi.responses import JSONResponse
        from memorizz.ui.extensions import UIExtension

        router = APIRouter()

        @router.post("/lab/publish")
        async def lab_publish(request: Request):
            if request.headers.get("authorization") != "Bearer scoped-key":
                return JSONResponse({"ok": False}, status_code=401)
            return {"ok": True}

        @router.get("/lab/private")
        async def lab_private():
            return {"secret": True}

        def ui_extension():
            return UIExtension(
                key="lab",
                label="Lab",
                href="/lab",
                router=router,
                self_authenticated_paths=("/lab/publish",),
            )
    """
    (tmp_path / "lab_ext.py").write_text(
        textwrap.dedent(self_auth_ext), encoding="utf-8"
    )
    app = app_with("lab_ext")
    client = TestClient(app, follow_redirects=False)

    # Declared path reaches the handler even cross-origin and without UI auth.
    ok = client.post(
        "/lab/publish",
        headers={
            "Authorization": "Bearer scoped-key",
            "Origin": "https://other.example",
        },
    )
    assert ok.status_code == 200 and ok.json() == {"ok": True}
    # Handler itself rejects bad key.
    bad = client.post("/lab/publish", headers={"Authorization": "Bearer wrong"})
    assert bad.status_code == 401
    # Undeclared path stays behind UI auth even when carrying the scoped key.
    blocked = client.get("/lab/private", headers={"Authorization": "Bearer scoped-key"})
    assert blocked.status_code == 401

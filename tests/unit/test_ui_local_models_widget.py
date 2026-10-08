"""The local-model availability widget lives once, in static/js/local-models.js.

Settings, the playground and the agent form mount it with their own copy,
element ids and data-* attribute names instead of each carrying a copy of the
``/api/ollama/installed`` / ``/api/huggingface/installed`` fetch, the
``:latest`` matching rule and the banner markup.
"""

import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

UI = Path(__file__).parents[2] / "src" / "memorizz" / "ui"
TEMPLATES = UI / "templates"
MODULE = UI / "static" / "js" / "local-models.js"
SCRIPT_TAG = '<script src="/static/js/local-models.js?v=1"></script>'
MOUNT = "window.MemorizzLocalModels.mount({"

# Each page's element id, refresh attribute and the copy that is its own.
PAGES = {
    "settings.html": [
        "statusId: 'local-model-status'",
        "providers: ['ollama', 'huggingface']",
        "refreshAttr: 'data-local-refresh'",
        "refreshText: 'refresh this check'",
        "ollamaMissing: 'Run {pull} in a terminal, or {browse}. Then {refresh}.'",
        "huggingfaceUnavailable: 'Install the SDK with <code>pip install "
        "memorizz[huggingface]</code>, then {refresh}.'",
        "huggingfaceMissing: 'It will be downloaded the first time the agent loads "
        "(a few hundred MB to several GB). You can pre-download with {download}, "
        "or {card}. Then {refresh}.'",
        "attrPrefix: 'data-'",
        "selectTitle: 'Use this model as the Default Model'",
        "onSelect: selectModelFromRow",
    ],
    "playground.html": [
        "statusId: 'cfg-local-model-status'",
        "refreshAttr: 'data-pg-refresh'",
        "refresh: playgroundCheckLocalModel",
        "warningPrefix: 'Pre-flight: '",
        "refreshText: 'refresh'",
        "ollamaDownRefreshText: 'refresh this check'",
        "ollamaMissing: 'Sending a message will fail with a 404. Run {pull}, click "
        "the download icon below, or {browse}. Then {refresh}.'",
        "huggingfaceUnavailable: 'Install it with <code>pip install "
        "memorizz[huggingface]</code>, then {refresh}.'",
        "huggingfaceMissing: 'Sending a message will trigger a download (a few "
        "hundred MB to several GB) and may stall. Click the download icon below, "
        "or pre-download with {download}, or {card}. Then {refresh}.'",
        "attrPrefix: 'data-pg-'",
        "selectTitle: 'Use this model in the playground'",
        "onSelect: pgSelectModel",
    ],
    "agent_form.html": [
        "statusId: 'llm_model_status'",
        "refreshAttr: 'data-preflight-refresh'",
        "hostFallback: 'localhost:11434'",
        "warningPrefix: 'Pre-flight: '",
        "refreshText: 'refresh'",
        "ollamaMissing: 'Saving this agent is fine, but the first chat will fail "
        'with a 404. Run {pull}, pull from <a href="/settings#local-models">'
        "Settings → Local models</a>, or {browse}. Then {refresh}.'",
        "huggingfaceUnavailable: 'Install it with <code>pip install "
        "memorizz[huggingface]</code>, then {refresh}.'",
        "huggingfaceMissing: 'First chat will trigger a multi-GB download and may "
        "stall. Pre-download with {download}, pull from "
        '<a href="/settings#local-models">Settings → Local models</a>, or {card}. '
        "Then {refresh}.'",
    ],
}

# The three inline implementations, by the names they used to go by, plus the
# fetch, matching and link details that now live only in the module.
OLD_INLINE = [
    "renderOllamaStatus",
    "renderHuggingFaceStatus",
    "manageListHtml",
    "wireRefreshLink",
    "checkLocalModelAvailability",
    "scheduleLocalCheck",
    "pgRenderOllama",
    "pgRenderHuggingFace",
    "pgManageListHtml",
    "pgModelRow",
    "pgWireRefresh",
    "pgScheduleLocalCheck",
    "pgHandleActionClick",
    "function renderOllama(",
    "function renderHF(",
    "/api/ollama/installed",
    "/api/huggingface/installed",
    "ollama.com/library/",
    "m + ':latest' === model",
    "Could not reach the UI server",
    "is not installed locally",
    "is not cached locally",
]


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(PAGES))
def test_page_loads_the_shared_widget_and_mounts_it_with_its_own_copy(name):
    page = (TEMPLATES / name).read_text()
    assert page.count(SCRIPT_TAG) == 1, name
    assert page.count(MOUNT) == 1, name
    # The page's inline scripts call mount() while the document is parsing,
    # so the module is a plain (not deferred) head script, after base.html's
    # escapeHtml() and before the mount call.
    assert page.index(SCRIPT_TAG) < page.index(MOUNT), name
    for fragment in PAGES[name]:
        assert fragment in page, f"{name} lost its copy: {fragment}"


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(PAGES))
def test_old_inline_implementation_is_gone(name):
    page = (TEMPLATES / name).read_text()
    for marker in OLD_INLINE:
        assert marker not in page, f"{name} still carries {marker!r}"
    module = MODULE.read_text()
    for marker in OLD_INLINE[-7:]:
        assert marker in module, marker
    assert "https://huggingface.co/" in module


@pytest.mark.unit
def test_module_loads_after_the_shared_helpers_it_depends_on():
    base = (TEMPLATES / "base.html").read_text()
    for helper in (
        "function escapeHtml(",
        "function trashIconSvg(",
        "function downloadIconSvg(",
    ):
        assert base.index(helper) < base.index("{% block head %}"), helper
    module = MODULE.read_text()
    assert "window.MemorizzLocalModels = { mount: mount }" in module


@pytest.fixture(scope="module")
def client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from memorizz.ui.app import create_app

    return TestClient(create_app(), follow_redirects=False)


@pytest.fixture()
def connected(tmp_path):
    from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
    from memorizz.ui import state

    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "ui", lazy_vector_indexes=True)
    )
    with patch.dict(
        state._state,
        {
            "provider": provider,
            "provider_type": "filesystem",
            "connection_info": {"root_path": str(provider.root_path)},
        },
    ):
        yield provider


@pytest.mark.unit
def test_rendered_pages_carry_the_widget(client, connected):
    client.post(
        "/agents/new",
        data={
            "agent_name": "Widget Agent",
            "instruction": "You check local models.",
            "application_mode": "assistant",
            "max_steps": "10",
            "tool_access": "private",
            "llm_provider": "ollama",
            "llm_model": "llama3.1",
        },
    )
    agent_id = connected.list_memagents()[0].agent_id
    for path in (
        "/settings",
        "/agents/new",
        f"/agents/{agent_id}/edit",
        f"/agents/{agent_id}/playground",
    ):
        resp = client.get(path)
        assert resp.status_code == 200, path
        html = resp.text
        assert html.count(SCRIPT_TAG) == 1, path
        assert (
            html.index("function escapeHtml(")
            < html.index(SCRIPT_TAG)
            < html.index(MOUNT)
        ), path


# ---------------------------------------------------------------------------
# Behaviour, run in Node when it is available: one mount with test copy and a
# fake DOM/fetch, exercising the fetch routing, the matching rules, escaping,
# the refresh link, the manage panel and the stale-response guard.
# ---------------------------------------------------------------------------

NODE_HARNESS = r"""
'use strict';
const fs = require('fs');
const status = { style: {}, innerHTML: '', wired: 0, onRefresh: null };
status.querySelector = function (sel) {
    if (sel !== '[data-t-refresh]' || status.innerHTML.indexOf(' data-t-refresh>') === -1) return null;
    return { addEventListener: function (type, fn) { status.wired += 1; status.onRefresh = fn; } };
};
const clicks = [], actions = [], selected = [], refreshes = [];
globalThis.document = {
    getElementById: function (id) { return id === 'status' ? status : null; },
    addEventListener: function (type, fn) { clicks.push(fn); },
};
globalThis.window = globalThis;
globalThis.escapeHtml = function (v) {
    return String(v ?? '').replace(/[&<>"']/g, function (c) {
        return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
};
globalThis.trashIconSvg = function () { return '<TRASH/>'; };
globalThis.downloadIconSvg = function () { return '<DOWNLOAD/>'; };
globalThis.handleLocalModelAction = function (btn, action, provider, model, refresh) {
    actions.push([action, provider, model]); refresh();
};
new Function(fs.readFileSync(process.argv[2], 'utf8'))();

let state = { provider: 'ollama', model: '' }, info = {}, fetched = [];
globalThis.fetch = async function (url, opts) {
    fetched.push(url + '|' + (opts && opts.cache));
    if (info instanceof Error) throw info;
    return { json: async function () { return info; } };
};
const widget = window.MemorizzLocalModels.mount({
    statusId: 'status',
    provider: function () { return state.provider; },
    model: function () { return state.model; },
    refreshAttr: 'data-t-refresh',
    refresh: function () { refreshes.push(state.model); return widget.check(); },
    hostFallback: 'fallback-host',
    copy: {
        warningPrefix: 'PFX ',
        refreshText: 'again',
        ollamaDownRefreshText: 'again later',
        ollamaMissing: 'Missing: {pull} / {browse} / {refresh}.',
        huggingfaceUnavailable: 'No SDK, {refresh}.',
        huggingfaceMissing: 'Not cached: {download} / {card} / {refresh}.',
    },
    manage: {
        catalog: {
            ollama: [{ value: 'llama3.1', label: 'Llama 3.1' }, { value: 'qwen:7b', label: 'qwen:7b' }],
            huggingface: [{ value: 'org/model', label: 'Org Model' }],
        },
        attrPrefix: 'data-t-',
        selectTitle: 'Use it',
        onSelect: function (v) { selected.push(v); },
    },
});
const out = {};
async function run(name, provider, model, reply) {
    state = { provider: provider, model: model }; info = reply; fetched = [];
    status.innerHTML = 'OLD'; status.style.display = 'OLD'; status.wired = 0; status.onRefresh = null;
    const pending = widget.check();
    const checking = status.innerHTML;
    const painted = await pending;
    out[name] = { checking, html: status.innerHTML, display: status.style.display, wired: status.wired, fetched, painted };
}
(async function () {
    await run('hosted', 'openai', 'gpt-4', {});
    await run('nomodel', 'ollama', '', {});
    await run('down', 'ollama', 'llama3.1', { reachable: false, error: 'a<b' });
    await run('error', 'ollama', 'llama3.1', new Error('net down'));
    await run('latest', 'ollama', 'llama3.1', { reachable: true, host: 'h', models: ['llama3.1:latest', 'extra:latest'] });
    await run('bare', 'ollama', 'qwen:latest', { reachable: true, host: 'h', models: ['qwen'] });
    await run('missing', 'ollama', 'qwen:7b', { reachable: true, host: 'h', models: ['llama3.1'] });
    await run('hf-cached', 'huggingface', 'org/model', { available: true, cache_dir: '/c', models: ['org/model'] });
    await run('hf-case', 'huggingface', 'Org/model', { available: true, models: ['org/model'] });
    await run('mlx-down', 'mlx', 'org/model', { available: false });
    await run('mlx-missing', 'mlx', 'o<r/g', { available: true, models: [] });

    // The refresh link re-runs the page-level refresh, not just check().
    await run('link', 'ollama', 'qwen:7b', { reachable: true, host: 'h', models: [] });
    status.onRefresh({ preventDefault: function () {} });
    out.refreshes = refreshes.slice();

    // Delegated clicks: one action button, one select button.
    const btn = { getAttribute: function (a) {
        return { 'data-t-action': 'pull', 'data-t-provider': 'ollama', 'data-t-model': 'qwen:7b' }[a];
    } };
    const sel = { getAttribute: function () { return 'llama3.1'; } };
    const target = { closest: function (s) {
        if (s === '.model-action-btn[data-t-action]') return btn;
        if (s === '[data-t-select-model]') return sel;
        return null;
    } };
    clicks.forEach(function (fn) { fn({ preventDefault: function () {}, target: target }); });
    await new Promise(function (r) { setTimeout(r, 0); });
    out.clicks = { handlers: clicks.length, actions: actions, selected: selected, refreshes: refreshes.slice() };

    // A superseded check never paints.
    let release;
    const slow = new Promise(function (r) { release = r; });
    globalThis.fetch = function (url) {
        if (url === '/api/ollama/installed') return slow;
        return Promise.resolve({ json: async function () { return { available: true, cache_dir: '/c', models: ['org/model'] }; } });
    };
    state = { provider: 'ollama', model: 'm1' };
    const p1 = widget.check();
    state = { provider: 'huggingface', model: 'org/model' };
    const p2 = widget.check();
    const second = await p2;
    const afterSecond = status.innerHTML;
    release({ json: async function () { return { reachable: true, host: 'late', models: ['m1'] }; } });
    const first = await p1;
    out.stale = { first: first, second: second, unchanged: afterSecond === status.innerHTML };
    process.stdout.write(JSON.stringify(out));
})().catch(function (e) { console.error(e); process.exit(1); });
"""


@pytest.fixture(scope="module")
def widget_run(tmp_path_factory):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    harness = tmp_path_factory.mktemp("local-models") / "harness.js"
    harness.write_text(NODE_HARNESS)
    proc = subprocess.run(
        [node, str(harness), str(MODULE)], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


@pytest.mark.unit
def test_module_parses():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    proc = subprocess.run(
        [node, "--check", str(MODULE)], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.unit
def test_hosted_providers_and_empty_models_hide_the_panel_without_fetching(widget_run):
    for name in ("hosted", "nomodel"):
        case = widget_run[name]
        assert case["display"] == "none" and case["html"] == "", name
        assert case["fetched"] == [] and case["painted"] is True, name


@pytest.mark.unit
def test_each_provider_hits_its_endpoint_and_shows_a_checking_hint(widget_run):
    assert widget_run["down"]["fetched"] == ["/api/ollama/installed|no-store"]
    assert widget_run["hf-cached"]["fetched"] == ["/api/huggingface/installed|no-store"]
    assert widget_run["mlx-down"]["fetched"] == ["/api/huggingface/installed|no-store"]
    assert (
        widget_run["down"]["checking"]
        == '<p class="form-hint">Checking local Ollama…</p>'
    )
    assert (
        widget_run["hf-cached"]["checking"]
        == '<p class="form-hint">Checking HuggingFace cache…</p>'
    )
    assert (
        widget_run["mlx-down"]["checking"]
        == '<p class="form-hint">Checking MLX cache (HuggingFace)…</p>'
    )
    assert all(
        widget_run[n]["display"] == "block" for n in ("down", "hf-cached", "mlx-down")
    )


@pytest.mark.unit
def test_unreachable_server_and_daemon_banners(widget_run):
    error = widget_run["error"]
    assert error["html"] == (
        '<div class="warning-message"><strong>Could not reach the UI server</strong>: '
        "Error: net down</div>"
    )
    assert error["painted"] is False and error["wired"] == 0

    down = widget_run["down"]
    assert down["html"].startswith(
        '<div class="warning-message"><strong>Ollama is not running</strong> at '
        "<code>fallback-host</code> &mdash; a&lt;b.<br>Install it from "
        '<a href="https://ollama.com/download" target="_blank" rel="noopener">'
        "ollama.com/download</a> and start it, then "
    )
    assert down["html"].endswith('<a href="#" data-t-refresh>again later</a>.</div>')
    assert down["wired"] == 1 and down["painted"] is True

    unavailable = widget_run["mlx-down"]["html"]
    assert unavailable == (
        '<div class="warning-message"><strong>HuggingFace cache unavailable</strong>.'
        '<br>No SDK, <a href="#" data-t-refresh>again</a>.</div>'
    )


@pytest.mark.unit
def test_ollama_matches_bare_and_latest_tags_both_ways(widget_run):
    for name in ("latest", "bare"):
        assert widget_run[name]["html"].startswith(
            '<div class="success-message"><strong><code>'
        ), name
        assert (
            "is available offline</strong> on Ollama at <code>h</code>."
            in widget_run[name]["html"]
        )
    missing = widget_run["missing"]["html"]
    assert missing.startswith(
        '<div class="warning-message"><strong>PFX <code>qwen:7b</code> is not installed '
        "locally.</strong><br>Missing: <code>ollama pull qwen:7b</code> / "
        '<a href="https://ollama.com/library/qwen" target="_blank" rel="noopener">'
        "browse <code>qwen</code> on ollama.com</a> / "
        '<a href="#" data-t-refresh>again</a>.</div>'
    )
    assert widget_run["missing"]["wired"] == 1


@pytest.mark.unit
def test_huggingface_matches_exactly_and_escapes_the_repo_id(widget_run):
    assert widget_run["hf-cached"]["html"].startswith(
        '<div class="success-message"><strong><code>org/model</code> is cached locally'
        "</strong> at <code>/c</code>.</div>"
    )
    assert (
        "<code>Org/model</code> is not cached locally." in widget_run["hf-case"]["html"]
    )
    missing = widget_run["mlx-missing"]["html"]
    assert missing.startswith(
        '<div class="warning-message"><strong>PFX <code>o&lt;r/g</code> is not cached '
        "locally.</strong><br>Not cached: <code>huggingface-cli download o&lt;r/g</code> / "
        '<a href="https://huggingface.co/o%3Cr/g" target="_blank" rel="noopener">'
        "view the model card</a> / "
        '<a href="#" data-t-refresh>again</a>.</div>'
    )


@pytest.mark.unit
def test_manage_panel_uses_the_page_attribute_prefix_and_catalog(widget_run):
    html = widget_run["latest"]["html"]
    assert (
        '<details class="model-manager" open><summary>Manage Ollama models</summary>'
        in html
    )
    assert '<div class="model-list-section">Available offline (2)</div>' in html
    assert '<div class="model-list-section">Available to download (1)</div>' in html
    assert (
        '<button type="button" class="model-row-select" data-t-select-model="llama3.1" '
        'aria-pressed="true" title="Use it">'
        '<span class="model-row-check" aria-hidden="true">✓</span>'
        "<code>llama3.1</code>"
        '<span class="model-row-label"> &mdash; Llama 3.1</span></button>'
    ) in html
    assert (
        '<button type="button" class="model-action-btn" data-t-action="delete" '
        'data-t-provider="ollama" data-t-model="llama3.1" title="Remove from local cache" '
        'aria-label="Remove llama3.1"><TRASH/></button>'
    ) in html
    assert (
        '<div class="model-row" data-row-action="pull"><div class="model-row-text">'
        "<code>qwen:7b</code></div>"
        '<button type="button" class="model-action-btn" data-t-action="pull" '
        'data-t-provider="ollama" data-t-model="qwen:7b" title="Download to local cache" '
        'aria-label="Download qwen:7b"><DOWNLOAD/></button></div>'
    ) in html
    # extra:latest is pulled but not in the catalog: listed under its own name.
    assert 'data-t-select-model="extra:latest"' in html
    assert (
        "<summary>Manage HuggingFace models</summary>"
        in widget_run["hf-cached"]["html"]
    )
    assert "Available to download (1)" in widget_run["mlx-missing"]["html"]


@pytest.mark.unit
def test_refresh_link_and_delegated_clicks_run_the_page_refresh(widget_run):
    assert widget_run["refreshes"] == ["qwen:7b"]
    clicks = widget_run["clicks"]
    assert clicks["handlers"] == 2
    assert clicks["actions"] == [["pull", "ollama", "qwen:7b"]]
    assert clicks["selected"] == ["llama3.1"]
    assert clicks["refreshes"] == ["qwen:7b", "qwen:7b"]


@pytest.mark.unit
def test_a_superseded_check_never_paints(widget_run):
    assert widget_run["stale"] == {"first": False, "second": True, "unchanged": True}

/* Memorizz UI JavaScript */

function initializeMemorizzUI() {
    const storageKeys = {
        sidebarCollapsed: "memorizz_ui_sidebar_collapsed",
        lightMode: "memorizz_ui_light_mode",
        benchmarkRunState: "memorizz_ui_benchmark_run_state",
        benchmarkDockCollapsed: "memorizz_ui_benchmark_dock_collapsed",
    };

    const iconMarkup = {
        sidebarCollapse:
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="m15 18-6-6 6-6"/></svg>',
        sidebarExpand:
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="m9 18 6-6-6-6"/></svg>',
        sun:
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.9 4.9 1.4 1.4"/><path d="m17.7 17.7 1.4 1.4"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m4.9 19.1 1.4-1.4"/><path d="m17.7 6.3 1.4-1.4"/></svg>',
        moon:
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>',
    };

    const body = document.body;
    const sidebarToggleBtn = document.getElementById("sidebar-toggle-btn");
    const themeToggleBtn = document.getElementById("theme-toggle-btn");
    const themeToggleIcon = document.getElementById("theme-toggle-icon");
    const themeToggleLabel = document.getElementById("theme-toggle-label");

    const storage = {
        get(key) {
            try {
                return window.localStorage.getItem(key);
            } catch (_) {
                return null;
            }
        },
        set(key, value) {
            try {
                window.localStorage.setItem(key, value);
            } catch (_) {
                // Ignore storage failures (private mode / restricted contexts).
            }
        },
    };

    function setSidebarCollapsed(isCollapsed) {
        body.classList.toggle("sidebar-collapsed", isCollapsed);
        if (sidebarToggleBtn) {
            sidebarToggleBtn.innerHTML = `<span class="sidebar-toggle-icon" id="sidebar-toggle-icon" aria-hidden="true">${isCollapsed ? iconMarkup.sidebarExpand : iconMarkup.sidebarCollapse}</span>`;
            sidebarToggleBtn.setAttribute(
                "aria-label",
                isCollapsed ? "Expand side navigation" : "Collapse side navigation"
            );
            sidebarToggleBtn.setAttribute(
                "title",
                isCollapsed ? "Expand navigation" : "Collapse navigation"
            );
        }
        storage.set(storageKeys.sidebarCollapsed, isCollapsed ? "1" : "0");
    }

    function setLightMode(enabled) {
        body.classList.toggle("theme-light", enabled);
        if (themeToggleIcon) {
            themeToggleIcon.innerHTML = enabled ? iconMarkup.moon : iconMarkup.sun;
        }
        if (themeToggleLabel) {
            themeToggleLabel.textContent = enabled ? "Dark mode" : "Light mode";
        }
        storage.set(storageKeys.lightMode, enabled ? "1" : "0");
    }

    function toggleLightMode() {
        setLightMode(!body.classList.contains("theme-light"));
    }

    const savedSidebarState = storage.get(storageKeys.sidebarCollapsed);
    if (savedSidebarState === "1") {
        setSidebarCollapsed(true);
    } else {
        setSidebarCollapsed(false);
    }

    const savedLightMode = storage.get(storageKeys.lightMode);
    if (savedLightMode === "1") {
        setLightMode(true);
    } else {
        setLightMode(false);
    }

    if (sidebarToggleBtn) {
        sidebarToggleBtn.addEventListener("click", function() {
            setSidebarCollapsed(!body.classList.contains("sidebar-collapsed"));
        });
    }

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener("click", function() {
            toggleLightMode();
        });
    }

    document.addEventListener("keydown", function(event) {
        const isL = event.key && event.key.toLowerCase() === "l";
        const hasToggleModifiers = event.shiftKey && (event.metaKey || event.ctrlKey);
        if (!isL || !hasToggleModifiers) {
            return;
        }
        event.preventDefault();
        toggleLightMode();
    });

    initializeBenchmarkMonitor({ body, storage, storageKeys });

    document.querySelectorAll("[data-confirm]").forEach(function(el) {
        el.addEventListener("click", function(e) {
            if (!confirm(el.dataset.confirm)) {
                e.preventDefault();
            }
        });
    });
}

function initializeBenchmarkMonitor({ body, storage, storageKeys }) {
    const dockRoot = document.getElementById("benchmark-run-dock");
    if (!dockRoot) {
        window.memorizzEvalMonitor = null;
        return;
    }

    const runIdEl = document.getElementById("benchmark-dock-run-id");
    const statusEl = document.getElementById("benchmark-dock-status");
    const outputEl = document.getElementById("benchmark-dock-output");
    const linkEl = document.getElementById("benchmark-dock-link");
    const clearBtn = document.getElementById("benchmark-dock-clear");
    const toggleBtn = document.getElementById("benchmark-dock-toggle");

    const activeStatuses = new Set(["queued", "running", "canceling"]);
    const terminalStatuses = new Set(["completed", "failed", "canceled"]);
    const pollIntervalMs = 1000;
    const maxLogLines = 500;
    const terminalRetentionMs = 15000;
    const isEvalgroundPage = window.location.pathname.startsWith("/evalground");

    let pollTimer = null;
    let terminalTimer = null;
    let pollInFlight = false;
    let dockCollapsed = storage.get(storageKeys.benchmarkDockCollapsed) === "1";
    let state = loadState();

    function emptyState() {
        return {
            runId: "",
            status: "",
            cursor: 0,
            logs: [],
            error: "",
            updatedAt: 0,
            terminalAt: 0,
        };
    }

    function safeRunId(value) {
        return typeof value === "string" ? value.trim() : "";
    }

    function isActiveStatus(status) {
        return activeStatuses.has(String(status || ""));
    }

    function isTerminalStatus(status) {
        return terminalStatuses.has(String(status || ""));
    }

    function loadState() {
        const raw = storage.get(storageKeys.benchmarkRunState);
        if (!raw) {
            return emptyState();
        }
        try {
            const parsed = JSON.parse(raw);
            const runId = safeRunId(parsed && parsed.runId);
            const status = typeof parsed?.status === "string" ? parsed.status : "";
            const cursor =
                typeof parsed?.cursor === "number" && Number.isFinite(parsed.cursor)
                    ? Math.max(0, parsed.cursor)
                    : 0;
            const logs = Array.isArray(parsed?.logs)
                ? parsed.logs.map((line) => String(line)).slice(-maxLogLines)
                : [];
            const terminalAt =
                typeof parsed?.terminalAt === "number" && Number.isFinite(parsed.terminalAt)
                    ? parsed.terminalAt
                    : 0;
            const normalized = {
                runId,
                status,
                cursor,
                logs,
                error: typeof parsed?.error === "string" ? parsed.error : "",
                updatedAt:
                    typeof parsed?.updatedAt === "number" && Number.isFinite(parsed.updatedAt)
                        ? parsed.updatedAt
                        : 0,
                terminalAt,
            };
            if (
                runId &&
                isTerminalStatus(status) &&
                terminalAt > 0 &&
                Date.now() - terminalAt > terminalRetentionMs
            ) {
                return emptyState();
            }
            return normalized;
        } catch (_) {
            return emptyState();
        }
    }

    function persistState() {
        storage.set(storageKeys.benchmarkRunState, JSON.stringify(state));
    }

    function persistDockCollapsed() {
        storage.set(storageKeys.benchmarkDockCollapsed, dockCollapsed ? "1" : "0");
    }

    function setBodyDockClasses(isVisible) {
        body.classList.toggle("benchmark-dock-visible", isVisible);
        body.classList.toggle("benchmark-dock-collapsed", isVisible && dockCollapsed);
    }

    function formatRunId(runId) {
        if (!runId) {
            return "-";
        }
        if (runId.length <= 14) {
            return runId;
        }
        return `${runId.slice(0, 8)}...${runId.slice(-4)}`;
    }

    function setStatusStyles(status) {
        if (!statusEl) {
            return;
        }
        statusEl.classList.remove(
            "is-running",
            "is-completed",
            "is-failed",
            "is-canceled"
        );
        if (isActiveStatus(status)) {
            statusEl.classList.add("is-running");
        } else if (status === "completed") {
            statusEl.classList.add("is-completed");
        } else if (status === "failed") {
            statusEl.classList.add("is-failed");
        } else if (status === "canceled") {
            statusEl.classList.add("is-canceled");
        }
    }

    function render() {
        const visible = Boolean(state.runId);
        dockRoot.hidden = !visible;
        dockRoot.classList.toggle("is-visible", visible);
        dockRoot.classList.toggle("is-collapsed", visible && dockCollapsed);
        setBodyDockClasses(visible);

        if (!visible) {
            return;
        }

        if (runIdEl) {
            runIdEl.textContent = formatRunId(state.runId);
        }

        const status = String(state.status || "running");
        if (statusEl) {
            statusEl.textContent = status;
        }
        setStatusStyles(status);

        if (linkEl) {
            linkEl.href = `/evalground?run_id=${encodeURIComponent(state.runId)}`;
        }

        if (outputEl) {
            outputEl.textContent = state.logs.length
                ? state.logs.join("\n")
                : "Waiting for benchmark logs...";
            outputEl.scrollTop = outputEl.scrollHeight;
        }

        if (clearBtn) {
            clearBtn.disabled = isActiveStatus(status);
        }

        if (toggleBtn) {
            toggleBtn.setAttribute(
                "aria-label",
                dockCollapsed ? "Expand benchmark monitor" : "Collapse benchmark monitor"
            );
            toggleBtn.setAttribute(
                "title",
                dockCollapsed ? "Expand benchmark monitor" : "Collapse benchmark monitor"
            );
        }
    }

    function stopPolling() {
        if (pollTimer) {
            window.clearTimeout(pollTimer);
            pollTimer = null;
        }
    }

    function schedulePolling(immediate = false) {
        if (isEvalgroundPage) {
            return;
        }
        if (!state.runId || !isActiveStatus(state.status || "running")) {
            stopPolling();
            return;
        }
        stopPolling();
        pollTimer = window.setTimeout(pollRun, immediate ? 0 : pollIntervalMs);
    }

    function scheduleTerminalClear() {
        if (terminalTimer) {
            window.clearTimeout(terminalTimer);
            terminalTimer = null;
        }
        if (!state.runId || !isTerminalStatus(state.status)) {
            return;
        }
        const terminalAt = state.terminalAt || Date.now();
        const elapsed = Date.now() - terminalAt;
        const remaining = terminalRetentionMs - elapsed;
        if (remaining <= 0) {
            clearRun(state.runId);
            return;
        }
        terminalTimer = window.setTimeout(function() {
            if (state.runId && isTerminalStatus(state.status)) {
                clearRun(state.runId);
            }
        }, remaining);
    }

    function mergePayload(runId, payload) {
        const targetRunId = safeRunId(runId);
        if (!targetRunId || !payload || typeof payload !== "object") {
            return;
        }

        if (!state.runId || state.runId !== targetRunId) {
            state = emptyState();
            state.runId = targetRunId;
        }

        const lines = Array.isArray(payload.logs)
            ? payload.logs.map((line) => String(line))
            : [];
        if (lines.length) {
            state.logs = state.logs.concat(lines).slice(-maxLogLines);
        }

        if (
            typeof payload.next_index === "number" &&
            Number.isFinite(payload.next_index)
        ) {
            state.cursor = Math.max(0, payload.next_index);
        } else {
            state.cursor += lines.length;
        }

        if (typeof payload.status === "string" && payload.status) {
            state.status = payload.status;
        } else if (!state.status) {
            state.status = "running";
        }

        state.error = payload.error ? String(payload.error) : "";
        state.updatedAt = Date.now();
        state.terminalAt = isTerminalStatus(state.status) ? state.terminalAt || Date.now() : 0;

        persistState();
        render();

        if (isActiveStatus(state.status)) {
            schedulePolling(false);
        } else {
            stopPolling();
            scheduleTerminalClear();
        }
    }

    async function pollRun() {
        if (isEvalgroundPage || pollInFlight || !state.runId) {
            return;
        }
        if (!isActiveStatus(state.status || "running")) {
            return;
        }

        pollInFlight = true;
        try {
            const response = await fetch(
                `/evalground/runs/${encodeURIComponent(state.runId)}?after=${state.cursor}`,
                { headers: { Accept: "application/json" } }
            );

            if (response.status === 404) {
                clearRun(state.runId);
                return;
            }

            if (!response.ok) {
                schedulePolling(false);
                return;
            }

            const payload = await response.json().catch(() => null);
            if (!payload || typeof payload !== "object") {
                schedulePolling(false);
                return;
            }

            mergePayload(state.runId, payload);
        } catch (_) {
            schedulePolling(false);
        } finally {
            pollInFlight = false;
        }
    }

    function trackRun(runId, options = {}) {
        const targetRunId = safeRunId(runId);
        if (!targetRunId) {
            return;
        }

        const sameRun = state.runId === targetRunId;
        const keepLogs = Boolean(options.keepLogs);

        if (!sameRun || !keepLogs) {
            state.logs = [];
            state.cursor = 0;
        }

        state.runId = targetRunId;
        if (typeof options.status === "string" && options.status) {
            state.status = options.status;
        } else if (!state.status) {
            state.status = "queued";
        }
        state.error = "";
        state.updatedAt = Date.now();
        state.terminalAt = isTerminalStatus(state.status) ? Date.now() : 0;

        persistState();
        render();

        if (isActiveStatus(state.status)) {
            schedulePolling(true);
        } else {
            scheduleTerminalClear();
        }
    }

    function syncRunPayload(runId, payload) {
        const targetRunId = safeRunId(runId);
        if (!targetRunId) {
            return;
        }
        if (!state.runId || state.runId !== targetRunId) {
            trackRun(targetRunId, { status: payload?.status || "running", keepLogs: false });
        }
        mergePayload(targetRunId, payload);
    }

    function clearRun(runId) {
        if (runId && state.runId && safeRunId(runId) !== state.runId) {
            return;
        }
        state = emptyState();
        stopPolling();
        if (terminalTimer) {
            window.clearTimeout(terminalTimer);
            terminalTimer = null;
        }
        persistState();
        render();
    }

    async function discoverActiveRun() {
        if (isEvalgroundPage || state.runId) {
            return;
        }
        try {
            const response = await fetch("/evalground/runs/active", {
                headers: { Accept: "application/json" },
            });
            if (!response.ok) {
                return;
            }
            const payload = await response.json().catch(() => null);
            const run = payload && typeof payload === "object" ? payload.run : null;
            if (!run || typeof run !== "object" || !run.run_id) {
                return;
            }
            trackRun(String(run.run_id), {
                status: typeof run.status === "string" ? run.status : "running",
                keepLogs: false,
            });
        } catch (_) {
            // Ignore discovery failures silently.
        }
    }

    if (toggleBtn) {
        toggleBtn.addEventListener("click", function() {
            dockCollapsed = !dockCollapsed;
            persistDockCollapsed();
            render();
        });
    }

    if (clearBtn) {
        clearBtn.addEventListener("click", function() {
            if (clearBtn.disabled) {
                return;
            }
            clearRun(state.runId);
        });
    }

    window.memorizzEvalMonitor = {
        trackRun,
        syncRunPayload,
        clearRun,
        getActiveRunId() {
            return state.runId || null;
        },
    };

    render();
    if (state.runId) {
        if (isActiveStatus(state.status || "running")) {
            schedulePolling(true);
        } else {
            scheduleTerminalClear();
        }
    } else {
        discoverActiveRun();
    }
}

/*
 * Shared pull/delete action for the local-model manage list rendered by
 * settings.html and playground.html. Confirms with the user, swaps the
 * button content for a spinner while the Ollama / HuggingFace API call
 * runs, alerts on failure, restores the button, then invokes `refresh`
 * so the caller can re-render its model list. Serialized: while one
 * action is in flight, further invocations are ignored.
 */
let _localModelActionInFlight = false;
async function handleLocalModelAction(button, action, provider, model, refresh) {
    if (_localModelActionInFlight) return;
    if (!action || !provider || !model) return;

    if (action === 'delete' && !window.confirm('Remove ' + model + ' from local cache?')) return;
    if (action === 'pull') {
        const ok = window.confirm('Download ' + model + '?\n\n'
            + 'Large models can take 5–15 minutes. The page will appear to hang while the download runs in the background.');
        if (!ok) return;
    }

    _localModelActionInFlight = true;
    button.disabled = true;
    button.classList.add('is-busy');
    const originalHtml = button.innerHTML;
    button.innerHTML = '<span class="model-action-spinner" aria-hidden="true"></span>';

    try {
        let res;
        if (action === 'delete' && provider === 'ollama') {
            res = await fetch('/api/ollama/models/' + encodeURIComponent(model), { method: 'DELETE' });
        } else if (action === 'delete' && provider === 'huggingface') {
            res = await fetch('/api/huggingface/models/' + encodeURIComponent(model), { method: 'DELETE' });
        } else if (action === 'pull') {
            const fd = new FormData();
            fd.append(provider === 'ollama' ? 'name' : 'repo_id', model);
            const url = provider === 'ollama' ? '/api/ollama/pull' : '/api/huggingface/pull';
            res = await fetch(url, { method: 'POST', body: fd });
        } else {
            return;
        }
        let data = {};
        try { data = await res.json(); } catch (_) { /* ignore */ }
        if (!data.ok) {
            window.alert('Failed: ' + (data.error || ('HTTP ' + res.status)));
        }
    } catch (err) {
        window.alert('Error: ' + err);
    } finally {
        _localModelActionInFlight = false;
        button.disabled = false;
        button.classList.remove('is-busy');
        button.innerHTML = originalHtml;
        if (typeof refresh === 'function') refresh();
    }
}

/*
 * Read a fetch() Response whose body is a Server-Sent-Events stream and
 * Dispatch complete frames, preserving IDs and multiline data. EOF without
 * run.done (or the legacy [DONE] sentinel) is an interrupted response.
 */
async function readSseStream(response, onEvent) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8', { fatal: true });
    let buffer = '';
    let data = [], event = '', id = '', terminal = false, frameSize = 0;
    function line(value) {
        if (value === '') {
            if (data.length) {
                const payload = data.join('\n');
                if (terminal) throw new Error('Event received after stream completion');
                let parsed;
                try { parsed = JSON.parse(payload); } catch (_) {}
                terminal = payload === '[DONE]' || (parsed && parsed.type === 'run.done');
                onEvent(payload, { event: event || 'message', id });
            }
            data = []; event = ''; frameSize = 0;
            return;
        }
        frameSize += value.length;
        if (frameSize > 131072) throw new Error('SSE frame exceeds size limit');
        if (value.startsWith(':')) return;
        const separator = value.indexOf(':');
        const field = separator < 0 ? value : value.slice(0, separator);
        let content = separator < 0 ? '' : value.slice(separator + 1);
        if (content.startsWith(' ')) content = content.slice(1);
        if (field === 'data') data.push(content);
        if (field === 'event') event = content;
        if (field === 'id' && !content.includes('\u0000')) id = content;
    }
    function consume(final = false) {
        while (true) {
            const index = buffer.search(/[\r\n]/);
            if (index < 0) break;
            if (!final && buffer[index] === '\r' && index === buffer.length - 1) break;
            const width = buffer[index] === '\r' && buffer[index + 1] === '\n' ? 2 : 1;
            const value = buffer.slice(0, index);
            buffer = buffer.slice(index + width);
            line(value);
        }
        if (buffer.length > 131072) throw new Error('SSE line exceeds size limit');
    }
    try {
        while (true) {
            const { done, value } = await reader.read();
            buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
            consume(done);
            if (done) break;
        }
        // An unterminated frame is discarded by SSE, never treated as completion.
        if (!terminal) throw new Error('Stream interrupted before run.done');
    } finally {
        try { await reader.cancel(); } finally { reader.releaseLock(); }
    }
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initializeMemorizzUI);
} else {
    initializeMemorizzUI();
}

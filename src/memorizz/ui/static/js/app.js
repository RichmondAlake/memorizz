/* Memorizz UI JavaScript */

// One rule for charges across the console's monitor pages:
// 2 decimals from $1, 4 under $1, 6 under one cent.
window.MemorizzFormat = {
    usd(value) {
        if (value == null) return "Unknown";
        if (value === 0) return "$0.00";
        const size = Math.abs(value);
        if (size >= 1) {
            return "$" + value.toLocaleString("en-US", {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2,
            });
        }
        return "$" + value.toFixed(size >= 0.01 ? 4 : 6);
    },
};

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

    function setSidebarCollapsed(isCollapsed, persist = true) {
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
        if (persist) {
            storage.set(storageKeys.sidebarCollapsed, isCollapsed ? "1" : "0");
        }
    }

    // The class sits on <html> (set before first paint in base.html) and on
    // <body>, which the stylesheets key on. Only a choice the viewer makes is
    // saved; until then the theme follows the system's.
    function setLightMode(enabled, persist = false) {
        document.documentElement.classList.toggle("theme-light", enabled);
        body.classList.toggle("theme-light", enabled);
        if (themeToggleIcon) {
            themeToggleIcon.innerHTML = enabled ? iconMarkup.moon : iconMarkup.sun;
        }
        if (themeToggleLabel) {
            themeToggleLabel.textContent = enabled ? "Dark mode" : "Light mode";
        }
        if (themeToggleBtn) {
            themeToggleBtn.setAttribute("aria-label", enabled ? "Switch to dark theme" : "Switch to light theme");
        }
        if (persist) {
            storage.set(storageKeys.lightMode, enabled ? "1" : "0");
        }
    }

    function toggleLightMode() {
        setLightMode(!body.classList.contains("theme-light"), true);
    }
    window.memorizzToggleTheme = toggleLightMode;

    // Full-screen workspaces (the agent playground) keep their own navigation
    // state and start collapsed, so the global preference is left untouched.
    const sidebarScope = body.dataset.sidebarScope;
    if (sidebarScope) {
        storageKeys.sidebarCollapsed = storageKeys.sidebarCollapsed + "." + sidebarScope;
    }
    // Phones open the full navigation as a drawer, so the icon-only preference
    // is neither applied nor overwritten there.
    const phoneQuery = window.matchMedia("(max-width: 768px)");
    const savedSidebarState = storage.get(storageKeys.sidebarCollapsed);
    if (phoneQuery.matches) {
        setSidebarCollapsed(false, false);
    } else if (savedSidebarState === "1" || (savedSidebarState === null && body.dataset.sidebarDefault === "collapsed")) {
        setSidebarCollapsed(true);
    } else {
        setSidebarCollapsed(false);
    }

    const mobileNavToggle = document.getElementById("mobile-nav-toggle");
    function setMobileNavOpen(open) {
        body.classList.toggle("mobile-nav-open", open);
        if (mobileNavToggle) {
            mobileNavToggle.setAttribute("aria-expanded", open ? "true" : "false");
            mobileNavToggle.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
        }
    }
    if (mobileNavToggle) {
        mobileNavToggle.addEventListener("click", function() {
            setMobileNavOpen(!body.classList.contains("mobile-nav-open"));
        });
    }
    document.querySelectorAll("[data-mobile-nav-close], .sidebar a").forEach(function(el) {
        el.addEventListener("click", function() {
            setMobileNavOpen(false);
        });
    });
    document.addEventListener("keydown", function(event) {
        if (event.key === "Escape" && body.classList.contains("mobile-nav-open")) {
            setMobileNavOpen(false);
            if (mobileNavToggle) mobileNavToggle.focus();
        }
    });

    const systemLight = window.matchMedia("(prefers-color-scheme: light)");
    const savedLightMode = storage.get(storageKeys.lightMode);
    setLightMode(savedLightMode === null ? systemLight.matches : savedLightMode === "1");
    systemLight.addEventListener("change", function(event) {
        if (storage.get(storageKeys.lightMode) === null) setLightMode(event.matches);
    });

    // Folded to icons, each item names itself on hover, and a group opens the
    // sidebar instead of unfolding a list nobody can see.
    document.querySelectorAll(".sidebar-nav > .nav-item, .sidebar-nav > .nav-dropdown > summary").forEach(function(item) {
        const label = item.querySelector("span:not(.nav-icon):not(.nav-caret)");
        if (label && !item.title) item.title = label.textContent.trim();
    });
    document.querySelectorAll(".sidebar-nav .nav-dropdown > summary").forEach(function(summary) {
        summary.addEventListener("click", function(event) {
            if (!body.classList.contains("sidebar-collapsed") || phoneQuery.matches) return;
            event.preventDefault();
            summary.parentElement.open = true;
            setSidebarCollapsed(false);
        });
    });

    // Disconnecting drops this window's store for everyone using this UI.
    const disconnectLink = document.getElementById("disconnect-btn");
    if (disconnectLink && window.MemorizzDialogs) {
        disconnectLink.addEventListener("click", function(event) {
            event.preventDefault();
            window.MemorizzDialogs.confirm("Disconnect from this memory provider?", {
                detail: "Nothing is deleted. The UI returns to the connect page until a provider is connected again.",
                confirmLabel: "Disconnect",
                cancelLabel: "Stay connected",
            }).then(function(ok) {
                if (ok) window.location.assign(disconnectLink.href);
            });
        });
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

    // [data-confirm]: ask in the page, then carry out the link or submit.
    document.addEventListener("click", function(event) {
        const el = event.target instanceof Element ? event.target.closest("[data-confirm]") : null;
        if (!el || el.dataset.confirmed === "1" || !window.MemorizzDialogs) return;
        event.preventDefault();
        event.stopPropagation();
        window.MemorizzDialogs.confirm(el.dataset.confirm, {
            confirmLabel: el.dataset.confirmLabel || "Confirm",
            cancelLabel: "Cancel",
        }).then(function(ok) {
            if (!ok) return;
            el.dataset.confirmed = "1";
            el.click();
            delete el.dataset.confirmed;
        });
    }, true);

    initializeCommandPalette();
    initializeShortcuts();
}

// Keyboard shortcuts: ? lists them, g then a letter opens a page. Neither
// fires while typing in a field or while a dialog is open.
function initializeShortcuts() {
    if (typeof HTMLDialogElement === "undefined") return;
    const isMac = /Mac|iPhone|iPad/.test(navigator.platform || "");
    const GO_TO = [
        ["o", "Overview", "/dashboard"],
        ["a", "Agents", "/agents"],
        ["h", "Agent harnesses", "/harnesses"],
        ["p", "Playground", "/playground"],
        ["u", "Automations", "/automations"],
        ["m", "Memory types", "/memory/knowledge-base"],
        ["l", "Continual learning", "/memory/workflows"],
        ["c", "MCP connections", "/mcp"],
        ["k", "Skills", "/vercel-skills"],
        ["t", "Observability", "/traces"],
        ["e", "Evalground", "/evalground"],
        ["s", "Settings", "/settings"],
        ["y", "Activity", "/activity"],
    ];
    let sheet = null;
    let pending = null;
    let pendingTimer = 0;

    function group(title, rows) {
        const section = document.createElement("section");
        section.className = "kbd-sheet-group";
        const heading = document.createElement("h3");
        heading.textContent = title;
        const list = document.createElement("dl");
        rows.forEach(function(row) {
            const line = document.createElement("div");
            line.className = "kbd-sheet-row";
            const term = document.createElement("dt");
            term.textContent = row.label;
            const keys = document.createElement("dd");
            row.keys.forEach(function(key, index) {
                if (index && row.then) keys.append("then");
                const cap = document.createElement("kbd");
                cap.textContent = key;
                keys.append(cap);
            });
            line.append(term, keys);
            list.append(line);
        });
        section.append(heading, list);
        return section;
    }

    // The hints a page prints above its table, e.g. "j k move".
    function pageRows() {
        const rows = [];
        document.querySelectorAll("main .fleet-keys > span").forEach(function(hint) {
            const keys = Array.from(hint.querySelectorAll("kbd")).map(function(key) { return key.textContent.trim(); });
            const copy = hint.cloneNode(true);
            copy.querySelectorAll("kbd").forEach(function(key) { key.remove(); });
            const label = copy.textContent.trim();
            if (!keys.length || !label) return;
            if (rows.some(function(row) { return row.label === label; })) return;
            rows.push({ label: label.charAt(0).toUpperCase() + label.slice(1), keys: keys });
        });
        return rows;
    }

    function build() {
        sheet = document.createElement("dialog");
        sheet.className = "kbd-sheet";
        sheet.setAttribute("aria-labelledby", "kbd-sheet-title");
        const head = document.createElement("div");
        head.className = "kbd-sheet-head";
        const title = document.createElement("h2");
        title.id = "kbd-sheet-title";
        title.textContent = "Keyboard shortcuts";
        const close = document.createElement("button");
        close.type = "button";
        close.className = "app-bar-btn";
        close.setAttribute("aria-label", "Close");
        close.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18"/></svg>';
        close.addEventListener("click", function() { sheet.close(); });
        head.append(title, close);
        const body = document.createElement("div");
        body.className = "kbd-sheet-body";
        const general = group("General", [
            { label: "Search pages, agents and actions", keys: [isMac ? "⌘" : "Ctrl", "K"] },
            { label: "Switch theme", keys: [isMac ? "⌘" : "Ctrl", "⇧", "L"] },
            { label: "Show this list", keys: ["?"] },
            { label: "Close a dialog", keys: ["esc"] },
        ]);
        const onPage = pageRows();
        if (onPage.length) general.append(...group("On this page", onPage).children);
        body.append(general, group("Go to", GO_TO.map(function(row) {
            return { label: row[1], keys: ["g", row[0]], then: true };
        })));
        sheet.append(head, body);
        sheet.addEventListener("click", function(event) {
            if (event.target === sheet) sheet.close();
        });
        document.body.append(sheet);
    }

    function show() {
        if (!sheet) build();
        if (!sheet.open) sheet.showModal();
    }
    window.memorizzShowShortcuts = show;
    document.querySelectorAll("[data-shortcuts-open]").forEach(function(button) {
        button.addEventListener("click", show);
    });

    function clearPending() {
        window.clearTimeout(pendingTimer);
        if (pending) pending.remove();
        pending = null;
    }

    function typing(target) {
        if (!(target instanceof Element)) return false;
        return Boolean(target.closest("input, textarea, select, [contenteditable]:not([contenteditable='false'])"));
    }

    document.addEventListener("keydown", function(event) {
        if (event.metaKey || event.ctrlKey || event.altKey || event.defaultPrevented) return;
        if (typing(event.target) || document.querySelector("dialog[open]")) {
            clearPending();
            return;
        }
        const key = event.key || "";
        if (pending) {
            const match = GO_TO.find(function(row) { return row[0] === key.toLowerCase(); });
            clearPending();
            if (match) {
                // The page's own single-key shortcuts must not also fire.
                event.preventDefault();
                event.stopPropagation();
                window.location.assign(match[2]);
            }
            return;
        }
        if (key === "?") {
            event.preventDefault();
            show();
        } else if (key === "g" && !event.shiftKey && !event.repeat) {
            pending = document.createElement("div");
            pending.className = "kbd-pending";
            pending.setAttribute("role", "status");
            const cap = document.createElement("kbd");
            cap.textContent = "g";
            pending.append(cap, "then a letter to open a page · ? for the list");
            document.body.append(pending);
            pendingTimer = window.setTimeout(clearPending, 1600);
        }
    }, true);
}

// Command palette (Ctrl/⌘ K, or the search field in the top bar): every page
// in the navigation, the store's agents and a few actions, filtered as you
// type. Arrow keys move, Enter opens, Escape closes.
function initializeCommandPalette() {
    const triggers = document.querySelectorAll("[data-cmdk-open]");
    if (!triggers.length || typeof HTMLDialogElement === "undefined") return;
    const isMac = /Mac|iPhone|iPad/.test(navigator.platform || "");
    const hint = document.getElementById("cmdk-trigger-kbd");
    if (hint) hint.textContent = isMac ? "⌘K" : "Ctrl K";

    let dialog = null;
    let input = null;
    let list = null;
    let items = [];
    let shown = [];
    let selected = 0;
    let agentsLoaded = false;

    function pageItems() {
        const pages = [];
        document.querySelectorAll(".sidebar-nav a.nav-item[href]").forEach(function(link) {
            if (link.classList.contains("nav-agent-item")) return;
            const label = (link.querySelector("span:not(.nav-icon)") || link).textContent.trim();
            const group = link.closest(".nav-dropdown");
            const parent = group ? group.querySelector("summary span:not(.nav-icon):not(.nav-caret)").textContent.trim() : "";
            pages.push({ group: "Pages", label: label, hint: parent, href: link.getAttribute("href") });
        });
        [
            ["Failures", "Observability", "/traces/failures"],
            ["Health", "Observability", "/traces/health"],
            ["Compare traces", "Observability", "/traces/compare"],
            ["Usage & cost", "Observability", "/traces/usage"],
            ["Rate cards", "Observability", "/traces/usage/rates"],
            ["Memory evolution", "Observability", "/traces/memory-evolution"],
            ["Compare models & rerankers", "Evalground", "/evalground/compare"],
            ["Connect a memory provider", "Admin", "/connect"],
        ].forEach(function(row) {
            pages.push({ group: "Pages", label: row[0], hint: row[1], href: row[2] });
        });
        return pages;
    }

    function actionItems() {
        const readOnly = document.body.dataset.readOnly === "true";
        const actions = [];
        if (!readOnly) {
            actions.push({ group: "Actions", label: "Create agent", hint: "Agents", href: "/agents/new" });
            actions.push({ group: "Actions", label: "Schedule an automation", hint: "Automations", href: "/automations/new" });
        }
        actions.push({ group: "Actions", label: "Switch theme", hint: isMac ? "⌘⇧L" : "Ctrl Shift L", run: function() { window.memorizzToggleTheme(); } });
        actions.push({ group: "Actions", label: "Keyboard shortcuts", hint: "?", run: function() { if (window.memorizzShowShortcuts) window.memorizzShowShortcuts(); } });
        actions.push({ group: "Actions", label: "Open documentation", hint: "New tab", run: function() { window.open("https://richmondalake.github.io/memorizz", "_blank", "noopener"); } });
        return actions;
    }

    function loadAgents() {
        if (agentsLoaded) return;
        agentsLoaded = true;
        fetch("/api/agents", { headers: { Accept: "application/json" } })
            .then(function(response) { return response.ok ? response.json() : { agents: [] }; })
            .then(function(data) {
                (data.agents || []).forEach(function(agent) {
                    if (!agent || !agent.agent_id) return;
                    const model = agent.llm_config && agent.llm_config.model ? String(agent.llm_config.model) : "";
                    items.push({
                        group: "Agents",
                        label: String(agent.name || agent.agent_id),
                        hint: model || "Open playground",
                        keywords: String(agent.agent_id),
                        href: "/agents/" + encodeURIComponent(agent.agent_id) + "/playground",
                    });
                });
                if (dialog && dialog.open) render();
            })
            .catch(function() { /* The pages and actions still work. */ });
    }

    // The pages this browser opened last, newest first, for the empty query.
    const RECENT_KEY = "memorizz_ui_recent_pages";
    function readRecent() {
        try {
            const saved = JSON.parse(window.localStorage.getItem(RECENT_KEY) || "[]");
            return Array.isArray(saved) ? saved.filter(function(href) { return typeof href === "string"; }) : [];
        } catch (error) {
            return [];
        }
    }
    function recordVisit() {
        const here = window.location.pathname;
        const known = pageItems().some(function(item) { return item.href.split("?")[0] === here; });
        if (!known) return;
        const recent = readRecent().filter(function(href) { return href !== here; });
        recent.unshift(here);
        try {
            window.localStorage.setItem(RECENT_KEY, JSON.stringify(recent.slice(0, 6)));
        } catch (error) { /* Private windows: the palette works without it. */ }
    }
    function recentItems() {
        const here = window.location.pathname;
        const recent = [];
        readRecent().forEach(function(href) {
            if (href === here || recent.length >= 4) return;
            const page = items.find(function(item) { return item.group === "Pages" && item.href.split("?")[0] === href; });
            if (page) recent.push({ group: "Recent", label: page.label, hint: page.hint, href: page.href });
        });
        return recent;
    }
    recordVisit();

    function matches(item, words) {
        const text = (item.label + " " + (item.hint || "") + " " + (item.keywords || "") + " " + item.group).toLowerCase();
        return words.every(function(word) { return text.includes(word); });
    }

    function render() {
        const words = input.value.toLowerCase().split(/\s+/).filter(Boolean);
        const order = ["Recent", "Pages", "Agents", "Actions"];
        shown = items.filter(function(item) { return matches(item, words); });
        // Without a query: the pages opened last, then the pages and actions
        // plus the first few agents.
        if (!words.length) {
            let agents = 0;
            shown = shown.filter(function(item) { return item.group !== "Agents" || ++agents <= 6; });
            shown = recentItems().concat(shown);
        }
        shown.sort(function(a, b) { return order.indexOf(a.group) - order.indexOf(b.group); });
        selected = Math.min(selected, Math.max(0, shown.length - 1));
        list.textContent = "";
        if (!shown.length) {
            const empty = document.createElement("li");
            empty.className = "cmdk-empty";
            empty.setAttribute("role", "presentation");
            empty.textContent = "Nothing matches “" + input.value.trim() + "”.";
            list.append(empty);
            input.removeAttribute("aria-activedescendant");
            return;
        }
        let group = "";
        shown.forEach(function(item, index) {
            if (item.group !== group) {
                group = item.group;
                const heading = document.createElement("li");
                heading.className = "cmdk-group";
                heading.setAttribute("role", "presentation");
                heading.textContent = group;
                list.append(heading);
            }
            const option = document.createElement("li");
            option.className = "cmdk-option";
            option.id = "cmdk-option-" + index;
            option.setAttribute("role", "option");
            option.setAttribute("aria-selected", index === selected ? "true" : "false");
            option.dataset.index = String(index);
            const label = document.createElement("span");
            label.className = "cmdk-option-label";
            label.textContent = item.label;
            option.append(label);
            if (item.hint) {
                const hintEl = document.createElement("span");
                hintEl.className = "cmdk-option-hint";
                hintEl.textContent = item.hint;
                option.append(hintEl);
            }
            list.append(option);
        });
        input.setAttribute("aria-activedescendant", "cmdk-option-" + selected);
    }

    function select(index) {
        if (!shown.length) return;
        selected = (index + shown.length) % shown.length;
        list.querySelectorAll(".cmdk-option").forEach(function(option) {
            const on = Number(option.dataset.index) === selected;
            option.setAttribute("aria-selected", on ? "true" : "false");
            if (on) option.scrollIntoView({ block: "nearest" });
        });
        input.setAttribute("aria-activedescendant", "cmdk-option-" + selected);
    }

    function activate(item, newTab) {
        if (!item) return;
        dialog.close();
        if (item.run) item.run();
        else if (newTab) window.open(item.href, "_blank", "noopener");
        else window.location.assign(item.href);
    }

    function build() {
        dialog = document.createElement("dialog");
        dialog.className = "cmdk";
        dialog.setAttribute("aria-label", "Search pages, agents and actions");
        dialog.innerHTML =
            '<div class="cmdk-head">' +
            '<svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>' +
            '<input class="cmdk-input" type="text" role="combobox" aria-expanded="true" aria-controls="cmdk-list" aria-autocomplete="list" autocomplete="off" spellcheck="false" placeholder="Search pages, agents and actions">' +
            '</div>' +
            '<ul class="cmdk-list" id="cmdk-list" role="listbox" aria-label="Results"></ul>' +
            '<div class="cmdk-foot" aria-hidden="true"><span><kbd>↑</kbd><kbd>↓</kbd> move</span><span><kbd>↵</kbd> open</span><span><kbd>esc</kbd> close</span></div>';
        input = dialog.querySelector(".cmdk-input");
        list = dialog.querySelector(".cmdk-list");
        input.addEventListener("input", function() { selected = 0; render(); });
        input.addEventListener("keydown", function(event) {
            if (event.key === "ArrowDown") { event.preventDefault(); select(selected + 1); }
            else if (event.key === "ArrowUp") { event.preventDefault(); select(selected - 1); }
            else if (event.key === "Home") { event.preventDefault(); select(0); }
            else if (event.key === "End") { event.preventDefault(); select(shown.length - 1); }
            else if (event.key === "Enter") { event.preventDefault(); activate(shown[selected], event.metaKey || event.ctrlKey); }
        });
        list.addEventListener("mousemove", function(event) {
            const option = event.target instanceof Element ? event.target.closest(".cmdk-option") : null;
            if (option && Number(option.dataset.index) !== selected) select(Number(option.dataset.index));
        });
        list.addEventListener("click", function(event) {
            const option = event.target instanceof Element ? event.target.closest(".cmdk-option") : null;
            if (option) activate(shown[Number(option.dataset.index)], event.metaKey || event.ctrlKey);
        });
        // A click on the backdrop lands on the dialog itself.
        dialog.addEventListener("click", function(event) {
            if (event.target === dialog) dialog.close();
        });
        document.body.append(dialog);
        items = pageItems().concat(actionItems());
    }

    function open() {
        if (!dialog) build();
        if (dialog.open) return;
        loadAgents();
        input.value = "";
        selected = 0;
        render();
        dialog.showModal();
        input.focus();
    }

    triggers.forEach(function(trigger) {
        trigger.addEventListener("click", open);
    });
    document.addEventListener("keydown", function(event) {
        // ⌘K on a Mac (Ctrl K there deletes to the end of a line), Ctrl K elsewhere.
        const modifier = isMac ? event.metaKey && !event.ctrlKey : event.ctrlKey && !event.metaKey;
        if (modifier && !event.shiftKey && !event.altKey && event.key && event.key.toLowerCase() === "k") {
            event.preventDefault();
            if (dialog && dialog.open) dialog.close();
            else open();
        }
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
    if (window.MemorizzReadOnly && window.MemorizzReadOnly.active) {
        window.MemorizzReadOnly.notify();
        return;
    }

    if (action === 'delete' && !(await window.MemorizzDialogs.confirm('Remove ' + model + ' from the local cache?', {
        detail: 'It can be downloaded again later.', confirmLabel: 'Remove model',
    }))) return;
    if (action === 'pull') {
        const ok = await window.MemorizzDialogs.confirm('Download ' + model + '?', {
            detail: 'Large models can take 5 to 15 minutes. The download runs in the background.',
            confirmLabel: 'Download', cancelLabel: 'Cancel', danger: false,
        });
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
            window.MemorizzDialogs.notify('Failed: ' + (data.error || ('HTTP ' + res.status)));
        }
    } catch (err) {
        window.MemorizzDialogs.notify('Error: ' + err);
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

// Click-to-copy for identifiers rendered with the id_chip macro ([data-copy]).
document.addEventListener("click", async (event) => {
    const chip = event.target.closest("[data-copy]");
    if (!chip) return;
    event.preventDefault();
    try {
        await navigator.clipboard.writeText(chip.dataset.copy);
    } catch (err) {
        return;
    }
    const label = chip.getAttribute("title") || "";
    chip.classList.add("is-copied");
    chip.setAttribute("title", "Copied");
    setTimeout(() => {
        chip.classList.remove("is-copied");
        chip.setAttribute("title", label);
    }, 1200);
});

// A link to a folded section (#section-tools) opens it, so the setting it
// points at is in view.
function openLinkedSection() {
    const id = decodeURIComponent(window.location.hash.slice(1));
    const target = id && document.getElementById(id);
    if (!target) return;
    const section = target.closest("details");
    if (section && !section.open) {
        section.open = true;
        target.scrollIntoView({ block: "start" });
    }
}
window.addEventListener("hashchange", openLinkedSection);
if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", openLinkedSection);
} else {
    openLinkedSection();
}

// A custom time range closes when the viewer clicks elsewhere or presses Esc
// (as does the usage export menu).
document.addEventListener("click", (event) => {
    document.querySelectorAll("details.range-custom[open]").forEach((picker) => {
        if (!picker.contains(event.target)) picker.open = false;
    });
});
document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    document.querySelectorAll("details.range-custom[open], details.usage-export[open]").forEach((picker) => {
        picker.open = false;
        picker.querySelector("summary")?.focus();
    });
});

// Sortable tables: a table[data-sortable] header with data-sort-type sorts its
// rows by each cell's data-value (or text), ascending then descending.
function initSortableTables(root = document) {
    root.querySelectorAll("table[data-sortable] th[data-sort-type]").forEach((th) => {
        if (th.querySelector("button.sort-button")) return;
        const button = document.createElement("button");
        button.type = "button";
        button.className = "sort-button";
        button.append(...th.childNodes);
        th.append(button);
        th.setAttribute("aria-sort", "none");
        button.addEventListener("click", () => {
            const table = th.closest("table");
            const index = [...th.parentElement.children].indexOf(th);
            const numeric = th.dataset.sortType === "number";
            const ascending = th.getAttribute("aria-sort") !== "ascending";
            table.querySelectorAll("th[aria-sort]").forEach((other) => other.setAttribute("aria-sort", "none"));
            th.setAttribute("aria-sort", ascending ? "ascending" : "descending");
            const body = table.tBodies[0];
            const key = (row) => {
                const cell = row.children[index];
                const value = cell ? (cell.dataset.value ?? cell.textContent.trim()) : "";
                return numeric ? parseFloat(value) || 0 : value.toLowerCase();
            };
            const rows = [...body.rows].sort((a, b) => {
                const x = key(a);
                const y = key(b);
                return (x < y ? -1 : x > y ? 1 : 0) * (ascending ? 1 : -1);
            });
            body.append(...rows);
        });
    });
}
if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => initSortableTables());
} else {
    initSortableTables();
}

// An open export menu closes when the viewer clicks elsewhere or presses Esc.
document.addEventListener("click", (event) => {
    document.querySelectorAll("details.usage-export[open]").forEach((menu) => {
        if (!menu.contains(event.target)) menu.open = false;
    });
});

// Long usage tables show their first rows; "Show all" opens the rest.
function initRowExpanders(root = document) {
    root.querySelectorAll("[data-show-rows]").forEach((button) => {
        const table = button.parentElement.querySelector("table");
        if (!table || !table.querySelector(".usage-extra")) return;
        table.classList.add("is-collapsed");
        button.hidden = false;
        button.addEventListener("click", () => {
            table.classList.remove("is-collapsed");
            button.hidden = true;
        });
    });
}
if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => initRowExpanders());
} else {
    initRowExpanders();
}

/*
 * Read-only mode. The server refuses every change while the UI is read-only
 * (MEMORIZZ_UI_READ_ONLY or `memorizz ui --read-only`); this turns the
 * controls that would make one off up front, with a tooltip saying why:
 *
 *   - forms that POST (except the sign-in, connect and trace-reveal steps,
 *     which the server still accepts): their submit buttons;
 *   - anything marked data-mutates (JS-driven sends, launches, saves);
 *   - destructive buttons (.btn-danger, .hx-btn-delete, .is-danger).
 *
 * GET forms (filters, search) keep working. Content rendered later (tables,
 * dialogs) is covered by a MutationObserver. Page scripts can ask
 * window.MemorizzReadOnly.active before sending a change.
 */
function initReadOnlyGuard() {
    const body = document.body;
    const active = Boolean(body && body.dataset.readOnly === "true");
    const tooltip = (body && body.dataset.readOnlyTooltip) || "The UI is read-only. Changes are turned off.";
    let toastTimer = null;

    function notify() {
        let toast = document.getElementById("ro-toast");
        if (!toast) {
            toast = document.createElement("div");
            toast.id = "ro-toast";
            toast.className = "ro-toast";
            toast.setAttribute("role", "status");
            document.body.appendChild(toast);
        }
        toast.textContent = tooltip;
        toast.hidden = false;
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => { toast.hidden = true; }, 2600);
    }

    window.MemorizzReadOnly = { active, tooltip, notify };
    if (!active) return;

    const allowed = new Set((body.dataset.readOnlyAllow || "").split(/\s+/).filter(Boolean));
    const ACTIONS = "[data-mutates], .btn-danger, .hx-btn-delete, .is-danger";

    // A change sent by a page script whose control isn't marked: stop it here
    // with the note, as the server would refuse it, rather than after a round
    // trip that reads as a failure.
    const send = window.fetch.bind(window);
    window.fetch = (resource, init) => {
        const method = String((init && init.method) || (resource && resource.method) || "GET").toUpperCase();
        if (!["GET", "HEAD", "OPTIONS"].includes(method)) {
            let url = null;
            try {
                url = new URL(typeof resource === "string" ? resource : resource.url, window.location.href);
            } catch (_) {
                url = null;
            }
            if (url && url.origin === window.location.origin && !allowed.has(url.pathname)) {
                notify();
                return Promise.resolve(new Response(JSON.stringify({ detail: tooltip }), {
                    status: 403,
                    headers: { "Content-Type": "application/json" },
                }));
            }
        }
        return send(resource, init);
    };
    const SUBMITS = 'button:not([type]), button[type="submit"], input[type="submit"], input[type="image"]';

    function isWriteForm(form) {
        if (!(form instanceof HTMLFormElement)) return false;
        if (form.matches("[data-mutates]")) return true;
        if ((form.getAttribute("method") || "get").toLowerCase() !== "post") return false;
        let path = "";
        try {
            path = new URL(form.getAttribute("action") || window.location.href, window.location.href).pathname;
        } catch (_) {
            return true;
        }
        return !allowed.has(path);
    }

    function block(element) {
        if (element.classList.contains("ro-blocked")) return;
        element.classList.add("ro-blocked");
        element.setAttribute("aria-disabled", "true");
        element.setAttribute("title", tooltip);
        if (element instanceof HTMLTextAreaElement || (element instanceof HTMLInputElement && !/^(submit|image|button)$/i.test(element.type))) {
            element.readOnly = true;
        } else if (element instanceof HTMLSelectElement) {
            element.disabled = true;
        }
    }

    function scan(root) {
        const scope = root && root.querySelectorAll ? root : document;
        const forms = scope instanceof HTMLFormElement ? [scope] : scope.querySelectorAll("form");
        forms.forEach((form) => {
            if (!isWriteForm(form)) return;
            form.querySelectorAll(SUBMITS).forEach(block);
            if (form.id) {
                document.querySelectorAll(`[form="${CSS.escape(form.id)}"]`).forEach((control) => {
                    if (control.matches(SUBMITS)) block(control);
                });
            }
        });
        scope.querySelectorAll(ACTIONS).forEach((element) => {
            if (element instanceof HTMLFormElement) return;
            // A confirmation dialog only opens from a control that is already off.
            if (!element.matches("[data-mutates]") && element.closest("dialog")) return;
            block(element);
        });
    }

    function stop(event) {
        event.preventDefault();
        event.stopImmediatePropagation();
        notify();
    }

    document.addEventListener("click", (event) => {
        if (event.target instanceof Element && event.target.closest(".ro-blocked")) stop(event);
    }, true);
    document.addEventListener("submit", (event) => {
        if (isWriteForm(event.target)) stop(event);
    }, true);
    document.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        const target = event.target instanceof Element ? event.target.closest(".ro-blocked") : null;
        if (!target) return;
        // Typing a space in a read-only text box is harmless; Enter would send.
        if (event.key === " " && (target instanceof HTMLTextAreaElement || target instanceof HTMLInputElement)) return;
        stop(event);
    }, true);
    // Dropping files adds them to a knowledge base.
    document.addEventListener("drop", (event) => {
        if (event.dataTransfer && Array.from(event.dataTransfer.types || []).includes("Files")) stop(event);
    }, true);

    scan(document);
    let pending = false;
    new MutationObserver(() => {
        if (pending) return;
        pending = true;
        requestAnimationFrame(() => {
            pending = false;
            scan(document);
        });
    }).observe(body, { childList: true, subtree: true });
}
if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initReadOnlyGuard);
} else {
    initReadOnlyGuard();
}

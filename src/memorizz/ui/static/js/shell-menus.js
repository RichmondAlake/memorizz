/* Top-bar menus: the inbox bell (what is waiting for a person) and the store
   switcher (FileSystem stores on this computer). Both read a small JSON route
   and draw a popover under the control that opened them. */
(function() {
    let openPop = null;

    function closePop() {
        if (!openPop) return;
        openPop.el.remove();
        openPop.trigger.setAttribute('aria-expanded', 'false');
        const trigger = openPop.trigger;
        openPop = null;
        return trigger;
    }

    function popover(trigger, label) {
        const same = openPop && openPop.trigger === trigger;
        closePop();
        if (same) return null;
        const el = document.createElement('div');
        el.className = 'shell-pop';
        el.setAttribute('role', 'dialog');
        el.setAttribute('aria-label', label);
        el.tabIndex = -1;
        document.body.appendChild(el);
        trigger.setAttribute('aria-expanded', 'true');
        openPop = { el: el, trigger: trigger };
        place(el, trigger);
        return el;
    }

    // Under the trigger, kept inside the window; above it when there is no
    // room below (the sidebar foot).
    function place(el, trigger) {
        const box = trigger.getBoundingClientRect();
        const width = Math.min(360, window.innerWidth - 16);
        el.style.width = width + 'px';
        const left = Math.max(8, Math.min(box.right - width, window.innerWidth - width - 8));
        el.style.left = (box.left < width / 2 ? Math.max(8, box.left) : left) + 'px';
        const below = window.innerHeight - box.bottom;
        if (below < 260 && box.top > below) {
            el.style.bottom = (window.innerHeight - box.top + 6) + 'px';
            el.style.maxHeight = Math.max(160, box.top - 16) + 'px';
        } else {
            el.style.top = (box.bottom + 6) + 'px';
            el.style.maxHeight = Math.max(160, below - 16) + 'px';
        }
    }

    function node(tag, className, text) {
        const el = document.createElement(tag);
        if (className) el.className = className;
        if (text !== undefined) el.textContent = text;
        return el;
    }

    document.addEventListener('click', function(event) {
        if (openPop && !openPop.el.contains(event.target) && !event.target.closest('[data-inbox-open], [data-store-switch], [data-profile-open]')) closePop();
    });
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape' && openPop) {
            event.stopPropagation();
            const trigger = closePop();
            if (trigger) trigger.focus();
        }
    }, true);
    window.addEventListener('resize', closePop);

    // ---------- Inbox ----------
    const KIND_LABEL = { approval: 'Approval', review: 'Review', failure: 'Failure' };
    let inbox = null;

    function paintCount() {
        const total = inbox ? inbox.count : 0;
        document.querySelectorAll('[data-inbox-open]').forEach(function(button) {
            const badge = button.querySelector('.inbox-count');
            if (badge) {
                badge.hidden = total === 0;
                badge.textContent = total > 9 ? '9+' : String(total);
            }
            button.setAttribute('aria-label', total ? 'Waiting for you: ' + total + (total === 1 ? ' item' : ' items') : 'Waiting for you: nothing');
        });
    }

    function loadInbox() {
        return fetch('/api/inbox', { headers: { Accept: 'application/json' } })
            .then(function(response) { return response.ok ? response.json() : null; })
            .then(function(data) {
                if (data && Array.isArray(data.items)) inbox = data;
                paintCount();
                return inbox;
            })
            .catch(function() { return inbox; });
    }

    function drawInbox(el) {
        el.textContent = '';
        const head = node('div', 'shell-pop-head');
        head.appendChild(node('strong', '', 'Waiting for you'));
        if (inbox && inbox.count) head.appendChild(node('span', 'shell-pop-meta', String(inbox.count)));
        el.appendChild(head);
        if (!inbox) {
            el.appendChild(node('p', 'shell-pop-empty', 'Checking…'));
            return;
        }
        if (!inbox.items.length) {
            const empty = node('div', 'shell-pop-empty');
            empty.appendChild(node('strong', '', 'Nothing is waiting'));
            empty.appendChild(node('span', '', 'Tool calls and harness runs held for approval, plans to review and failed automations appear here.'));
            el.appendChild(empty);
            return;
        }
        const list = node('ul', 'inbox-list');
        inbox.items.forEach(function(item) {
            const row = node('li');
            const link = node('a', 'inbox-item inbox-item--' + item.kind);
            link.href = item.href;
            link.appendChild(node('span', 'inbox-kind', KIND_LABEL[item.kind] || 'Notice'));
            link.appendChild(node('strong', '', item.title));
            if (item.detail) link.appendChild(node('span', 'inbox-detail', item.detail));
            link.appendChild(node('span', 'inbox-action', (item.action || 'Open') + ' →'));
            row.appendChild(link);
            list.appendChild(row);
        });
        el.appendChild(list);
    }

    function initInbox() {
        const triggers = document.querySelectorAll('[data-inbox-open]');
        if (!triggers.length) return;
        triggers.forEach(function(trigger) {
            trigger.addEventListener('click', function() {
                const el = popover(trigger, 'Waiting for you');
                if (!el) return;
                drawInbox(el);
                loadInbox().then(function() { if (openPop && openPop.el === el) drawInbox(el); });
                el.focus();
            });
        });
        loadInbox();
        // A plan or an approval can arrive while a page sits open.
        setInterval(function() { if (!document.hidden) loadInbox(); }, 60000);
    }

    // ---------- Store switcher ----------
    function drawStores(el, data) {
        el.textContent = '';
        const head = node('div', 'shell-pop-head');
        head.appendChild(node('strong', '', 'Memory store'));
        el.appendChild(head);
        const stores = (data && data.stores) || [];
        if (!data) {
            el.appendChild(node('p', 'shell-pop-empty', 'Looking for stores on this computer…'));
        } else if (stores.length) {
            el.appendChild(node('p', 'shell-pop-label', 'FileSystem stores on this computer'));
            const list = node('div', 'store-list');
            stores.forEach(function(store) {
                const form = node('form');
                form.method = 'post';
                form.action = '/connect';
                [['provider_type', 'filesystem'], ['filesystem_path', store.path]].forEach(function(pair) {
                    const input = node('input');
                    input.type = 'hidden';
                    input.name = pair[0];
                    input.value = pair[1];
                    form.appendChild(input);
                });
                const button = node('button', 'store-option' + (store.current ? ' is-current' : ''));
                button.type = store.current ? 'button' : 'submit';
                if (store.current) button.setAttribute('aria-current', 'true');
                button.title = store.path;
                button.appendChild(node('span', 'store-option-path', store.display));
                const agents = store.agents + (store.agents === 1 ? ' agent' : ' agents');
                button.appendChild(node('span', 'store-option-meta', store.current ? agents + ' · connected' : agents));
                form.appendChild(button);
                form.addEventListener('submit', function() { button.disabled = true; button.classList.add('is-busy'); });
                list.appendChild(form);
            });
            el.appendChild(list);
        } else {
            el.appendChild(node('p', 'shell-pop-empty', 'No other FileSystem store was found on this computer.'));
        }
        const foot = node('div', 'shell-pop-foot');
        const link = node('a', '', 'Connect to another store or database');
        link.href = '/connect';
        foot.appendChild(link);
        el.appendChild(foot);
    }

    function initStores() {
        document.querySelectorAll('[data-store-switch]').forEach(function(trigger) {
            trigger.addEventListener('click', function(event) {
                // Ctrl/⌘-click on the chip still opens the connect page in a tab.
                if (event.metaKey || event.ctrlKey || event.shiftKey) return;
                event.preventDefault();
                const el = popover(trigger, 'Memory store');
                if (!el) return;
                drawStores(el, null);
                el.focus();
                fetch('/api/local-stores', { headers: { Accept: 'application/json' } })
                    .then(function(response) { return response.ok ? response.json() : { stores: [] }; })
                    .catch(function() { return { stores: [] }; })
                    .then(function(data) { if (openPop && openPop.el === el) drawStores(el, data); });
            });
        });
    }

    // ---------- Budget meter at the foot of the sidebar ----------
    function initBudget() {
        const meter = document.getElementById('sidebar-budget');
        if (!meter) return;
        fetch('/api/budget', { headers: { Accept: 'application/json' } })
            .then(function(response) { return response.ok ? response.json() : null; })
            .then(function(data) {
                if (!data || !data.set || data.state === 'unknown') return;
                const shown = Math.floor(data.percent);
                document.getElementById('sidebar-budget-figures').textContent = data.spend + ' of ' + data.budget;
                document.getElementById('sidebar-budget-fill').style.width = Math.min(100, Math.max(2, data.percent)) + '%';
                const bar = meter.querySelector('[role="progressbar"]');
                bar.setAttribute('aria-valuenow', String(Math.min(100, shown)));
                bar.setAttribute('aria-valuetext', shown + '% of the budget');
                meter.classList.add('is-' + data.state);
                meter.title = 'This month: ' + data.spend + ' of ' + data.budget + ' (' + shown + '%). Open usage and cost.';
                meter.hidden = false;
            })
            .catch(function() {});
    }

    // ---------- Your name and preferences ----------
    // The name signs approvals, reviews and undo forms, and the activity log.
    // It lives in this browser; the cookie lets the server put it in the log.
    const OPERATOR_KEY = 'memorizzApproverId';
    const APPROVER_FIELDS = 'input[name="approved_by"], input[name="approver_id"], input.harness-approver, input[aria-label="Approver identity"]';

    function operator() {
        try { return (localStorage.getItem(OPERATOR_KEY) || '').trim(); } catch (_) { return ''; }
    }

    function setOperator(name) {
        const clean = name.replace(/[\u0000-\u001f\u007f]/g, '').trim().slice(0, 80);
        try { if (clean) localStorage.setItem(OPERATOR_KEY, clean); else localStorage.removeItem(OPERATOR_KEY); } catch (_) {}
        syncOperatorCookie();
        paintOperator();
        return clean;
    }

    function syncOperatorCookie() {
        const name = operator();
        document.cookie = 'memorizz_operator=' + encodeURIComponent(name) + '; path=/; SameSite=Lax; max-age=' + (name ? 31536000 : 0);
    }

    function initials(name) {
        const parts = name.split(/[\s@._-]+/).filter(Boolean);
        return ((parts[0] || '')[0] || '').toUpperCase() + (parts.length > 1 ? parts[1][0].toUpperCase() : '');
    }

    function paintOperator() {
        const name = operator();
        document.querySelectorAll('[data-profile-open]').forEach(function(button) {
            const badge = button.querySelector('[data-profile-initials]');
            const glyph = button.querySelector('.profile-glyph');
            if (badge) { badge.hidden = !name; badge.textContent = initials(name); }
            if (glyph) glyph.style.display = name ? 'none' : '';
            button.title = name ? name + ' · your name and preferences' : 'Set your name and preferences';
        });
    }

    function fillApprover(input) {
        const name = operator();
        if (name && input && !input.value && !input.readOnly && !input.disabled) input.value = name;
    }

    function drawProfile(el) {
        el.textContent = '';
        const head = node('div', 'shell-pop-head');
        head.appendChild(node('strong', '', 'You'));
        el.appendChild(head);
        const form = node('form', 'profile-form');
        const label = node('label', '', 'Your name or email');
        const input = node('input');
        input.type = 'text';
        input.maxLength = 80;
        input.autocomplete = 'name';
        input.placeholder = 'Ada Lovelace';
        input.value = operator();
        label.appendChild(input);
        form.appendChild(label);
        form.appendChild(node('p', 'profile-note', 'Signs your approvals, reviews and undo actions, and the activity log. Kept in this browser.'));
        const save = node('button', 'btn btn-sm btn-primary', 'Save');
        save.type = 'submit';
        form.appendChild(save);
        form.addEventListener('submit', function(event) {
            event.preventDefault();
            setOperator(input.value);
            document.querySelectorAll(APPROVER_FIELDS).forEach(fillApprover);
            save.textContent = 'Saved';
            setTimeout(function() { save.textContent = 'Save'; }, 1200);
        });
        el.appendChild(form);

        const prefs = node('div', 'profile-prefs');
        const toggle = node('label', 'profile-switch');
        const box = node('input');
        box.type = 'checkbox';
        box.checked = document.documentElement.classList.contains('density-compact');
        box.addEventListener('change', function() {
            document.documentElement.classList.toggle('density-compact', box.checked);
            try { localStorage.setItem('memorizz_ui_density', box.checked ? 'compact' : 'comfortable'); } catch (_) {}
        });
        toggle.appendChild(box);
        const text = node('span');
        text.appendChild(node('strong', '', 'Compact tables'));
        text.appendChild(node('span', '', 'More rows on screen, on every table.'));
        toggle.appendChild(text);
        prefs.appendChild(toggle);
        el.appendChild(prefs);

        const foot = node('div', 'shell-pop-foot');
        const link = node('a', '', 'See the activity log');
        link.href = '/activity';
        foot.appendChild(link);
        el.appendChild(foot);
        return input;
    }

    function initProfile() {
        syncOperatorCookie();
        paintOperator();
        document.querySelectorAll('[data-profile-open]').forEach(function(trigger) {
            trigger.addEventListener('click', function() {
                const el = popover(trigger, 'Your name and preferences');
                if (el) drawProfile(el).focus();
            });
        });
        document.querySelectorAll(APPROVER_FIELDS).forEach(fillApprover);
        // Forms drawn later (dialogs, rows loaded by script) fill on focus.
        document.addEventListener('focusin', function(event) {
            if (event.target.matches && event.target.matches(APPROVER_FIELDS)) fillApprover(event.target);
        });
        // A name typed into any approval form becomes the saved one.
        document.addEventListener('change', function(event) {
            if (event.target.matches && event.target.matches(APPROVER_FIELDS) && event.target.value.trim() && !operator()) setOperator(event.target.value);
        });
    }

    // The Overview's first-steps list ticks "open a trace" from this.
    if (window.location.pathname.indexOf('/traces') === 0) {
        try { localStorage.setItem('memorizz_ui_seen_trace', '1'); } catch (_) {}
    }

    function start() { initInbox(); initStores(); initProfile(); initBudget(); }
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
    else start();
})();

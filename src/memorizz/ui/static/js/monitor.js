/* Monitor grid: the shared behaviour behind the fleet, memory and other
   console list pages. Markup contract (see console.css "Fleet monitor"):

   <table class="fleet-table" id="...">           rows: <tr data-key data-search
                                                    [data-tags="a b"] [data-<field>]>
   <section class="fleet-detail" data-key hidden>  one per row, in the pane
   <input data-monitor-filter-for="<table id>">    free-text filter
   <button data-monitor-tag="<tag>|''">            chip group, matches data-tags
   <select data-monitor-field="<field>">           matches data-<field> exactly
   <button data-sort="<field>" [data-sort-type="text"]> in a <th aria-sort>

   MemorizzMonitor.init({ table, pane, storageKey, keys: { e: row => ... } })
   Keys: j/k or arrows move, / filters, Enter/other keys call keys[key](row).

   Every grid also gets, with no markup of its own:
   - chips under the bar for each filter in use, each with a remove button;
   - a Views menu in the bar that saves the filters, the sort and the page's
     query string under a name, in this browser (memorizz_ui_views:<path>).

   MemorizzMonitor.bulk(table, { noun, actions: [{ label, run(keys, rows),
   danger, enabled(keys) }] }) adds a checkbox column and an action bar for
   the checked rows. It works on any table whose rows carry data-key. */
(function() {
    function init(options) {
        const table = typeof options.table === 'string' ? document.getElementById(options.table) : options.table;
        if (!table || !table.tBodies[0]) return null;
        const tbody = table.tBodies[0];
        const scope = options.scope || table.closest('.fleet') || document;
        const pane = options.pane || scope.querySelector('.fleet-pane');
        const filterInput = document.querySelector('[data-monitor-filter-for="' + table.id + '"]');
        const none = options.empty || scope.querySelector('.fleet-none');
        const stacked = window.matchMedia(options.stackedQuery || '(max-width: 1100px)');
        const keys = options.keys || {};
        let tag = '';
        let selected = null;

        const rows = () => Array.from(tbody.rows).filter(function(row) { return row.dataset.key !== undefined; });
        const visible = () => rows().filter(function(row) { return !row.hidden; });
        const store = {
            get() {
                if (!options.storageKey) return null;
                try { return localStorage.getItem(options.storageKey); } catch (_) { return null; }
            },
            set(value) {
                if (!options.storageKey) return;
                try { localStorage.setItem(options.storageKey, value); } catch (_) {}
            },
        };

        function select(row, focus) {
            if (!row) return;
            rows().forEach(function(other) {
                const active = other === row;
                other.classList.toggle('is-selected', active);
                other.tabIndex = active ? 0 : -1;
                if (active) other.setAttribute('aria-current', 'true');
                else other.removeAttribute('aria-current');
            });
            if (pane) {
                pane.querySelectorAll('.fleet-detail').forEach(function(detail) {
                    detail.hidden = detail.dataset.key !== row.dataset.key;
                });
            }
            selected = row;
            store.set(row.dataset.key);
            if (typeof options.onSelect === 'function') options.onSelect(row);
            if (focus) {
                row.focus({ preventScroll: true });
                row.scrollIntoView({ block: 'nearest' });
            }
        }

        function matches(row) {
            const query = (filterInput && filterInput.value || '').trim().toLowerCase();
            if (query && !(row.dataset.search || '').includes(query)) return false;
            if (tag && !(' ' + (row.dataset.tags || '') + ' ').includes(' ' + tag + ' ')) return false;
            const fields = document.querySelectorAll('[data-monitor-field]');
            for (const field of fields) {
                if (field.value && row.dataset[field.dataset.monitorField] !== field.value) return false;
            }
            return true;
        }

        function applyFilter() {
            rows().forEach(function(row) { row.hidden = !matches(row); });
            const shown = visible();
            if (none) none.hidden = shown.length > 0;
            if (shown.length && (!selected || selected.hidden)) select(shown[0]);
            renderChips();
        }

        // ---- Filters in use, as chips, and named views ----
        const bar = (filterInput && filterInput.closest('.fleet-bar'))
            || (scope !== document && scope.previousElementSibling && scope.previousElementSibling.matches('.fleet-bar') ? scope.previousElementSibling : null);
        const tagButtons = () => Array.from(document.querySelectorAll('[data-monitor-tag]'));
        const fieldInputs = () => Array.from(document.querySelectorAll('[data-monitor-field]'));
        let chips = null;

        function setTag(value) {
            const buttons = tagButtons();
            const target = buttons.find(function(button) { return button.dataset.monitorTag === value; });
            tag = target ? value : '';
            buttons.forEach(function(button) {
                button.setAttribute('aria-pressed', button.dataset.monitorTag === tag ? 'true' : 'false');
            });
        }

        function labelOf(field) {
            const label = field.id && document.querySelector('label[for="' + field.id + '"]');
            return (label ? label.textContent : field.getAttribute('aria-label') || field.dataset.monitorField).trim();
        }

        function activeFilters() {
            const active = [];
            const query = (filterInput && filterInput.value || '').trim();
            if (query) active.push({ name: 'Search', value: query, clear: function() { filterInput.value = ''; } });
            if (tag) {
                const button = tagButtons().find(function(item) { return item.dataset.monitorTag === tag; });
                const text = button ? (button.firstChild && button.firstChild.nodeType === 3 ? button.firstChild.textContent : button.textContent).trim() : tag;
                active.push({ name: 'Show', value: text, clear: function() { setTag(''); } });
            }
            fieldInputs().forEach(function(field) {
                if (!field.value) return;
                const option = field.options ? field.options[field.selectedIndex] : null;
                active.push({ name: labelOf(field), value: option ? option.textContent.trim() : field.value, clear: function() { field.value = ''; } });
            });
            return active;
        }

        function renderChips() {
            if (!bar) return;
            const active = activeFilters();
            if (!chips) {
                chips = document.createElement('div');
                chips.className = 'fleet-chips';
                chips.setAttribute('role', 'group');
                chips.setAttribute('aria-label', 'Filters in use');
                bar.insertAdjacentElement('afterend', chips);
            }
            chips.hidden = !active.length;
            chips.textContent = '';
            if (!active.length) return;
            const shown = visible().length;
            const count = document.createElement('span');
            count.className = 'fleet-chips-count';
            count.textContent = shown + ' of ' + rows().length + ' shown';
            chips.appendChild(count);
            active.forEach(function(filter) {
                const chip = document.createElement('span');
                chip.className = 'fleet-chip';
                const name = document.createElement('span');
                name.className = 'fleet-chip-name';
                name.textContent = filter.name;
                const value = document.createElement('strong');
                value.textContent = filter.value;
                const remove = document.createElement('button');
                remove.type = 'button';
                remove.setAttribute('aria-label', 'Remove filter ' + filter.name + ': ' + filter.value);
                remove.textContent = '×';
                remove.addEventListener('click', function() { filter.clear(); applyFilter(); });
                chip.append(name, value, remove);
                chips.appendChild(chip);
            });
            if (active.length > 1) {
                const clear = document.createElement('button');
                clear.type = 'button';
                clear.className = 'fleet-chips-clear';
                clear.textContent = 'Clear all';
                clear.addEventListener('click', function() {
                    active.forEach(function(filter) { filter.clear(); });
                    applyFilter();
                });
                chips.appendChild(clear);
            }
        }

        function currentSort() {
            const th = table.querySelector('thead th[aria-sort="ascending"], thead th[aria-sort="descending"]');
            const button = th && th.querySelector('button[data-sort]');
            return button ? { key: button.dataset.sort, dir: th.getAttribute('aria-sort') } : null;
        }

        function readState() {
            const fields = {};
            fieldInputs().forEach(function(field) { if (field.value) fields[field.dataset.monitorField] = field.value; });
            return { q: (filterInput && filterInput.value || '').trim(), tag: tag, fields: fields, sort: currentSort() };
        }

        function writeState(state) {
            if (filterInput) filterInput.value = state.q || '';
            setTag(state.tag || '');
            fieldInputs().forEach(function(field) {
                const wanted = (state.fields || {})[field.dataset.monitorField] || '';
                field.value = wanted;
                if (field.value !== wanted) field.value = '';
            });
            if (state.sort && state.sort.key) {
                const button = table.querySelector('thead button[data-sort="' + state.sort.key + '"]');
                // A click toggles: one reaches a direction, a second the other.
                for (let turn = 0; button && turn < 2; turn++) {
                    if (button.closest('th').getAttribute('aria-sort') === state.sort.dir) break;
                    button.click();
                }
            }
            applyFilter();
        }

        function move(step) {
            const shown = visible();
            if (!shown.length) return;
            const index = shown.indexOf(selected);
            select(shown[Math.max(0, Math.min(shown.length - 1, index + step))], true);
        }

        // Column sort: numbers descend first, text ascends first.
        table.querySelectorAll('thead button[data-sort]').forEach(function(button) {
            button.addEventListener('click', function() {
                const key = button.dataset.sort;
                const th = button.closest('th');
                const text = button.dataset.sortType === 'text';
                const current = th.getAttribute('aria-sort');
                const ascending = current === 'none' || !current ? text : current === 'descending';
                table.querySelectorAll('thead th[aria-sort]').forEach(function(other) {
                    other.setAttribute('aria-sort', 'none');
                });
                th.setAttribute('aria-sort', ascending ? 'ascending' : 'descending');
                const value = function(row) {
                    return text ? String(row.dataset[key] || '') : Number(row.dataset[key] || 0);
                };
                rows().sort(function(a, b) {
                    const order = text ? value(a).localeCompare(value(b)) : value(a) - value(b);
                    return ascending ? order : -order;
                }).forEach(function(row) { tbody.appendChild(row); });
            });
        });

        document.querySelectorAll('[data-monitor-tag]').forEach(function(button) {
            button.addEventListener('click', function() {
                setTag(button.dataset.monitorTag);
                applyFilter();
            });
        });
        document.querySelectorAll('[data-monitor-field]').forEach(function(field) {
            field.addEventListener('change', applyFilter);
        });
        if (filterInput) filterInput.addEventListener('input', applyFilter);

        tbody.addEventListener('click', function(event) {
            if (event.target.closest('a, button, form, input, select, textarea')) return;
            const row = event.target.closest('tr');
            if (!row || row.dataset.key === undefined) return;
            select(row, true);
            // When the pane sits under the grid (phones, tablets), bring it into view.
            if (stacked.matches && pane) {
                const motion = window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth';
                pane.scrollIntoView({ behavior: motion, block: 'start' });
            }
        });
        if (typeof keys.Enter === 'function') {
            tbody.addEventListener('dblclick', function(event) {
                const row = event.target.closest('tr');
                if (row && !event.target.closest('a, button, form, input, select')) keys.Enter(row);
            });
        }

        document.addEventListener('keydown', function(event) {
            if (event.metaKey || event.ctrlKey || event.altKey) return;
            const target = event.target;
            if (target.closest && target.closest('input, textarea, select, [contenteditable="true"]')) {
                if (event.key === 'Escape' && target === filterInput) filterInput.blur();
                if (event.key === 'Enter' && target === filterInput) {
                    event.preventDefault();
                    const first = visible()[0];
                    if (first) select(first, true);
                }
                return;
            }
            if (event.key === '/' && filterInput) {
                event.preventDefault();
                filterInput.focus();
                return;
            }
            if (!selected) return;
            // Links and buttons keep their own Enter behaviour.
            if (event.key === 'Enter' && target.closest && target.closest('a, button')) return;
            if (event.key === 'j' || event.key === 'ArrowDown') {
                event.preventDefault();
                move(1);
            } else if (event.key === 'k' || event.key === 'ArrowUp') {
                event.preventDefault();
                move(-1);
            } else if (typeof keys[event.key] === 'function') {
                keys[event.key](selected, event);
            }
        });

        const remembered = rows().find(function(row) { return row.dataset.key === store.get(); });
        select(remembered || rows()[0]);
        initViews({ bar: bar, read: readState, write: writeState });
        initExport(bar, table, visible);
        return { select: select, applyFilter: applyFilter, selected: function() { return selected; }, state: readState, setState: writeState };
    }

    // ---- Named views: filters, sort and the page's query, kept in this browser ----
    const VIEWS_KEY = 'memorizz_ui_views:' + window.location.pathname;
    const PENDING_KEY = 'memorizz_ui_view_pending';

    function loadViews() {
        try {
            const views = JSON.parse(localStorage.getItem(VIEWS_KEY) || '[]');
            return Array.isArray(views) ? views.filter(function(view) { return view && typeof view.name === 'string'; }) : [];
        } catch (_) { return []; }
    }

    function saveViews(views) {
        try { localStorage.setItem(VIEWS_KEY, JSON.stringify(views.slice(0, 20))); return true; } catch (_) { return false; }
    }

    function initViews(grid) {
        if (!grid.bar || grid.bar.querySelector('.fleet-views')) return;
        const wrap = document.createElement('div');
        wrap.className = 'fleet-views';
        const toggle = document.createElement('button');
        toggle.type = 'button';
        toggle.className = 'btn btn-sm fleet-views-toggle';
        toggle.setAttribute('aria-haspopup', 'true');
        toggle.setAttribute('aria-expanded', 'false');
        toggle.textContent = 'Views';
        const menu = document.createElement('div');
        menu.className = 'fleet-views-menu';
        menu.hidden = true;
        wrap.append(toggle, menu);
        const keys = grid.bar.querySelector('.fleet-keys');
        if (keys) grid.bar.insertBefore(wrap, keys); else grid.bar.appendChild(wrap);

        function close() { menu.hidden = true; toggle.setAttribute('aria-expanded', 'false'); }

        function apply(view) {
            close();
            if ((view.search || '') !== window.location.search) {
                // The view was saved under another time range or scope: open
                // that page, then apply the filters once it has loaded.
                try { sessionStorage.setItem(PENDING_KEY, JSON.stringify({ path: window.location.pathname, state: view.state })); } catch (_) {}
                window.location.href = window.location.pathname + (view.search || '');
                return;
            }
            grid.write(view.state || {});
        }

        function render() {
            const views = loadViews();
            menu.textContent = '';
            toggle.textContent = views.length ? 'Views · ' + views.length : 'Views';
            if (!views.length) {
                const empty = document.createElement('p');
                empty.className = 'fleet-views-empty';
                empty.textContent = 'No saved views yet. Set the filters you use often, then save them here.';
                menu.appendChild(empty);
            }
            views.forEach(function(view, index) {
                const row = document.createElement('div');
                row.className = 'fleet-views-row';
                const open = document.createElement('button');
                open.type = 'button';
                open.className = 'fleet-views-open';
                open.textContent = view.name;
                open.addEventListener('click', function() { apply(view); });
                const remove = document.createElement('button');
                remove.type = 'button';
                remove.className = 'fleet-views-remove';
                remove.setAttribute('aria-label', 'Delete the view ' + view.name);
                remove.textContent = '×';
                remove.addEventListener('click', function() {
                    const next = loadViews();
                    next.splice(index, 1);
                    saveViews(next);
                    render();
                });
                row.append(open, remove);
                menu.appendChild(row);
            });
            const save = document.createElement('button');
            save.type = 'button';
            save.className = 'fleet-views-save';
            save.textContent = 'Save current view…';
            save.addEventListener('click', async function() {
                close();
                if (!window.MemorizzDialogs) return;
                const name = await MemorizzDialogs.prompt('Save this view', {
                    detail: 'Keeps the filters, the sort order and the time range of this page, in this browser.',
                    label: 'Name',
                    placeholder: 'Failed runs this week',
                    required: true,
                    confirmLabel: 'Save view',
                });
                const trimmed = typeof name === 'string' ? name.trim().slice(0, 60) : '';
                if (!trimmed) return;
                const next = loadViews().filter(function(view) { return view.name !== trimmed; });
                next.unshift({ name: trimmed, search: window.location.search, state: grid.read() });
                if (!saveViews(next) && MemorizzDialogs.notify) MemorizzDialogs.notify('This browser would not store the view.');
                render();
            });
            menu.appendChild(save);
        }

        toggle.addEventListener('click', function() {
            const open = menu.hidden;
            if (open) render();
            menu.hidden = !open;
            toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
        });
        document.addEventListener('click', function(event) { if (!wrap.contains(event.target)) close(); });
        wrap.addEventListener('keydown', function(event) {
            if (event.key === 'Escape' && !menu.hidden) { event.stopPropagation(); close(); toggle.focus(); }
        });
        render();

        try {
            const pending = JSON.parse(sessionStorage.getItem(PENDING_KEY) || 'null');
            if (pending && pending.path === window.location.pathname) {
                sessionStorage.removeItem(PENDING_KEY);
                grid.write(pending.state || {});
            }
        } catch (_) {}
    }

    // ---- The rows on screen as a CSV file ----
    function cellText(cell) {
        const clone = cell.cloneNode(true);
        clone.querySelectorAll('.visually-hidden, .row-menu, input, svg, .ov-share, form').forEach(function(el) { el.remove(); });
        return clone.textContent.replace(/\s+/g, ' ').trim();
    }

    function toCsv(table, rows) {
        const skip = [];
        const head = Array.from(table.tHead ? table.tHead.rows[0].cells : []).map(function(cell, index) {
            const text = cellText(cell);
            if (cell.classList.contains('bulk-col') || cell.classList.contains('row-menu-col') || !text) skip.push(index);
            return text;
        });
        const quote = function(value) {
            // A leading =, +, - or @ would run as a formula in a spreadsheet.
            const safe = /^[=+\-@]/.test(value) ? "'" + value : value;
            return '"' + safe.replace(/"/g, '""') + '"';
        };
        const lines = [head.filter(function(_, index) { return !skip.includes(index); }).map(quote).join(',')];
        rows.forEach(function(row) {
            lines.push(Array.from(row.cells).filter(function(_, index) { return !skip.includes(index); }).map(function(cell) { return quote(cellText(cell)); }).join(','));
        });
        return lines.join('\r\n') + '\r\n';
    }

    function initExport(bar, table, visible) {
        if (!bar || bar.querySelector('.fleet-export')) return;
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'btn btn-sm fleet-export';
        button.textContent = 'Export CSV';
        button.title = 'Download the rows shown, with the filters applied';
        button.addEventListener('click', function() {
            const name = (window.location.pathname.replace(/^\/+|\/+$/g, '').replace(/[^a-z0-9]+/gi, '-') || 'table') + '.csv';
            download(name, toCsv(table, visible()), 'text/csv;charset=utf-8');
        });
        const views = bar.querySelector('.fleet-views');
        if (views) views.insertAdjacentElement('afterend', button); else bar.appendChild(button);
    }

    // ---- A "more" menu at the end of each row ----
    // MemorizzMonitor.rowMenu(table, function(row) { return [{ label, href |
    // run(row), danger, mutates }]; })
    let openMenu = null;
    function closeRowMenu() {
        if (!openMenu) return;
        openMenu.list.remove();
        openMenu.button.setAttribute('aria-expanded', 'false');
        openMenu = null;
    }
    document.addEventListener('click', function(event) {
        if (openMenu && !openMenu.list.contains(event.target) && event.target !== openMenu.button) closeRowMenu();
    });
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape' && openMenu) { const button = openMenu.button; closeRowMenu(); button.focus(); }
    });
    window.addEventListener('scroll', closeRowMenu, true);

    function rowMenu(tableOrId, itemsFor) {
        const table = typeof tableOrId === 'string' ? document.getElementById(tableOrId) : tableOrId;
        if (!table || !table.tBodies[0] || table.dataset.rowMenuReady) return;
        table.dataset.rowMenuReady = '1';
        const headRow = table.tHead && table.tHead.rows[0];
        if (headRow) {
            const th = document.createElement('th');
            th.scope = 'col';
            th.className = 'row-menu-col';
            th.innerHTML = '<span class="visually-hidden">Actions</span>';
            headRow.appendChild(th);
        }
        Array.from(table.tBodies[0].rows).forEach(function(row) {
            const td = document.createElement('td');
            td.className = 'row-menu-col';
            if (row.dataset.key !== undefined) {
                const button = document.createElement('button');
                button.type = 'button';
                button.className = 'row-menu';
                button.setAttribute('aria-haspopup', 'menu');
                button.setAttribute('aria-expanded', 'false');
                button.setAttribute('aria-label', 'Actions for this row');
                button.textContent = '⋮';
                button.addEventListener('click', function(event) {
                    event.stopPropagation();
                    const same = openMenu && openMenu.button === button;
                    closeRowMenu();
                    if (same) return;
                    const list = document.createElement('div');
                    list.className = 'row-menu-list';
                    list.setAttribute('role', 'menu');
                    itemsFor(row).filter(Boolean).forEach(function(item) {
                        const el = document.createElement(item.href ? 'a' : 'button');
                        el.setAttribute('role', 'menuitem');
                        el.textContent = item.label;
                        if (item.danger) el.classList.add('is-danger');
                        if (item.mutates) el.setAttribute('data-mutates', '');
                        if (item.href) el.href = item.href;
                        else { el.type = 'button'; el.addEventListener('click', function() { closeRowMenu(); item.run(row); }); }
                        list.appendChild(el);
                    });
                    document.body.appendChild(list);
                    const box = button.getBoundingClientRect();
                    const height = list.offsetHeight;
                    list.style.left = Math.max(8, box.right - list.offsetWidth) + 'px';
                    list.style.top = (box.bottom + height + 8 > window.innerHeight ? Math.max(8, box.top - height - 4) : box.bottom + 4) + 'px';
                    button.setAttribute('aria-expanded', 'true');
                    openMenu = { list: list, button: button };
                    const first = list.querySelector('a, button');
                    if (first) first.focus();
                });
                td.appendChild(button);
            }
            row.appendChild(td);
        });
    }

    // ---- Several rows at once: a checkbox column and an action bar ----
    function bulk(tableOrId, options) {
        const table = typeof tableOrId === 'string' ? document.getElementById(tableOrId) : tableOrId;
        if (!table || !table.tBodies[0] || table.dataset.bulkReady) return null;
        const actions = (options.actions || []).filter(Boolean);
        if (!actions.length) return null;
        table.dataset.bulkReady = '1';
        const noun = options.noun || 'row';
        const tbody = table.tBodies[0];
        const rows = () => Array.from(tbody.rows).filter(function(row) { return row.dataset.key !== undefined; });
        const box = (row) => row.querySelector('input.bulk-check');
        const checked = () => rows().filter(function(row) { const input = box(row); return input && input.checked; });

        function checkbox(label) {
            const input = document.createElement('input');
            input.type = 'checkbox';
            input.className = 'bulk-check';
            input.setAttribute('aria-label', label);
            return input;
        }

        const all = checkbox('Select every ' + noun + ' shown');
        const headRow = table.tHead && table.tHead.rows[0];
        if (headRow) {
            const th = document.createElement('th');
            th.scope = 'col';
            th.className = 'bulk-col';
            th.appendChild(all);
            headRow.insertBefore(th, headRow.firstChild);
        }
        Array.from(tbody.rows).forEach(function(row) {
            const td = document.createElement('td');
            td.className = 'bulk-col';
            if (row.dataset.key !== undefined) td.appendChild(checkbox('Select this ' + noun));
            row.insertBefore(td, row.firstChild);
        });

        const barEl = document.createElement('div');
        barEl.className = 'bulk-bar';
        barEl.setAttribute('role', 'region');
        barEl.setAttribute('aria-label', 'Actions for the selected ' + noun + 's');
        barEl.hidden = true;
        const count = document.createElement('span');
        count.className = 'bulk-bar-count';
        count.setAttribute('aria-live', 'polite');
        barEl.appendChild(count);
        const hint = document.createElement('span');
        hint.className = 'bulk-bar-hint';
        const buttons = actions.map(function(action) {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'btn btn-sm' + (action.danger ? ' btn-danger' : '');
            button.textContent = action.label;
            if (action.mutates) button.setAttribute('data-mutates', '');
            button.addEventListener('click', function() {
                const picked = checked();
                if (picked.length) action.run(picked.map(function(row) { return row.dataset.key; }), picked, api);
            });
            barEl.appendChild(button);
            return button;
        });
        barEl.appendChild(hint);
        const clear = document.createElement('button');
        clear.type = 'button';
        clear.className = 'bulk-bar-clear';
        clear.setAttribute('aria-label', 'Clear the selection');
        clear.textContent = '×';
        barEl.appendChild(clear);
        document.body.appendChild(barEl);

        function refresh() {
            const picked = checked();
            const shown = rows().filter(function(row) { return !row.hidden; });
            rows().forEach(function(row) { row.classList.toggle('is-checked', picked.includes(row)); });
            all.checked = shown.length > 0 && shown.every(function(row) { return picked.includes(row); });
            all.indeterminate = picked.length > 0 && !all.checked;
            barEl.hidden = picked.length === 0;
            count.textContent = picked.length + ' ' + noun + (picked.length === 1 ? '' : 's') + ' selected';
            const keys = picked.map(function(row) { return row.dataset.key; });
            let why = '';
            actions.forEach(function(action, index) {
                const reason = typeof action.enabled === 'function' ? action.enabled(keys, picked) : true;
                buttons[index].disabled = reason !== true;
                if (typeof reason === 'string') { buttons[index].title = reason; why = why || reason; } else buttons[index].removeAttribute('title');
            });
            // Say why an action is off instead of leaving a dead button.
            hint.textContent = why;
            hint.hidden = !why;
        }

        function clearAll() {
            rows().forEach(function(row) { const input = box(row); if (input) input.checked = false; });
            refresh();
        }

        const api = { clear: clearAll, refresh: refresh, checked: checked };
        all.addEventListener('change', function() {
            rows().forEach(function(row) { const input = box(row); if (input && !row.hidden) input.checked = all.checked; });
            refresh();
        });
        tbody.addEventListener('change', function(event) {
            if (event.target.classList && event.target.classList.contains('bulk-check')) refresh();
        });
        clear.addEventListener('click', clearAll);
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape' && !barEl.hidden && !document.querySelector('dialog[open]')) clearAll();
        });
        return api;
    }

    function download(name, text, type) {
        const link = document.createElement('a');
        link.href = URL.createObjectURL(new Blob([text], { type: type || 'application/json' }));
        link.download = name;
        document.body.appendChild(link);
        link.click();
        link.remove();
        setTimeout(function() { URL.revokeObjectURL(link.href); }, 1000);
    }

    // Copy buttons: <button data-copy-target="element id">Copy</button>.
    document.addEventListener('click', function(event) {
        const button = event.target.closest('[data-copy-target]');
        if (!button || !navigator.clipboard) return;
        const source = document.getElementById(button.dataset.copyTarget);
        if (!source) return;
        const label = button.textContent;
        navigator.clipboard.writeText(source.textContent).then(function() {
            button.textContent = 'Copied';
            setTimeout(function() { button.textContent = label; }, 1200);
        });
    });

    // ---- A record in a side drawer, over the page it came from ----
    // MemorizzDrawer.open({ title, subtitle, status: Node, facts: [[label,
    // value]], body: Node, actions: [{ label, href | run, primary }], returnTo })
    let drawer = null;
    function openDrawer(options) {
        if (!drawer) {
            drawer = document.createElement('dialog');
            drawer.className = 'mz-drawer';
            drawer.addEventListener('click', function(event) { if (event.target === drawer) drawer.close(); });
            document.body.appendChild(drawer);
        }
        drawer.textContent = '';
        drawer.setAttribute('aria-label', options.title || 'Details');
        const head = document.createElement('header');
        head.className = 'mz-drawer-head';
        const titles = document.createElement('div');
        const title = document.createElement('h2');
        title.textContent = options.title || 'Details';
        titles.appendChild(title);
        if (options.subtitle) {
            const sub = document.createElement('p');
            sub.textContent = options.subtitle;
            titles.appendChild(sub);
        }
        head.appendChild(titles);
        if (options.status) head.appendChild(options.status);
        const close = document.createElement('button');
        close.type = 'button';
        close.className = 'mz-drawer-close';
        close.setAttribute('aria-label', 'Close');
        close.textContent = '×';
        close.addEventListener('click', function() { drawer.close(); });
        head.appendChild(close);
        drawer.appendChild(head);
        const body = document.createElement('div');
        body.className = 'mz-drawer-body';
        const actions = (options.actions || []).filter(Boolean);
        if (actions.length) {
            const row = document.createElement('div');
            row.className = 'mz-drawer-actions';
            actions.forEach(function(action) {
                const el = document.createElement(action.href ? 'a' : 'button');
                el.className = 'btn btn-sm' + (action.primary ? ' btn-primary' : '');
                el.textContent = action.label;
                if (action.href) el.href = action.href;
                else { el.type = 'button'; el.addEventListener('click', function() { action.run(drawer); }); }
                row.appendChild(el);
            });
            body.appendChild(row);
        }
        if (options.facts && options.facts.length) {
            const list = document.createElement('dl');
            list.className = 'mz-drawer-facts';
            options.facts.forEach(function(fact) {
                const item = document.createElement('div');
                const dt = document.createElement('dt');
                dt.textContent = fact[0];
                const dd = document.createElement('dd');
                dd.textContent = fact[1];
                item.append(dt, dd);
                list.appendChild(item);
            });
            body.appendChild(list);
        }
        if (options.body) body.appendChild(options.body);
        drawer.appendChild(body);
        const back = options.returnTo;
        drawer.addEventListener('close', function restore() {
            drawer.removeEventListener('close', restore);
            if (back && back.focus) back.focus({ preventScroll: true });
        });
        if (!drawer.open) drawer.showModal();
        close.focus();
        return drawer;
    }
    window.MemorizzDrawer = { open: openDrawer, close: function() { if (drawer && drawer.open) drawer.close(); } };

    window.MemorizzMonitor = { init: init, bulk: bulk, rowMenu: rowMenu, download: download };
})();

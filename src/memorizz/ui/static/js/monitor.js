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
   Keys: j/k or arrows move, / filters, Enter/other keys call keys[key](row). */
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
                tag = button.dataset.monitorTag;
                document.querySelectorAll('[data-monitor-tag]').forEach(function(other) {
                    other.setAttribute('aria-pressed', other === button ? 'true' : 'false');
                });
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
        return { select: select, applyFilter: applyFilter, selected: function() { return selected; } };
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

    window.MemorizzMonitor = { init: init };
})();

/* The Observability waterfall: folding, lane filters, search, zoom, the
   detail panel and keyboard movement. Rows, bars and the first axis are
   rendered by the server (trace_waterfall.html); this only rearranges them. */
(() => {
    'use strict';
    const section = document.getElementById('trace-waterfall');
    if (!section) return;

    // ---------- Formatting, the same rules as ui/waterfall_view.py ----------
    function duration(ms) {
        if (ms === null || ms === undefined || Number.isNaN(Number(ms))) return '';
        ms = Number(ms);
        if (ms < 1) return ms.toFixed(2) + ' ms';
        if (ms < 100) return ms.toFixed(1) + ' ms';
        if (ms < 1000) return Math.round(ms) + ' ms';
        let seconds = ms / 1000;
        if (seconds < 10) return seconds.toFixed(2) + ' s';
        if (seconds < 60) return seconds.toFixed(1) + ' s';
        seconds = Math.round(seconds);
        let minutes = Math.floor(seconds / 60);
        if (minutes < 60) return `${minutes}m ${String(seconds % 60).padStart(2, '0')}s`;
        let hours = Math.floor(minutes / 60);
        minutes %= 60;
        if (hours < 24) return `${hours}h ${String(minutes).padStart(2, '0')}m`;
        return `${Math.floor(hours / 24)}d ${hours % 24}h`;
    }
    function tickStep(span) {
        const raw = Math.max(span, 1e-6) / 6;
        const magnitude = 10 ** Math.floor(Math.log10(raw));
        for (const multiple of [1, 2, 2.5, 5, 10]) if (multiple * magnitude >= raw) return multiple * magnitude;
        return 10 * magnitude;
    }
    function tickLabel(ms, step) {
        if (Math.abs(ms) < 1e-9) return '0';
        if (step < 1) return ms.toFixed(Math.min(3, Math.max(0, -Math.floor(Math.log10(step))))) + ' ms';
        if (step < 1000) return Math.round(ms) + ' ms';
        if (step < 60000) {
            const seconds = ms / 1000;
            return (Math.abs(seconds - Math.round(seconds * 10) / 10) < 1e-9 ? String(Math.round(seconds * 10) / 10) : seconds.toFixed(1)) + ' s';
        }
        return duration(ms);
    }
    const count = new Intl.NumberFormat();
    const clock = new Intl.DateTimeFormat(undefined, {hour: '2-digit', minute: '2-digit', second: '2-digit'});
    const fullTime = new Intl.DateTimeFormat(undefined, {dateStyle: 'medium', timeStyle: 'medium'});
    section.querySelectorAll('time[data-wf-clock]').forEach(node => {
        const date = new Date(node.getAttribute('datetime'));
        if (Number.isNaN(date.getTime())) return;
        node.textContent = clock.format(date);
        node.title = fullTime.format(date);
    });

    // ---------- The split between names and bars, shared by every turn ----------
    const SPLIT_KEY = 'memorizz.waterfall.split';
    function setSplit(value, save) {
        const split = Math.min(70, Math.max(20, value));
        section.style.setProperty('--wf-split', split + '%');
        section.querySelectorAll('[data-wf-divider]').forEach(divider => divider.setAttribute('aria-valuenow', String(Math.round(split))));
        if (save) { try { localStorage.setItem(SPLIT_KEY, String(split)); } catch (error) { /* private mode */ } }
        return split;
    }
    let split = 38;
    try { split = Number(localStorage.getItem(SPLIT_KEY)) || 38; } catch (error) { /* private mode */ }
    split = setSplit(split, false);

    // The axis sticks under the page's own sticky section bar.
    const nav = document.querySelector('.trace-section-nav');
    function stickUnderNav() { section.style.setProperty('--wf-sticky-top', (nav ? nav.offsetHeight : 0) + 'px'); }
    stickUnderNav();
    window.addEventListener('resize', stickUnderNav);

    const filter = {lane: 'all', query: ''};
    const isFiltering = () => filter.lane !== 'all' || filter.query !== '';
    const status = section.querySelector('[data-wf-status]');
    const announce = text => { if (status) status.textContent = text; };
    const laneColor = {};
    function readColors() {
        const style = getComputedStyle(section);
        for (const lane of ['model', 'tool', 'memory', 'output', 'run', 'gap', 'danger']) {
            laneColor[lane] = style.getPropertyValue('--wf-' + lane).trim() || '#888';
        }
    }
    readColors();

    const turns = [...section.querySelectorAll('[data-wf-turn]')].map(setupTurn);

    function setupTurn(details) {
        const grid = details.querySelector('[data-wf-grid]');
        const rows = [...grid.querySelectorAll('[data-wf-row]')];
        const byId = new Map(rows.map(row => [row.id, row]));
        const kids = new Map();
        rows.forEach(row => {
            const parent = row.dataset.parent;
            if (!parent) return;
            if (!kids.has(parent)) kids.set(parent, []);
            kids.get(parent).push(row);
        });
        let info = {};
        try { info = JSON.parse(details.querySelector('[data-wf-details]').textContent); } catch (error) { info = {}; }
        const turn = {
            details, grid, rows, byId, kids, info,
            extent: Number(grid.dataset.wfExtent) || 1,
            collapsed: new Set(rows.filter(row => row.getAttribute('aria-expanded') === 'false').map(row => row.id)),
            zoom: [0, 100],
            selected: null,
            panel: null,
            axis: grid.querySelector('[data-wf-axis]'),
            gridlines: grid.querySelector('[data-wf-gridlines]'),
            minimap: grid.querySelector('[data-wf-minimap]'),
            windowBox: grid.querySelector('[data-wf-window]'),
            reset: grid.querySelector('[data-wf-reset]'),
            hiddenNote: grid.querySelector('[data-wf-hidden-note]'),
            breakdown: details.querySelector('.wf-breakdown'),
            empty: details.querySelector('[data-wf-empty]'),
        };
        turn.parentOf = row => (row.dataset.parent ? byId.get(row.dataset.parent) : null);
        // A row picked from the URL shows, even inside a fold.
        rows.filter(row => row.classList.contains('is-selected')).forEach(row => {
            for (let parent = turn.parentOf(row); parent; parent = turn.parentOf(parent)) turn.collapsed.delete(parent.id);
        });
        const first = rows.find(row => row.classList.contains('is-selected')) || rows[0];
        if (first) first.tabIndex = 0;
        wireTurn(turn);
        return turn;
    }

    // ---------- Visibility: folding, then the filter on top ----------
    function matches(turn, row) {
        if (filter.lane === 'failed' && row.dataset.failed !== 'true') return false;
        if (!['all', 'failed'].includes(filter.lane) && row.dataset.lane !== filter.lane) return false;
        if (filter.query) {
            const data = turn.info[row.id] || {};
            const haystack = [row.dataset.search, data.span_id, data.parent_span_id, data.task_id].join(' ').toLowerCase();
            if (!haystack.includes(filter.query)) return false;
        }
        return true;
    }
    function render(turn) {
        const filtering = isFiltering();
        let hiddenSpans = 0;
        let matched = 0;
        if (!filtering) {
            turn.rows.forEach(row => {
                let hidden = false;
                for (let parent = turn.parentOf(row); parent; parent = turn.parentOf(parent)) {
                    if (turn.collapsed.has(parent.id)) { hidden = true; break; }
                }
                row.hidden = hidden;
                row.classList.remove('is-context');
            });
        } else {
            const hits = new Set(turn.rows.filter(row => row.dataset.lane !== 'gap' && matches(turn, row)));
            const shown = new Set();
            hits.forEach(row => {
                for (let node = row; node && !shown.has(node); node = turn.parentOf(node)) shown.add(node);
            });
            turn.rows.forEach(row => {
                row.hidden = !shown.has(row);
                row.classList.toggle('is-context', shown.has(row) && !hits.has(row));
                if (row.hidden && row.classList.contains('wf-row--span')) hiddenSpans += 1;
            });
            matched = [...hits].filter(row => row.classList.contains('wf-row--span')).length;
        }
        turn.rows.forEach(row => {
            if (!row.hasAttribute('aria-expanded')) return;
            const open = filtering || !turn.collapsed.has(row.id);
            row.setAttribute('aria-expanded', String(open));
            const toggle = row.querySelector('[data-wf-toggle]');
            if (toggle) toggle.setAttribute('aria-label', (open ? 'Collapse ' : 'Expand ') + (row.querySelector('.wf-label')?.textContent || 'span'));
        });
        if (turn.panel && turn.selected && turn.selected.hidden) closePanel(turn);
        if (turn.hiddenNote) {
            turn.hiddenNote.hidden = !filtering || !hiddenSpans;
            turn.hiddenNote.replaceChildren();
            if (filtering && hiddenSpans) {
                turn.hiddenNote.append(`${count.format(hiddenSpans)} span${hiddenSpans === 1 ? '' : 's'} hidden by this filter. `);
                const all = document.createElement('button');
                all.type = 'button';
                all.className = 'wf-text-button';
                all.textContent = 'Show all spans';
                all.addEventListener('click', () => setFilter('all', ''));
                turn.hiddenNote.append(all);
            }
        }
        const collapseAll = turn.grid.querySelector('[data-wf-collapse-all]');
        if (collapseAll) {
            const expandable = turn.rows.filter(row => row.hasAttribute('aria-expanded'));
            collapseAll.hidden = !expandable.length || filtering;
            collapseAll.textContent = expandable.some(row => !turn.collapsed.has(row.id)) ? 'Collapse all' : 'Expand all';
        }
        // A turn with nothing to show says so, rather than an empty grid.
        const empty = filtering && !matched;
        turn.grid.hidden = empty;
        if (turn.breakdown) turn.breakdown.hidden = empty;
        if (turn.empty) turn.empty.hidden = !empty;
        drawMinimap(turn);
        return matched;
    }
    function renderAll() {
        let matched = 0;
        let turnsWithHits = 0;
        turns.forEach(turn => {
            const hits = render(turn);
            matched += hits;
            if (hits) {
                turnsWithHits += 1;
                if (filter.query) turn.details.open = true;
            }
        });
        if (isFiltering()) {
            announce(matched
                ? `${count.format(matched)} span${matched === 1 ? '' : 's'} match${matched === 1 ? 'es' : ''}${turns.length > 1 ? ` in ${turnsWithHits} turn${turnsWithHits === 1 ? '' : 's'}` : ''}.`
                : 'No spans match. Clear the search or pick another lane.');
        } else announce('');
    }
    function setFilter(lane, query) {
        filter.lane = lane;
        if (query !== undefined) {
            filter.query = query.trim().toLowerCase();
            const search = section.querySelector('[data-wf-search]');
            if (search && search.value !== query) search.value = query;
        }
        section.querySelectorAll('[data-wf-filter]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.wfFilter === lane)));
        renderAll();
    }
    section.querySelectorAll('[data-wf-filter]').forEach(button => button.addEventListener('click', () => setFilter(button.dataset.wfFilter)));
    const search = section.querySelector('[data-wf-search]');
    let searchTimer = null;
    search?.addEventListener('input', () => {
        clearTimeout(searchTimer);
        searchTimer = setTimeout(() => setFilter(filter.lane, search.value), 120);
    });
    search?.addEventListener('keydown', event => {
        if (event.key === 'Escape' && search.value) { event.preventDefault(); setFilter(filter.lane, ''); }
    });

    function toggleRow(turn, row, open) {
        if (!row.hasAttribute('aria-expanded')) return;
        if (isFiltering()) setFilter('all', '');
        const willOpen = open === undefined ? turn.collapsed.has(row.id) : open;
        if (willOpen) turn.collapsed.delete(row.id); else turn.collapsed.add(row.id);
        render(turn);
    }

    // ---------- Zoom: bars, ticks and gridlines follow the window ----------
    function applyZoom(turn) {
        const [from, to] = turn.zoom;
        const scale = 100 / (to - from);
        const place = value => (value - from) * scale;
        turn.grid.querySelectorAll('.wf-track').forEach(track => {
            const bar = track.querySelector('.wf-bar');
            if (!bar) return;
            const left = Number(bar.dataset.left);
            const width = Number(bar.dataset.width);
            bar.style.left = place(left) + '%';
            bar.style.width = width * scale + '%';
            const label = track.querySelector('[data-wf-dur]');
            if (label) {
                const start = place(left);
                const end = place(left + width);
                label.hidden = end < 0 || start > 100;
                const side = end <= 82 ? 'right' : start >= 15 ? 'left' : 'inside';
                for (const name of ['left', 'right', 'inside']) label.classList.toggle('wf-dur--' + name, side === name);
                label.style.left = side === 'right' ? Math.max(0, end) + '%' : '';
                label.style.right = side === 'left' ? 100 - start + '%' : side === 'inside' ? Math.max(0, 100 - end) + '%' : '';
            }
            track.querySelectorAll('.wf-mark').forEach(mark => {
                const at = place(Number(mark.dataset.left));
                mark.style.left = at + '%';
                mark.hidden = at < 0 || at > 100;
            });
        });
        const startMs = (from / 100) * turn.extent;
        const endMs = (to / 100) * turn.extent;
        const step = tickStep(endMs - startMs);
        const ticks = [];
        for (let index = Math.ceil(startMs / step - 1e-9); ; index += 1) {
            const value = index * step;
            if (value > endMs + 1e-9 || ticks.length > 50) break;
            ticks.push([((value - startMs) / (endMs - startMs)) * 100, tickLabel(value, step)]);
        }
        turn.axis.replaceChildren(...ticks.map(([left, text]) => {
            const tick = document.createElement('span');
            tick.style.left = left + '%';
            tick.textContent = text;
            return tick;
        }));
        turn.gridlines.replaceChildren(...ticks.map(([left]) => {
            const line = document.createElement('i');
            line.style.left = left + '%';
            return line;
        }));
        const zoomed = from > 0 || to < 100;
        if (turn.reset) turn.reset.hidden = !zoomed;
        if (turn.windowBox) {
            turn.windowBox.hidden = !zoomed;
            turn.windowBox.style.left = from + '%';
            turn.windowBox.style.width = (to - from) + '%';
        }
        turn.grid.classList.toggle('is-zoomed', zoomed);
    }
    function zoomTo(turn, from, to) {
        const minimum = Math.max(0.05, (0.05 / turn.extent) * 100);
        if (to - from < minimum) {
            const middle = (from + to) / 2;
            from = middle - minimum / 2;
            to = middle + minimum / 2;
        }
        turn.zoom = [Math.max(0, from), Math.min(100, to)];
        applyZoom(turn);
        drawMinimap(turn);
    }

    // ---------- The minimap: every span at once, drag across it to zoom ----------
    function drawMinimap(turn) {
        const box = turn.minimap;
        const canvas = box?.querySelector('canvas');
        if (!canvas) return;
        const width = box.clientWidth;
        const height = box.clientHeight;
        if (!width || !height) return;
        const ratio = window.devicePixelRatio || 1;
        canvas.width = Math.round(width * ratio);
        canvas.height = Math.round(height * ratio);
        canvas.style.width = width + 'px';
        canvas.style.height = height + 'px';
        const context = canvas.getContext('2d');
        context.setTransform(ratio, 0, 0, ratio, 0, 0);
        context.clearRect(0, 0, width, height);
        const drawn = turn.rows.filter(row => !row.classList.contains('wf-row--fold'));
        const lane = (height - 4) / Math.max(drawn.length, 1);
        drawn.forEach((row, index) => {
            const bar = row.querySelector('.wf-bar');
            if (!bar) return;
            const y = 2 + index * lane;
            const x = (Number(bar.dataset.left) / 100) * width;
            const w = Math.max(1.5, (Number(bar.dataset.width) / 100) * width);
            const failed = row.dataset.failed === 'true';
            context.globalAlpha = row.hidden && !failed ? 0.35 : 0.95;
            context.fillStyle = failed ? laneColor.danger : (laneColor[row.dataset.lane] || laneColor.run);
            if (bar.classList.contains('wf-bar--gap')) context.globalAlpha = 0.25;
            context.fillRect(x, y, w, Math.max(1, lane - (lane > 3 ? 1 : 0)));
            if (bar.classList.contains('wf-bar--point')) context.fillRect(x - 0.75, y, 1.5, Math.max(1, lane));
        });
        context.globalAlpha = 1;
    }
    function wireMinimap(turn) {
        const box = turn.minimap;
        if (!box) return;
        let drag = null;
        const percentAt = event => {
            const rect = box.getBoundingClientRect();
            return Math.min(100, Math.max(0, ((event.clientX - rect.left) / rect.width) * 100));
        };
        box.addEventListener('pointerdown', event => {
            if (event.button !== 0) return;
            const at = percentAt(event);
            const [from, to] = turn.zoom;
            const zoomed = from > 0 || to < 100;
            drag = zoomed && at >= from && at <= to ? {mode: 'pan', start: at, from, to} : {mode: 'brush', start: at};
            box.setPointerCapture(event.pointerId);
            event.preventDefault();
        });
        box.addEventListener('pointermove', event => {
            if (!drag) return;
            const at = percentAt(event);
            if (drag.mode === 'pan') {
                const span = drag.to - drag.from;
                const from = Math.min(100 - span, Math.max(0, drag.from + at - drag.start));
                turn.zoom = [from, from + span];
                applyZoom(turn);
            } else {
                const [from, to] = [Math.min(drag.start, at), Math.max(drag.start, at)];
                turn.windowBox.hidden = false;
                turn.windowBox.style.left = from + '%';
                turn.windowBox.style.width = (to - from) + '%';
            }
        });
        const finish = event => {
            if (!drag) return;
            const at = percentAt(event);
            if (drag.mode === 'brush') {
                if (Math.abs(at - drag.start) > 0.75) {
                    zoomTo(turn, Math.min(drag.start, at), Math.max(drag.start, at));
                    announce(`Zoomed to ${duration((turn.zoom[0] / 100) * turn.extent)} – ${duration((turn.zoom[1] / 100) * turn.extent)} of the turn.`);
                } else applyZoom(turn);
            }
            drag = null;
        };
        box.addEventListener('pointerup', finish);
        box.addEventListener('pointercancel', () => { drag = null; applyZoom(turn); });
        box.addEventListener('dblclick', () => { zoomTo(turn, 0, 100); announce('Showing the whole turn.'); });
        if ('ResizeObserver' in window) new ResizeObserver(() => drawMinimap(turn)).observe(box);
        turn.details.addEventListener('toggle', () => { if (turn.details.open) drawMinimap(turn); });
    }

    // ---------- The detail panel under a selected span ----------
    const FIELDS = [
        ['Model', data => data.attributes?.model],
        ['Returned by', data => data.attributes?.response_model !== data.attributes?.model ? data.attributes?.response_model : ''],
        ['Provider', data => data.attributes?.provider],
        ['Tokens', data => {
            const a = data.attributes || {};
            if (a.input_tokens === undefined && a.output_tokens === undefined) return '';
            const parts = [`${count.format(Number(a.input_tokens) || 0)} in`, `${count.format(Number(a.output_tokens) || 0)} out`];
            if (Number(a.cached_tokens)) parts.push(`${count.format(Number(a.cached_tokens))} cached`);
            return parts.join(' · ');
        }],
        ['First token', data => data.attributes?.ttft_ms !== undefined ? duration(data.attributes.ttft_ms) : ''],
        ['Charge', data => {
            const value = Number(data.attributes?.cost_usd);
            if (!data.attributes || data.attributes.cost_usd === undefined || Number.isNaN(value)) return '';
            return value >= 1 ? '$' + value.toFixed(2) : value >= 0.01 ? '$' + value.toFixed(4) : '$' + value.toFixed(6);
        }],
        ['Tool', data => data.attributes?.logical_tool_name || data.attributes?.tool_name],
        ['Tool provider', data => data.attributes?.tool_provider],
        ['Outcome', data => String(data.attributes?.outcome || '').replaceAll('_', ' ')],
        ['Reason', data => String(data.attributes?.outcome_reason_code || '').replaceAll('_', ' ')],
        ['Error code', data => data.attributes?.error_code],
        ['Memory type', data => String(data.attributes?.memory_type || '').replaceAll('_', ' ')],
        ['Step', data => data.attributes?.iteration],
        ['Stage', data => String(data.attributes?.stage || '').replaceAll('_', ' ')],
    ];
    const ID_FIELDS = [
        ['Span', 'span_id'], ['Parent span', 'parent_span_id'], ['Caused by event', 'caused_by_event_id'], ['Task', 'task_id'],
    ];
    function term(list, label, value, mono) {
        if (value === undefined || value === null || value === '') return;
        const pair = document.createElement('div');
        const dt = document.createElement('dt');
        dt.textContent = label;
        const dd = document.createElement('dd');
        dd.textContent = String(value);
        if (mono) dd.className = 'wf-mono';
        pair.append(dt, dd);
        list.append(pair);
    }
    function closePanel(turn) {
        turn.panel?.remove();
        turn.panel = null;
        if (turn.selected) {
            turn.selected.classList.remove('is-selected');
            turn.selected.removeAttribute('aria-selected');
        }
        turn.selected = null;
    }
    function openPanel(turn, row) {
        closePanel(turn);
        const data = turn.info[row.id] || {};
        turn.selected = row;
        row.classList.add('is-selected');
        row.setAttribute('aria-selected', 'true');
        const panel = document.createElement('div');
        panel.className = 'wf-panel';
        panel.setAttribute('role', 'row');
        panel.dataset.lane = row.dataset.lane;
        const cell = document.createElement('div');
        cell.setAttribute('role', 'gridcell');
        cell.className = 'wf-panel-cell';
        panel.append(cell);

        const head = document.createElement('div');
        head.className = 'wf-panel-head';
        const title = document.createElement('h3');
        title.textContent = data.label || 'Span';
        if (data.detail) {
            const detail = document.createElement('span');
            detail.textContent = data.detail;
            title.append(' ', detail);
        }
        head.append(title);
        if (data.type === 'span') {
            const chip = document.createElement('span');
            chip.className = 'wf-panel-status' + (data.failed ? ' is-failed' : '');
            chip.textContent = data.failed ? 'Failed' : String(data.status || 'unknown').replaceAll('_', ' ');
            head.append(chip);
        }
        const close = document.createElement('button');
        close.type = 'button';
        close.className = 'wf-text-button wf-panel-close';
        close.textContent = 'Close';
        close.addEventListener('click', () => { closePanel(turn); row.focus(); });
        head.append(close);
        cell.append(head);

        if (data.why || data.warning) {
            const note = document.createElement('p');
            note.className = data.why ? 'wf-panel-why' : 'wf-panel-warn';
            note.textContent = data.why || data.warning + '.';
            cell.append(note);
        }
        const list = document.createElement('dl');
        list.className = 'wf-panel-facts';
        if (data.offset_ms !== null && data.offset_ms !== undefined) {
            let started = Number(data.offset_ms) < 0.5 ? 'At the start of the turn' : '+' + duration(data.offset_ms) + ' into the turn';
            const date = data.timestamp ? new Date(data.timestamp) : null;
            if (date && !Number.isNaN(date.getTime())) started += ' · ' + clock.format(date);
            term(list, 'Started', started);
        }
        if (data.type === 'gap') {
            term(list, 'Length', duration(data.duration_ms));
            cell.append(list);
            const note = document.createElement('p');
            note.className = 'wf-panel-note';
            note.textContent = 'The turn was open for this stretch, but no span was recorded in it. Work that isn\'t instrumented, or waiting between steps, shows up here.';
            cell.append(note);
        } else if (data.type === 'fold') {
            term(list, 'Spans', data.count);
            term(list, 'Covers', data.duration_ms !== null && data.duration_ms !== undefined ? duration(data.duration_ms) : 'No duration recorded');
            cell.append(list);
        } else {
            term(list, 'Duration', data.duration_ms !== null && data.duration_ms !== undefined ? duration(data.duration_ms)
                : data.timing_state === 'missing_completion' ? 'Started, no end recorded' : 'No duration recorded');
            if (data.self_ms !== null && data.self_ms !== undefined) term(list, 'Own time', duration(data.self_ms) + ' outside the spans inside it');
            FIELDS.forEach(([label, read]) => term(list, label, read(data)));
            term(list, 'Kind', String(data.kind || '').replaceAll('_', ' '));
            term(list, 'Events', data.events);
            ID_FIELDS.forEach(([label, key]) => term(list, label, data[key], true));
            cell.append(list);
        }
        const actions = document.createElement('div');
        actions.className = 'wf-panel-actions';
        if (data.anchor && document.getElementById(data.anchor)) {
            const link = document.createElement('a');
            link.className = 'btn btn-sm';
            link.href = '#' + data.anchor;
            link.textContent = 'Show in timeline';
            actions.append(link);
        }
        const bar = row.querySelector('.wf-bar');
        if (bar && Number(bar.dataset.width) > 0) {
            const zoom = document.createElement('button');
            zoom.type = 'button';
            zoom.className = 'btn btn-sm';
            zoom.textContent = 'Zoom to this span';
            zoom.addEventListener('click', () => {
                const left = Number(bar.dataset.left);
                const width = Number(bar.dataset.width);
                const pad = Math.max(width * 0.08, 0.02);
                zoomTo(turn, left - pad, left + width + pad);
                announce(`Zoomed to ${data.label || 'the span'}.`);
            });
            actions.append(zoom);
        }
        if (data.span_id && navigator.clipboard) {
            const copy = document.createElement('button');
            copy.type = 'button';
            copy.className = 'btn btn-sm';
            copy.textContent = 'Copy span ID';
            copy.addEventListener('click', async () => {
                try { await navigator.clipboard.writeText(String(data.span_id)); copy.textContent = 'Copied'; } catch (error) { copy.textContent = 'Copy failed'; }
                setTimeout(() => { copy.textContent = 'Copy span ID'; }, 1500);
            });
            actions.append(copy);
        }
        if (actions.childElementCount) cell.append(actions);
        row.after(panel);
        turn.panel = panel;
    }

    // ---------- Moving around: pointer, keyboard, failures ----------
    function focusRow(turn, row) {
        turn.rows.forEach(other => { other.tabIndex = -1; });
        row.tabIndex = 0;
        row.focus({preventScroll: true});
        row.scrollIntoView({block: 'nearest'});
    }
    function activate(turn, row) {
        if (row.classList.contains('wf-row--fold')) { toggleRow(turn, row); return; }
        if (turn.selected === row) closePanel(turn); else openPanel(turn, row);
    }
    function reveal(turn, row) {
        turn.details.open = true;
        if (row.hidden && isFiltering()) setFilter('all', '');
        for (let parent = turn.parentOf(row); parent; parent = turn.parentOf(parent)) turn.collapsed.delete(parent.id);
        render(turn);
        const bar = row.querySelector('.wf-bar');
        if (bar) {
            const left = Number(bar.dataset.left);
            const [from, to] = turn.zoom;
            if (left < from || left > to) zoomTo(turn, 0, 100);
        }
        openPanel(turn, row);
        focusRow(turn, row);
        row.scrollIntoView({block: 'center'});
    }
    function wireTurn(turn) {
        turn.grid.addEventListener('click', event => {
            const goto = event.target.closest('[data-wf-goto]');
            if (goto) { const target = turn.byId.get(goto.dataset.wfGoto); if (target) reveal(turn, target); return; }
            const toggle = event.target.closest('[data-wf-toggle]');
            const row = event.target.closest('[data-wf-row]');
            if (toggle && row) { toggleRow(turn, row); focusRow(turn, row); return; }
            if (row && !event.target.closest('a, button, input')) { activate(turn, row); focusRow(turn, row); }
        });
        turn.grid.addEventListener('keydown', event => {
            const row = event.target.closest?.('[data-wf-row]');
            if (!row || event.target !== row) return;
            const visible = turn.rows.filter(other => !other.hidden);
            const at = visible.indexOf(row);
            const move = target => { if (target) { event.preventDefault(); focusRow(turn, target); } };
            const open = row.getAttribute('aria-expanded');
            switch (event.key) {
            case 'ArrowDown': move(visible[at + 1]); break;
            case 'ArrowUp': move(visible[at - 1]); break;
            case 'Home': move(visible[0]); break;
            case 'End': move(visible[visible.length - 1]); break;
            case 'ArrowRight':
                if (open === 'false') { event.preventDefault(); toggleRow(turn, row, true); } else if (open === 'true') move((turn.kids.get(row.id) || []).find(child => !child.hidden));
                break;
            case 'ArrowLeft':
                if (open === 'true' && !isFiltering()) { event.preventDefault(); toggleRow(turn, row, false); } else move(turn.parentOf(row));
                break;
            case 'Enter': case ' ':
                event.preventDefault();
                activate(turn, row);
                break;
            case 'Escape':
                if (turn.panel) { event.preventDefault(); closePanel(turn); }
                break;
            default:
            }
        });
        turn.grid.querySelector('[data-wf-collapse-all]')?.addEventListener('click', event => {
            const expandable = turn.rows.filter(row => row.hasAttribute('aria-expanded'));
            const collapse = expandable.some(row => !turn.collapsed.has(row.id));
            expandable.forEach(row => { if (collapse) turn.collapsed.add(row.id); else turn.collapsed.delete(row.id); });
            render(turn);
            event.currentTarget.focus();
        });
        turn.reset?.addEventListener('click', () => { zoomTo(turn, 0, 100); announce('Showing the whole turn.'); });
        wireMinimap(turn);
        wireDivider(turn);
    }
    function wireDivider(turn) {
        const divider = turn.grid.querySelector('[data-wf-divider]');
        if (!divider) return;
        let dragging = false;
        divider.addEventListener('pointerdown', event => { dragging = true; divider.setPointerCapture(event.pointerId); event.preventDefault(); });
        divider.addEventListener('pointermove', event => {
            if (!dragging) return;
            const rect = turn.grid.getBoundingClientRect();
            split = setSplit(((event.clientX - rect.left) / rect.width) * 100, false);
        });
        const stop = () => { if (dragging) { dragging = false; setSplit(split, true); turns.forEach(drawMinimap); } };
        divider.addEventListener('pointerup', stop);
        divider.addEventListener('pointercancel', stop);
        divider.addEventListener('keydown', event => {
            const delta = {ArrowLeft: -2, ArrowRight: 2}[event.key];
            if (!delta) return;
            event.preventDefault();
            split = setSplit(split + delta, true);
            turns.forEach(drawMinimap);
        });
    }
    let failureIndex = -1;
    section.querySelector('[data-wf-next-failure]')?.addEventListener('click', () => {
        const failures = turns.flatMap(turn => turn.rows.filter(row => row.dataset.failed === 'true').map(row => [turn, row]));
        if (!failures.length) return;
        failureIndex = (failureIndex + 1) % failures.length;
        const [turn, row] = failures[failureIndex];
        reveal(turn, row);
        const label = [row.querySelector('.wf-label')?.textContent, row.querySelector('.wf-detail')?.textContent].filter(Boolean).join(' ');
        announce(`Failure ${failureIndex + 1} of ${failures.length}: ${label}.`);
    });

    turns.forEach(turn => { render(turn); applyZoom(turn); });
    const media = window.matchMedia?.('(prefers-color-scheme: dark)');
    new MutationObserver(() => { readColors(); turns.forEach(drawMinimap); }).observe(document.body, {attributes: true, attributeFilter: ['class']});
    media?.addEventListener?.('change', () => { readColors(); turns.forEach(drawMinimap); });
})();

/* Trace inspection only. No tool/model execution or production replay. */
(() => {
    'use strict';
    const status = document.getElementById('incident-status');
    const scope = new URLSearchParams(window.location.search);
    const scopeParams = () => {
        const params = new URLSearchParams();
        for (const key of ['user_id', 'application_id']) {
            if (scope.has(key)) params.set(key, scope.get(key));
        }
        return params;
    };
    const selectionBody = () => {
        const body = Object.fromEntries(scopeParams());
        for (const key of ['thread_id', 'thread_memory_id', 'turn_id', 'task_id', 'run_id', 'start_time', 'end_time']) {
            if (scope.get(key)) body[key] = scope.get(key);
        }
        return body;
    };
    async function request(url, body) {
        const options = body ? {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)} : {};
        const response = await fetch(url, {...options, credentials: 'same-origin', cache: 'no-store'});
        const payload = await response.json();
        if (!response.ok) throw new Error(typeof payload.detail === 'string' ? payload.detail : 'Trace request failed');
        return payload;
    }
    const form = document.getElementById('incident-finder');
    let cursor = null;
    let searchParams = null;
    const groups = new Map();
    async function search(next = false) {
        if (!form) return;
        const params = next ? new URLSearchParams(searchParams) : scopeParams();
        if (!next) {
            const values = new FormData(form);
            for (const key of ['q', 'start_time', 'end_time']) {
                if (values.get(key)) params.set(key, key === 'q' ? values.get(key) : new Date(values.get(key)).toISOString());
            }
            searchParams = params.toString();
        }
        if (next && cursor) params.set('cursor', cursor);
        const result = await request('/traces/find.json?' + params);
        const target = document.getElementById('incident-results');
        if (!next) { target.replaceChildren(); groups.clear(); }
        for (const event of result.items) {
            const groupKey = JSON.stringify([event.agent_id, event.thread_id, event.root_trace_id, event.task_id]);
            let group = groups.get(groupKey);
            if (!group) {
                group = document.createElement('details');
                const label = document.createElement('summary');
                label.textContent = `Root ${event.root_trace_id || 'not captured'} · task ${event.task_id || 'unspecified'} · thread ${event.thread_id || 'unknown'}`;
                group.append(label);
                group.open = groups.size === 0;
                groups.set(groupKey, group);
                target.append(group);
            }
            const row = document.createElement('p');
            const link = document.createElement('a');
            const route = scopeParams();
            for (const key of ['start_time', 'end_time']) {
                if (searchParams && new URLSearchParams(searchParams).has(key)) route.set(key, new URLSearchParams(searchParams).get(key));
            }
            route.set('agent_id', event.agent_id || '');
            for (const key of ['thread_id', 'root_trace_id', 'turn_id', 'task_id', 'event_id']) {
                if (event[key]) route.set(key, event[key]);
            }
            link.href = '/traces?' + route + (event.anchor ? '#' + encodeURIComponent(event.anchor) : '');
            link.textContent = `${event.result_type}: ${event.operation || event.logical_tool_name || event.tool_name || event.kind || 'trace'} · ${event.event_id || ''}`;
            row.append(link);
            group.append(row);
        }
        cursor = result.next_cursor;
        document.getElementById('incident-next').hidden = !cursor;
        status.textContent = `${result.normalized_events} recorded events in this page · stored-event read ${result.coverage}. End-to-end evidence must be inspected separately.`;
    }
    form?.addEventListener('submit', event => { event.preventDefault(); search().catch(error => {status.textContent = error.message;}); });
    document.getElementById('incident-next')?.addEventListener('click', () => search(true).catch(error => {status.textContent = error.message;}));
    document.getElementById('account-lookup')?.addEventListener('submit', async event => {
        event.preventDefault();
        const input = document.getElementById('account-email');
        const email = input.value;
        input.value = '';
        try {
            const result = await request('/traces/account/resolve?' + scopeParams(), {email});
            scope.set('user_id', result.user_id);
            if (result.application_id) scope.set('application_id', result.application_id);
            status.textContent = 'Account resolved. Searching its authorized traces.';
            form.elements.q.value = '';
            await search();
        } catch (error) { status.textContent = error.message; }
    });
    function showContent(output, content) {
        let text = content || 'Nothing was captured for this event.';
        let json = false;
        if (content && /^[[{]/.test(content.trim())) {
            try { text = JSON.stringify(JSON.parse(content), null, 2); json = true; } catch (error) { /* plain text */ }
        }
        output.textContent = text;
        output.classList.toggle('is-json', json);
        output.classList.toggle('is-note', !content);
    }
    function setRevealed(button, open) {
        button.nextElementSibling.hidden = !open;
        button.setAttribute('aria-expanded', String(open));
        const label = button.querySelector('span');
        if (label) label.textContent = open ? 'Hide' : button.dataset.label;
    }
    // Each reveal is its own audited request; a second click only folds it.
    async function reveal(button) {
        const output = button.nextElementSibling;
        if (button.dataset.loaded) { setRevealed(button, output.hidden); return; }
        button.setAttribute('aria-busy', 'true');
        try {
            const body = {...selectionBody(), event_id: button.dataset.revealEvent, agent_id: button.dataset.agentId};
            for (const [key, field] of [['root_trace_id', 'rootId'], ['source_id', 'sourceId'], ['thread_id', 'threadId'], ['turn_id', 'turnId'], ['run_id', 'runId']]) {
                if (button.dataset[field]) body[key] = button.dataset[field];
            }
            const result = await request('/traces/reveal', body);
            showContent(output, result.content);
            button.dataset.loaded = 'true';
        } catch (error) {
            output.textContent = error.message;
            output.classList.remove('is-json');
            output.classList.add('is-note');
        } finally { button.removeAttribute('aria-busy'); }
        setRevealed(button, true);
    }
    document.querySelectorAll('[data-reveal-event]').forEach(button => button.addEventListener('click', () => reveal(button)));
    document.getElementById('trace-reveal-all')?.addEventListener('click', async event => {
        const all = event.currentTarget;
        const buttons = [...document.querySelectorAll('.obs-tl-event:not([hidden]) [data-reveal-event]')];
        if (all.dataset.shown) {
            buttons.filter(button => button.dataset.loaded).forEach(button => setRevealed(button, false));
            delete all.dataset.shown;
            all.textContent = 'Reveal all content';
            return;
        }
        const queue = buttons.filter(button => !button.dataset.loaded || button.nextElementSibling.hidden);
        all.disabled = true;
        const worker = async () => { while (queue.length) await reveal(queue.shift()); };
        await Promise.all(Array.from({length: Math.min(4, queue.length)}, worker));
        all.disabled = false;
        all.dataset.shown = 'true';
        all.textContent = 'Hide all content';
    });
    const timelineEvents = [...document.querySelectorAll('.obs-tl-event')];
    function markRailEnds() {
        const visible = timelineEvents.filter(row => !row.hidden);
        timelineEvents.forEach(row => row.classList.remove('is-first', 'is-last'));
        visible[0]?.classList.add('is-first');
        visible[visible.length - 1]?.classList.add('is-last');
        // A date shows where the day changes, not on every event.
        let day = null;
        for (const row of visible) {
            const date = row.querySelector('[data-tl-date]');
            if (!date) continue;
            date.hidden = date.dataset.tlDay === day;
            day = date.dataset.tlDay;
        }
    }
    document.querySelectorAll('[data-tl-filter]').forEach(button => button.addEventListener('click', () => {
        const key = button.dataset.tlFilter;
        document.querySelectorAll('[data-tl-filter]').forEach(other => other.setAttribute('aria-pressed', String(other === button)));
        timelineEvents.forEach(row => {
            row.hidden = key !== 'all' && (key === 'errors' ? row.dataset.tlError !== 'true' : row.dataset.tlGroup !== key);
        });
        markRailEnds();
    }));
    const clock = new Intl.DateTimeFormat(undefined, {hour: '2-digit', minute: '2-digit', second: '2-digit'});
    const day = new Intl.DateTimeFormat(undefined, {month: 'short', day: 'numeric'});
    const full = new Intl.DateTimeFormat(undefined, {dateStyle: 'medium', timeStyle: 'medium'});
    const stamp = new Intl.DateTimeFormat(undefined, {dateStyle: 'medium', timeStyle: 'short'});
    const parsed = node => { const date = new Date(node.getAttribute('datetime')); return Number.isNaN(date.getTime()) ? null : date; };
    document.querySelectorAll('time[data-tl-clock]').forEach(node => {
        const date = parsed(node);
        if (!date) return;
        node.textContent = clock.format(date);
        node.title = full.format(date);
        const label = node.parentElement.querySelector('[data-tl-date]');
        if (label) { label.textContent = day.format(date); label.dataset.tlDay = date.toDateString(); }
    });
    // The summary's range names the day once when it starts and ends on it.
    const range = [...document.querySelectorAll('time[data-tl-stamp]')].map(node => [node, parsed(node)]);
    range.forEach(([node, date]) => { if (date) node.textContent = stamp.format(date); });
    if (range.length === 2 && range[0][1] && range[1][1] && range[0][1].toDateString() === range[1][1].toDateString()) {
        const minutes = new Intl.DateTimeFormat(undefined, {hour: '2-digit', minute: '2-digit'});
        const [start, end] = [minutes.format(range[0][1]), minutes.format(range[1][1])];
        if (start === end) { range[1][0].previousSibling.remove(); range[1][0].remove(); } else { range[1][0].textContent = end; }
    }
    markRailEnds();
    document.querySelectorAll('[data-tl-details]').forEach(button => button.addEventListener('click', () => {
        const panel = document.getElementById(button.getAttribute('aria-controls'));
        panel.hidden = !panel.hidden;
        button.setAttribute('aria-expanded', String(!panel.hidden));
    }));
    document.querySelectorAll('[data-tl-copy]').forEach(button => button.addEventListener('click', async () => {
        const json = button.closest('.obs-tl-raw').querySelector('.obs-tl-json');
        try {
            await navigator.clipboard.writeText(json.textContent);
            button.textContent = 'Copied';
        } catch (error) { button.textContent = 'Copy failed'; }
        setTimeout(() => { button.textContent = 'Copy JSON'; }, 1500);
    }));
    const category = document.getElementById('trace-category');
    document.querySelectorAll('[data-artifact-ref]').forEach(button => button.addEventListener('click', async () => {
        const output = button.nextElementSibling;
        try {
            const params = scopeParams();
            params.set('resource_type', button.dataset.artifactType);
            params.set('ref', button.dataset.artifactRef);
            const result = await request('/traces/artifact.json?' + params);
            output.textContent = JSON.stringify(result, null, 2);
        } catch (error) { output.textContent = error.message; }
        output.hidden = false;
    }));
    const successes = document.getElementById('trace-show-success');
    function filterWaterfall() {
        document.querySelectorAll('[data-waterfall-kind]').forEach(row => {
            const kind = row.dataset.waterfallKind;
            const group = kind.startsWith('memory') || kind === 'context_binding' ? 'memory' : kind.startsWith('model') ? 'model' : kind.startsWith('tool') ? 'tool' : kind.startsWith('artifact') ? 'artifact' : ['ui_delivery', 'output_contract'].includes(kind) ? 'delivery' : 'host';
            const error = ['error', 'partial'].includes(row.dataset.waterfallStatus) || row.dataset.waterfallGap === 'true';
            row.hidden = (category && category.value !== 'all' && (category.value === 'errors' ? !error : group !== category.value)) || (!successes?.checked && ['model', 'tool'].includes(group) && row.dataset.waterfallStatus === 'success' && !error);
        });
    }
    category?.addEventListener('change', filterWaterfall);
    successes?.addEventListener('change', filterWaterfall);
    filterWaterfall();
    const selected = document.querySelector('[data-selected-event]');
    if (selected) { selected.scrollIntoView({block: 'center'}); selected.focus({preventScroll: true}); }
    document.getElementById('trace-replay-create')?.addEventListener('click', async event => {
        const button = event.currentTarget;
        const root = document.getElementById('trace-replay-root').value;
        try {
            const body = {...selectionBody(), agent_id: button.dataset.agentId, root_trace_id: root};
            const draft = await request('/traces/replays', body);
            document.getElementById('trace-replay-status').textContent = `Draft ${draft.experiment_id} created. Execution is disabled; review in Evalground.`;
        } catch (error) { document.getElementById('trace-replay-status').textContent = error.message; }
    });
})();

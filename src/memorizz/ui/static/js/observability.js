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
    document.querySelectorAll('[data-reveal-event]').forEach(button => button.addEventListener('click', async () => {
        const output = button.nextElementSibling;
        try {
            const body = {...selectionBody(), event_id: button.dataset.revealEvent, agent_id: button.dataset.agentId, root_trace_id: button.dataset.rootId};
            for (const [key, field] of [['thread_id', 'threadId'], ['turn_id', 'turnId'], ['run_id', 'runId']]) {
                if (button.dataset[field]) body[key] = button.dataset[field];
            }
            const result = await request('/traces/reveal', body);
            output.textContent = result.content || 'No preview captured';
            output.hidden = false;
        } catch (error) { output.textContent = error.message; output.hidden = false; }
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

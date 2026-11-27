/* Full-page evidence. Clipboard exports use stored events, never DOM excerpts. */
(() => {
    const root = document.getElementById('run-evidence');
    if (!root) return;
    const T = window.MemorizzTrajectory;
    const runId = root.dataset.runId;
    const api = '/api/harness-runs/' + encodeURIComponent(runId);
    const container = document.getElementById('evidence-steps');
    const loadStatus = document.getElementById('evidence-load-status');
    const copyStatus = document.getElementById('evidence-copy-status');
    const copy = document.getElementById('evidence-copy');
    const raw = document.getElementById('evidence-json');
    const fallback = document.getElementById('evidence-copy-fallback');
    const state = {events: [], after: 0, loaded: false};
    const tabs = [...root.querySelectorAll('[role="tab"]')];
    const activeStatuses = new Set(['pending_approval', 'queued', 'running']);
    let timer;
    let pending = false;
    let stopped = false;
    let active = container.dataset.active === 'true';
    let memoryMounted = false;
    const request = async (url) => {
        const response = await fetch(url, {cache: 'no-store'});
        if (!response.ok) throw new Error('Could not load activity (HTTP ' + response.status + ').');
        return response.json();
    };
    const loadRecord = async () => {
        raw.textContent = 'Loading every recorded event…';
        try { raw.textContent = JSON.stringify(await request(api + '/activity'), null, 2); }
        catch (error) { raw.textContent = error.message; }
    };
    // What the run did with memory: its limit, what it was given at the
    // start, what it looked up itself, and what it was refused.
    const used = document.getElementById('evidence-used-body');
    const refusedBadge = document.getElementById('evidence-refused');
    const esc = T.esc;
    const memoryLink = (id) => '<a class="hx-used-memory" href="/harnesses/memory/' + encodeURIComponent(id) + '">' + esc(id) + '</a>';
    const drawUsed = (view) => {
        const scope = view.scope;
        const reach = scope.open
            ? 'This run was opened to every memory of the user.'
            : scope.private
                ? 'No Memory ID was given, so the run had a private memory of its own. Nothing it learned is shared with other runs.'
                : 'Limited to the memory ' + memoryLink(scope.memory_id) + ': it could read ' + (scope.can_write ? 'and propose changes to ' : '') + 'this memory and no other.';
        const readOnly = scope.read_only.length ? ' It could also read, never write: ' + scope.read_only.map(memoryLink).join(', ') + '.' : '';
        const tools = scope.tools ? '' : ' MemoRizz tools were turned off, so it had only what it was given at the start.';
        const given = view.given.length
            ? '<ol class="hx-used-list">' + view.given.map(record => '<li>'
                + '<div class="hx-used-head"><span class="hx-used-kind">' + esc(T.humanize(record.memory_type || 'memory')) + '</span>'
                + (record.written_by ? '<span>written by <strong>' + esc(record.written_by) + '</strong></span>' : '')
                + (record.run_id ? '<a href="/harnesses/runs/' + encodeURIComponent(record.run_id) + '">its run</a>' : '')
                + (record.read_only ? '<span class="hx-used-tag">read-only · ' + esc(record.memory_id) + '</span>' : '')
                + (record.score != null ? '<span class="hx-used-score" title="How closely it matched the task">match ' + Number(record.score).toFixed(2) + '</span>' : '')
                + '</div><p>' + esc(record.text) + '</p></li>').join('') + '</ol>'
            : '<p class="hx-used-none">' + (view.given_known ? 'Nothing in memory matched the task, so it started with none.' : 'Not recorded yet: the run has not finished.') + '</p>';
        const outcome = (call) => call.outcome === 'refused' ? 'Refused' : call.outcome === 'failed' ? 'Failed' : call.outcome === 'nothing' ? 'Found nothing' : call.found != null ? 'Found ' + call.found : call.outcome === 'ok' ? 'Done' : 'No result recorded';
        const asked = view.lookups.length
            ? '<ol class="hx-used-list">' + view.lookups.map(call => '<li class="hx-used-call hx-used-call--' + esc(call.outcome || 'none') + '">'
                + '<div class="hx-used-head"><span class="hx-used-kind">' + esc(T.humanize(call.tool)) + '</span>'
                + (call.writes ? '<span class="hx-used-tag">write</span>' : '')
                + '<span class="hx-used-outcome">' + esc(outcome(call)) + '</span></div>'
                + (Object.keys(call.arguments).length ? '<p class="mono">' + esc(Object.entries(call.arguments).map(pair => pair[0] + ': ' + pair[1]).join(' · ')) + '</p>' : '')
                + (call.message ? '<p class="hx-used-why">' + esc(call.message) + '</p>' : '')
                + '</li>').join('') + '</ol>'
            : '<p class="hx-used-none">It did not use MemoRizz tools itself.</p>';
        used.innerHTML = '<p class="hx-used-scope">' + reach + readOnly + tools + '</p>'
            + (view.refused ? '<p class="hx-used-refusals">' + view.refused + (view.refused === 1 ? ' lookup was' : ' lookups were') + ' refused because ' + (view.refused === 1 ? 'it' : 'they') + ' reached outside this run\u2019s memory. That is why it may have found nothing.</p>' : '')
            + '<h3>Given at the start <span>' + view.given.length + '</span></h3>' + given
            + '<h3>Looked up itself <span>' + view.lookups.length + '</span></h3>' + asked
            + (scope.memory_id ? '<p class="hx-used-saved">' + (view.saved ? 'Its answer was saved to ' + memoryLink(scope.memory_id) + ' for later runs to recall.' : 'Its answer is saved to ' + memoryLink(scope.memory_id) + ' when it finishes with one.') + '</p>' : '');
    };
    const loadUsed = async () => {
        try {
            const view = await request(api + '/memory');
            refusedBadge.hidden = !view.refused;
            refusedBadge.textContent = view.refused ? view.refused + ' refused' : '';
            drawUsed(view);
        } catch (error) { used.innerHTML = '<p class="hx-evidence-note">' + esc(error.message) + '</p>'; }
    };

    const setTab = () => {
        const name = ['record', 'memory', 'used'].includes(location.hash.slice(1)) ? location.hash.slice(1) : 'trajectory';
        tabs.forEach(tab => {
            const selected = tab.hash === '#' + name;
            tab.setAttribute('aria-selected', String(selected));
            tab.tabIndex = selected ? 0 : -1;
            document.getElementById(tab.getAttribute('aria-controls')).hidden = !selected;
        });
        if (name === 'record') loadRecord();
        if (name === 'used') loadUsed();
        if (name === 'memory' && !memoryMounted && window.MemorizzMemoryEvolution) {
            const view = root.querySelector('[data-memory-evolution]');
            window.MemorizzMemoryEvolution.mount(view).setScope({runId, agentId: '', memoryId: ''});
            memoryMounted = true;
        }
    };
    root.querySelector('[role="tablist"]').addEventListener('keydown', event => {
        const index = tabs.indexOf(document.activeElement);
        if (index < 0) return;
        let next;
        if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
        if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
        if (event.key === 'Home') next = 0;
        if (event.key === 'End') next = tabs.length - 1;
        if (next == null) return;
        event.preventDefault();
        tabs[next].focus();
        location.hash = tabs[next].hash;
    });
    window.addEventListener('hashchange', setTab);
    setTab();

    const refresh = async () => {
        if (pending || stopped || document.hidden) return;
        pending = true;
        clearTimeout(timer);
        try {
            const {run} = await request(api);
            active = activeStatuses.has(run.status);
            container.dataset.active = String(active);
            document.getElementById('evidence-run-status').textContent = T.humanize(run.status);
            // Follow the sequence cursor until the store is exhausted, even for
            // sessions larger than 10,000 events. No page-count cutoff.
            while (!stopped) {
                const body = await request(api + '/events?after=' + state.after + '&limit=1000');
                const events = (body.events || []).filter(event => event.sequence > state.after);
                if (!events.length) break;
                state.events.push(...events);
                state.after = events[events.length - 1].sequence;
            }
            state.loaded = true;
            T.render(container, state);
            // The tab shows a count of refused lookups without being opened.
            if (!active || location.hash === '#used') loadUsed();
            loadStatus.textContent = state.events.length + ' recorded events' + (active ? ' · Live' : '');
        } catch (error) {
            loadStatus.textContent = error.message;
        } finally {
            pending = false;
            if (!stopped && (active || !state.loaded)) timer = setTimeout(refresh, 2000);
        }
    };
    document.addEventListener('visibilitychange', () => { if (!document.hidden) refresh(); });
    window.addEventListener('pagehide', () => { stopped = true; clearTimeout(timer); });
    window.addEventListener('pageshow', event => { if (event.persisted) { stopped = false; refresh(); } });
    refresh();

    copy.addEventListener('click', async () => {
        copy.disabled = true;
        copyStatus.textContent = 'Preparing full activity…';
        fallback.hidden = true;
        try {
            // Fetch a fresh snapshot, including delegate runs and event kinds
            // the visual trajectory does not project into steps.
            const activity = await request(api + '/activity');
            const text = JSON.stringify(activity, null, 2);
            const runs = [activity.run, ...activity.delegated_runs];
            const count = runs.reduce((sum, run) => sum + run.events.length, 0);
            let copied = false;
            try {
                await navigator.clipboard.writeText(text);
                copied = true;
            } catch (_) {
                fallback.value = text;
                fallback.hidden = false;
                fallback.focus();
                fallback.select();
                copied = document.execCommand('copy');
                if (copied) { fallback.hidden = true; copy.focus(); }
            }
            copyStatus.textContent = copied ? 'Copied all ' + count + ' recorded events.' : 'Full activity selected below. Press ⌘C or Ctrl+C to copy.';
        } catch (error) {
            copyStatus.textContent = 'Copy failed. ' + error.message;
        } finally { copy.disabled = false; }
    });
})();

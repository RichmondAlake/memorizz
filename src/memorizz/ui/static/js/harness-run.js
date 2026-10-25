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
    const setTab = () => {
        const name = ['record', 'memory'].includes(location.hash.slice(1)) ? location.hash.slice(1) : 'trajectory';
        tabs.forEach(tab => {
            const selected = tab.hash === '#' + name;
            tab.setAttribute('aria-selected', String(selected));
            tab.tabIndex = selected ? 0 : -1;
            document.getElementById(tab.getAttribute('aria-controls')).hidden = !selected;
        });
        if (name === 'record') loadRecord();
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

// A playground conversation answered by a harness (Codex, Claude Code, ...)
// in place of the agent's own model. The choice is per conversation and
// lives in this browser; nothing is saved on the agent. A harness turn is a
// job that can take minutes, so the reply shows it as one: who is working,
// for how long, each step as it happens, and a way to stop it.
window.pgHarness = (function() {
    'use strict';
    const pick = document.getElementById('pg-answer');
    const model = document.getElementById('pg-answer-model');
    const models = document.getElementById('pg-answer-models');
    if (!pick) return null;
    const AGENT = window.PG_DISPLAY_AGENT_ID || '';
    const KEY = 'memorizz.pg.answeredBy.' + AGENT;
    const T = window.MemorizzTrajectory;
    const jobs = new Map();
    let harnesses = [];
    let runtime = null;
    let shownFor = null;

    const memoryId = function() {
        const field = document.getElementById('pg-memory-id');
        return field ? field.value.trim() : '';
    };
    function all() {
        try { return JSON.parse(sessionStorage.getItem(KEY) || '{}') || {}; } catch (_) { return {}; }
    }
    function choice(memory) { return all()[memory || ''] || {harness: '', model: ''}; }
    function remember(memory, value) {
        const kept = all();
        if (value.harness) kept[memory || ''] = value; else delete kept[memory || ''];
        try { sessionStorage.setItem(KEY, JSON.stringify(kept)); } catch (_) { /* this tab only */ }
    }
    function label(name) {
        const found = harnesses.find(function(item) { return item.name === name; });
        return found ? found.label : name;
    }

    // What a harness turn changes around the composer: live voice assumes a
    // quick model reply, and the access panel now applies to this turn.
    function reflect() {
        const chosen = pick.value;
        const found = harnesses.find(function(item) { return item.name === chosen; });
        model.hidden = !chosen;
        models.replaceChildren.apply(models, ((found && found.models) || []).map(function(name) {
            const option = document.createElement('option');
            option.value = name;
            return option;
        }));
        model.placeholder = chosen ? label(chosen) + '’s default model' : '';
        document.body.classList.toggle('pg-answered-by-harness', !!chosen);
        ['pg-call-btn', 'pg-voice-btn'].forEach(function(id) {
            const button = document.getElementById(id);
            if (!button) return;
            button.disabled = !!chosen;
            if (chosen) button.title = 'Voice is for the agent’s own model; a harness turn takes longer.';
        });
        const section = document.getElementById('pg-hx');
        if (section) {
            section.hidden = !chosen && section.dataset.delegates !== 'true';
            const title = document.getElementById('pg-hx-title');
            if (title) title.textContent = chosen ? label(chosen) + '’s access' : 'Harness delegates';
            const note = document.getElementById('pg-hx-answer-note');
            if (note) note.hidden = !chosen;
        }
        const box = document.getElementById('pg-query');
        if (box && box.dataset.ownPlaceholder == null) box.dataset.ownPlaceholder = box.placeholder;
        if (box) box.placeholder = chosen ? 'Ask ' + label(chosen) + '… (a harness turn can take a few minutes)' : box.dataset.ownPlaceholder;
    }
    function show(memory) {
        shownFor = memory;
        const kept = choice(memory);
        const ready = harnesses.some(function(item) { return item.name === kept.harness && item.ready; });
        pick.value = ready ? kept.harness : '';
        model.value = ready ? kept.model || '' : '';
        reflect();
    }
    function fill() {
        Array.from(pick.options).slice(1).forEach(function(option) { option.remove(); });
        // An agent that runs on a harness is answered by it, not by its model.
        const own = pick.options[0];
        if (own.dataset.own == null) own.dataset.own = own.textContent;
        const on = runtime && harnesses.find(function(item) { return item.name === runtime; });
        own.textContent = runtime ? (on ? on.label : 'A harness') + ' · this agent runs on it' : own.dataset.own;
        own.title = runtime ? 'This agent is set to run on a harness, so its turns go there. Change it under Harness in the agent’s settings.' : '';
        harnesses.forEach(function(item) {
            const option = document.createElement('option');
            option.value = item.name;
            option.textContent = item.label + (item.ready ? '' : ' · not ready');
            option.disabled = !item.ready;
            if (!item.ready) option.title = item.why;
            pick.append(option);
        });
        pick.closest('.pg-answer').hidden = !harnesses.length;
        show(memoryId());
    }
    pick.addEventListener('change', function() {
        remember(memoryId(), {harness: pick.value, model: ''});
        model.value = '';
        reflect();
    });
    model.addEventListener('change', function() {
        remember(memoryId(), {harness: pick.value, model: model.value.trim()});
    });
    // Each conversation has its own choice; follow the one on screen.
    setInterval(function() { if (memoryId() !== shownFor) show(memoryId()); }, 700);
    fetch('/api/agents/' + encodeURIComponent(AGENT) + '/answer-harnesses')
        .then(function(response) { return response.ok ? response.json() : {harnesses: []}; })
        .then(function(data) { harnesses = data.harnesses || []; runtime = data.runtime || null; fill(); })
        .catch(function() { pick.closest('.pg-answer').hidden = true; });

    // --- the job in the reply ---

    function elapsed(ms) {
        const seconds = Math.max(0, Math.round(ms / 1000));
        return Math.floor(seconds / 60) + ':' + String(seconds % 60).padStart(2, '0');
    }
    function build(messageEl, runId, harness) {
        const root = document.createElement('section');
        root.className = 'pgh is-working';
        root.innerHTML = '<header class="pgh-head">'
            + '<span class="pgh-state"><span class="pgh-dot" aria-hidden="true"></span><strong class="pgh-who"></strong> <span class="pgh-doing"></span></span>'
            + '<span class="pgh-side"><span class="pgh-time" aria-live="off">0:00</span>'
            + '<button type="button" class="pgh-link pgh-toggle" aria-expanded="true">Hide steps</button>'
            + '<a class="pgh-link" target="_blank" rel="noopener">Open run</a>'
            + '<button type="button" class="pgh-link pgh-stop">Stop</button></span></header>'
            + '<div class="hx-traj pgh-steps" data-active="true"></div>';
        root.querySelector('.pgh-who').textContent = label(harness);
        root.querySelector('a.pgh-link').href = '/harnesses/runs/' + encodeURIComponent(runId);
        const text = messageEl.querySelector('.pg-message-text');
        if (text) text.before(root); else messageEl.append(root);
        const job = {
            root: root, runId: runId, harness: harness, started: Date.now(), done: false,
            state: {events: [], after: 0, loaded: false},
            steps: root.querySelector('.pgh-steps'), time: root.querySelector('.pgh-time'),
            doing: root.querySelector('.pgh-doing'), timer: null, clock: null, loading: false,
        };
        root.querySelector('.pgh-toggle').addEventListener('click', function() {
            const open = job.steps.hidden;
            job.steps.hidden = !open;
            this.setAttribute('aria-expanded', String(open));
            this.textContent = open ? 'Hide steps' : 'Show steps';
        });
        root.querySelector('.pgh-stop').addEventListener('click', function() { stop(runId); });
        job.clock = setInterval(function() { if (!job.done) job.time.textContent = elapsed(Date.now() - job.started); }, 1000);
        jobs.set(runId, job);
        // The run is announced a moment before it is saved, so the first
        // look waits: asking at once only gets "not found".
        job.timer = setTimeout(function() { follow(job); }, 1000);
        return job;
    }
    // Found again after a reload, a job counts from when its run started.
    function age(job) {
        if (job.aged) return;
        job.aged = true;
        fetch('/api/harness-runs/' + encodeURIComponent(job.runId), {cache: 'no-store'})
            .then(function(response) { return response.ok ? response.json() : null; })
            .then(function(data) {
                const at = data && data.run && Date.parse(data.run.started_at || data.run.created_at);
                if (at && at < job.started) job.started = at;
            }).catch(function() {});
    }
    async function follow(job) {
        if (job.loading || !job.root.isConnected) return;
        job.loading = true;
        try {
            while (true) {
                const response = await fetch('/api/harness-runs/' + encodeURIComponent(job.runId) + '/events?after=' + job.state.after + '&limit=500', {cache: 'no-store'});
                if (!response.ok) break;
                age(job);
                const events = ((await response.json()).events || []).filter(function(event) { return event.sequence > job.state.after; });
                if (!events.length) break;
                job.state.events.push.apply(job.state.events, events);
                job.state.after = events[events.length - 1].sequence;
            }
            job.state.loaded = true;
            if (T) T.render(job.steps, job.state);
        } catch (_) { /* the next tick tries again */ }
        job.loading = false;
        if (!job.done) job.timer = setTimeout(function() { follow(job); }, 1500);
    }

    // A status event of a harness turn: start the job's panel, or say what
    // it is doing now.
    function job(messageEl, event) {
        const runId = String(event.harness_run_id || '');
        if (!runId || !messageEl) return;
        let current = jobs.get(runId);
        if (!current || !current.root.isConnected) current = build(messageEl, runId, event.harness || pick.value || 'harness');
        if (event.harness && label(event.harness)) current.root.querySelector('.pgh-who').textContent = label(event.harness);
        // "Codex is working", "Codex: writing": the name is already shown.
        const doing = String(event.message || '').replace(/^[^:]{1,40}:\s*/, '').replace(/^.*? is /, '');
        if (doing && !current.done) current.doing.textContent = 'is ' + doing;
    }
    async function finish(messageEl) {
        const done = Array.from(jobs.values()).filter(function(item) { return !item.done && messageEl && messageEl.contains(item.root); });
        for (const item of done) {
            item.done = true;
            clearTimeout(item.timer);
            clearInterval(item.clock);
            item.steps.dataset.active = 'false';
            await follow(item);
            let run = null;
            try {
                const response = await fetch('/api/harness-runs/' + encodeURIComponent(item.runId), {cache: 'no-store'});
                if (response.ok) run = (await response.json()).run;
            } catch (_) { /* the panel keeps what it has */ }
            const result = (run && run.result) || {};
            const status = (run && run.status) || 'finished';
            const good = status === 'succeeded';
            item.root.classList.remove('is-working');
            item.root.classList.add(good ? 'is-done' : 'is-stopped');
            const facts = [
                good ? 'finished' : status.replace(/_/g, ' '),
                result.latency_ms != null && T ? T.fmtDuration(result.latency_ms) : elapsed(Date.now() - item.started),
                result.cost_usd != null ? '$' + Number(result.cost_usd).toFixed(2) : '',
            ].filter(Boolean).join(' · ');
            item.doing.textContent = facts;
            item.time.textContent = '';
            item.root.querySelector('.pgh-stop').remove();
            const memory = document.createElement('a');
            memory.className = 'pgh-link';
            memory.target = '_blank';
            memory.rel = 'noopener';
            memory.href = '/harnesses/runs/' + encodeURIComponent(item.runId) + '#used';
            memory.textContent = 'Memory it used';
            item.root.querySelector('a.pgh-link').after(memory);
            // Finished, the answer is what matters; the steps fold away.
            item.steps.hidden = true;
            const toggle = item.root.querySelector('.pgh-toggle');
            toggle.setAttribute('aria-expanded', 'false');
            toggle.textContent = 'Show steps';
            if (!good && !result.final_response && result.error) {
                const why = document.createElement('p');
                why.className = 'pgh-error';
                why.textContent = String(result.error).slice(0, 600);
                item.root.append(why);
            }
        }
    }
    // Stop asks the run itself to stop; closing the page does not.
    async function stop(runId) {
        const target = runId ? [jobs.get(runId)] : Array.from(jobs.values());
        const running = target.filter(function(item) { return item && !item.done; });
        await Promise.all(running.map(function(item) {
            item.doing.textContent = 'is stopping…';
            return fetch('/api/harness-runs/' + encodeURIComponent(item.runId) + '/cancel', {method: 'POST'}).catch(function() {});
        }));
        return running.length;
    }

    // A stored answer a harness gave: one quiet line with the way to its run.
    function stored(messageEl, run) {
        if (!messageEl || !run || !run.run_id) return;
        const good = run.status === 'succeeded';
        const line = document.createElement('p');
        line.className = 'pgh-stored';
        const facts = [
            'Answered by ' + label(String(run.harness || 'a harness')),
            good ? '' : String(run.status || '').replace(/_/g, ' '),
            run.latency_ms != null && T ? T.fmtDuration(run.latency_ms) : '',
            run.cost_usd != null ? '$' + Number(run.cost_usd).toFixed(2) : '',
        ].filter(Boolean).join(' \u00b7 ');
        line.append(facts + ' \u00b7 ');
        [['Open run', ''], ['Memory it used', '#used']].forEach(function(pair, index) {
            const link = document.createElement('a');
            link.href = '/harnesses/runs/' + encodeURIComponent(run.run_id) + pair[1];
            link.target = '_blank';
            link.rel = 'noopener';
            link.textContent = pair[0];
            if (index) line.append(' \u00b7 ');
            line.append(link);
        });
        const text = messageEl.querySelector('.pg-message-text');
        if (text) text.before(line); else messageEl.append(line);
    }

    return {
        stored: stored,
        current: function() { return pick.value ? {harness: pick.value, model: model.value.trim()} : null; },
        // A new conversation gets its memory ID with its first turn.
        adopt: function(from, to) {
            if (from === to || !to) return;
            const kept = choice(from);
            if (kept.harness && !choice(to).harness) remember(to, kept);
            if (!from) remember('', {harness: ''});
            shownFor = to;
        },
        job: job, finish: finish, stop: stop,
        working: function() { return Array.from(jobs.values()).some(function(item) { return !item.done; }); },
    };
})();

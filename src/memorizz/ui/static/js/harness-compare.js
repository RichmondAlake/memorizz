/* Compare harness runs side by side: their facts, a shared timeline and their
   trajectories. Needs harness-trajectory.js (window.MemorizzTrajectory); uses
   window.MemorizzMarkdown for answers when the page loads it.

   MemorizzCompare.open(runIds, {title}) opens a dialog for 2–4 runs and keeps it
   live while any of them is still going. MemorizzCompare.summarize(run, events)
   and MemorizzCompare.highlights(summaries) are the pure parts it draws from. */
(function () {
    const T = () => window.MemorizzTrajectory;
    const ACTIVE = new Set(['queued', 'running', 'pending_approval']);
    const MAX_RUNS = 4;
    // Timeline marks: kinds drawn as bars (start to end) and as ticks (a moment).
    const BAR_KINDS = {model: 'Model call', tool: 'Tool call', command: 'Command', subagent: 'Subagent'};
    const TICK_KINDS = {thinking: 'Reasoning', message: 'Message', file: 'File change', error: 'Error'};
    const ACTION_KINDS = ['model', 'tool', 'command', 'subagent'];

    const num = (value) => {
        if (value === null || value === undefined || value === '') return null;
        const n = Number(value);
        return Number.isFinite(n) ? n : null;
    };
    const when = (value) => { const t = Date.parse(value || ''); return Number.isFinite(t) ? t : null; };
    const esc = (value) => T().esc(value);

    // Anthropic reports cache reads and writes beside input_tokens; OpenAI's
    // input_tokens already include cached ones.
    const tokensOf = (usage) => {
        const u = usage && typeof usage === 'object' ? usage : {};
        const read = num(u.cache_read_input_tokens);
        const write = num(u.cache_creation_input_tokens);
        const input = num(u.input_tokens ?? u.prompt_tokens);
        const output = num(u.output_tokens ?? u.completion_tokens);
        if (read !== null || write !== null) {
            return {input: (input || 0) + (read || 0) + (write || 0), output, cached: read};
        }
        return {input, output, cached: num(u.cached_input_tokens ?? u.cached_tokens)};
    };
    // Older Claude Code runs recorded usage for their last model call only;
    // runs that list per-model usage ("models") carry whole-run totals.
    const PARTIAL_USAGE = new Set(['claude-code', 'deepseek']);

    // Steps nest the way the trajectory shows them: under the subagent that ran them.
    const nest = (steps) => {
        const hosts = new Map(steps.filter(step => step.toolId).map(step => [step.toolId, step]));
        return steps.map(step => {
            const host = step.parent && hosts.get(step.parent);
            return {step, host: host && host !== step ? host : null};
        });
    };

    /* One run's facts and timeline marks. Times are seconds from the moment the
       harness started (after any approval wait), so runs line up at zero. */
    const summarize = (run, events, now = Date.now()) => {
        run = run || {};
        events = events || [];
        const {steps, finished} = T().buildSteps(events);
        const status = String(run.status || 'queued');
        const active = ACTIVE.has(status);
        const task = run.task || {};
        const result = run.result || {};
        const first = events.length ? when(events[0].timestamp) : null;
        const started = steps.find(step => step.kind === 'status' && step.title === 'Started');
        const origin = started ? started.at : 0;
        const last = events.length ? when(events[events.length - 1].timestamp) : null;
        const nowAt = first !== null ? (now - first) / 1000 - origin : 0;
        const endAt = active ? Math.max(nowAt, 0) : (first !== null && last !== null ? (last - first) / 1000 - origin : 0);
        let durationMs = num(result.latency_ms);
        if (durationMs === null) {
            const a = when(run.started_at);
            const b = when(run.finished_at) ?? (active ? now : null);
            durationMs = a !== null && b !== null ? Math.max(0, b - a) : Math.max(0, endAt * 1000);
        }
        if (active) durationMs = Math.max(0, endAt * 1000);

        const counts = {model: 0, tool: 0, command: 0, subagent: 0, message: 0, thinking: 0, file: 0, error: 0};
        const bars = [];
        const ticks = [];
        let hiddenReasoning = 0;
        // MemAgent tool-cache results: hits and the tool time they saved.
        const cache = {seen: false, hits: 0, savedMs: 0};
        nest(steps).forEach(({step, host}) => {
            if (step.at < origin && (step.kind === 'status' || step.kind === 'approval')) return;
            if (counts[step.kind] !== undefined) counts[step.kind] += 1;
            const cacheHit = step.kind === 'tool' && !!step.cache && step.cache.status === 'hit';
            if (step.kind === 'tool' && step.cache) cache.seen = true;
            if (cacheHit) { cache.hits += 1; cache.savedMs += num(step.cache.saved_ms) || 0; }
            const at = Math.max(0, step.at - origin);
            if (BAR_KINDS[step.kind]) {
                const running = !!step.running && active;
                const end = step.end !== undefined && step.end !== null
                    ? Math.max(at, step.end - origin)
                    : (running ? Math.max(at, endAt) : at);
                bars.push({
                    kind: step.kind, title: step.title, summary: step.summary || '', start: at, end,
                    running, error: !!step.error || (step.exit !== undefined && step.exit !== null && Number(step.exit) !== 0), cached: cacheHit,
                    depth: host ? 1 : 0, within: host ? host.title : '', host: host ? host.key : '', key: step.key,
                });
            } else if (TICK_KINDS[step.kind]) {
                if (step.kind === 'thinking' && step.hidden) hiddenReasoning += 1;
                ticks.push({kind: step.kind, title: step.title, summary: step.summary || '', at, depth: host ? 1 : 0, within: host ? host.title : '', host: host ? host.key : ''});
            }
        });

        // A subagent lasts until its last nested step ends: Claude Code's Task
        // call returns at once while the subagent carries on in the background.
        bars.filter(bar => bar.kind === 'subagent').forEach(bar => {
            const inside = bars.filter(other => other.host === bar.key).map(other => other.end)
                .concat(ticks.filter(t => t.host === bar.key).map(t => t.at));
            if (inside.length) bar.end = Math.max(bar.end, ...inside);
        });

        const modelEvent = events.find(event => event && event.data && typeof event.data.model === 'string' && event.data.model);
        const usage = result.usage || {};
        const tokens = tokensOf(usage);
        const harness = String(run.harness || task.harness || 'harness');
        tokens.partial = PARTIAL_USAGE.has(harness) && !(result.usage && result.usage.models) && (tokens.input !== null || tokens.output !== null);
        const verification = result.verification || {};
        const verified = verification.command || verification.exit_code !== undefined || verification.status
            ? (result.verified ? 'passed' : 'failed')
            : (result.verified ? 'passed' : 'not run');
        const agentId = String(task.agent_id || run.agent_id || '');
        // Only a MemAgent lane runs the agent: a comparison copies agent_id onto every lane.
        const agentName = harness === 'memagent' && agentId && window.MemorizzAgentNames ? window.MemorizzAgentNames[agentId] : '';
        return {
            runId: String(run.run_id || task.run_id || ''),
            harness,
            label: agentName ? harness + ' · ' + agentName : harness + ((task.metadata || {}).source === 'plugin' ? ' · plugin session' : ''),
            cache,
            model: String(usage.model || task.model || (modelEvent ? modelEvent.data.model : '') || ''),
            task: String(task.task || ''),
            status,
            active,
            finished: finished || !active,
            durationMs,
            waitedMs: Math.max(0, origin * 1000),
            costUsd: num(result.cost_usd),
            costEstimated: usage.cost_basis === 'list_rate_estimate',
            tokens,
            counts,
            hiddenReasoning,
            actions: ACTION_KINDS.reduce((sum, kind) => sum + counts[kind], 0),
            answer: String(result.final_response || ''),
            error: String(result.error || ''),
            verified,
            endAt: Math.max(endAt, ...bars.map(bar => bar.end), 0),
            bars,
            ticks,
        };
    };

    /* Which run is fastest, cheapest, uses the fewest actions or tokens — among
       runs that succeeded, and only when at least two of them can be compared.
       These are facts about effort, not a verdict on the answers. */
    const highlights = (summaries) => {
        const done = summaries.filter(s => s.status === 'succeeded');
        const marks = {};
        const pick = (field, label, value) => {
            const rows = done.map(s => ({s, v: value(s)})).filter(row => row.v !== null && row.v !== undefined);
            if (rows.length < 2) return;
            const best = Math.min(...rows.map(row => row.v));
            if (rows.every(row => row.v === best)) return;
            rows.filter(row => row.v === best).forEach(row => { (marks[row.s.runId] = marks[row.s.runId] || {})[field] = label; });
        };
        pick('duration', 'Fastest', s => s.durationMs);
        pick('cost', 'Cheapest', s => s.costUsd);
        pick('actions', 'Fewest actions', s => s.actions);
        pick('tokens', 'Fewest tokens', s => (s.tokens.partial || (s.tokens.input === null && s.tokens.output === null) ? null : (s.tokens.input || 0) + (s.tokens.output || 0)));
        return marks;
    };

    // Bars that overlap in time go on separate rows (parallel subagents, say).
    const pack = (bars) => {
        const rows = [];
        bars.slice().sort((a, b) => a.start - b.start || b.end - a.end).forEach(bar => {
            let row = rows.findIndex(end => end <= bar.start + 1e-6);
            if (row === -1) { rows.push(bar.end); row = rows.length - 1; } else rows[row] = bar.end;
            bar.row = row;
        });
        return rows.length;
    };

    const niceStep = (span) => {
        const target = span / 6;
        for (const step of [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600]) if (step >= target) return step;
        return 3600;
    };
    const axisLabel = (seconds) => seconds < 60 ? seconds + 's' : Math.floor(seconds / 60) + 'm' + (seconds % 60 ? String(seconds % 60).padStart(2, '0') + 's' : '');

    // ------------------------------------------------------------------
    // Drawing
    // ------------------------------------------------------------------
    // The page's status dots: fleet-health--healthy, --failing and so on.
    const HEALTH = {succeeded: 'healthy', failed: 'failing', verification_failed: 'failing', budget_exceeded: 'failing', interrupted: 'failing', canceled: 'idle', pending_approval: 'degraded', queued: 'active', running: 'active'};
    const fmtUsd = (value) => value === null ? '—' : '$' + (value < 0.01 && value > 0 ? value.toFixed(4) : value.toFixed(2));

    const renderFacts = (summaries) => {
        const {fmtDuration, fmtCount, humanize} = T();
        const marks = highlights(summaries);
        const badge = (s, field) => {
            const label = (marks[s.runId] || {})[field];
            return label ? ' <span class="hxcmp-best">' + esc(label) + '</span>' : '';
        };
        const tokenCell = (s) => {
            const {input, output, cached} = s.tokens;
            if (input === null && output === null) return '<span class="hxcmp-muted">not reported</span>';
            return esc((input === null ? '—' : fmtCount(input)) + ' in · ' + (output === null ? '—' : fmtCount(output)) + ' out')
                + (cached ? '<small>' + esc(fmtCount(cached) + ' cached') + '</small>' : '')
                + (s.tokens.partial ? '<small>last model call only, as the harness reports it</small>' : '')
                + badge(s, 'tokens');
        };
        const rows = [
            ['Status', s => '<span class="fleet-health fleet-health--' + (HEALTH[s.status] || 'idle') + '">' + esc(humanize(s.status)) + '</span>'],
            ['Model', s => s.model ? '<span class="mono">' + esc(s.model) + '</span>' : '<span class="hxcmp-muted">harness default</span>'],
            ['Time', s => esc(fmtDuration(s.durationMs)) + (s.waitedMs >= 1000 ? '<small>' + esc('after ' + fmtDuration(s.waitedMs) + ' waiting for approval') + '</small>' : '') + badge(s, 'duration')],
            ['Cost', s => (s.costEstimated ? '≈' : '') + esc(fmtUsd(s.costUsd)) + (s.costUsd === null ? ' <span class="hxcmp-muted">not reported</span>' : '') + badge(s, 'cost')],
            ['Tokens', tokenCell],
            ['Actions', s => esc(String(s.actions)) + '<small>' + esc(ACTION_KINDS.filter(k => s.counts[k]).map(k => s.counts[k] + ' ' + BAR_KINDS[k].toLowerCase() + (s.counts[k] === 1 ? '' : 's')).join(' · ') || 'none') + '</small>' + badge(s, 'actions')],
            ['Tool cache', s => !s.cache || !s.cache.seen
                ? '<span class="hxcmp-muted">off or not used</span>'
                : esc(s.cache.hits + ' of ' + s.counts.tool + ' tool call' + (s.counts.tool === 1 ? '' : 's') + ' from cache')
                    + '<small>' + esc(s.cache.hits ? 'the original calls took ' + fmtDuration(s.cache.savedMs) : 'results kept for the next run') + '</small>'],
            ['Reasoning', s => s.counts.thinking ? esc(String(s.counts.thinking)) + (s.hiddenReasoning === s.counts.thinking ? '<small>text not shared by the harness</small>' : '') : '<span class="hxcmp-muted">none reported</span>'],
            ['Messages', s => esc(String(s.counts.message))],
            ['Verification', s => s.verified === 'passed' ? '<span class="is-good">Passed</span>' : s.verified === 'failed' ? '<span class="is-bad">Failed</span>' : '<span class="hxcmp-muted">Not run</span>'],
        ];
        const head = '<tr><th scope="col"><span class="visually-hidden">Fact</span></th>' + summaries.map((s, i) => '<th scope="col"><span class="hxcmp-key hxcmp-key--' + i + '">' + (i + 1) + '</span> ' + esc(s.label) + '</th>').join('') + '</tr>';
        const body = rows.map(([label, cell]) => '<tr><th scope="row">' + esc(label) + '</th>' + summaries.map(s => '<td>' + cell(s) + '</td>').join('') + '</tr>').join('');
        const sameTask = new Set(summaries.map(s => s.task.trim())).size <= 1;
        return (sameTask ? '' : '<p class="hxcmp-note">These runs were given different tasks, so compare them with care.</p>')
            + '<div class="hxcmp-table-wrap"><table class="hxcmp-facts"><thead>' + head + '</thead><tbody>' + body + '</tbody></table></div>'
            + '<p class="hxcmp-note">Highlights compare effort among runs that succeeded. Which answer is right is for you to judge below.</p>';
    };

    const renderTimeline = (summaries) => {
        const {fmtDuration} = T();
        const span = Math.max(1, ...summaries.map(s => s.endAt));
        const step = niceStep(span);
        const max = Math.ceil(span / step) * step;
        const pos = (seconds) => (100 * Math.min(Math.max(seconds, 0), max) / max).toFixed(3) + '%';
        const width = (a, b) => (100 * Math.max(b - a, 0) / max).toFixed(3) + '%';
        const axis = [];
        for (let t = 0; t <= max + 1e-9; t += step) axis.push('<span class="hxcmp-axis-tick" style="left:' + pos(t) + '">' + esc(axisLabel(Math.round(t * 10) / 10)) + '</span>');
        const lanes = summaries.map((s, i) => {
            const main = s.bars.filter(bar => bar.depth === 0);
            const inner = s.bars.filter(bar => bar.depth > 0);
            const mainRows = pack(main);
            const innerRows = pack(inner);
            const bar = (b, offset) => {
                const label = b.title + (b.within ? ' — in ' + b.within : '') + ' · ' + fmtDuration((b.end - b.start) * 1000) + (b.running ? ' · running' : '') + (b.summary ? '\n' + b.summary : '');
                return '<span class="hxcmp-bar hxcmp-bar--' + b.kind + (b.depth ? ' is-inner' : '') + (b.running ? ' is-running' : '') + (b.error ? ' is-error' : '') + (b.cached ? ' is-cached' : '') + '" style="left:' + pos(b.start) + ';width:' + width(b.start, b.end) + ';--row:' + (b.row + offset) + '" title="' + esc(label) + '"></span>';
            };
            const tick = (k) => '<span class="hxcmp-tick hxcmp-tick--' + k.kind + '" style="left:' + pos(k.at) + '" title="' + esc(k.title + (k.within ? ' — in ' + k.within : '') + ' · +' + k.at.toFixed(1) + 's' + (k.summary ? '\n' + k.summary : '')) + '"></span>';
            const rows = Math.max(1, mainRows + innerRows);
            const endLabel = s.active ? 'running' : (s.status === 'succeeded' ? 'done' : s.status.replace(/_/g, ' '));
            return '<div class="hxcmp-lane" style="--rows:' + rows + ';--inner-from:' + mainRows + '">'
                + '<div class="hxcmp-lane-label"><span class="hxcmp-key hxcmp-key--' + i + '">' + (i + 1) + '</span><strong>' + esc(s.label) + '</strong><small>' + esc(fmtDuration(s.durationMs)) + '</small></div>'
                + '<div class="hxcmp-lane-track" role="img" aria-label="' + esc(s.label + ': ' + s.bars.length + ' timed steps over ' + fmtDuration(s.durationMs)) + '">'
                + '<div class="hxcmp-ticks">' + s.ticks.map(tick).join('') + '</div>'
                + '<div class="hxcmp-bars">' + main.map(b => bar(b, 0)).join('') + inner.map(b => bar(b, mainRows)).join('') + '</div>'
                + '<span class="hxcmp-end' + (s.active ? ' is-running' : s.status === 'succeeded' ? '' : ' is-bad') + '" style="left:' + pos(s.endAt) + '" title="' + esc(endLabel + ' at +' + s.endAt.toFixed(1) + 's') + '"><span>' + esc(endLabel) + '</span></span>'
                + '</div></div>';
        }).join('');
        const legend = '<ul class="hxcmp-legend" aria-label="Legend">'
            + '<li><i class="hxcmp-swatch hxcmp-bar--model"></i>Model call</li>'
            + '<li><i class="hxcmp-swatch hxcmp-bar--tool"></i>Tool call</li>'
            + (summaries.some(s => s.cache && s.cache.hits) ? '<li><i class="hxcmp-swatch hxcmp-bar--tool is-cached"></i>Tool call from cache</li>' : '')
            + '<li><i class="hxcmp-swatch hxcmp-bar--command"></i>Command</li>'
            + '<li><i class="hxcmp-swatch hxcmp-bar--subagent"></i>Subagent, its steps in thinner bars below</li>'
            + '<li><i class="hxcmp-swatch-tick hxcmp-tick--thinking"></i>Reasoning</li>'
            + '<li><i class="hxcmp-swatch-tick hxcmp-tick--message"></i>Message</li>'
            + '<li><i class="hxcmp-swatch-end"></i>Finished</li>'
            + '</ul>';
        return legend + '<p class="hxcmp-note">Each lane starts when its harness started, after any approval wait, so lengths compare directly. Hover a bar for its details.</p>'
            + '<div class="hxcmp-scroll"><div class="hxcmp-timeline"><div class="hxcmp-axis"><span class="hxcmp-lane-label"></span><div class="hxcmp-axis-track">' + axis.join('') + '</div></div>' + lanes + '</div></div>';
    };

    const answerHtml = (text) => {
        if (!text) return '<p class="hxcmp-muted">No answer yet.</p>';
        if (window.MemorizzMarkdown && typeof window.MemorizzMarkdown.render === 'function') return window.MemorizzMarkdown.render(text);
        return '<p class="hxcmp-plain">' + esc(text) + '</p>';
    };

    const renderColumnShells = (summaries) => summaries.map((s, i) => '<article class="hxcmp-col" data-run="' + esc(s.runId) + '">'
        + '<header><span class="hxcmp-key hxcmp-key--' + i + '">' + (i + 1) + '</span><strong>' + esc(s.label) + '</strong>' + (s.model ? '<span class="mono hxcmp-model">' + esc(s.model) + '</span>' : '') + '<span class="mono hxcmp-muted">' + esc(s.runId.slice(0, 8)) + '</span></header>'
        + '<section class="hxcmp-answer" aria-label="Answer"></section>'
        + '<h4>Trajectory</h4><div class="hx-traj hxcmp-traj" data-active="false"></div></article>').join('');

    // ------------------------------------------------------------------
    // The dialog
    // ------------------------------------------------------------------
    const getJSON = async (url) => {
        const response = await fetch(url, {headers: {Accept: 'application/json'}, credentials: 'same-origin'});
        const body = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(body.detail || body.error || ('HTTP ' + response.status));
        return body;
    };

    const open = (runIds, options = {}) => {
        if (!T()) throw new Error('harness-trajectory.js must load before harness-compare.js');
        const ids = [...new Set((runIds || []).map(String).filter(Boolean))];
        if (ids.length < 2) throw new Error('Pick at least two runs to compare');
        const shown = ids.slice(0, MAX_RUNS);
        const states = shown.map(id => ({id, run: null, events: [], after: 0, loaded: false, pending: null}));
        let timer = null;
        let closed = false;

        const dialog = document.createElement('dialog');
        dialog.className = 'hxcmp';
        dialog.setAttribute('aria-labelledby', 'hxcmp-title');
        dialog.innerHTML = '<header class="hxcmp-head"><div><h2 id="hxcmp-title">Compare ' + shown.length + ' runs</h2><p class="hxcmp-task"></p></div>'
            + '<button type="button" class="btn btn-sm" data-hxcmp-close>Close</button></header>'
            + (ids.length > MAX_RUNS ? '<p class="hxcmp-note">Showing the first ' + MAX_RUNS + ' of ' + ids.length + ' runs.</p>' : '')
            + '<p class="hxcmp-status hxcmp-muted" aria-live="polite">Loading…</p>'
            + '<section class="hxcmp-section" aria-labelledby="hxcmp-facts-title"><h3 id="hxcmp-facts-title">At a glance</h3><div class="hxcmp-facts-slot"></div></section>'
            + '<section class="hxcmp-section" aria-labelledby="hxcmp-time-title"><h3 id="hxcmp-time-title">Timeline</h3><div class="hxcmp-timeline-slot"></div></section>'
            + '<section class="hxcmp-section" aria-labelledby="hxcmp-cols-title"><h3 id="hxcmp-cols-title">Answers and steps</h3><div class="hxcmp-cols" style="--cols:' + shown.length + '"></div></section>';
        document.body.append(dialog);
        const $ = (selector) => dialog.querySelector(selector);

        const fetchEvents = async (state) => {
            if (state.pending) return state.pending;
            state.pending = (async () => {
                try {
                    for (let page = 0; page < 20; page += 1) {
                        const body = await getJSON('/api/harness-runs/' + encodeURIComponent(state.id) + '/events?after=' + state.after + '&limit=1000');
                        const fresh = (body.events || []).filter(event => event.sequence > state.after);
                        if (!fresh.length) break;
                        state.events.push(...fresh);
                        state.after = fresh[fresh.length - 1].sequence;
                        if (fresh.length < 1000) break;
                    }
                    state.loaded = true;
                } finally { state.pending = null; }
            })();
            return state.pending;
        };
        const refresh = async (all) => {
            await Promise.all(states.map(async (state) => {
                if (!all && state.run && !ACTIVE.has(String(state.run.status))) return;
                const body = await getJSON('/api/harness-runs/' + encodeURIComponent(state.id));
                state.run = body.run || state.run;
                await fetchEvents(state);
            }));
        };

        let columnsBuilt = false;
        const draw = () => {
            const now = Date.now();
            const summaries = states.map(state => summarize(state.run || {run_id: state.id}, state.events, now));
            const task = options.title || (summaries[0] && summaries[0].task) || '';
            $('.hxcmp-task').textContent = task;
            $('.hxcmp-facts-slot').innerHTML = renderFacts(summaries);
            const scroller = $('.hxcmp-timeline-slot .hxcmp-scroll');
            const left = scroller ? scroller.scrollLeft : 0;
            $('.hxcmp-timeline-slot').innerHTML = renderTimeline(summaries);
            const again = $('.hxcmp-timeline-slot .hxcmp-scroll');
            if (again) again.scrollLeft = left;
            if (!columnsBuilt) { $('.hxcmp-cols').innerHTML = renderColumnShells(summaries); columnsBuilt = true; }
            summaries.forEach((s, i) => {
                const col = dialog.querySelectorAll('.hxcmp-col')[i];
                if (!col) return;
                const answer = col.querySelector('.hxcmp-answer');
                const signature = s.status + ':' + s.answer.length + ':' + s.error.length;
                if (answer.dataset.signature !== signature) {
                    answer.dataset.signature = signature;
                    answer.innerHTML = (s.error && s.status !== 'succeeded' ? '<p class="hxcmp-error">' + esc(s.error) + '</p>' : '') + answerHtml(s.answer);
                }
                const traj = col.querySelector('.hxcmp-traj');
                traj.dataset.active = String(s.active);
                T().render(traj, states[i]);
            });
            const live = summaries.some(s => s.active);
            $('.hxcmp-status').textContent = live ? 'Updating while runs are active.' : '';
            return live;
        };

        const tick = async () => {
            if (closed) return;
            try { await refresh(false); } catch (error) { $('.hxcmp-status').textContent = 'Could not refresh: ' + error.message; }
            if (closed) return;
            const live = draw();
            // One more pass after the last run finishes picks up its final events.
            if (live) timer = setTimeout(tick, 2000);
            else if (!tick.settled) { tick.settled = true; timer = setTimeout(tick, 1000); }
        };

        const close = () => {
            if (closed) return;
            closed = true;
            clearTimeout(timer);
            if (dialog.open) dialog.close();
            dialog.remove();
        };
        dialog.addEventListener('close', close);
        dialog.addEventListener('click', (event) => {
            if (event.target.closest('[data-hxcmp-close]')) close();
            else if (event.target === dialog) close();  // a click on the backdrop
        });
        dialog.showModal();
        (async () => {
            try {
                await refresh(true);
                if (closed) return;
                const live = draw();
                if (live) timer = setTimeout(tick, 2000);
            } catch (error) {
                $('.hxcmp-status').textContent = 'Could not load the runs: ' + error.message;
            }
        })();
        return {close, dialog};
    };

    window.MemorizzCompare = {open, summarize, highlights, pack, tokensOf};
})();

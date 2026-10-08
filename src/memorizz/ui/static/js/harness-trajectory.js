/* Harness trajectories: a run's events as the steps the harness took.
   Shared by the harnesses page and the harness chat.
   MemorizzTrajectory.render(container, {events, loaded}) draws the steps into a
   container whose data-active="true" marks a run still in progress. */
(function () {
    // base.html's escapeHtml; the inline copy keeps the module working standalone.
    const esc = typeof escapeHtml === 'function' ? escapeHtml : (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
    const fmtDuration = (ms) => {
        if (ms == null || !isFinite(ms)) return '—';
        if (ms < 1000) return Math.round(ms) + ' ms';
        const s = ms / 1000;
        if (s < 60) return s.toFixed(1) + ' s';
        return Math.floor(s / 60) + 'm ' + String(Math.floor(s % 60)).padStart(2, '0') + 's';
    };
    const fmtCount = (value) => {
        const n = Number(value || 0);
        if (n >= 1e6) return (n / 1e6).toFixed(1).replace(/\.0$/, '') + 'M';
        if (n >= 1e3) return (n / 1e3).toFixed(1).replace(/\.0$/, '') + 'k';
        return String(Math.round(n));
    };
    const humanize = (value) => { const text = String(value || '').replace(/_/g, ' '); return text.charAt(0).toUpperCase() + text.slice(1); };
    const toolName = (name) => {
        const parts = String(name || 'tool').split('__');
        return parts.length >= 3 && parts[0] === 'mcp' ? parts[1] + ' › ' + parts.slice(2).join('__') : String(name || 'tool');
    };
    const excerpt = (text, limit) => { const value = String(text ?? ''); return value.length > limit ? value.slice(0, limit) + '…' : value; };
    const pretty = (value) => { if (typeof value === 'string') return value; try { return JSON.stringify(value, null, 2); } catch (_) { return String(value); } };
    const buildSteps = (events) => {
        const steps = [];
        const items = new Map();
        const start = events.length ? Date.parse(events[0].timestamp) : 0;
        const seconds = (event) => (Date.parse(event.timestamp) - start) / 1000;
        let finished = false;
        events.forEach(event => {
            const data = event.data || {};
            const t = seconds(event);
            const add = (step) => { step.at = t; step.key = String(event.sequence); steps.push(step); return step; };
            switch (event.type) {
            case 'status':
                if (data.phase === 'model_call') {
                    const number = steps.filter(step => step.kind === 'model').length + 1;
                    const step = add({kind: 'model', title: 'Model call ' + number, summary: data.stage ? humanize(data.stage) : 'Waiting for the model', running: true});
                    if (data.id) items.set('model:' + data.id, step);
                } else if (data.phase === 'model_result') {
                    const step = data.id && items.get('model:' + data.id);
                    if (step) {
                        step.running = false;
                        step.end = t;
                        step.error = data.success === false;
                        step.summary = data.success === false ? 'Failed' + (data.error_code ? ' · ' + data.error_code : '') : (step.summary === 'Waiting for the model' ? 'Answered' : step.summary);
                    }
                } else if (data.status) {
                    const facts = [data.model, data.web_access ? 'web: ' + data.web_access : '', data.notice || data.reason].filter(Boolean).join(' · ');
                    const started = data.status === 'running' && steps.find(step => step.kind === 'status' && step.title === 'Started');
                    // A harness restating that it started only adds what it runs on.
                    if (started) { if (facts) started.summary = facts; break; }
                    add({kind: 'status', title: data.status === 'running' ? 'Started' : humanize(data.status), summary: facts});
                } else if (data.memory_context && data.memory_context.source === 'memorizz_plugin') {
                    // What the MemoRizz plugin's hook added to the session.
                    const ctx = data.memory_context;
                    const when = ctx.hook === 'SessionStart' ? 'at session start' : 'with the prompt';
                    add({kind: 'memory', title: 'MemoRizz memory', summary: 'The plugin added project memory ' + when + ' · ~' + fmtCount(ctx.token_estimate) + ' tokens', detail: ctx.text || ''});
                } else if (data.memory_context) {
                    const ctx = data.memory_context;
                    add({kind: 'memory', title: 'Memory context', summary: (ctx.source_ids || []).length + ' memories · ~' + fmtCount(ctx.token_estimate) + ' tokens' + (ctx.truncated ? ' · truncated' : '')});
                } else if (data.session_id || data.thread_id) {
                    const summary = [data.model, excerpt(data.session_id || data.thread_id, 13)].filter(Boolean).join(' · ');
                    if (!steps.some(step => step.kind === 'session' && step.summary === summary)) add({kind: 'session', title: 'Session', summary});
                }
                break;
            case 'log':
                // Errors a harness only prints to stderr (a failed subagent spawn, say).
                if (data.stream === 'stderr' && /\bERROR\b/.test(String(data.text || ''))) {
                    const text = String(data.text || '').replace(/^\S+\s+ERROR\s+\S+:\s*/, '');
                    add({kind: 'error', title: 'Harness error', summary: excerpt(text, 280), detail: String(data.text || ''), error: true});
                    break;
                }
                if (data.subtype === 'thinking_tokens' || data.type === 'reasoning') {
                    const last = steps[steps.length - 1];
                    if (last && last.kind === 'thinking') { last.count += 1; last.end = t; } else add({kind: 'thinking', title: 'Thinking', count: 1});
                }
                break;
            case 'reasoning': {
                const text = String(data.text || '');
                if (!text) {
                    // The harness reasoned but kept the text to itself: one step per run of them.
                    const last = steps[steps.length - 1];
                    if (last && last.kind === 'thinking' && last.hidden) { last.count = (last.count || 1) + 1; last.end = t; break; }
                    add({kind: 'thinking', title: 'Reasoning', summary: 'The harness reasoned here but did not share the text.', hidden: true, count: 1, parent: data.parent_id});
                    break;
                }
                add({kind: 'thinking', title: 'Reasoning', summary: excerpt(text.replace(/\s+/g, ' '), 280), detail: text.length > 280 ? text : '', parent: data.parent_id});
                break;
            }
            case 'message':
                add({kind: 'message', title: data.role === 'user' ? 'User' : 'Assistant', summary: excerpt(data.text, 280), detail: String(data.text || '').length > 280 ? data.text : '', parent: data.parent_id});
                break;
            case 'command': {
                const id = data.id;
                const existing = id && items.get('cmd:' + id);
                if (existing) {
                    existing.end = t;
                    existing.exit = data.exit_code;
                    existing.running = data.status === 'in_progress';
                    if (data.aggregated_output) existing.detail = '$ ' + data.command + '\n\n' + excerpt(data.aggregated_output, 4000);
                } else {
                    const step = add({kind: 'command', title: 'Command', summary: '$ ' + excerpt(String(data.command || '').replace(/^\/bin\/\w+ -lc /, ''), 180), detail: '$ ' + data.command + (data.aggregated_output ? '\n\n' + excerpt(data.aggregated_output, 4000) : ''), exit: data.exit_code, running: data.status === 'in_progress'});
                    if (id) items.set('cmd:' + id, step);
                }
                break;
            }
            case 'tool_call': {
                const query = data.query || (data.action && data.action.query);
                const input = data.input ?? data.arguments ?? data.args ?? (query ? {query} : (data.type === 'web_search' ? 'query not shared by the harness' : undefined));
                const done = data.status && data.status !== 'in_progress' && data.status !== 'started';
                const existing = data.id && items.get('tool:' + data.id);
                if (existing) {
                    if (data.harness_run_id) { existing.runId = data.harness_run_id; existing.harness = data.harness; }
                    // The same item reported again as it progresses (Codex).
                    if (input && Object.keys(input).length) { existing.summary = excerpt(typeof input === 'string' ? input : JSON.stringify(input), 160); existing.detail = pretty(input); }
                    if (done) { existing.running = false; existing.end = t; existing.error = data.status === 'failed'; }
                    break;
                }
                const summary = input && (typeof input === 'string' || Object.keys(input).length) ? excerpt(typeof input === 'string' ? input : JSON.stringify(input), 160) : 'no arguments';
                // Codex's sub-agent calls: spawning one, messaging it, waiting for them.
                const COLLAB = {spawn_agent: 'Subagent', send_input: 'Message to subagent', wait: 'Waiting for subagents', close_agent: 'Closed subagent'};
                const label = data.subagent
                    ? (data.type === 'collab_tool_call' && COLLAB[data.tool] && data.tool !== 'spawn_agent'
                        ? COLLAB[data.tool] + ((data.receiver_thread_ids || []).length ? ' · ' + data.receiver_thread_ids.length : '')
                        : 'Subagent · ' + ((input && (input.description || input.subagent_type)) || (data.type === 'collab_tool_call' ? 'spawned' : toolName(data.name))))
                    : 'Tool · ' + toolName(data.name || data.tool || data.server || data.type);
                // Only a started subagent counts as one; waiting on or messaging them is a step.
                const spawned = data.subagent && !(data.type === 'collab_tool_call' && data.tool !== 'spawn_agent');
                const prompt = input && typeof input === 'object' ? input.prompt : '';
                const step = add({
                    kind: spawned ? 'subagent' : 'tool',
                    title: label,
                    summary: data.subagent ? (prompt ? excerpt(String(prompt).replace(/\s+/g, ' '), 200) : '') : summary,
                    detail: input && !(data.subagent && !prompt) ? pretty(input) : '',
                    running: !done, toolId: data.id, parent: data.parent_id,
                });
                if (data.id) items.set('tool:' + data.id, step);
                break;
            }
            case 'tool_result': {
                const blocks = ((data.message || {}).content) || [data];
                blocks.filter(block => block && (block.tool_use_id || block.id)).forEach(block => {
                    const step = items.get('tool:' + (block.tool_use_id || block.id));
                    const output = typeof block.content === 'string' ? block.content : pretty(block.content);
                    if (step) {
                        step.running = false;
                        step.end = t;
                        step.error = !!block.is_error;
                        step.cache = block.cache || data.cache || null;
                        step.detail = (step.detail ? 'Input\n' + step.detail + '\n\n' : '') + 'Result\n' + excerpt(output, 4000);
                        if (block.is_error) step.summary = excerpt(output, 200);
                        else if (data.status === 'approval_required') step.summary = 'Waiting for approval';
                    }
                });
                break;
            }
            case 'file_change': {
                const paths = (data.changes || []).map(change => change.path || change.file_path).filter(Boolean);
                add({kind: 'file', title: 'File change', summary: paths.length ? paths.join(', ') : (data.path || 'files changed'), detail: pretty(data)});
                break;
            }
            case 'usage': {
                const cached = data.cached_input_tokens ?? data.cache_read_input_tokens;
                add({kind: 'usage', title: 'Usage', summary: fmtCount(data.input_tokens) + ' in' + (cached ? ' (' + fmtCount(cached) + ' cached)' : '') + ' · ' + fmtCount(data.output_tokens) + ' out' + (data.model ? ' · ' + data.model : '')});
                break;
            }
            case 'error':
                add({kind: 'error', title: 'Error', summary: excerpt(data.error || data.message || pretty(data), 280), error: true});
                break;
            case 'approval':
                add({kind: 'approval', title: 'Approval', summary: humanize(data.status || data.decision || 'requested')});
                break;
            case 'verification':
                add({kind: 'verify', title: 'Verification', summary: data.ok === false || data.passed === false ? 'Failed' : (data.ok || data.passed ? 'Passed' : humanize(data.status || 'ran')), detail: pretty(data), error: data.ok === false || data.passed === false});
                break;
            case 'complete':
                if (data.status) {
                    finished = true;
                    add({kind: 'complete', title: humanize(data.status), summary: [data.latency_ms != null ? fmtDuration(data.latency_ms) : '', data.cost_usd != null ? '$' + Number(data.cost_usd).toFixed(4) : '', data.error_code ? humanize(data.error_code) : ''].filter(Boolean).join(' · '), error: data.ok === false});
                } else if (data.usage) {
                    const usage = data.usage;
                    add({kind: 'usage', title: 'Usage', summary: fmtCount((usage.input_tokens || 0) + (usage.cache_read_input_tokens || 0) + (usage.cache_creation_input_tokens || 0)) + ' in (' + fmtCount(usage.cache_read_input_tokens) + ' cached) · ' + fmtCount(usage.output_tokens) + ' out'});
                }
                break;
            default:
                break;
            }
        });
        return {steps, finished};
    };
    const renderTrajectory = (container, state) => {
        if (!container) return;
        const active = container.dataset.active === 'true';
        const signature = state.events.length + ':' + active + ':' + state.loaded;
        if (container.dataset.signature === signature) return;
        container.dataset.signature = signature;
        if (!state.loaded && !state.events.length) { container.innerHTML = '<p class="hx-muted">Loading steps…</p>'; return; }
        const {steps, finished} = buildSteps(state.events);
        if (!steps.length) { container.innerHTML = '<p class="hx-muted">' + (active ? 'Waiting for the first step…' : 'No steps were recorded.') + '</p>'; return; }
        const open = new Set([...container.querySelectorAll('details[open]')].map(el => el.dataset.step));
        const counts = steps.reduce((acc, step) => { acc[step.kind] = (acc[step.kind] || 0) + 1; return acc; }, {});
        const parts = [['subagent', 'subagent'], ['model', 'model call'], ['thinking', 'reasoning step'], ['tool', 'tool call'], ['command', 'command'], ['file', 'file change'], ['message', 'message']].filter(([kind]) => counts[kind]).map(([kind, label]) => counts[kind] + ' ' + label + (counts[kind] === 1 ? '' : 's'));
        const fromCache = steps.filter(step => step.kind === 'tool' && step.cache && step.cache.status === 'hit').length;
        if (fromCache) parts.push(fromCache + ' from cache');
        const last = steps[steps.length - 1];
        const hosts = new Map(steps.filter(step => step.toolId).map(step => [step.toolId, step]));
        const roots = [];
        steps.forEach(step => {
            step.children = [];
            const host = step.parent && hosts.get(step.parent);
            if (host && host !== step) host.children.push(step); else roots.push(step);
        });
        // A subagent's launching call can return at once while it keeps
        // working: it lasts until its last nested step.
        const settle = (step) => {
            step.children.forEach(settle);
            step.children.forEach(child => {
                const until = child.end != null ? child.end : child.at;
                if (step.end == null || until > step.end) step.end = until;
                if (child.running) step.running = true;
            });
        };
        roots.forEach(settle);
        const item = (step) => {
            const running = step.running || (active && !finished && step === last && step.end == null);
            const took = step.end != null ? fmtDuration((step.end - step.at) * 1000) : '';
            const cachedHit = step.cache && step.cache.status === 'hit';
            // saved_ms is how long the call took when its result was stored.
            const cacheNote = cachedHit ? 'from cache · the original call took ' + fmtDuration(Number(step.cache.saved_ms) || 0) : '';
            const meta = [step.count > 1 ? step.count + ' updates' : '', step.exit != null ? 'exit ' + step.exit : '', step.children.length ? step.children.length + ' inside' : '', cacheNote, took].filter(Boolean).join(' · ');
            const detail = step.detail ? '<details data-step="' + esc(step.key) + '"' + (open.has(step.key) ? ' open' : '') + '><summary>Details</summary><pre>' + esc(step.detail) + '</pre></details>' : '';
            const inner = step.children.length ? '<ol class="hx-traj-list hx-traj-children">' + step.children.map(item).join('') + '</ol>' : '';
            // A harness delegate's own run, one click away.
            const runLink = step.runId ? '<button type="button" class="hx-traj-run" data-select-run="' + esc(step.runId) + '">Open its ' + esc(step.harness || 'harness') + ' run</button>' : '';
            return '<li class="hx-traj-step hx-traj--' + step.kind + (step.error || (step.exit != null && Number(step.exit) !== 0) ? ' is-error' : '') + (running ? ' is-running' : '') + (cachedHit ? ' is-cached' : '') + '">'
                + '<span class="hx-traj-icon" aria-hidden="true"></span>'
                + '<div class="hx-traj-body"><div class="hx-traj-head"><strong>' + esc(step.title) + '</strong>' + (meta ? '<span class="hx-traj-meta">' + esc(meta) + '</span>' : '') + '<time>+' + step.at.toFixed(1) + 's</time></div>'
                + (step.summary ? '<p class="hx-traj-summary">' + esc(step.summary) + '</p>' : '') + runLink + detail + inner + '</div></li>';
        };
        container.innerHTML = '<p class="hx-traj-counts">' + esc(steps.length + ' steps' + (parts.length ? ' · ' + parts.join(' · ') : '')) + '</p><ol class="hx-traj-list">' + roots.map(item).join('') + '</ol>';
    };
    window.MemorizzTrajectory = {esc, fmtDuration, fmtCount, humanize, buildSteps, render: renderTrajectory};
})();

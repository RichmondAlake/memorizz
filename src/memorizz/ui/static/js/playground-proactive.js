// The agent's proactive side in the playground: a level control, and a card
// above the chat box when the agent has a next step to offer or has taken one.
(function() {
    'use strict';
    const dock = document.getElementById('pg-proactive');
    if (!dock || typeof AGENT_ID === 'undefined') return;

    const readOnly = Boolean(document.body && document.body.dataset.readOnly === 'true');
    const cards = dock.querySelector('.pg-proactive-cards');
    const levelEl = dock.querySelector('#pg-proactive-level');
    const predictEl = dock.querySelector('#pg-proactive-predict');
    const scanEl = dock.querySelector('#pg-proactive-scan');
    const noteEl = dock.querySelector('.pg-proactive-note');
    const chipEl = document.getElementById('pg-proactive-chip');   // the badge beside the agent's name
    const factEl = document.getElementById('pg-proactive-fact');   // and its row in the Agent panel
    const base = `/api/agents/${encodeURIComponent(AGENT_ID)}/proactive`;
    const approvalsBase = `/api/agents/${encodeURIComponent(AGENT_ID)}/approvals`;

    const QUIET_MS = 6000;      // leave someone who is typing alone
    const POLL_MS = 15000;      // how often an idle, visible page asks
    const LEVEL_NOTES = {
        off: 'Off. The agent answers when asked.',
        low: 'Suggests a next step. Nothing runs until you say so.',
        medium: 'Suggests, and does steps that change nothing by itself. Anything that changes something waits for you.',
        high: 'Does its steps by itself. A tool that needs approval still waits for you.'
    };
    const BADGE_NOTES = {
        low: 'This agent suggests a next step. Nothing runs until you say so.',
        medium: 'This agent suggests, and takes steps that change nothing by itself. Anything that changes something waits for you.',
        high: 'This agent takes its steps by itself. A tool that needs approval still waits for you.',
        ultra: 'This agent takes its steps by itself and predicts steps from your habits (experimental). A tool that needs approval still waits for you.'
    };

    let state = null;
    let shown = null;           // the suggestion on the card
    let lastInput = Date.now();
    let nextCheck = Date.now() + 4000;
    let checking = false;
    let trigger = 'view';

    function memoryId() {
        const el = document.getElementById('pg-memory-id');
        return el ? el.value.trim() : '';
    }

    function busy() {
        try { return typeof isStreaming !== 'undefined' && isStreaming; } catch (_) { return false; }
    }

    function plainPreview(text) {
        return String(text || '')
            .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '$1')
            .replace(/^\s*[-*]\s*\[[ xX]\]\s*/gm, '• ')
            .replace(/^#+\s*/gm, '')
            .replace(/\*\*|__|`|~~/g, '')
            .replace(/(^|\s)\*([^*\n]+)\*(?=\s|[.,;:!?]|$)/g, '$1$2');
    }

    async function call(path, body, method) {
        const response = await fetch(base + path, {
            method: method || (body === undefined ? 'GET' : 'POST'),
            headers: body === undefined ? {} : {'Content-Type': 'application/json'},
            body: body === undefined ? undefined : JSON.stringify(body)
        });
        if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || 'Request failed');
        return response.json();
    }

    async function wake() {
        if (readOnly) return;
        try {
            state = await call('/snooze', {seconds: 0});
            renderControl();
            nextCheck = Date.now() + 1500;
        } catch (error) {
            noteEl.textContent = error.message;
        }
    }

    function renderControl() {
        if (!state) return;
        levelEl.value = state.enabled ? state.level : 'off';
        predictEl.checked = Boolean(state.predict);
        scanEl.checked = Boolean(state.scan);
        levelEl.disabled = readOnly;
        predictEl.disabled = scanEl.disabled = readOnly || !state.enabled;
        let note = LEVEL_NOTES[state.enabled ? state.level : 'off'];
        if (state.enabled && state.evaluations_allowed) {
            note += ` Looked ${state.evaluations_today || 0} of ${state.evaluations_allowed} times today`;
            if (state.level !== 'low') note += `, acted ${state.actions_today || 0} of ${state.actions_allowed}`;
            note += '.';
        }
        if (readOnly) note += ' (Read-only mode.)';
        noteEl.replaceChildren(document.createTextNode(note));
        if (state.snoozed_until) {
            noteEl.append(document.createTextNode(' Quiet for now. '));
            if (!readOnly) {
                const wakeBtn = button('Wake', '', wake);
                noteEl.append(wakeBtn);
            }
        }
        dock.dataset.level = state.enabled ? state.level : 'off';
        renderBadge();
    }

    // The badge beside the agent's name, as the Self-aware badge is shown:
    // there while the agent is proactive, naming its level.
    function renderBadge() {
        const label = state.enabled ? String(state.label || state.level) : 'off';
        const name = label.charAt(0).toUpperCase() + label.slice(1);
        if (chipEl) {
            chipEl.hidden = !state.enabled;
            chipEl.dataset.level = label;
            chipEl.textContent = state.enabled ? `Proactive · ${name}` : '';
            chipEl.title = state.enabled ? `${BADGE_NOTES[label] || BADGE_NOTES.low} Change the level below the conversation.` : '';
        }
        if (factEl) factEl.textContent = state.enabled ? name : 'Off';
    }

    function showControl() {
        dock.scrollIntoView({ block: 'nearest' });
        if (!levelEl.disabled) levelEl.focus();
    }
    if (chipEl) chipEl.addEventListener('click', showControl);
    if (factEl) factEl.addEventListener('click', showControl);

    function surfacePendingApproval() {
        if (!state || !state.enabled || shown || readOnly) return;
        const currentMem = memoryId();
        const pending = (state.recent || []).find(item =>
            item && item.status === 'awaiting_approval' && item.proposal_id
            && (!currentMem || !item.memory_id || item.memory_id === currentMem)
        );
        if (pending) show(pending);
    }

    async function loadState() {
        try {
            state = await call('');
            renderControl();
            surfacePendingApproval();
        } catch (_) { /* the control stays as it was */ }
    }

    async function configure(change) {
        if (readOnly) return;
        levelEl.disabled = true;
        try {
            state = await call('', change);
            renderControl();
            clearCard();
            nextCheck = Date.now() + 1500;
        } catch (error) {
            noteEl.textContent = error.message;
        } finally {
            levelEl.disabled = readOnly;
        }
    }

    function clearCard() {
        shown = null;
        cards.replaceChildren();
    }

    function button(label, kind, action) {
        const el = document.createElement('button');
        el.type = 'button';
        el.className = 'btn btn-sm' + (kind ? ' ' + kind : '');
        el.textContent = label;
        el.addEventListener('click', action);
        return el;
    }

    function line(className, text) {
        const el = document.createElement('div');
        el.className = className;
        el.textContent = text;
        return el;
    }

    async function askApproverId(action) {
        let saved = 'local-ui';
        try { saved = localStorage.getItem('memorizzApproverId') || saved; } catch (_) {}
        if (typeof MemorizzDialogs !== 'undefined' && MemorizzDialogs && typeof MemorizzDialogs.prompt === 'function') {
            const entered = await MemorizzDialogs.prompt(
                action === 'approve' ? 'Approve this call?' : 'Reject this call?',
                {
                    detail: 'The decision is recorded under this identity.',
                    label: 'Approver identity (email or operator ID)',
                    value: saved,
                    required: true,
                    confirmLabel: action === 'approve' ? 'Approve & run' : 'Reject'
                }
            );
            if (entered === null) return null;
            const trimmed = entered.trim();
            try { localStorage.setItem('memorizzApproverId', trimmed); } catch (_) {}
            return trimmed;
        }
        return saved;
    }

    async function loadProposalArgs(proposalId, preEl) {
        try {
            const response = await fetch(`${approvalsBase}?status=pending&limit=100`);
            const body = await response.json();
            const match = (body.approvals || []).find(item => item.proposal_id === proposalId);
            if (!match) {
                preEl.textContent = 'This approval is no longer pending.';
                return;
            }
            preEl.textContent = `${match.tool_name}(${JSON.stringify(match.arguments || {}, null, 2)})`;
        } catch (_) {
            preEl.textContent = 'Could not load tool arguments.';
        }
    }

    async function decideApproval(suggestion, action, statusEl, buttons) {
        if (readOnly || !suggestion.proposal_id) return;
        const approverId = await askApproverId(action);
        if (approverId === null) return;
        buttons.forEach(btn => { btn.disabled = true; });
        statusEl.textContent = action === 'approve' ? 'Recording approval…' : 'Rejecting…';
        try {
            const pid = encodeURIComponent(suggestion.proposal_id);
            const decisionResp = await fetch(`${approvalsBase}/${pid}/${action}`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({approver_id: approverId})
            });
            const decision = await decisionResp.json().catch(() => ({}));
            if (!decisionResp.ok) throw new Error(decision.detail || `HTTP ${decisionResp.status}`);

            if (action === 'reject') {
                clearCard();
                if (typeof loadThreadMessages === 'function' && memoryId()) loadThreadMessages(memoryId());
                loadState();
                return;
            }

            statusEl.textContent = 'Approved. Running the approved step…';
            const resumeResp = await fetch(`${approvalsBase}/${pid}/resume`, {method: 'POST'});
            const resumed = await resumeResp.json().catch(() => ({}));
            if (!resumeResp.ok) throw new Error(resumed.detail || `HTTP ${resumeResp.status}`);

            const evidence = resumed.result;
            const resultValue = evidence && typeof evidence === 'object'
                ? (evidence.assistant_response ?? evidence.tool_result ?? evidence)
                : evidence;
            let chained = null;
            [resultValue, evidence && evidence.tool_result, evidence].forEach(value => {
                if (chained || !value) return;
                if (typeof value === 'string' && value.trim().startsWith('{')) {
                    try { value = JSON.parse(value); } catch (_) { return; }
                }
                if (value && typeof value === 'object' && value.error_code === 'approval_required' && value.proposal) {
                    chained = value.proposal;
                }
            });
            if (typeof loadThreadMessages === 'function' && memoryId()) loadThreadMessages(memoryId());
            if (chained && chained.proposal_id) {
                show({
                    ...suggestion,
                    status: 'awaiting_approval',
                    proposal_id: chained.proposal_id,
                    result: `Waiting for your approval to run ${chained.tool_name || 'a tool'}.`
                });
            } else {
                const text = typeof resultValue === 'string'
                    ? resultValue
                    : (resultValue ? JSON.stringify(resultValue) : 'Approved and completed.');
                show({...suggestion, status: 'done', proposal_id: null, result: text});
            }
            loadState();
        } catch (err) {
            statusEl.textContent = `Approval action failed: ${err.message}`;
            buttons.forEach(btn => { btn.disabled = false; });
        }
    }

    function show(suggestion) {
        shown = suggestion;
        const root = document.createElement('div');
        root.className = 'pg-suggestion';
        root.dataset.status = suggestion.status;
        root.dataset.suggestionId = suggestion.id;
        root.setAttribute('role', 'status');

        const kinds = {
            offered: suggestion.origin === 'predicted' ? 'Predicted next step' : 'Suggested next step',
            done: 'Done for you',
            awaiting_approval: 'Waiting for your approval',
            failed: 'Could not do this',
            running: 'Working on it'
        };
        root.append(line('pg-suggestion-kind', kinds[suggestion.status] || 'Suggestion'));
        root.append(line('pg-suggestion-title', suggestion.title));
        root.append(line('pg-suggestion-message', suggestion.message));
        if (suggestion.status === 'done' && suggestion.result) {
            // The full, formatted reply is in the conversation; this is a glance.
            const plain = plainPreview(suggestion.result);
            root.append(line('pg-suggestion-result', plain.length > 320 ? plain.slice(0, 319) + '…' : plain));
        }
        if (suggestion.status === 'awaiting_approval' && suggestion.result) {
            root.append(line('pg-suggestion-result', suggestion.result));
        }
        if (suggestion.status === 'failed' && suggestion.error) {
            root.append(line('pg-suggestion-result', suggestion.error));
        }

        let approvalStatus = null;
        if (suggestion.status === 'awaiting_approval' && suggestion.proposal_id) {
            const details = document.createElement('details');
            details.className = 'pg-suggestion-details';
            const summary = document.createElement('summary');
            summary.textContent = 'Exact tool call';
            const pre = document.createElement('pre');
            pre.className = 'pg-suggestion-result';
            pre.textContent = 'Loading…';
            details.append(summary, pre);
            root.append(details);
            loadProposalArgs(suggestion.proposal_id, pre);
            approvalStatus = line('pg-suggestion-why', '');
            root.append(approvalStatus);
        }

        const why = [];
        const sources = (suggestion.evidence || []).map(item => item.label).filter(Boolean);
        if (sources.length) why.push('From: ' + sources.join(', '));
        if (suggestion.reason) why.push(suggestion.reason);
        if (why.length) root.append(line('pg-suggestion-why', why.join(' · ')));

        const actions = document.createElement('div');
        actions.className = 'pg-suggestion-actions';
        if (suggestion.status === 'offered') {
            if (!readOnly) {
                actions.append(
                    button('Do it', 'btn-primary', () => accept(suggestion)),
                    button('Not now', '', () => answer(suggestion, 'snooze')),
                    button('Dismiss', '', () => answer(suggestion, 'dismiss'))
                );
            }
        } else if (suggestion.status === 'awaiting_approval' && suggestion.proposal_id && !readOnly) {
            const approveBtn = button('Approve & run', 'btn-primary', () => {});
            const rejectBtn = button('Reject', '', () => {});
            const dismissBtn = button('OK', '', clearCard);
            approveBtn.addEventListener('click', () => decideApproval(suggestion, 'approve', approvalStatus, [approveBtn, rejectBtn]));
            rejectBtn.addEventListener('click', () => decideApproval(suggestion, 'reject', approvalStatus, [approveBtn, rejectBtn]));
            actions.append(approveBtn, rejectBtn, dismissBtn);
        } else {
            actions.append(button('OK', '', clearCard));
        }
        root.append(actions);
        cards.replaceChildren(root);
    }

    async function accept(suggestion) {
        if (readOnly) return;
        const queryEl = document.getElementById('pg-query');
        if (!queryEl || busy() || typeof sendMessage !== 'function') return;
        clearCard();
        try { await call(`/suggestions/${encodeURIComponent(suggestion.id)}/accept`, {run: false}); } catch (_) { /* the step still runs */ }
        const draft = queryEl.value;
        queryEl.value = 'Please carry out this next step:\n\n' + suggestion.prompt;
        queryEl.dispatchEvent(new Event('input', {bubbles: true}));
        nextCheck = Date.now() + POLL_MS;
        sendMessage();
        if (draft) {
            queryEl.value = draft;
            queryEl.dispatchEvent(new Event('input', {bubbles: true}));
        }
        loadState();
    }

    async function answer(suggestion, action) {
        if (readOnly) return;
        clearCard();
        try { await call(`/suggestions/${encodeURIComponent(suggestion.id)}/${action}`, {}); } catch (_) { /* gone already */ }
        nextCheck = Date.now() + POLL_MS;
        loadState();
    }

    async function check(force) {
        if (readOnly || !state || !state.enabled || checking) return;
        if (document.visibilityState !== 'visible' || busy()) return;
        if (shown && (shown.status === 'offered' || shown.status === 'awaiting_approval')) return;
        const now = Date.now();
        if (!force && (now < nextCheck || now - lastInput < QUIET_MS)) return;
        checking = true;
        nextCheck = now + POLL_MS;
        try {
            const asked = memoryId();
            const result = await call('/suggestions', {memory_id: asked, trigger: trigger, force: Boolean(force)});
            trigger = 'view';
            const suggestion = (result.suggestions || [])[0];
            if (result.status === 'ready' || result.status === 'acted' || result.status === 'snoozed') {
                loadState();
            }
            // A different conversation is open by now, or a message is going out.
            if (!suggestion || asked !== memoryId() || busy()) return;
            show(suggestion);
            if (result.status === 'acted') {
                if (typeof loadThreadMessages === 'function') loadThreadMessages(asked);
            }
        } catch (_) {
            // Asked again on the next round.
        } finally {
            checking = false;
        }
    }

    if (!readOnly) {
        levelEl.addEventListener('change', () => configure({level: levelEl.value}));
        predictEl.addEventListener('change', () => configure({predict: predictEl.checked}));
        scanEl.addEventListener('change', () => configure({scan: scanEl.checked}));
        ['keydown', 'pointerdown'].forEach(name => window.addEventListener(name, () => { lastInput = Date.now(); }, {passive: true}));
        document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') { trigger = 'return'; nextCheck = Date.now() + 2000; } });
        setInterval(() => check(false), 3000);
    }

    loadState();
    window.pgProactive = {refresh: loadState, check: () => check(true), clear: clearCard};
})();

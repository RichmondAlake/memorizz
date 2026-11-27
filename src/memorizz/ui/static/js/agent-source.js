// An agent's own source: its versions, its errors, and the coding model's work.
(function() {
    const root = document.getElementById('src-root');
    if (!root) return;
    const agentId = root.dataset.agentId;
    const agentName = root.dataset.agentName;
    const readOnly = root.dataset.readOnly === '1';
    const base = '/api/agents/' + encodeURIComponent(agentId) + '/source';
    const page = '/agents/' + encodeURIComponent(agentId) + '/source';
    const $ = function(id) { return document.getElementById(id); };
    let state = null, timer = null, shownJob = null;

    function el(tag, attrs, children) {
        const node = document.createElement(tag);
        Object.keys(attrs || {}).forEach(function(key) {
            if (key === 'class') node.className = attrs[key];
            else if (key === 'text') node.textContent = attrs[key];
            else if (key.indexOf('on') === 0) node.addEventListener(key.slice(2), attrs[key]);
            else if (attrs[key] !== false && attrs[key] != null) node.setAttribute(key, attrs[key]);
        });
        (children || []).forEach(function(child) {
            if (child != null) node.append(child);
        });
        return node;
    }

    async function call(path, body) {
        const response = await fetch(base + path, body === undefined ? {} : {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body || {}),
        });
        const data = await response.json().catch(function() { return {}; });
        if (!response.ok) throw new Error(data.detail || ('HTTP ' + response.status));
        return data;
    }

    function fail(error) {
        if (window.MemorizzDialogs) window.MemorizzDialogs.notify(String(error.message || error));
    }

    function when(value) {
        const date = typeof value === 'number' ? new Date(value * 1000) : new Date(value);
        return isNaN(date) ? '' : date.toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
    }

    function modelLabel(model) {
        return model && model.model ? model.provider + ' · ' + model.model : null;
    }

    function diffBlock(text) {
        const block = el('pre', { class: 'src-diff' });
        (text || 'No changes.').split('\n').forEach(function(line) {
            let kind = '';
            if (line.indexOf('diff --git') === 0 || line.indexOf('+++') === 0 || line.indexOf('---') === 0) kind = 'is-file';
            else if (line.indexOf('@@') === 0) kind = 'is-hunk';
            else if (line[0] === '+') kind = 'is-add';
            else if (line[0] === '-') kind = 'is-del';
            block.append(el('span', { class: kind, text: line || ' ' }));
        });
        return block;
    }

    function kpi(label, value, note) {
        return el('div', { class: 'kpi' }, [
            el('span', { class: 'kpi-label', text: label }),
            el('strong', { class: 'kpi-value', text: value }),
            note ? el('small', { class: 'kpi-note', text: note }) : null,
        ]);
    }

    function renderVersions() {
        const list = $('src-versions');
        list.replaceChildren();
        state.versions.forEach(function(version, index) {
            const body = el('div', { class: 'src-version-body' });
            const stats = version.installed ? null : el('span', { class: 'src-version-meta' }, [
                el('span', { class: 'src-plus', text: '+' + version.insertions }), ' ',
                el('span', { class: 'src-minus', text: '−' + version.deletions }),
                ' · ' + version.files + (version.files === 1 ? ' file' : ' files'),
            ]);
            const item = el('details', { class: 'src-version' }, [
                el('summary', {}, [
                    el('span', { class: 'src-version-number', text: 'v' + version.number }),
                    el('span', { class: 'src-version-message', text: version.message }),
                    index === 0 ? el('span', { class: 'pill pill--success', text: 'Current' }) : null,
                    stats,
                    el('span', { class: 'src-version-meta', text: when(version.at) }),
                ]),
                body,
            ]);
            item.addEventListener('toggle', async function() {
                if (!item.open || body.childElementCount) return;
                if (version.installed) {
                    body.append(el('p', { class: 'src-none', text: 'MemoRizz as it was installed. Nothing was changed here.' }));
                } else {
                    try {
                        const data = await call('/diff?version=' + version.id);
                        body.append(diffBlock(data.diff));
                    } catch (error) { body.append(el('p', { class: 'src-none', text: String(error.message) })); }
                }
                if (!version.installed) {
                    body.append(el('a', { class: 'btn btn-sm', href: page + '/diff?version=' + version.id, text: 'Open the full diff' }), ' ');
                }
                // See this version's console for yourself, in its own window.
                body.append(el('button', { class: 'btn btn-sm', type: 'button', text: 'Preview this version', onclick: function() { preview(version, this); } }), ' ');
                if (index !== 0 && !readOnly) {
                    body.append(el('button', { class: 'btn btn-sm', text: 'Go back to this version', onclick: function() { restore(version); } }));
                }
            });
            list.append(el('li', {}, [item]));
        });
    }

    // --- a version running as its own, read-only console ---

    async function preview(version, button) {
        const label = button ? button.textContent : '';
        if (button) { button.disabled = true; button.textContent = 'Starting the preview…'; }
        // Opened now, while the click still counts, so the browser allows it.
        const tab = window.open('about:blank', '_blank');
        if (tab) tab.document.write('<p style="font:14px system-ui;padding:24px">Starting v' + version.number + '…</p>');
        try {
            const data = await call('/preview', { version: version.id });
            if (tab) tab.location = data.preview.url;
            await load();
        } catch (error) {
            if (tab) tab.close();
            fail(error);
        }
        if (button) { button.disabled = false; button.textContent = label; }
    }

    async function stopPreview() {
        try {
            await fetch(base + '/preview', { method: 'DELETE' });
            await load();
        } catch (error) { fail(error); }
    }

    function renderPreview() {
        const panel = $('src-preview');
        const running = state.preview;
        panel.hidden = !running;
        if (!running) return;
        const minutes = Math.max(1, Math.round((running.ends_at - Date.now() / 1000) / 60));
        panel.replaceChildren(
            el('div', { class: 'src-preview-text' }, [
                el('strong', { text: 'Preview of v' + running.number + ' is running' }),
                el('span', { text: ' at ' + running.url + ' · read-only · stops by itself in about ' + minutes + ' min' }),
            ]),
            el('div', { class: 'src-actions' }, [
                el('a', { class: 'btn btn-sm btn-primary', href: running.url, target: '_blank', rel: 'noopener', text: 'Open' }),
                el('button', { class: 'btn btn-sm', type: 'button', text: 'Stop', onclick: stopPreview }),
            ])
        );
    }

    async function restore(version) {
        const ok = !window.MemorizzDialogs || await window.MemorizzDialogs.confirm(
            'Go back to v' + version.number + '?',
            { detail: 'The source returns to how it was then, as a new version. Nothing kept is lost.', confirmLabel: 'Go back' }
        );
        if (!ok) return;
        try {
            await call('/restore', { version: version.installed ? 'installed' : version.id });
            await load();
        } catch (error) { fail(error); }
    }

    function renderErrors() {
        const list = $('src-errors');
        list.replaceChildren();
        $('src-error-count').textContent = state.errors.length ? state.errors.length + ' recorded' : 'None';
        if (!state.errors.length) {
            list.append(el('li', { class: 'src-none', text: 'No errors recorded. They appear here when one of ' + agentName + "'s turns fails." }));
            return;
        }
        state.errors.forEach(function(error) {
            const frames = error.frames || [];
            const frame = frames[frames.length - 1];
            list.append(el('li', { class: 'src-error' }, [
                el('div', { class: 'src-error-head' }, [
                    el('span', { text: when(error.at) }),
                    el('span', { class: 'pill', text: error.level || error.source || 'error' }),
                    error.where ? el('span', { text: error.where }) : null,
                ]),
                el('p', { class: 'src-error-text', text: error.error || '' }),
                frame && frame.file ? el('div', { class: 'src-error-frame', text: frame.file + ':' + frame.line + (frame.function ? ' in ' + frame.function : '') }) : null,
            ]));
        });
    }

    function renderWork() {
        const panel = $('src-work');
        const job = state.job;
        const running = !!job && job.state === 'running';
        $('src-recommend').disabled = running || !state.enabled;
        $('src-fix').disabled = running || !state.enabled || readOnly;
        if (!job) { panel.hidden = true; return; }
        panel.hidden = false;
        // A finished result is drawn once, so an open diff is not redrawn under the reader.
        const key = job.id + ':' + job.state + ':' + (running ? job.step : '');
        if (key === shownJob) return;
        shownJob = key;
        panel.replaceChildren();
        const title = job.kind === 'fix' ? 'Fixing the source' : 'Finding the cause';
        if (running) {
            panel.append(el('div', { class: 'src-work-head' }, [
                el('span', { class: 'src-spinner', 'aria-hidden': 'true' }),
                el('h2', { class: 'panel-title', text: title }),
                el('span', { class: 'src-step', text: (job.step || 'Working') + '…' }),
            ]));
            return;
        }
        if (job.state === 'error') {
            panel.append(el('div', { class: 'src-work-head' }, [el('h2', { class: 'panel-title', text: title + ' did not finish' })]));
            panel.append(el('pre', { class: 'src-text', text: job.error || 'Unknown error' }));
            return;
        }
        const result = job.result || {};
        const model = result.coding_model ? ' · ' + result.coding_model : '';
        if (job.kind === 'recommend') {
            panel.append(el('div', { class: 'src-work-head' }, [
                el('h2', { class: 'panel-title', text: 'Recommended fix' }),
                el('span', { class: 'src-version-meta', text: result.failures + ' recent error' + (result.failures === 1 ? '' : 's') + ' read' + model }),
            ]));
            panel.append(el('pre', { class: 'src-text', text: result.recommendation || 'The model gave no answer.' }));
            if (!readOnly) {
                panel.append(el('div', { class: 'src-actions' }, [
                    el('button', { class: 'btn btn-primary', text: 'Make this fix', onclick: function() { start('fix', { recommendation: result.recommendation }); } }),
                ]));
            }
            return;
        }
        const heads = {
            fixed: 'Fixed and kept as v' + (result.version ? result.version.number : ''),
            no_change: 'No change was made',
            check_failed: 'The fix was dropped: the source no longer passed its check',
        };
        panel.append(el('div', { class: 'src-work-head' }, [
            el('h2', { class: 'panel-title', text: heads[result.outcome] || 'Finished' }),
            el('span', { class: 'src-version-meta', text: model.replace(' · ', '') }),
        ]));
        panel.append(el('pre', { class: 'src-text', text: result.answer || '' }));
        if (result.check && !result.check.ok) {
            result.check.steps.filter(function(step) { return !step.ok; }).forEach(function(step) {
                panel.append(el('pre', { class: 'src-text', text: step.name + '\n' + step.output }));
            });
        }
        if (result.diff) panel.append(diffBlock(result.diff));
        if (result.version) {
            panel.append(el('div', { class: 'src-actions' }, [
                el('a', { class: 'btn', href: page + '/diff?version=' + result.version.id, text: 'Open the full diff' }),
            ]));
        }
    }

    function render() {
        $('src-off').hidden = state.enabled;
        $('src-on').hidden = !state.enabled;
        if (!state.enabled) {
            $('src-recommend').disabled = true;
            $('src-fix').disabled = true;
            return;
        }
        const source = state.source;
        const outsideAgent = state.coding_agent && state.coding_agent.kind ? state.coding_agent : null;
        const chosen = outsideAgent ? 'droid' : modelLabel(state.coding_model);
        $('src-kpis').replaceChildren(
            kpi('Started from', 'MemoRizz ' + (source.base_version || '?'), source.base_version !== source.installed_version ? 'Installed is now ' + source.installed_version : 'Same as installed'),
            kpi('Current version', 'v' + (source.version_number || 0), source.head || ''),
            kpi('Fixes kept', String(source.versions || 0), (source.changed_files || []).length ? source.changed_files.length + ' file(s) edited, not kept' : 'Nothing half-made'),
            kpi('Coding model', outsideAgent ? 'Factory Droid' : chosen ? state.coding_model.model : 'Agent’s own',
                outsideAgent ? (outsideAgent.model || 'Droid’s own model')
                : chosen ? state.coding_model.provider + (state.coding_model.host || state.coding_model.base_url ? ' · ' + (state.coding_model.host || state.coding_model.base_url) : '') : state.agent_model)
        );
        $('src-model-pill').textContent = outsideAgent ? 'Factory Droid' : chosen ? state.coding_model.model : 'Using ' + state.agent_model;
        $('src-path').textContent = source.path || '';
        $('src-patch').hidden = !(source.versions > 0);
        $('src-diff-all').hidden = !(source.versions > 0);
        const form = $('src-model-form');
        if (!form.dataset.filled) {
            form.dataset.filled = '1';
            // An outside coding agent (Droid) shows as a choice in the same menu.
            const outside = state.coding_agent && state.coding_agent.kind ? state.coding_agent : null;
            const model = outside ? { provider: outside.kind, model: outside.model || '' } : (state.coding_model || {});
            form.model.value = model.model || '';
            form.host.value = model.host || '';
            form.base_url.value = model.base_url || '';
            form.api_key_env.value = model.api_key_env || '';
            pickProvider(outside ? outside.kind : (model.model ? (model.provider || 'ollama') : ''));
        }
        keyStatus();
        renderPreview();
        renderVersions();
        renderErrors();
        renderWork();
    }

    // --- the coding model form: a provider menu, and fields for that provider ---
    const pickButton = $('cm-pick-button'), pickMenu = $('cm-menu');
    const options = Array.from(pickMenu.querySelectorAll('.cm-option'));
    const catalog = (function() { try { return JSON.parse($('cm-catalog').textContent); } catch (error) { return {}; } })();

    function pickProvider(value) {
        const form = $('src-model-form');
        const option = options.find(function(item) { return item.dataset.value === value; }) || options[0];
        form.provider.value = option.dataset.value;
        options.forEach(function(item) { item.setAttribute('aria-selected', item === option ? 'true' : 'false'); });
        pickButton.querySelector('[data-cm-name]').textContent = option.querySelector('strong').textContent;
        const mark = pickButton.querySelector('[data-cm-mark]');
        mark.textContent = option.querySelector('.cm-mark').textContent;
        mark.dataset.provider = option.dataset.value || 'own';
        const own = !option.dataset.value;
        form.querySelector('[data-cm-needs-model]').hidden = own;
        form.querySelectorAll('[data-for-provider]').forEach(function(field) {
            field.hidden = field.dataset.forProvider !== option.dataset.value;
        });
        form.model.placeholder = option.dataset.example || 'Model name';
        // Droid picks its own model unless one is named.
        form.querySelector('[data-cm-needs-model] .cm-label').textContent = option.dataset.value === 'droid' ? 'Model (optional)' : 'Model';
        form.api_key_env.placeholder = option.dataset.key || 'OPENAI_API_KEY';
        $('cm-models').replaceChildren.apply($('cm-models'), (catalog[option.dataset.value] || []).map(function(name) {
            return el('option', { value: name });
        }));
        keyStatus();
    }

    // Whether the key this choice needs is set where MemoRizz runs.
    function keyStatus() {
        const form = $('src-model-form'), line = $('src-key-status');
        const option = options.find(function(item) { return item.dataset.value === form.provider.value; }) || options[0];
        const named = form.api_key_env.value.trim();
        const key = option.dataset.value === 'openai' && (named || form.base_url.value.trim()) ? named : option.dataset.key;
        const saved = state && state.coding_agent && state.coding_agent.kind ? { provider: state.coding_agent.kind }
            : state && state.coding_model && state.coding_model.model ? state.coding_model : null;
        const unchanged = saved ? saved.provider === option.dataset.value && (saved.api_key_env || '') === named : !option.dataset.value;
        let text = 'Keys are read from the environment and never saved here.', kind = '';
        if (option.dataset.value === 'droid' && state && !state.droid_installed) {
            line.textContent = 'Droid is not installed where MemoRizz runs. Install Factory\u2019s CLI so `droid` is on the PATH, then restart.';
            line.className = 'cm-key is-missing';
            return;
        }
        if (!option.dataset.value) text = 'Nothing to set up: fixes use ' + (state ? state.agent_model : "the agent's model") + '.';
        else if (!key) text = option.dataset.value === 'ollama' ? 'No key needed.' : 'No key will be sent to this endpoint.';
        else if (unchanged && state && state.coding_key_missing) { text = key + ' is not set where MemoRizz runs. Add it in Settings or ~/.memorizz/.env, then restart.'; kind = 'is-missing'; }
        else if (state && state.coding_keys && key === option.dataset.key) {
            const set = state.coding_keys[option.dataset.value];
            text = set ? key + ' is set.' : key + ' is not set where MemoRizz runs. Add it in Settings or ~/.memorizz/.env, then restart.';
            kind = set ? 'is-set' : 'is-missing';
        } else text = 'The key is read from ' + key + ' when a fix runs.';
        line.textContent = text;
        line.className = 'cm-key ' + kind;
    }

    function openMenu(open) {
        pickMenu.hidden = !open;
        pickButton.setAttribute('aria-expanded', open ? 'true' : 'false');
        if (!open) return;
        const current = options.find(function(item) { return item.getAttribute('aria-selected') === 'true'; }) || options[0];
        focusOption(current);
    }

    function focusOption(option) {
        options.forEach(function(item) { item.classList.toggle('is-active', item === option); });
        pickMenu.setAttribute('aria-activedescendant', option.id);
        pickMenu.focus();
        option.scrollIntoView({ block: 'nearest' });
    }

    pickButton.addEventListener('click', function() { openMenu(pickMenu.hidden); });
    pickButton.addEventListener('keydown', function(event) {
        if (event.key === 'ArrowDown' || event.key === 'ArrowUp') { event.preventDefault(); openMenu(true); }
    });
    options.forEach(function(option) {
        option.addEventListener('click', function() { pickProvider(option.dataset.value); openMenu(false); pickButton.focus(); });
        option.addEventListener('mousemove', function() { options.forEach(function(item) { item.classList.toggle('is-active', item === option); }); });
    });
    pickMenu.addEventListener('keydown', function(event) {
        const at = options.findIndex(function(item) { return item.classList.contains('is-active'); });
        if (event.key === 'ArrowDown') { event.preventDefault(); focusOption(options[Math.min(options.length - 1, at + 1)]); }
        else if (event.key === 'ArrowUp') { event.preventDefault(); focusOption(options[Math.max(0, at - 1)]); }
        else if (event.key === 'Home') { event.preventDefault(); focusOption(options[0]); }
        else if (event.key === 'End') { event.preventDefault(); focusOption(options[options.length - 1]); }
        else if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); pickProvider(options[Math.max(0, at)].dataset.value); openMenu(false); pickButton.focus(); }
        else if (event.key === 'Escape' || event.key === 'Tab') { openMenu(false); pickButton.focus(); }
    });
    document.addEventListener('click', function(event) {
        if (!pickMenu.hidden && !$('cm-pick').contains(event.target)) openMenu(false);
    });

    async function load() {
        window.clearTimeout(timer);
        try {
            const openVersions = Array.from(document.querySelectorAll('.src-version[open]')).length;
            const next = await call('');
            const running = next.job && next.job.state === 'running';
            const changed = !state || JSON.stringify(next.versions) !== JSON.stringify(state.versions) || next.enabled !== state.enabled || JSON.stringify(next.coding_model) !== JSON.stringify(state.coding_model) || next.errors.length !== state.errors.length || JSON.stringify(next.preview) !== JSON.stringify(state.preview);
            state = next;
            // While a reader has a diff open and nothing changed, only the work panel moves.
            if (changed || !openVersions) render(); else renderWork();
            if (running) timer = window.setTimeout(load, 1500);
        } catch (error) { fail(error); }
    }

    async function start(kind, body) {
        try {
            await call('/' + kind, body || {});
            window.dispatchEvent(new Event('memorizz:agent-activity'));
            await load();
        } catch (error) { fail(error); }
    }

    $('src-recommend').addEventListener('click', function() { start('recommend'); });
    $('src-fix').addEventListener('click', function() { start('fix'); });
    $('src-enable').addEventListener('click', async function() {
        this.disabled = true;
        this.textContent = 'Copying the source…';
        try { await call('/enable', {}); await load(); } catch (error) { fail(error); this.disabled = false; }
    });
    $('src-model-form').addEventListener('submit', async function(event) {
        event.preventDefault();
        const button = $('src-model-save');
        const own = !this.provider.value;
        if (!own && this.provider.value !== 'droid' && !this.model.value.trim()) { this.model.focus(); return fail(new Error('Name the model to use, or choose the agent\u2019s own model.')); }
        button.disabled = true;
        try {
            await call('/coding-model', own ? { provider: 'ollama', model: '' } : { provider: this.provider.value, model: this.model.value.trim(), host: this.host.value.trim(), base_url: this.base_url.value.trim(), api_key_env: this.api_key_env.value.trim() });
            await load();
            button.textContent = 'Saved';
            window.setTimeout(function() { button.textContent = 'Save'; }, 1600);
        } catch (error) { fail(error); }
        button.disabled = readOnly;
    });
    $('src-model-form').addEventListener('input', keyStatus);
    load();
})();

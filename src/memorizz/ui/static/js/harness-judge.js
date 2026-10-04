/* Shared judge controls for launches and saved-output comparisons. */
(() => {
    'use strict';
    const esc = text => String(text == null ? '' : text).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
    let defaults = null;
    let loading = null;
    let serial = 0;
    const request = async (url, options = {}) => {
        const response = await fetch(url, {credentials: 'same-origin', headers: {'Content-Type': 'application/json', Accept: 'application/json'}, ...options});
        const body = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : 'Evaluation request failed');
        return body;
    };
    const settings = () => {
        if (defaults) return Promise.resolve(defaults);
        if (!loading) loading = request('/api/harness-judge/settings').then(body => { defaults = body; return body; }).finally(() => { loading = null; });
        return loading;
    };
    const configFrom = root => {
        const value = key => root.querySelector('[data-judge-field="' + key + '"]').value.trim();
        const config = {provider: value('provider'), model: value('model'), prompt: value('prompt'), reference: value('reference'), pass_score: Number(value('pass_score'))};
        if (!config.model || !config.prompt) throw new Error('Enter the judge model and evaluation prompt.');
        if (!Number.isFinite(config.pass_score) || config.pass_score < 0 || config.pass_score > 100) throw new Error('Accuracy target must be between 0 and 100.');
        return config;
    };
    const setConfig = (root, config) => {
        if (!root || !config) return;
        Object.entries(config).forEach(([key, value]) => {
            const input = root.querySelector('[data-judge-field="' + key + '"]');
            if (input) input.value = value == null ? '' : String(value);
        });
        root.dispatchEvent(new Event('judge-config'));
    };
    const mount = async (root, options = {}) => {
        if (!root) return;
        const id = 'hx-judge-' + (++serial);
        root.classList.add('hxjudge');
        root.innerHTML = '<p class="form-hint">Loading judge settings…</p>';
        try {
            const body = await settings();
            if (!root.isConnected) return;
            const readOnly = options.readOnly || body.read_only;
            root.innerHTML = (options.automatic ? '<label class="harness-check hxjudge-enable"><input type="checkbox" data-judge-auto> Evaluate answers after the runs finish</label>' : '')
                + '<div class="hxjudge-fields"' + (options.automatic ? ' hidden' : '') + '>'
                + '<div class="hxjudge-grid"><div class="form-group"><label for="' + id + '-provider">Judge provider</label><select id="' + id + '-provider" data-judge-field="provider">'
                + Object.entries(body.providers).map(([value, label]) => '<option value="' + esc(value) + '">' + esc(label) + '</option>').join('') + '</select></div>'
                + '<div class="form-group"><label for="' + id + '-model">Judge model</label><input id="' + id + '-model" data-judge-field="model" list="' + id + '-models" maxlength="240" placeholder="Select or enter a model"><datalist id="' + id + '-models"></datalist></div>'
                + '<div class="form-group"><label for="' + id + '-target">Accuracy target / 100</label><input id="' + id + '-target" type="number" min="0" max="100" step="1" data-judge-field="pass_score"></div></div>'
                + '<p class="form-hint hxjudge-provider-note"></p>'
                + '<div class="form-group"><label for="' + id + '-prompt">Evaluation prompt</label><textarea id="' + id + '-prompt" data-judge-field="prompt" rows="4" maxlength="8000"></textarea></div>'
                + '<details class="hxjudge-reference"><summary>Reference answer or supporting evidence (optional)</summary><textarea id="' + id + '-reference" data-judge-field="reference" rows="3" maxlength="16000" aria-label="Reference answer or evidence" placeholder="Expected facts, an answer key, acceptance criteria or evidence the judge should use"></textarea></details>'
                + '<p class="form-hint">The score estimates accuracy from the task, answer and supplied evidence. The judge does not inspect files or run tests. The accuracy target highlights the cheapest and fastest answers that meet it.</p>'
                + '<div class="hxjudge-actions">' + (!options.automatic && !readOnly ? '<button class="btn btn-primary btn-sm" type="button" data-judge-start>Evaluate answers</button>' : '')
                + (!readOnly ? '<button class="btn btn-sm" type="button" data-judge-save>Save judge settings</button>' : '')
                + '<span class="form-hint hxjudge-status" role="status" aria-live="polite"></span></div></div>';
            const $ = selector => root.querySelector(selector);
            const status = $('.hxjudge-status');
            const note = $('.hxjudge-provider-note');
            let modelRequest = 0;
            const choices = async () => {
                const provider = $('[data-judge-field="provider"]').value;
                const current = ++modelRequest;
                let models = (body.models || {})[provider];
                if (!models) {
                    try { models = (await request('/api/harness-judge/models?provider=' + encodeURIComponent(provider))).models; }
                    catch (_) { models = []; }
                }
                if (current !== modelRequest || !root.isConnected) return;
                $('#' + id + '-models').innerHTML = (models || []).map(model => '<option value="' + esc(model) + '"></option>').join('');
                note.textContent = provider === 'ollama'
                    ? (models.length ? 'Uses your local Ollama model. No external API charge; evaluation time is reported separately.' : 'No installed Ollama chat model was found. Enter an installed model or choose another provider.')
                    : 'Uses this provider’s configured API credentials. Evaluation charges are reported separately from the harness run.';
            };
            root.addEventListener('judge-config', choices);
            $('[data-judge-field="provider"]').addEventListener('change', async () => {
                const provider = $('[data-judge-field="provider"]').value;
                $('[data-judge-field="model"]').value = provider === 'ollama' ? (body.config.provider === 'ollama' ? body.config.model : (body.models.ollama.includes('qwen2.5:3b') ? 'qwen2.5:3b' : body.models.ollama[0] || '')) : '';
                await choices();
            });
            setConfig(root, options.config || body.config);
            const enabled = $('[data-judge-auto]');
            if (enabled) enabled.addEventListener('change', () => { $('.hxjudge-fields').hidden = !enabled.checked; });
            $('[data-judge-save]')?.addEventListener('click', async event => {
                const button = event.currentTarget;
                button.disabled = true;
                try {
                    const saved = await request('/api/harness-judge/settings', {method: 'PUT', body: JSON.stringify(configFrom(root))});
                    defaults = {...body, config: saved.config};
                    status.textContent = 'Judge settings saved.';
                } catch (error) { status.textContent = error.message; }
                finally { button.disabled = false; }
            });
            const start = $('[data-judge-start]');
            if (start) start.addEventListener('click', async () => {
                start.disabled = true;
                status.textContent = 'Queuing evaluation…';
                try {
                    const runIds = options.runIds();
                    if (!runIds.length) throw new Error('Wait for a successful run with an answer.');
                    const result = await request('/api/harness-judgments', {method: 'POST', body: JSON.stringify({run_ids: runIds, config: configFrom(root)})});
                    status.textContent = 'Evaluation queued. Scores will update here.';
                    options.onStarted?.(result.judgment);
                } catch (error) { status.textContent = error.message; start.disabled = false; }
            });
            return {root, setConfig: config => setConfig(root, config), setBusy: busy => { if (start) start.disabled = busy; }, status: text => { status.textContent = text; }};
        } catch (error) {
            root.innerHTML = '<p class="form-hint">Could not load judge settings: ' + esc(error.message) + '</p>';
        }
    };
    const renderResult = judgment => {
        if (!judgment) return '';
        const config = judgment.config || {};
        const completed = judgment.status === 'completed' && typeof judgment.score === 'number';
        const title = completed ? 'Judged accuracy: ' + judgment.score + '/100' : 'Evaluation: ' + judgment.status;
        return '<section class="hxjudge-result" aria-label="Judge result"><strong>' + esc(title) + '</strong>'
            + '<small>' + esc(config.provider + ' · ' + config.model) + '</small>'
            + (judgment.error ? '<p class="hxcmp-error">' + esc(judgment.error) + '</p>' : '')
            + (completed ? '<p>' + esc(judgment.rationale) + '</p>' + ((judgment.issues || []).length ? '<ul>' + judgment.issues.map(issue => '<li>' + esc(issue) + '</li>').join('') + '</ul>' : '') : '')
            + '<details><summary>Evaluation prompt and reference</summary><p class="hxcmp-plain">' + esc(config.prompt) + '</p>'
            + (config.reference ? '<p class="hxcmp-plain"><strong>Reference</strong><br>' + esc(config.reference) + '</p>' : '') + '</details></section>';
    };
    window.MemorizzJudge = {mount, configFrom, setConfig, renderResult};
})();

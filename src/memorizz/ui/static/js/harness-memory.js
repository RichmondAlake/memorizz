// A shared memory's refine cycle: preview it, run it, choose its model.
(() => {
    const root = document.getElementById('hx-knows');
    if (!root) return;
    const memory = root.dataset.memory;
    const api = '/api/harness-memories/' + encodeURIComponent(memory) + '/refine';
    const status = document.getElementById('hx-refine-status');
    const report = document.getElementById('hx-refine-report');
    const preview = document.getElementById('hx-refine-preview');
    const run = document.getElementById('hx-refine-run');
    const esc = (value) => String(value == null ? '' : value).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
    const LABEL = {added: 'New', confirmed: 'Agrees', replaced: 'Replaces', not_taken: 'Not taken'};

    const draw = (value, dry) => {
        const findings = (value.jobs || [])[0] || {items: [], notes: []};
        const items = findings.items || [];
        const head = value.status === 'failed'
            ? 'The cycle did not run: ' + (value.reason || 'it failed') + '.'
            : value.status === 'skipped'
                ? 'Nothing new to distil.'
                : dry ? 'This is what a cycle would change. Nothing has been written.' : 'Done. Reloading\u2026';
        report.hidden = false;
        report.innerHTML = '<p class="hx-refine-head">' + esc(head) + '</p>'
            + (items.length ? '<ul class="hx-refine-items">' + items.map(item => '<li class="hx-refine-item hx-refine-item--' + esc(item.kind) + '">'
                + '<span class="hx-refine-kind">' + esc(LABEL[item.kind] || item.kind) + '</span>'
                + '<div><p>' + esc(item.text) + '</p><p class="hx-refine-detail">' + esc(item.detail) + '</p></div></li>').join('') + '</ul>'
            : value.status === 'failed' || value.status === 'skipped' ? '' : '<p class="hx-refine-detail">The model found nothing worth keeping in the new answers.</p>');
    };

    const start = async (dry) => {
        preview.disabled = run.disabled = true;
        status.textContent = dry ? 'Reading the new answers\u2026' : 'Refining\u2026';
        try {
            const response = await fetch(api, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({dry_run: dry})});
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(data.detail || 'HTTP ' + response.status);
            draw(data.report, dry);
            status.textContent = '';
            if (!dry && data.report.status === 'done') { setTimeout(() => window.location.reload(), 900); return; }
        } catch (error) {
            status.textContent = String(error.message || error);
        }
        preview.disabled = false;
        run.disabled = root.dataset.readOnly === 'true';
    };
    preview.addEventListener('click', () => start(true));
    run.addEventListener('click', () => start(false));

    const form = document.getElementById('hx-refine-settings');
    const saved = document.getElementById('hx-refine-saved');
    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        const data = new FormData(form);
        saved.textContent = 'Saving\u2026';
        try {
            const response = await fetch('/api/harness-memory-refine/settings', {method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({
                provider: data.get('provider'), model: String(data.get('model') || '').trim(),
                auto: data.has('auto'), after_answers: Number(data.get('after_answers') || 3),
            })});
            const body = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(body.detail || 'HTTP ' + response.status);
            saved.textContent = body.settings.model ? 'Saved.' : 'Saved. Choose a provider and a model to refine.';
            form.querySelector('[name="auto"]').checked = body.settings.auto;
        } catch (error) { saved.textContent = String(error.message || error); }
    });
})();

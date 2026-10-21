// Agent templates on the agent form. "Use template" fills the form in from a
// template; "Save as template" keeps the form's settings as one of your own.
// Loaded after the form's own scripts, so the model pickers are ready.
(function () {
    const form = document.querySelector('form.agent-form');
    if (!form) return;
    const templates = Array.isArray(window.AGENT_FORM_TEMPLATES) ? window.AGENT_FORM_TEMPLATES : [];
    const byKey = {};
    templates.forEach(function (template) { byKey[template.key] = template; });
    const keyInput = document.getElementById('template_key');
    const banner = document.getElementById('agent-template-banner');
    const TRUE = ['1', 'on', 'true', 'yes'];
    // Fields a template never resets: its own key, the Google client, the
    // name being typed for a new template, and the pickers that only help
    // fill other fields (they load their options after the page).
    const KEEP = /^(template_|google_client_|harness_delegate_|persona_preset_select|persona_saved_select)/;
    let blank = null;
    let applied = null;

    document.querySelectorAll('[data-callback-url]').forEach(function (el) {
        el.textContent = window.location.origin + '/api/mcp/oauth/callback';
    });

    function controls() {
        return Array.from(form.elements).filter(function (el) {
            const key = el.name || el.id;
            return key && !KEEP.test(key) && !['submit', 'button', 'file'].includes(el.type) && el.tagName !== 'FIELDSET';
        });
    }

    function snapshot() {
        return controls().map(function (el) {
            return {el: el, value: el.value, checked: el.checked, disabled: el.disabled, hidden: el.hidden};
        });
    }

    // Whether anything the form submits was changed by hand.
    function differs(state) {
        return (state || []).some(function (item) {
            if (!item.el.name && item.el.id !== 'continuous_mode') return false;
            return item.el.value !== item.value || item.el.checked !== item.checked;
        });
    }

    function restore(state) {
        (state || []).forEach(function (item) {
            item.el.value = item.value;
            item.el.checked = item.checked;
            item.el.disabled = item.disabled;
            item.el.hidden = item.hidden;
        });
        afterChange();
    }

    // The pickers and hints that are derived from other fields.
    function afterChange() {
        if (typeof syncAgentModelOptions === 'function') syncAgentModelOptions();
        if (typeof loadBackgroundModel === 'function') loadBackgroundModel();
        if (typeof syncEntityScope === 'function') syncEntityScope();
        const continuous = document.getElementById('continuous_mode');
        if (continuous && typeof toggleContinuousMode === 'function') toggleContinuousMode(continuous);
        ['skills_marketplace_provider', 'default_harness'].forEach(function (id) {
            document.getElementById(id)?.dispatchEvent(new Event('change', {bubbles: true}));
        });
    }

    function truthy(value) {
        return value === true || TRUE.includes(String(value).toLowerCase());
    }

    function setField(name, value) {
        form.querySelectorAll('[name="' + CSS.escape(name) + '"]').forEach(function (el) {
            if (el.type === 'checkbox') {
                el.checked = truthy(value);
            } else if (el.tagName === 'SELECT') {
                // A list is an order of preference: the first option offered wins.
                const wanted = (Array.isArray(value) ? value : [value]).map(String);
                const offered = Array.from(el.options).filter(function (o) { return !o.disabled; }).map(function (o) { return o.value; });
                const pick = wanted.find(function (v) { return offered.includes(v); });
                if (pick !== undefined) el.value = pick;
            } else {
                el.value = value == null ? '' : String(value);
            }
            el.dispatchEvent(new Event('change', {bubbles: true}));
        });
    }

    function setHarnessModel(value) {
        const input = document.getElementById('harness_model');
        const select = document.getElementById('harness_model_select');
        if (!input) return;
        input.value = value || '';
        if (!select) return;
        const known = Array.from(select.options).some(function (o) { return o.value === value; });
        select.value = !value ? '' : (known ? value : '__other__');
        input.hidden = select.value !== '__other__';
    }

    function apply(values) {
        const done = new Set(['llm_provider', 'llm_model', 'background_model', 'max_steps', 'harness_model']);
        if ('llm_provider' in values || 'llm_model' in values) {
            const provider = document.getElementById('llm_provider');
            if (values.llm_provider) provider.value = values.llm_provider;
            document.getElementById('llm_model').value = values.llm_model || '';
            document.getElementById('llm_model_custom').value = values.llm_model || '';
        }
        if ('background_model' in values) {
            document.getElementById('background_model').value = values.background_model || '';
        }
        if ('max_steps' in values) {
            const steps = parseInt(values.max_steps, 10);
            document.getElementById('continuous_mode').checked = steps === 0;
            if (steps > 0) document.getElementById('max_steps').value = String(steps);
        }
        Object.keys(values).forEach(function (name) {
            if (!done.has(name)) setField(name, values[name]);
        });
        afterChange();
        if ('harness_model' in values) setHarnessModel(values.harness_model);
        // The Ollama thinking checkbox mirrors the LLM config.
        const thinking = document.getElementById('ollama_thinking');
        if (thinking) {
            let config = {};
            try { config = JSON.parse(document.getElementById('llm_config_json').value || '{}') || {}; } catch (_) { config = {}; }
            thinking.checked = Boolean(config.think === true || (config.additional_config && config.additional_config.think === true));
        }
        // A template's persona is new, not a saved one.
        document.getElementById('persona_id').value = '';
        ['persona_preset_select', 'persona_saved_select'].forEach(function (id) {
            const el = document.getElementById(id);
            if (el) el.value = '';
        });
        const hint = document.getElementById('persona_version_hint');
        if (hint) hint.style.display = 'none';
    }

    // Open the sections the template filled in, so they can be reviewed.
    function openChanged() {
        (blank || []).forEach(function (item) {
            if (item.el.value === item.value && item.el.checked === item.checked) return;
            let details = item.el.closest('details');
            while (details) {
                details.open = true;
                details = details.parentElement && details.parentElement.closest('details');
            }
        });
    }

    function showBanner(template) {
        document.querySelectorAll('[data-use-template]').forEach(function (button) {
            const selected = Boolean(template) && button.dataset.useTemplate === template.key;
            button.setAttribute('aria-pressed', String(selected));
            button.textContent = selected ? 'In use' : 'Use template';
            button.closest('[data-template-card]')?.classList.toggle('is-selected', selected);
        });
        if (!banner) return;
        banner.hidden = !template;
        if (!template) return;
        banner.querySelector('[data-template-title]').textContent = template.title;
        const connections = template.connections || [];
        banner.querySelector('[data-template-connections]').hidden = !connections.length;
        banner.querySelector('[data-template-connection-list]').textContent = connections.join(', ');
        banner.querySelector('[data-template-google]').hidden = !template.needs_google_client;
    }

    function setUrl(key) {
        const url = key ? '/agents/new?template=' + encodeURIComponent(key) : '/agents/new';
        try { history.replaceState(null, '', url); } catch (_) { /* file:// or sandboxed */ }
    }

    async function useTemplate(key, scroll) {
        const template = byKey[key];
        if (!template) return;
        if (applied && differs(applied) && window.MemorizzDialogs) {
            const replace = await MemorizzDialogs.confirm('Replace what you entered with the ' + template.title + ' template?', {
                detail: 'The fields go back to their defaults, then the template fills them in.',
                confirmLabel: 'Use template',
                cancelLabel: 'Keep my changes',
                danger: false,
            });
            if (!replace) return;
        }
        restore(blank);
        apply(template.values || {});
        keyInput.value = template.key;
        showBanner(template);
        openChanged();
        applied = snapshot();
        setUrl(template.key);
        if (scroll && banner) banner.scrollIntoView({behavior: 'smooth', block: 'start'});
    }

    function clearTemplate() {
        restore(blank);
        keyInput.value = '';
        showBanner(null);
        applied = snapshot();
        setUrl('');
    }

    document.addEventListener('DOMContentLoaded', function () {
        blank = snapshot();
        applied = blank;
        document.querySelectorAll('[data-use-template]').forEach(function (button) {
            button.addEventListener('click', function () { useTemplate(button.dataset.useTemplate, true); });
        });
        banner?.querySelector('[data-clear-template]')?.addEventListener('click', clearTemplate);
        document.querySelectorAll('form[data-delete-template]').forEach(function (deleteForm) {
            deleteForm.addEventListener('submit', async function (event) {
                event.preventDefault();
                const ok = await MemorizzDialogs.confirm('Delete the template ' + deleteForm.dataset.deleteTemplate + '?', {
                    detail: 'Agents made from it keep their settings.',
                });
                if (ok) deleteForm.submit();
            });
        });
        if (keyInput && keyInput.value && byKey[keyInput.value]) {
            // The form came back after a failed check: keep what was entered.
            showBanner(byKey[keyInput.value]);
        } else if (window.AGENT_FORM_APPLY_TEMPLATE && byKey[window.AGENT_FORM_APPLY_TEMPLATE]) {
            useTemplate(window.AGENT_FORM_APPLY_TEMPLATE, false);
        }
    });

    // Save as template.
    const toggle = document.getElementById('save-template-toggle');
    const panel = document.getElementById('save-template-panel');
    const title = document.getElementById('template_title');
    const summary = document.getElementById('template_summary');
    const submit = document.getElementById('save-template-submit');
    const status = document.getElementById('save-template-status');
    if (!toggle || !panel || !title || !submit) return;

    toggle.addEventListener('click', function () {
        panel.hidden = !panel.hidden;
        toggle.setAttribute('aria-expanded', String(!panel.hidden));
        if (panel.hidden) return;
        if (!title.value.trim()) {
            title.value = (document.getElementById('agent_name')?.value || document.getElementById('persona_role')?.value || '').trim();
        }
        panel.scrollIntoView({behavior: 'smooth', block: 'nearest'});
        title.focus();
    });
    [title, summary].forEach(function (input) {
        input?.addEventListener('keydown', function (event) {
            // Enter here saves the template, not the agent.
            if (event.key === 'Enter') { event.preventDefault(); saveTemplate(); }
        });
    });
    submit.addEventListener('click', saveTemplate);

    // The form's values as a submit would send them.
    function formData() {
        const data = new FormData(form);
        data.delete('google_client_id');
        data.delete('google_client_secret');
        if (document.getElementById('continuous_mode')?.checked) data.set('max_steps', '0');
        let config = {};
        try { config = JSON.parse(data.get('llm_config_json') || '{}'); } catch (_) { config = null; }
        if (config && typeof config === 'object') {
            const baseUrl = (document.getElementById('llm_base_url')?.value || '').trim();
            if (baseUrl) config.base_url = baseUrl;
            if ((data.get('llm_provider') || '') === 'ollama' && document.getElementById('ollama_thinking')?.checked) {
                config.additional_config = Object.assign({}, config.additional_config, {think: true});
            }
            if (Object.keys(config).length) data.set('llm_config_json', JSON.stringify(config));
        }
        return data;
    }

    async function saveTemplate() {
        if (!title.value.trim()) {
            status.classList.add('is-bad');
            status.textContent = 'Give the template a name.';
            title.focus();
            return;
        }
        submit.disabled = true;
        status.classList.remove('is-bad');
        status.textContent = 'Saving…';
        try {
            const response = await fetch('/agents/templates/save', {method: 'POST', body: formData(), headers: {Accept: 'application/json'}});
            const body = await response.json().catch(function () { return {}; });
            if (!response.ok) throw new Error(body.detail || 'Could not save the template');
            if (!form.dataset.isEdit) {
                window.location.href = body.url;
                return;
            }
            status.innerHTML = 'Saved <strong>' + escapeHtml(body.title) + '</strong>. <a href="' + escapeHtml(body.url) + '">Create an agent from it</a>.';
        } catch (error) {
            status.classList.add('is-bad');
            status.textContent = error.message;
        } finally {
            submit.disabled = false;
        }
    }
})();

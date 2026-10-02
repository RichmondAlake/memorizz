/* Find MCP servers and agent skills, and attach them to an agent.
 *
 * Shared by the MCP connections, skills and playground pages. Markup:
 *   <div data-catalog="mcp|skills" data-agent-id="...">
 *     <div data-catalog-form><input type="search"><button type="button">Search</button></div>
 *     <p data-catalog-status></p>
 *     <ul data-catalog-results></ul>
 *     <button data-catalog-more hidden>More results</button>
 *   </div>
 * Nothing is attached until the user presses Attach; secrets go straight to
 * the encrypted credential store through the MCP API. There are no <form>
 * elements, so the catalog can sit inside another form (the playground's).
 */
(function () {
    'use strict';

    const enc = encodeURIComponent;

    async function request(url, options) {
        const response = await fetch(url, options);
        let body = {};
        try { body = await response.json(); } catch (_) { /* not JSON */ }
        if (!response.ok || body.ok === false) {
            const detail = body.detail;
            const message = (detail && typeof detail === 'object' && (detail.message || detail.error))
                || (typeof detail === 'string' ? detail : '')
                || body.error
                || `Request failed (${response.status})`;
            const error = new Error(message);
            error.body = body;
            throw error;
        }
        return body;
    }

    function postJson(url, payload) {
        return request(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload || {}) });
    }

    function node(tag, className, text) {
        const el = document.createElement(tag);
        if (className) el.className = className;
        if (text != null) el.textContent = text;
        return el;
    }

    function button(label, className) {
        const el = node('button', className || 'btn btn-sm', label);
        el.type = 'button';
        return el;
    }

    function link(label, href) {
        const el = node('a', 'catalog-link', label);
        el.href = href; el.target = '_blank'; el.rel = 'noopener noreferrer';
        return el;
    }

    function field(label, input, hint) {
        const wrap = node('label', 'catalog-field');
        wrap.append(node('span', 'catalog-field-label', label), input);
        if (hint) wrap.appendChild(node('small', 'form-hint', hint));
        return wrap;
    }

    function textInput(value, { secret = false, placeholder = '' } = {}) {
        const input = document.createElement('input');
        input.type = secret ? 'password' : 'text';
        input.autocomplete = secret ? 'new-password' : 'off';
        input.value = value || '';
        input.placeholder = placeholder;
        return input;
    }

    // Enter in any input runs the container's main action, never an outer form.
    function onEnter(container, action) {
        container.addEventListener('keydown', event => {
            if (event.key !== 'Enter' || event.target.tagName !== 'INPUT') return;
            event.preventDefault();
            event.stopPropagation();
            action();
        });
    }

    function callbackUrl() { return `${window.location.origin}/api/mcp/oauth/callback`; }

    // ------------------------------------------------------------------ MCP

    const mcp = {
        search(query, cursor) {
            const params = new URLSearchParams({ q: query || '' });
            if (cursor) params.set('cursor', cursor);
            return request(`/api/mcp/catalog?${params}`);
        },
        detectAuth(url) {
            return postJson('/api/mcp/catalog/auth', { url }).then(body => body.auth).catch(() => 'none');
        },
        attachPreset(agentId, key, extra) {
            return postJson(`/api/mcp/agents/${enc(agentId)}/presets/${enc(key)}`, extra);
        },
        attachServer(agentId, config) {
            return postJson(`/api/mcp/agents/${enc(agentId)}/servers`, config);
        },
        async signIn(agentId, name) {
            const body = await postJson(`/api/mcp/agents/${enc(agentId)}/servers/${enc(name)}/authorize`);
            if (body.already_authenticated) return 'already';
            if (!body.authorization_url) throw new Error(body.message || 'Authorization URL was not returned');
            window.location.assign(body.authorization_url);
            return 'redirected';
        },
    };

    // A server configuration from a registry option and what the user typed.
    function serverConfig(option, name, authType, values) {
        const config = JSON.parse(JSON.stringify(option.config));
        const fill = text => String(text).replace(/\{([A-Za-z0-9_.-]+)\}/g, (match, key) => (values[`placeholder:${key}`] || '').trim() || match);
        config.name = name;
        config.require_approval = true;
        config.auth = { ...(config.auth || {}), type: authType || (config.auth || {}).type || 'none' };
        if (config.auth.type === 'oauth') config.auth.redirect_uri = callbackUrl();
        if (config.url) config.url = fill(config.url);
        if (config.args) config.args = config.args.map(fill);
        (option.inputs || []).forEach(input => {
            const value = (values[`${input.kind}:${input.name}`] || '').trim();
            if (input.kind === 'token') {
                if (value) config.auth.token = value;
            } else if (input.kind === 'header') {
                if (value) config.headers[input.name] = value; else delete config.headers[input.name];
            } else if (input.kind === 'env') {
                if (value) config.env[input.name] = value; else delete config.env[input.name];
            }
        });
        return config;
    }

    function missing(option, values) {
        const needed = (option.inputs || []).filter(input => input.required && !(values[`${input.kind}:${input.name}`] || '').trim()).map(input => input.name);
        (option.placeholders || []).forEach(key => { if (!(values[`placeholder:${key}`] || '').trim()) needed.push(key); });
        return needed;
    }

    function runsLine(config) {
        if (config.transport === 'stdio') return `Runs on this machine: ${[config.command, ...(config.args || [])].join(' ')}`;
        return `Connects to ${config.url}`;
    }

    // The review-and-attach form for one registry option.
    function optionForm(entry, option, context) {
        const form = node('div', 'catalog-attach');
        const values = {};
        const inputs = [];
        form.appendChild(node('p', 'catalog-runs mono', runsLine(option.config)));
        if (option.kind !== 'remote') {
            form.appendChild(node('p', 'form-hint', 'Published by a third party. Check the repository before you connect; the command runs only when the agent or Test uses the server.'));
        }
        const nameInput = textInput(entry.name);
        form.appendChild(field('Connection name', nameInput));

        let authSelect = null;
        if (option.kind === 'remote') {
            authSelect = document.createElement('select');
            [['none', 'None'], ['oauth', 'Sign in (OAuth)'], ['bearer', 'Token']].forEach(([value, label]) => {
                const opt = node('option', '', label); opt.value = value; authSelect.appendChild(opt);
            });
            authSelect.value = (option.config.auth || {}).type || 'none';
            const authHint = node('small', 'form-hint', option.auth_known ? '' : 'Checking how this server signs in…');
            const authField = field('Authentication', authSelect);
            authField.appendChild(authHint);
            form.appendChild(authField);
            if (!option.auth_known) {
                mcp.detectAuth(option.config.url).then(auth => {
                    authSelect.value = auth;
                    authHint.textContent = auth === 'oauth' ? 'This server uses sign-in. Attach, then select Sign in.' : auth === 'bearer' ? 'This server needs a token.' : 'No sign-in detected.';
                    syncToken();
                });
            }
        }

        const tokenInput = textInput('', { secret: true });
        const tokenField = field('Token', tokenInput, 'Encrypted at rest; never stored in agent memory.');
        const hasTokenInput = (option.inputs || []).some(input => input.kind === 'token');
        function syncToken() {
            tokenField.hidden = !(hasTokenInput || (authSelect && authSelect.value === 'bearer'));
        }
        form.appendChild(tokenField);
        syncToken();
        if (authSelect) authSelect.addEventListener('change', syncToken);

        (option.inputs || []).filter(input => input.kind !== 'token').forEach(input => {
            const el = textInput('', { secret: input.secret, placeholder: input.required ? 'Required' : 'Optional' });
            inputs.push([`${input.kind}:${input.name}`, el]);
            form.appendChild(field(input.name, el, input.description));
        });
        (option.placeholders || []).forEach(key => {
            const el = textInput('', { placeholder: 'Required' });
            inputs.push([`placeholder:${key}`, el]);
            form.appendChild(field(key, el, `Replaces {${key}} in the ${option.kind === 'remote' ? 'URL' : 'arguments'}.`));
        });

        const status = node('p', 'catalog-form-status');
        status.setAttribute('role', 'status');
        const submit = button(context.agentName ? `Attach to ${context.agentName}` : 'Attach', 'btn btn-primary btn-sm');
        const actions = node('div', 'catalog-actions');
        actions.append(submit, status);
        form.appendChild(actions);
        onEnter(form, () => submit.click());

        submit.addEventListener('click', async () => {
            inputs.forEach(([key, el]) => { values[key] = el.value; });
            values['token:token'] = tokenInput.value;
            const needed = missing(option, values);
            if (needed.length) { status.textContent = `Fill in ${needed.join(', ')}.`; return; }
            submit.disabled = true;
            status.textContent = 'Attaching…';
            try {
                const config = serverConfig(option, nameInput.value.trim(), authSelect && authSelect.value, values);
                if (config.auth.type === 'bearer' && tokenInput.value.trim()) config.auth.token = tokenInput.value.trim();
                const body = await mcp.attachServer(context.agentId, config);
                attached(form, body.server, context);
            } catch (error) {
                status.textContent = error.message;
                submit.disabled = false;
            }
        });
        return form;
    }

    function presetForm(preset, context) {
        const form = node('div', 'catalog-attach');
        form.appendChild(node('p', 'form-hint', preset.setup));
        const nameInput = textInput(preset.name);
        form.appendChild(field('Connection name', nameInput));
        const clientId = textInput('');
        const clientSecret = textInput('', { secret: true });
        if (preset.needs_client) {
            form.appendChild(field('OAuth client ID', clientId, `Register ${callbackUrl()} as a redirect URI.`));
            form.appendChild(field('OAuth client secret', clientSecret, 'Encrypted at rest.'));
        }
        const status = node('p', 'catalog-form-status');
        status.setAttribute('role', 'status');
        const submit = button(context.agentName ? `Attach to ${context.agentName}` : 'Attach', 'btn btn-primary btn-sm');
        const actions = node('div', 'catalog-actions');
        actions.append(submit, link('Setup guide', preset.docs), status);
        form.appendChild(actions);
        onEnter(form, () => submit.click());
        submit.addEventListener('click', async () => {
            submit.disabled = true;
            status.textContent = 'Attaching…';
            try {
                const body = await mcp.attachPreset(context.agentId, preset.key, {
                    name: nameInput.value.trim(),
                    client_id: clientId.value.trim(),
                    client_secret: clientSecret.value.trim(),
                });
                attached(form, body.server, context);
            } catch (error) {
                status.textContent = error.message;
                submit.disabled = false;
            }
        });
        return form;
    }

    // After attaching: offer sign-in for OAuth servers, then tell the page.
    function attached(form, server, context) {
        form.textContent = '';
        const done = node('div', 'catalog-done');
        done.appendChild(node('p', '', `Attached ${server.name}.`));
        const auth = (server.auth || {}).type;
        if (auth === 'oauth') {
            const signIn = button(`Sign in to ${server.name}`, 'btn btn-primary btn-sm');
            const note = node('p', 'form-hint', 'Opens the provider’s sign-in page and returns to MCP connections.');
            signIn.addEventListener('click', async () => {
                signIn.disabled = true;
                try {
                    const outcome = await mcp.signIn(context.agentId, server.name);
                    if (outcome === 'already') note.textContent = 'Already signed in.';
                } catch (error) {
                    note.textContent = error.message;
                    signIn.disabled = false;
                }
            });
            done.append(signIn, note);
        }
        form.appendChild(done);
        if (context.onAttached) context.onAttached(server);
    }

    function toggleForm(item, trigger, build) {
        const open = item.querySelector('.catalog-attach, .catalog-done');
        if (open && trigger.getAttribute('aria-expanded') === 'true') {
            open.remove();
            trigger.setAttribute('aria-expanded', 'false');
            return;
        }
        item.querySelectorAll('.catalog-attach').forEach(el => el.remove());
        item.querySelectorAll('[aria-expanded]').forEach(el => el.setAttribute('aria-expanded', 'false'));
        trigger.setAttribute('aria-expanded', 'true');
        const form = build();
        item.appendChild(form);
        const first = form.querySelector('input');
        if (first) first.focus({ preventScroll: true });
    }

    function presetItem(preset, context) {
        const item = node('li', 'catalog-item');
        const head = node('div', 'catalog-item-head');
        const title = node('div', 'catalog-item-title');
        title.append(node('strong', '', preset.title), node('span', 'pill pill--info', 'Official'));
        const pick = button('Attach');
        pick.setAttribute('aria-expanded', 'false');
        pick.addEventListener('click', () => toggleForm(item, pick, () => presetForm(preset, context)));
        head.append(title, pick);
        item.append(head, node('p', 'catalog-desc', preset.description), node('p', 'catalog-meta mono', preset.url));
        return item;
    }

    function serverItem(entry, context) {
        const item = node('li', 'catalog-item');
        const head = node('div', 'catalog-item-head');
        const title = node('div', 'catalog-item-title');
        title.append(node('strong', '', entry.title), node('span', 'catalog-meta mono', entry.registry_name));
        const picks = node('div', 'catalog-options');
        entry.options.forEach(option => {
            const pick = button(option.label);
            pick.setAttribute('aria-expanded', 'false');
            pick.title = runsLine(option.config);
            pick.addEventListener('click', () => toggleForm(item, pick, () => optionForm(entry, option, context)));
            picks.appendChild(pick);
        });
        head.append(title, picks);
        item.append(head);
        if (entry.description) item.appendChild(node('p', 'catalog-desc', entry.description));
        const links = node('p', 'catalog-links');
        if (entry.repository) links.appendChild(link('Repository', entry.repository));
        if (entry.website) links.appendChild(link('Website', entry.website));
        if (links.childNodes.length) item.appendChild(links);
        return item;
    }

    // --------------------------------------------------------------- skills

    const skills = {
        search(query) {
            return request(`/vercel-skills/api/search?q=${enc(query)}&limit=20`);
        },
        list(agentId) {
            return request(`/api/agents/${enc(agentId)}/skills`);
        },
        attach(agentId, skill) {
            return postJson(`/api/agents/${enc(agentId)}/skills`, { repo: skill.repo, skill_name: skill.skill_name || '' });
        },
        detach(agentId, path) {
            return request(`/api/agents/${enc(agentId)}/skills?path=${enc(path)}`, { method: 'DELETE' });
        },
    };

    function skillItem(skill, context) {
        const item = node('li', 'catalog-item');
        const head = node('div', 'catalog-item-head');
        const title = node('div', 'catalog-item-title');
        title.append(node('strong', '', skill.skill_name || skill.name), node('span', 'catalog-meta mono', skill.repo));
        const pick = button(context.agentName ? `Attach to ${context.agentName}` : 'Attach');
        const status = node('p', 'catalog-form-status');
        status.setAttribute('role', 'status');
        pick.addEventListener('click', async () => {
            pick.disabled = true;
            status.textContent = 'Saving the skill…';
            try {
                const body = await skills.attach(context.agentId, skill);
                status.textContent = `Attached. The agent lists it with list_skills.`;
                pick.textContent = 'Attached';
                if (context.onAttached) context.onAttached(body.skill, body.skills);
            } catch (error) {
                const choices = (error.body && error.body.available_skills) || [];
                status.textContent = error.message + (choices.length ? ` Try: ${choices.slice(0, 6).map(c => c.name).join(', ')}.` : '');
                pick.disabled = false;
            }
        });
        head.append(title, pick);
        item.append(head);
        const meta = [];
        if (skill.installs) meta.push(`${skill.installs.toLocaleString()} installs`);
        if (skill.stars) meta.push(`${skill.stars.toLocaleString()} stars`);
        if (skill.description) item.appendChild(node('p', 'catalog-desc', skill.description));
        const links = node('p', 'catalog-links');
        if (meta.length) links.appendChild(node('span', 'catalog-meta', meta.join(' · ')));
        if (skill.html_url) links.appendChild(link('Details', skill.html_url));
        if (skill.repo_url) links.appendChild(link('Repository', skill.repo_url));
        item.append(links, status);
        return item;
    }

    // ---------------------------------------------------------------- mount

    function mount(root, options) {
        if (!root) return null;
        const kind = root.dataset.catalog;
        const form = root.querySelector('[data-catalog-form]');
        const input = form && form.querySelector('input');
        const go = form && form.querySelector('button');
        const list = root.querySelector('[data-catalog-results]');
        const status = root.querySelector('[data-catalog-status]');
        const more = root.querySelector('[data-catalog-more]');
        const context = {
            agentId: root.dataset.agentId || (options && options.agentId),
            agentName: root.dataset.agentName || (options && options.agentName) || '',
            onAttached: options && options.onAttached,
        };
        let cursor = null;
        let query = '';

        async function run(append) {
            if (!context.agentId) { status.textContent = 'Choose an agent first.'; return; }
            status.textContent = !query ? ''
                : kind === 'mcp' ? `Searching the MCP Registry for “${query}”… this can take up to a minute when the registry is busy.`
                : `Searching for “${query}”…`;
            try {
                if (kind === 'skills') {
                    if (!query) { list.textContent = ''; status.textContent = ''; return; }
                    const body = await skills.search(query);
                    list.textContent = '';
                    (body.skills || []).forEach(skill => list.appendChild(skillItem(skill, context)));
                    status.textContent = (body.skills || []).length ? `${body.skills.length} skills from ${body.source === 'github' ? 'GitHub' : 'skills.sh'}, most installed first.` : `No skills matched “${query}”.`;
                    return;
                }
                const body = await mcp.search(query, append ? cursor : null);
                if (!append) list.textContent = '';
                (body.presets || []).forEach(preset => list.appendChild(presetItem(preset, context)));
                (body.servers || []).forEach(entry => list.appendChild(serverItem(entry, context)));
                cursor = body.next_cursor || null;
                if (more) more.hidden = !cursor;
                const count = list.children.length;
                status.textContent = !query ? 'Official servers. Search the MCP Registry for more.'
                    : count ? `${count} result${count === 1 ? '' : 's'} for “${query}”. Registry servers are community-published.`
                    : `Nothing matched “${query}”.`;
            } catch (error) {
                status.textContent = error.message;
            }
        }

        function start() { query = input.value.trim(); cursor = null; run(false); }
        if (go) go.addEventListener('click', start);
        if (form) onEnter(form, start);
        if (more) more.addEventListener('click', () => run(true));
        if (kind === 'mcp') run(false);
        return {
            setAgent(agentId, agentName) { context.agentId = agentId; context.agentName = agentName || ''; },
            search(value) { if (input) input.value = value; query = value; cursor = null; return run(false); },
        };
    }

    window.MemorizzCatalog = { mount, mcp, skills };
})();

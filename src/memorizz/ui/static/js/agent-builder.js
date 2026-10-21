// The agent form's overview: section navigation with each section's status,
// a summary of the configuration, and the same agent as Python SDK code.
// Everything is read from the form, so it follows edits and templates.
(function () {
    const form = document.querySelector('form.agent-form');
    const summary = document.getElementById('builder-summary');
    const code = document.getElementById('builder-code');
    if (!form || !summary || !code) return;
    const context = window.AGENT_BUILDER || {};
    const SECRET = /api[_-]?key|secret|token|password|passphrase|authorization|credential/i;
    const PROVIDERS = {
        openai: 'OpenAI', anthropic: 'Anthropic', azure: 'Azure OpenAI', ollama: 'Ollama',
        huggingface: 'HuggingFace', mlx: 'MLX', 'local-openai': 'Local OpenAI-compatible',
    };

    function field(name) { return form.querySelector('[name="' + name + '"]'); }
    function value(name) { const el = field(name); return el ? String(el.value || '').trim() : ''; }
    function checked(name) { const el = field(name); return Boolean(el && el.checked); }
    function optionLabel(name) {
        const el = field(name);
        if (!el || el.tagName !== 'SELECT' || !el.value) return '';
        return (el.options[el.selectedIndex]?.textContent || el.value).replace(/\s*\(.*\)\s*$/, '').trim();
    }

    // The form's configuration, in the terms the SDK uses.
    function read() {
        const provider = value('llm_provider') || 'openai';
        const model = value('llm_model');
        let advanced = {};
        try { advanced = JSON.parse(value('llm_config_json') || '{}') || {}; } catch (_) { advanced = {}; }
        const baseUrl = (document.getElementById('llm_base_url')?.value || '').trim();
        if (baseUrl) advanced.base_url = baseUrl;
        if (provider === 'ollama' && document.getElementById('ollama_thinking')?.checked) {
            advanced.additional_config = Object.assign({}, advanced.additional_config, {think: true});
        }
        const background = value('background_model');
        const split = background.indexOf(':');
        const mode = value('application_mode') || 'assistant';
        const continuous = Boolean(document.getElementById('continuous_mode')?.checked);
        return {
            name: value('agent_name'),
            instruction: (field('instruction')?.value || '').replace(/\s+$/, ''),
            mode: mode,
            provider: provider,
            model: model,
            advanced: withoutSecrets(advanced),
            background: background ? {provider: split > 0 ? background.slice(0, split) : provider, model: split > 0 ? background.slice(split + 1) : background} : null,
            persona: {name: value('persona_name'), role: value('persona_role'), goals: value('persona_goals'), background: value('persona_background')},
            memoryIds: value('memory_ids').split(/[\n,]/).map(function (id) { return id.trim(); }).filter(Boolean),
            entity: checked('enable_entity_memory'),
            workflow: checked('enable_workflow_memory'),
            semanticCache: checked('semantic_cache'),
            toolCache: checked('tool_cache'),
            sealed: checked('sealed_conversations'),
            entityScope: checked('sealed_conversations') ? 'thread' : (value('entity_scope') || 'memory'),
            consolidation: checked('memory_consolidation') ? {turns: parseInt(value('consolidation_every_n_turns'), 10) || 20, autoApply: checked('consolidation_auto_apply')} : null,
            forgetting: Array.from(form.querySelectorAll('[name^="forgetting_"]')).some(function (el) { return String(el.value || '').trim(); }),
            continual: checked('continual_learning'),
            controlPlane: checked('learning_control_plane'),
            toolAccess: value('tool_access') || 'private',
            internet: value('internet_provider'),
            sandbox: value('sandbox_provider'),
            browser: value('browser_control_provider'),
            skills: value('skills_marketplace_provider'),
            maxSteps: continuous ? 0 : (parseInt(value('max_steps'), 10) || 20),
            harnessMode: value('meta_harness_mode'),
            harness: value('default_harness') || 'auto',
            harnessModel: value('harness_model'),
            harnessWorkspace: value('harness_workspace'),
            delegates: Array.from(form.querySelectorAll('input[name="delegate_ids"]:checked')).map(function (el) { return el.value; }),
            selfAware: checked('self_aware'),
            selfAwareRoots: value('self_aware_root_paths').split(/[\n,]/).map(function (p) { return p.trim(); }).filter(Boolean),
            selfAwareWrites: checked('self_aware_allow_writes'),
            selfAwareDeletes: checked('self_aware_allow_deletes'),
            automations: checked('automations_enabled'),
            timezone: value('default_timezone'),
            whatsapp: checked('whatsapp_enabled'),
            mcp: mcpConnections(),
        };
    }

    // The agent's MCP servers, and those a template attaches when it is saved.
    function mcpConnections() {
        const key = (document.getElementById('template_key')?.value || '').trim();
        const template = (window.AGENT_FORM_TEMPLATES || []).find(function (item) { return item.key === key; });
        return {saved: context.mcp_servers || [], pending: template ? template.connections || [] : []};
    }

    function withoutSecrets(item) {
        if (Array.isArray(item)) return item.map(withoutSecrets);
        if (!item || typeof item !== 'object') return item;
        const out = {};
        Object.keys(item).forEach(function (key) { if (!SECRET.test(key)) out[key] = withoutSecrets(item[key]); });
        return out;
    }

    // Mirrors the server: the mode's defaults, the entity and workflow
    // toggles, and summaries always.
    function memoryTypes(config) {
        const defaults = ((context.mode_memory_types || {})[config.mode] || []).slice();
        let types = defaults.slice();
        const toggle = function (name, on) {
            if (on && !types.includes(name)) types.push(name);
            if (!on) types = types.filter(function (t) { return t !== name; });
        };
        toggle('entity_memory', config.entity);
        toggle('workflow_memory', config.workflow);
        if (!types.includes('summaries')) types.push('summaries');
        return {types: types, changed: types.join() !== defaults.join()};
    }

    function words(type) { return type.replace(/_memory$/, '').replace(/_/g, ' '); }

    // --- Section status -----------------------------------------------------

    function statuses(config) {
        const tools = [
            (config.mcp.saved.length || config.mcp.pending.length) && 'MCP',
            config.internet && 'Web search',
            config.sandbox && 'Sandbox',
            config.browser && 'Browser',
            config.skills && 'Skills',
        ].filter(Boolean);
        const memory = [
            config.entity && 'Entity',
            config.workflow && 'Workflow',
            config.sealed ? 'Sealed' : (config.entityScope !== 'memory' ? 'Facts per conversation' : ''),
            config.consolidation && 'Consolidation',
        ].filter(Boolean);
        const runtime = [
            config.maxSteps === 0 ? 'Unlimited steps' : config.maxSteps + ' steps',
            config.harnessMode && (config.harnessMode === 'runtime' ? 'Harness runtime' : 'Harness tools'),
            config.delegates.length && config.delegates.length + ' delegate' + (config.delegates.length === 1 ? '' : 's'),
            config.selfAware && 'Codebase',
        ].filter(Boolean);
        return {
            identity: config.name || 'Unnamed',
            model: config.model || 'No model',
            persona: config.persona.name || config.persona.role || 'None',
            memory: memory.join(' · ') || 'Conversation only',
            learning: [config.continual && 'Skills', config.controlPlane && 'Control plane'].filter(Boolean).join(' · ') || 'Off',
            tools: tools.join(' · ') || 'None',
            runtime: runtime.join(' · '),
            channels: [config.automations ? 'Automations' : '', config.whatsapp ? 'WhatsApp' : ''].filter(Boolean).join(' · ') || 'Off',
        };
    }

    // --- Summary --------------------------------------------------------------

    function renderSummary(config) {
        const types = memoryTypes(config).types;
        const rows = [
            ['group', 'Model'],
            ['Chat model', config.model ? (PROVIDERS[config.provider] || config.provider) + ' · ' + config.model : 'Not set'],
            ['Background', config.background ? (PROVIDERS[config.background.provider] || config.background.provider) + ' · ' + config.background.model : 'Same as chat model'],
            ['group', 'Memory & data'],
            ['Memory types', types.map(words).join(', ')],
            ['Memory', config.memoryIds.length ? config.memoryIds.join(', ') : 'Its own, created on first run'],
            ['Conversations', config.sealed ? 'Sealed: nothing crosses between them' : 'Recall across a memory’s conversations'],
            ['Entity facts', {memory: 'Shared within a memory', thread: 'Kept to each conversation', disabled: 'Not recalled automatically'}[config.entityScope]],
            ['Consolidation', config.consolidation ? 'Every ' + config.consolidation.turns + ' turns, ' + (config.consolidation.autoApply ? 'applied automatically' : 'waits for review') : 'Off'],
            ['Caching', [config.semanticCache && 'answers', config.toolCache && 'tool results'].filter(Boolean).join(', ') || 'Off'],
            ['group', 'Capabilities'],
            ['Tool access', {private: 'Private', public: 'Public', global: 'Global'}[config.toolAccess] || config.toolAccess],
            ['MCP servers', config.mcp.saved.length ? config.mcp.saved.join(', ') : (config.mcp.pending.length ? config.mcp.pending.join(', ') + ', attached on save' : 'None')],
            ['Web search', config.internet ? optionLabel('internet_provider') : 'Off'],
            ['Code sandbox', config.sandbox ? optionLabel('sandbox_provider') : 'Off'],
            ['Browser control', config.browser ? optionLabel('browser_control_provider') + ', each task approved' : 'Off'],
            ['Skills', [config.continual && 'learned from work', config.skills && optionLabel('skills_marketplace_provider')].filter(Boolean).join(', ') || 'Off'],
            ['group', 'Runtime'],
            ['Tool rounds', config.maxSteps === 0 ? 'Unlimited' : 'Up to ' + config.maxSteps],
            ['Harness', config.harnessMode ? optionLabel('meta_harness_mode') + ' · ' + optionLabel('default_harness') : 'Off'],
            ['Delegates', config.delegates.length ? String(config.delegates.length) : 'None'],
            ['Automations', config.automations ? 'On' + (config.timezone ? ' · ' + config.timezone : '') : 'Off'],
        ];
        summary.innerHTML = rows.map(function (row) {
            if (row[0] === 'group') return '<dt class="builder-summary-group">' + escapeHtml(row[1]) + '</dt>';
            return '<dt>' + escapeHtml(row[0]) + '</dt><dd>' + escapeHtml(row[1] || '') + '</dd>';
        }).join('');
    }

    // --- Python ----------------------------------------------------------------

    function pyString(text) {
        text = String(text);
        if (!text.includes('\n')) return JSON.stringify(text);
        let body = text.replace(/\\/g, '\\\\').replace(/"""/g, '\\"\\"\\"');
        if (body.endsWith('"')) body = body.slice(0, -1) + '\\"';
        return '"""\\\n' + body + '"""';
    }

    function py(item) {
        if (item === null || item === undefined) return 'None';
        if (item === true) return 'True';
        if (item === false) return 'False';
        if (typeof item === 'number') return String(item);
        if (typeof item === 'string') return pyString(item);
        if (Array.isArray(item)) return '[' + item.map(py).join(', ') + ']';
        return '{' + Object.keys(item).map(function (key) { return JSON.stringify(key) + ': ' + py(item[key]); }).join(', ') + '}';
    }

    function storeCode(store, imports) {
        if (store.type === 'mongodb') {
            imports.add('MongoDBProvider').add('MongoDBConfig');
            return {env: true, line: 'MongoDBProvider(MongoDBConfig(uri=os.environ["MONGODB_URI"], db_name=' + py(store.db_name || 'memorizz') + '))'};
        }
        if (store.type === 'oracle') {
            imports.add('OracleProvider').add('OracleConfig');
            return {env: true, line: 'OracleProvider(\n    OracleConfig(user=' + py(store.user || 'MEMORIZZ') + ', password=os.environ["ORACLE_PASSWORD"], dsn=' + py(store.dsn || 'localhost:1521/FREEPDB1') + ')\n)'};
        }
        imports.add('FileSystemProvider').add('FileSystemConfig');
        return {env: false, line: 'FileSystemProvider(FileSystemConfig(root_path=' + py(store.path || '~/.memorizz/memory') + '))'};
    }

    function renderCode(config) {
        const imports = new Set(['MemAgent']);
        const extra = [];
        const args = [];
        // Multi-line text becomes a constant above the agent, so indenting
        // the call never changes it.
        const constants = [];
        const text = function (name, item) {
            if (!String(item).includes('\n')) return py(item);
            constants.push(name + ' = ' + pyString(item));
            return name;
        };
        const arg = function (name, code) { args.push('    ' + name + '=' + code.replace(/\n/g, '\n    ') + ','); };
        if (config.name) arg('name', py(config.name));
        if (config.instruction) arg('instruction', text('INSTRUCTION', config.instruction));
        const persona = config.persona;
        if (persona.name || persona.role || persona.goals || persona.background) {
            imports.add('Persona');
            const fields = ['name=' + py(persona.name || config.name || 'Assistant')];
            if (persona.role) fields.push('role=' + py(persona.role));
            if (persona.goals) fields.push('goals=' + text('PERSONA_GOALS', persona.goals));
            if (persona.background) fields.push('background=' + text('PERSONA_BACKGROUND', persona.background));
            arg('persona', 'Persona(\n    ' + fields.join(',\n    ') + ',\n)');
        }
        arg('llm_config', py(Object.assign({provider: config.provider, model: config.model}, config.advanced)));
        if (config.background) arg('background_llm_config', py(config.background));
        if (config.mode !== 'assistant') arg('application_mode', py(config.mode));
        const types = memoryTypes(config);
        if (types.changed) arg('memory_types', py(types.types));
        arg('memory_provider', 'memory_provider');
        if (config.memoryIds.length) arg('memory_ids', py(config.memoryIds));
        if (config.maxSteps !== 20) arg('max_steps', String(config.maxSteps));
        if (config.toolAccess !== 'private') arg('tool_access', py(config.toolAccess));
        if (config.semanticCache) arg('semantic_cache', 'True');
        if (config.toolCache) arg('tool_cache', 'True');
        if (config.sealed) {
            arg('retrieval_policy', '"sealed"');
        } else if (config.entityScope !== 'memory') {
            imports.add('RetrievalPolicy');
            arg('retrieval_policy', 'RetrievalPolicy(entity_scope=' + py(config.entityScope) + ')');
        }
        if (config.consolidation) arg('memory_consolidation', py({every_n_turns: config.consolidation.turns, auto_apply: config.consolidation.autoApply}));
        if (config.continual) arg('continual_learning', 'True');
        if (config.controlPlane) arg('learning_control_plane', 'True');
        if (config.internet) {
            extra.push('from memorizz.internet_access import create_internet_access_provider');
            arg('internet_access_provider', 'create_internet_access_provider(' + py(config.internet) + ')');
        }
        if (config.sandbox) arg('sandbox_provider', py(config.sandbox));
        if (config.browser) arg('browser_control', py(config.browser));
        if (config.skills) arg('skills_marketplace_provider', py(config.skills));
        if (config.harnessMode) {
            arg('meta_harness_mode', py(config.harnessMode));
            arg('default_harness', py(config.harness));
            const harnessConfig = {};
            if (config.harnessModel) harnessConfig.model = config.harnessModel;
            if (config.harnessWorkspace) harnessConfig.workspace = config.harnessWorkspace;
            if (Object.keys(harnessConfig).length) arg('harness_config', py(harnessConfig));
        }
        if (config.selfAware) {
            arg('self_aware', 'True');
            arg('self_aware_config', py({root_paths: config.selfAwareRoots, allow_writes: config.selfAwareWrites, allow_deletes: config.selfAwareWrites && config.selfAwareDeletes}));
        }
        if (!config.automations) arg('automations_enabled', 'False');
        if (config.timezone) arg('default_timezone', py(config.timezone));

        const store = storeCode(context.store || {}, imports);
        const lines = [];
        if (store.env) lines.push('import os', '');
        lines.push('from memorizz import ' + Array.from(imports).join(', '));
        lines.push.apply(lines, extra);
        lines.push('', 'memory_provider = ' + store.line);
        if (context.agent_id) {
            lines.push('', '# This agent is saved as ' + context.agent_id + '. To use it as saved:',
                '# agent = MemAgent.load(' + py(context.agent_id) + ', memory_provider=memory_provider)');
        }
        if (constants.length) lines.push('', constants.join('\n\n'));
        lines.push('', 'agent = MemAgent(');
        lines.push.apply(lines, args);
        lines.push(')');
        if (config.delegates.length) {
            lines.push('# Delegates: load them with MemAgent.load(id, memory_provider=memory_provider)',
                '# and pass delegates=[...]: ' + config.delegates.join(', '));
        }
        if (config.mcp.saved.length || config.mcp.pending.length) {
            lines.push('# MCP servers (' + (config.mcp.saved.length ? config.mcp.saved : config.mcp.pending).join(', ') + '): sign in on MCP connections,',
                '# or pass mcp_servers=[...] configs from memorizz.mcp.catalog.preset_config().');
        }
        if (config.forgetting) lines.push('# Forgetting overrides are set on RetrievalPolicy(scoring=..., retention=...).');
        lines.push('agent.save()', '', 'print(agent.run("Hello!"))');
        code.textContent = lines.join('\n');
    }

    // --- Navigation --------------------------------------------------------------

    const sections = Array.from(form.querySelectorAll('details.builder-section'));
    document.querySelectorAll('[data-nav]').forEach(function (link) {
        link.addEventListener('click', function (event) {
            const section = document.getElementById('section-' + link.dataset.nav);
            if (!section) return;
            event.preventDefault();
            section.open = true;
            section.scrollIntoView({behavior: 'smooth', block: 'start'});
            section.querySelector('summary')?.focus({preventScroll: true});
        });
    });
    if ('IntersectionObserver' in window) {
        const visible = new Map();
        const observer = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) { visible.set(entry.target.id, entry.isIntersecting ? entry.boundingClientRect.top : null); });
            let current = null;
            let best = Infinity;
            visible.forEach(function (top, id) {
                if (top !== null && Math.abs(top) < best) { best = Math.abs(top); current = id; }
            });
            document.querySelectorAll('[data-nav]').forEach(function (link) {
                link.classList.toggle('is-active', 'section-' + link.dataset.nav === current);
            });
        }, {rootMargin: '0px 0px -55% 0px'});
        sections.forEach(function (section) { observer.observe(section); });
    }

    // --- Copy --------------------------------------------------------------------

    // The wide view is a dialog, so it sits above the app's sidebar.
    const dialog = document.getElementById('builder-code-dialog');
    const full = document.getElementById('builder-code-full');
    document.getElementById('builder-code-expand')?.addEventListener('click', function () {
        full.textContent = code.textContent;
        dialog.showModal();
    });
    document.getElementById('builder-code-dialog-close')?.addEventListener('click', function () { dialog.close(); });
    dialog?.addEventListener('click', function (event) {
        if (event.target === dialog) dialog.close();  // a click on the backdrop
    });

    async function copyCode(button, source) {
        try {
            await navigator.clipboard.writeText(source.textContent);
            button.textContent = 'Copied';
        } catch (_) {
            const range = document.createRange();
            range.selectNodeContents(source);
            const selection = window.getSelection();
            selection.removeAllRanges();
            selection.addRange(range);
            button.textContent = 'Press ⌘C';
        }
        setTimeout(function () { button.textContent = 'Copy'; }, 1600);
    }
    const copy = document.getElementById('builder-code-copy');
    copy?.addEventListener('click', function () { copyCode(copy, code); });
    const dialogCopy = document.getElementById('builder-code-dialog-copy');
    dialogCopy?.addEventListener('click', function () { copyCode(dialogCopy, full); });

    // --- Refresh -----------------------------------------------------------------

    let pending = false;
    function refresh() {
        if (pending) return;
        pending = true;
        requestAnimationFrame(function () {
            pending = false;
            const config = read();
            const status = statuses(config);
            Object.keys(status).forEach(function (key) {
                document.querySelectorAll('[data-section-status="' + key + '"], [data-nav-status="' + key + '"]').forEach(function (el) {
                    el.textContent = status[key];
                    el.title = status[key];
                });
            });
            renderSummary(config);
            renderCode(config);
        });
    }
    form.addEventListener('input', refresh);
    form.addEventListener('change', refresh);
    document.addEventListener('DOMContentLoaded', refresh);
    refresh();
})();

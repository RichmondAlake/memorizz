/*
 * Local-model availability widget shared by Settings, the playground and the
 * agent form: one fetch of /api/ollama/installed or /api/huggingface/installed,
 * one "is this model installed" rule, one set of success / warning banners
 * and (optionally) the "Manage … models" pull/delete panel.
 *
 * Each page keeps its own copy, element id and data-* attribute names:
 *
 *   const widget = window.MemorizzLocalModels.mount({
 *       statusId: 'local-model-status',       // the banner container
 *       provider: function () { ... },        // current provider value
 *       model: function () { ... },           // current model value
 *       providers: ['ollama', 'huggingface'], // optional; default adds 'mlx'
 *       refreshAttr: 'data-local-refresh',    // attribute on the refresh link
 *       refresh: pageLevelCheck,              // optional; what the refresh
 *                                             // link, pull and delete re-run
 *                                             // (defaults to widget.check)
 *       hostFallback: 'localhost:11434',      // optional; shown when the
 *                                             // Ollama reply has no host
 *       copy: {
 *           warningPrefix: 'Pre-flight: ',    // optional; inside <strong>
 *           refreshText: 'refresh',
 *           ollamaDownRefreshText: 'refresh this check', // optional; link text
 *                                             // in the "Ollama is not running"
 *                                             // banner (defaults to refreshText)
 *           ollamaMissing: 'Run {pull}, or {browse}. Then {refresh}.',
 *           huggingfaceUnavailable: 'Install it with ..., then {refresh}.',
 *           huggingfaceMissing: 'Pre-download with {download}, or {card}. Then {refresh}.',
 *       },
 *       manage: {                             // optional: the pull/delete panel
 *           catalog: PROVIDER_MODELS,         // provider -> [{value, label}]
 *           attrPrefix: 'data-pg-',           // -> data-pg-action, -provider, -model, -select-model
 *           selectTitle: 'Use this model in the playground',
 *           onSelect: function (value) { ... },
 *       },
 *   });
 *   widget.check();     // re-check now; resolves true once the panel reflects
 *                       // the current inputs, false on a fetch error or when
 *                       // a newer check superseded it
 *   widget.schedule();  // debounced (350ms) re-check for keystrokes
 *
 * Copy slots: {pull} = <code>ollama pull …</code>, {browse} = the
 * ollama.com/library link, {download} = <code>huggingface-cli download …</code>,
 * {card} = the huggingface.co model-card link, {refresh} = the refresh link.
 *
 * Depends on escapeHtml(), trashIconSvg() and downloadIconSvg() (base.html)
 * and, at click time only, handleLocalModelAction() (app.js).
 */
(function () {
    'use strict';

    const CHECKING_LABEL = {
        ollama: 'local Ollama',
        huggingface: 'HuggingFace cache',
        mlx: 'MLX cache (HuggingFace)',
    };
    const MANAGE_LABEL = { ollama: 'Ollama', huggingface: 'HuggingFace' };
    const DEFAULT_PROVIDERS = ['ollama', 'huggingface', 'mlx'];

    function fill(template, slots) {
        return String(template || '').replace(/\{(\w+)\}/g, function (match, key) {
            return Object.prototype.hasOwnProperty.call(slots, key) ? slots[key] : match;
        });
    }

    function mount(options) {
        const opts = options || {};
        const copy = opts.copy || {};
        const providers = opts.providers || DEFAULT_PROVIDERS;
        const manage = opts.manage || null;
        const hostFallback = opts.hostFallback || '';
        const warningPrefix = copy.warningPrefix || '';
        let seq = 0;
        let debounce = null;

        function refresh() {
            return (opts.refresh || check)();
        }

        function schedule() {
            if (debounce) clearTimeout(debounce);
            debounce = setTimeout(refresh, 350);
        }

        function refreshLink(text) {
            return '<a href="#" ' + opts.refreshAttr + '>' + (text || copy.refreshText) + '</a>';
        }

        function wireRefreshLink(scope) {
            const link = scope.querySelector('[' + opts.refreshAttr + ']');
            if (!link) return;
            link.addEventListener('click', function (ev) {
                ev.preventDefault();
                refresh();
            });
        }

        // Race protection: if a second fetch starts before the first returns
        // (e.g. the user flips providers quickly) only the most recent paints.
        async function check() {
            const statusEl = document.getElementById(opts.statusId);
            if (!statusEl) return false;

            const provider = opts.provider();
            const model = opts.model();
            if (providers.indexOf(provider) === -1 || !model) {
                statusEl.style.display = 'none';
                statusEl.innerHTML = '';
                return true;
            }

            const mine = ++seq;
            statusEl.style.display = 'block';
            statusEl.innerHTML = '<p class="form-hint">Checking ' + CHECKING_LABEL[provider] + '…</p>';

            // MLX reads the same cache directory as huggingface_hub, so the
            // HuggingFace endpoint covers it.
            const url = provider === 'ollama'
                ? '/api/ollama/installed'
                : '/api/huggingface/installed';

            let info;
            try {
                const res = await fetch(url, { cache: 'no-store' });
                info = await res.json();
            } catch (err) {
                if (mine !== seq) return false;
                statusEl.innerHTML =
                    '<div class="warning-message"><strong>Could not reach the UI server</strong>: '
                    + escapeHtml(String(err)) + '</div>';
                return false;
            }
            if (mine !== seq) return false;

            if (provider === 'ollama') {
                renderOllama(statusEl, model, info);
            } else {
                renderHuggingFace(statusEl, model, info);
            }
            return true;
        }

        function renderOllama(statusEl, model, info) {
            const host = escapeHtml(info.host || hostFallback);
            if (!info.reachable) {
                statusEl.innerHTML =
                    '<div class="warning-message">'
                    + '<strong>Ollama is not running</strong> at <code>' + host + '</code>'
                    + (info.error ? ' &mdash; ' + escapeHtml(info.error) : '')
                    + '.<br>Install it from <a href="https://ollama.com/download" target="_blank" rel="noopener">ollama.com/download</a> and start it, then '
                    + refreshLink(copy.ollamaDownRefreshText) + '.'
                    + '</div>';
                wireRefreshLink(statusEl);
                return;
            }

            // Ollama treats a bare name as `:latest`; match both directions so
            // `llama3.1` counts as installed when the user pulled `llama3.1:latest`.
            const installed = (info.models || []).some(function (m) {
                return m === model
                    || m === model + ':latest'
                    || m + ':latest' === model;
            });

            const base = model.split(':')[0];
            const libUrl = 'https://ollama.com/library/' + encodeURIComponent(base);

            const banner = installed
                ? '<div class="success-message">'
                    + '<strong><code>' + escapeHtml(model) + '</code> is available offline</strong>'
                    + ' on Ollama at <code>' + host + '</code>.'
                    + '</div>'
                : '<div class="warning-message">'
                    + '<strong>' + warningPrefix + '<code>' + escapeHtml(model) + '</code> is not installed locally.</strong><br>'
                    + fill(copy.ollamaMissing, {
                        pull: '<code>ollama pull ' + escapeHtml(model) + '</code>',
                        browse: '<a href="' + libUrl + '" target="_blank" rel="noopener">browse <code>'
                            + escapeHtml(base) + '</code> on ollama.com</a>',
                        refresh: refreshLink(),
                    })
                    + '</div>';

            statusEl.innerHTML = banner + manageListHtml(info.models, 'ollama');
            wireRefreshLink(statusEl);
        }

        function renderHuggingFace(statusEl, model, info) {
            if (!info.available) {
                statusEl.innerHTML =
                    '<div class="warning-message">'
                    + '<strong>HuggingFace cache unavailable</strong>'
                    + (info.error ? ' &mdash; ' + escapeHtml(info.error) : '')
                    + '.<br>' + fill(copy.huggingfaceUnavailable, { refresh: refreshLink() })
                    + '</div>';
                wireRefreshLink(statusEl);
                return;
            }

            // HF repo IDs are case-sensitive; only an exact match counts.
            const cached = (info.models || []).indexOf(model) !== -1;
            const repoUrl = 'https://huggingface.co/'
                + model.split('/').map(encodeURIComponent).join('/');
            const cacheLoc = info.cache_dir || 'local HF cache';

            const banner = cached
                ? '<div class="success-message">'
                    + '<strong><code>' + escapeHtml(model) + '</code> is cached locally</strong>'
                    + ' at <code>' + escapeHtml(cacheLoc) + '</code>.'
                    + '</div>'
                : '<div class="warning-message">'
                    + '<strong>' + warningPrefix + '<code>' + escapeHtml(model) + '</code> is not cached locally.</strong><br>'
                    + fill(copy.huggingfaceMissing, {
                        download: '<code>huggingface-cli download ' + escapeHtml(model) + '</code>',
                        card: '<a href="' + repoUrl + '" target="_blank" rel="noopener">view the model card</a>',
                        refresh: refreshLink(),
                    })
                    + '</div>';

            statusEl.innerHTML = banner + manageListHtml(info.models, 'huggingface');
            wireRefreshLink(statusEl);
        }

        // The manage panel: every locally-pulled model with a trash button,
        // plus every catalog model that isn't pulled yet with a download
        // button. The catalog is the same data the page's dropdown uses, so
        // the two surfaces never drift.
        function manageListHtml(installedModels, provider) {
            if (!manage) return '';
            const installed = (installedModels || []).filter(Boolean);
            const installedSet = new Set(installed);
            const isInstalled = function (val) {
                if (installedSet.has(val)) return true;
                if (provider === 'ollama') {
                    if (installedSet.has(val + ':latest')) return true;
                    if (val.endsWith(':latest') && installedSet.has(val.slice(0, -7))) return true;
                }
                return false;
            };

            const catalog = (manage.catalog[provider] || []);
            const catalogValueSet = new Set(catalog.map(function (m) { return m.value; }));
            const catalogInstalled = catalog.filter(function (m) { return isInstalled(m.value); });
            const catalogMissing = catalog.filter(function (m) { return !isInstalled(m.value); });
            // User-pulled models that we don't have a catalog row for.
            const extras = installed.filter(function (name) {
                if (catalogValueSet.has(name)) return false;
                if (provider === 'ollama' && name.endsWith(':latest')) {
                    if (catalogValueSet.has(name.slice(0, -7))) return false;
                }
                return true;
            });
            const totalInstalled = catalogInstalled.length + extras.length;

            if (totalInstalled === 0 && catalogMissing.length === 0) {
                return '';
            }

            const rows = [];
            if (totalInstalled > 0) {
                rows.push('<div class="model-list-section">Available offline ('
                    + totalInstalled + ')</div>');
                catalogInstalled.forEach(function (m) {
                    rows.push(modelRow(m.value, m.label, provider, 'delete'));
                });
                extras.forEach(function (name) {
                    rows.push(modelRow(name, name, provider, 'delete'));
                });
            }
            if (catalogMissing.length > 0) {
                rows.push('<div class="model-list-section">Available to download ('
                    + catalogMissing.length + ')</div>');
                catalogMissing.forEach(function (m) {
                    rows.push(modelRow(m.value, m.label, provider, 'pull'));
                });
            }

            return '<details class="model-manager" open>'
                + '<summary>Manage ' + escapeHtml(MANAGE_LABEL[provider]) + ' models</summary>'
                + '<div class="model-list">' + rows.join('') + '</div>'
                + '</details>';
        }

        function modelRow(value, label, provider, action) {
            const icon = action === 'delete' ? trashIconSvg() : downloadIconSvg();
            const title = action === 'delete' ? 'Remove from local cache' : 'Download to local cache';
            const aria = action === 'delete' ? ('Remove ' + value) : ('Download ' + value);
            const labelHtml = (label && label !== value)
                ? '<span class="model-row-label"> &mdash; ' + escapeHtml(label) + '</span>'
                : '';
            // Only "Available offline" rows (action === 'delete') are
            // selectable; catalog rows that aren't pulled yet would 404.
            const isSelectable = action === 'delete';
            const isSelected = isSelectable && opts.model() === value;
            const inner = '<code>' + escapeHtml(value) + '</code>' + labelHtml;
            const textHtml = isSelectable
                ? '<button type="button" class="model-row-select"'
                    + ' ' + manage.attrPrefix + 'select-model="' + escapeHtml(value) + '"'
                    + ' aria-pressed="' + (isSelected ? 'true' : 'false') + '"'
                    + ' title="' + escapeHtml(manage.selectTitle) + '">'
                    + (isSelected
                        ? '<span class="model-row-check" aria-hidden="true">✓</span>'
                        : '')
                    + inner
                    + '</button>'
                : '<div class="model-row-text">' + inner + '</div>';
            return '<div class="model-row" data-row-action="' + action + '"'
                + (isSelected ? ' data-selected="true"' : '')
                + '>'
                + textHtml
                + '<button type="button" class="model-action-btn"'
                + ' ' + manage.attrPrefix + 'action="' + action + '"'
                + ' ' + manage.attrPrefix + 'provider="' + escapeHtml(provider) + '"'
                + ' ' + manage.attrPrefix + 'model="' + escapeHtml(value) + '"'
                + ' title="' + escapeHtml(title) + '"'
                + ' aria-label="' + escapeHtml(aria) + '">'
                + icon + '</button>'
                + '</div>';
        }

        // Click handlers delegated from document, so freshly rendered rows
        // after a refresh keep working without re-binding. Pull/delete
        // plumbing lives in handleLocalModelAction() (app.js); this maps the
        // page's data attributes onto it and re-checks afterwards.
        if (manage) {
            const actionAttr = manage.attrPrefix + 'action';
            const selectAttr = manage.attrPrefix + 'select-model';
            document.addEventListener('click', function (ev) {
                const btn = ev.target.closest && ev.target.closest('.model-action-btn[' + actionAttr + ']');
                if (!btn) return;
                ev.preventDefault();
                handleLocalModelAction(
                    btn,
                    btn.getAttribute(actionAttr),
                    btn.getAttribute(manage.attrPrefix + 'provider'),
                    btn.getAttribute(manage.attrPrefix + 'model'),
                    refresh
                );
            });
            document.addEventListener('click', function (ev) {
                const sel = ev.target.closest && ev.target.closest('[' + selectAttr + ']');
                if (!sel) return;
                ev.preventDefault();
                manage.onSelect(sel.getAttribute(selectAttr));
            });
        }

        return { check: check, schedule: schedule };
    }

    window.MemorizzLocalModels = { mount: mount };
})();

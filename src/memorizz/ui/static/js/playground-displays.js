// What a tool found, shown in the reply itself: the pages of a web search,
// a webset as a table that fills in, a monitor and what its runs found, and
// a snapshot of a page the agent made.
// A tool attaches a small `display` to its result (see ToolResult.display);
// the rows are read from the console as they arrive.
(function() {
    'use strict';

    const POLL_MS = 4000;
    const POLL_FOR_MS = 20 * 60 * 1000;
    const agentId = function() { return window.PG_DISPLAY_AGENT_ID || ''; };
    function el(tag, props, children) {
        const node = document.createElement(tag);
        Object.entries(props || {}).forEach(function(pair) {
            const key = pair[0], value = pair[1];
            if (value == null || value === false) return;
            if (key === 'class') node.className = value;
            else if (key === 'text') node.textContent = value;
            else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
            else node.setAttribute(key, value === true ? '' : value);
        });
        (children || []).forEach(function(child) { if (child != null) node.append(child); });
        return node;
    }

    // Only web addresses become links; anything else is shown as text.
    function link(url, text, cls) {
        if (!/^https?:\/\//i.test(String(url || ''))) return el('span', {class: cls, text: text});
        return el('a', {class: cls, href: url, target: '_blank', rel: 'noopener noreferrer', text: text});
    }

    function site(url) {
        try { return new URL(url).hostname.replace(/^www\./, ''); } catch (_) { return ''; }
    }

    function day(value) {
        const date = new Date(value);
        return value && !isNaN(date) ? date.toLocaleDateString(undefined, {month: 'short', day: 'numeric', year: 'numeric'}) : '';
    }

    function when(value) {
        const date = new Date(value);
        return value && !isNaN(date) ? date.toLocaleString(undefined, {month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit'}) : '';
    }

    async function api(path, options) {
        const response = await fetch(`/api/agents/${encodeURIComponent(agentId())}/exa/${path}`, options);
        const data = await response.json().catch(function() { return {}; });
        if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
        return data;
    }

    function pages(results, shown) {
        const list = el('ol', {class: 'pgd-pages'});
        results.forEach(function(page, index) {
            const host = site(page.url);
            list.append(el('li', {class: 'pgd-page', hidden: index >= shown}, [
                el('span', {class: 'pgd-mark', 'aria-hidden': 'true', text: (host[0] || '·').toUpperCase()}),
                el('div', {class: 'pgd-page-body'}, [
                    link(page.url, page.title || page.url, 'pgd-page-title'),
                    el('div', {class: 'pgd-page-meta', text: [host, day(page.published)].filter(Boolean).join(' · ')}),
                    // Page text arrives as Markdown; a snippet reads better without its marks.
                    page.snippet ? el('p', {class: 'pgd-page-snippet', text: String(page.snippet).replace(/(^|\s)#{1,6}\s+/g, '$1').replace(/\*\*|__/g, '').trim()}) : null,
                ]),
            ]));
        });
        if (results.length > shown) {
            const more = el('button', {class: 'pgd-more', type: 'button', text: `Show ${results.length - shown} more`, onclick: function() {
                list.querySelectorAll('li[hidden]').forEach(function(item) { item.hidden = false; });
                more.remove();
            }});
            const wrap = el('div', {}, [list, more]);
            return wrap;
        }
        return list;
    }

    function head(label, title, right) {
        return el('header', {class: 'pgd-head'}, [
            el('div', {class: 'pgd-head-text'}, [
                el('span', {class: 'pgd-label', text: label}),
                el('strong', {class: 'pgd-title', text: title}),
            ]),
            right ? el('div', {class: 'pgd-head-side'}, right) : null,
        ]);
    }

    // --- a web search ---

    function search(card, display) {
        const results = display.results || [];
        const provider = display.provider ? display.provider[0].toUpperCase() + display.provider.slice(1) : 'the web';
        card.replaceChildren(
            head(`Searched with ${provider}`, display.query || '', [
                el('span', {class: 'pgd-count', text: `${display.count != null ? display.count : results.length} result${(display.count != null ? display.count : results.length) === 1 ? '' : 's'}`}),
            ]),
            results.length ? pages(results, 3) : el('p', {class: 'pgd-empty', text: 'Nothing was found.'})
        );
    }

    // --- a webset: a table that fills in ---

    const GLYPH = {yes: '✓', no: '✕', unclear: '?'};

    function websetTable(table) {
        // A fact Exa had for none of the rows is not worth a column.
        const columns = (table.columns || []).filter(function(column) {
            return column.kind !== 'field' || (table.rows || []).some(function(row) { return (row.cells || {})[column.key]; });
        });
        const headRow = el('tr', {}, [el('th', {class: 'pgd-col-name', text: table.entity ? table.entity.replace('_', ' ') : 'Item'})]);
        columns.forEach(function(column) {
            headRow.append(el('th', {class: `pgd-col-${column.kind}`, title: column.label, text: column.label}));
        });
        const body = el('tbody');
        (table.rows || []).forEach(function(row) {
            const line = el('tr', {}, [el('td', {class: 'pgd-col-name'}, [
                link(row.url, row.name || row.url, 'pgd-row-name'),
                row.description ? el('span', {class: 'pgd-row-about', text: row.description}) : null,
            ])]);
            columns.forEach(function(column) {
                const value = (row.cells || {})[column.key] || '';
                const why = (row.why || {})[column.key] || '';
                if (column.kind === 'criterion') {
                    line.append(el('td', {class: 'pgd-cell-criterion', title: why || value}, [
                        value ? el('span', {class: `pgd-met pgd-met--${value}`, 'aria-label': value, text: GLYPH[value] || ''}) : null,
                    ]));
                } else {
                    line.append(el('td', {class: value === '…' ? 'pgd-cell-pending' : '', title: why, text: value}));
                }
            });
            body.append(line);
        });
        return el('div', {class: 'pgd-table-wrap', tabindex: '0'}, [
            el('table', {class: 'pgd-table'}, [el('thead', {}, [headRow]), body]),
        ]);
    }

    function webset(card, display) {
        let stopAt = Date.now() + POLL_FOR_MS;
        let timer = null;

        function draw(table) {
            const running = table.status === 'running';
            const percent = Math.max(0, Math.min(100, Math.round(table.percent || 0)));
            const saved = table.in_memory || 0;
            const save = el('button', {class: 'pgd-action', type: 'button', disabled: !table.found || saved >= table.found,
                text: saved && saved >= table.found ? `In memory (${saved})` : saved ? `Save ${table.found - saved} new to memory` : 'Save to memory',
                onclick: async function() {
                    save.disabled = true;
                    save.textContent = 'Saving…';
                    try {
                        const done = await api(`websets/${encodeURIComponent(table.id)}/save`, {method: 'POST'});
                        save.textContent = `In memory (${done.in_memory})`;
                    } catch (error) {
                        save.disabled = false;
                        save.textContent = 'Save to memory';
                        note.textContent = String(error.message || error);
                    }
                }});
            const note = el('span', {class: 'pgd-note', role: 'status'});
            card.replaceChildren(
                head('Webset', table.title || display.title || '', [
                    el('span', {class: `pgd-status pgd-status--${running ? 'running' : 'done'}`, text: running ? `Building · ${percent}%` : 'Done'}),
                ]),
                el('div', {class: 'pgd-progress', hidden: !running, role: 'progressbar', 'aria-valuenow': String(percent), 'aria-valuemin': '0', 'aria-valuemax': '100'}, [
                    el('span', {style: `width:${percent}%`}),
                ]),
                el('div', {class: 'pgd-bar'}, [
                    el('span', {class: 'pgd-count', text: `${table.found} of ${table.wanted || '?'} found` + (table.analyzed ? ` · ${table.analyzed} checked` : '')}),
                    el('span', {class: 'pgd-actions'}, [
                        note,
                        save,
                        table.found ? el('a', {class: 'pgd-action', href: `/api/agents/${encodeURIComponent(agentId())}/exa/websets/${encodeURIComponent(table.id)}.csv`, text: 'CSV'}) : null,
                        table.url ? link(table.url, 'Open in Exa', 'pgd-action') : null,
                    ]),
                ]),
                table.rows && table.rows.length
                    ? websetTable(table)
                    : el('p', {class: 'pgd-empty', text: running ? 'Searching and checking each candidate. Rows appear here as they pass.' : 'Nothing matched.'})
            );
            return running;
        }

        async function load() {
            if (!card.isConnected) return;
            try {
                const running = draw(await api(`websets/${encodeURIComponent(display.id)}`));
                if (running && Date.now() < stopAt) timer = setTimeout(load, POLL_MS);
            } catch (error) {
                card.replaceChildren(head('Webset', display.title || display.id || ''), el('p', {class: 'pgd-empty', text: String(error.message || error)}));
            }
        }

        card.replaceChildren(head('Webset', display.title || ''), el('p', {class: 'pgd-empty', text: 'Loading…'}));
        card._refresh = function() { clearTimeout(timer); stopAt = Date.now() + POLL_FOR_MS; load(); };
        load();
    }

    // --- a monitor ---

    function monitor(card, display) {
        function draw(view) {
            const latest = (view.runs || []).find(function(run) { return run.status === 'completed'; });
            const working = (view.runs || []).some(function(run) { return run.status === 'pending' || run.status === 'running'; });
            const failed = !latest && (view.runs || []).find(function(run) { return run.status === 'failed'; });
            const note = el('span', {class: 'pgd-note', role: 'status'});
            const run = el('button', {class: 'pgd-action', type: 'button', disabled: working || view.status === 'disabled', text: working ? 'Running…' : 'Run now', onclick: async function() {
                run.disabled = true;
                run.textContent = 'Running…';
                try { await api(`monitors/${encodeURIComponent(view.id)}/run`, {method: 'POST'}); setTimeout(load, POLL_MS); }
                catch (error) { run.disabled = false; run.textContent = 'Run now'; note.textContent = String(error.message || error); }
            }});
            const facts = [
                view.every ? `Every ${view.every}` : '',
                view.next_run_at && view.status === 'active' ? `next ${when(view.next_run_at)}` : '',
            ].filter(Boolean).join(' · ');
            card.replaceChildren(
                head('Monitor', view.name || display.name || '', [
                    el('span', {class: `pgd-status pgd-status--${view.status === 'active' ? 'done' : 'paused'}`, text: view.status === 'active' ? 'Watching' : view.status === 'disabled' ? 'Disabled' : 'Paused'}),
                ]),
                el('div', {class: 'pgd-bar'}, [
                    el('span', {class: 'pgd-count', text: facts}),
                    el('span', {class: 'pgd-actions'}, [note, run]),
                ]),
                view.query && view.query !== view.name ? el('p', {class: 'pgd-query', text: view.query}) : null,
                // Exa must have an address to report to; say when one was made for it.
                view.reports_to === 'temporary' ? el('details', {class: 'pgd-guide'}, [
                    el('summary', {text: 'Reports to a temporary webhook.site address'}),
                    el('p', {}, [
                        'Exa needs a public address to call when a run ends. None was set, so MemoRizz made a temporary one at ',
                        link(view.webhook_url.replace('webhook.site/', 'webhook.site/#!/view/'), 'webhook.site', ''),
                        '. The results shown here are read from Exa and do not depend on it.',
                    ]),
                    el('ul', {}, [
                        el('li', {text: 'Anyone who has the address can read what each run found. Do not use it for private topics.'}),
                        el('li', {text: 'It expires after about a week. The monitor keeps running; only Exa\u2019s calls to the address stop being received.'}),
                        el('li', {}, ['To use your own, set ', el('code', {text: 'EXA_MONITOR_WEBHOOK_URL'}), ' where MemoRizz runs (Settings, or ~/.memorizz/.env), then create the monitor again.']),
                    ]),
                ]) : null,
                latest
                    ? el('div', {}, [
                        el('div', {class: 'pgd-run-head', text: `Latest run · ${when(latest.at)} · ${latest.results.length} result${latest.results.length === 1 ? '' : 's'}`}),
                        latest.summary ? el('p', {class: 'pgd-run-summary', text: latest.summary}) : null,
                        latest.results.length ? pages(latest.results, 3) : null,
                    ])
                    : el('p', {class: 'pgd-empty', text: failed ? `The last run failed: ${String(failed.failed_because || 'no reason given').replace(/_/g, ' ')}.` : working ? 'The first run is under way.' : 'No run has finished yet.'})
            );
            return working;
        }

        let stopAt = Date.now() + POLL_FOR_MS;
        async function load() {
            if (!card.isConnected) return;
            try {
                const working = draw(await api(`monitors/${encodeURIComponent(display.id)}`));
                if (working && Date.now() < stopAt) setTimeout(load, POLL_MS);
            } catch (error) {
                card.replaceChildren(head('Monitor', display.name || display.id || ''), el('p', {class: 'pgd-empty', text: String(error.message || error)}));
            }
        }
        card.replaceChildren(head('Monitor', display.name || ''), el('p', {class: 'pgd-empty', text: 'Loading…'}));
        card._refresh = load;
        load();
    }

    // --- a page the agent made: what is on it now ---

    function snapshotBlock(block) {
        if (!block || typeof block !== 'object') return null;
        const text = String(block.text || '');
        switch (block.type) {
            case 'heading': return el('div', {class: 'pgd-snap-h pgd-snap-h' + Math.min(3, Math.max(1, Number(block.level) || 1)), role: 'heading', 'aria-level': String(3 + (Number(block.level) || 1)), text: text});
            case 'bullet': return el('div', {class: 'pgd-snap-li', text: text});
            case 'number': return el('div', {class: 'pgd-snap-li pgd-snap-li--num', text: text});
            case 'todo': return el('div', {class: 'pgd-snap-todo' + (block.checked ? ' is-done' : '')}, [
                el('span', {class: 'pgd-snap-box', 'aria-hidden': 'true', text: block.checked ? '\u2713' : ''}),
                el('span', {text: text}),
                el('span', {class: 'sr-only', text: block.checked ? ' (done)' : ' (not done)'}),
            ]);
            case 'quote': return el('blockquote', {class: 'pgd-snap-quote', text: text});
            case 'callout': return el('div', {class: 'pgd-snap-callout', text: text});
            case 'code': return el('pre', {class: 'pgd-snap-code', text: text});
            case 'divider': return el('hr', {class: 'pgd-snap-rule'});
            case 'row': return el('div', {class: 'pgd-snap-row'}, (block.cells || []).map(function(cell) { return el('span', {text: String(cell || '')}); }));
            case 'page':
            case 'database': return el('div', {class: 'pgd-snap-sub'}, [
                el('span', {class: 'pgd-snap-sub-mark', 'aria-hidden': 'true', text: block.type === 'database' ? '\u25A6' : '\u25A4'}),
                el('span', {text: text}),
                el('span', {class: 'pgd-snap-sub-kind', text: block.type === 'database' ? 'database' : 'page'}),
            ]);
            default: return text ? el('p', {class: 'pgd-snap-p', text: text}) : null;
        }
    }

    function pageSnapshot(card, display) {
        const notion = display.source === 'notion';
        const where = notion ? 'Notion' : (display.site || 'the page');
        const blocks = (display.blocks || []).map(snapshotBlock).filter(Boolean);
        const properties = display.properties || [];
        const title = display.title || display.url || 'Page';
        // The page itself, as it read when the snapshot was taken.
        const paper = el('div', {class: 'pgd-snap-paper'}, [
            el('div', {class: 'pgd-snap-title'}, [
                display.icon ? el('span', {class: 'pgd-snap-icon', 'aria-hidden': 'true', text: display.icon}) : null,
                el('span', {text: title}),
            ]),
            properties.length ? el('dl', {class: 'pgd-snap-props'}, properties.reduce(function(list, property) {
                list.push(el('dt', {text: property.name}), el('dd', {text: property.value}));
                return list;
            }, [])) : null,
            display.summary ? el('p', {class: 'pgd-snap-summary', text: display.summary}) : null,
            blocks.length ? el('div', {class: 'pgd-snap-body'}, blocks) : null,
            !blocks.length && !properties.length && !display.summary
                ? el('p', {class: 'pgd-empty', text: display.read ? 'The page is empty.' : 'A link to the page; its content was not read.'})
                : null,
        ]);
        const frame = el('div', {class: 'pgd-snap-frame'}, [paper]);
        const more = el('button', {class: 'pgd-more', type: 'button', hidden: true, text: 'Show more of the page', onclick: function() {
            const open = frame.classList.toggle('is-open');
            more.textContent = open ? 'Show less' : 'Show more of the page';
        }});
        const taken = when(display.captured_at);
        const note = display.trimmed
            ? 'The snapshot was too large to keep; open the page to see it.'
            : display.read
                ? 'Read from ' + where + (taken ? ' \u00b7 ' + taken : '') + (display.truncated ? ' \u00b7 the start of the page' : '')
                : (display.unread_because ? 'Not read: ' + display.unread_because + '. ' : '') + 'Shown from what the agent reported.';
        card.replaceChildren(
            head('Page snapshot \u00b7 ' + where, title, [
                link(display.url, notion ? 'Open in Notion' : 'Open page', 'pgd-action'),
            ]),
            display.trimmed ? null : frame,
            display.trimmed ? null : more,
            el('p', {class: 'pgd-snap-note', text: note})
        );
        // Tall pages are folded; the button shows when there is more below.
        requestAnimationFrame(function() { more.hidden = paper.scrollHeight <= frame.clientHeight + 4; });
        card._update = function(next) { pageSnapshot(card, next); };
    }

    const KINDS = {web_search: search, exa_webset: webset, exa_monitor: monitor, page_snapshot: pageSnapshot};

    // Put a display in a message. One card per webset or monitor in a
    // message: a later step about the same one refreshes its card.
    function render(messageEl, display) {
        if (!messageEl || !display || !KINDS[display.kind]) return;
        let holder = messageEl.querySelector(':scope > .pg-displays, :scope .pg-message-body > .pg-displays');
        if (!holder) {
            holder = el('div', {class: 'pg-displays'});
            const text = messageEl.querySelector('.pg-message-text');
            if (text) text.before(holder); else messageEl.append(holder);
        }
        const key = display.kind + ':' + (display.id || display.query || '');
        const existing = Array.from(holder.children).find(function(card) { return card.dataset.key === key; });
        if (existing) {
            // A page snapshotted again shows the newer snapshot.
            if (existing._update) existing._update(display);
            else if (existing._refresh) existing._refresh();
            return;
        }
        const card = el('section', {class: `pgd pgd--${display.kind.replace(/_/g, '-')}`, 'data-key': key});
        holder.append(card);
        // A stored reply is built before it is put on the page; a card reads
        // its rows only once it is there.
        if (card.isConnected) KINDS[display.kind](card, display);
        else setTimeout(function() { KINDS[display.kind](card, display); }, 0);
    }

    window.pgDisplays = {render: render};
})();

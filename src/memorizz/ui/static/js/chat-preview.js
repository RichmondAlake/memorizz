/* A preview pane beside the chat. A reply that carries a document (HTML, SVG,
   Markdown, JSON, CSV) or a long block of code gets an "Open" button on that
   block; the pane shows it rendered, with the source one click away, and the
   conversation stays usable next to it.

   HTML and SVG render in a sandboxed frame with no access to this page, its
   cookies or its storage. Scripts stay off until the reader turns them on. */
(function() {
    const RENDERED = { html: 'HTML page', svg: 'SVG image', markdown: 'Markdown document', md: 'Markdown document', json: 'JSON', csv: 'CSV table' };
    const EXT = { html: 'html', svg: 'svg', markdown: 'md', md: 'md', json: 'json', csv: 'csv', python: 'py', javascript: 'js', typescript: 'ts', bash: 'sh', sql: 'sql', yaml: 'yml' };
    const LONG_BLOCK = 14;
    let pane = null;
    let current = null;
    let returnTo = null;
    let hidInspector = false;

    function language(code) {
        const match = (code.className || '').match(/language-([\w+-]+)/);
        let lang = match ? match[1].toLowerCase() : '';
        const text = code.textContent.trimStart();
        if (lang === 'xml' && /^<svg[\s>]/i.test(text)) lang = 'svg';
        if (!lang && /^<!doctype html|^<html[\s>]/i.test(text)) lang = 'html';
        if (!lang && /^<svg[\s>]/i.test(text)) lang = 'svg';
        return lang;
    }

    function decorate(root) {
        root.querySelectorAll('pre > code').forEach(function(code) {
            const pre = code.parentElement;
            if (pre.dataset.preview) return;
            const lang = language(code);
            const lines = code.textContent.split('\n').length;
            if (!RENDERED[lang] && lines < LONG_BLOCK) return;
            pre.dataset.preview = '1';
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'pg-preview-open';
            button.textContent = RENDERED[lang] ? 'Open preview' : 'Open';
            button.title = RENDERED[lang] ? 'Show this ' + RENDERED[lang] + ' beside the chat' : 'Show this code beside the chat';
            button.addEventListener('click', function() { open({ lang: lang, text: code.textContent.replace(/\n$/, '') }, button); });
            pre.insertBefore(button, pre.firstChild);
        });
    }

    function el(tag, className, text) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text !== undefined) node.textContent = text;
        return node;
    }

    function csvRows(text) {
        const rows = [];
        let row = [], cell = '', quoted = false;
        for (let i = 0; i < text.length; i++) {
            const ch = text[i];
            if (quoted) {
                if (ch === '"' && text[i + 1] === '"') { cell += '"'; i++; }
                else if (ch === '"') quoted = false;
                else cell += ch;
            } else if (ch === '"') quoted = true;
            else if (ch === ',') { row.push(cell); cell = ''; }
            else if (ch === '\n') { row.push(cell); rows.push(row); row = []; cell = ''; }
            else if (ch !== '\r') cell += ch;
        }
        if (cell || row.length) { row.push(cell); rows.push(row); }
        return rows;
    }

    function rendered(item, scripts) {
        if (item.lang === 'html' || item.lang === 'svg') {
            const frame = el('iframe', 'pg-preview-frame');
            frame.setAttribute('sandbox', scripts ? 'allow-scripts' : '');
            frame.setAttribute('referrerpolicy', 'no-referrer');
            frame.title = 'Preview';
            frame.srcdoc = item.lang === 'svg'
                ? '<!doctype html><meta charset="utf-8"><style>html,body{margin:0;height:100%;display:grid;place-items:center;background:#fff}svg{max-width:96%;max-height:96%}</style>' + item.text
                : item.text;
            return frame;
        }
        if (item.lang === 'markdown' || item.lang === 'md') {
            const doc = el('div', 'pg-preview-doc pg-message-body');
            if (window.MemorizzMarkdown) doc.innerHTML = window.MemorizzMarkdown.render(item.text);
            else doc.textContent = item.text;
            return doc;
        }
        if (item.lang === 'json') {
            const block = el('pre', 'pg-preview-code');
            try { block.textContent = JSON.stringify(JSON.parse(item.text), null, 2); } catch (_) { block.textContent = item.text; }
            return block;
        }
        if (item.lang === 'csv') {
            const rows = csvRows(item.text).slice(0, 501);
            const wrap = el('div', 'table-scroll pg-preview-table');
            const table = el('table', 'data-table');
            rows.forEach(function(cells, index) {
                const tr = el('tr');
                cells.forEach(function(value) { tr.appendChild(el(index === 0 ? 'th' : 'td', '', value)); });
                (index === 0 ? table.createTHead() : (table.tBodies[0] || table.createTBody())).appendChild(tr);
            });
            wrap.appendChild(table);
            return wrap;
        }
        return null;
    }

    function draw(view, scripts) {
        const item = current;
        const body = pane.querySelector('.pg-preview-body');
        body.textContent = '';
        const canRender = Boolean(RENDERED[item.lang]);
        const show = canRender ? view : 'code';
        pane.querySelectorAll('[data-view]').forEach(function(button) {
            button.hidden = !canRender;
            button.setAttribute('aria-pressed', button.dataset.view === show ? 'true' : 'false');
        });
        const run = pane.querySelector('.pg-preview-scripts');
        run.hidden = !(show === 'preview' && item.lang === 'html');
        run.querySelector('input').checked = Boolean(scripts);
        if (show === 'preview') body.appendChild(rendered(item, scripts));
        else body.appendChild(el('pre', 'pg-preview-code', item.text));
    }

    function build() {
        pane = el('aside', 'pg-preview');
        pane.setAttribute('aria-label', 'Preview');
        pane.tabIndex = -1;
        const head = el('header', 'pg-preview-head');
        const title = el('strong', 'pg-preview-title');
        head.appendChild(title);
        const views = el('div', 'segmented pg-preview-views');
        views.setAttribute('role', 'group');
        views.setAttribute('aria-label', 'Show');
        [['preview', 'Preview'], ['code', 'Code']].forEach(function(pair) {
            const button = el('button', 'segmented-item', pair[1]);
            button.type = 'button';
            button.dataset.view = pair[0];
            button.addEventListener('click', function() { draw(pair[0], false); });
            views.appendChild(button);
        });
        head.appendChild(views);
        const copy = el('button', 'btn btn-sm', 'Copy');
        copy.type = 'button';
        copy.addEventListener('click', function() {
            if (!navigator.clipboard) return;
            navigator.clipboard.writeText(current.text).then(function() {
                copy.textContent = 'Copied';
                setTimeout(function() { copy.textContent = 'Copy'; }, 1200);
            });
        });
        const save = el('button', 'btn btn-sm', 'Download');
        save.type = 'button';
        save.addEventListener('click', function() {
            const name = 'memorizz-reply.' + (EXT[current.lang] || 'txt');
            if (window.MemorizzMonitor) MemorizzMonitor.download(name, current.text, 'text/plain;charset=utf-8');
        });
        const close = el('button', 'pg-preview-close', '×');
        close.type = 'button';
        close.setAttribute('aria-label', 'Close the preview');
        close.addEventListener('click', shut);
        head.append(copy, save, close);
        pane.appendChild(head);
        const run = el('label', 'pg-preview-scripts');
        const box = el('input');
        box.type = 'checkbox';
        box.addEventListener('change', function() { draw('preview', box.checked); });
        run.append(box, document.createTextNode(' Run this page\'s scripts. It stays in a sandbox with no access to MemoRizz, but it can reach the network.'));
        pane.appendChild(run);
        pane.appendChild(el('div', 'pg-preview-body'));
        pane.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') { event.stopPropagation(); shut(); }
        });
        document.body.appendChild(pane);
    }

    function open(item, trigger) {
        if (!pane) build();
        current = item;
        returnTo = trigger || null;
        const lines = item.text.split('\n').length;
        pane.querySelector('.pg-preview-title').textContent = (RENDERED[item.lang] || (item.lang ? item.lang + ' code' : 'Code')) + ' · ' + lines + (lines === 1 ? ' line' : ' lines');
        draw('preview', false);
        pane.hidden = false;
        // The inspector steps aside while the pane is open, and comes back
        // with it closed if it was showing before.
        const app = document.querySelector('.pg-app');
        if (app && !document.body.classList.contains('pg-preview-open-state')) {
            hidInspector = !app.classList.contains('inspector-collapsed');
            if (hidInspector) app.classList.add('inspector-collapsed');
        }
        document.body.classList.add('pg-preview-open-state');
        pane.focus();
    }

    function shut() {
        if (!pane || pane.hidden) return;
        pane.hidden = true;
        pane.querySelector('.pg-preview-body').textContent = '';
        document.body.classList.remove('pg-preview-open-state');
        const app = document.querySelector('.pg-app');
        if (app && hidInspector) app.classList.remove('inspector-collapsed');
        hidInspector = false;
        if (returnTo && document.contains(returnTo)) returnTo.focus();
    }

    function start() {
        const thread = document.getElementById('playground-messages');
        if (!thread) return;
        let queued = false;
        const run = function() { queued = false; decorate(thread); };
        // Replies re-render while they stream: decorate once things settle.
        new MutationObserver(function() {
            if (queued) return;
            queued = true;
            setTimeout(run, 250);
        }).observe(thread, { childList: true, subtree: true });
        decorate(thread);
    }

    window.MemorizzChatPreview = { open: open, close: shut };
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
    else start();
})();

// One whole file of an agent's source, with the lines a change touched marked.
(function() {
    const root = document.getElementById('sfile-root');
    if (!root) return;
    const agentId = root.dataset.agentId;
    const base = '/api/agents/' + encodeURIComponent(agentId) + '/source';
    const page = '/agents/' + encodeURIComponent(agentId) + '/source';
    const picker = document.getElementById('sfile-version');
    const main = document.getElementById('sfile-main');
    const query = new URLSearchParams(window.location.search);
    const path = query.get('path') || '';
    let side = query.get('side') === 'old' ? 'old' : 'new';
    let target = parseInt((window.location.hash.match(/^#L(\d+)/) || [])[1], 10) || 0;
    let marks = [];

    function el(tag, attrs, children) {
        const node = document.createElement(tag);
        Object.keys(attrs || {}).forEach(function(key) {
            if (key === 'class') node.className = attrs[key];
            else if (key === 'text') node.textContent = attrs[key];
            else if (key.indexOf('on') === 0) node.addEventListener(key.slice(2), attrs[key]);
            else if (attrs[key] !== false && attrs[key] != null) node.setAttribute(key, attrs[key]);
        });
        (children || []).forEach(function(child) { if (child != null) node.append(child); });
        return node;
    }

    async function get(url) {
        const response = await fetch(url);
        const data = await response.json().catch(function() { return {}; });
        if (!response.ok) throw new Error(data.detail || ('HTTP ' + response.status));
        return data;
    }

    function empty(title, text) {
        main.replaceChildren(el('div', { class: 'empty-state' }, [el('strong', { text: title }), el('p', { text: text })]));
    }

    function address() {
        const params = new URLSearchParams({ path: path, version: picker.value });
        if (side === 'old') params.set('side', 'old');
        return page + '/file?' + params.toString() + (target ? '#L' + target : '');
    }

    function go(line, smooth) {
        const previous = main.querySelector('tr.is-target');
        if (previous) previous.classList.remove('is-target');
        const row = document.getElementById('L' + line);
        if (!row) return;
        target = line;
        row.classList.add('is-target');
        row.scrollIntoView({ block: 'center', behavior: smooth ? 'smooth' : 'auto' });
        window.history.replaceState(null, '', address());
    }

    // Changed lines that touch are one change, as an editor counts them.
    function blocks(lines) {
        const starts = [];
        lines.forEach(function(line, index) {
            if (!index || line !== lines[index - 1] + 1) starts.push(line);
        });
        return starts;
    }

    function draw(file) {
        const lines = file.text.split('\n');
        if (lines.length > 1 && lines[lines.length - 1] === '') lines.pop();
        const changed = new Set(file.changed), cut = new Set(file.removed_before);
        marks = blocks(file.changed.concat(file.removed_before).sort(function(a, b) { return a - b; }).filter(function(line, index, all) { return line !== all[index - 1]; }));
        const read = window.MemorizzCode.tokenizer(file.path);
        const body = el('tbody');
        lines.forEach(function(text, index) {
            const number = index + 1;
            const kind = changed.has(number) ? (side === 'old' ? 'is-old' : 'is-new') : '';
            const row = el('tr', { id: 'L' + number, class: (kind + (cut.has(number) ? ' was-cut' : '')).trim() || null }, [
                el('td', { class: 'ed-num' }, [el('a', { href: '#L' + number, text: String(number), onclick: function(event) { event.preventDefault(); go(number, false); } })]),
                el('td', { class: 'ed-edge', title: kind ? (side === 'old' ? 'Removed by this change' : 'Changed here') : (cut.has(number) ? 'Lines were removed above this one' : null) }),
                window.MemorizzCode.render(el('td', { class: 'ed-code' }), read(text)),
            ]);
            body.append(row);
        });
        const parts = file.path.split('/');
        const crumbs = el('div', { class: 'ed-crumbs' });
        parts.forEach(function(part, index) {
            if (index) crumbs.append(el('i', { text: '›' }));
            crumbs.append(el('span', { text: part }));
        });
        const position = el('span');
        const step = function(by) {
            const at = marks.findIndex(function(line) { return line === target; });
            let next;
            if (at >= 0) next = marks[(at + by + marks.length) % marks.length];
            else if (by > 0) next = marks.find(function(line) { return line > target; }) || marks[0];
            else next = marks.slice().reverse().find(function(line) { return line < target; }) || marks[marks.length - 1];
            go(next, true);
            label();
        };
        const label = function() {
            const at = marks.indexOf(target);
            position.textContent = marks.length ? (at >= 0 ? (at + 1) + ' of ' : '') + marks.length + (marks.length === 1 ? ' change' : ' changes') : 'No changes in this file';
        };
        const up = el('button', { type: 'button', text: '↑', title: 'Previous change', 'aria-label': 'Previous change', disabled: marks.length ? null : 'disabled', onclick: function() { step(-1); } });
        const down = el('button', { type: 'button', text: '↓', title: 'Next change', 'aria-label': 'Next change', disabled: marks.length ? null : 'disabled', onclick: function() { step(1); } });
        main.replaceChildren(el('div', { class: 'ed-file' }, [
            el('div', { class: 'ed-bar' }, [crumbs, el('div', { class: 'ed-bar-side' }, [position, up, down])]),
            el('div', { class: 'ed-scroll' }, [el('table', { class: 'ed-table ed-table--file' }, [body])]),
            el('div', { class: 'ed-foot', text: lines.length + ' lines' + (file.cut_short ? ' shown; the file is longer and was cut short' : '') + ' · ' + (side === 'old' ? 'before the change' : 'after the change') }),
        ]));
        // No line asked for: start at the first change.
        if (!document.getElementById('L' + target)) target = marks[0] || 0;
        if (target) go(target, false);
        label();
    }

    async function load() {
        document.querySelectorAll('[data-side]').forEach(function(button) {
            button.setAttribute('aria-pressed', button.dataset.side === side ? 'true' : 'false');
        });
        document.querySelector('.page-back').href = page + '/diff?version=' + encodeURIComponent(picker.value);
        window.history.replaceState(null, '', address());
        try {
            const params = new URLSearchParams({ path: path, version: picker.value });
            if (side === 'old') params.set('side', 'old');
            draw(await get(base + '/file?' + params.toString()));
        } catch (error) {
            empty('This file could not be shown', String(error.message || error));
        }
    }

    async function start() {
        const title = document.getElementById('sfile-title');
        if (path && title) title.textContent = path.split('/').pop();
        if (!path) { empty('No file was chosen', 'Open a file from the list of changes.'); return; }
        try {
            const state = await get(base);
            const versions = (state.versions || []).filter(function(version) { return !version.installed; });
            versions.forEach(function(version) {
                picker.append(el('option', { value: version.id, text: 'Marking v' + version.number + ' · ' + version.message }));
            });
            picker.append(el('option', { value: 'all', text: 'Marking everything since install' }));
            picker.append(el('option', { value: 'working', text: 'Marking edits not yet kept' }));
            const asked = query.get('version');
            const known = Array.from(picker.options).some(function(option) { return option.value === asked; });
            picker.value = known ? asked : 'all';
            await load();
        } catch (error) {
            empty('The source could not be read', String(error.message || error));
        }
    }

    picker.addEventListener('change', function() { target = 0; load(); });
    document.querySelectorAll('[data-side]').forEach(function(button) {
        button.addEventListener('click', function() { side = button.dataset.side; target = 0; load(); });
    });
    window.addEventListener('hashchange', function() {
        const line = parseInt((window.location.hash.match(/^#L(\d+)/) || [])[1], 10);
        if (line) go(line, false);
    });
    start();
})();

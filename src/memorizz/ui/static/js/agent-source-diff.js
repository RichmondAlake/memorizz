// What an agent changed in its own source: files, hunks and lines.
(function() {
    const root = document.getElementById('sdiff-root');
    if (!root) return;
    const agentId = root.dataset.agentId;
    const agentName = root.dataset.agentName;
    const base = '/api/agents/' + encodeURIComponent(agentId) + '/source';
    const page = '/agents/' + encodeURIComponent(agentId) + '/source';
    const $ = function(id) { return document.getElementById(id); };
    const picker = $('sdiff-version');
    const narrow = window.matchMedia('(max-width: 760px)');
    let versions = [], files = [], story = null, blamed = {}, layout = 'split', storyOpen = true;
    try { storyOpen = window.localStorage.getItem('memorizz.diffStory') !== '0'; } catch (error) { /* stays open */ }
    try { layout = window.localStorage.getItem('memorizz.diffView') || 'split'; } catch (error) { /* stays side by side */ }

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

    async function get(path) {
        const response = await fetch(base + path);
        const data = await response.json().catch(function() { return {}; });
        if (!response.ok) throw new Error(data.detail || ('HTTP ' + response.status));
        return data;
    }

    function when(value) {
        const date = new Date(value);
        return isNaN(date) ? '' : date.toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
    }

    function counts(plus, minus) {
        return el('span', { class: 'sdiff-counts' }, [
            el('span', { class: 'sdiff-plus', text: '+' + plus }), ' ',
            el('span', { class: 'sdiff-minus', text: '−' + minus }),
        ]);
    }

    // The part of a changed line that differs from the line it replaced.
    function changedSpan(before, after) {
        let start = 0;
        const max = Math.min(before.length, after.length);
        while (start < max && before[start] === after[start]) start++;
        let end = 0;
        while (end < max - start && before[before.length - 1 - end] === after[after.length - 1 - end]) end++;
        // A line rewritten from end to end is marked as a line, not as text.
        if (!start && !end) return null;
        return { start: start, beforeEnd: before.length - end, afterEnd: after.length - end };
    }

    // The whole file, at this line: the new side, or how it was before.
    function fileUrl(path, line, old) {
        const params = new URLSearchParams({ path: path, version: picker.value });
        if (old) params.set('side', 'old');
        return page + '/file?' + params.toString() + (line ? '#L' + line : '');
    }

    function code(tokens, from, to) {
        return window.MemorizzCode.render(el('td', { class: 'ed-code' }), tokens, from, to);
    }

    function num(file, value, old) {
        const cell = el('td', { class: 'ed-num' });
        // The line an error named is pointed out on the side it was read from.
        if (old && value != null && (blamed[file.old_path] || {})[value]) {
            cell.classList.add('is-blamed');
            cell.id = cell.id || 'sdiff-blamed';
            cell.title = 'The error was raised on this line';
        }
        if (value != null) cell.append(el('a', { href: fileUrl(old ? file.old_path : file.path, value, old), text: String(value), title: 'Open the file at line ' + value }));
        return cell;
    }

    // Lines that were replaced sit next to what replaced them.
    function pairs(lines) {
        const rows = [];
        let index = 0;
        while (index < lines.length) {
            if (lines[index].kind === 'same') { rows.push([lines[index], lines[index]]); index++; continue; }
            const removed = [], added = [];
            while (index < lines.length && lines[index].kind === 'del') removed.push(lines[index++]);
            while (index < lines.length && lines[index].kind === 'add') added.push(lines[index++]);
            for (let i = 0; i < Math.max(removed.length, added.length); i++) rows.push([removed[i] || null, added[i] || null]);
        }
        return rows;
    }

    function unifiedRows(body, file, hunk, read) {
        let waiting = [];
        const flush = function() { waiting.forEach(function(row) { body.append(row); }); waiting = []; };
        pairs(hunk.lines).forEach(function(pair) {
            const before = pair[0], after = pair[1];
            if (before && before.kind === 'same') {
                flush();
                const tokens = read.both(before.text);
                body.append(el('tr', {}, [num(file, before.old, true), num(file, before.new), el('td', { class: 'ed-sign' }), code(tokens)]));
                return;
            }
            const span = before && after ? changedSpan(before.text, after.text) : null;
            if (before) body.append(el('tr', { class: 'is-del' }, [num(file, before.old, true), num(file, null), el('td', { class: 'ed-sign', text: '−' }), code(read.old(before.text), span && span.start, span && span.beforeEnd)]));
            // Additions follow every removal of the same block, as in a patch.
            if (after) waiting.push(el('tr', { class: 'is-add' }, [num(file, null), num(file, after.new), el('td', { class: 'ed-sign', text: '+' }), code(read.fresh(after.text), span && span.start, span && span.afterEnd)]));
        });
        flush();
    }

    function splitRows(body, file, hunk, read) {
        pairs(hunk.lines).forEach(function(pair) {
            const before = pair[0], after = pair[1];
            const same = before && before.kind === 'same';
            const span = !same && before && after ? changedSpan(before.text, after.text) : null;
            const sameTokens = same ? read.both(before.text) : null;
            const left = before ? code(same ? sameTokens : read.old(before.text), span && span.start, span && span.beforeEnd) : el('td', { class: 'ed-code is-blank' });
            const right = after ? code(same ? sameTokens : read.fresh(after.text), span && span.start, span && span.afterEnd) : el('td', { class: 'ed-code is-blank' });
            const leftNum = num(file, before ? before.old : null, true), rightNum = num(file, after ? after.new : null);
            if (!same && before) { left.classList.add('is-del'); leftNum.classList.add('is-del'); }
            if (!same && after) { right.classList.add('is-add'); rightNum.classList.add('is-add'); }
            if (!before) leftNum.classList.add('is-blank');
            if (!after) rightNum.classList.add('is-blank');
            body.append(el('tr', {}, [leftNum, left, rightNum, right]));
        });
    }

    // Each side of a diff is its own file, so each is read by its own tokenizer.
    function readers(path) {
        const old = window.MemorizzCode.tokenizer(path), fresh = window.MemorizzCode.tokenizer(path);
        return {
            old: old,
            fresh: fresh,
            both: function(text) { old(text); return fresh(text); },
        };
    }

    function fold(file, hidden, line) {
        return el('tbody', { class: 'ed-fold' }, [el('tr', {}, [el('td', { colspan: '4' }, [
            el('a', { href: fileUrl(file.path, line), title: 'Open the whole file' }, [
                el('span', { class: 'ed-fold-dots', 'aria-hidden': 'true', text: '⋯' }),
                hidden + (hidden === 1 ? ' unchanged line' : ' unchanged lines'),
            ]),
        ])])]);
    }

    function fileCard(file, index) {
        const letters = { added: ['A', 'Added'], deleted: ['D', 'Deleted'], renamed: ['R', 'Renamed'], changed: ['M', 'Modified'] };
        const letter = letters[file.status] || letters.changed;
        // Two halves do not fit a phone; it always gets one column.
        const split = layout === 'split' && !narrow.matches;
        const table = el('table', { class: 'ed-table ed-table--' + (split ? 'split' : 'unified') });
        if (split) {
            // The fold rows span the table, so the widths are set here.
            table.append(el('colgroup', {}, [el('col', { class: 'ed-col-num' }), el('col'), el('col', { class: 'ed-col-num' }), el('col')]));
        }
        const read = readers(file.path);
        let shown = 0;
        file.hunks.forEach(function(hunk) {
            const first = hunk.lines.find(function(line) { return line.new != null; });
            const start = first ? first.new : shown + 1;
            if (start - shown - 1 > 0) table.append(fold(file, start - shown - 1, shown + 1));
            const body = el('tbody');
            if (split) splitRows(body, file, hunk, read); else unifiedRows(body, file, hunk, read);
            table.append(body);
            hunk.lines.forEach(function(line) { if (line.new != null) shown = line.new; });
        });
        const content = file.binary
            ? el('p', { class: 'sdiff-note', text: 'A binary file; its contents are not shown.' })
            : file.hunks.length ? el('div', { class: 'ed-scroll' }, [table])
            : el('p', { class: 'sdiff-note', text: 'No lines changed.' });
        const slash = file.path.lastIndexOf('/');
        const gone = file.status === 'deleted';
        return el('details', { class: 'ed-file', id: 'sdiff-file-' + index, open: 'open' }, [
            el('summary', {}, [
                el('span', { class: 'ed-file-name' }, [
                    el('strong', { text: file.path.slice(slash + 1) }),
                    el('span', { class: 'ed-file-dir', text: file.status === 'renamed' ? file.old_path + ' → ' + file.path : file.path.slice(0, Math.max(slash, 0)) }),
                ]),
                counts(file.insertions, file.deletions),
                el('span', { class: 'ed-status ed-status--' + file.status, title: letter[1], text: letter[0] }),
                el('a', { class: 'ed-open', href: fileUrl(gone ? file.old_path : file.path, null, gone), text: 'Open file' }),
            ]),
            whyBlock(file),
            content,
        ]);
    }

    // Why this file was changed, in the coding model's words (the explainer
    // kept with the version), each reason leading to its line.
    function whyBlock(file) {
        const reasons = (((story || {}).explanation || {}).changes || []).filter(function(change) {
            const path = String(change.file || '');
            return path === file.path || file.path.endsWith('/' + path) || path.endsWith('/' + file.path);
        });
        if (!reasons.length) return null;
        return el('div', { class: 'ed-why' }, [
            el('span', { class: 'ed-why-label', text: 'Why this changed' }),
            el('ul', {}, reasons.map(function(change) {
                return el('li', {}, [
                    change.line ? el('a', { class: 'ed-why-line', href: fileUrl(file.path, change.line, false), text: 'Line ' + change.line }) : null,
                    el('span', { class: 'ed-why-what', text: change.what || '' }),
                    change.why ? el('span', { class: 'ed-why-reason', text: change.why }) : null,
                ]);
            })),
        ]);
    }

    function traceUrl(work) {
        if (!work || !work.agent_id) return '';
        const params = new URLSearchParams({ agent_id: work.agent_id });
        if (work.thread_id) params.set('thread_id', work.thread_id);
        if (work.memory_id) params.set('thread_memory_id', work.memory_id);
        return '/traces?' + params.toString() + '#trace-timeline';
    }


    function seconds(from, to) {
        const span = (new Date(to) - new Date(from)) / 1000;
        if (!(span >= 0)) return '';
        return span < 90 ? Math.round(span) + ' s' : Math.round(span / 60) + ' min';
    }

    // Long text starts folded, with the first lines in view.
    function longText(text, lines) {
        const block = el('pre', { class: 'story-text', text: text });
        if ((text || '').split('\n').length <= lines + 2 && (text || '').length < 700) return block;
        block.classList.add('is-folded');
        block.style.setProperty('--story-lines', String(lines));
        const more = el('button', { type: 'button', class: 'story-more', text: 'Show all', onclick: function() {
            const folded = block.classList.toggle('is-folded');
            more.textContent = folded ? 'Show all' : 'Show less';
        } });
        return el('div', {}, [block, more]);
    }

    function errorCard(entry) {
        const frames = entry.frames || [];
        const frame = frames[frames.length - 1];
        const text = entry.error || entry.error_message || entry.message || entry.content || entry.status || 'A step failed';
        const at = entry.at || entry.timestamp;
        const where = frame && frame.file ? el('a', {
            class: 'story-frame',
            href: page + '/file?' + new URLSearchParams({ path: frame.file, version: picker.value, side: 'old' }).toString() + '#L' + frame.line,
            title: 'Open the file as it was, at this line',
            text: frame.file + ':' + frame.line + (frame.function ? ' in ' + frame.function : ''),
        }) : null;
        return el('div', { class: 'story-error' }, [
            el('code', { class: 'story-error-text', text: String(text) }),
            frame && frame.code ? el('code', { class: 'story-error-code', text: String(frame.code).trim() }) : null,
            el('div', { class: 'story-meta' }, [
                where,
                el('span', { text: [entry.from === 'trace' ? 'from a trace' : 'from its error log', when(at)].filter(Boolean).join(' · ') }),
            ]),
        ]);
    }

    function storyStep(number, kind, title, meta, children) {
        return el('li', { class: 'story-step story-step--' + kind }, [
            el('span', { class: 'story-dot', 'aria-hidden': 'true', text: String(number) }),
            el('div', { class: 'story-body' }, [
                el('div', { class: 'story-head' }, [el('h3', { text: title }), meta ? el('span', { class: 'story-meta', text: meta }) : null]),
            ].concat(children || [])),
        ]);
    }

    function drawStory() {
        const panel = $('sdiff-story');
        const version = versions.find(function(item) { return item.id === picker.value; });
        panel.hidden = !version;
        if (!version) return;
        if (!story) {
            panel.replaceChildren(el('p', { class: 'story-none', text: 'No account was kept with this version: it was made by hand, restored, or made before accounts were kept.' }));
            return;
        }
        const errors = story.errors || [], diagnosis = story.diagnosis, prompt = story.prompt || {};
        const steps = [];
        let number = 0;
        const shown = errors.slice(-3).reverse();
        const pointed = document.getElementById('sdiff-blamed');
        steps.push(storyStep(++number, 'error',
            errors.length ? (errors.length === 1 ? 'It noticed an error' : 'It noticed ' + errors.length + ' errors') : 'No error was on record',
            errors.length ? (story.errors_read_by_model ? 'read by the coding model' : 'on record when the fix began') : '',
            shown.map(errorCard).concat([
                errors.length > shown.length ? el('p', { class: 'story-note', text: 'and ' + (errors.length - shown.length) + ' earlier.' }) : null,
                (story.traces_read || []).length ? el('p', { class: 'story-note', text: 'It also opened ' + story.traces_read.length + (story.traces_read.length === 1 ? ' trace' : ' traces') + ' of the turn that failed.' }) : null,
                pointed ? el('p', { class: 'story-note' }, [el('a', { href: '#sdiff-blamed', text: 'The line the error named is marked in the change below' }), '.']) : null,
            ])));
        if (diagnosis && diagnosis.text) {
            steps.push(storyStep(++number, 'think', 'It found the cause',
                [diagnosis.coding_model, seconds(diagnosis.started_at, diagnosis.finished_at)].filter(Boolean).join(' · '),
                [
                    longText(diagnosis.text, 7),
                    traceUrl(diagnosis.work_trace) ? el('p', { class: 'story-note' }, [
                        el('a', { href: traceUrl(diagnosis.work_trace), text: 'Read how it got there in Traces' }), '.',
                    ]) : null,
                ]));
        }
        steps.push(storyStep(++number, 'ask', 'It was asked to fix it',
            'the prompt sent to ' + (story.coding_model || 'the coding model'),
            [
                longText(prompt.task || '', 6),
                prompt.instruction ? el('details', { class: 'story-details' }, [
                    el('summary', { text: 'Standing instructions sent with it' }),
                    el('pre', { class: 'story-text', text: prompt.instruction }),
                ]) : null,
            ]));
        // The same step twice in a row (the model checks, then MemoRizz does) reads as one.
        const work = (story.steps || []).filter(function(step, index, all) {
            return step.text !== 'Working on a fix' && !(index && all[index - 1].text === step.text);
        });
        steps.push(storyStep(++number, 'work', 'It made the change',
            [seconds(story.started_at, story.finished_at), story.repaired ? 'repaired once after a failed check' : ''].filter(Boolean).join(' · '),
            [
                el('ol', { class: 'story-trail' }, work.filter(function(step) { return step.text !== 'Explaining the change'; }).map(function(step) { return el('li', { text: step.text }); })),
                traceUrl(story.work_trace) ? el('p', { class: 'story-note' }, [
                    el('a', { href: traceUrl(story.work_trace), text: 'Read every model call and tool call of this work in Traces' }),
                    // An account kept before both were one thread names the explanation's separately.
                    story.work_trace.explained_in ? ' (and ' : null,
                    story.work_trace.explained_in ? el('a', { href: traceUrl(Object.assign({}, story.work_trace, { thread_id: story.work_trace.explained_in })), text: 'of the explanation' }) : null,
                    story.work_trace.explained_in ? ').' : '.',
                ]) : null,
            ]));
        const told = story.explanation;
        if (told) {
            const fact = function(label, text) {
                return text ? el('div', { class: 'story-fact' }, [el('dt', { text: label }), el('dd', { text: text })]) : null;
            };
            steps.push(storyStep(++number, 'why', 'It explained the change', 'in ' + (story.coding_model || 'the coding model') + '\u2019s words',
                [
                    told.summary ? el('p', { class: 'story-summary', text: told.summary }) : null,
                    el('dl', { class: 'story-facts' }, [
                        fact('Root cause', told.root_cause),
                        fact('Why this fix', told.why_this_way),
                        fact('Risk', told.risk),
                        fact('To confirm', told.check_it),
                    ]),
                    (told.changes || []).length ? el('p', { class: 'story-note' }, [el('a', { href: '#sdiff-file-0', text: 'The reason for each edit is shown above its file below' }), '.']) : null,
                ]));
        }
        const checks = ((story.check || {}).steps || []);
        steps.push(storyStep(++number, 'kept', 'The copy passed its check and was kept as v' + version.number, when(story.finished_at),
            [
                el('ul', { class: 'story-checks' }, checks.map(function(step) {
                    return el('li', { class: step.ok ? 'is-ok' : 'is-bad', text: (step.ok ? '✓ ' : '✗ ') + step.name });
                })),
                story.answer ? el('details', { class: 'story-details' }, [
                    el('summary', { text: 'What the model said when it finished' }),
                    el('pre', { class: 'story-text', text: story.answer }),
                ]) : null,
            ]));
        const first = errors.length ? String(errors[errors.length - 1].error || '').split('. ')[0] : '';
        panel.replaceChildren(el('details', { class: 'story-fold', open: storyOpen ? 'open' : null, ontoggle: function() {
            storyOpen = this.open;
            try { window.localStorage.setItem('memorizz.diffStory', storyOpen ? '1' : '0'); } catch (error) { /* not remembered */ }
        } }, [
            el('summary', {}, [
                el('h2', { class: 'panel-title', text: 'How this change came about' }),
                el('span', { class: 'story-meta', text: steps.length + ' steps' + (first ? ' · ' + first : '') }),
                el('a', { class: 'btn btn-sm story-jump', href: '#sdiff-bar', text: 'Skip to the change', onclick: function(event) { event.stopPropagation(); } }),
            ]),
            el('ol', { class: 'story-steps' }, steps),
        ]));
    }

    function draw() {
        const main = $('sdiff-main'), nav = $('sdiff-files');
        main.replaceChildren();
        nav.replaceChildren();
        nav.hidden = files.length < 2;
        const plus = files.reduce(function(sum, file) { return sum + file.insertions; }, 0);
        const minus = files.reduce(function(sum, file) { return sum + file.deletions; }, 0);
        $('sdiff-bar').hidden = !files.length;
        $('sdiff-bar-count').replaceChildren(counts(plus, minus), ' in ' + files.length + (files.length === 1 ? ' file' : ' files'));
        root.classList.toggle('has-files', files.length >= 2);
        if (!files.length) {
            main.append(el('div', { class: 'empty-state' }, [
                el('strong', { text: 'Nothing was changed here' }),
                el('p', { text: picker.value === 'working' ? agentName + ' has no edits waiting to be kept.' : 'This is the same as the version before it.' }),
            ]));
            return;
        }
        nav.append(el('h2', { text: files.length + ' files changed' }));
        files.forEach(function(file, index) {
            nav.append(el('a', { href: '#sdiff-file-' + index, title: file.path }, [
                el('span', { class: 'sdiff-name', text: file.path.split('/').pop() }),
                el('span', { class: 'ed-status ed-status--' + file.status, text: { added: 'A', deleted: 'D', renamed: 'R' }[file.status] || 'M' }),
            ]));
            main.append(fileCard(file, index));
        });
    }

    function summary() {
        const panel = $('sdiff-summary');
        const plus = files.reduce(function(sum, file) { return sum + file.insertions; }, 0);
        const minus = files.reduce(function(sum, file) { return sum + file.deletions; }, 0);
        const version = versions.find(function(item) { return item.id === picker.value; });
        const position = versions.indexOf(version);
        let title, note;
        if (version) {
            title = 'v' + version.number + ' · ' + version.message;
            note = when(version.at) + ' · compared with v' + (version.number - 1);
        } else if (picker.value === 'all') {
            title = 'Everything since MemoRizz was installed';
            note = 'v' + (versions.length ? versions[0].number : 0) + ' compared with v0, the installed program';
        } else {
            title = 'Edits not yet kept';
            note = 'Made since the last version and not checked';
        }
        const step = function(label, target) {
            return el('button', { class: 'btn btn-sm', type: 'button', text: label, disabled: target ? null : 'disabled', onclick: function() { picker.value = target.id; load(); } });
        };
        // The list runs newest first; v0 has no change of its own to show.
        const older = version && versions[position + 1] && !versions[position + 1].installed ? versions[position + 1] : null;
        const newer = version && position > 0 ? versions[position - 1] : null;
        panel.hidden = false;
        panel.replaceChildren(
            el('div', { class: 'sdiff-summary-text' }, [el('h2', { class: 'panel-title', text: title }), el('p', { class: 'panel-description', text: note })]),
            el('div', { class: 'sdiff-summary-side' }, [
                el('span', { class: 'sdiff-total' }, [counts(plus, minus), ' in ' + files.length + (files.length === 1 ? ' file' : ' files')]),
                version ? step('Older', older) : null,
                version ? step('Newer', newer) : null,
                version ? el('button', { class: 'btn btn-sm btn-primary', type: 'button', text: 'Preview this version', title: 'Run this version as its own read-only console', onclick: function() { previewVersion(version, this); } }) : null,
            ])
        );
    }

    async function load() {
        const main = $('sdiff-main');
        const params = new URLSearchParams(window.location.search);
        params.set('version', picker.value);
        window.history.replaceState(null, '', window.location.pathname + '?' + params.toString());
        try {
            const data = await get('/diff?version=' + encodeURIComponent(picker.value));
            files = data.files || [];
            story = data.story || null;
            blamed = {};
            ((story && story.errors) || []).forEach(function(entry) {
                (entry.frames || []).forEach(function(frame) {
                    if (!frame.file || !frame.line) return;
                    (blamed[frame.file] = blamed[frame.file] || {})[frame.line] = true;
                });
            });
            summary();
            draw();
            // After the files, so the account can point at the marked line.
            drawStory();
        } catch (error) {
            $('sdiff-summary').hidden = true;
            $('sdiff-story').hidden = true;
            $('sdiff-bar').hidden = true;
            $('sdiff-files').hidden = true;
            main.replaceChildren(el('div', { class: 'empty-state' }, [el('strong', { text: 'This change could not be shown' }), el('p', { text: String(error.message || error) })]));
        }
    }

    // Run this version as its own read-only console, in a new window.
    async function previewVersion(version, button) {
        const label = button.textContent;
        button.disabled = true;
        button.textContent = 'Starting…';
        const tab = window.open('about:blank', '_blank');
        if (tab) tab.document.write('<p style="font:14px system-ui;padding:24px">Starting v' + version.number + '…</p>');
        try {
            const response = await fetch(base + '/preview', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ version: version.id }) });
            const data = await response.json().catch(function() { return {}; });
            if (!response.ok) throw new Error(data.detail || ('HTTP ' + response.status));
            if (tab) tab.location = data.preview.url; else window.location = data.preview.url;
        } catch (error) {
            if (tab) tab.close();
            if (window.MemorizzDialogs) window.MemorizzDialogs.notify(String(error.message || error));
        }
        button.disabled = false;
        button.textContent = label;
    }

    function setLayout(next, remember) {
        layout = next === 'split' ? 'split' : 'unified';
        document.querySelectorAll('[data-layout]').forEach(function(button) {
            button.setAttribute('aria-pressed', button.dataset.layout === layout ? 'true' : 'false');
        });
        // Only a choice is remembered, so a new default reaches everyone else.
        if (remember) { try { window.localStorage.setItem('memorizz.diffView', layout); } catch (error) { /* not remembered */ } }
    }

    async function start() {
        setLayout(layout);
        try {
            const state = await get('');
            versions = state.versions || [];
            if (!versions.length) {
                $('sdiff-main').replaceChildren(el('div', { class: 'empty-state' }, [
                    el('strong', { text: agentName + ' has no copy of its source yet' }),
                    el('p', {}, [el('a', { href: '/agents/' + encodeURIComponent(agentId) + '/source', text: 'Give it one on the source page' }), '.']),
                ]));
                picker.disabled = true;
                return;
            }
            const kept = versions.filter(function(version) { return !version.installed; });
            kept.forEach(function(version) {
                picker.append(el('option', { value: version.id, text: 'v' + version.number + ' · ' + version.message }));
            });
            picker.append(el('option', { value: 'all', text: 'Everything since install' }));
            picker.append(el('option', { value: 'working', text: 'Edits not yet kept' }));
            $('sdiff-patch').hidden = !kept.length;
            const asked = new URLSearchParams(window.location.search).get('version');
            const known = Array.from(picker.options).some(function(option) { return option.value === asked; });
            picker.value = known ? asked : (kept.length ? kept[0].id : 'all');
            await load();
        } catch (error) {
            $('sdiff-main').replaceChildren(el('div', { class: 'empty-state' }, [el('strong', { text: 'The source could not be read' }), el('p', { text: String(error.message || error) })]));
        }
    }

    narrow.addEventListener('change', draw);
    picker.addEventListener('change', load);
    document.querySelectorAll('[data-layout]').forEach(function(button) {
        button.addEventListener('click', function() { setLayout(button.dataset.layout, true); draw(); });
    });
    start();
})();

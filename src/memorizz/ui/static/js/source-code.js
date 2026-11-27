// Code drawn the way an editor draws it: Python coloured by token, and the
// changed part of a line marked. Shared by the diff and the file pages.
window.MemorizzCode = (function() {
    const FLOW = new Set('if elif else for while try except finally return raise break continue pass yield import from as with await async match case'.split(' '));
    const WORD = new Set('def class lambda and or not in is None True False global nonlocal del assert'.split(' '));
    const SELF = new Set(['self', 'cls']);
    const BUILTIN = new Set('print len range str int float bool list dict set tuple isinstance getattr setattr hasattr super type open sorted enumerate zip map filter any all min max sum repr bytes object Exception ValueError TypeError KeyError RuntimeError'.split(' '));
    const TOKEN = /(#.*)|((?:[rbfuRBFU]{1,2})?(?:"""|'''|"|'))|(\b(?:0[xX][\da-fA-F_]+|\d[\d_]*(?:\.[\d_]+)?(?:[eE][+-]?\d+)?)\b)|(@[A-Za-z_][\w.]*)|([A-Za-z_]\w*)|(\s+)|([^\s\w])/y;

    // A tokenizer keeps whether a triple-quoted string is still open, so
    // feed it the lines of one file in order.
    function python() {
        let open = null;
        return function(line) {
            const out = [];
            let at = 0, previous = '';
            const push = function(kind, text) { if (text) out.push([kind, text]); };
            const string = function(start, quote) {
                // Reads to the closing quote; a triple quote may run on.
                let index = start;
                while (index < line.length) {
                    if (line[index] === '\\') { index += 2; continue; }
                    if (line.startsWith(quote, index)) return index + quote.length;
                    index++;
                }
                if (quote.length === 3) open = quote;
                return line.length;
            };
            if (open) {
                const quote = open;
                open = null;
                at = string(0, quote);
                push('str', line.slice(0, at));
            }
            while (at < line.length) {
                TOKEN.lastIndex = at;
                const found = TOKEN.exec(line);
                if (!found) { push('', line.slice(at)); break; }
                const text = found[0];
                let end = at + text.length;
                if (found[1]) push('com', text);
                else if (found[2]) {
                    const quote = text.replace(/^[rbfuRBFU]+/, '');
                    end = string(end, quote);
                    push('str', line.slice(at, end));
                } else if (found[3]) push('num', text);
                else if (found[4]) push('fn', text);
                else if (found[5]) {
                    let kind = '';
                    if (previous === 'def') kind = 'fn';
                    else if (previous === 'class') kind = 'cls';
                    else if (FLOW.has(text)) kind = 'flow';
                    else if (WORD.has(text)) kind = 'kw';
                    else if (SELF.has(text)) kind = 'self';
                    else if (line[end] === '(') kind = BUILTIN.has(text) || text[0] === text[0].toLowerCase() ? 'fn' : 'cls';
                    else if (/^[A-Z][a-z]/.test(text)) kind = 'cls';
                    else if (/^[A-Z][A-Z0-9_]+$/.test(text)) kind = 'const';
                    push(kind, text);
                    previous = text;
                } else push('', text);
                at = end;
            }
            return out;
        };
    }

    function plain() {
        return function(line) { return line ? [['', line]] : []; };
    }

    function tokenizer(path) {
        return /\.pyi?$/.test(path || '') ? python() : plain();
    }

    // One line of code as nodes; [from, to) is wrapped in <mark>.
    function render(target, tokens, from, to) {
        const marked = from != null && to > from;
        let offset = 0;
        const piece = function(kind, text, parent) {
            if (!text) return;
            if (!kind) { parent.append(text); return; }
            const span = document.createElement('span');
            span.className = 'tk-' + kind;
            span.textContent = text;
            parent.append(span);
        };
        let mark = null;
        tokens.forEach(function(token) {
            const kind = token[0], text = token[1];
            const start = offset, end = offset + text.length;
            offset = end;
            if (!marked || end <= from || start >= to) { piece(kind, text, target); return; }
            const a = Math.max(from, start) - start, b = Math.min(to, end) - start;
            piece(kind, text.slice(0, a), target);
            if (!mark) { mark = document.createElement('mark'); }
            if (!mark.parentNode) target.append(mark);
            piece(kind, text.slice(a, b), mark);
            if (b < text.length) { piece(kind, text.slice(b), target); }
        });
        if (!target.childNodes.length) target.append(' ');
        return target;
    }

    return { tokenizer: tokenizer, render: render };
})();

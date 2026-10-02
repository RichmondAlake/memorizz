/* Markdown for untrusted model and harness output.
   Shared by the playground and the harness chat. Load marked first.
   MemorizzMarkdown.render(text) returns HTML in which raw HTML shows as text,
   links keep only web, mail and in-app targets, and images load only from
   this app or inline data (a remote image is a link, so a reply cannot make
   the browser call out, say to leak what is on screen through its URL). */
(function () {
    const esc = escapeHtml; // base.html
    const SAFE_LINK = /^(https?:|mailto:|\/|#)/i;
    const LOCAL_IMAGE = /^(\/(?!\/)|data:image\/(png|jpe?g|gif|webp);base64,)/i;
    const plain = (text) => '<p>' + esc(text || '').replace(/\n/g, '<br>') + '</p>';

    const parser = typeof marked === 'undefined' ? null : new marked.Marked({
        gfm: true,
        breaks: true,
        renderer: {
            html(token) { return esc(token.text || token.raw || ''); },
            image(token) {
                const src = String(token.href || '');
                const alt = esc(token.text || '');
                if (LOCAL_IMAGE.test(src)) {
                    const title = token.title ? ' title="' + esc(token.title) + '"' : '';
                    return '<img src="' + esc(src) + '" alt="' + alt + '"' + title + ' loading="lazy">';
                }
                if (/^https?:/i.test(src)) {
                    return '<a href="' + esc(src) + '" target="_blank" rel="noopener noreferrer">' + (alt || 'Image') + '</a>';
                }
                return alt;
            },
            link(token) {
                const text = this.parser.parseInline(token.tokens);
                const href = String(token.href || '');
                if (!SAFE_LINK.test(href)) return text;
                const title = token.title ? ' title="' + esc(token.title) + '"' : '';
                return '<a href="' + esc(href) + '"' + title + ' target="_blank" rel="noopener noreferrer">' + text + '</a>';
            },
        },
    });

    const render = (text) => {
        if (!parser || !text || !String(text).trim()) return plain(text);
        try { return parser.parse(String(text)); } catch (_) { return plain(text); }
    };

    window.MemorizzMarkdown = { render, esc };
})();

/* In-page confirmations and notices, in place of the browser's confirm() and
   alert(), which block the page (and any automation driving it).
   MemorizzDialogs.confirm(message, {detail, confirmLabel, cancelLabel, danger})
   resolves true only when the confirming button is pressed; Escape or the
   other button resolves false. MemorizzDialogs.notify(message, {tone}) shows a
   notice that closes itself.
   MemorizzDialogs.prompt(message, {detail, facts, label, value, placeholder,
   multiline, mono, readOnly, required, validate, confirmLabel, cancelLabel,
   altLabel}) asks for one value. It resolves the text, or null when closed.
   `facts` is a list of [label, value] rows shown above the field; `validate`
   returns a message to refuse a value. With `altLabel` (a second, destructive
   answer such as Reject) it resolves {action: 'ok' | 'alt', value}. */
(function () {
    const confirmDialog = (message, options = {}) => new Promise((resolve) => {
        const dialog = document.createElement('dialog');
        dialog.className = 'mz-dialog';
        dialog.setAttribute('aria-labelledby', 'mz-dialog-text');
        const form = document.createElement('form');
        form.method = 'dialog';
        const text = document.createElement('p');
        text.className = 'mz-dialog-text';
        text.id = 'mz-dialog-text';
        text.textContent = message;
        form.append(text);
        if (options.detail) {
            const detail = document.createElement('p');
            detail.className = 'mz-dialog-detail';
            detail.textContent = options.detail;
            form.append(detail);
        }
        const actions = document.createElement('div');
        actions.className = 'mz-dialog-actions';
        const cancel = document.createElement('button');
        cancel.className = 'btn';
        cancel.value = 'cancel';
        cancel.textContent = options.cancelLabel || 'Keep';
        const ok = document.createElement('button');
        ok.className = 'btn ' + (options.danger === false ? 'btn-primary' : 'btn-danger');
        ok.value = 'ok';
        ok.textContent = options.confirmLabel || 'Delete';
        actions.append(cancel, ok);
        form.append(actions);
        dialog.append(form);
        dialog.addEventListener('close', () => {
            resolve(dialog.returnValue === 'ok');
            dialog.remove();
        });
        document.body.append(dialog);
        dialog.showModal();
        cancel.focus();  // The safe choice is the default.
    });

    let promptCount = 0;
    const promptDialog = (message, options = {}) => new Promise((resolve) => {
        const id = 'mz-prompt-' + (++promptCount);
        const dialog = document.createElement('dialog');
        dialog.className = 'mz-dialog mz-dialog--prompt';
        dialog.setAttribute('aria-labelledby', id + '-text');
        const form = document.createElement('form');
        form.method = 'dialog';
        form.noValidate = true;
        const text = document.createElement('p');
        text.className = 'mz-dialog-text';
        text.id = id + '-text';
        text.textContent = message;
        form.append(text);
        if (options.detail) {
            const detail = document.createElement('p');
            detail.className = 'mz-dialog-detail';
            detail.textContent = options.detail;
            form.append(detail);
        }
        if (options.facts && options.facts.length) {
            const list = document.createElement('dl');
            list.className = 'mz-dialog-facts';
            options.facts.forEach(([name, value]) => {
                const term = document.createElement('dt');
                term.textContent = name;
                const data = document.createElement('dd');
                data.textContent = value == null || value === '' ? '—' : String(value);
                list.append(term, data);
            });
            form.append(list);
        }
        const label = document.createElement('label');
        label.className = 'mz-dialog-label';
        label.htmlFor = id + '-input';
        label.textContent = options.label || message;
        if (!options.label) label.classList.add('visually-hidden');
        const input = document.createElement(options.multiline ? 'textarea' : 'input');
        input.id = id + '-input';
        input.className = 'mz-dialog-input' + (options.mono ? ' mono' : '');
        if (options.multiline) input.rows = options.rows || 8;
        else input.type = 'text';
        input.value = options.value == null ? '' : String(options.value);
        if (options.placeholder) input.placeholder = options.placeholder;
        if (options.readOnly) input.readOnly = true;
        input.spellcheck = false;
        input.autocomplete = 'off';
        input.setAttribute('aria-describedby', id + '-error');
        const error = document.createElement('p');
        error.className = 'mz-dialog-error';
        error.id = id + '-error';
        error.setAttribute('role', 'alert');
        form.append(label, input, error);
        const actions = document.createElement('div');
        actions.className = 'mz-dialog-actions';
        const button = (labelText, className) => {
            const el = document.createElement('button');
            el.type = 'button';
            el.className = className;
            el.textContent = labelText;
            actions.append(el);
            return el;
        };
        const cancel = button(options.cancelLabel || 'Cancel', 'btn');
        const alt = options.altLabel ? button(options.altLabel, 'btn btn-danger') : null;
        const ok = button(options.confirmLabel || 'Save', 'btn btn-primary');
        form.append(actions);
        dialog.append(form);

        let answer = null;
        const finish = (action) => {
            const value = input.value;
            const problem = (options.required && !value.trim() ? 'Enter a value.' : '')
                || (options.validate ? options.validate(value, action) : '');
            if (problem) {
                error.textContent = problem;
                input.setAttribute('aria-invalid', 'true');
                input.focus();
                return;
            }
            answer = options.altLabel ? {action, value} : value;
            dialog.close();
        };
        cancel.addEventListener('click', () => dialog.close());
        ok.addEventListener('click', () => finish('ok'));
        if (alt) alt.addEventListener('click', () => finish('alt'));
        input.addEventListener('input', () => {
            error.textContent = '';
            input.removeAttribute('aria-invalid');
        });
        // Enter answers from a one-line field; in a text area it is a new line
        // (Ctrl/⌘ + Enter answers there).
        input.addEventListener('keydown', (event) => {
            if (event.key !== 'Enter' || event.isComposing) return;
            if (options.multiline && !(event.metaKey || event.ctrlKey)) return;
            event.preventDefault();
            finish('ok');
        });
        dialog.addEventListener('close', () => {
            resolve(answer);
            dialog.remove();
        });
        document.body.append(dialog);
        dialog.showModal();
        input.focus();
        if (!options.multiline || options.readOnly) input.select();
    });

    let region = null;
    const notify = (message, options = {}) => {
        if (!region) {
            region = document.createElement('div');
            region.className = 'mz-notices';
            region.setAttribute('role', 'status');
            region.setAttribute('aria-live', 'polite');
            document.body.append(region);
        }
        const notice = document.createElement('div');
        notice.className = 'mz-notice mz-notice--' + (options.tone || 'error');
        const text = document.createElement('span');
        text.textContent = message;
        const close = document.createElement('button');
        close.type = 'button';
        close.className = 'mz-notice-close';
        close.setAttribute('aria-label', 'Dismiss');
        close.textContent = '×';
        close.addEventListener('click', () => notice.remove());
        notice.append(text, close);
        region.append(notice);
        setTimeout(() => notice.remove(), options.timeout || 8000);
    };

    window.MemorizzDialogs = {confirm: confirmDialog, prompt: promptDialog, notify};
})();

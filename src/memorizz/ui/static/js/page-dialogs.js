/* In-page confirmations and notices, in place of the browser's confirm() and
   alert(), which block the page (and any automation driving it).
   MemorizzDialogs.confirm(message, {detail, confirmLabel, cancelLabel, danger})
   resolves true only when the confirming button is pressed; Escape or the
   other button resolves false. MemorizzDialogs.notify(message, {tone}) shows a
   notice that closes itself. */
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

    window.MemorizzDialogs = {confirm: confirmDialog, notify};
})();

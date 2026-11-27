// A menu drawn by the console in place of the browser's own for a <select>.
// The <select> stays in the page as the value that is submitted and that
// other scripts read and set; this only replaces what is seen and clicked.
//   MemorizzSelect.enhance(select)        one field
//   MemorizzSelect.enhanceAll(container)  every <select> inside
window.MemorizzSelect = (function() {
    const enhanced = new WeakMap();
    let openOne = null;

    function make(tag, className, text) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text != null) node.textContent = text;
        return node;
    }

    function enhance(select) {
        if (!select || select.multiple || select.size > 1 || select.hasAttribute('data-native') || enhanced.has(select)) return enhanced.get(select) || null;
        const wrap = make('div', 'ms');
        // A select that filled its row still does; one sized to its text sits inline.
        const width = getComputedStyle(select).width;
        const parentWidth = select.parentElement ? select.parentElement.clientWidth : 0;
        const fills = /%$/.test(width) ? parseFloat(width) >= 90 : (parentWidth > 0 && select.offsetWidth >= parentWidth * 0.9);
        if (!fills) {
            wrap.classList.add('ms--inline');
            const capped = getComputedStyle(select).maxWidth;
            if (/px$/.test(capped)) wrap.style.maxWidth = capped;
            const floor = getComputedStyle(select).minWidth;
            if (/px$/.test(floor) && parseFloat(floor) > 0) wrap.style.minWidth = floor;
        }
        const button = make('button', 'ms-button');
        button.type = 'button';
        button.setAttribute('aria-haspopup', 'listbox');
        button.setAttribute('aria-expanded', 'false');
        const label = make('span', 'ms-value');
        button.append(label);
        button.insertAdjacentHTML('beforeend', '<svg class="ms-chevron" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg>');
        const menu = make('ul', 'ms-menu');
        menu.setAttribute('role', 'listbox');
        menu.tabIndex = -1;
        menu.hidden = true;
        const base = select.id || 'ms-' + Math.random().toString(36).slice(2, 8);
        menu.id = base + '-menu';
        button.setAttribute('aria-controls', menu.id);
        select.parentNode.insertBefore(wrap, select.nextSibling);
        wrap.append(button, menu);
        select.classList.add('ms-native');
        select.tabIndex = -1;
        select.setAttribute('aria-hidden', 'true');
        // The field's <label> names the button too.
        const named = select.id && document.querySelector('label[for="' + select.id + '"]');
        if (named) {
            named.id = named.id || base + '-label';
            button.id = base + '-button';
            button.setAttribute('aria-labelledby', named.id + ' ' + button.id);
            named.addEventListener('click', function(event) { event.preventDefault(); button.focus(); });
        } else if (select.getAttribute('aria-label')) {
            button.setAttribute('aria-label', select.getAttribute('aria-label'));
        }

        let items = [], active = -1, typed = '', typedAt = 0;

        function build() {
            menu.replaceChildren();
            items = [];
            Array.from(select.children).forEach(function(child) {
                const options = child.tagName === 'OPTGROUP' ? Array.from(child.children) : [child];
                if (child.tagName === 'OPTGROUP') {
                    const head = make('li', 'ms-group', child.label);
                    head.setAttribute('role', 'presentation');
                    menu.append(head);
                }
                options.forEach(function(option) {
                    if (option.tagName !== 'OPTION' || option.hidden) return;
                    const item = make('li', 'ms-option');
                    item.setAttribute('role', 'option');
                    item.id = menu.id + '-' + items.length;
                    item.append(make('span', 'ms-option-text', option.textContent.trim() || ' '));
                    item.insertAdjacentHTML('beforeend', '<svg class="ms-check" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m5 12 5 5 9-10"/></svg>');
                    if (option.disabled) item.setAttribute('aria-disabled', 'true');
                    const index = items.length;
                    item.addEventListener('click', function() { if (!option.disabled) choose(index); });
                    item.addEventListener('mousemove', function() { setActive(index, false); });
                    items.push({ node: item, option: option });
                    menu.append(item);
                });
            });
            sync();
        }

        // Shows what the <select> holds; safe to call at any time.
        function sync() {
            const chosen = select.options[select.selectedIndex];
            label.textContent = chosen ? chosen.textContent.trim() : '';
            label.classList.toggle('is-empty', !chosen || !chosen.value);
            items.forEach(function(item) { item.node.setAttribute('aria-selected', item.option === chosen ? 'true' : 'false'); });
            button.disabled = select.disabled;
            wrap.hidden = select.hidden || select.style.display === 'none';
        }

        function setActive(index, scroll) {
            active = index;
            items.forEach(function(item, at) { item.node.classList.toggle('is-active', at === index); });
            if (items[index]) {
                menu.setAttribute('aria-activedescendant', items[index].node.id);
                if (scroll !== false) items[index].node.scrollIntoView({ block: 'nearest' });
            }
        }

        function place() {
            const box = button.getBoundingClientRect();
            const room = window.innerHeight - box.bottom - 12, above = box.top - 12;
            const up = room < 180 && above > room;
            menu.style.minWidth = box.width + 'px';
            menu.style.left = Math.max(8, Math.min(box.left, window.innerWidth - Math.max(box.width, menu.offsetWidth) - 8)) + 'px';
            menu.style.maxHeight = Math.max(140, Math.min(320, up ? above : room)) + 'px';
            if (up) { menu.style.top = ''; menu.style.bottom = (window.innerHeight - box.top + 6) + 'px'; }
            else { menu.style.bottom = ''; menu.style.top = (box.bottom + 6) + 'px'; }
            // Inside a moved or scaled container "fixed" is measured from it,
            // not the window: see where the menu landed and correct.
            const landed = menu.getBoundingClientRect();
            const wantLeft = parseFloat(menu.style.left);
            const wantTop = up ? box.top - 6 - landed.height : box.bottom + 6;
            const dx = wantLeft - landed.left, dy = wantTop - landed.top;
            if (Math.abs(dx) > 1) menu.style.left = (wantLeft + dx) + 'px';
            if (Math.abs(dy) > 1) {
                if (up) menu.style.bottom = (parseFloat(menu.style.bottom) - dy) + 'px';
                else menu.style.top = (parseFloat(menu.style.top) + dy) + 'px';
            }
        }

        function open() {
            if (button.disabled || !menu.hidden) return;
            if (openOne) openOne();
            menu.hidden = false;
            button.setAttribute('aria-expanded', 'true');
            place();
            setActive(Math.max(0, items.findIndex(function(item) { return item.option.selected; })));
            menu.focus({ preventScroll: true });
            openOne = close;
        }

        function close(refocus) {
            if (menu.hidden) return;
            menu.hidden = true;
            button.setAttribute('aria-expanded', 'false');
            if (openOne === close) openOne = null;
            if (refocus === true) button.focus({ preventScroll: true });
        }

        function choose(index) {
            const item = items[index];
            if (!item || item.option.disabled) return;
            const changed = !item.option.selected;
            item.option.selected = true;
            sync();
            close(true);
            if (changed) {
                select.dispatchEvent(new Event('input', { bubbles: true }));
                select.dispatchEvent(new Event('change', { bubbles: true }));
            }
        }

        function step(by) {
            let next = active;
            for (let tries = 0; tries < items.length; tries++) {
                next = Math.min(items.length - 1, Math.max(0, next + by));
                if (!items[next].option.disabled) break;
            }
            setActive(next);
        }

        button.addEventListener('click', function() { if (menu.hidden) open(); else close(true); });
        button.addEventListener('keydown', function(event) {
            if (event.key === 'ArrowDown' || event.key === 'ArrowUp') { event.preventDefault(); open(); }
        });
        menu.addEventListener('keydown', function(event) {
            if (event.key === 'ArrowDown') { event.preventDefault(); step(1); }
            else if (event.key === 'ArrowUp') { event.preventDefault(); step(-1); }
            else if (event.key === 'Home') { event.preventDefault(); setActive(0); }
            else if (event.key === 'End') { event.preventDefault(); setActive(items.length - 1); }
            else if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); choose(active); }
            else if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); close(true); }
            else if (event.key === 'Tab') { close(true); }
            else if (event.key.length === 1) {
                // Typing a name jumps to it.
                const now = Date.now();
                typed = now - typedAt > 700 ? event.key : typed + event.key;
                typedAt = now;
                const found = items.findIndex(function(item) { return item.option.textContent.trim().toLowerCase().indexOf(typed.toLowerCase()) === 0; });
                if (found >= 0) setActive(found);
            }
        });
        menu.addEventListener('focusout', function(event) {
            if (!wrap.contains(event.relatedTarget)) close();
        });
        select.addEventListener('change', sync);
        select.addEventListener('focus', function() { button.focus(); });

        // A script that sets .value or .selectedIndex is shown at once.
        ['value', 'selectedIndex'].forEach(function(name) {
            const own = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, name);
            Object.defineProperty(select, name, {
                configurable: true,
                get: function() { return own.get.call(select); },
                set: function(next) { own.set.call(select, next); sync(); },
            });
        });
        // Options that are added, removed or relabelled, and a field that is
        // shown, hidden or disabled, are followed.
        new MutationObserver(function(changes) {
            if (changes.some(function(change) { return change.type !== 'attributes' || change.target !== select; })) build();
            else sync();
        }).observe(select, { childList: true, subtree: true, characterData: true, attributes: true, attributeFilter: ['style', 'hidden', 'disabled', 'label'] });

        build();
        const api = { sync: sync, rebuild: build, close: close };
        enhanced.set(select, api);
        return api;
    }

    document.addEventListener('pointerdown', function(event) {
        if (openOne && !event.target.closest('.ms')) openOne();
    });
    window.addEventListener('resize', function() { if (openOne) openOne(); });
    // A menu is pinned to the window, so scrolling the page behind it closes it.
    document.addEventListener('scroll', function(event) {
        if (openOne && !(event.target.closest && event.target.closest('.ms-menu'))) openOne();
    }, true);

    // Every select on the page, and any added later, gets the same menu.
    // A select marked data-native keeps the browser's.
    function everywhere() {
        const all = function(root) {
            if (root.tagName === 'SELECT') enhance(root);
            else if (root.querySelectorAll) root.querySelectorAll('select').forEach(enhance);
        };
        all(document);
        new MutationObserver(function(changes) {
            changes.forEach(function(change) { change.addedNodes.forEach(function(node) { if (node.nodeType === 1) all(node); }); });
        }).observe(document.body, { childList: true, subtree: true });
    }
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', everywhere);
    else everywhere();

    return {
        enhance: enhance,
        enhanceAll: function(container) {
            return Array.from((container || document).querySelectorAll('select')).map(enhance);
        },
    };
})();

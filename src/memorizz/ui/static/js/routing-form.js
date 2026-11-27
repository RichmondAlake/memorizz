// The Routing fields for a model on OpenRouter (the routing_fields macro):
// filled from a saved routing, read back as one, and shown only for that provider.
window.MemorizzRouting = (function() {
    const LISTS = ['fallback_models', 'providers', 'only', 'ignore'];

    function names(text) {
        return String(text || '').split(/[\n,]+/).map(function(part) { return part.trim(); }).filter(Boolean);
    }

    function read(box) {
        const routing = {};
        box.querySelectorAll('[data-routing-field]').forEach(function(field) {
            const name = field.dataset.routingField;
            if (field.type === 'checkbox') {
                if (!field.checked) return;
                const on = field.dataset.routingOn;
                routing[name] = on === 'true' ? true : on === 'false' ? false : on;
            } else if (name.indexOf('max_price.') === 0) {
                const price = parseFloat(field.value);
                if (!(price >= 0)) return;
                (routing.max_price = routing.max_price || {})[name.slice(10)] = price;
            } else if (LISTS.indexOf(name) >= 0) {
                const list = names(field.value);
                if (list.length) routing[name] = list;
            } else if (field.value) {
                routing[name] = field.value;
            }
        });
        return routing;
    }

    function fill(box, routing) {
        const saved = routing && typeof routing === 'object' ? routing : {};
        box.querySelectorAll('[data-routing-field]').forEach(function(field) {
            const name = field.dataset.routingField;
            if (field.type === 'checkbox') {
                field.checked = name in saved && String(saved[name]) === field.dataset.routingOn;
            } else if (name.indexOf('max_price.') === 0) {
                const price = (saved.max_price || {})[name.slice(10)];
                field.value = price == null ? '' : price;
            } else if (LISTS.indexOf(name) >= 0) {
                field.value = (saved[name] || []).join(field.tagName === 'TEXTAREA' ? '\n' : ', ');
            } else {
                field.value = saved[name] || '';
            }
        });
    }

    // Returns { read, show } for one set of fields; provider is the <select>
    // that names the model's provider.
    function attach(box, provider, routing) {
        if (!box) return null;
        const json = box.querySelector('[data-routing-json]');
        const sync = function() { if (json) json.value = JSON.stringify(read(box)); };
        const show = function() { box.hidden = !provider || String(provider.value).toLowerCase() !== 'openrouter'; };
        fill(box, routing);
        sync();
        show();
        box.addEventListener('input', sync);
        box.addEventListener('change', sync);
        if (provider) provider.addEventListener('change', show);
        return { read: function() { return read(box); }, show: show };
    }

    return { attach: attach, read: read, fill: fill };
})();

// Timers above the chat box: live countdowns the agent or the CLI started.
(function() {
    'use strict';
    const dock = document.getElementById('pg-timers');
    if (!dock) return;

    const POLL_MS = 5000;
    const cards = new Map();
    const pageTitle = document.title;
    let refreshing = false;

    function format(seconds) {
        const whole = Math.max(0, Math.ceil(seconds));
        const hours = Math.floor(whole / 3600);
        const minutes = Math.floor((whole % 3600) / 60);
        const secs = String(whole % 60).padStart(2, '0');
        return hours ? `${hours}:${String(minutes).padStart(2, '0')}:${secs}` : `${minutes}:${secs}`;
    }

    function chime() {
        // Browsers can refuse sound before the page is touched; the card still changes.
        try {
            const audio = new (window.AudioContext || window.webkitAudioContext)();
            [0, 0.22].forEach(function(delay) {
                const tone = audio.createOscillator();
                const gain = audio.createGain();
                tone.frequency.value = 880;
                gain.gain.setValueAtTime(0.0001, audio.currentTime + delay);
                gain.gain.exponentialRampToValueAtTime(0.12, audio.currentTime + delay + 0.02);
                gain.gain.exponentialRampToValueAtTime(0.0001, audio.currentTime + delay + 0.18);
                tone.connect(gain).connect(audio.destination);
                tone.start(audio.currentTime + delay);
                tone.stop(audio.currentTime + delay + 0.2);
            });
        } catch (_) { /* sound is optional */ }
    }

    function act(timer, action) {
        return fetch(`/api/timers/${encodeURIComponent(timer.timer_id)}/${action}`, {method: 'POST'})
            .finally(function() {
                remove(timer.timer_id);
                refresh();
            });
    }

    function remove(id) {
        const card = cards.get(id);
        if (card) card.root.remove();
        cards.delete(id);
        dock.hidden = cards.size === 0;
    }

    function build(timer) {
        const root = document.createElement('div');
        root.className = 'pg-timer';
        root.dataset.timerId = timer.timer_id;

        const label = document.createElement('span');
        label.className = 'pg-timer-label';
        label.textContent = timer.label;

        const clock = document.createElement('span');
        clock.className = 'pg-timer-clock';
        clock.setAttribute('role', 'timer');
        clock.setAttribute('aria-label', `${timer.label} time left`);

        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'btn btn-sm';

        const track = document.createElement('div');
        track.className = 'pg-timer-track';
        track.setAttribute('aria-hidden', 'true');
        const fill = document.createElement('div');
        fill.className = 'pg-timer-fill';
        track.appendChild(fill);

        root.append(label, clock, button, track);
        dock.appendChild(root);
        const card = {root, clock, button, fill, timer, finished: false};
        cards.set(timer.timer_id, card);
        button.addEventListener('click', function() {
            button.disabled = true;
            act(card.timer, card.finished ? 'dismiss' : 'cancel');
        });
        return card;
    }

    function paint(card, announce) {
        const left = card.timer.ends_at - Date.now() / 1000;
        const finished = left <= 0;
        if (finished && !card.finished) {
            card.finished = true;
            card.root.classList.add('is-finished');
            card.clock.removeAttribute('role');
            card.clock.setAttribute('role', 'status');
            if (announce) {
                chime();
                if (document.hidden) document.title = `${card.timer.label} finished`;
            }
        }
        card.clock.textContent = finished ? 'Finished' : format(left);
        card.button.textContent = finished ? 'Dismiss' : 'Cancel';
        const done = finished ? 1 : 1 - left / card.timer.seconds;
        card.fill.style.width = `${Math.min(100, Math.max(0, done * 100))}%`;
    }

    async function refresh() {
        if (refreshing) return;
        refreshing = true;
        try {
            const response = await fetch('/api/timers', {cache: 'no-store'});
            if (!response.ok) return;
            const {timers} = await response.json();
            const seen = new Set();
            for (const timer of timers || []) {
                seen.add(timer.timer_id);
                const known = cards.get(timer.timer_id);
                const card = known || build(timer);
                card.timer = timer;
                // One that ended while the page was closed is shown, not rung.
                paint(card, false);
            }
            for (const id of [...cards.keys()]) {
                if (!seen.has(id)) remove(id);
            }
            dock.hidden = cards.size === 0;
        } catch (_) {
            // The next poll tries again.
        } finally {
            refreshing = false;
        }
    }

    window.setInterval(function() {
        cards.forEach(function(card) { paint(card, true); });
    }, 250);
    window.setInterval(refresh, POLL_MS);
    document.addEventListener('visibilitychange', function() {
        if (!document.hidden) document.title = pageTitle;
    });
    window.pgTimers = {refresh};
    refresh();
})();

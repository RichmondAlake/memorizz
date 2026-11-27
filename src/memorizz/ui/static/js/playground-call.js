// A live, spoken conversation with the agent in the playground, on LiveKit.
// LiveKit hears you, decides when you have finished, and plays the answer;
// speaking again cuts the answer short. Each thing said is a turn of the
// open conversation, so the chat shows it once the call ends.
(function() {
    const button = document.getElementById('pg-call-btn');
    const panel = document.getElementById('pg-call-panel');
    if (!button || !panel || typeof AGENT_ID === 'undefined' || !AGENT_ID) return;
    const lines = document.getElementById('pg-call-lines');
    const stateEl = document.getElementById('pg-call-state');
    const CLIENT = 'https://cdn.jsdelivr.net/npm/livekit-client@2.24.0/dist/livekit-client.umd.min.js';
    let call = null;   // {room, sessionId, memoryId, players}
    let starting = false;

    function notify(message) {
        if (window.MemorizzDialogs) window.MemorizzDialogs.notify(message);
    }

    function loadScript(src) {
        return new Promise(function(resolve, reject) {
            const script = document.createElement('script');
            script.src = src;
            script.onload = resolve;
            script.onerror = function() { reject(new Error('The call library could not be loaded')); };
            document.head.appendChild(script);
        });
    }

    function setState(text) { stateEl.textContent = text; }

    // One line per thing said; a line still being heard is updated in place.
    function showLine(id, who, text, final) {
        let line = lines.querySelector('[data-segment="' + id + '"]');
        if (!line) {
            line = document.createElement('p');
            line.dataset.segment = id;
            line.className = 'pg-call-line' + (who === 'You' ? ' is-you' : '');
            const name = document.createElement('strong');
            name.textContent = who;
            line.append(name, document.createElement('span'));
            lines.appendChild(line);
        }
        line.lastChild.textContent = text;
        line.classList.toggle('is-interim', !final);
        lines.scrollTop = lines.scrollHeight;
    }

    function sync() {
        const live = Boolean(call);
        button.classList.toggle('is-live', live);
        button.setAttribute('aria-pressed', String(live));
        button.title = live ? 'End the live conversation' : button.getAttribute('aria-label');
        panel.hidden = !live && !starting;
    }

    async function start() {
        if (starting || call) return;
        starting = true;
        lines.replaceChildren();
        setState('Connecting…');
        sync();
        try {
            const response = await fetch('/api/agents/' + encodeURIComponent(AGENT_ID) + '/voice/session', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ memory_id: typeof pgActiveMemoryId === 'function' ? pgActiveMemoryId() : '' }),
            });
            const details = await response.json().catch(function() { return {}; });
            if (!response.ok) throw new Error(details.detail || ('HTTP ' + response.status));
            if (details.provider !== 'livekit') throw new Error('A live conversation here needs LiveKit.');
            if (!window.LivekitClient) await loadScript(CLIENT);
            const lk = window.LivekitClient;
            const room = new lk.Room();
            const players = [];
            room.on(lk.RoomEvent.TrackSubscribed, function(track) {
                if (track.kind !== lk.Track.Kind.Audio) return;
                const player = track.attach();
                players.push(player);
                document.body.appendChild(player);
            });
            room.on(lk.RoomEvent.TranscriptionReceived, function(segments, participant) {
                const mine = participant && participant.identity === room.localParticipant.identity;
                segments.forEach(function(segment) {
                    if (segment.text) showLine(segment.id, mine ? 'You' : (participant && participant.name) || 'Agent', segment.text, segment.final);
                });
            });
            room.on(lk.RoomEvent.ActiveSpeakersChanged, function(speakers) {
                const names = speakers.map(function(speaker) { return speaker.identity === room.localParticipant.identity ? 'you' : 'agent'; });
                setState(names.indexOf('agent') >= 0 ? 'Speaking' : names.indexOf('you') >= 0 ? 'Listening' : 'Listening · just talk');
            });
            room.on(lk.RoomEvent.Disconnected, function() { if (call) end(true); });
            await room.connect(details.server_url, details.participant_token);
            await room.localParticipant.setMicrophoneEnabled(true);
            await room.startAudio();
            call = { room: room, sessionId: details.session_id, memoryId: details.memory_id, players: players };
            setState('Listening · just talk');
        } catch (error) {
            notify('The live conversation could not start: ' + (error.message || error));
        } finally {
            starting = false;
            sync();
        }
    }

    function end(remoteEnded) {
        const ended = call;
        if (!ended) return;
        call = null;
        ended.players.forEach(function(player) { player.remove(); });
        if (!remoteEnded) { try { ended.room.disconnect(); } catch (error) { /* already gone */ } }
        fetch('/api/agents/' + encodeURIComponent(AGENT_ID) + '/voice/session/' + encodeURIComponent(ended.sessionId), { method: 'DELETE' }).catch(function() {});
        sync();
        // The spoken turns are now part of the conversation: show them as chat.
        const spoke = lines.querySelector('.pg-call-line');
        if (spoke && ended.memoryId && typeof useThreadMemoryId === 'function') {
            window.setTimeout(function() {
                const known = Array.from(document.querySelectorAll('.playground-thread-tab')).some(function(tab) { return tab.dataset.memoryId === ended.memoryId; });
                useThreadMemoryId(ended.memoryId, !known);
            }, 600);
        }
    }

    button.addEventListener('click', function() { if (call) end(false); else start(); });
    document.getElementById('pg-call-end').addEventListener('click', function() { end(false); });
    window.addEventListener('pagehide', function() { end(false); });
})();

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('src/memorizz/ui/static/js/app.js', 'utf8');
const parser = source.slice(source.indexOf('async function readSseStream('), source.indexOf('\nif (document.readyState', source.indexOf('async function readSseStream(')));
const context = { TextDecoder };
vm.createContext(context);
vm.runInContext(parser, context);
const wire = ': heartbeat\r\nid: first\r\nevent: answer.delta\r\ndata: {"type":"answer.delta",\r\ndata: "delta":"世界 😀"}\r\n\r\nid: last\nevent: run.done\ndata: {"type":"run.done"}\n\n';
async function parse(bytes, width) {
    let position = 0, released = false, cancelled = false;
    const response = {body: {getReader() { return {
        async read() {
            if (position >= bytes.length) return {done: true};
            const value = bytes.subarray(position, position + width);
            position += width;
            return {done: false, value};
        },
        async cancel() { cancelled = true; },
        releaseLock() { released = true; },
    }; }}};
    const events = [];
    try {
        await context.readSseStream(response, (payload, frame) => events.push([JSON.parse(payload), frame]));
        return events;
    } finally {
        assert.ok(released && cancelled);
    }
}
(async () => {
    for (const width of [1, 2, 3, 7, 64, 10000]) {
        const events = await parse(Buffer.from(wire), width);
        assert.equal(events.length, 2);
        assert.equal(events[0][0].delta, '世界 😀');
        assert.equal(events[0][1].event, 'answer.delta');
        assert.equal(events[0][1].id, 'first');
        assert.equal(events[1][1].id, 'last');
    }
    await assert.rejects(parse(Buffer.from('data: {"type":"answer.delta"}\n\n'), 1), /interrupted/);
    await assert.rejects(parse(Buffer.from('data: {"type":"run.done"}'), 1), /interrupted/);
    await assert.rejects(parse(Buffer.from('data: {"type":"run.done"}\n\ndata: {}\n\n'), 1), /after/);
    await assert.rejects(parse(Buffer.from('data: ' + 'x'.repeat(140000)), 4096), /size limit/);
    await assert.rejects(parse(Buffer.from([0xff, 0xfe]), 1));
    console.log('SSE parser: six fragment widths and five interruption/size/encoding checks passed');
})().catch(error => { console.error(error); process.exitCode = 1; });

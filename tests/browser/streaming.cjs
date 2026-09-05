// Real rendered Playground against an isolated barrier-controlled provider.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawn } = require('node:child_process');
const { chromium } = require(process.env.MEMORIZZ_PLAYWRIGHT_MODULE || 'playwright');
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'memorizz-stream-browser-'));
const port = Number(process.env.MEMORIZZ_STREAM_BROWSER_PORT || 8794);
const python = process.env.MEMORIZZ_TEST_PYTHON || '.venv/bin/python';
const server = spawn(python, ['tests/integration/streaming_fixture.py', 'ui', String(port)], {
    env: {...process.env, PYTHONPATH: 'src', MEMORIZZ_STREAM_FIXTURE_DIR: root},
    stdio: ['ignore', 'ignore', 'pipe'],
});
let serverErrors = '';
server.stderr.on('data', data => { serverErrors += data; });
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(check, description) {
    const deadline = Date.now() + 15000;
    while (!(await check())) {
        if (Date.now() > deadline) throw Error(description + '\n' + serverErrors);
        await pause(25);
    }
}
(async () => {
    let browser;
    try {
        await until(async () => {
            try { return (await fetch('http://127.0.0.1:' + port + '/health')).status < 500; }
            catch (_) { return false; }
        }, 'Fixture startup');
        browser = await chromium.launch({headless: true, executablePath: process.env.MEMORIZZ_CHROMIUM || undefined});
        const page = await browser.newPage({viewport: {width: 1280, height: 900}, extraHTTPHeaders: {Authorization: 'Bearer stream-fixture-token'}});
        const errors = [];
        page.on('pageerror', error => errors.push(String(error)));
        await page.goto('http://127.0.0.1:' + port + '/agents/stream-fixture/playground');
        await page.locator('#pg-query').fill('hello');
        await page.evaluate(() => { void sendMessage(); });
        await until(async () => (await page.locator('.pg-stream-status').count()) > 0
            && await page.locator('.pg-message-text').last().textContent() === 'Hello ', 'First visible delta');
        assert.ok(!fs.existsSync(path.join(root, 'provider_complete')));
        await page.screenshot({path: path.join(root, 'first-delta.png')});
        fs.writeFileSync(path.join(root, 'release'), '');
        await until(() => fs.existsSync(path.join(root, 'provider_complete')), 'Provider completion');
        await until(async () => (await page.locator('.pg-message-text').last().textContent()).includes('世界'), 'Second visible delta');
        await until(async () => await page.locator('.pg-stream-status').last().textContent() === '', 'Terminal completion');
        fs.unlinkSync(path.join(root, 'release'));
        fs.unlinkSync(path.join(root, 'provider_complete'));
        fs.unlinkSync(path.join(root, 'provider_closed'));
        await page.locator('#pg-query').fill('stop test');
        await page.evaluate(() => { void sendMessage(); });
        await until(async () => await page.locator('.pg-message-text').last().textContent() === 'Hello ', 'Second run first paint');
        await page.evaluate(() => currentAbortController.abort());
        await until(() => fs.existsSync(path.join(root, 'provider_closed')), 'Provider cancellation');
        assert.ok((await page.locator('.pg-message-text').last().textContent()).includes('Hello'));
        assert.ok((await page.locator('.pg-stream-status').last().textContent()).includes('Partial answer retained'));
        await page.screenshot({path: path.join(root, 'stopped-partial.png')});
        assert.deepEqual(errors, []);
        console.log(JSON.stringify({passed: true, root, checks: ['visible-before-provider-completion', 'two-visible-updates', 'stop-preserves-partial', 'provider-cancelled', 'no-page-errors']}));
    } finally {
        if (browser) await browser.close();
        server.kill('SIGTERM');
    }
})().catch(error => { console.error(error); process.exitCode = 1; });

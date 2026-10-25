/* Verify real clipboard contents against complete stored activity, beyond 10k events. */
const assert = require('node:assert/strict');
const {chromium} = require(process.env.MEMORIZZ_PLAYWRIGHT_MODULE || 'playwright');
(async () => {
    const browser = await chromium.launch({headless: true, ...(process.env.MEMORIZZ_BROWSER_EXECUTABLE ? {executablePath: process.env.MEMORIZZ_BROWSER_EXECUTABLE} : {})});
    try {
        const base = process.env.MEMORIZZ_BROWSER_TEST_URL;
        const page = await browser.newPage({viewport: {width: 1440, height: 1000}, permissions: ['clipboard-read', 'clipboard-write'], extraHTTPHeaders: {Authorization: 'Bearer memorizz-browser-fixture-token'}});
        const errors = [];
        page.on('pageerror', error => errors.push(error.message));
        await page.route('https://cdn.jsdelivr.net/**', route => route.abort());
        const activity = await (await page.request.get(base + '/api/harness-runs/evidence-fixture/activity')).json();
        assert(activity.run.events.length > 10_000);
        await page.goto(base + '/harnesses', {waitUntil: 'domcontentloaded'});
        assert.equal(await page.locator('#harness-run-dialog').count(), 0);
        await page.locator('#hx-table tr[data-run-id="evidence-fixture"]').click();
        await page.locator('#hx-detail-evidence-fixture .harness-inspect').click();
        await page.waitForURL('**/harnesses/runs/evidence-fixture');
        await page.waitForFunction(count => document.getElementById('evidence-load-status').textContent.includes(count + ' recorded events'), activity.run.events.length);
        assert((await page.locator('#evidence-steps').textContent()).includes('LAST-EVENT-SENTINEL'));
        assert((await page.locator('#evidence-steps').textContent()).includes('OUTPUT-END'));
        assert((await page.locator('#evidence-steps').textContent()).includes('COMMAND-END'));
        // Details are collapsed: copying must include their full data anyway.
        assert.equal(await page.locator('#evidence-steps details[open]').count(), 0);
        await page.getByRole('button', {name: 'Copy full activity', exact: true}).click();
        await page.waitForFunction(() => document.getElementById('evidence-copy-status').textContent.startsWith('Copied all '));
        const copied = await page.evaluate(() => navigator.clipboard.readText());
        assert.deepEqual(JSON.parse(copied), activity);
        assert(copied.includes('INPUT-END') && copied.includes('OUTPUT-END') && copied.includes('DIFF-END'));
        assert(copied.includes('delegate-fixture-ACTIVITY') && copied.includes('nested-fixture-ACTIVITY'));
        await page.getByRole('tab', {name: 'Raw record', exact: true}).click();
        await page.waitForFunction(() => document.getElementById('evidence-json').textContent.includes('LAST-EVENT-SENTINEL'));
        assert.deepEqual(JSON.parse(await page.locator('#evidence-json').textContent()), activity);
        await page.reload({waitUntil: 'domcontentloaded'});
        assert(await page.locator('#evidence-record').isVisible());
        await page.getByRole('tab', {name: 'Memory evolution', exact: true}).click();
        await page.locator('#evidence-memory').waitFor({state: 'visible'});
        await page.getByRole('tab', {name: 'Trajectory', exact: true}).click();
        await page.getByRole('tab', {name: 'Trajectory', exact: true}).focus();
        await page.keyboard.press('ArrowRight');
        await page.waitForFunction(() => document.getElementById('tab-record').getAttribute('aria-selected') === 'true');
        await page.getByRole('tab', {name: 'Trajectory', exact: true}).click();
        // A clipboard denial keeps the entire export selected for manual copy.
        await page.evaluate(() => {
            navigator.clipboard.writeText = async () => { throw new Error('Denied'); };
            document.execCommand = () => false;
        });
        await page.getByRole('button', {name: 'Copy full activity', exact: true}).click();
        await page.waitForFunction(() => !document.getElementById('evidence-copy-fallback').hidden);
        assert.deepEqual(JSON.parse(await page.locator('#evidence-copy-fallback').inputValue()), activity);
        assert.equal(await page.locator('#evidence-copy-fallback').evaluate(el => el.selectionEnd - el.selectionStart), copied.length);
        await page.route('**/api/harness-runs/evidence-fixture/activity', route => route.fulfill({status: 503, body: '{}'}));
        await page.getByRole('button', {name: 'Copy full activity', exact: true}).click();
        await page.waitForFunction(() => document.getElementById('evidence-copy-status').textContent.startsWith('Copy failed.'));
        await page.setViewportSize({width: 390, height: 844});
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
        if (process.env.MEMORIZZ_EVIDENCE_SCREENSHOT) await page.screenshot({path: process.env.MEMORIZZ_EVIDENCE_SCREENSHOT});
        // The back link reaches the ledger (never a ?run= link that bounces an
        // older run back here) with the run still selected. A ?run= link the
        // ledger cannot show opens evidence, and Back does not bounce again.
        await page.setViewportSize({width: 1440, height: 1000});
        await page.locator('.page-back').click();
        await page.waitForURL(url => url.pathname === '/harnesses' && !url.search);
        await page.locator('#hx-table tr[data-run-id="evidence-fixture"][aria-current="true"]').waitFor();
        await page.goto(base + '/harnesses?run=not-in-ledger', {waitUntil: 'domcontentloaded'});
        await page.waitForURL('**/harnesses/runs/not-in-ledger');
        await page.goBack({waitUntil: 'domcontentloaded'});
        const back = new URL(page.url());
        assert.equal(back.pathname + back.search, '/harnesses');
        assert.deepEqual(errors, []);
        console.log('Evidence is a full page; clipboard and raw record preserve every event, long payload, nested delegate, and late event. Mobile, deep links, keyboard tabs and copy failures passed.');
    } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

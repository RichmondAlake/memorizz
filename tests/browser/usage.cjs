/* Run against usage_app.py. No provider credentials, CDN or paid calls. */
const assert = require('node:assert/strict');
const {chromium} = require(process.env.MEMORIZZ_PLAYWRIGHT_MODULE || 'playwright');
(async () => {
    const browser = await chromium.launch({headless: true, ...(process.env.MEMORIZZ_BROWSER_EXECUTABLE ? {executablePath: process.env.MEMORIZZ_BROWSER_EXECUTABLE} : {})});
    try {
        const base = process.env.MEMORIZZ_BROWSER_TEST_URL || 'http://127.0.0.1:8781';
        const page = await browser.newPage({viewport: {width: 1440, height: 1000}, extraHTTPHeaders: {Authorization: 'Bearer memorizz-browser-fixture-token'}});
        const errors = [];
        page.on('pageerror', error => errors.push(error.message));
        // The synthetic calls are from September, so open on all recorded usage.
        await page.goto(`${base}/traces/usage?window=all`);
        const figure = index => page.locator('.usage-figures .kpi-value').nth(index).textContent();
        assert.equal(await page.locator('h1').textContent(), 'Usage & cost');
        assert.equal(await figure(0), '$0.001170');
        assert.equal(await figure(1), '6.6k');
        assert.equal(await figure(2), '6');
        assert.equal(await page.locator('.usage-slice').count(), 3);
        assert(!(await page.content()).includes('PRIVATE SYNTHETIC PROMPT'));
        // A bar shows its period's figures and opens that period.
        const bar = page.locator('.usage-over-time a.ov-bar').first();
        await bar.hover();
        assert(await page.locator('.usage-over-time a.ov-bar:hover .ov-tip').isVisible());
        assert((await bar.getAttribute('href')).includes('window=custom'));
        await page.locator('.usage-export > summary').click();
        const exportURL = await page.getByRole('menuitem', {name: 'JSON, everything'}).getAttribute('href');
        const data = await (await page.request.get(base + exportURL)).json();
        assert.equal(data.totals.calls, 6);
        assert.equal(data.memory[0].memory_tokens_estimate, 1800);
        assert.equal(data.coverage.read_complete, true);
        const csvURL = await page.getByRole('menuitem', {name: 'CSV by agent'}).getAttribute('href');
        const csv = await (await page.request.get(base + csvURL)).text();
        assert(csv.startsWith('agent_id,agent_name,calls'), csv.slice(0, 80));
        await Promise.all([page.waitForNavigation(), page.locator('select[name=agent_id]').selectOption('researcher')]);
        assert.equal(await figure(2), '3');
        await page.locator('.usage-runs a[href*="root_trace_id"]').first().click();
        assert.equal(new URL(page.url()).searchParams.get('agent_id'), 'researcher');
        assert(new URL(page.url()).searchParams.get('root_trace_id'));
        await page.locator('#trace-usage > summary').click();
        assert.equal(await page.locator('#trace-usage .usage-kpis strong').first().textContent(), '1');
        await page.locator('#trace-usage-link').click();
        assert.equal(await figure(2), '1');
        await page.locator('.usage-export > summary').click();
        const selected = await page.getByRole('menuitem', {name: 'JSON, everything'}).getAttribute('href');
        assert(new URL(selected, base).searchParams.get('root_trace_id'));
        await page.screenshot({path: process.env.MEMORIZZ_USAGE_SCREENSHOT || '/private/tmp/memorizz-usage-desktop.png', fullPage: true});
        await page.setViewportSize({width: 390, height: 844});
        // Wait for the base layout's 200 ms sidebar transition to settle.
        await page.waitForFunction(() => {
            const box = document.querySelector('.main-content').getBoundingClientRect();
            return box.x < 1 && box.width > 389;
        });
        const layout = await page.locator('.main-content').boundingBox();
        assert(layout.x < 30 && layout.width > 330, JSON.stringify(layout));
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= 390));
        await page.screenshot({path: process.env.MEMORIZZ_USAGE_MOBILE_SCREENSHOT || '/private/tmp/memorizz-usage-mobile.png', fullPage: true});
        await page.goto(`${base}/evalground?run_id=usage-eval`);
        assert.equal(await page.locator('.usage-chart').count(), 3);
        assert((await page.content()).includes('Evaluation latency'));
        assert((await page.content()).includes('$0.002000'));
        assert.deepEqual(errors, []);
        console.log('PASS: usage figures, exact costs, bars, memory charts, JSON and CSV export, agent filter, trace selection, mobile and Evalground');
    } finally {
        await browser.close();
    }
})().catch(error => { console.error(error); process.exitCode = 1; });

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
        await page.goto(`${base}/traces/usage`);
        assert.equal(await page.locator('h1').textContent(), 'Usage & memory analytics');
        assert.equal(await page.locator('.usage-kpis strong').first().textContent(), '6');
        assert.equal(await page.locator('.usage-kpis strong').nth(1).textContent(), '6600');
        assert.equal(await page.locator('.usage-kpis strong').nth(2).textContent(), '$0.001170');
        assert.equal(await page.locator('.usage-slice').count(), 3);
        assert.equal(await page.locator('.usage-line').count(), 2);
        assert(!(await page.content()).includes('PRIVATE SYNTHETIC PROMPT'));
        const exportURL = await page.getByRole('link', {name: 'Export JSON'}).getAttribute('href');
        const response = await page.request.get(base + exportURL);
        const data = await response.json();
        assert.equal(data.totals.calls, 6);
        assert.equal(data.memory[0].memory_tokens_estimate, 1800);
        assert.equal(data.coverage.read_complete, true);
        await page.locator('input[name=agent_id]').fill('researcher');
        await page.getByRole('button', {name: 'Apply filters'}).click();
        assert.equal(await page.locator('.usage-kpis strong').first().textContent(), '3');
        await page.locator('.usage-table a[href*="root_trace_id"]').first().click();
        assert.equal(new URL(page.url()).searchParams.get('agent_id'), 'researcher');
        assert(new URL(page.url()).searchParams.get('root_trace_id'));
        await page.locator('#trace-usage > summary').click();
        assert.equal(await page.locator('#trace-usage .usage-kpis strong').first().textContent(), '1');
        await page.locator('#trace-usage-link').click();
        assert.equal(await page.locator('.usage-kpis strong').first().textContent(), '1');
        assert(new URL(await page.getByRole('link', {name: 'Export JSON'}).getAttribute('href'), base).searchParams.get('root_trace_id'));
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
        console.log('PASS: daily usage, exact costs, memory charts, export, agent filter, trace selection, mobile and Evalground');
    } finally {
        await browser.close();
    }
})().catch(error => { console.error(error); process.exitCode = 1; });

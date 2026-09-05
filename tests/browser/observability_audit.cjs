/* Run against observability_app.py with MEMORIZZ_BROWSER_TEST_HOOKS=off. */
const assert = require('node:assert/strict');
const {chromium} = require(process.env.MEMORIZZ_PLAYWRIGHT_MODULE || 'playwright');
(async () => {
    const browser = await chromium.launch({headless: true, ...(process.env.MEMORIZZ_BROWSER_EXECUTABLE ? {executablePath: process.env.MEMORIZZ_BROWSER_EXECUTABLE} : {})});
    try {
        const page = await browser.newPage({extraHTTPHeaders: {Authorization: 'Bearer memorizz-browser-fixture-token'}, viewport: {width: 1440, height: 1050}});
        const errors = [];
        page.on('pageerror', error => errors.push(error.message));
        const base = process.env.MEMORIZZ_BROWSER_TEST_URL || 'http://127.0.0.1:8779/traces';
        const url = new URL(base);
        url.search = 'agent_id=browser-agent&thread_id=legacy-thread&root_trace_id=legacy-root';
        await page.goto(url.toString());
        assert((await page.locator('#trace-insights').textContent()).includes('All stored events loaded; end-to-end coverage unknown'));
        assert((await page.locator('#trace-insights').textContent()).includes('No recorded evidence'));
        assert.equal(await page.locator('#trace-agent-navigator').getAttribute('open'), null);
        assert(await page.locator('#trace-insights').evaluate(node => node.getBoundingClientRect().top + window.scrollY < 1200));
        assert((await page.locator('#trace-artifact-provenance').textContent()).includes('No source-to-artifact link can be verified'));
        assert((await page.locator('#trace-output-contracts').textContent()).includes('does not establish interactive delivery'));
        assert((await page.locator('#trace-lineage-inspectors').textContent()).includes('Observed supplied items: 33'));
        await page.locator('#trace-capability-banner summary').click();
        assert((await page.locator('#trace-capability-banner').textContent()).includes('Not configured'));
        await page.locator('#trace-capability-banner summary').click();
        assert(await page.locator('#trace-replay-create').isDisabled());
        await page.locator('#incident-finder-panel summary').click();
        assert(await page.locator('#account-email').isDisabled());
        assert(await page.locator('#account-lookup button').isDisabled());
        await page.locator('#incident-finder-panel summary').click();
        await page.locator('#trace-health-link').click();
        assert(new URL(page.url()).searchParams.get('root_trace_id') === 'legacy-root');
        await page.locator('.page-header a').click();
        assert(new URL(page.url()).searchParams.get('root_trace_id') === 'legacy-root');
        await page.setViewportSize({width: 390, height: 844});
        await page.goto(url.toString());
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= 390));
        assert(await page.locator('#trace-insights').evaluate(node => node.getBoundingClientRect().top + window.scrollY < 1400));
        await page.locator('.trace-section-nav a[href="#trace-lineage-inspectors"]').click();
        assert(await page.locator('.trace-section-nav').evaluate(node => Math.abs(node.getBoundingClientRect().top) < 2));
        assert(await page.locator('#trace-lineage-inspectors').isVisible());
        assert(await page.locator('.trace-scroll-hint').isVisible());
        await page.screenshot({path: process.env.MEMORIZZ_AUDIT_SCREENSHOT || '/private/tmp/memorizz-observability-audit-mobile.png'});
        assert.deepEqual(errors, []);
        console.log('PASS: legacy unknown coverage, explicit missing-evidence panels, disabled hooks, scoped round-trip navigation, bounded busy overview and usable mobile incident placement');
    } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

/* Async launch navigation against synthetic harnesses; no external API calls. */
const assert = require('node:assert/strict');
const {chromium} = require(process.env.MEMORIZZ_PLAYWRIGHT_MODULE || 'playwright');
(async () => {
    const browser = await chromium.launch({headless:true, ...(process.env.MEMORIZZ_BROWSER_EXECUTABLE ? {executablePath:process.env.MEMORIZZ_BROWSER_EXECUTABLE} : {})});
    try {
        const base = process.env.MEMORIZZ_BROWSER_TEST_URL || 'http://127.0.0.1:8793';
        const page = await browser.newPage({viewport:{width:1440,height:900}, reducedMotion:'reduce', extraHTTPHeaders:{Authorization:'Bearer memorizz-browser-fixture-token'}});
        const errors=[];
        page.on('pageerror', e=>errors.push(e.message));
        await page.route('https://cdn.jsdelivr.net/**', route=>route.abort());
        await page.addInitScript(() => {
            const scroll = Element.prototype.scrollIntoView;
            window.comparisonScrolls = [];
            Element.prototype.scrollIntoView = function(options) {
                window.comparisonScrolls.push({id:this.id,run:this.dataset.approvalRun,options});
                return scroll.call(this,options);
            };
        });
        await page.goto(base+'/harnesses',{waitUntil:'domcontentloaded'});
        await page.locator('#harness-launch > summary').click();
        await page.locator('[data-mode-choice=compare]').click();
        await page.locator('#harness-task').fill('Compare with approval');
        await page.locator('[name=network]').selectOption('full');
        await page.locator('[name=mcp_access]').selectOption('none');
        await page.locator('[name=compare_harness][value=codex]').check();
        await page.locator('[name=compare_harness][value=claude-code]').check();
        const started = page.waitForResponse(r=>r.url().endsWith('/api/harness-orchestrations') && r.request().method()==='POST');
        await page.locator('#harness-submit').click();
        const first = (await (await started).json()).orchestration;
        await page.waitForFunction(() => window.comparisonScrolls.some(s=>s.run),{},{timeout:15000});
        let detail;
        const deadline = Date.now() + 15000;
        while (true) {
            detail = await (await page.request.get(base+'/api/harness-orchestrations/'+first.orchestration_id)).json();
            if (detail.runs.length === 2) break;
            assert(Date.now() < deadline, 'Both comparison runs were not created: ' + JSON.stringify(detail));
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        const ids = detail.runs.map(run=>run.run_id);
        assert.equal(ids.length,2);
        const scrolls = await page.evaluate(()=>window.comparisonScrolls.filter(s=>s.run));
        assert.equal(scrolls.length,1);
        assert(ids.includes(scrolls[0].run));
        assert.equal(scrolls[0].options.behavior,'auto');
        const input = page.locator('#hx-region-approvals [data-approval-run="'+scrolls[0].run+'"] .harness-approver');
        assert(await input.evaluate(el=>el===document.activeElement));
        assert.equal(await input.getAttribute('placeholder'),'Your name or email');
        const bounds = await input.boundingBox();
        assert(bounds.y>=0 && bounds.y<900,JSON.stringify({bounds,scrolls}));
        assert((await page.locator('#hx-region-approvals').textContent()).includes('record who approves'));
        // Remembering the identity never approves another run automatically.
        await input.fill('Fixture approver');
        const card = input.locator('..');
        await card.locator('[data-decision=reject]').click();
        await page.locator('#hx-region-approvals .harness-approver').first().waitFor();
        assert.equal(await page.locator('#hx-region-approvals .harness-approver').first().inputValue(),'Fixture approver');
        const remaining=(await (await page.request.get(base+'/api/harness-orchestrations/'+first.orchestration_id)).json()).runs;
        assert(remaining.some(run=>run.status==='pending_approval'));
        // A new run with no approval navigates to its own workflow despite
        // the existing approval queue. Live refresh does not keep scrolling.
        if (!(await page.locator('#harness-launch').evaluate(el=>el.open))) await page.locator('#harness-launch > summary').click();
        await page.locator('[data-mode-choice=compare]').click();
        await page.locator('#harness-task').fill('Compare without approval');
        await page.locator('[name=network]').selectOption('none');
        await page.locator('[name=mcp_access]').selectOption('none');
        await page.locator('[name=compare_harness][value=codex]').check();
        await page.locator('[name=compare_harness][value=claude-code]').check();
        const secondStarted = page.waitForResponse(r=>r.url().endsWith('/api/harness-orchestrations') && r.request().method()==='POST');
        await page.locator('#harness-submit').click();
        const second=(await (await secondStarted).json()).orchestration;
        await page.waitForFunction(id=>window.comparisonScrolls.some(s=>s.id==='hx-workflow-'+id),second.orchestration_id,{timeout:15000});
        assert.equal(await page.locator('#hx-workflow-'+second.orchestration_id).getAttribute('data-collapsed'),null);
        const count = await page.evaluate(()=>window.comparisonScrolls.length);
        await page.waitForTimeout(3000);
        assert.equal(await page.evaluate(()=>window.comparisonScrolls.length),count);
        assert.deepEqual(errors,[]);
        console.log('Approval and no-approval comparisons navigate once to their own results; approver identity is remembered.');
    } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});

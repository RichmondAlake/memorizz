/* Acceptance of saved-output and launch judging; optional real local Ollama. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const {chromium} = require(process.env.MEMORIZZ_PLAYWRIGHT_MODULE || 'playwright');
(async () => {
    const browser = await chromium.launch({headless: true, ...(process.env.MEMORIZZ_BROWSER_EXECUTABLE ? {executablePath: process.env.MEMORIZZ_BROWSER_EXECUTABLE} : {})});
    const output = process.env.MEMORIZZ_JUDGE_EVIDENCE || require('node:path').join(require('node:os').tmpdir(), 'memorizz-harness-judge-evidence');
    fs.mkdirSync(output, {recursive: true});
    try {
        const base = process.env.MEMORIZZ_BROWSER_TEST_URL || 'http://127.0.0.1:8792';
        const page = await browser.newPage({viewport: {width: 1440, height: 1000}, extraHTTPHeaders: {Authorization: 'Bearer memorizz-browser-fixture-token'}});
        const errors = [];
        page.on('pageerror', error => errors.push(error.message));
        await page.route('https://cdn.jsdelivr.net/**', route => route.abort());
        await page.goto(base + '/harnesses');
        await page.locator('[data-compare-pick="judge-fixture-correct"]').check();
        await page.locator('[data-compare-pick="judge-fixture-wrong"]').check();
        await page.locator('#hx-compare-picked').click();
        const dialog = page.locator('dialog.hxcmp');
        await dialog.locator('.hxcmp-judge-settings > summary').click();
        await dialog.locator('[data-judge-field=model]').waitFor();
        assert.equal(await dialog.locator('[data-judge-field=provider]').inputValue(), 'ollama');
        assert.equal(await dialog.locator('[data-judge-field=model]').inputValue(), 'qwen2.5:3b');
        await dialog.locator('.hxjudge-reference > summary').click();
        await dialog.locator('[data-judge-field=reference]').fill('The capital of France is Paris. London is the capital of the United Kingdom.');
        const queued = page.waitForResponse(response => response.url().endsWith('/api/harness-judgments') && response.request().method() === 'POST');
        await dialog.locator('[data-judge-start]').click();
        const job = (await (await queued).json()).judgment;
        await dialog.locator('.hxcmp-best').filter({hasText: 'Most accurate (judge)'}).waitFor({timeout: 180000});
        const judged = (await (await page.request.get(base + '/api/harness-judgments/' + job.judgment_id)).json()).judgment;
        assert.equal(judged.status, 'completed', JSON.stringify(judged.results));
        assert(judged.results['judge-fixture-correct'].score > judged.results['judge-fixture-wrong'].score);
        assert.equal(judged.results['judge-fixture-correct'].cost_usd, 0);
        assert(await dialog.getByText('Cheapest meeting target', {exact:true}).count());
        fs.writeFileSync(output + '/local-judgment.json', JSON.stringify(judged, null, 2));
        await page.screenshot({path:output + '/comparison-judge-desktop.png', fullPage:true});
        const savedSettings = page.waitForResponse(response => response.url().endsWith('/api/harness-judge/settings') && response.request().method() === 'PUT');
        await dialog.locator('[data-judge-save]').click();
        assert.equal((await (await savedSettings).json()).config.reference, judged.config.reference);
        // Inspectability and persistence after reopening.
        await dialog.locator('[data-hxcmp-close]').click();
        await page.reload();
        await page.locator('tr[data-run-id="judge-fixture-correct"] .hx-col-accuracy').filter({hasText:'/100'}).waitFor();
        await page.locator('tr[data-run-id="judge-fixture-correct"]').click();
        await page.locator('[data-judge-run="judge-fixture-correct"]').click();
        await dialog.locator('.hxjudge-result').waitFor();
        assert((await dialog.locator('.hxjudge-result').textContent()).includes('qwen2.5:3b'));
        await page.setViewportSize({width:390, height:844});
        await page.screenshot({path:output + '/judge-mobile.png', fullPage:true});
        assert(await dialog.locator('[data-judge-field=prompt]').isVisible());
        await dialog.locator('[data-hxcmp-close]').click();
        await page.setViewportSize({width:1440, height:1000});
        // Launch a comparison with auto-judging; these harnesses are fixtures.
        await page.locator('#harness-launch > summary').click();
        await page.locator('[data-mode-choice=compare]').click();
        await page.locator('#harness-task').fill('What is the capital of France?');
        await page.locator('[name=network]').selectOption('none');
        await page.locator('[name=mcp_access]').selectOption('none');
        await page.locator('[data-judge-auto]').check();
        const launchJudge = page.locator('#hx-launch-judge');
        await launchJudge.locator('.hxjudge-reference > summary').click();
        await launchJudge.locator('[data-judge-field=reference]').fill('The capital is Paris.');
        await launchJudge.locator('[data-judge-field=prompt]').fill('Score whether the answer correctly names Paris as the capital of France. Give 100 for Paris and 0 for a different city, with a short explanation.');
        await launchJudge.locator('[data-judge-field=model]').fill('qwen2.5:7b');
        const started = page.waitForResponse(response => response.url().endsWith('/api/harness-orchestrations') && response.request().method() === 'POST');
        await page.locator('#harness-submit').click();
        const workflow = (await (await started).json()).orchestration;
        assert(workflow, 'Workflow launched');
        let done;
        const deadline = Date.now() + 180000;
        while (true) {
            done = await (await page.request.get(base + '/api/harness-orchestrations/' + workflow.orchestration_id)).json();
            if (done.orchestration.status === 'succeeded' && done.runs.length === 2 && done.runs.every(run => run.judgment?.status === 'completed')) break;
            assert(Date.now() < deadline, 'Automatic evaluation timed out: ' + JSON.stringify(done));
            await new Promise(resolve => setTimeout(resolve, 300));
        }
        assert.equal(done.runs.length, 2);
        assert(done.runs.every(run => run.result.cost_usd === 0.01));
        assert(done.runs.every(run => run.judgment.config.prompt.includes('correctly names Paris')));
        assert(done.runs.every(run => run.judgment.config.model === 'qwen2.5:7b' && run.judgment.response_model === 'qwen2.5:7b'));
        fs.writeFileSync(output + '/auto-judged-comparison.json', JSON.stringify(done, null, 2));
        await page.locator('[data-edit-workflow="' + workflow.orchestration_id + '"]').click();
        assert(await page.locator('[data-judge-auto]').isChecked());
        assert.equal(await launchJudge.locator('[data-judge-field=model]').inputValue(), 'qwen2.5:7b');
        assert((await launchJudge.locator('[data-judge-field=prompt]').inputValue()).includes('correctly names Paris'));
        assert.deepEqual(errors, []);
        console.log('Saved-output and automatic local judging passed. Evidence: ' + output);
    } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exitCode = 1;});

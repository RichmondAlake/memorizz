/* Explicit opt-in: incurs small API charges using keys already in the UI server. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const { chromium } = require(
  process.env.MEMORIZZ_PLAYWRIGHT_MODULE || "playwright",
);
(async () => {
  if (process.env.MEMORIZZ_LIVE_HOSTED_EVAL !== "1")
    throw Error(
      "Set MEMORIZZ_LIVE_HOSTED_EVAL=1 to authorize this paid acceptance run.",
    );
  const base = process.env.MEMORIZZ_BROWSER_TEST_URL || "http://localhost:8765";
  const out =
    process.env.MEMORIZZ_COMPARISON_EVIDENCE ||
    "/private/tmp/memorizz-hosted-evidence";
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({
        viewport: { width: 1500, height: 1100 },
      }),
      errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(base + "/evalground/compare");
    await page
      .locator("#experiment-name")
      .fill("OpenAI vs Anthropic · six strict memory checks");
    await page.locator("#dataset").selectOption("memory_checks");
    await page.locator("#limit").fill("6");
    await page.click("#setup-next");
    const rows = page.locator("#readers .model-row");
    for (const [i, provider, model, options] of [
      [0, "openai", "gpt-4.1-mini-2025-04-14", { max_completion_tokens: 256 }],
      [1, "anthropic", "claude-haiku-4-5-20251001", { max_tokens: 256 }],
    ]) {
      const row = rows.nth(i);
      await row.locator("[data-key=provider]").selectOption(provider);
      await row
        .locator(`[data-key=model] option[value="${model}"]`)
        .waitFor({ state: "attached", timeout: 30000 });
      assert((await row.locator("[data-key=model] option").count()) > 2);
      await row.locator("[data-key=model]").selectOption(model);
      await row.locator("summary").click();
      await row.locator("[data-key=options]").fill(JSON.stringify(options));
      await row.locator("[data-key=stream]").check();
      const catalog = await (
        await page.request.get(
          base + "/evalground/model-catalog?provider=" + provider,
        )
      ).json();
      assert.equal(catalog.source, "provider_api");
      fs.writeFileSync(
        out + "/" + provider + "-model-catalog.json",
        JSON.stringify(catalog, null, 2),
      );
    }
    await page.click("#setup-next");
    await page.locator("#max-cost").fill("0.25");
    await page.screenshot({
      path: out + "/model-menus-desktop.png",
      fullPage: true,
    });
    await page.locator("#start-comparison").click();
    await page.waitForFunction(
      () =>
        [
          "completed",
          "completed_with_errors",
          "failed",
          "spend_limit",
          "unknown_cost",
        ].includes(document.getElementById("result-status").textContent),
      {},
      { timeout: 300000 },
    );
    const id = new URL(page.url()).searchParams.get("experiment");
    const result = await (
      await page.request.get(`${base}/evalground/comparisons/${id}/export`)
    ).json();
    fs.writeFileSync(
      out + "/hosted-six-questions.json",
      JSON.stringify(result, null, 2),
    );
    assert.equal(
      result.status,
      "completed",
      JSON.stringify(result.runs.map((r) => r.error)),
    );
    assert.equal(result.runs.length, 2);
    assert.equal(result.case_ids.length, 6);
    const quantile = (xs, p) => {
      const a = [...xs].sort((x, y) => x - y),
        v = (a.length - 1) * p,
        lo = Math.floor(v);
      return a[lo] + (a[Math.min(lo + 1, a.length - 1)] - a[lo]) * (v - lo);
    };
    const near = (a, b) => assert(Math.abs(a - b) < 1e-10, `${a} != ${b}`);
    for (const run of result.runs) {
      assert.equal(run.cases.length, 6);
      assert.equal(run.summary.unpriced_calls, 0);
      assert(run.summary.total_cost_usd > 0);
      let spend = 0;
      for (const c of run.calls) {
        assert.equal(c.status, "completed");
        assert(c.response_metadata.response_model);
        const u = c.usage,
          p = c.pricing,
          w = u.cache_write_tokens || 0,
          h = u.cache_write_1h_tokens || 0,
          cache = u.cached_tokens || 0;
        const expected =
          ((u.prompt_tokens - cache - w) * p.input +
            cache * p.cached_input +
            (w - h) * (p.cache_write || 0) +
            h * (p.cache_write_1h || 0) +
            u.completion_tokens * p.output) /
          1e6;
        near(c.cost_usd, expected);
        spend += expected;
      }
      near(run.summary.total_cost_usd, spend);
      near(run.summary.cost_per_question_usd, spend / 6);
      near(
        run.summary.prompt_tokens,
        run.calls.reduce((n, c) => n + c.usage.prompt_tokens, 0),
      );
      near(
        run.summary.completion_tokens,
        run.calls.reduce((n, c) => n + c.usage.completion_tokens, 0),
      );
      const lat = run.cases.map(
        (c) => c.retrieval_seconds + c.generation_seconds,
      );
      near(run.summary.latency_p50_seconds, quantile(lat, 0.5));
      near(run.summary.latency_p95_seconds, quantile(lat, 0.95));
      near(run.summary.accuracy, run.cases.filter((c) => c.correct).length / 6);
      assert(run.summary.accuracy_ci95.lower < run.summary.accuracy_ci95.upper);
      assert(
        run.cases.every((c) => c.scorer === "exact_match" && c.evidence.length),
      );
    }
    assert.deepEqual(
      result.runs[0].cases.map((c) => c.retrieved_source_ids),
      result.runs[1].cases.map((c) => c.retrieved_source_ids),
    );
    assert.equal(await page.locator(".comparison-chart svg").count(), 4);
    assert.equal(await page.locator("#comparison-form").isVisible(), false);
    await page.screenshot({
      path: out + "/hosted-charts-desktop.png",
      fullPage: true,
    });
    const series = page.locator(".chart-series input");
    await series.first().uncheck();
    assert.equal(await page.locator(".chart-series input:checked").count(), 1);
    await series.first().check();
    await page.click("[data-result-tab=measurements]");
    await page.locator("#inspect-run").selectOption(result.runs[1].id);
    await page.locator(".case-detail summary").first().click();
    assert((await page.locator(".evidence-card").count()) > 0);
    await page.setViewportSize({ width: 390, height: 844 });
    await page.waitForFunction(
      () =>
        document.querySelector(".main-content").getBoundingClientRect().x < 1,
    );
    assert(
      await page.evaluate(() => document.documentElement.scrollWidth <= 390),
    );
    await page.screenshot({
      path: out + "/hosted-charts-mobile.png",
      fullPage: true,
    });
    const csv = await page.request.get(
      `${base}/evalground/comparisons/${id}/export?format=csv`,
    );
    assert((await csv.text()).includes("cost_per_question_usd"));
    assert.deepEqual(errors, []);
    console.log(
      JSON.stringify(
        {
          status: "PASS",
          url: page.url(),
          models: result.runs.map((r) => ({
            model: r.reader.model,
            accuracy: r.summary.accuracy,
            cost: r.summary.total_cost_usd,
            p50: r.summary.latency_p50_seconds,
            p95: r.summary.latency_p95_seconds,
            input: r.summary.prompt_tokens,
            output: r.summary.completion_tokens,
          })),
        },
        null,
        2,
      ),
    );
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});

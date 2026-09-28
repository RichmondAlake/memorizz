/* Live acceptance: installed Ollama models, isolated fixture, no paid APIs. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const { chromium } = require(
  process.env.MEMORIZZ_PLAYWRIGHT_MODULE || "playwright",
);
(async () => {
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.MEMORIZZ_BROWSER_EXECUTABLE
      ? { executablePath: process.env.MEMORIZZ_BROWSER_EXECUTABLE }
      : {}),
  });
  const output = process.env.MEMORIZZ_COMPARISON_EVIDENCE || "/private/tmp";
  fs.mkdirSync(output, { recursive: true });
  try {
    const base =
      process.env.MEMORIZZ_BROWSER_TEST_URL || "http://127.0.0.1:8785";
    const page = await browser.newPage({
      viewport: { width: 1440, height: 1000 },
      extraHTTPHeaders: {
        Authorization: "Bearer memorizz-browser-fixture-token",
      },
    });
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(base + "/evalground");
    await page.getByRole("link", { name: "New comparison" }).click();
    assert.equal(await page.locator("h1").textContent(), "New comparison");
    await page.locator("#limit").fill("2");
    await page.locator("#experiment-name").fill("Live local reader comparison");
    await page.click("#setup-next");
    await page.locator('[data-panel="1"] > details > summary').click();
    for (const [selector, model] of [
      ["#readers .model-row >> nth=0", "qwen2.5:0.5b"],
      ["#readers .model-row >> nth=1", "gemma3:1b"],
      ["#judge .model-row", "qwen2.5:3b"],
    ]) {
      const row = page.locator(selector);
      await row.locator("[data-key=provider]").selectOption("ollama");
      await row
        .locator(`[data-key=model] option[value="${model}"]`)
        .waitFor({ state: "attached" });
      await row.locator("[data-key=model]").selectOption(model);
      if (!selector.startsWith("#judge")) {
        await row.locator("summary").click();
        await row.locator("[data-key=stream]").check();
      }
    }
    await page.click("#setup-next");
    await page.locator("#start-comparison").click();
    await page.waitForFunction(
      () =>
        [
          "completed",
          "completed_with_errors",
          "failed",
          "interrupted",
        ].includes(document.getElementById("result-status").textContent),
      {},
      { timeout: 600000 },
    );
    const id = new URL(page.url()).searchParams.get("experiment");
    const response = await page.request.get(
      `${base}/evalground/comparisons/${id}/export`,
    );
    const result = await response.json();
    fs.writeFileSync(
      output + "/live-readers.json",
      JSON.stringify(result, null, 2),
    );
    assert.equal(
      result.status,
      "completed",
      JSON.stringify(result.runs.map((r) => r.error)),
    );
    assert.equal(result.runs.length, 2);
    assert.equal(result.case_ids.length, 2);
    for (const run of result.runs) {
      assert.equal(run.summary.samples, 2);
      assert(run.summary.prompt_tokens > 0);
      assert(run.summary.latency_p50_seconds > 0);
      assert(run.summary.ttft_p50_seconds > 0);
      assert.equal(run.summary.total_cost_usd, 0);
      assert.equal(run.summary.unpriced_calls, 0);
    }
    assert.deepEqual(
      result.runs[0].cases.map((c) => c.retrieved_source_ids),
      result.runs[1].cases.map((c) => c.retrieved_source_ids),
    );
    assert.equal(result.comparisons[0].paired_cases, 2);
    const csv = await page.request.get(
      `${base}/evalground/comparisons/${id}/export?format=csv`,
    );
    assert((await csv.text()).includes("cost_per_question_usd"));
    await page.click("[data-result-tab=measurements]");
    await page.locator("#inspect-run").selectOption(result.runs[1].id);
    await page.locator(".case-detail summary").first().click();
    await page.screenshot({
      path: output + "/comparison-desktop.png",
      fullPage: true,
    });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.waitForFunction(
      () =>
        document.querySelector(".main-content").getBoundingClientRect().x < 1,
    );
    assert(
      await page.evaluate(() => document.documentElement.scrollWidth <= 390),
    );
    await page.screenshot({
      path: output + "/comparison-mobile.png",
      fullPage: true,
    });
    await page.reload();
    await page.waitForFunction(
      () =>
        document.getElementById("result-status").textContent === "completed",
    );
    await page.locator("#reuse-config").click();
    assert.equal(
      await page.locator("#experiment-name").inputValue(),
      "Live local reader comparison",
    );
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.locator("#experiment-type").selectOption("reranker");
    await page
      .locator("#experiment-name")
      .fill("Live local reranker comparison");
    await page.click("#setup-next");
    await page.locator("#add-reranker").click();
    const rerank = page.locator("#rerankers .model-row").last();
    await rerank.locator("[data-key=provider]").selectOption("llm");
    await rerank.locator("[data-key=llm_provider]").selectOption("ollama");
    await rerank
      .locator('[data-key=model] option[value="qwen2.5:3b"]')
      .waitFor({ state: "attached" });
    await rerank.locator("[data-key=model]").selectOption("qwen2.5:3b");
    await page.click("#setup-next");
    await page.locator("#start-comparison").click();
    await page.waitForFunction(
      () =>
        [
          "completed",
          "completed_with_errors",
          "failed",
          "interrupted",
        ].includes(document.getElementById("result-status").textContent) &&
        document.getElementById("result-name").textContent ===
          "Live local reranker comparison",
      {},
      { timeout: 600000 },
    );
    const rid = new URL(page.url()).searchParams.get("experiment");
    const reranked = await (
      await page.request.get(`${base}/evalground/comparisons/${rid}`)
    ).json();
    fs.writeFileSync(
      output + "/live-rerankers.json",
      JSON.stringify(reranked, null, 2),
    );
    assert.equal(
      reranked.status,
      "completed",
      JSON.stringify(reranked.runs.map((r) => r.error)),
    );
    assert(reranked.runs[1].calls.some((c) => c.lane === "reranker"));
    assert.deepEqual(
      reranked.runs[0].cases.map((c) => c.candidate_source_ids),
      reranked.runs[1].cases.map((c) => c.candidate_source_ids),
    );
    await page.screenshot({
      path: output + "/reranker-desktop.png",
      fullPage: true,
    });
    // A new real worker can be stopped and remains inspectable.
    await page.locator("#reuse-config").click();
    await page.locator("#experiment-name").fill("Cancellation check");
    await page.locator("#limit").fill("6");
    await page.click("#setup-next");
    await page.click("#setup-next");
    await page.locator("#repeats").fill("5");
    await page.locator("#start-comparison").click();
    await page.waitForFunction(
      () =>
        document.getElementById("result-name").textContent ===
        "Cancellation check",
    );
    await page.locator("#stop-comparison").click();
    await page.waitForFunction(
      () =>
        document.getElementById("result-status").textContent === "cancelled",
      {},
      { timeout: 30000 },
    );
    assert.deepEqual(errors, []);
    console.log(
      "PASS: real reader and LLM reranker comparisons, streamed TTFT, fixed candidates, cost/token reporting, case inspection, exports, reload/history, reuse, mobile, cancellation.",
    );
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});

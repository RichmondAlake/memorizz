/* Read-only acceptance against real saved runs. No hosted inference. */
const assert = require("node:assert/strict"),
  fs = require("node:fs"),
  path = require("node:path");
const { chromium } = require(
  process.env.MEMORIZZ_PLAYWRIGHT_MODULE || "playwright",
);
const base = process.env.MEMORIZZ_BROWSER_TEST_URL || "http://127.0.0.1:8766";
const out =
  process.env.MEMORIZZ_COMPARISON_EVIDENCE ||
  "/private/tmp/evalground-explainer";
fs.mkdirSync(out, { recursive: true });
(async () => {
  const browser = await chromium.launch({ headless: true });
  try {
    const p = await browser.newPage({
        viewport: { width: 1512, height: 1050 },
      }),
      errors = [];
    p.on("pageerror", (e) => errors.push(e.message));
    await p.goto(base + "/evalground");
    await p.waitForSelector("#run-library-rows tr[data-run-id]");
    const library = await (
      await p.request.get(base + "/evalground/run-library")
    ).json();
    const ranking = library.runs.find(
      (r) => r.kind === "reranker" && r.status === "completed",
    );
    const readers = library.runs.find(
      (r) => r.kind === "reader" && r.status === "completed",
    );
    assert.ok(
      ranking && readers,
      "Completed reranker and reader runs are discoverable",
    );
    assert.equal(
      await p
        .locator(`#run-library-rows tr[data-run-id="${ranking.id}"]`)
        .count(),
      1,
    );
    await p.selectOption("#run-kind", "reader");
    assert.equal(
      await p.locator("#run-library-rows tr[data-kind=reranker]").count(),
      0,
    );
    await p.selectOption("#run-kind", "all");
    await p.fill("#run-search", "not-present-test");
    assert.ok(
      (await p.locator("#run-library-rows").innerText()).includes(
        "No runs match",
      ),
    );
    await p.fill("#run-search", "");
    await p.selectOption("#run-status", "completed");
    await p.locator("#run-library").scrollIntoViewIfNeeded();
    await p.screenshot({ path: path.join(out, "home-runs.png") });
    await p.locator(`tr[data-run-id="${ranking.id}"] a.run-name`).click();
    await p.waitForSelector("#comparison-findings");
    assert.equal(
      await p.locator("[data-scope=reranker]").getAttribute("aria-pressed"),
      "true",
    );
    assert.ok(
      (await p.locator(".comparison-chart h3").first().innerText()).includes(
        "nDCG@3",
      ),
    );
    const text = await p.locator("#comparison-visuals").innerText();
    assert.ok(!text.includes("none · noul"));
    assert.ok(text.includes("Original candidate order"));
    for (const recipe of ["Jev Noul", "Jev Score", "Jev Choice"])
      assert.ok(text.includes(recipe));
    const result = await (
      await p.request.get(base + "/evalground/comparisons/" + ranking.id)
    ).json();
    for (const row of result.runs) {
      const sum = row.calls
        .filter((c) => c.lane === "reranker")
        .reduce((s, c) => s + c.cost_usd, 0);
      const cells = p
        .locator("#results-table tbody tr")
        .nth(result.runs.indexOf(row))
        .locator("td");
      assert.equal(
        await cells.last().innerText(),
        "$" + (sum / row.cases.length).toFixed(6),
      );
    }
    await p.selectOption("#ranking-metric", "mrr");
    assert.ok(
      (await p.locator(".comparison-chart h3").first().innerText()).includes(
        "MRR@3",
      ),
    );
    await p.selectOption("#ranking-metric", "ndcg_at_k");
    await p.locator("[data-scope=pipeline]").click();
    assert.ok(
      (await p.locator(".comparison-chart h3").nth(1).innerText()).includes(
        "Full answer pipeline",
      ),
    );
    await p.locator("[data-scope=reranker]").click();
    await p.locator("#comparison-visuals").scrollIntoViewIfNeeded();
    await p.screenshot({ path: path.join(out, "ranking-metrics.png") });
    await p.click("[data-result-tab=evidence]");
    await p.selectOption(
      "#inspect-run",
      result.runs.find(
        (r) =>
          r.reranker.jev_method === "choice" && r.reranker.provider === "jev",
      ).id,
    );
    await p.waitForSelector("#ranking-question");
    await p.selectOption("#ranking-question", "q01");
    await p.waitForFunction(() =>
      document
        .querySelector("#ranking-explorer")
        .textContent.includes("↑ Up 1"),
    );
    assert.ok(
      (
        await p.locator(".ranking-columns section").first().innerText()
      ).includes("#1 · m01"),
    );
    assert.ok(
      (await p.locator(".ranking-columns section").nth(1).innerText()).includes(
        "#1 · m02",
      ),
    );
    assert.ok(
      (await p.locator("#ranking-explorer .notice").innerText()).includes(
        "Precision@3 = 1/3",
      ),
    );
    assert.ok(
      (await p.locator("#ranking-explorer").innerText()).includes(
        "not absolute relevance probabilities",
      ),
    );
    await p.locator("#ranking-explorer").scrollIntoViewIfNeeded();
    await p.screenshot({ path: path.join(out, "evidence-before-after.png") });
    await p.goto(base + readers.url);
    await p.waitForSelector("#comparison-findings");
    assert.equal(
      await p.locator("[data-scope=pipeline]").getAttribute("aria-pressed"),
      "true",
    );
    assert.ok(
      (await p.locator(".comparison-chart h3").first().innerText()).includes(
        "Lexical answer check",
      ),
    );
    assert.ok(
      (await p.locator("#comparison-findings").innerText()).includes(
        "ceiling effect",
      ),
    );
    await p.goto(base + "/evalground/compare#experiment-library");
    assert.equal(
      await p.locator("#experiment-library").getAttribute("open"),
      "",
    );
    const links = await p
      .locator(".lesson-links a[target=_blank]")
      .evaluateAll((a) => a.map((x) => x.href));
    assert.equal(links.length, 4);
    for (const link of links) {
      const chapter = await browser.newPage();
      await chapter.goto(link);
      const lab = new URL(link).searchParams.get("lab");
      await chapter.waitForSelector(`nav [data-lab=${lab}].active`);
      await chapter.waitForSelector("#experiment:not([hidden])");
      await chapter.waitForFunction(() =>
        document
          .querySelector("#status-line")
          .textContent.startsWith("COMPLETED"),
      );
      await chapter.close();
    }
    await p.setViewportSize({ width: 390, height: 844 });
    await p.goto(base + "/evalground");
    await p.waitForSelector("#run-library-rows tr[data-run-id]");
    assert.ok(
      await p.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth + 1,
      ),
    );
    await p.locator("#run-library").scrollIntoViewIfNeeded();
    await p.screenshot({ path: path.join(out, "home-mobile.png") });
    await p.goto(base + ranking.url);
    await p.waitForSelector("#comparison-findings");
    await p.click("[data-result-tab=evidence]");
    await p.waitForSelector(".ranking-columns");
    assert.ok(
      await p.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth + 1,
      ),
    );
    await p.locator("#ranking-explorer").scrollIntoViewIfNeeded();
    await p.screenshot({ path: path.join(out, "evidence-mobile.png") });
    assert.deepEqual(errors, []);
    fs.writeFileSync(
      path.join(out, "checks.json"),
      JSON.stringify(
        {
          status: "passed",
          checks: [
            "home table lists saved runs with working result links",
            "type/status/search filters",
            "reranking-first metrics and explicit stage controls",
            "displayed rank costs match measured ledger",
            "before/after evidence and real rank movements",
            "question-level precision denominator",
            "reader ceiling-effect interpretation",
            "four appbook chapter links open completed Oracle results",
            "desktop and mobile layouts",
            "no JavaScript errors",
          ],
        },
        null,
        2,
      ),
    );
    console.log(
      "Evalground home, interpretation, stage costs, evidence and lesson navigation passed.",
    );
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exit(1);
});

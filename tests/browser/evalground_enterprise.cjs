/* Enterprise workflow acceptance. Saved results are real; submission is intercepted before inference. */
const assert = require("node:assert/strict"),
  fs = require("node:fs"),
  path = require("node:path");
const { chromium } = require(
  process.env.MEMORIZZ_PLAYWRIGHT_MODULE || "playwright",
);
const base = process.env.MEMORIZZ_BROWSER_TEST_URL || "http://127.0.0.1:8766";
const out =
  process.env.MEMORIZZ_COMPARISON_EVIDENCE ||
  "/private/tmp/evalground-enterprise";
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
    assert.equal(await p.locator(".evaluation-starts a").count(), 3);
    for (const id of [
      "agent-evaluation",
      "dataset-library",
      "suggested-experiments",
    ])
      assert.equal(await p.locator("#" + id).getAttribute("open"), null);
    await p.screenshot({ path: path.join(out, "home-desktop.png") });
    await p.locator(".evaluation-actions [data-open-section]").click();
    assert.equal(await p.locator("#agent-evaluation").getAttribute("open"), "");
    await p.locator('.evaluation-starts a[href$="type=reranker"]').click();
    assert.equal(await p.locator("#experiment-type").inputValue(), "reranker");
    assert.ok(await p.locator('[data-panel="0"]').isVisible());
    assert.ok(!(await p.locator('[data-panel="1"]').isVisible()));
    await p.selectOption("#dataset", "custom");
    await p.click("#setup-next");
    assert.equal(
      await p.locator('[data-step="0"]').getAttribute("aria-current"),
      "step",
    );
    assert.equal(
      await p.locator("#data-path").evaluate((e) => e.validity.valueMissing),
      true,
    );
    await p.selectOption("#dataset", "memory_checks");
    await p.screenshot({ path: path.join(out, "setup-dataset.png") });
    await p.locator("#sample-presets summary").click();
    await p.click("#load-system-one");
    await p.waitForFunction(
      () => document.querySelector("#candidate-snapshot").value.length > 0,
    );
    await p.waitForFunction(() =>
      [...document.querySelectorAll(".model-row")].every(
        (r) => r.dataset.loading === "false",
      ),
    );
    await p.click("#setup-next");
    assert.ok(await p.locator('[data-panel="1"]').isVisible());
    assert.equal(await p.locator("#readers .model-row").count(), 1);
    assert.equal(await p.locator("#rerankers .model-row").count(), 7);
    assert.ok(
      (await p.locator("#rerankers .model-role").first().innerText()).includes(
        "Baseline",
      ),
    );
    await p.screenshot({ path: path.join(out, "setup-models.png") });
    await p.click("#setup-next");
    assert.ok(await p.locator('[data-panel="2"]').isVisible());
    assert.ok(
      (await p.locator("#run-plan").innerText()).includes(
        "42 evaluated answers",
      ),
    );
    assert.ok(
      (await p.locator("#review-summary").innerText()).includes(
        "Fixed evidence",
      ),
    );
    assert.equal(await p.locator("#embedding").isDisabled(), true);
    await p.fill("#max-cost", ".50");
    assert.ok(
      (await p.locator("#review-summary").innerText()).includes("$0.50"),
    );
    for (const recipe of ["Jev Noul", "Jev Score", "Jev Choice"])
      assert.ok(
        (await p.locator("#review-summary").innerText()).includes(recipe),
      );
    await p.screenshot({ path: path.join(out, "setup-review.png") });
    let submitted;
    await p.route("**/evalground/comparisons", async (route) => {
      if (route.request().method() !== "POST") return route.continue();
      submitted = route.request().postDataJSON();
      await route.fulfill({
        status: 422,
        json: { detail: "Browser validation only: inference was not started." },
      });
    });
    await p.click("#start-comparison");
    await p.waitForFunction(() =>
      document
        .querySelector("#form-error")
        .textContent.includes("inference was not started"),
    );
    assert.equal(submitted.rerankers.length, 7);
    assert.equal(submitted.max_cost_usd, 0.5);
    assert.equal(submitted.top_k, 3);
    fs.writeFileSync(
      path.join(out, "validated-submission.json"),
      JSON.stringify(submitted, null, 2),
    );
    await p.unroute("**/evalground/comparisons");
    const library = await (
      await p.request.get(base + "/evalground/run-library")
    ).json();
    const ranking = library.runs.find(
      (r) => r.kind === "reranker" && r.status === "completed",
    );
    await p.goto(base + ranking.url);
    await p.waitForSelector("#comparison-findings");
    assert.ok(await p.locator("#result-overview").isVisible());
    assert.ok(!(await p.locator("#result-evidence").isVisible()));
    assert.equal(await p.locator("#comparison-form").isVisible(), false);
    await p.click("#toggle-setup");
    assert.ok(await p.locator("#saved-configuration").isVisible());
    assert.ok(
      (await p.locator("#saved-configuration-summary").innerText()).includes(
        "3 / 20",
      ),
    );
    assert.equal(await p.locator("#comparison-form").isVisible(), false);
    await p.click("#toggle-setup");
    assert.ok(
      // Labels are sentence case; compare without depending on CSS casing.
      (await p.locator("#comparison-findings").innerText())
        .toLowerCase()
        .includes("sample dataset"),
    );
    await p.screenshot({ path: path.join(out, "results-overview.png") });
    await p.click("[data-result-tab=evidence]");
    await p.waitForSelector(".ranking-columns");
    assert.ok(await p.locator("#configuration-inspector").isVisible());
    const detail = await (
      await p.request.get(base + "/evalground/comparisons/" + ranking.id)
    ).json();
    await p.selectOption(
      "#inspect-run",
      detail.runs.find(
        (r) =>
          r.reranker.provider === "jev" && r.reranker.jev_method === "choice",
      ).id,
    );
    await p.waitForFunction(() =>
      document
        .querySelector("#ranking-explorer")
        .textContent.includes("↑ Up 1"),
    );
    await p.screenshot({ path: path.join(out, "results-evidence.png") });
    await p.click("[data-result-tab=measurements]");
    assert.ok(await p.locator("#run-details").isVisible());
    assert.equal(await p.locator("#run-details .case-detail").count(), 6);
    await p.click("#reuse-config");
    assert.ok(await p.locator('[data-panel="0"]').isVisible());
    assert.ok(!(await p.locator("#experiment-results").isVisible()));
    assert.equal(
      await p.locator("#candidate-snapshot").inputValue(),
      detail.config.candidate_snapshot_path,
    );
    assert.equal(
      await p.locator("#experiment-name").inputValue(),
      "Reranking comparison · 7 methods",
    );
    await p.setViewportSize({ width: 390, height: 844 });
    await p.goto(base + "/evalground");
    await p.waitForSelector("#run-library-rows tr[data-run-id]");
    assert.ok(
      await p.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth + 1,
      ),
    );
    await p.screenshot({ path: path.join(out, "home-mobile.png") });
    await p.goto(base + "/evalground/compare?type=reader");
    await p.waitForSelector('[data-step="0"][aria-current=step]');
    assert.ok(
      await p.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth + 1,
      ),
    );
    await p.screenshot({ path: path.join(out, "setup-mobile.png") });
    assert.deepEqual(errors, []);
    fs.writeFileSync(
      path.join(out, "checks.json"),
      JSON.stringify(
        {
          status: "passed",
          inference_started: false,
          checks: [
            "three task-based entry points",
            "advanced home sections are collapsed and open correctly",
            "three-step comparison with required dataset validation",
            "provider models and explicit baseline roles",
            "snapshot-aware review and live cost controls",
            "validated submission intercepted before inference",
            "overview/evidence/measurement tabs with real saved data",
            "duplicate keeps original evidence and settings",
            "sample-data scope remains visible",
            "desktop/mobile layouts",
            "no JavaScript runtime errors",
          ],
        },
        null,
        2,
      ),
    );
    console.log("Enterprise workflow checks passed; no inference started.");
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exit(1);
});

/* Read-only UI acceptance against a completed comparison; no model inference. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const { chromium } = require(
  process.env.MEMORIZZ_PLAYWRIGHT_MODULE || "playwright",
);
(async () => {
  const base = process.env.MEMORIZZ_BROWSER_TEST_URL || "http://localhost:8765";
  const out =
    process.env.MEMORIZZ_COMPARISON_EVIDENCE ||
    "/private/tmp/memorizz-hosted-evidence";
  const result = JSON.parse(
    fs.readFileSync(out + "/hosted-six-questions.json", "utf8"),
  );
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({
        viewport: { width: 1500, height: 1100 },
      }),
      errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(base + "/evalground/compare?experiment=" + result.id);
    await page.waitForSelector(".comparison-chart svg");
    assert.equal(await page.locator(".comparison-chart svg").count(), 4);
    await page.getByRole("button", { name: "Duplicate configuration" }).click();
    await page.waitForFunction(() =>
      [...document.querySelectorAll("#readers .model-row")].every(
        (r) => r.dataset.loading === "false",
      ),
    );
    await page.click("#setup-next");
    const row = page.locator("#readers .model-row").first();
    assert.equal(
      await row.locator("[data-key=model]").inputValue(),
      result.runs[0].reader.model,
    );
    await row.locator("[data-key=model]").selectOption("__custom__");
    await row.locator("[data-key=custom_model]").fill("custom-deployment");
    assert(await row.locator("[data-key=custom_model]").isVisible());
    await row.locator("[data-key=provider]").selectOption("azure");
    await page.waitForFunction(
      () =>
        document.querySelector("#readers .model-row").dataset.loading ===
        "false",
    );
    assert(
      (await row.locator(".model-discovery").textContent()).includes("Azure"),
    );
    // Simulate a slow previous provider response; it must not overwrite the newer selection.
    await page.route("**/evalground/model-catalog?**", async (route) => {
      const provider = new URL(route.request().url()).searchParams.get(
        "provider",
      );
      if (provider === "openai")
        await new Promise((resolve) => setTimeout(resolve, 250));
      await route.fulfill({
        json: {
          provider,
          source: "provider_api",
          message: "Acceptance fixture",
          models: [
            {
              id: provider + "-fixture",
              name: provider + "-fixture",
              pricing: null,
            },
          ],
        },
      });
    });
    await row.locator("[data-key=provider]").selectOption("openai");
    await row.locator("[data-key=provider]").selectOption("anthropic");
    await row
      .locator("[data-key=model] option[value=anthropic-fixture]")
      .waitFor({ state: "attached" });
    await page.waitForTimeout(400);
    assert.equal(
      await row.locator("[data-key=model]").inputValue(),
      "anthropic-fixture",
    );
    assert.equal(
      await row
        .locator("[data-key=model] option[value=openai-fixture]")
        .count(),
      0,
    );
    await row.getByRole("button", { name: "Refresh model list" }).click();
    await row
      .locator("[data-key=model] option[value=anthropic-fixture]")
      .waitFor({ state: "attached" });
    await page.unroute("**/evalground/model-catalog?**");
    // Restore actual saved results without making paid calls.
    await page.goto(base + "/evalground/compare?experiment=" + result.id);
    await page.waitForSelector(".comparison-chart svg");
    await page.screenshot({
      path: out + "/hosted-charts-desktop.png",
      fullPage: true,
    });
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
    assert.deepEqual(errors, []);
    console.log(
      "PASS: saved results, live model menus, custom deployment, refresh, stale-response protection, four data charts, evidence cards, mobile layout; no inference calls.",
    );
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});

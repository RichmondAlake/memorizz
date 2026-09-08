const assert = require('node:assert/strict')
const { chromium } = require(process.env.MEMORIZZ_PLAYWRIGHT_MODULE || 'playwright')
const path = require('node:path')
;(async () => {
  const browser = await chromium.launch({ headless: true, ...(process.env.MEMORIZZ_BROWSER_EXECUTABLE ? { executablePath: process.env.MEMORIZZ_BROWSER_EXECUTABLE } : {}) })
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1100 }, extraHTTPHeaders: { Authorization: 'Bearer persona-browser-fixture-token' } })
    const errors = []
    page.on('pageerror', error => errors.push(error.message))
    await page.route('**/*', route => new URL(route.request().url()).hostname === '127.0.0.1' ? route.continue() : route.abort())
    await page.goto(`${process.env.MEMORIZZ_PERSONA_TEST_URL || 'http://127.0.0.1:8783'}/persona-evolution?user_id=synthetic-learner`)
    await page.getByRole('button', { name: 'Reflect now', exact: true }).click()
    await page.getByText('SUGGESTED ADAPTATION · NOT APPLIED', { exact: true }).waitFor()
    assert(await page.getByText('Version 1', { exact: true }).count())
    assert((await page.locator('.pe-evidence a').getAttribute('href')).includes('thread_id=synthetic-thread'))
    const output = process.env.PERSONA_SCREENSHOT_DIR || '/private/tmp'
    await page.screenshot({ path: path.join(output, '2026-09-07-memorizz-persona-review.png'), fullPage: true })
    await page.getByRole('button', { name: 'Approve for next turn', exact: true }).click()
    await page.getByText('Version 2', { exact: true }).waitFor()
    await page.locator('.pe-event').first().locator('summary').click()
    await page.screenshot({ path: path.join(output, '2026-09-07-memorizz-persona-history.png'), fullPage: true })
    await page.getByRole('button', { name: 'Undo last change', exact: true }).click()
    await page.getByText('Version 3', { exact: true }).waitFor()
    await page.getByRole('button', { name: 'Pause reflection', exact: true }).click()
    await page.getByRole('button', { name: 'Resume reflection', exact: true }).waitFor()
    assert(await page.getByRole('button', { name: 'Reflect now', exact: true }).isDisabled())
    await page.setViewportSize({ width: 390, height: 844 })
    await page.waitForFunction(() => {
      const box = document.querySelector('.main-content').getBoundingClientRect()
      return box.x < 1 && box.width > 389
    })
    const layout = await page.locator('.main-content').boundingBox()
    assert(layout.x < 30 && layout.width > 330, JSON.stringify(layout))
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= 390))
    await page.screenshot({ path: path.join(output, '2026-09-07-memorizz-persona-mobile.png'), fullPage: true })
    assert.deepEqual(errors, [])
    console.log('PASS: Memorizz persona reflection, evidence link, approval, history, undo, pause, mobile overflow; no browser errors. Synthetic host fixture.')
  } finally { await browser.close() }
})().catch(error => { console.error(error); process.exitCode = 1 })

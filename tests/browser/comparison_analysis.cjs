/* Offline arithmetic checks for the UI's derived views. */
const assert = require("node:assert/strict");
const A = require("../../src/memorizz/ui/static/js/comparison-analysis.js");
const near = (a, b) => assert.ok(Math.abs(a - b) < 1e-12, `${a} != ${b}`);
const row = {
  reranker: { provider: "jev" },
  summary: {},
  calls: [
    { lane: "reranker", cost_usd: 0.002 },
    { lane: "reader", cost_usd: 0.01 },
    { lane: "judge", cost_usd: 0.05 },
    { lane: "reranker", cost_usd: 0.003, status: "failed" },
  ],
  cases: [
    {
      case_id: "a",
      retrieval_seconds: 2,
      generation_seconds: 8,
      retrieval_timing: { reranker_seconds: 1 },
    },
    {
      case_id: "b",
      retrieval_seconds: 8,
      generation_seconds: 2,
      retrieval_timing: { reranker_seconds: 7 },
    },
  ],
};
near(A.economics(row, "pipeline").p50, 10);
near(A.economics(row, "reranker").p50, 4);
near(A.economics(row, "reranker").p95, 6.7);
near(A.cost(row, "pipeline"), 0.015);
near(A.economics(row, "reranker").perQuestion, 0.0025);
assert.equal(A.latency({}, "pipeline"), null);
assert.equal(A.percentile([1, null], 0.5), null);
assert.equal(
  A.cost({ ...row, calls: [{ lane: "reranker", cost_usd: null }] }, "reranker"),
  null,
);
assert.equal(
  A.cost({ ...row, summary: { billing_incomplete: true } }, "pipeline"),
  null,
);
assert.equal(
  A.cost({ reranker: { provider: "none" }, calls: [] }, "reranker"),
  0,
);
assert.equal(
  A.cost({ reranker: { provider: "jev" }, calls: [] }, "reranker"),
  null,
);
assert.equal(
  A.modelName({ provider: "none", jev_method: "noul" }),
  "Original candidate order",
);
assert.equal(
  A.modelName({ provider: "jev", model: "jev-1", jev_method: "score" }),
  "Jev Score · jev-1",
);
const pair = A.pairedMetric(
  {
    cases: [
      { case_id: "a", retrieval: { ndcg_at_k: 1 } },
      { case_id: "b", retrieval: { ndcg_at_k: 0.2 } },
    ],
  },
  {
    cases: [
      { case_id: "a", retrieval: { ndcg_at_k: 0.5 } },
      { case_id: "c", retrieval: { ndcg_at_k: 0 } },
    ],
  },
  "ndcg_at_k",
);
assert.equal(pair.n, 1);
near(pair.delta, 0.5);
assert.equal(pair.improved.length, 1);
console.log(
  "UI arithmetic, unknown measurements, stage isolation and paired-case checks passed.",
);

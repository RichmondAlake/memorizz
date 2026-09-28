/* Derived views of saved measurements. No model calls or fabricated observations. */
(() => {
  const finite = (v) => typeof v === "number" && Number.isFinite(v);
  const mean = (xs) =>
    xs.length && xs.every(finite)
      ? xs.reduce((s, v) => s + v, 0) / xs.length
      : null;
  const percentile = (xs, p) => {
    if (!xs.length || !xs.every(finite)) return null;
    const a = [...xs].sort((x, y) => x - y),
      n = (a.length - 1) * p,
      lo = Math.floor(n);
    return a[lo] + (a[Math.ceil(n)] - a[lo]) * (n - lo);
  };
  const fmt = (v, n = 3) => (finite(v) ? v.toFixed(n) : "Unknown");
  const money = (v) => (finite(v) ? "$" + v.toFixed(6) : "Unknown");
  function experimentName(name) {
    return name === "System One lesson · Oracle / Voyage · 7 reranking methods"
      ? "Reranking comparison · 7 methods"
      : name || "Model comparison";
  }
  function modelName(spec = {}) {
    if (spec.provider === "none") return "Original candidate order";
    if (spec.label) return spec.label;
    if (spec.provider === "jev")
      return `Jev ${{ noul: "Noul", score: "Score", choice: "Choice" }[spec.jev_method || "noul"]} · ${spec.model || "default"}`;
    return spec.model || spec.provider || "Unknown model";
  }
  function runName(row, data) {
    const type = data?.config?.experiment_type;
    const primary =
      type === "reranker"
        ? modelName(row.reranker)
        : type === "reader"
          ? modelName(row.reader)
          : `${modelName(row.reader)} / ${modelName(row.reranker)}`;
    return (
      primary +
      ((data?.config?.repeats || 1) > 1 ? ` · repeat ${row.repeat}` : "")
    );
  }
  function latency(c, scope) {
    if (scope === "reranker")
      return finite(c.retrieval_timing?.reranker_seconds)
        ? c.retrieval_timing.reranker_seconds
        : null;
    if (scope === "reader")
      return finite(c.generation_seconds) ? c.generation_seconds : null;
    return finite(c.retrieval_seconds) && finite(c.generation_seconds)
      ? c.retrieval_seconds + c.generation_seconds
      : null;
  }
  const lanes = (scope) =>
    scope === "pipeline" ? ["reader", "reranker"] : [scope];
  function callCost(calls) {
    return calls.every((c) => finite(c.cost_usd))
      ? calls.reduce((s, c) => s + c.cost_usd, 0)
      : null;
  }
  function cost(row, scope) {
    if (row.summary?.billing_incomplete) return null;
    const wanted = lanes(scope),
      calls = (row.calls || []).filter((c) => wanted.includes(c.lane));
    // A missing charge is zero only for the explicit no-reranker stage.
    if (!calls.length)
      return scope === "reranker" && row.reranker?.provider === "none"
        ? 0
        : null;
    return callCost(calls);
  }
  function caseCost(c, row, scope) {
    const calls = (c.measurements || []).filter((call) =>
      lanes(scope).includes(call.lane),
    );
    if (!calls.length)
      return scope === "reranker" && row.reranker?.provider === "none"
        ? 0
        : null;
    return callCost(calls);
  }
  function economics(row, scope) {
    const values = (row.cases || []).map((c) => latency(c, scope)),
      total = cost(row, scope);
    return {
      p50: percentile(values, 0.5),
      p95: percentile(values, 0.95),
      mean: mean(values),
      total,
      perQuestion:
        values.length && finite(total) ? total / values.length : null,
      n: values.length,
    };
  }
  function pairedMetric(row, baseline, key) {
    const other = new Map((baseline?.cases || []).map((c) => [c.case_id, c]));
    const value = (c) => (key === "score" ? c.score : c.retrieval?.[key]);
    const pairs = (row.cases || [])
      .filter((c) => other.has(c.case_id))
      .map((c) => ({
        id: c.case_id,
        question: c.question,
        after: value(c),
        before: value(other.get(c.case_id)),
      }))
      .filter((p) => finite(p.after) && finite(p.before));
    return {
      n: pairs.length,
      before: mean(pairs.map((p) => p.before)),
      after: mean(pairs.map((p) => p.after)),
      delta: mean(pairs.map((p) => p.after - p.before)),
      improved: pairs.filter((p) => p.after - p.before > 1e-9),
      regressed: pairs.filter((p) => p.before - p.after > 1e-9),
    };
  }
  const api = {
    finite,
    mean,
    percentile,
    fmt,
    money,
    modelName,
    runName,
    latency,
    cost,
    caseCost,
    economics,
    pairedMetric,
    experimentName,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof window !== "undefined") window.MemorizzComparisonAnalysis = api;
})();

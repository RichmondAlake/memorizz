/* Accessible SVG charts built only from exported measurements. */
(() => {
  const A = window.MemorizzComparisonAnalysis;
  const colors = [
    "#47d7cb",
    "#b09aff",
    "#ffbe76",
    "#71baff",
    "#f58cbe",
    "#c5da75",
    "#ff8879",
    "#9ad5ef",
  ];
  const el = (tag, text, cls) => {
    const n = document.createElement(tag);
    if (text !== undefined) n.textContent = text;
    if (cls) n.className = cls;
    return n;
  };
  const svg = (tag, attrs = {}, text) => {
    const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, v));
    if (text !== undefined) n.textContent = text;
    return n;
  };
  let experiment,
    hidden = new Set(),
    scope = "pipeline",
    metric = "ndcg_at_k";
  const scopeNames = {
    reranker: "Reranker only",
    reader: "Answer model only",
    pipeline: "Full answer pipeline",
  };
  const metrics = {
    ndcg_at_k: "nDCG",
    mrr: "MRR",
    recall_at_k: "Recall",
    precision_at_k: "Precision",
  };
  const title = (n, t) => n.append(svg("title", {}, t));
  function panel(root, name, note) {
    const p = el("section", undefined, "comparison-chart");
    p.append(el("h3", name), el("p", note, "muted"));
    root.append(p);
    return p;
  }
  function frame(parent, name, height = 270) {
    const s = svg("svg", {
      viewBox: `0 0 600 ${height}`,
      role: "img",
      "aria-label": name,
    });
    title(s, name);
    parent.append(s);
    return s;
  }
  function interact(node, row, onSelect) {
    node.setAttribute("tabindex", "0");
    node.addEventListener("click", () => onSelect(row.id));
    node.addEventListener("keydown", (e) => {
      if (e.key === "Enter") onSelect(row.id);
    });
  }
  function axis(s, max, kind) {
    for (let i = 0; i <= 4; i++) {
      const y = 220 - i * 46;
      const v = (max * i) / 4;
      s.append(
        svg("line", { x1: 78, x2: 574, y1: y, y2: y, class: "chart-grid" }),
        svg(
          "text",
          { x: 70, y: y + 4, "text-anchor": "end", class: "chart-axis" },
          kind === "cost" ? "$" + A.fmt(v, 6) : A.fmt(v, 2) + "s",
        ),
      );
    }
  }
  function metricBars(parent, runs, key, k, data, onSelect) {
    const ranking = key !== "accuracy",
      name = ranking
        ? `${metrics[key]}@${k}`
        : data.accuracy_label || "Answer accuracy";
    const h = Math.max(225, 65 + runs.length * 47),
      s = frame(parent, name, h);
    for (let i = 0; i <= 4; i++) {
      const x = 70 + i * 115;
      s.append(
        svg("line", { x1: x, x2: x, y1: 18, y2: h - 28, class: "chart-grid" }),
        svg(
          "text",
          { x, y: h - 7, "text-anchor": "middle", class: "chart-axis" },
          ranking ? A.fmt(i / 4, 2) : `${i * 25}%`,
        ),
      );
    }
    runs.forEach((r, i) => {
      const y = 38 + i * 47,
        v = r.summary?.[key];
      s.append(
        svg(
          "text",
          { x: 58, y: y + 4, "text-anchor": "end", class: "chart-axis" },
          r.short,
        ),
      );
      if (!A.finite(v)) {
        s.append(
          svg("text", { x: 80, y: y + 4, class: "chart-axis" }, "Not measured"),
        );
        return;
      }
      const bar = svg("rect", {
        x: 70,
        y: y - 9,
        width: v * 460,
        height: 18,
        rx: 3,
        fill: r.color,
        opacity: 0.75,
      });
      title(
        bar,
        `${A.runName(r, data)}: ${name} ${A.fmt(v)} · ${r.summary.samples} questions`,
      );
      interact(bar, r, onSelect);
      s.append(bar);
      if (!ranking && r.summary.accuracy_ci95) {
        const ci = r.summary.accuracy_ci95,
          a = 70 + ci.lower * 460,
          b = 70 + ci.upper * 460;
        s.append(
          svg("path", {
            d: `M ${a} ${y - 7} V ${y + 7} M ${a} ${y} H ${b} M ${b} ${y - 7} V ${y + 7}`,
            stroke: r.color,
            "stroke-width": 2,
          }),
        );
      }
      s.append(
        svg(
          "text",
          { x: 542, y: y + 4, class: "chart-axis" },
          ranking ? A.fmt(v) : `${A.fmt(v * 100, 0)}%`,
        ),
      );
    });
    parent.append(
      el(
        "p",
        ranking
          ? "Mean across labeled questions. Higher is better. These bars do not establish statistical significance."
          : "Whiskers: Wilson 95% intervals. Lexical checks do not establish factual correctness.",
        "chart-note",
      ),
    );
  }
  function costBars(parent, runs, data, onSelect) {
    const s = frame(parent, scopeNames[scope] + " API cost per question");
    const max = Math.max(
      ...runs.map((r) => A.economics(r, scope).perQuestion || 0),
      0.000001,
    );
    axis(s, max, "cost");
    const w = 490 / Math.max(1, runs.length);
    runs.forEach((r, i) => {
      const x = 82 + i * w,
        v = A.economics(r, scope).perQuestion;
      if (A.finite(v)) {
        const h = (v / max) * 184,
          bar = svg("rect", {
            x,
            y: 220 - h,
            width: w * 0.68,
            height: Math.max(h, 1),
            rx: 3,
            fill: r.color,
          });
        title(
          bar,
          `${A.runName(r, data)}\n${scopeNames[scope]}: ${A.money(v)} / completed question`,
        );
        interact(bar, r, onSelect);
        s.append(bar);
      } else
        s.append(
          svg(
            "text",
            {
              x: x + w * 0.34,
              y: 205,
              "text-anchor": "middle",
              class: "chart-axis",
            },
            "?",
          ),
        );
      s.append(
        svg(
          "text",
          {
            x: x + w * 0.34,
            y: 242,
            "text-anchor": "middle",
            class: "chart-axis",
          },
          r.short,
        ),
      );
    });
    parent.append(
      el(
        "p",
        "Selected-stage API ledger total ÷ completed questions, including recorded failed attempts. Unknown billing stays unknown. Local API cost is zero; hardware costs are separate.",
        "chart-note",
      ),
    );
  }
  function lines(parent, runs, data, kind, onSelect) {
    const ids = data.case_ids || [],
      isCost = kind === "cost";
    const series = runs.map((r) => {
      let cumulative = 0,
        unknown = false;
      const byId = new Map((r.cases || []).map((c) => [c.case_id, c]));
      return {
        ...r,
        points: ids.map((id, i) => {
          const c = byId.get(id);
          if (!c) return null;
          let v = A.latency(c, scope);
          if (isCost) {
            const amount = A.caseCost(c, r, scope);
            if (!A.finite(amount)) unknown = true;
            else cumulative += amount;
            v = unknown ? null : cumulative;
          }
          return { i, v, question: c.question };
        }),
      };
    });
    const max = Math.max(
      ...series.flatMap((r) =>
        r.points.filter((p) => p && A.finite(p.v)).map((p) => p.v),
      ),
      isCost ? 0.000001 : 1,
    );
    const s = frame(
      parent,
      isCost
        ? "Cumulative selected-stage API spend"
        : "Selected-stage latency per question",
    );
    axis(s, max, kind);
    const x = (i) => (ids.length < 2 ? 325 : 78 + (i * 496) / (ids.length - 1)),
      y = (v) => 220 - (v / max) * 184;
    series.forEach((r) => {
      let d = "",
        start = false;
      r.points.forEach((p) => {
        if (!p || !A.finite(p.v)) {
          start = false;
          return;
        }
        d += `${start ? "L" : "M"} ${x(p.i)} ${y(p.v)} `;
        start = true;
      });
      s.append(
        svg("path", { d, fill: "none", stroke: r.color, "stroke-width": 2.5 }),
      );
      r.points.forEach((p) => {
        if (!p || !A.finite(p.v)) return;
        const dot = svg("circle", {
          cx: x(p.i),
          cy: y(p.v),
          r: 4.5,
          fill: r.color,
          stroke: "var(--bg-card)",
          "stroke-width": 2,
        });
        title(
          dot,
          `${A.runName(r, data)}\n${p.question}\n${isCost ? A.money(p.v) : A.fmt(p.v) + " seconds"}`,
        );
        interact(dot, r, onSelect);
        s.append(dot);
      });
    });
    ids.forEach((id, i) => {
      if (ids.length > 20 && i % Math.ceil(ids.length / 20) !== 0) return;
      s.append(
        svg(
          "text",
          { x: x(i), y: 242, "text-anchor": "middle", class: "chart-axis" },
          `Q${i + 1}`,
        ),
      );
    });
    parent.append(
      el(
        "p",
        isCost
          ? "Completed cases only; failed calls outside a completed case remain in ledger totals. Evaluation calls are excluded."
          : "Dataset order, not wall-clock time. Hover or focus a point for the question and measured time. Model loading is separate.",
        "chart-note",
      ),
    );
  }
  function findings(root, data, all, k) {
    const ranking = data.config?.experiment_type === "reranker",
      baseline = all[0];
    const hero = el("div", undefined, "comparison-insight");
    hero.id = "comparison-findings";
    const synthetic =
      ["demo", "memory_checks"].includes(data.config?.dataset) ||
      /synthetic/i.test(data.dataset_label || "");
    hero.append(
      el(
        "span",
        synthetic
          ? "Sample dataset · exploratory results"
          : "Evaluation summary",
        "chart-eyebrow",
      ),
    );
    hero.append(el("h2", ranking ? "Retrieval quality" : "Answer quality"));
    const count = data.case_ids?.length || 0,
      answered = all.filter((r) => r.summary?.samples);
    hero.append(
      el(
        "p",
        `${count} questions · ${all.length} configurations · ${data.config?.repeats || 1} repeat(s) · top ${k} evidence records per answer.`,
      ),
    );
    if (ranking && baseline) {
      const valid = answered.filter((r) => A.finite(r.summary?.ndcg_at_k));
      const max = Math.max(...valid.map((r) => r.summary.ndcg_at_k));
      const leaders = valid.filter(
        (r) => Math.abs(r.summary.ndcg_at_k - max) < 1e-9,
      );
      if (leaders.length) {
        const pair = A.pairedMetric(leaders[0], baseline, "ndcg_at_k");
        hero.append(
          el(
            "p",
            `Highest nDCG@${k}: ${A.fmt(max)} · baseline: ${A.fmt(baseline.summary?.ndcg_at_k)}. ${leaders.length} configuration${leaders.length === 1 ? "" : "s"} at the highest score. ${A.runName(leaders[0], data)} changed nDCG by ${pair.delta >= 0 ? "+" : ""}${A.fmt(pair.delta)} on ${pair.n} matched questions.`,
            "finding-main",
          ),
        );
      }
    }
    if (answered.length && answered.every((r) => r.summary.accuracy === 1))
      hero.append(
        el(
          "p",
          `Every scored answer passed the ${data.accuracy_label ? "lexical check" : "configured scorer"}. This ceiling effect cannot distinguish factual answer quality.`,
          "notice",
        ),
      );
    hero.append(
      el(
        "p",
        synthetic
          ? "Validate these findings on a held-out dataset from your application before selecting a production model."
          : "Results apply to the selected dataset. Compare matched questions and inspect regressions before selecting a model.",
      ),
    );
    const workflow = el(
      "p",
      `${data.config?.candidate_pool_size || "?"} initial candidates → ${ranking ? "selected reranker" : "fixed evidence selection"} → top ${k} → ${ranking ? A.modelName(baseline?.reader) : "selected answer model"} → ${data.accuracy_label || "configured scorer"}`,
      "experiment-flow",
    );
    const design = el("details");
    design.append(el("summary", "Experiment design"), workflow);
    hero.append(design);
    root.append(hero);
  }
  function stageTable(root, data, runs) {
    const details = el("details", undefined, "stage-breakdown");
    details.open = false;
    details.append(
      el("summary", "Stage breakdown · mean time and API cost per question"),
    );
    const wrap = el("div", undefined, "compare-table-wrap"),
      table = el("table"),
      head = el("tr");
    [
      "Configuration",
      "Initial retrieval mean",
      "Reranker mean / API $",
      "Answer mean / API $",
      "Evaluation API total",
      "One-time model load",
    ].forEach((t) => head.append(el("th", t)));
    table.append(head);
    runs.forEach((r) => {
      const tr = el("tr"),
        rank = A.economics(r, "reranker"),
        reader = A.economics(r, "reader");
      const first = A.mean(
        (r.cases || []).map((c) =>
          A.finite(c.retrieval_seconds) &&
          A.finite(c.retrieval_timing?.reranker_seconds)
            ? Math.max(
                0,
                c.retrieval_seconds - c.retrieval_timing.reranker_seconds,
              )
            : null,
        ),
      );
      [
        A.runName(r, data),
        data.config?.candidate_snapshot_path
          ? "Excluded · frozen snapshot"
          : A.fmt(first) + "s",
        `${A.fmt(rank.mean)}s / ${A.money(rank.perQuestion)}`,
        `${A.fmt(reader.mean)}s / ${A.money(reader.perQuestion)}`,
        A.money(r.summary?.evaluation_cost_usd),
        A.fmt(r.model_load_seconds) + "s",
      ].forEach((v) => tr.append(el("td", v)));
      table.append(tr);
    });
    wrap.append(table);
    details.append(
      wrap,
      el(
        "p",
        "Means can be compared by stage; p50/p95 are calculated from per-question totals, never by adding stage percentiles. API estimates exclude local hardware. Evaluation and one-time loading are outside serving charts.",
        "muted",
      ),
    );
    root.append(details);
  }
  function render(data, onSelect) {
    const root = document.getElementById("comparison-visuals");
    if (!root) return;
    if (experiment !== data.id) {
      experiment = data.id;
      hidden = new Set();
      scope =
        data.config?.experiment_type === "reranker" ? "reranker" : "pipeline";
      metric = "ndcg_at_k";
    }
    root.replaceChildren();
    const all = (data.runs || []).map((r, i) => ({
      ...r,
      color: colors[i % colors.length],
      short: `C${i + 1}`,
    }));
    if (!all.some((r) => r.summary?.samples)) {
      root.append(el("p", "Charts appear as scored cases arrive.", "muted"));
      return;
    }
    const ranking = data.config?.experiment_type === "reranker",
      k = data.config?.top_k || "?";
    findings(root, data, all, k);
    const controls = el("div", undefined, "chart-controls");
    const group = el("div", undefined, "scope-buttons");
    group.setAttribute("role", "group");
    group.setAttribute("aria-label", "Measurement stage");
    Object.entries(scopeNames).forEach(([value, name]) => {
      const b = el("button", name, "btn btn-secondary");
      b.type = "button";
      b.dataset.scope = value;
      b.setAttribute("aria-pressed", String(scope === value));
      b.onclick = () => {
        scope = value;
        render(data, onSelect);
      };
      group.append(b);
    });
    controls.append(group);
    if (ranking) {
      const label = el("label", "Ranking metric"),
        select = el("select");
      select.id = "ranking-metric";
      Object.entries(metrics).forEach(([key, name]) => {
        const o = el("option", `${name}@${k}`);
        o.value = key;
        select.append(o);
      });
      select.value = metric;
      select.onchange = () => {
        metric = select.value;
        render(data, onSelect);
      };
      label.append(select);
      controls.append(label);
    }
    root.append(controls);
    root.append(
      el(
        "p",
        `${scopeNames[scope]} selected. ${data.config?.candidate_snapshot_path ? "Original Oracle/Voyage retrieval is excluded from this replay." : "Full pipeline includes shared initial retrieval, reranking and answer generation."} Judge/oracle charges and model loading are separate. Switching stages changes costs and timings, not quality.`,
        "notice",
      ),
    );
    const legend = el("div", undefined, "chart-legend");
    all.forEach((r) => {
      const l = el("label", undefined, "chart-series"),
        input = el("input");
      input.type = "checkbox";
      input.checked = !hidden.has(r.id);
      input.setAttribute("aria-label", `Show ${A.runName(r, data)} in charts`);
      input.onchange = () => {
        input.checked ? hidden.delete(r.id) : hidden.add(r.id);
        render(data, onSelect);
      };
      const swatch = el("span", undefined, "chart-swatch");
      swatch.style.background = r.color;
      l.append(input, swatch, el("span", `${r.short} · ${A.runName(r, data)}`));
      legend.append(l);
    });
    root.append(legend);
    const runs = all.filter((r) => !hidden.has(r.id));
    if (!runs.length) {
      root.append(
        el("p", "Select a configuration to display its measurements."),
      );
      return;
    }
    const grid = el("div", undefined, "comparison-chart-grid");
    root.append(grid);
    metricBars(
      panel(
        grid,
        ranking
          ? `${metrics[metric]}@${k} · evidence quality`
          : data.accuracy_label || "Answer accuracy",
        ranking
          ? "How well did this method place labeled evidence near the top?"
          : "Answer checks and sample uncertainty; review factual support separately.",
      ),
      runs,
      ranking ? metric : "accuracy",
      k,
      data,
      onSelect,
    );
    costBars(
      panel(
        grid,
        `${scopeNames[scope]} · API cost / question`,
        "Compare the component you are considering replacing.",
      ),
      runs,
      data,
      onSelect,
    );
    lines(
      panel(
        grid,
        `${scopeNames[scope]} · latency`,
        "Measured client wall time for each question.",
      ),
      runs,
      data,
      "latency",
      onSelect,
    );
    lines(
      panel(
        grid,
        `${scopeNames[scope]} · cumulative API spend`,
        "API estimates in dataset order.",
      ),
      runs,
      data,
      "cost",
      onSelect,
    );
    stageTable(root, data, runs);
  }
  window.MemorizzComparisonCharts = { render };
})();

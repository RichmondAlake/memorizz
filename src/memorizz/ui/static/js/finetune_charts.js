// Line charts for training curves, shared by the Fine-tuning page and the
// Model registry: several series, a legend, hover values, epoch markers and
// reference lines (the score before training).
(() => {
  const NS = "http://www.w3.org/2000/svg";
  const svg = (tag, attrs = {}) => {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) node.setAttribute(k, v);
    return node;
  };
  const fmt = (v, asPct) => {
    if (v === null || v === undefined || Number.isNaN(v)) return "–";
    if (asPct) return `${(v * 100).toFixed(1)}%`;
    if (Number.isInteger(v)) return String(v);
    if (v !== 0 && Math.abs(v) < 0.01) return v.toExponential(1);
    return Math.abs(v) >= 100 ? v.toFixed(0) : String(Number(v.toPrecision(3)));
  };

  // Exponential moving average, for noisy per-step losses.
  function smooth(points, alpha = 0.1) {
    let value = null;
    return points.map(([x, y]) => {
      value = value === null ? y : alpha * y + (1 - alpha) * value;
      return [x, value];
    });
  }

  /**
   * series: [{ name, points: [[x, y]], tone: 0-5, faint, dots, dashed }]
   * opts:   { width, height, yMin, yMax, asPct, logY, xLabel, yLabel, markers: [{ x, label }], refs: [{ y, label, tone }] }
   * logY plots on a log scale (losses that span orders of magnitude).
   */
  function line(series, opts = {}) {
    const width = opts.width || 420, height = opts.height || 210;
    const pad = { l: 52, r: 14, t: 14, b: 34 };
    const wrap = document.createElement("figure");
    wrap.className = "ch";
    const chart = svg("svg", { viewBox: `0 0 ${width} ${height}`, class: "ch-svg", role: "img", "aria-label": opts.yLabel || "chart" });
    const logY = !!opts.logY;
    const all = series.flatMap((s) => s.points).concat((opts.refs || []).map((r) => [null, r.y]));
    const ys = all.map((p) => p[1]).filter((v) => v !== null && v !== undefined && !Number.isNaN(v) && (!logY || v > 0));
    const xs = series.flatMap((s) => s.points.map((p) => p[0]));
    if (!ys.length || !xs.length) {
      const t = svg("text", { x: pad.l, y: height / 2, class: "ch-empty" });
      t.textContent = opts.empty || "Nothing recorded.";
      chart.append(t);
      wrap.append(chart);
      return wrap;
    }
    const x0 = Math.min(...xs), x1 = Math.max(...xs, x0 + 1e-9);
    const t = (v) => (logY ? Math.log10(Math.max(v, 1e-12)) : v);
    let y0 = t(opts.yMin ?? Math.min(...ys)), y1 = t(opts.yMax ?? Math.max(...ys));
    if (y1 - y0 < 1e-9) { y0 -= 0.5; y1 += 0.5; }
    const X = (x) => pad.l + ((x - x0) / (x1 - x0 || 1)) * (width - pad.l - pad.r);
    const Y = (y) => height - pad.b - ((t(y) - y0) / (y1 - y0)) * (height - pad.t - pad.b);
    for (let i = 0; i <= 4; i++) {
      const v = logY ? 10 ** (y0 + ((y1 - y0) * i) / 4) : y0 + ((y1 - y0) * i) / 4;
      chart.append(svg("line", { class: "ch-grid", x1: pad.l, x2: width - pad.r, y1: Y(v), y2: Y(v) }));
      const t = svg("text", { x: pad.l - 6, y: Y(v) + 4, "text-anchor": "end", class: "ch-tick" });
      t.textContent = fmt(v, opts.asPct);
      chart.append(t);
    }
    chart.append(svg("line", { class: "ch-axis", x1: pad.l, x2: width - pad.r, y1: height - pad.b, y2: height - pad.b }));
    for (const [i, v] of [[0, x0], [1, x1]]) {
      const t = svg("text", { x: i ? width - pad.r : pad.l, y: height - pad.b + 16, "text-anchor": i ? "end" : "start", class: "ch-tick" });
      t.textContent = opts.xFormat ? opts.xFormat(v) : fmt(v);
      chart.append(t);
    }
    if (opts.xLabel) {
      const t = svg("text", { x: (pad.l + width - pad.r) / 2, y: height - 4, "text-anchor": "middle", class: "ch-tick" });
      t.textContent = opts.xLabel;
      chart.append(t);
    }
    for (const m of opts.markers || []) {
      if (m.x <= x0 || m.x > x1) continue;
      chart.append(svg("line", { class: "ch-marker", x1: X(m.x), x2: X(m.x), y1: pad.t, y2: height - pad.b }));
      if (m.label) {
        const t = svg("text", { x: X(m.x) - 3, y: pad.t + 9, "text-anchor": "end", class: "ch-tick" });
        t.textContent = m.label;
        chart.append(t);
      }
    }
    for (const r of opts.refs || []) {
      const ref = svg("line", { class: `ch-ref tone-${r.tone ?? 5}`, x1: pad.l, x2: width - pad.r, y1: Y(r.y), y2: Y(r.y) });
      const title = svg("title");
      title.textContent = `${r.label}: ${fmt(r.y, opts.asPct)}`;
      ref.append(title);
      chart.append(ref);
    }
    for (const s of series) {
      if (!s.points.length) continue;
      const path = svg("polyline", {
        class: `ch-line tone-${s.tone ?? 0}${s.faint ? " ch-faint" : ""}${s.dashed ? " ch-dashed" : ""}`,
        points: s.points.filter(([, y]) => !logY || y > 0).map(([x, y]) => `${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join(" "),
      });
      chart.append(path);
      if (s.dots) {
        for (const [x, y] of s.points) {
          const dot = svg("circle", { class: `ch-dot tone-${s.tone ?? 0}`, cx: X(x), cy: Y(y), r: 3.5 });
          const title = svg("title");
          title.textContent = `${s.name}: ${fmt(y, opts.asPct)} at ${opts.xFormat ? opts.xFormat(x) : fmt(x)}`;
          dot.append(title);
          chart.append(dot);
        }
      }
    }
    wrap.append(chart);
    const legendItems = series.filter((s) => s.name && !s.hideInLegend).map((s) => [s.name, s.tone ?? 0, s.faint]).concat((opts.refs || []).map((r) => [`${r.label} (${fmt(r.y, opts.asPct)})`, r.tone ?? 5, false, true]));
    if (legendItems.length > 1 || opts.legend) {
      const legend = document.createElement("figcaption");
      legend.className = "ch-legend";
      for (const [name, tone, faint, ref] of legendItems) {
        const item = document.createElement("span");
        item.className = `ch-key tone-${tone}${faint ? " ch-faint" : ""}${ref ? " ch-key-ref" : ""}`;
        item.textContent = name;
        legend.append(item);
      }
      wrap.append(legend);
    }
    return wrap;
  }

  window.MemorizzCharts = { line, smooth };
})();

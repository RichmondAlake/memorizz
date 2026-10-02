/* Read-only home-page index. Refreshing never starts inference.
   The table is a monitor grid: sortable columns, / to search, j/k to move,
   Enter to open the selected result (keys and selection via MemorizzMonitor). */
(() => {
  const $ = (id) => document.getElementById(id);
  if (!$("run-library")) return;
  const el = (tag, text, cls) => {
    const n = document.createElement(tag);
    if (text !== undefined) n.textContent = text;
    if (cls) n.className = cls;
    return n;
  };
  const active = new Set(["queued", "running", "cancelling", "canceling"]);
  const kinds = {
    agent: "Agent evaluation",
    harness: "Harness benchmark",
    reranker: "Reranking",
    reader: "Answer models",
    pipeline: "Memory pipeline",
  };
  const statusLabels = {
    completed: "Completed",
    running: "Running",
    queued: "Queued",
    completed_with_errors: "Completed with errors",
    interrupted: "Interrupted",
    failed: "Failed",
    cancelled: "Cancelled",
    canceled: "Cancelled",
    spend_limit: "Spend limit reached",
    time_limit: "Time limit reached",
  };
  const sortLabels = {
    newest: "Newest first",
    oldest: "Oldest first",
    cost: "Highest API cost",
  };
  const columnLabels = {
    name: "Experiment",
    kind: "Type",
    status: "Status",
    progress: "Progress",
    quality: "Quality",
    cost: "API cost",
    created: "Started",
  };
  let rows = [],
    page = 0,
    timer,
    busy = false,
    loaded = false,
    // Column sort; the Sort by menu maps onto the same state.
    sort = { key: "created", ascending: false };
  function group(status) {
    return active.has(status)
      ? "active"
      : status === "completed"
        ? "completed"
        : "attention";
  }
  function tone(status) {
    if (active.has(status)) return "active";
    if (status === "completed") return "good";
    return status === "failed" || status === "interrupted" ? "bad" : "warn";
  }
  const usd = window.MemorizzFormat.usd; // app.js
  function age(seconds) {
    if (!seconds) return "Unknown";
    const delta = Date.now() / 1000 - seconds;
    const size = Math.abs(delta);
    for (const [unit, span] of [
      ["d", 86400],
      ["h", 3600],
      ["m", 60],
    ]) {
      if (size >= span) {
        const amount = Math.floor(size / span) + unit;
        return delta < 0 ? "in " + amount : amount + " ago";
      }
    }
    return delta < 0 ? "soon" : "just now";
  }
  function progressRatio(r) {
    return r.completed != null && r.planned > 0 ? r.completed / r.planned : -1;
  }
  const sortValue = {
    name: (r) => (r.name || "").toLocaleLowerCase(),
    kind: (r) => kinds[r.kind] || r.kind || "",
    status: (r) => statusLabels[r.status] || r.status || "",
    progress: progressRatio,
    quality: (r) => (r.quality_max == null ? -1 : r.quality_max),
    cost: (r) => (r.cost_usd == null ? -1 : r.cost_usd),
    created: (r) => r.created_at || 0,
  };
  function compare(a, b) {
    const value = sortValue[sort.key] || sortValue.created;
    const x = value(a),
      y = value(b);
    const order = typeof x === "string" ? x.localeCompare(y) : x - y;
    return sort.ascending ? order : -order;
  }
  function syncSortControls() {
    const menu = $("run-sort");
    const preset =
      sort.key === "created"
        ? sort.ascending
          ? "oldest"
          : "newest"
        : sort.key === "cost" && !sort.ascending
          ? "cost"
          : "column";
    menu.value = preset;
    document
      .querySelectorAll("#run-library-table thead th[aria-sort]")
      .forEach((th) => {
        const button = th.querySelector("[data-run-sort]");
        th.setAttribute(
          "aria-sort",
          button && button.dataset.runSort === sort.key
            ? sort.ascending
              ? "ascending"
              : "descending"
            : "none",
        );
      });
  }
  function sortLabel() {
    const preset = $("run-sort").value;
    if (sortLabels[preset]) return sortLabels[preset];
    return `${columnLabels[sort.key] || "Column"} ${sort.ascending ? "ascending" : "descending"}`;
  }
  function cell(text, cls) {
    return el("td", text, cls);
  }
  function buildRow(r) {
    const tr = el("tr");
    tr.dataset.key = r.id;
    tr.dataset.runId = r.id;
    tr.dataset.kind = r.kind;
    tr.dataset.search = [r.name, r.id, ...r.models].join(" ").toLowerCase();
    tr.tabIndex = -1;

    const name = el("td", undefined, "eval-col-name");
    const link = el(
      "a",
      r.name === "System One lesson · Oracle / Voyage · 7 reranking methods"
        ? "Reranking comparison · 7 methods"
        : r.name,
      "run-name",
    );
    link.href = r.url;
    link.title = r.name;
    name.append(link);
    if (/synthetic|memory_checks|demo/i.test(r.scope))
      name.append(el("span", "Sample dataset", "dataset-tag"));
    tr.append(name);

    const models = cell(undefined, "eval-col-models");
    const modelSummary = el(
      "span",
      r.models.slice(0, 2).join(" · ") +
        (r.models.length > 2 ? ` · +${r.models.length - 2} more` : ""),
      "run-subtitle",
    );
    modelSummary.title = r.models.join(" · ");
    models.append(modelSummary);
    tr.append(models);

    tr.append(cell(kinds[r.kind] || r.kind, "eval-col-type run-type"));

    const status = cell();
    const statusText = statusLabels[r.status] || r.status.replaceAll("_", " ");
    const badge = el(
      "span",
      undefined,
      `eval-status eval-status--${tone(r.status)} run-status-${group(r.status)}`,
    );
    badge.title = statusText;
    badge.append(el("span", statusText, "eval-status-label"));
    status.append(badge);
    tr.append(status);

    const progress = cell(undefined, "num eval-col-progress");
    progress.title = `${r.completed ?? "Unknown"} of ${r.planned ?? "unknown"} ${r.progress_unit}`;
    if (r.completed != null && r.planned > 0) {
      const bar = el("progress");
      bar.value = r.completed;
      bar.max = r.planned;
      bar.setAttribute(
        "aria-label",
        `${r.completed} of ${r.planned} ${r.progress_unit}`,
      );
      progress.append(bar);
    }
    progress.append(
      el("span", `${r.completed ?? "—"}/${r.planned ?? "—"}`, "eval-count"),
    );
    tr.append(progress);

    const quality = cell(undefined, "num eval-col-quality");
    const format = (v) =>
      r.quality_label.startsWith("nDCG")
        ? v.toFixed(3)
        : (v * 100).toFixed(1) + "%";
    quality.title =
      r.quality_label +
      (r.quality_min !== r.quality_max ? " · range across configurations" : "");
    quality.append(
      el(
        "span",
        r.quality_min == null
          ? "Not measured"
          : format(r.quality_min) +
              (r.quality_max !== r.quality_min
                ? "–" + format(r.quality_max)
                : ""),
        r.quality_min == null ? "eval-muted" : "",
      ),
    );
    tr.append(quality);

    const cost = cell(usd(r.cost_usd), "num");
    if (r.cost_usd == null) cost.classList.add("eval-muted");
    else cost.title = "Exact $" + r.cost_usd;
    tr.append(cost);

    const date = r.created_at ? new Date(r.created_at * 1000) : null;
    const when = cell(date ? age(r.created_at) : "Unknown", "num eval-col-started");
    if (date)
      when.title = date.toLocaleString(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      });
    tr.append(when);

    const action = cell(undefined, "eval-col-result");
    const open = el(
      "a",
      active.has(r.status) ? "View progress" : "Open result",
      "btn btn-sm",
    );
    open.href = r.url;
    open.setAttribute("aria-label", `Open ${r.name}`);
    action.append(open);
    tr.append(action);
    return tr;
  }
  function renderStats() {
    const known = rows.filter((r) => r.cost_usd != null);
    const spend = known.reduce((sum, r) => sum + r.cost_usd, 0);
    const counts = { active: 0, completed: 0, attention: 0 };
    rows.forEach((r) => (counts[group(r.status)] += 1));
    const newest = rows.reduce((max, r) => Math.max(max, r.created_at || 0), 0);
    const items = [
      ["Evaluations", String(rows.length), ""],
      ["In progress", String(counts.active), ""],
      ["Completed", String(counts.completed), ""],
      ["Needs attention", String(counts.attention), counts.attention ? "is-warn" : "is-good"],
      [
        "Known API cost",
        known.length ? usd(spend) : "Unknown",
        "",
        known.length
          ? `Exact $${spend.toPrecision(12).replace(/\.?0+$/, "")} across ${known.length} run${known.length === 1 ? "" : "s"}` +
            (rows.length > known.length
              ? `; ${rows.length - known.length} without a known cost`
              : "")
          : "",
      ],
      ["Latest run", newest ? age(newest) : "—", "", newest ? new Date(newest * 1000).toLocaleString() : ""],
    ];
    const stats = $("run-library-stats");
    stats.removeAttribute("aria-busy");
    stats.replaceChildren(
      ...items.map(([label, value, cls, title]) => {
        const item = el("div");
        item.setAttribute("role", "listitem");
        if (label === "Latest run") item.className = "eval-tape-text";
        const strong = el("strong", value, cls || undefined);
        if (title) strong.title = title;
        item.append(el("span", label), strong);
        return item;
      }),
    );
  }
  let monitor = null;
  function render() {
    const search = $("run-search").value.toLocaleLowerCase().trim(),
      kind = $("run-kind").value,
      status = $("run-status").value;
    const matches = rows.filter(
      (r) =>
        (kind === "all" || r.kind === kind) &&
        (status === "all" || group(r.status) === status) &&
        (!search ||
          [r.name, r.id, ...r.models]
            .join(" ")
            .toLocaleLowerCase()
            .includes(search)),
    );
    matches.sort(compare);
    page = Math.min(page, Math.max(0, Math.ceil(matches.length / 20) - 1));
    const slice = matches.slice(page * 20, page * 20 + 20);
    const previous = monitor && monitor.selected();
    const previousKey = previous ? previous.dataset.key : null;
    $("run-library-rows").replaceChildren(...slice.map(buildRow));
    if (!matches.length) {
      const tr = el("tr"),
        td = el(
          "td",
          rows.length
            ? "No runs match these filters. Clear the search or choose another run type."
            : "No runs yet. Start a model comparison above or an agent evaluation below.",
          "eval-empty",
        );
      td.colSpan = 9;
      tr.append(td);
      $("run-library-rows").append(tr);
    }
    $("run-library-message").textContent =
      `${matches.length} evaluation${matches.length === 1 ? "" : "s"}${matches.length !== rows.length ? " of " + rows.length : ""} · ${sortLabel()}`;
    $("run-page").textContent = matches.length
      ? `${page * 20 + 1}–${Math.min(matches.length, page * 20 + 20)} of ${matches.length}`
      : "0 runs";
    $("run-prev").disabled = page === 0;
    $("run-next").disabled = (page + 1) * 20 >= matches.length;
    renderStats();
    if (monitor) {
      const next =
        Array.from($("run-library-rows").rows).find(
          (row) => row.dataset.key === previousKey,
        ) || $("run-library-rows").rows[0];
      if (next && next.dataset.key !== undefined) monitor.select(next);
    }
  }
  async function refresh() {
    if (busy) return;
    busy = true;
    $("refresh-run-library").disabled = true;
    try {
      const res = await fetch("/evalground/run-library");
      const data = await res.json();
      if (!res.ok) throw Error(data.detail || "Could not load run history");
      rows = data.runs;
      loaded = true;
      render();
      $("run-library-scope").textContent =
        data.scope +
        (data.warnings.length ? " " + data.warnings.join(" ") : "");
    } catch (e) {
      $("run-library-message").textContent =
        (loaded ? "Showing the last loaded history. " : "") + e.message;
    } finally {
      busy = false;
      $("refresh-run-library").disabled = false;
      clearTimeout(timer);
      timer = setTimeout(() => {
        if (!document.hidden) refresh();
        else timer = setTimeout(refresh, 15000);
      }, 15000);
    }
  }
  ["run-search", "run-kind", "run-status", "run-sort"].forEach((id) =>
    $(id).addEventListener(id === "run-search" ? "input" : "change", () => {
      page = 0;
      if (id === "run-sort") {
        const preset = $("run-sort").value;
        if (preset === "cost") sort = { key: "cost", ascending: false };
        else if (preset === "oldest") sort = { key: "created", ascending: true };
        else if (preset === "newest") sort = { key: "created", ascending: false };
        syncSortControls();
      }
      render();
    }),
  );
  // Numbers sort high to low first, words A to Z, as on the fleet grid.
  document
    .querySelectorAll("#run-library-table thead [data-run-sort]")
    .forEach((button) =>
      button.addEventListener("click", () => {
        const key = button.dataset.runSort;
        const text = button.dataset.sortType === "text";
        sort =
          sort.key === key
            ? { key, ascending: !sort.ascending }
            : { key, ascending: text };
        page = 0;
        syncSortControls();
        render();
      }),
    );
  $("run-prev").onclick = () => {
    page--;
    render();
  };
  $("run-next").onclick = () => {
    page++;
    render();
  };
  $("refresh-run-library").onclick = refresh;
  function revealSection(id) {
    const target = document.getElementById(id);
    if (target?.tagName === "DETAILS") {
      target.open = true;
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }
  document
    .querySelectorAll("[data-open-section]")
    .forEach((a) =>
      a.addEventListener("click", () => revealSection(a.dataset.openSection)),
    );
  window.addEventListener("hashchange", () =>
    revealSection(location.hash.slice(1)),
  );
  if (location.hash) revealSection(location.hash.slice(1));
  if (window.MemorizzMonitor) {
    monitor = window.MemorizzMonitor.init({
      table: "run-library-table",
      keys: {
        Enter: (row) => {
          const link = row.querySelector("a.run-name");
          if (link) window.location.href = link.href;
        },
      },
    });
  }
  refresh();
})();

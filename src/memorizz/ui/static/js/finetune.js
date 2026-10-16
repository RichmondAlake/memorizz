(() => {
  const app = document.getElementById("ft-app");
  if (!app) return;
  const readOnly = app.dataset.readOnly === "true";
  const S = {
    overview: null,
    datasetId: app.dataset.dataset || "",
    dataset: null,
    tab: "data",
    filter: { status: "", split: "" },
    examples: [],
    total: 0,
    limit: 0,
    selected: new Set(),
    runs: [],
    runId: "",
    events: [],
    after: 0,
    timers: {},
    preselect: null,
    keepInView: false,
  };

  // -- helpers -------------------------------------------------------------
  const h = (tag, attrs = {}, ...kids) => {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (value === null || value === undefined || value === false) continue;
      if (key === "class") node.className = value;
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else if (key === "value") node.value = value;
      else node.setAttribute(key, value === true ? "" : value);
    }
    for (const kid of kids.flat(Infinity)) {
      if (kid === null || kid === undefined || kid === false) continue;
      node.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    }
    return node;
  };
  // replaceChildren() would print "null" for a skipped part; drop those.
  const fill = (node, ...kids) => node.replaceChildren(...kids.flat(Infinity).filter((k) => k !== null && k !== undefined && k !== false));
  const api = async (path, opts = {}) => {
    const init = { ...opts };
    if (opts.body !== undefined && typeof opts.body !== "string") {
      init.body = JSON.stringify(opts.body);
      init.headers = { "Content-Type": "application/json" };
    }
    const response = await fetch(path, init);
    const type = response.headers.get("content-type") || "";
    const data = type.includes("json") ? await response.json() : await response.text();
    if (!response.ok) {
      const detail = data && data.detail;
      throw new Error(typeof detail === "string" ? detail : `Request failed (${response.status})`);
    }
    return data;
  };
  const fmt = (value, digits = 3) => {
    if (value === null || value === undefined || value === "") return "–";
    if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(digits);
    return String(value);
  };
  const pct = (v) => (v === null || v === undefined || Number.isNaN(v) ? "–" : `${(v * 100).toFixed(1)}%`);
  const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
  const say = (node, text, error = false) => {
    node.textContent = text;
    node.classList.toggle("ft-down", !!error);
  };
  const busy = async (button, work) => {
    button.disabled = true;
    try {
      return await work();
    } finally {
      button.disabled = readOnly;
    }
  };
  const download = async (url, fallback) => {
    const response = await fetch(url);
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.detail || "Download failed");
    }
    const blob = await response.blob();
    const link = h("a", { href: URL.createObjectURL(blob) });
    link.download = response.headers.get("Content-Disposition")?.match(/filename="([^"]+)"/)?.[1] || fallback;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  };
  // Textareas grow with what they hold.
  const autosize = (el) => {
    const fit = () => {
      el.style.height = "auto";
      el.style.height = `${el.scrollHeight + 2}px`;
    };
    el.addEventListener("input", fit);
    requestAnimationFrame(fit);
    return el;
  };
  const narrow = () => window.matchMedia("(max-width: 960px)").matches;
  const KIND = { decision: "Decision model", embedding: "Embedding model", reranker: "Reranker" };
  const DEVICE = { mps: "Apple GPU (MPS)", cuda: "NVIDIA GPU (CUDA)", cpu: "CPU" };
  const STATUS = { pending: "to review", accepted: "accepted", rejected: "rejected" };
  const statusPill = (status) => {
    const tone = { succeeded: "success", accepted: "success", running: "info", queued: "info", pending: "warning", failed: "danger", rejected: "danger", canceled: "warning", interrupted: "danger" }[status] || "info";
    return h("span", { class: `pill pill--${tone}` }, STATUS[status] || status);
  };
  // The job a dataset trains a model for; datasets are grouped by it.
  const JOBS = [
    { key: "memory_routing", title: "Memory routing", match: (d) => d.kind === "decision" && d.task === "memory_routing" },
    { key: "embedding", title: "Memory embeddings", match: (d) => d.kind === "embedding" },
    { key: "reranking", title: "Memory reranking", match: (d) => d.kind === "reranker" || (d.kind === "decision" && d.task === "relevance") },
    { key: "other", title: "Other decisions", match: () => true },
  ];
  const jobOf = (d) => JOBS.find((j) => j.match(d)).key;
  const sameJob = (a, b) => a.kind === b.kind && jobOf(a) === jobOf(b);
  const primaryMetric = (d) => (d.kind === "decision" && d.task !== "relevance" ? "accuracy" : "ndcg@10");
  const METRICS = {
    accuracy: { words: "Right answers", short: "accuracy", up: true, pct: true },
    macro_f1: { words: "Right answers, each option weighted equally", short: "macro-F1", up: true, pct: true },
    nll: { words: "Surprise at the right answer", short: "log loss", up: false, pct: false },
    brier: { words: "Probability error", short: "Brier score", up: false, pct: false },
    ece: { words: "Gap between confidence and accuracy", short: "calibration error", up: false, pct: true },
    order_flip_rate: { words: "Answers that change when options are reordered", short: "order flips", up: false, pct: true },
    "mrr@10": { words: "How high the right memory ranks", short: "MRR@10", up: true, pct: true },
    "ndcg@10": { words: "Right memory near the top", short: "nDCG@10", up: true, pct: true },
    "recall@1": { words: "Right memory ranked first", short: "recall@1", up: true, pct: true },
    "recall@5": { words: "Right memory in the top 5", short: "recall@5", up: true, pct: true },
    "recall@10": { words: "Right memory in the top 10", short: "recall@10", up: true, pct: true },
    map: { words: "Right memories ranked high", short: "MAP", up: true, pct: true },
  };
  const metricWords = (key, job) => (key === "accuracy" && job === "memory_routing" ? "Right store chosen" : (METRICS[key] || { words: key }).words);
  const metricValue = (key, v) => (METRICS[key] && !METRICS[key].pct ? fmt(v) : pct(v));
  const SPLIT_WORDS = { test: "test", validation: "validation", held_out: "test and validation", all: "" };
  const taskTitle = (d) => ((S.overview.tasks || []).find((t) => t.kind === d.kind && t.task === d.task) || { title: d.task.replace(/_/g, " ") }).title;
  const question = () => {
    const questions = (S.dataset && S.dataset.spec && S.dataset.spec.questions) || {};
    const [qid, q] = Object.entries(questions)[0] || [];
    return qid ? { qid, ...q } : null;
  };
  const optionKeys = (q) => {
    if (!q) return [];
    if (q.type === "choice") return Object.keys(q.criteria || {});
    if (q.type === "noul") return ["false", "true"];
    return (q.criteria || []).map((_, i) => String(i));
  };
  const goldKey = (q, label) => {
    const answer = label && label[q.qid];
    if (!answer) return "";
    if (q.type === "choice") return String(answer.choice);
    if (q.type === "noul") return answer.noul === true || Number(answer.noul) >= 0.5 ? "true" : "false";
    return String(answer.level);
  };
  // Long text shows a few lines, with a toggle for the rest.
  function clampText(text, lines = 3) {
    const body = h("div", { class: "ft-clamp", style: `--lines:${lines}` }, String(text ?? ""));
    const more = h("button", { type: "button", class: "ft-more", hidden: true, onclick: () => { more.textContent = body.classList.toggle("open") ? "Show less" : "Show all"; } }, "Show all");
    requestAnimationFrame(() => { more.hidden = body.scrollHeight <= body.clientHeight + 2; });
    return h("div", { class: "ft-clamp-box" }, body, more);
  }
  const exampleCell = (e) => {
    const input = e.input || {};
    if (input.state !== undefined) return clampText(typeof input.state === "string" ? input.state : JSON.stringify(input.state, null, 1));
    const negatives = input.negatives || [];
    return h(
      "div",
      { class: "ft-pair" },
      h("div", {}, h("span", { class: "ft-k" }, "Question"), clampText(input.query, 2)),
      h("div", {}, h("span", { class: "ft-k" }, "Memory that answers it"), clampText(input.positive, 2)),
      negatives.length ? h("div", {}, h("span", { class: "ft-k" }, `Look-alikes that don't (${negatives.length})`), clampText(negatives.join("\n"), 2)) : null,
    );
  };

  // -- polling a background job ------------------------------------------------
  async function runJob(path, body, button, note, message, done) {
    button.disabled = true;
    say(note, message);
    try {
      const { job_id } = await api(path, { method: "POST", body });
      const poll = async () => {
        const state = await api(`/api/finetune/jobs/${job_id}`);
        if (state.status === "running") {
          say(note, state.message);
          return setTimeout(poll, 1200);
        }
        button.disabled = readOnly;
        if (state.status === "failed") return say(note, state.error, true);
        say(note, "Done.");
        await done(state);
      };
      poll();
    } catch (error) {
      button.disabled = readOnly;
      say(note, error.message, true);
    }
  }

  // -- comparing models: the same rows on the cards and in Results ------------
  function compareRows(c) {
    const e = c.evidence;
    const rows = [{ label: `Fine-tuned ${c.version}`, sub: c.short_base, value: e.fine_tuned, main: true }];
    if (e.in_use) rows.push({ label: e.in_use.when === "in use now" ? "In use now" : "Before the swap", sub: e.in_use.model, value: e.in_use.value });
    if (e.base !== null && e.base !== undefined) {
      rows.push(e.base_kind === "parent" ? { label: `Started from ${e.base_label}`, sub: "before this training", value: e.base } : { label: "Untrained base", sub: String(e.base_label || "").replace(/, untrained$/, ""), value: e.base });
    }
    if (e.reference) rows.push({ label: "Jev, zero-shot", sub: "TypeSafe", value: e.reference.value });
    return h(
      "div",
      { class: "ft-compare", role: "table", "aria-label": "Scores" },
      rows.map((r) =>
        h(
          "div",
          { class: `ft-crow${r.main ? " ft-crow--main" : ""}`, role: "row" },
          h("span", { class: "ft-clabel", role: "rowheader" }, h("span", {}, r.label), r.sub ? h("small", { title: r.sub }, r.sub) : null),
          h("span", { class: "ft-ctrack", "aria-hidden": "true" }, h("span", { class: "ft-cfill", style: `width:${Math.max(0, Math.min(1, r.value ?? 0)) * 100}%` })),
          h("span", { class: "ft-cval", role: "cell" }, metricValue(e.metric, r.value)),
        ),
      ),
    );
  }

  function caption(c, job) {
    const e = c.evidence;
    const unit = c.kind === "decision" ? "examples" : "questions";
    const split = SPLIT_WORDS[e.split] ?? e.split;
    const parts = [`${metricWords(e.metric, job)} (${(METRICS[e.metric] || { short: e.metric }).short})`];
    let set = e.sets && e.sets.length > 1
      ? `${e.count ?? "?"} held-out ${unit} across ${e.sets.length} fixed sets: ${e.sets.map((x) => `“${x.name}”`).join(", ")}`
      : `${e.count ?? "?"} ${split ? split + " " : ""}${unit} from “${e.set_name}”`;
    if (e.candidates) set += `, each searched among ${e.candidates} memories`;
    if (!e.frozen) set += ", scored when it trained";
    if (e.dimensions) set += `, vectors cut to this store's ${e.dimensions} dimensions`;
    parts.push(set);
    return h("p", { class: "ft-caption" }, parts.join(" · "));
  }

  // "p = 0.004; 9 of 17 questions differ": how likely the difference is chance.
  const chanceNote = (t, metric) => {
    if (!t) return "";
    const unit = metric === "accuracy" ? "examples" : "questions";
    return `${t.p < 0.001 ? "p < 0.001" : `p = ${t.p}`}; ${t.differ} of ${t.n} ${unit} scored differently`;
  };

  function verdictLine(c) {
    const t = c.evidence && c.evidence.significance;
    if (c.verdict === "none" || c.gain_points === null || c.gain_points === undefined) return h("div", { class: "ft-verdict" }, "Nothing to compare with yet.");
    const points = `${c.gain_points > 0 ? "+" : ""}${c.gain_points} points`;
    const note = chanceNote(t, c.evidence.metric);
    if (c.verdict === "better") return h("div", { class: "ft-verdict good" }, `✓ Fine-tuning helped: ${points} over ${c.compared_with}, unlikely to be chance`, h("small", {}, note));
    if (c.verdict === "worse") return h("div", { class: "ft-verdict bad" }, `Worse than ${c.compared_with}: ${points}, unlikely to be chance`, h("small", {}, note));
    if (c.verdict === "untested") return h("div", { class: "ft-verdict unsure" }, `${points} vs ${c.compared_with}, not yet tested for chance`, h("small", {}, "Score it again to keep each example's result and test it."));
    return h("div", { class: "ft-verdict unsure" }, `${points} vs ${c.compared_with}, but that could be chance`, h("small", {}, `${note}. More test examples would settle it.`));
  }

  // How fast the version is next to what it would replace, timed together.
  function speedLine(c, done) {
    const sp = c.speed;
    const note = h("span", { class: "ft-muted ft-note", role: "status" });
    const measure = readOnly
      ? null
      : h("button", { class: "ft-link", type: "button", onclick: (ev) => runJob(`/api/finetune/runs/${c.run_id}/speed`, {}, ev.currentTarget, note, "Timing…", done) }, sp ? "Measure again" : "Measure speed");
    if (!sp) return h("div", { class: "ft-speed" }, h("span", { class: "ft-muted" }, "Speed not measured yet. "), measure, note);
    const ms = (v) => (v < 10 ? v.toFixed(1) : String(Math.round(v)));
    const mine = sp.rows.find((r) => r.role === "fine-tuned");
    const other = sp.rows.find((r) => r.role !== "fine-tuned");
    let compare = null;
    if (mine && mine.n && other && other.n) {
      const ratio = other.median_ms / mine.median_ms;
      const than = `${other.label} (${other.role})`;
      compare = ratio >= 1.1 ? h("b", { class: "ft-up" }, ` ${ratio.toFixed(1)}× faster than ${than}`) : ratio <= 0.9 ? h("b", { class: "ft-down" }, ` ${(1 / ratio).toFixed(1)}× slower than ${than}`) : h("b", {}, ` about as fast as ${than}`);
    } else if (mine && mine.n && other) {
      compare = h("b", {}, ` adds ${ms(mine.median_ms)} ms; ${other.role}: ${other.label.toLowerCase()}`);
    }
    const timed = sp.rows.filter((r) => r.n);
    const where = (d) => (d === "mps" ? "the Apple GPU" : d === "cuda" ? "the GPU" : d === "cpu" ? "the CPU" : d);
    return h(
      "div",
      { class: "ft-speed" },
      h("span", {}, `Speed ${sp.unit}:`, compare),
      h("small", {}, timed.map((r) => `${r.role === "fine-tuned" ? r.label.split(" · ")[0] : r.label}: ${ms(r.median_ms)} ms median, ${ms(r.p95_ms)} ms p95 on ${where(r.device)}`).join(" · "), ` (${timed.length ? timed[0].n : 0} calls each). Fine-tuning keeps the base model's size, so speed changes only with what runs.`),
      measure,
      note,
    );
  }

  function swapButtons(card, c, canSwapBack, note, done, { pill = true } = {}) {
    if (readOnly) return [];
    const out = [];
    const embedding = card === "embedding";
    // Every swap shows what it touches first (store, agents, other processes).
    const confirmed = (action, label, runId) => (window.MemorizzSwap ? window.MemorizzSwap.confirm({ card, action, label, runId }) : Promise.resolve(true));
    const back = () => h("button", { class: "btn btn-sm", type: "button", onclick: async (ev) => { const button = ev.currentTarget; if (await confirmed("back")) runJob(`/api/finetune/models/${card}/swap-back`, {}, button, note, embedding ? "Swapping back and re-embedding memories…" : "Swapping back…", done); } }, "Swap back");
    if (c && c.in_use) {
      if (pill) out.push(h("span", { class: "pill pill--success" }, "In use"));
      out.push(back());
    } else {
      if (c) {
        out.push(
          h(
            "button",
            {
              class: c.better ? "btn btn-primary" : "btn",
              type: "button",
              title: c.better ? null : `It did not beat ${c.compared_with}`,
              onclick: async (ev) => {
                const button = ev.currentTarget;
                const answer = await confirmed("in", c.version, c.run_id);
                if (answer) runJob(`/api/finetune/models/${card}/swap`, { run_id: c.run_id, dimensions: answer.dimensions || null }, button, note, embedding ? "Swapping in and re-embedding memories…" : "Swapping in…", done);
              },
            },
            embedding ? "Swap in (re-embeds memories)" : "Swap in",
          ),
        );
      }
      if (canSwapBack) out.push(back());
    }
    if (c && (!c.evidence.frozen || c.verdict === "untested" || (card !== "memory_routing" && !c.evidence.in_use))) {
      out.push(
        h("button", { class: "btn btn-sm", type: "button", onclick: (ev) => runJob(`/api/finetune/models/${card}/compare`, { run_id: c.run_id }, ev.target, note, "Scoring on the test examples…", done) }, card === "memory_routing" ? "Score on its test examples" : "Compare with the model in use"),
      );
    }
    return out;
  }

  // -- models in use (the simple view) ------------------------------------
  async function loadModels() {
    const grid = document.getElementById("ft-model-grid");
    let cards;
    try {
      cards = (await api("/api/finetune/models")).models;
    } catch (error) {
      fill(grid, h("p", { class: "ft-down" }, error.message));
      return;
    }
    fill(grid, cards.map(modelCard));
    // The cards sit above the dataset on a phone: if one was picked while
    // they loaded, they just pushed it down, so bring it back into view.
    if (S.keepInView && narrow()) document.getElementById("ft-main").scrollIntoView({ block: "start" });
    S.keepInView = false;
  }

  function modelCard(card) {
    const c = card.candidate;
    const note = h("span", { class: "ft-muted ft-note", role: "status" });
    const done = async () => {
      await refreshOverview();
      loadModels();
      if (S.dataset && S.tab === "results") renderResults();
    };
    const parts = [
      h("header", { class: "ft-model-head" }, h("h3", {}, card.title), c && c.in_use ? h("span", { class: "pill pill--success" }, `${c.version} in use`) : null),
      h("p", { class: "ft-muted ft-model-note" }, card.note),
      h("div", { class: "ft-inuse" }, "In use now: ", h("b", {}, card.in_use.label)),
    ];
    if (!c) {
      parts.push(h("p", { class: "ft-muted ft-empty" }, "No fine-tuned model yet. Create a dataset for it below and train one; the comparison appears here."));
      if (card.can_swap_back && !readOnly) parts.push(h("div", { class: "ft-actions" }, swapButtons(card.use, null, true, note, done), note));
      return h("article", { class: "ft-model" }, parts);
    }
    parts.push(
      compareRows(c),
      caption(c, card.use),
      verdictLine(c),
      speedLine(c, done),
      h("div", { class: "ft-actions" }, swapButtons(card.use, c, card.can_swap_back, note, done, { pill: false }), note),
      h("button", { type: "button", class: "ft-link", onclick: () => selectDataset(c.dataset_id, { tab: "results", runId: c.run_id, scroll: true }) }, `How ${c.version} was trained →`),
    );
    return h("article", { class: "ft-model" }, parts);
  }

  // -- overview ------------------------------------------------------------
  async function loadOverview() {
    try {
      S.overview = await api("/api/finetune/overview");
    } catch (error) {
      const alert = document.getElementById("ft-alert");
      alert.hidden = false;
      say(alert, error.message, true);
      return;
    }
    renderAlert();
    renderDatasets();
    loadModels();
    if (S.datasetId) selectDataset(S.datasetId, { tab: app.dataset.tab || undefined, runId: app.dataset.run || "" });
    else renderStart();
  }

  function renderAlert() {
    const env = S.overview.environment;
    const alert = document.getElementById("ft-alert");
    alert.hidden = env.ready;
    if (!env.ready) fill(alert, h("strong", {}, "Training is not set up on this machine. "), `Missing ${env.missing.join(", ")}: ${env.install}`);
  }

  async function refreshOverview() {
    S.overview = await api("/api/finetune/overview");
    renderDatasets();
    if (S.dataset) renderSteps();
  }

  function renderDatasets() {
    const list = document.getElementById("ft-datasets");
    const rows = S.overview ? S.overview.datasets : [];
    if (!rows.length) {
      fill(list, h("p", { class: "ft-muted" }, "No datasets yet. Create one to start."));
      return;
    }
    const groups = JOBS.map((job) => ({
      ...job,
      items: rows
        .filter((d) => jobOf(d) === job.key)
        .sort((a, b) => Number(a.test_set) - Number(b.test_set) || String(a.created_at).localeCompare(String(b.created_at))),
    })).filter((g) => g.items.length);
    fill(list, groups.map((g) => h("div", { class: "ft-group" }, h("h3", {}, g.title), g.items.map(datasetButton))));
  }

  function datasetButton(d) {
    const accepted = d.counts.status.accepted || 0;
    const pending = d.counts.status.pending || 0;
    const runs = d.runs || {};
    const bits = [`${accepted} accepted`];
    if (pending) bits.push(`${pending} to review`);
    if (runs.running) bits.push(`${runs.running} training`);
    else if (runs.latest && runs.latest.after !== null && runs.latest.after !== undefined) bits.push(`${runs.latest.label} ${metricValue(runs.latest.metric, runs.latest.after)}`);
    return h(
      "button",
      { class: "ft-dataset", type: "button", "aria-current": d.dataset_id === S.datasetId ? "true" : "false", onclick: () => selectDataset(d.dataset_id, { scroll: true }) },
      h("span", { class: "ft-dataset-name" }, h("span", {}, d.name), d.test_set ? h("span", { class: "ft-tag" }, "Test set") : null),
      h("small", {}, bits.join(" · ")),
    );
  }

  // -- a new dataset (also the page's starting view) ---------------------------
  function renderStart(lead) {
    Object.values(S.timers).forEach(clearTimeout);
    S.datasetId = "";
    S.dataset = null;
    history.replaceState(null, "", "/fine-tuning");
    renderDatasets();
    const hasData = S.overview && S.overview.datasets.length;
    const main = document.getElementById("ft-main");
    fill(main,
      h(
        "section",
        { class: "ft-panel ft-start" },
        h("h2", {}, "New dataset"),
        h("p", { class: "ft-muted" }, lead ?? (hasData ? "Pick a dataset on the left, or start a new one. " : "") + "A dataset holds the examples for one job; you train and compare versions of its model from there."),
        newForm(),
      ),
    );
    return main;
  }

  function newForm() {
    const tasks = S.overview.tasks || [];
    const groups = [["decision", "Decision model (System One)"], ["embedding", "Embedding model"], ["reranker", "Reranker"]];
    const pick = h("select", { name: "task" }, groups.map(([kind, label]) => h("optgroup", { label }, tasks.filter((t) => t.kind === kind).map((t) => h("option", { value: `${t.kind}:${t.task}` }, t.title)))));
    const describe = h("p", { class: "form-note" });
    const name = h("input", { name: "name", maxlength: 120 });
    const note = h("span", { class: "ft-muted", role: "status" });
    const update = () => {
      const [kind, task] = pick.value.split(":");
      const t = tasks.find((x) => x.kind === kind && x.task === task);
      describe.textContent = t ? `${KIND[kind]}. ${t.description}` : "";
      name.placeholder = t ? t.title : "";
    };
    pick.onchange = update;
    update();
    return h(
      "form",
      {
        class: "ft-new",
        onsubmit: async (event) => {
          event.preventDefault();
          const button = event.target.querySelector("button[type=submit]");
          const [kind, task] = pick.value.split(":");
          await busy(button, async () => {
            try {
              const dataset = await api("/api/finetune/datasets", { method: "POST", body: { kind, task, name: name.value } });
              await refreshOverview();
              S.tab = "data";
              await selectDataset(dataset.dataset_id);
            } catch (error) {
              say(note, error.message, true);
            }
          });
        },
      },
      h("div", { class: "ft-row" }, h("label", { class: "ft-wide" }, "What to train", pick), h("label", { class: "ft-wide" }, "Name", name)),
      describe,
      h("div", { class: "ft-actions" }, h("button", { class: "btn btn-primary", type: "submit", disabled: readOnly }, "Create dataset"), note),
    );
  }

  document.getElementById("ft-new-btn").addEventListener("click", () => {
    const main = renderStart("A dataset holds the examples for one job; you train and compare versions of its model from there.");
    main.querySelector("select")?.focus();
    if (narrow()) main.scrollIntoView({ behavior: "smooth", block: "start" });
  });

  // -- dataset ---------------------------------------------------------------
  async function selectDataset(id, opts = {}) {
    Object.values(S.timers).forEach(clearTimeout);
    S.timers = {};
    if (id !== S.datasetId) {
      S.filter = { status: "", split: "" };
      S.limit = 0;
    }
    S.datasetId = id;
    S.selected.clear();
    S.runId = opts.runId || "";
    S.events = [];
    S.after = 0;
    if (opts.tab) S.tab = opts.tab;
    history.replaceState(null, "", `/fine-tuning?dataset=${encodeURIComponent(id)}`);
    const main = document.getElementById("ft-main");
    try {
      S.dataset = await api(`/api/finetune/datasets/${id}`);
    } catch (error) {
      S.dataset = null;
      fill(main, h("div", { class: "empty-state" }, h("strong", {}, "Dataset not found"), h("p", {}, error.message)));
      return;
    }
    if (S.dataset.test_set && ["train", "results"].includes(S.tab)) S.tab = "scores";
    if (!S.dataset.test_set && S.tab === "scores") S.tab = "results";
    renderDatasets();
    renderMain();
    if (opts.scroll && (narrow() || opts.tab)) {
      main.scrollIntoView({ behavior: "smooth", block: "start" });
      S.keepInView = !document.querySelector(".ft-model");
    }
  }

  async function reloadDataset() {
    S.dataset = await api(`/api/finetune/datasets/${S.datasetId}`);
    renderHead();
    renderSteps();
  }

  function setTab(key) {
    Object.values(S.timers).forEach(clearTimeout);
    S.tab = key;
    renderMain();
  }

  function renderMain() {
    const main = document.getElementById("ft-main");
    fill(main, h("section", { class: "ft-panel ft-head-panel", id: "ft-head" }), h("nav", { class: "ft-steps", id: "ft-steps", "aria-label": "Steps" }), h("div", { id: "ft-tab", class: "ft-tab" }));
    renderHead();
    renderSteps();
    ({ data: renderData, task: renderTask, format: renderFormat, train: renderTrain, results: renderResults, scores: renderScores })[S.tab]();
  }

  function renderSteps() {
    const d = S.dataset;
    const nav = document.getElementById("ft-steps");
    if (!nav || !d) return;
    const c = d.counts;
    const summary = ((S.overview && S.overview.datasets) || []).find((x) => x.dataset_id === d.dataset_id) || {};
    const runs = summary.runs || {};
    const step = (key, n, title, sub) =>
      h("button", { type: "button", class: "ft-step", "aria-current": S.tab === key ? "step" : null, onclick: () => setTab(key) }, h("span", { class: "ft-step-n", "aria-hidden": "true" }, n), h("span", { class: "ft-step-t" }, h("b", {}, title), h("small", {}, sub)));
    const accepted = c.status.accepted || 0;
    const pending = c.status.pending || 0;
    const dataSub = `${accepted} accepted${pending ? ` · ${pending} to review` : ""}`;
    const latest = runs.latest;
    const steps = d.test_set
      ? [step("data", 1, "Data", dataSub), step("scores", 2, "Scores", "versions compared on this set")]
      : [
          step("data", 1, "Data", dataSub),
          step("train", 2, "Train", runs.running ? `${runs.running} training now` : runs.total ? plural(runs.total, "run") : "not trained yet"),
          step("results", 3, "Results", latest ? `${latest.label}: ${metricValue(latest.metric, latest.before)} → ${metricValue(latest.metric, latest.after)}` : "after training"),
        ];
    const adv = (key, label) => h("button", { type: "button", class: "ft-adv-btn", "aria-current": S.tab === key ? "step" : null, onclick: () => setTab(key) }, label);
    fill(nav,
      h("div", { class: "ft-steps-row" }, steps),
      h("div", { class: "ft-adv" }, h("span", { class: "ft-muted" }, "Advanced:"), adv("task", d.kind === "decision" ? "Question and options" : "Query instruction"), adv("format", "Model input format")),
    );
  }

  function renderHead() {
    const d = S.dataset;
    const head = document.getElementById("ft-head");
    if (!head || !d) return;
    const c = d.counts;
    const title = h("h2", {}, h("span", { class: "ft-title" }, d.name), d.test_set ? h("span", { class: "ft-tag" }, "Test set") : null);
    const stats = h(
      "div",
      { class: "ft-stats" },
      h("span", {}, `${KIND[d.kind]} · ${taskTitle(d)}`),
      h("span", {}, h("b", {}, c.total), " examples"),
      d.test_set
        ? h("span", {}, "only scored on, never trained on")
        : h("span", {}, "held out for testing: ", h("b", {}, (c.split.test || 0) + (c.split.validation || 0)), h("span", { class: "ft-muted" }, ` (test ${c.split.test || 0}, validation ${c.split.validation || 0})`)),
    );
    const rename = () => {
      const input = h("input", { value: d.name, maxlength: 120, "aria-label": "Dataset name" });
      const save = async () => {
        try {
          await api(`/api/finetune/datasets/${d.dataset_id}`, { method: "PATCH", body: { name: input.value } });
          await reloadDataset();
          await refreshOverview();
        } catch (error) {
          say(stats, error.message, true);
        }
      };
      input.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter") save();
        if (ev.key === "Escape") renderHead();
      });
      fill(title, input, h("button", { class: "btn btn-sm btn-primary", type: "button", onclick: save }, "Save"), h("button", { class: "btn btn-sm", type: "button", onclick: renderHead }, "Cancel"));
      input.focus();
      input.select();
    };
    const toggleTest = async () => {
      try {
        await api(`/api/finetune/datasets/${d.dataset_id}`, { method: "PATCH", body: { test_set: !d.test_set } });
        await refreshOverview();
        await selectDataset(d.dataset_id, { tab: d.test_set ? "data" : S.tab });
      } catch (error) {
        say(stats, error.message, true);
      }
    };
    const remove = async () => {
      if (!confirm(`Delete "${d.name}" with its ${c.total} examples and its run records? The trained model folders stay on disk.`)) return;
      await api(`/api/finetune/datasets/${d.dataset_id}`, { method: "DELETE" });
      await refreshOverview();
      loadModels();
      renderStart(`Deleted “${d.name}”. Pick another dataset on the left, or start a new one.`);
    };
    const menu = readOnly
      ? null
      : h(
          "details",
          { class: "ft-menu" },
          h("summary", { class: "btn btn-sm" }, "More ▾"),
          h(
            "div",
            { class: "ft-menu-list" },
            h("button", { type: "button", onclick: (ev) => { ev.target.closest("details").open = false; rename(); } }, "Rename"),
            h("button", { type: "button", title: d.test_set ? "Train on it again" : "Only score on it from now on: keep it out of training and replay", onclick: toggleTest }, d.test_set ? "Use for training again" : "Mark as test set"),
            h("button", { type: "button", class: "ft-danger", onclick: remove }, "Delete dataset…"),
          ),
        );
    fill(head, h("div", { class: "ft-head" }, h("div", { class: "ft-head-text" }, title, stats), menu));
  }

  // -- data tab ----------------------------------------------------------------
  function labelBars() {
    const q = question();
    const labels = S.dataset.labels || {};
    if (!q || !Object.keys(labels).length) return null;
    const max = Math.max(1, ...Object.values(labels).map((v) => (v.accepted || 0) + (v.pending || 0)));
    return h(
      "div",
      {},
      h("h3", {}, "Examples per option"),
      h(
        "div",
        { class: "ft-bars" },
        Object.entries(labels).map(([key, v]) => {
          const accepted = v.accepted || 0;
          const pending = v.pending || 0;
          return h(
            "div",
            { class: "ft-bar", title: `${accepted} accepted, ${pending} to review, ${v.rejected || 0} rejected` },
            h("span", {}, key),
            h("span", { class: "ft-bar-track" }, h("span", { class: "ft-bar-fill", style: `width:${(accepted / max) * 100}%` }), h("span", { class: "ft-bar-fill ft-bar-fill--pending", style: `width:${(pending / max) * 100}%` })),
            h("span", { class: "ft-bar-num" }, `${accepted}${pending ? "+" + pending : ""}`),
          );
        }),
      ),
      h("p", { class: "ft-muted" }, "Solid: accepted · light: waiting for review. Options with few examples are the ones to generate more of."),
    );
  }

  function renderData() {
    const d = S.dataset;
    const tab = document.getElementById("ft-tab");
    const retrievalLike = d.kind !== "decision" || d.task === "relevance";
    const teacher = `${S.overview.teacher.provider}/${S.overview.teacher.model}`;
    const genStatus = h("div", { class: "ft-progress", hidden: true });
    const verifier = h(
      "select",
      { name: "verifier" },
      h("option", { value: "" }, "The teacher (same model)"),
      h("option", { value: "typesafe/jev-latest", disabled: !S.overview.jev_available }, "Jev (TypeSafe, calibrated)" + (S.overview.jev_available ? "" : " — set TYPESAFE_API_KEY")),
      h("option", { value: "other" }, "Another LLM…"),
    );
    const otherVerifier = h("input", { name: "verifier_other", placeholder: "provider/model", hidden: true });
    verifier.onchange = () => { otherVerifier.hidden = verifier.value !== "other"; };
    const wanted = new Set(S.preselect || []);
    S.preselect = null;
    const chips = optionKeys(question()).map((k) => {
      const box = h("input", { type: "checkbox", name: "labels", value: k });
      box.checked = wanted.has(k);
      return h("label", { class: "ft-chip" }, box, h("span", {}, k));
    });
    const generateForm = h(
      "form",
      {
        id: "ft-generate",
        onsubmit: async (event) => {
          event.preventDefault();
          const f = event.target.elements;
          const picked = [...event.target.querySelectorAll("input[name=labels]:checked")].map((b) => b.value);
          const body = {
            labels: picked.length ? picked : null,
            count: Number(f.count.value),
            teacher: f.teacher.value.trim() || null,
            verifier: verifier.value === "other" ? otherVerifier.value.trim() || null : verifier.value || null,
            context: f.context ? f.context.value : "",
            topic: f.topic ? f.topic.value : "",
            from_memory: f.from_memory ? f.from_memory.checked : false,
          };
          const button = event.target.querySelector("button[type=submit]");
          button.disabled = true;
          genStatus.hidden = false;
          const bar = h("div", { class: "ft-progress-fill", style: "width:2%" });
          const label = h("span", {}, "Starting…");
          fill(genStatus, label, h("div", { class: "ft-progress-track" }, bar));
          try {
            const { job_id } = await api(`/api/finetune/datasets/${d.dataset_id}/generate`, { method: "POST", body });
            const poll = async () => {
              const job = await api(`/api/finetune/jobs/${job_id}`);
              bar.style.width = `${Math.max(2, (job.done / job.total) * 100)}%`;
              label.textContent = job.message;
              if (job.status === "running") {
                S.timers.job = setTimeout(poll, 1500);
                return;
              }
              button.disabled = readOnly;
              if (job.status === "failed") {
                say(label, job.error, true);
                return;
              }
              const r = job.report || {};
              const rejected = Object.entries(r.rejected || {}).map(([k, v]) => `${v} ${k.replace(/_/g, " ")}`).join(", ");
              const cost = [r.usage, r.verifier_usage].filter(Boolean).map((u) => u.cost_usd).filter((v) => v !== null && v !== undefined);
              label.textContent = `Added ${r.added} of ${r.written ?? r.passages ?? "?"} written` + (rejected ? ` · rejected: ${rejected}` : "") + (r.short && r.short.length ? ` · still short: ${r.short.join(", ")}` : "") + (cost.length ? ` · cost $${cost.reduce((a, b) => a + b, 0).toFixed(4)}` : "") + ". Review them below.";
              await reloadDataset();
              await refreshOverview();
              loadExamples();
            };
            poll();
          } catch (error) {
            button.disabled = readOnly;
            say(label, error.message, true);
          }
        },
      },
      h(
        "div",
        { class: "ft-row" },
        h("label", {}, "How many", h("input", { type: "number", name: "count", min: 1, max: 2000, value: retrievalLike ? 24 : 60 })),
        h("label", { class: "ft-wide" }, "Teacher (provider/model)", h("input", { name: "teacher", value: teacher })),
        h("label", { class: "ft-wide" }, "Blind check", verifier, otherVerifier),
      ),
      retrievalLike
        ? h(
            "div",
            { class: "ft-row" },
            h("label", { class: "ft-wide" }, "Topic of generated memories", h("input", { name: "topic", placeholder: "e.g. a product team's projects, people and decisions" })),
            h("label", { class: "ft-check" }, h("input", { type: "checkbox", name: "from_memory" }), h("span", {}, "Use my stored memories instead")),
          )
        : [
            h("div", { class: "ft-row" }, h("label", { class: "ft-wide" }, "Setting (optional)", h("input", { name: "context", placeholder: "e.g. a support assistant for a SaaS billing product" }))),
            chips.length
              ? h("fieldset", { class: "ft-chips" }, h("legend", {}, "Only for these options ", h("span", { class: "ft-muted" }, "(none ticked: every option)")), chips)
              : null,
          ],
      h("div", { class: "ft-actions" }, h("button", { class: "btn btn-primary", type: "submit", disabled: readOnly }, "Generate examples")),
      genStatus,
    );

    const importNote = h("span", { class: "ft-muted" });
    const importBox =
      d.kind === "decision" && d.task !== "relevance"
        ? h(
            "div",
            {},
            h("h3", {}, "From your memory"),
            h("p", { class: "ft-muted" }, "Each stored memory becomes an example labelled with the store it lives in. Review them: a memory can be filed in the wrong store."),
            h("div", { class: "ft-actions" }, h("button", { class: "btn", type: "button", disabled: readOnly, onclick: (event) => busy(event.target, async () => {
              try {
                const r = await api(`/api/finetune/datasets/${d.dataset_id}/import-memories`, { method: "POST", body: {} });
                importNote.textContent = `Imported ${r.added}: ` + Object.entries(r.by_store).map(([k, v]) => `${k} ${v}`).join(", ");
                await reloadDataset();
                await refreshOverview();
                loadExamples();
              } catch (error) {
                say(importNote, error.message, true);
              }
            }) }, "Import memories"), importNote),
          )
        : null;

    fill(tab,
      h("div", { class: "ft-grid2" },
        h("section", { class: "ft-panel" }, h("h2", {}, "Generate examples"), h("p", { class: "ft-muted" }, retrievalLike ? "The teacher writes a question each memory answers and a look-alike memory that doesn't; a blind check keeps only the pairs it confirms." : "The teacher writes states for each option in turn (plain, negated, borderline and look-alike cases); a blind check keeps only the ones it labels the same way."), generateForm),
        h("section", { class: "ft-panel" }, labelBars() || h("div", {}, h("h3", {}, "Where the examples came from"), h("p", { class: "ft-muted" }, Object.entries(d.sources || {}).map(([k, v]) => `${k} ${v}`).join(" · ") || "No examples yet.")), importBox, addExampleForm()),
      ),
      h("section", { class: "ft-panel", id: "ft-examples" }),
    );
    loadExamples();
    if (wanted.size) {
      const form = document.getElementById("ft-generate");
      form.scrollIntoView({ behavior: "smooth", block: "center" });
      form.elements.count.focus({ preventScroll: true });
    }
  }

  function addExampleForm() {
    const d = S.dataset;
    const q = question();
    const note = h("span", { class: "ft-muted" });
    if (d.kind === "decision" && d.task !== "relevance") {
      return h("form", {
        onsubmit: async (event) => {
          event.preventDefault();
          const f = event.target.elements;
          try {
            const label = { [q.qid]: q.type === "choice" ? { choice: f.label.value } : q.type === "noul" ? { noul: f.label.value === "true" } : { level: Number(f.label.value) } };
            await api(`/api/finetune/datasets/${d.dataset_id}/examples`, { method: "POST", body: { input: { state: f.state.value }, label } });
            f.state.value = "";
            say(note, "Added.");
            await reloadDataset();
            loadExamples();
          } catch (error) {
            say(note, error.message, true);
          }
        },
      },
      h("h3", {}, "Add one by hand"),
      autosize(h("textarea", { name: "state", rows: 2, required: true, placeholder: "A piece of state, e.g. a message or tool result", class: "ft-full" })),
      h("div", { class: "ft-row" }, h("label", { class: "ft-wide" }, "Right option", h("select", { name: "label" }, optionKeys(q).map((k) => h("option", { value: k }, k)))), h("button", { class: "btn", type: "submit", disabled: readOnly }, "Add"), note));
    }
    if (d.kind === "decision") return null;
    return h("form", {
      onsubmit: async (event) => {
        event.preventDefault();
        const f = event.target.elements;
        try {
          await api(`/api/finetune/datasets/${d.dataset_id}/examples`, { method: "POST", body: { input: { query: f.query.value, positive: f.positive.value, negatives: f.negatives.value.split("\n").map((s) => s.trim()).filter(Boolean) } } });
          event.target.reset();
          say(note, "Added.");
          await reloadDataset();
          loadExamples();
        } catch (error) {
          say(note, error.message, true);
        }
      },
    },
    h("h3", {}, "Add one by hand"),
    h("input", { name: "query", required: true, placeholder: "A question", class: "ft-full" }),
    autosize(h("textarea", { name: "positive", rows: 2, required: true, placeholder: "The memory that answers it", class: "ft-full" })),
    autosize(h("textarea", { name: "negatives", rows: 2, placeholder: "Look-alike memories that don't, one per line (optional)", class: "ft-full" })),
    h("div", { class: "ft-actions" }, h("button", { class: "btn", type: "submit", disabled: readOnly }, "Add"), note));
  }

  // A retrieval example (question, memory, look-alikes) is taller than a state.
  const pageSize = () => (S.dataset && S.dataset.kind === "decision" && S.dataset.task !== "relevance" ? 50 : 25);

  async function loadExamples() {
    const box = document.getElementById("ft-examples");
    if (!box) return;
    if (!S.limit) S.limit = pageSize();
    const query = new URLSearchParams({ status: S.filter.status, split: S.filter.split, limit: S.limit });
    let data;
    try {
      data = await api(`/api/finetune/datasets/${S.datasetId}/examples?${query}`);
    } catch (error) {
      fill(box, h("p", { class: "ft-down" }, error.message));
      return;
    }
    S.examples = data.examples;
    S.total = data.total;
    renderExamples();
  }

  function renderExamples() {
    const box = document.getElementById("ft-examples");
    if (!box) return;
    const q = question();
    const isDecision = S.dataset.kind === "decision";
    const keys = optionKeys(q);
    const note = h("span", { class: "ft-muted", role: "status" });
    const pick = (name, label, options) => {
      const select = h("select", { "aria-label": label, onchange: () => { S.filter[name] = select.value; loadExamples(); } }, options.map(([v, l]) => h("option", { value: v }, l)));
      select.value = S.filter[name];
      return select;
    };
    const after = async () => {
      S.selected.clear();
      await reloadDataset();
      await refreshOverview();
      loadExamples();
    };
    const bulk = async (status) => {
      if (!S.selected.size) return say(note, "Tick some examples first.");
      await api("/api/finetune/examples/review", { method: "POST", body: { ids: [...S.selected], status } });
      await after();
    };
    const rows = S.examples.map((e) => {
      const check = h("input", { type: "checkbox", "aria-label": "Select example", onchange: (ev) => (ev.target.checked ? S.selected.add(e.example_id) : S.selected.delete(e.example_id)) });
      check.checked = S.selected.has(e.example_id);
      const split = h("select", { disabled: readOnly, "aria-label": "Split", onchange: async () => { await api(`/api/finetune/examples/${e.example_id}`, { method: "PATCH", body: { split: split.value } }); reloadDataset(); } }, ["train", "validation", "test"].map((s) => h("option", { value: s }, s)));
      split.value = e.split;
      let label = "";
      if (isDecision && q) {
        const relabel = h("select", { disabled: readOnly, "aria-label": "Label", onchange: async () => { await api(`/api/finetune/examples/${e.example_id}`, { method: "PATCH", body: { label_key: relabel.value } }); reloadDataset(); } }, keys.map((k) => h("option", { value: k }, k)));
        relabel.value = goldKey(q, e.label);
        label = relabel;
      }
      const probs = e.meta && e.meta.verifier_probabilities;
      const top = probs ? Object.entries(probs).sort((a, b) => b[1] - a[1])[0] : null;
      const source = [e.source, e.case_type].filter(Boolean).join(" · ").replace(/_/g, " ");
      return h(
        "tr",
        {},
        h("td", {}, check),
        h("td", { class: "ft-status" }, statusPill(e.status), e.verified ? h("small", { title: top ? `The blind check picked ${top[0]} (${(top[1] * 100).toFixed(0)}%)` : "The blind check agreed" }, "✓ checked", top ? ` ${(top[1] * 100).toFixed(0)}%` : "") : null),
        isDecision ? h("td", {}, label) : null,
        h("td", { class: "ft-text" }, exampleCell(e), source ? h("small", { class: "ft-source" }, source) : null),
        h("td", {}, split),
      );
    });
    fill(box,
      h("div", { class: "ft-panel-head" }, h("h2", {}, `Examples (${S.total})`), h("span", { class: "ft-muted" }, "Accepted train examples are trained on; test and validation ones are held out for scoring.")),
      h(
        "div",
        { class: "ft-actions ft-toolbar" },
        pick("status", "Status", [["", "All statuses"], ["pending", "To review"], ["accepted", "Accepted"], ["rejected", "Rejected"]]),
        pick("split", "Split", [["", "All splits"], ["train", "Train"], ["validation", "Validation"], ["test", "Test"]]),
        h("button", { class: "btn btn-sm btn-primary", type: "button", disabled: readOnly, title: "Accept every example the blind check agreed with", onclick: async () => { const r = await api(`/api/finetune/datasets/${S.datasetId}/accept-verified`, { method: "POST" }); await after(); say(note, `Accepted ${r.accepted} checked examples.`); } }, "Accept all checked (✓)"),
        h("button", { class: "btn btn-sm", type: "button", disabled: readOnly, onclick: () => bulk("accepted") }, "Accept ticked"),
        h("button", { class: "btn btn-sm", type: "button", disabled: readOnly, onclick: () => bulk("rejected") }, "Reject ticked"),
        h("button", { class: "btn btn-sm btn-danger", type: "button", disabled: readOnly, onclick: async () => { if (!S.selected.size) return say(note, "Tick some examples first."); if (!confirm(`Delete ${S.selected.size} examples?`)) return; await api("/api/finetune/examples/delete", { method: "POST", body: { ids: [...S.selected] } }); await after(); } }, "Delete ticked"),
        note,
      ),
      S.examples.length
        ? h("div", { class: "ft-table-wrap" }, h("table", { class: "ft-table" }, h("thead", {}, h("tr", {}, h("th", {}, h("span", { class: "sr-only" }, "Select")), h("th", {}, "Status"), isDecision ? h("th", {}, "Label") : null, h("th", {}, isDecision ? "State" : "Example"), h("th", {}, "Split"))), h("tbody", {}, rows)))
        : h("p", { class: "ft-muted" }, "No examples match."),
      S.total > S.examples.length ? h("button", { class: "btn btn-sm", type: "button", onclick: () => { S.limit += pageSize(); loadExamples(); } }, `Show ${Math.min(pageSize(), S.total - S.examples.length)} more (${S.total - S.examples.length} left)`) : null,
    );
  }

  // -- task tab (advanced) ------------------------------------------------------
  function renderTask() {
    const tab = document.getElementById("ft-tab");
    const d = S.dataset;
    const note = h("span", { class: "ft-muted" });
    if (d.kind !== "decision") {
      const input = autosize(h("textarea", { class: "ft-instructions", value: d.spec.instruction || "" }));
      fill(tab, h("section", { class: "ft-panel" }, h("h2", {}, "Query instruction"), h("p", { class: "ft-muted" }, "Instruction-tuned embedders (Qwen3-Embedding) read this before each query. EmbeddingGemma uses its own trained search prompt."), input, h("div", { class: "ft-actions" }, h("button", { class: "btn btn-primary", type: "button", disabled: readOnly, onclick: async () => { try { await api(`/api/finetune/datasets/${d.dataset_id}`, { method: "PATCH", body: { spec: { ...d.spec, instruction: input.value } } }); say(note, "Saved."); await reloadDataset(); } catch (error) { say(note, error.message, true); } } }, "Save"), note)));
      return;
    }
    const q = question();
    const instructions = autosize(h("textarea", { class: "ft-instructions", value: q.instructions || "" }));
    const editor = h("div", { class: "ft-options-editor" });
    const addRow = (key = "", desc = "") => {
      const row = h("div", { class: "ft-option" }, h("input", { value: key, placeholder: "option key", "aria-label": "Option key" }), autosize(h("textarea", { value: desc, rows: 1, placeholder: "When to choose it", "aria-label": "Option description" })), h("button", { class: "btn btn-sm", type: "button", onclick: () => row.remove() }, "Remove"));
      editor.append(row);
    };
    if (q.type === "choice") Object.entries(q.criteria || {}).forEach(([k, v]) => addRow(k, v));
    fill(tab,
      h(
        "section",
        { class: "ft-panel" },
        h("h2", {}, `The question (${q.type})`),
        h("p", { class: "ft-muted" }, "A decision model reads these instructions and option descriptions with every state, so they are part of its input. Changing an option key does not relabel existing examples."),
        h("label", { class: "form-group" }, "Instructions", instructions),
        q.type === "choice" ? h("div", {}, h("h3", {}, "Options"), editor, h("button", { class: "btn btn-sm", type: "button", onclick: () => addRow() }, "Add option")) : h("pre", { class: "ft-code" }, JSON.stringify(q.criteria, null, 2)),
        h("div", { class: "ft-actions" }, h("button", { class: "btn btn-primary", type: "button", disabled: readOnly, onclick: async () => {
          const spec = JSON.parse(JSON.stringify(d.spec));
          const target = spec.questions[q.qid];
          target.instructions = instructions.value;
          if (q.type === "choice") {
            target.criteria = {};
            for (const row of editor.children) {
              const [key, desc] = row.querySelectorAll("input, textarea");
              if (key.value.trim()) target.criteria[key.value.trim()] = desc.value.trim();
            }
          }
          try {
            await api(`/api/finetune/datasets/${d.dataset_id}`, { method: "PATCH", body: { spec } });
            say(note, "Saved.");
            await reloadDataset();
          } catch (error) {
            say(note, error.message, true);
          }
        } }, "Save question"), note),
      ),
    );
  }

  // -- format tab (advanced) ------------------------------------------------------
  function baseSelect(kind, value) {
    const options = S.overview.base_models.filter((m) => m.kinds.includes(kind));
    const select = h("select", { name: "base_model" }, options.map((m) => h("option", { value: m.model_id, title: m.notes }, `${m.label} · ${m.params}${m.backbone === "lm" ? " · small LM" : ""}`)));
    if (value) select.value = value;
    return select;
  }

  function renderFormat() {
    const tab = document.getElementById("ft-tab");
    const d = S.dataset;
    const formats = d.kind === "decision" ? [["", "The base model's own"], ["systemone", "System One / typed-decisions (Unsloth, /v1/systemone)"], ["jevembed", "JevEmbed (embedding backbone)"], ["chat", "Chat prompt (small-LM backbone)"], ["memorizz", "MemoRizz records"]] : [["", "The base model's own"], ["sentence-transformers", "Sentence Transformers (anchor, positive, negative)"], ["cross-encoder", "Cross-encoder pairs (query, passage, label)"], ["memorizz", "MemoRizz records"]];
    const base = baseSelect(d.kind);
    const format = h("select", {}, formats.map(([v, l]) => h("option", { value: v }, l)));
    const out = h("div", {});
    const note = h("span", { class: "ft-muted" });
    const show = async () => {
      try {
        const p = await api(`/api/finetune/datasets/${d.dataset_id}/preview?${new URLSearchParams({ base_model: base.value, format: format.value, limit: 3 })}`);
        fill(out,
          h("p", {}, h("strong", {}, p.how.style), h("br"), h("span", { class: "ft-muted" }, "Objective: ", p.how.objective)),
          h("p", { class: "ft-muted" }, `Format: ${p.format} · ${p.rows.length} example rows shown`),
          ...p.rows.map((row) => h("pre", { class: "ft-code" }, JSON.stringify(row, null, 2))),
        );
      } catch (error) {
        say(out, error.message, true);
      }
    };
    base.onchange = show;
    format.onchange = show;
    const bundleBase = h("input", { value: "unsloth/Qwen3.5-4B", placeholder: "Unsloth base model" });
    fill(tab,
      h(
        "section",
        { class: "ft-panel" },
        h("h2", {}, "Training input, as each model expects it"),
        h("p", { class: "ft-muted" }, "What a base model actually reads for each accepted example. Download it to train elsewhere."),
        h("div", { class: "ft-row" }, h("label", { class: "ft-wide" }, "Base model", base), h("label", { class: "ft-wide" }, "Format", format)),
        h(
          "div",
          { class: "ft-actions" },
          h("button", { class: "btn", type: "button", onclick: () => download(`/api/finetune/datasets/${d.dataset_id}/export?${new URLSearchParams({ format: format.value || (d.kind === "decision" ? "systemone" : d.kind === "reranker" ? "cross-encoder" : "sentence-transformers"), base_model: base.value })}`, "finetune.jsonl").catch((e) => say(note, e.message, true)) }, "Download accepted examples (JSONL)"),
          note,
        ),
        out,
      ),
      d.kind === "decision"
        ? h(
            "section",
            { class: "ft-panel" },
            h("h2", {}, "Train with Unsloth on a GPU"),
            h("p", { class: "ft-muted" }, `Unsloth trains decision models on an LLM with Cloudflare Clef's option head. It needs an NVIDIA GPU (${S.overview.environment.unsloth.available ? "available here" : "not on this machine"}). The bundle has your accepted examples in its typed-decisions format and a script to run on Colab or any CUDA machine.`),
            h("div", { class: "ft-row" }, h("label", { class: "ft-wide" }, "Unsloth base model", bundleBase), h("button", { class: "btn", type: "button", onclick: () => download(`/api/finetune/datasets/${d.dataset_id}/unsloth-bundle?${new URLSearchParams({ base_model: bundleBase.value })}`, "unsloth.zip").catch((e) => say(note, e.message, true)) }, "Download Unsloth bundle")),
          )
        : null,
    );
    show();
  }

  // -- training curves (shared chart module) -------------------------------------
  // A curves object ({steps, validation, before, after, metric}) from a run's
  // raw events, for a run still training (the server builds it otherwise).
  function curvesFromEvents(run, events) {
    const phase = (name) => (events.find((e) => e.type === "eval" && e.data.phase === name) || { data: { metrics: {} } }).data.metrics;
    return {
      metric: run.kind === "decision" && run.task !== "relevance" ? "accuracy" : "ndcg@10",
      steps: events.filter((e) => e.type === "step").map((e) => e.data),
      total_steps: run.progress && run.progress.total,
      validation: events.filter((e) => e.type === "eval" && e.data.phase === "validation").map((e) => ({ epoch: e.data.epoch, step: e.data.step, metrics: e.data.metrics || {} })),
      before: phase("before"),
      after: phase("after"),
    };
  }

  function curveCharts(c) {
    const C = window.MemorizzCharts;
    if (!C) return h("p", { class: "ft-muted" }, "Charts unavailable.");
    const metric = c.metric;
    const steps = c.steps.filter((s) => typeof s.loss === "number");
    const raw = steps.map((s) => [s.step, s.loss]);
    // Epoch boundaries: where validation was scored, else where the epoch count turns over.
    let ends = c.validation.map((v) => v.step).filter((x) => x);
    if (!ends.length) {
      for (let k = 1; k < 50; k++) {
        const hit = steps.find((s) => s.epoch >= k - 1e-6);
        if (!hit) break;
        ends.push(hit.step);
      }
    }
    const markers = ends.slice(0, -1).map((x, i) => ({ x, label: `epoch ${i + 1}` }));
    const lossSeries = raw.length > 20
      ? [{ name: "loss per step", points: raw, tone: 0, faint: true }, { name: "loss (smoothed)", points: C.smooth(raw, 0.12), tone: 0 }]
      : [{ name: "loss", points: raw, tone: 0, dots: true }];
    const loss = C.line(lossSeries, { logY: true, xLabel: "training step", markers, empty: "Waiting for the first steps…" });
    const value = (m) => (m ? m[metric] ?? m.accuracy : undefined);
    const words = metricWords(metric, S.dataset ? jobOf(S.dataset) : "");
    const points = c.validation.map((v) => [v.epoch, value(v.metrics)]).filter((p) => typeof p[1] === "number");
    const refs = [];
    if (typeof value(c.before) === "number") refs.push({ y: value(c.before), label: "test, before training", tone: 5 });
    if (typeof value(c.after) === "number") refs.push({ y: value(c.after), label: "test, after training", tone: 1 });
    const all = points.map((p) => p[1]).concat(refs.map((r) => r.y));
    const range = all.length ? { yMin: Math.max(0, Math.min(...all) - 0.05), yMax: Math.min(1, Math.max(...all) + 0.05) } : {};
    const validation = points.length
      ? C.line([{ name: `validation: ${words.toLowerCase()}`, points, tone: 2, dots: true }], { ...range, asPct: true, xLabel: "epoch", refs, xFormat: (x) => `${Math.round(x)}`, legend: true })
      : C.line(
          refs.length === 2 ? [{ name: "test score", points: [[0, refs[0].y], [1, refs[1].y]], tone: 1, dots: true }] : [],
          { ...range, asPct: true, xFormat: (x) => (x < 0.5 ? "before" : "after"), empty: "No validation score was recorded.", legend: true },
        );
    return h("div", { class: "ft-curves" },
      h("div", {}, h("h3", {}, "Training loss (lower is better)"), loss),
      h("div", {}, h("h3", {}, points.length ? `Validation each epoch: ${words.toLowerCase()}` : `Test score before and after: ${words.toLowerCase()}`), validation));
  }

  function trainingPanel(run) {
    const panel = h("section", { class: "ft-panel", id: "ft-curves" }, h("h2", {}, "How it trained"), h("p", { class: "ft-muted" }, "Loading its curves…"));
    api(`/api/finetune/runs/${run.run_id}/curves`)
      .then((c) => fill(panel,
        h("div", { class: "ft-panel-head" }, h("h2", {}, "How it trained"), h("span", { class: "ft-muted" }, `${c.steps.length ? `${c.total_steps || c.steps[c.steps.length - 1].step} steps` : ""}${c.validation.length ? ` · ${c.validation.length} epochs scored on validation` : ""}`)),
        curveCharts(c),
        c.convergence.length ? h("ul", { class: "ft-notes" }, c.convergence.map((n) => h("li", {}, n))) : null))
      .catch((error) => fill(panel, h("h2", {}, "How it trained"), h("p", { class: "ft-down" }, error.message)));
    return panel;
  }

  function dataPanel(run) {
    const d = S.dataset;
    const panel = h("section", { class: "ft-panel", id: "ft-data" });
    const state = { split: "", source: "", limit: 25 };
    const load = async () => {
      let data;
      try {
        data = await api(`/api/finetune/runs/${run.run_id}/data?${new URLSearchParams({ split: state.split, source: state.source, limit: state.limit })}`);
      } catch (error) {
        fill(panel, h("h2", {}, "The data it trained on"), h("p", { class: "ft-muted" }, error.message));
        return;
      }
      const sm = data.summary;
      const replayed = Object.entries(sm.replayed);
      const pick = (key, label, options) => {
        const select = h("select", { "aria-label": label, onchange: () => { state[key] = select.value; state.limit = 25; load(); } }, options.map(([v, l]) => h("option", { value: v }, l)));
        select.value = state[key];
        return select;
      };
      const labels = Object.entries(sm.labels || {}).filter(([k]) => k);
      fill(panel,
        h("div", { class: "ft-panel-head" }, h("h2", {}, "The data it trained on"), h("span", { class: "ft-muted" }, `${run.label}'s own snapshot, taken when it started`)),
        h("div", { class: "ft-data-summary" },
          ["train", "validation", "test"].map((sp) => h("span", {}, h("b", {}, sm.splits[sp] || 0), ` ${sp}`)),
          replayed.length ? h("span", {}, "including ", replayed.map(([n, c]) => `${c} replayed from “${n}”`).join(", ")) : null,
          h("span", { class: "ft-muted" }, `“${data.dataset_name}” has changed since: ${data.since.added} added, ${data.since.removed} removed`)),
        labels.length
          ? h("details", { class: "ft-advanced" }, h("summary", {}, `Examples per option (${labels.length})`),
              h("div", { class: "ft-table-wrap" }, h("table", { class: "ft-metrics" },
                h("thead", {}, h("tr", {}, h("th", {}, "Option"), h("th", {}, "Train"), h("th", {}, "Validation"), h("th", {}, "Test"))),
                h("tbody", {}, labels.sort((a, b) => (b[1].train || 0) - (a[1].train || 0)).map(([k, v]) => h("tr", {}, h("td", {}, k), h("td", {}, v.train || 0), h("td", {}, v.validation || 0), h("td", {}, v.test || 0)))))))
          : null,
        h("div", { class: "ft-actions ft-toolbar" },
          pick("split", "Split", [["", "All splits"], ["train", "Train"], ["validation", "Validation"], ["test", "Test"]]),
          replayed.length ? pick("source", "Source", [["", "All examples"], ["own", "This dataset's"], ["replay", "Replayed"]]) : null,
          h("span", { class: "ft-muted" }, `${data.total} example${data.total === 1 ? "" : "s"}`)),
        data.rows.length
          ? h("div", { class: "ft-table-wrap" }, h("table", { class: "ft-table" },
              h("thead", {}, h("tr", {}, h("th", {}, "Split"), d.kind === "decision" ? h("th", {}, "Label") : null, h("th", {}, "Example"))),
              h("tbody", {}, data.rows.map((r) => h("tr", {},
                h("td", {}, r.split, r.replayed_from ? h("div", { class: "ft-replay", title: `Replayed from “${r.replayed_from}”` }, "replayed") : null),
                d.kind === "decision" ? h("td", {}, r.label) : null,
                h("td", { class: "ft-text" }, r.state !== undefined ? clampText(r.state) : exampleCell({ input: { query: r.query, positive: r.positive, negatives: r.negatives } })))))))
          : h("p", { class: "ft-muted" }, "No examples match."),
        data.total > data.rows.length ? h("button", { class: "btn btn-sm", type: "button", onclick: () => { state.limit += 25; load(); } }, `Show 25 more (${data.total - data.rows.length} left)`) : null,
      );
      if (location.hash === "#ft-data") panel.scrollIntoView({ block: "start" });
    };
    fill(panel, h("h2", {}, "The data it trained on"), h("p", { class: "ft-muted" }, "Loading…"));
    load();
    return panel;
  }

  // -- train tab -----------------------------------------------------------------
  function envLine() {
    const env = S.overview.environment;
    return h(
      "p",
      { class: "ft-env" },
      h("span", { class: "ft-env-k" }, "This machine"),
      h("span", {}, "trains on ", h("strong", {}, DEVICE[env.device] || env.device || "?")),
      env.ready ? h("span", {}, "packages ", h("strong", {}, "ready")) : h("span", { class: "ft-down" }, `missing ${env.missing.join(", ")}: ${env.install}`),
      h("span", { title: env.unsloth.reason || null }, "Unsloth ", h("strong", {}, env.unsloth.available ? "available" : "needs an NVIDIA GPU")),
    );
  }

  const runOption = (r, metric) => `${r.label}${r.lineage ? ` (${r.lineage})` : ""} · ${r.short_base} · ${metricValue(metric, (r.metrics_after || {})[metric])}`;

  async function renderTrain() {
    const tab = document.getElementById("ft-tab");
    const d = S.dataset;
    const metric = primaryMetric(d);
    const base = baseSelect(d.kind);
    let allRuns = [];
    try {
      allRuns = (await api("/api/finetune/runs")).runs;
    } catch (error) {
      allRuns = [];
    }
    if (S.tab !== "train" || !S.dataset || S.dataset.dataset_id !== d.dataset_id) return;
    const finished = allRuns.filter((r) => r.kind === d.kind && r.status === "succeeded" && jobOf(r) === jobOf(d));
    const baseLabel = h("label", { class: "ft-wide" }, "Base model", base);
    const startFrom = h("select", { name: "from_run" }, h("option", { value: "" }, "The base model (a fresh start)"), finished.map((r) => h("option", { value: r.run_id }, `Continue ${runOption(r, metric)}`)));
    startFrom.onchange = () => { baseLabel.hidden = !!startFrom.value; };
    const replayable = (S.overview.datasets || []).filter((x) => x.kind === d.kind && x.dataset_id !== d.dataset_id && !x.test_set && sameJob(x, d));
    const share = h("label", { hidden: true }, "Share of it", h("input", { type: "number", name: "replay_fraction", min: 0.1, max: 1, step: 0.1, value: 0.5 }));
    const replaySelect = h("select", { name: "replay" }, h("option", { value: "" }, "None"), replayable.map((x) => h("option", { value: x.dataset_id }, `${x.name} (${x.counts.split.train || 0} train)`)));
    replaySelect.onchange = () => { share.hidden = !replaySelect.value; };
    const note = h("span", { class: "ft-muted", role: "status" });
    const embedOrDecision = d.kind !== "reranker";
    const form = h(
      "form",
      {
        onsubmit: async (event) => {
          event.preventDefault();
          const f = event.target.elements;
          const params = { epochs: Number(f.epochs.value), batch_size: Number(f.batch_size.value), learning_rate: Number(f.learning_rate.value), max_length: Number(f.max_length.value) };
          if (f.lora_r) params.lora_r = Number(f.lora_r.value);
          const replay = f.replay.value ? [{ dataset_id: f.replay.value, fraction: Number(f.replay_fraction.value) }] : null;
          const button = event.target.querySelector("button[type=submit]");
          await busy(button, async () => {
            try {
              const run = await api(`/api/finetune/datasets/${d.dataset_id}/runs`, { method: "POST", body: { base_model: base.value, params, backend: f.backend.value, from_run: startFrom.value || null, replay, label: f.label.value } });
              S.runId = run.run_id;
              S.events = [];
              S.after = 0;
              await refreshOverview();
              await renderTrain();
            } catch (error) {
              say(note, error.message, true);
            }
          });
        },
      },
      h("div", { class: "ft-row" }, h("label", { class: "ft-wide" }, "Start from", startFrom), baseLabel),
      replayable.length
        ? h("div", { class: "ft-row" }, h("label", { class: "ft-wide", title: "Mix in training examples from an earlier dataset so the model keeps what it learned there." }, "Replay earlier data (keeps what it learned)", replaySelect), share)
        : h("input", { type: "hidden", name: "replay", value: "" }),
      h("div", { class: "ft-row" }, h("label", { class: "ft-wide" }, "Name (optional)", h("input", { name: "label", maxlength: 40, placeholder: "e.g. round 3; otherwise the next version number" }))),
      h(
        "details",
        { class: "ft-advanced" },
        h("summary", {}, "Advanced settings"),
        h(
          "div",
          { class: "ft-row" },
          h("label", {}, "Trainer", h("select", { name: "backend" }, h("option", { value: "auto" }, "Automatic"), h("option", { value: "local" }, "This machine (PEFT)"), h("option", { value: "unsloth", disabled: !S.overview.environment.unsloth.available }, "Unsloth (NVIDIA)"))),
          h("label", {}, "Epochs", h("input", { type: "number", name: "epochs", min: 1, max: 20, step: 1, value: 3 })),
          h("label", {}, "Batch size", h("input", { type: "number", name: "batch_size", min: 1, max: 256, value: 8 })),
          h("label", {}, "Learning rate", h("input", { type: "number", name: "learning_rate", step: "0.00001", min: "0.000001", value: d.kind === "reranker" ? 0.00002 : 0.0002 })),
          embedOrDecision ? h("label", {}, "LoRA rank", h("input", { type: "number", name: "lora_r", min: 4, max: 128, value: 16 })) : null,
          h("label", {}, "Max tokens", h("input", { type: "number", name: "max_length", min: 64, max: 4096, step: 32, value: 256 })),
        ),
      ),
      h("div", { class: "ft-actions" }, h("button", { class: "btn btn-primary", type: "submit", disabled: readOnly || !S.overview.environment.ready }, "Start training"), note),
      h("p", { class: "ft-muted" }, "Trains on the accepted train examples, checks validation each epoch, and scores the test examples before and after. It keeps running if you close this page."),
    );
    const runsBox = h("div", { class: "ft-run-list" });
    const live = h("section", { class: "ft-panel", id: "ft-live" });
    fill(tab, envLine(), h("div", { class: "ft-grid2" }, h("section", { class: "ft-panel" }, h("h2", {}, "Train a new version"), form), h("section", { class: "ft-panel" }, h("div", { class: "ft-panel-head" }, h("h2", {}, "Versions trained on this dataset"), h("a", { href: "/fine-tuning/models", class: "ft-link" }, "All versions of this job →")), runsBox)), live);
    S.runs = allRuns.filter((r) => r.dataset_id === d.dataset_id);
    if (!S.runId && S.runs.length) S.runId = S.runs[0].run_id;
    fill(runsBox,
      ...(S.runs.length
        ? S.runs.map((r) =>
            h(
              "button",
              { class: "ft-run", type: "button", "aria-current": r.run_id === S.runId ? "true" : "false", onclick: () => { S.runId = r.run_id; S.events = []; S.after = 0; renderTrain(); } },
              h("span", { class: "ft-run-name" }, h("b", {}, r.label), r.lineage ? h("span", { class: "ft-muted" }, ` ${r.lineage}`) : null, h("br"), h("small", {}, `${r.short_base} · ${(r.created_at || "").slice(0, 16).replace("T", " ")}`)),
              h("span", { class: "ft-run-score" }, statusPill(r.status), h("br"), r.status === "succeeded" ? h("small", {}, `${metricValue(metric, (r.metrics_before || {})[metric])} → ${metricValue(metric, (r.metrics_after || {})[metric])}`) : null),
            ),
          )
        : [h("p", { class: "ft-muted" }, "No versions yet. Accept some examples, then start training.")]),
    );
    if (S.runId) followRun();
  }

  async function followRun() {
    clearTimeout(S.timers.run);
    const live = document.getElementById("ft-live");
    if (!live || S.tab !== "train") return;
    let run;
    try {
      run = await api(`/api/finetune/runs/${S.runId}`);
      const { events, next } = await api(`/api/finetune/runs/${S.runId}/events?after=${S.after}`);
      S.events.push(...events);
      S.after = next;
    } catch (error) {
      fill(live, h("p", { class: "ft-down" }, error.message));
      return;
    }
    if (!document.getElementById("ft-live") || S.tab !== "train") return;
    const p = run.progress || {};
    const done = p.total ? Math.round((p.step / p.total) * 100) : 0;
    const eta = p.eta_seconds ? `${Math.round(p.eta_seconds / 60)}m ${Math.round(p.eta_seconds % 60)}s left` : "";
    const note = h("span", { class: "ft-muted", role: "status" });
    const rename = readOnly
      ? null
      : h("button", { class: "btn btn-sm", type: "button", onclick: () => {
          const input = h("input", { value: run.label, maxlength: 40, "aria-label": "Version name" });
          const save = async () => {
            try {
              await api(`/api/finetune/runs/${run.run_id}`, { method: "PATCH", body: { label: input.value } });
              await refreshOverview();
              loadModels();
              renderTrain();
            } catch (error) {
              say(note, error.message, true);
            }
          };
          input.addEventListener("keydown", (ev) => { if (ev.key === "Enter") save(); });
          fill(titleBox, input, h("button", { class: "btn btn-sm btn-primary", type: "button", onclick: save }, "Save"), h("span", { class: "ft-muted" }, "Empty: the version number"));
          input.focus();
        } }, "Rename");
    const titleBox = h("h2", {}, `${run.label} · ${run.short_base}`, run.lineage ? h("span", { class: "ft-muted ft-lineage" }, ` ${run.lineage}`) : null);
    fill(live,
      h("div", { class: "ft-panel-head" }, titleBox, h("span", { class: "ft-actions" }, statusPill(run.status), h("span", { class: "ft-muted" }, `${run.backbone} · ${run.backend}${run.device ? " · " + run.device : ""}`), rename)),
      h("div", { class: "ft-progress" }, h("span", {}, p.step ? `Step ${p.step} of ${p.total} · loss ${fmt(p.loss, 4)} ${eta ? "· " + eta : ""}` : p.message || "Starting…"), h("div", { class: "ft-progress-track" }, h("div", { class: "ft-progress-fill", style: `width:${run.status === "succeeded" ? 100 : done}%` }))),
      run.error ? h("p", { class: "ft-down" }, run.error) : null,
      curveCharts(curvesFromEvents(run, S.events)),
      run.status === "succeeded" ? h("ul", { class: "ft-notes", id: "ft-live-notes" }) : null,
      h("details", { class: "ft-advanced" }, h("summary", {}, "Log"), h("div", { class: "ft-log" }, S.events.filter((e) => e.type !== "step").map((e) => h("div", {}, `${e.timestamp.slice(11, 19)}  ${e.type === "eval" ? `${e.data.phase}: ` + Object.entries(e.data.metrics || {}).filter(([, v]) => typeof v === "number").slice(0, 5).map(([k, v]) => `${k} ${fmt(v)}`).join(", ") : e.data.message || e.data.status || ""}`)))),
      h("div", { class: "ft-actions" },
        run.status === "running" && !readOnly ? h("button", { class: "btn btn-sm btn-danger", type: "button", onclick: async () => { await api(`/api/finetune/runs/${run.run_id}/cancel`, { method: "POST" }); followRun(); } }, "Cancel run") : null,
        run.status === "succeeded" ? h("button", { class: "btn btn-sm btn-primary", type: "button", onclick: () => { S.runId = run.run_id; setTab("results"); } }, "See results") : null,
        note),
    );
    if (run.status === "succeeded") {
      api(`/api/finetune/runs/${run.run_id}/curves`).then((c) => {
        const notes = document.getElementById("ft-live-notes");
        if (notes) fill(notes, c.convergence.map((n) => h("li", {}, n)));
      }).catch(() => {});
    }
    if (run.status === "running" || run.status === "queued") S.timers.run = setTimeout(followRun, 2000);
    else if (S.runs.find((r) => r.run_id === run.run_id && r.status !== run.status)) {
      await refreshOverview();
      loadModels();
      renderTrain();
    }
  }

  // -- results tab -----------------------------------------------------------------
  async function renderResults() {
    const tab = document.getElementById("ft-tab");
    const d = S.dataset;
    let runs;
    try {
      runs = (await api(`/api/finetune/runs?dataset_id=${d.dataset_id}`)).runs.filter((r) => r.status === "succeeded");
    } catch (error) {
      fill(tab, h("p", { class: "ft-down" }, error.message));
      return;
    }
    if (S.tab !== "results" || !S.dataset || S.dataset.dataset_id !== d.dataset_id) return;
    if (!runs.length) {
      fill(tab, h("div", { class: "empty-state" }, h("strong", {}, "No finished versions yet"), h("p", {}, "Train one; its scores before and after appear here."), h("button", { class: "btn btn-primary", type: "button", onclick: () => setTab("train") }, "Go to Train")));
      return;
    }
    if (!runs.find((r) => r.run_id === S.runId)) S.runId = runs[0].run_id;
    const run = runs.find((r) => r.run_id === S.runId);
    const select = h("select", { "aria-label": "Version", onchange: () => { S.runId = select.value; renderResults(); } }, runs.map((r) => h("option", { value: r.run_id }, `${r.label}${r.lineage ? ` (${r.lineage})` : ""} · ${r.short_base}`)));
    select.value = S.runId;
    const verdictBox = h("section", { class: "ft-panel ft-verdict-panel" }, h("p", { class: "ft-muted" }, "Loading the comparison…"));
    fill(tab,
      h("div", { class: "ft-results-head" }, h("label", { class: "ft-inline" }, "Version ", select)),
      verdictBox,
      trainingPanel(run),
      reportPanel(run),
      dataPanel(run),
      historyPanel(run),
      h("div", { class: "ft-grid2" }, h("section", { class: "ft-panel" }, h("h2", {}, "Try it"), tryIt(run)), usePanel(run)),
    );
    loadVerdict(run, verdictBox);
  }

  async function loadVerdict(run, box) {
    let v;
    try {
      v = await api(`/api/finetune/runs/${run.run_id}/verdict`);
    } catch (error) {
      fill(box, h("p", { class: "ft-down" }, error.message));
      return;
    }
    const note = h("span", { class: "ft-muted ft-note", role: "status" });
    const done = async () => {
      await refreshOverview();
      loadModels();
      renderResults();
    };
    const headline = { better: "fine-tuning helped", worse: "worse than what it is compared with", unclear: "no clear difference yet", untested: "not yet tested for chance" }[v.verdict] || "before and after";
    const sig = document.getElementById("ft-train-sig");
    if (sig) {
      const t = v.training_significance;
      sig.textContent = t
        ? `Is the change in ${metricWords(v.evidence.metric, jobOf(S.dataset)).toLowerCase()} more than chance? ${t.p < 0.05 ? "Yes" : "Not clearly"}: ${chanceNote(t, v.evidence.metric)} (paired sign test).`
        : "This run was trained before each example's result was kept, so its change cannot be tested for chance.";
    }
    fill(box,
      h("div", { class: "ft-panel-head" }, h("h2", {}, `${v.version}: ${headline}`), v.card_title ? h("span", { class: "ft-muted" }, `Job: ${v.card_title.toLowerCase()}`) : null),
      compareRows(v),
      caption(v, v.card || jobOf(S.dataset)),
      verdictLine(v),
      speedLine(v, done),
      v.card
        ? h("div", { class: "ft-actions" }, swapButtons(v.card, v, false, note, done), note)
        : h("p", { class: "ft-muted" }, "This task has no slot in MemoRizz's memory layer; use the model from Python or the CLI (see Use it below)."),
    );
  }

  function reportPanel(run) {
    const d = S.dataset;
    const before = run.metrics_before || {};
    const after = run.metrics_after || {};
    const counts = (run.progress && run.progress.counts) || {};
    const scored = before.count ?? before.queries ?? counts.test ?? counts.validation;
    const keys = ["accuracy", "macro_f1", "ndcg@10", "mrr@10", "recall@1", "recall@5", "recall@10", "map", "ece", "nll", "brier", "order_flip_rate"].filter((k) => k in before || k in after);
    const metrics = h(
      "table",
      { class: "ft-metrics" },
      h("thead", {}, h("tr", {}, h("th", {}, "Measure"), h("th", {}, run.parent_label ? `Started from ${run.parent_label}` : "Untrained base"), h("th", {}, run.label), h("th", {}, "Change"))),
      h(
        "tbody",
        {},
        keys.map((k) => {
          const m = METRICS[k] || { words: k, short: k, up: true, pct: false };
          const a = before[k], b = after[k];
          const delta = typeof a === "number" && typeof b === "number" ? b - a : null;
          const good = delta !== null && (m.up ? delta > 0 : delta < 0);
          const change = delta === null ? "–" : m.pct ? `${delta > 0 ? "+" : ""}${(delta * 100).toFixed(1)} pts` : `${delta > 0 ? "+" : ""}${delta.toFixed(3)}`;
          return h(
            "tr",
            {},
            h("td", {}, h("span", {}, metricWords(k, jobOf(d))), h("small", {}, `${m.short} · ${m.up ? "higher" : "lower"} is better`)),
            h("td", {}, metricValue(k, a)),
            h("td", {}, metricValue(k, b)),
            h("td", { class: delta === null || Math.abs(delta) < 1e-9 ? "" : good ? "ft-up" : "ft-down" }, change),
          );
        }),
      ),
    );
    const conf = after.confusion;
    let confusion = null;
    let weak = [];
    if (conf) {
      const n = conf.labels.length;
      const rowSum = (i) => conf.matrix[i].reduce((a, b) => a + b, 0);
      const colSum = (j) => conf.matrix.reduce((a, row) => a + row[j], 0);
      const keep = conf.labels.map((_, i) => i).filter((i) => rowSum(i) + colSum(i) > 0);
      weak = keep.filter((i) => rowSum(i) > conf.matrix[i][i]).sort((a, b) => (rowSum(b) - conf.matrix[b][b]) / rowSum(b) - (rowSum(a) - conf.matrix[a][a]) / rowSum(a)).map((i) => conf.labels[i]);
      const max = Math.max(1, ...conf.matrix.flat());
      confusion = h(
        "div",
        { class: "ft-confusion-box" },
        h("h3", {}, "Where it goes wrong"),
        h("p", { class: "ft-muted" }, `Rows: the right option · columns: what ${run.label} picked. Green is right; red is a mix-up.${keep.length < n ? ` Options with no test examples are hidden (${n - keep.length}).` : ""}`),
        h("div", { class: "ft-table-wrap" }, h("table", { class: "ft-confusion" }, h("thead", {}, h("tr", {}, h("th", {}, ""), keep.map((j) => h("th", { class: "rot", title: conf.labels[j] }, conf.labels[j])))), h("tbody", {}, keep.map((i) => h("tr", {}, h("th", { class: "ft-row-label" }, conf.labels[i]), keep.map((j) => { const v = conf.matrix[i][j]; return h("td", { style: v ? `background: color-mix(in srgb, var(${i === j ? "--accent-secondary" : "--accent-danger"}) ${Math.round((v / max) * 70) + 10}%, transparent)` : null, title: `${conf.labels[i]} → ${conf.labels[j]}: ${v}` }, v || ""); })))))),
        weak.length && !readOnly && !d.test_set
          ? h("div", { class: "ft-actions" }, h("button", { class: "btn btn-sm", type: "button", onclick: () => { S.preselect = weak; setTab("data"); } }, `Generate more for what it gets wrong (${weak.slice(0, 4).join(", ")}${weak.length > 4 ? "…" : ""})`))
          : null,
      );
    }
    return h(
      "section",
      { class: "ft-panel" },
      h("h2", {}, "Training report"),
      h("p", { class: "ft-muted" }, `Scored when it trained, on ${scored ?? "the"} test ${d.kind === "decision" ? "examples" : "questions"} it never saw${counts.train ? ` (trained on ${counts.train})` : ""}. Small test sets swing a lot; add data for a firmer comparison.`),
      h("div", { class: "ft-table-wrap" }, metrics),
      h("p", { class: "ft-muted", id: "ft-train-sig" }),
      // Runs trained before the base was calibrated too record no temperature.
      d.kind === "decision" && keys.some((k) => ["nll", "brier", "ece"].includes(k)) && before.temperature === undefined
        ? h("p", { class: "ft-muted" }, "Trained before its starting point was calibrated: the changes in surprise, probability error and the confidence gap partly come from calibrating only the trained model. Accuracy is unaffected.")
        : null,
      confusion,
    );
  }

  function usePanel(run) {
    const d = S.dataset;
    const job = jobOf(d);
    const how = {
      memory_routing: ["Swap it in (above) and writes that ask for memory_type \"auto\" go where it decides.", "Python: ft.route(text) · CLI: memorizz finetune route \"…\" · MCP: memorizz_finetune_decide"],
      embedding: ["Swap it in (above): MemoRizz switches its embedding model and re-embeds every stored memory. Swap back the same way.", `CLI: memorizz finetune swap embedding ${run.run_id}`],
      reranking: ["Swap it in (above): recall and search rescore what vector search found.", `CLI: memorizz finetune swap reranking ${run.run_id} · Python: ft.rank(run_id, query, passages)`],
      other: ["Ask it directly from Python or the CLI.", `Python: ft.decide("${run.run_id}", state) · CLI: memorizz finetune decide ${run.run_id} "…"`],
    }[job];
    return h(
      "section",
      { class: "ft-panel" },
      h("h2", {}, "Use it"),
      h("p", {}, how[0]),
      h("p", { class: "ft-muted" }, how[1]),
      h("p", { class: "ft-muted" }, "Model folder on this machine:"),
      h("pre", { class: "ft-code" }, (run.progress && run.progress.model_dir) || run.output_dir + "/model"),
    );
  }

  // -- comparing versions on fixed sets -------------------------------------------
  function renderScores() {
    fill(document.getElementById("ft-tab"), historyPanel(null));
  }

  function historyPanel(run) {
    const d = S.dataset;
    const metric = primaryMetric(d);
    const panel = h("section", { class: "ft-panel" });
    const sets = (S.overview.datasets || [])
      .filter((x) => sameJob(x, d))
      .sort((a, b) => Number(b.test_set) - Number(a.test_set) || String(a.created_at).localeCompare(String(b.created_at)));
    const datasetPick = h("select", {}, sets.map((x) => h("option", { value: x.dataset_id }, `${x.name}${x.test_set ? " (test set)" : ""}`)));
    datasetPick.value = d.test_set || !sets.some((x) => x.test_set) ? d.dataset_id : sets[0].dataset_id;
    const split = h("select", {});
    const fillSplits = () => {
      const chosen = sets.find((x) => x.dataset_id === datasetPick.value);
      const options = chosen && chosen.test_set ? [["all", "All its examples"]] : [["test", "Test"], ["held_out", "Test and validation"], ["all", "All accepted"]];
      fill(split, options.map(([v, l]) => h("option", { value: v }, l)));
    };
    datasetPick.onchange = fillSplits;
    fillSplits();
    const modelPick = h("select", {}, h("option", { value: "" }, "Loading versions…"));
    const note = h("span", { class: "ft-muted", role: "status" });
    const table = h("div", { class: "ft-table-wrap" });
    const scoreButton = h("button", { class: "btn", type: "button", disabled: readOnly, onclick: async (event) => {
      if (!modelPick.value) return;
      const [type, value] = modelPick.value.split(/:(.*)/s);
      const body = { dataset_id: datasetPick.value, split: split.value, run_id: type === "run" ? value : null, base_model: type === "base" ? value : null, jev_model: type === "jev" ? value : null, in_use: type === "in_use" };
      event.target.disabled = true;
      say(note, "Scoring…");
      try {
        const { job_id } = await api("/api/finetune/evaluate", { method: "POST", body });
        const poll = async () => {
          const job = await api(`/api/finetune/jobs/${job_id}`);
          if (job.status === "running") return setTimeout(poll, 1500);
          event.target.disabled = readOnly;
          if (job.status === "failed") return say(note, job.error, true);
          say(note, `${metricValue(metric, job.report.metrics[metric] ?? job.report.metrics.accuracy)} on ${job.report.count} examples (${job.report.seconds}s).`);
          load();
          loadModels();
        };
        poll();
      } catch (error) {
        event.target.disabled = readOnly;
        say(note, error.message, true);
      }
    } }, "Score");
    const load = async () => {
      const [{ evaluations }, { runs }] = await Promise.all([api("/api/finetune/evaluations"), api("/api/finetune/runs")]);
      const versions = runs.filter((r) => r.status === "succeeded" && sameJob(r, d));
      const bases = [...new Set(versions.map((r) => r.base_model))];
      const keepPick = modelPick.value || (run ? `run:${run.run_id}` : versions[0] ? `run:${versions[0].run_id}` : "");
      fill(modelPick,
        versions.map((r) => h("option", { value: `run:${r.run_id}` }, `${r.label}${r.lineage ? ` (${r.lineage})` : ""} · ${r.short_base}`)),
        bases.map((b) => h("option", { value: `base:${b}` }, `${b.split("/").pop()}, untrained`)),
        d.kind === "decision" ? h("option", { value: "jev:jev-latest", disabled: !S.overview.jev_available }, "Jev, zero-shot (TypeSafe)") : h("option", { value: "in_use:" }, "The setup in use now"),
      );
      if (keepPick) modelPick.value = keepPick;
      const ids = new Set(sets.map((x) => x.dataset_id));
      const rows = evaluations.filter((e) => ids.has(e.dataset_id) && (e.metrics[metric] !== undefined || e.metrics.accuracy !== undefined));
      if (!rows.length) {
        fill(table, h("p", { class: "ft-muted" }, "Nothing scored yet. Score each version, its untrained base and the setup in use on the same set to see what training changed and what it kept."));
        return;
      }
      const rank = (id) => (id === d.dataset_id ? 0 : sets.find((x) => x.dataset_id === id)?.test_set ? 1 : 2);
      const columns = [...new Map(rows.map((e) => [`${e.dataset_id}|${e.split}`, `${e.dataset_name}${SPLIT_WORDS[e.split] ? ` · ${SPLIT_WORDS[e.split]}` : ""}`])).entries()]
        .sort(([a], [b]) => rank(a.split("|")[0]) - rank(b.split("|")[0]));
      const order = { base: 0, jev: 1, in_use: 2, run: 3 };
      const version = Object.fromEntries(versions.map((r) => [r.run_id, r.version || 0]));
      const models = [...new Map(rows.map((e) => [e.model.label, e.model])).entries()]
        .sort(([, a], [, b]) => (order[a.type] ?? 4) - (order[b.type] ?? 4) || (version[a.run_id] || 0) - (version[b.run_id] || 0))
        .map(([label]) => label);
      const latest = {};
      for (const e of rows) latest[`${e.model.label}|${e.dataset_id}|${e.split}`] = e;
      fill(table,
        h("table", { class: "ft-metrics ft-history" },
          h("thead", {}, h("tr", {}, h("th", {}, "Model"), columns.map(([, label]) => h("th", {}, label)))),
          h("tbody", {}, models.map((label) => h("tr", {}, h("td", {}, label), columns.map(([key]) => {
            const e = latest[`${label}|${key}`];
            const value = e ? e.metrics[metric] ?? e.metrics.accuracy : null;
            return h("td", { title: e ? `${e.count} examples · ${new Date(e.created_at).toLocaleString()}` : null }, e ? metricValue(metric, value) : "–");
          }))))),
        h("p", { class: "ft-muted" }, `${metricWords(metric, jobOf(d))} (${METRICS[metric].short}). Read a row for what one version can do; read a column top to bottom for what training changed. Hover a score for its size and date.`),
      );
    };
    fill(panel,
      h("h2", {}, d.test_set ? "Scores on this test set" : "Across versions"),
      h("p", { class: "ft-muted" }, "Continual learning is judged on fixed sets: does a new version gain where it was trained without losing what earlier versions could do?" + (sets.some((x) => x.test_set) ? "" : " Mark a dataset as a test set (More ▾) to keep it out of training.")),
      h("div", { class: "ft-row" }, h("label", { class: "ft-wide" }, "Set", datasetPick), h("label", {}, "Examples", split), h("label", { class: "ft-wide" }, "Model", modelPick), scoreButton, note),
      table,
    );
    load().catch((error) => say(note, error.message, true));
    return panel;
  }

  function tryIt(run) {
    const out = h("div", { class: "ft-probs" });
    const isPointwiseDecision = run.kind === "decision" && run.task !== "relevance";
    if (isPointwiseDecision) {
      const state = autosize(h("textarea", { rows: 2, placeholder: "e.g. Remember that Jordan from finance always wants reports as PDF.", class: "ft-full", "aria-label": "State" }));
      return h("div", {}, state, h("div", { class: "ft-actions" }, h("button", { class: "btn", type: "button", onclick: (event) => busy(event.target, async () => {
        try {
          const { answers } = await api(`/api/finetune/runs/${run.run_id}/decide`, { method: "POST", body: { state: state.value } });
          const answer = Object.values(answers)[0];
          const probs = Object.entries(answer.probabilities || { false: 1 - answer.noul, true: answer.noul }).sort((a, b) => b[1] - a[1]);
          fill(out, h("p", {}, h("strong", {}, answer.choice ?? String(answer.answer ?? answer.level)), " ", h("span", { class: "ft-muted" }, `confidence ${pct(answer.confidence ?? answer.noul)}`)), ...probs.slice(0, 6).map(([k, p]) => h("div", { class: "ft-bar" }, h("span", {}, k), h("span", { class: "ft-bar-track" }, h("span", { class: "ft-bar-fill", style: `width:${p * 100}%` })), h("span", { class: "ft-bar-num" }, pct(p)))));
        } catch (error) {
          say(out, error.message, true);
        }
      }) }, "Decide")), out);
    }
    const query = h("input", { placeholder: "A question", class: "ft-full", "aria-label": "Question" });
    const passages = autosize(h("textarea", { rows: 3, placeholder: "Candidate memories, one per line", class: "ft-full", "aria-label": "Candidate memories" }));
    return h("div", {}, query, passages, h("div", { class: "ft-actions" }, h("button", { class: "btn", type: "button", onclick: (event) => busy(event.target, async () => {
      try {
        const { ranking } = await api(`/api/finetune/runs/${run.run_id}/rank`, { method: "POST", body: { query: query.value, passages: passages.value.split("\n").map((s) => s.trim()).filter(Boolean) } });
        fill(out, ...ranking.map((r, i) => h("div", { class: "ft-bar" }, h("span", { title: r.passage }, `${i + 1}. ${r.passage}`), h("span", { class: "ft-bar-track" }, h("span", { class: "ft-bar-fill", style: `width:${Math.max(2, Math.min(100, r.score * 100))}%` })), h("span", { class: "ft-bar-num" }, fmt(r.score)))));
      } catch (error) {
        say(out, error.message, true);
      }
    }) }, "Rank")), out);
  }

  // Close the More menu when clicking elsewhere.
  document.addEventListener("click", (event) => {
    for (const menu of document.querySelectorAll(".ft-menu[open]")) if (!menu.contains(event.target)) menu.open = false;
  });

  loadOverview();
})();

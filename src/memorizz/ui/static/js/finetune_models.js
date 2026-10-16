(() => {
  const app = document.getElementById("mr-app");
  if (!app) return;
  const readOnly = app.dataset.readOnly === "true";
  const open = new Set(); // version cards left open across reloads
  const openAgents = new Set(); // per-agent lists left open across reloads
  const curves = {}; // run id -> its recorded curves (fetched once)
  const getCurves = (runId) => (curves[runId] ||= api(`/api/finetune/runs/${runId}/curves`));
  const validationOf = (c) => c.validation.map((v) => [v.epoch, v.metrics[c.metric] ?? v.metrics.accuracy]).filter((p) => typeof p[1] === "number");

  // -- helpers (as on the Fine-tuning page) -----------------------------------
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
  const fill = (node, ...kids) => node.replaceChildren(...kids.flat(Infinity).filter((k) => k !== null && k !== undefined && k !== false));
  const api = async (path, opts = {}) => {
    const init = { ...opts };
    if (opts.body !== undefined) {
      init.body = JSON.stringify(opts.body);
      init.headers = { "Content-Type": "application/json" };
    }
    const response = await fetch(path, init);
    const data = (response.headers.get("content-type") || "").includes("json") ? await response.json() : await response.text();
    if (!response.ok) throw new Error((data && data.detail) || `Request failed (${response.status})`);
    return data;
  };
  const pct = (v) => (v === null || v === undefined ? "–" : `${(v * 100).toFixed(1)}%`);
  const METRIC = { accuracy: "right answers", "ndcg@10": "right memory near the top (nDCG@10)" };
  const when = (iso) => {
    if (!iso) return "–";
    const d = new Date(iso);
    return d.toLocaleDateString(undefined, { day: "numeric", month: "short" }) + " " + d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
  };
  const duration = (s) => (s === null || s === undefined ? "" : s < 90 ? `${s}s` : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`);
  const size = (mb) => (mb === null || mb === undefined ? "–" : mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${Math.round(mb)} MB`);
  const chance = (t) => (t ? (t.p < 0.05 ? `unlikely to be chance (p ${t.p < 0.001 ? "< 0.001" : "= " + t.p})` : `could be chance (p = ${t.p})`) : "not tested for chance");
  const say = (node, text, error = false) => {
    node.textContent = text;
    node.classList.toggle("ft-down", !!error);
  };

  // -- load ----------------------------------------------------------------------
  async function load() {
    let data;
    try {
      data = await api("/api/finetune/registry");
    } catch (error) {
      fill(document.getElementById("mr-summary"), h("span", { class: "ft-down" }, error.message));
      return;
    }
    const t = data.totals;
    const inUse = data.jobs.reduce((n, j) => n + j.versions.filter((v) => v.in_use).length, 0);
    fill(document.getElementById("mr-summary"),
      h("span", {}, h("b", {}, t.versions), ` fine-tuned version${t.versions === 1 ? "" : "s"}`),
      h("span", {}, h("b", {}, inUse), " in use"),
      t.failed ? h("span", {}, h("b", {}, t.failed), " failed or canceled") : null,
      h("span", {}, h("b", {}, size(t.disk_mb)), " of model files"),
      h("a", { href: "/fine-tuning" }, "Train a new version →"),
    );
    fill(document.getElementById("mr-jobs"), data.jobs.map(jobSection));
  }

  // -- one job ---------------------------------------------------------------
  // Versions in family order: each under the one it continued from.
  function familyOrder(versions) {
    const ids = new Set(versions.map((v) => v.run_id));
    const children = {};
    const roots = [];
    for (const v of [...versions].sort((a, b) => (a.version || 0) - (b.version || 0))) {
      if (v.parent_run_id && ids.has(v.parent_run_id)) (children[v.parent_run_id] ||= []).push(v);
      else roots.push(v);
    }
    const out = [];
    const walk = (v, depth) => {
      out.push({ v, depth });
      for (const child of children[v.run_id] || []) walk(child, depth + 1);
    };
    roots.forEach((v) => walk(v, 0));
    return out;
  }

  function jobSection(job) {
    const rows = familyOrder(job.versions);
    const table = h(
      "table",
      { class: "ft-table mr-table" },
      h("thead", {}, h("tr", {}, h("th", {}, "Version"), h("th", {}, "Trained on"), h("th", {}, "Training result"), h("th", {}, "Best fixed-set score"), h("th", {}, "Trained"), h("th", { class: "mr-num" }, "Size"))),
      h("tbody", {}, rows.map(({ v, depth }) => versionRows(job, v, depth))),
    );
    return h(
      "section",
      { class: "ft-panel mr-job", id: `job-${job.key.replace(/[^a-z0-9]/gi, "-")}` },
      h("div", { class: "ft-panel-head" },
        h("h2", {}, job.title),
        job.memory_job && job.in_use ? h("span", { class: "mr-inuse" }, "In use now: ", h("b", {}, job.in_use.label)) : h("span", { class: "ft-muted" }, job.memory_job ? "" : "Not part of the memory layer")),
      h("p", { class: "ft-muted mr-note" }, job.note),
      job.versions.length ? h("div", { class: "ft-table-wrap" }, table) : h("p", { class: "ft-muted" }, "No fine-tuned versions yet."),
      compareSection(job),
      job.agents ? agentChoices(job) : null,
      job.key === "embedding" && job.agents_total
        ? h("p", { class: "ft-muted mr-agents-note" }, `All ${job.agents_total} agents in this store use the embedding model in use: every memory is one model's vectors, so it cannot differ by agent.`)
        : null,
      job.failed.length
        ? h("details", { class: "mr-more" }, h("summary", {}, `${job.failed.length} failed or canceled run${job.failed.length === 1 ? "" : "s"}`),
            h("ul", { class: "mr-failed" }, job.failed.map((f) => h("li", {}, h("b", {}, f.status), ` · ${f.short_base} on “${f.dataset_name}” · ${when(f.created_at)}`, f.error ? h("div", { class: "ft-down mr-error" }, f.error) : null, readOnly ? null : h("button", { class: "ft-link", type: "button", onclick: (ev) => remove(f, ev.target) }, "Delete record")))))
        : null,
      h("details", { class: "mr-more" }, h("summary", {}, `Base models to start from (${job.bases.length})`),
        h("div", { class: "ft-table-wrap" }, h("table", { class: "ft-table" },
          h("thead", {}, h("tr", {}, h("th", {}, "Model"), h("th", {}, "Size"), h("th", {}, "On this machine"), h("th", {}, "Notes"))),
          h("tbody", {}, job.bases.map((b) => h("tr", {},
            h("td", {}, h("b", {}, b.label), b.recommended && b.recommended.length ? h("span", { class: "ft-tag mr-tag" }, "recommended") : null, h("small", { class: "mr-sub" }, b.model_id)),
            h("td", {}, b.params || "–"),
            h("td", {}, b.on_disk ? `✓ ${size(b.size_mb)}` : h("span", { class: "ft-muted" }, "downloads on first use")),
            h("td", { class: "ft-muted mr-notes" }, b.notes || ""),
          )))))),
    );
  }

  function versionRows(job, v, depth) {
    const t = v.training;
    const best = v.scores[0];
    const badges = [
      v.in_use ? h("span", { class: "pill pill--success" }, "in use") : null,
      v.best && !v.in_use ? h("span", { class: "pill pill--info" }, "best") : null,
      v.status !== "succeeded" ? h("span", { class: "pill pill--warning" }, v.status) : null,
    ];
    const detail = h("tr", { class: "mr-detail", hidden: !open.has(v.run_id) }, h("td", { colspan: 6 }, open.has(v.run_id) ? card(job, v) : null));
    const toggle = () => {
      const show = detail.hidden;
      detail.hidden = !show;
      row.setAttribute("aria-expanded", String(show));
      if (show) {
        open.add(v.run_id);
        fill(detail.firstChild, card(job, v));
      } else open.delete(v.run_id);
    };
    const row = h(
      "tr",
      { class: "mr-row", tabindex: "0", role: "button", "aria-expanded": String(open.has(v.run_id)), onclick: toggle, onkeydown: (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); toggle(); } } },
      h("td", {}, h("div", { class: "mr-version", style: `--depth:${depth}` },
        depth ? h("span", { class: "mr-branch", "aria-hidden": "true" }, "↳") : null,
        h("span", {}, h("b", {}, v.label), " ", badges, h("small", { class: "mr-sub" }, `${v.short_base}${v.lineage ? " · " + v.lineage : ""}`)))),
      h("td", {}, v.dataset_name, h("small", { class: "mr-sub" }, `${v.counts.train ?? "?"} train · ${(v.counts.test ?? 0) + (v.counts.validation ?? 0)} held out${v.replay.length ? " · + replay" : ""}`)),
      h("td", {}, t.before !== undefined && t.before !== null ? `${pct(t.before)} → ${pct(t.after)}` : pct(t.after), h("small", { class: `mr-sub ${t.significance && t.significance.p < 0.05 ? (t.after > t.before ? "ft-up" : "ft-down") : ""}` }, `from ${t.start} · ${chance(t.significance)}`)),
      h("td", {}, best ? pct(best.value) : "–", best ? h("small", { class: "mr-sub" }, `${best.count} ${best.test_set ? "test-set" : best.split} examples · ${best.dataset_name}`) : h("small", { class: "mr-sub" }, "not scored on a fixed set yet")),
      h("td", {}, when(v.created_at), h("small", { class: "mr-sub" }, [duration(v.duration_s), v.device].filter(Boolean).join(" · "))),
      h("td", { class: "mr-num" }, v.model.on_disk ? size(v.model.size_mb) : h("span", { class: "ft-muted" }, "not here")),
    );
    return [row, detail];
  }

  // -- each agent's own router or reranker -------------------------------------------
  function agentChoices(job) {
    const ownCount = () => job.agents.filter((a) => a.choice).length;
    const summary = h("summary", {});
    const setSummary = () => {
      summary.textContent = `Per-agent choices: ${ownCount()} of ${job.agents.length} agent${job.agents.length === 1 ? "" : "s"} use their own`;
    };
    setSummary();
    const versions = job.versions.filter((v) => v.status === "succeeded" && v.model.on_disk);
    const storeChoice = job.in_use ? job.in_use.label : "none";
    const none = job.key === "memory_routing" ? "None: “auto” writes go to the default store" : "None: vector search order";
    const valueOf = (choice) => (!choice ? "default" : choice.mode === "off" ? "off" : choice.run_id);
    const rows = job.agents.map((a) => {
      const note = h("span", { class: "ft-muted", role: "status" });
      const select = h(
        "select",
        { "aria-label": `${job.title} for ${a.name}`, disabled: readOnly },
        h("option", { value: "default" }, `The store's (${storeChoice})`),
        h("option", { value: "off" }, none),
        versions.map((v) => h("option", { value: v.run_id }, `${v.label} · ${v.short_base}`)),
      );
      select.value = valueOf(a.choice);
      select.onchange = async () => {
        select.disabled = true;
        try {
          const updated = await api(`/api/finetune/agents/${encodeURIComponent(a.agent_id)}/models/${job.key}`, { method: "PUT", body: { choice: select.value } });
          a.choice = updated.choices[job.key];
          setSummary();
          say(note, "Saved: applies within 30 seconds everywhere.");
        } catch (error) {
          select.value = valueOf(a.choice);
          say(note, error.message, true);
        } finally {
          select.disabled = readOnly;
        }
      };
      return h("tr", {}, h("td", {}, a.name, h("small", { class: "mr-sub" }, a.agent_id)), h("td", {}, select, h("div", {}, note)));
    });
    const details = h(
      "details",
      { class: "mr-more mr-agents", open: openAgents.has(job.key) },
      summary,
      h("p", { class: "ft-muted" }, job.key === "memory_routing"
        ? "An agent can use its own router for writes into its memory that ask for memory_type “auto”, or none. Everything else follows the store's choice."
        : "An agent can use its own reranker when it recalls (and for searches in its memory), or none. Everything else follows the store's choice."),
      h("div", { class: "ft-table-wrap" }, h("table", { class: "ft-table" }, h("thead", {}, h("tr", {}, h("th", {}, "Agent"), h("th", {}, "Uses"))), h("tbody", {}, rows))),
    );
    details.addEventListener("toggle", () => (details.open ? openAgents.add(job.key) : openAgents.delete(job.key)));
    return details;
  }

  // -- a version's card ------------------------------------------------------------
  function card(job, v) {
    const note = h("span", { class: "ft-muted", role: "status" });
    const params = Object.entries(v.params || {});
    const results = `/fine-tuning?dataset=${encodeURIComponent(v.dataset_id)}&tab=results&run=${encodeURIComponent(v.run_id)}`;
    const actions = [h("a", { class: "btn btn-sm btn-primary", href: results }, "Open results")];
    if (!readOnly) {
      if (job.memory_job && !v.in_use && v.status === "succeeded" && v.model.on_disk) {
        actions.push(h("button", { class: "btn btn-sm", type: "button", onclick: (ev) => swap(job, v, ev.target, note) }, job.key === "embedding" ? "Swap in (re-embeds memories)" : "Swap in"));
      }
      actions.push(h("button", { class: "btn btn-sm", type: "button", onclick: () => rename(v, note) }, "Rename"));
      actions.push(h("button", { class: "btn btn-sm btn-danger", type: "button", disabled: v.in_use, title: v.in_use ? "In use: swap back or swap in another version first" : null, onclick: (ev) => remove(v, ev.target, note) }, "Delete…"));
    }
    return h(
      "div",
      { class: "mr-card" },
      h("div", { class: "mr-card-grid" },
        h("section", {},
          h("h3", {}, "How it was trained"),
          h("dl", { class: "mr-dl" },
            h("dt", {}, "Data"), h("dd", {}, h("a", { href: `/fine-tuning?dataset=${encodeURIComponent(v.dataset_id)}` }, v.dataset_name), ` · ${v.counts.train ?? "?"} train, ${v.counts.validation ?? 0} validation, ${v.counts.test ?? 0} test`),
            v.replay.length ? [h("dt", {}, "Replayed"), h("dd", {}, v.replay.map((r) => `${r.examples} from “${r.dataset_name}”`).join("; "))] : null,
            h("dt", {}, "Started from"), h("dd", {}, v.parent_label ? `${v.parent_label} (continued training)` : `${v.base_model} (untrained)`),
            h("dt", {}, "Settings"), h("dd", {}, params.length ? params.map(([k, val]) => `${k.replace(/_/g, " ")} ${val}`).join(" · ") : "the trainer's defaults"),
            h("dt", {}, "Trainer"), h("dd", {}, [v.backbone, v.backend, v.device].filter(Boolean).join(" · ")),
            h("dt", {}, "When"), h("dd", {}, `${when(v.created_at)}${v.duration_s !== null ? ", " + duration(v.duration_s) : ""}`),
          )),
        h("section", {},
          h("h3", {}, "How it scores"),
          h("p", { class: "mr-train" }, `In training (${METRIC[v.training.metric] || v.training.metric}): ${pct(v.training.before)} → ${pct(v.training.after)} on its test examples, from ${v.training.start}; ${chance(v.training.significance)}.`),
          v.scores.length
            ? h("table", { class: "ft-metrics mr-scores" }, h("thead", {}, h("tr", {}, h("th", {}, "Fixed set"), h("th", {}, "Examples"), h("th", {}, "Score"))),
                h("tbody", {}, v.scores.map((s) => h("tr", {}, h("td", {}, s.dataset_name, s.test_set ? h("span", { class: "ft-tag mr-tag" }, "Test set") : null), h("td", {}, s.count), h("td", {}, pct(s.value))))))
            : h("p", { class: "ft-muted" }, "Not scored on a fixed set yet: use Compare on its results page."),
          v.speed
            ? h("p", { class: "mr-train" }, `Speed ${v.speed.unit}: ` + v.speed.rows.map((r) => `${r.role === "fine-tuned" ? v.label : `${r.label} (${r.role})`} ${r.n ? `${r.median_ms < 10 ? r.median_ms.toFixed(1) : Math.round(r.median_ms)} ms` : "adds nothing"}`).join(" · ") + ` (median of ${v.speed.rows[0].n} calls on ${v.speed.rows[0].device === "mps" ? "the Apple GPU" : v.speed.rows[0].device}).`)
            : h("p", { class: "ft-muted" }, "Speed not measured yet: Measure speed on its results page."),
        ),
        h("section", {},
          h("h3", {}, "Files"),
          h("p", { class: "ft-muted" }, v.model.on_disk ? `Model folder (${size(v.model.size_mb)}):` : "The model folder is not on this machine:"),
          h("pre", { class: "ft-code" }, v.model.dir || "–"),
          h("p", { class: "ft-muted" }, `Run ${v.run_id}`),
        )),
      runCurves(v),
      h("div", { class: "ft-actions" }, actions, h("a", { class: "ft-link", href: `${results}#ft-data` }, "The data it trained on →"), note),
    );
  }

  function runCurves(v) {
    const box = h("div", { class: "mr-curves" }, h("p", { class: "ft-muted" }, "Loading its training curves…"));
    getCurves(v.run_id)
      .then((c) => {
        const C = window.MemorizzCharts;
        const raw = c.steps.filter((x) => typeof x.loss === "number").map((x) => [x.step, x.loss]);
        const loss = C.line(raw.length > 20 ? [{ name: "loss per step", points: raw, tone: 0, faint: true }, { name: "loss (smoothed)", points: C.smooth(raw, 0.12), tone: 0 }] : [{ name: "loss", points: raw, tone: 0, dots: true }], { logY: true, xLabel: "training step", height: 170, empty: "No loss recorded." });
        const points = validationOf(c);
        const val = (m) => (m ? m[c.metric] ?? m.accuracy : undefined);
        const refs = [[c.before, "test, before", 5], [c.after, "test, after", 1]].filter(([m]) => typeof val(m) === "number").map(([m, label, tone]) => ({ y: val(m), label, tone }));
        const chart = points.length
          ? C.line([{ name: "validation", points, tone: 2, dots: true }], { asPct: true, xLabel: "epoch", height: 170, refs, xFormat: (x) => `${Math.round(x)}`, legend: true })
          : C.line(refs.length === 2 ? [{ name: "test score", points: [[0, refs[0].y], [1, refs[1].y]], tone: 1, dots: true }] : [], { asPct: true, height: 170, xFormat: (x) => (x < 0.5 ? "before" : "after"), empty: "No validation score recorded." });
        fill(box,
          h("div", {}, h("h3", {}, "Training loss"), loss),
          h("div", {}, h("h3", {}, points.length ? "Validation each epoch" : "Test score before and after"), chart),
          c.convergence.length ? h("ul", { class: "ft-notes", style: "grid-column: 1 / -1" }, c.convergence.map((n) => h("li", {}, n))) : null);
      })
      .catch((error) => fill(box, h("p", { class: "ft-muted" }, `No curves: ${error.message}`)));
    return box;
  }

  // Overlaid training curves of several versions of one job.
  function compareSection(job) {
    const done = job.versions.filter((v) => v.status === "succeeded");
    if (done.length < 2) return null;
    const chosen = new Set(done.slice(0, 4).map((v) => v.run_id));
    const charts = h("div", { class: "mr-curves" });
    const draw = async () => {
      const picked = done.filter((v) => chosen.has(v.run_id));
      if (!picked.length) return fill(charts, h("p", { class: "ft-muted" }, "Tick versions to compare."));
      fill(charts, h("p", { class: "ft-muted" }, "Loading…"));
      const all = await Promise.all(picked.map((v) => getCurves(v.run_id).catch(() => null)));
      const C = window.MemorizzCharts;
      const lossSeries = [], valSeries = [];
      picked.forEach((v, i) => {
        const c = all[i];
        if (!c) return;
        const raw = c.steps.filter((x) => typeof x.loss === "number").map((x) => [x.step, x.loss]);
        const last = raw.length ? raw[raw.length - 1][0] : 1;
        // Progress through training (0-100%), so runs of different lengths line up.
        lossSeries.push({ name: v.label, points: C.smooth(raw.map(([x, y]) => [(x / last) * 100, y]), 0.12), tone: i % 5 });
        const pts = validationOf(c);
        if (pts.length) valSeries.push({ name: v.label, points: pts, tone: i % 5, dots: true });
      });
      fill(charts,
        h("div", {}, h("h3", {}, "Loss through training (smoothed)"), C.line(lossSeries, { logY: true, xLabel: "% of training", height: 190, legend: true, xFormat: (x) => `${Math.round(x)}%` })),
        h("div", {}, h("h3", {}, "Validation each epoch"), valSeries.length
          ? C.line(valSeries, { asPct: true, xLabel: "epoch", height: 190, legend: true, xFormat: (x) => `${Math.round(x)}` })
          : h("p", { class: "ft-muted" }, "These versions recorded no validation score during training (embedding and reranker runs before 13 Oct 2026 scored only before and after).")),
      );
    };
    const picks = h("div", { class: "mr-compare-picks" }, done.map((v) => {
      const box = h("input", { type: "checkbox", value: v.run_id });
      box.checked = chosen.has(v.run_id);
      box.onchange = () => { box.checked ? chosen.add(v.run_id) : chosen.delete(v.run_id); draw(); };
      return h("label", { class: "ft-chip" }, box, h("span", {}, `${v.label} · ${v.short_base}`));
    }));
    const details = h("details", { class: "mr-more" }, h("summary", {}, `Compare versions' training (${done.length})`), picks, charts);
    let drawn = false;
    details.addEventListener("toggle", () => { if (details.open && !drawn) { drawn = true; draw(); } });
    return details;
  }

  async function swap(job, v, button, note) {
    const answer = window.MemorizzSwap ? await window.MemorizzSwap.confirm({ card: job.key, action: "in", label: v.label, runId: v.run_id }) : true;
    if (!answer) return;
    button.disabled = true;
    say(note, job.key === "embedding" ? "Swapping in and re-embedding memories…" : "Swapping in…");
    try {
      const { job_id } = await api(`/api/finetune/models/${job.key}/swap`, { method: "POST", body: { run_id: v.run_id, dimensions: answer.dimensions || null } });
      const poll = async () => {
        const state = await api(`/api/finetune/jobs/${job_id}`);
        if (state.status === "running") {
          say(note, state.message);
          return setTimeout(poll, 1200);
        }
        if (state.status === "failed") {
          button.disabled = false;
          return say(note, state.error, true);
        }
        load();
      };
      poll();
    } catch (error) {
      button.disabled = false;
      say(note, error.message, true);
    }
  }

  async function rename(v, note) {
    const label = prompt(`Name ${v.label} (empty for its version number):`, v.label.startsWith("v") ? "" : v.label);
    if (label === null) return;
    try {
      await api(`/api/finetune/runs/${v.run_id}`, { method: "PATCH", body: { label } });
      load();
    } catch (error) {
      say(note, error.message, true);
    }
  }

  async function remove(v, button, note) {
    if (!confirm(`Delete ${v.label} (${v.short_base}) and its model files? Its scores stay in the history.`)) return;
    button.disabled = true;
    try {
      await api(`/api/finetune/runs/${v.run_id}`, { method: "DELETE" });
      open.delete(v.run_id);
      load();
    } catch (error) {
      button.disabled = false;
      if (note) say(note, error.message, true);
      else alert(error.message);
    }
  }

  load();
})();

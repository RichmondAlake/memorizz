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
    limit: 100,
    selected: new Set(),
    runs: [],
    runId: "",
    events: [],
    after: 0,
    timers: {},
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
  const svgNode = (tag, attrs = {}) => {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    return node;
  };
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
  const KIND = { decision: "Decision model", embedding: "Embedding model", reranker: "Reranker" };
  const DEVICE = { mps: "Apple GPU (MPS)", cuda: "NVIDIA GPU (CUDA)", cpu: "CPU" };
  const statusPill = (status) => {
    const tone = { succeeded: "success", accepted: "success", running: "info", queued: "info", pending: "warning", failed: "danger", rejected: "danger", canceled: "warning", interrupted: "danger" }[status] || "info";
    return h("span", { class: `pill pill--${tone}` }, status);
  };
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
  const exampleText = (e) => {
    const input = e.input || {};
    if (input.state !== undefined) return typeof input.state === "string" ? input.state : JSON.stringify(input.state, null, 1);
    return `Q: ${input.query}\n+ ${input.positive}` + (input.negatives || []).map((n) => `\n− ${n}`).join("");
  };

  // -- overview ------------------------------------------------------------
  async function loadOverview() {
    try {
      S.overview = await api("/api/finetune/overview");
    } catch (error) {
      say(document.getElementById("ft-env"), error.message, true);
      return;
    }
    renderEnv();
    renderDatasets();
    renderActive();
    fillTasks();
    if (S.datasetId) selectDataset(S.datasetId);
  }

  function renderEnv() {
    const env = S.overview.environment;
    const node = document.getElementById("ft-env");
    fill(node,
      h("span", {}, "Trains on ", h("strong", {}, DEVICE[env.device] || env.device || "?")),
      env.ready
        ? h("span", {}, "Packages ", h("strong", {}, "ready"))
        : h("span", { class: "ft-down" }, `Missing ${env.missing.join(", ")} · ${env.install}`),
      h("span", {}, "Unsloth ", h("strong", {}, env.unsloth.available ? "available" : "not on this machine"), env.unsloth.available ? "" : h("span", { class: "ft-muted", title: env.unsloth.reason }, " ⓘ")),
      h("span", {}, "Teacher ", h("strong", {}, `${S.overview.teacher.provider}/${S.overview.teacher.model}`)),
      h("span", {}, "Jev check ", h("strong", {}, S.overview.jev_available ? "available" : "set TYPESAFE_API_KEY")),
    );
  }

  function renderDatasets() {
    const list = document.getElementById("ft-datasets");
    const rows = S.overview ? S.overview.datasets : [];
    if (!rows.length) {
      fill(list, h("p", { class: "ft-muted" }, "No datasets yet."));
      return;
    }
    fill(list,
      ...rows.map((d) =>
        h(
          "button",
          { class: "ft-dataset", type: "button", "aria-current": d.dataset_id === S.datasetId ? "true" : "false", onclick: () => selectDataset(d.dataset_id) },
          h("span", {}, d.name),
          h("small", {}, `${KIND[d.kind]} · ${d.task.replace("_", " ")} · ${d.counts.total} examples, ${d.counts.status.accepted || 0} accepted`),
        ),
      ),
    );
  }

  function renderActive() {
    const node = document.getElementById("ft-active");
    const active = S.overview.active || {};
    const uses = Object.keys(active);
    if (!uses.length) {
      fill(node, h("p", { class: "ft-muted" }, "Train a model and activate it for memory routing, relevance, embeddings or reranking."));
      return;
    }
    fill(node,
      ...uses.map((use) =>
        h(
          "div",
          {},
          h("strong", {}, use.replace("_", " ")),
          h("div", { class: "ft-muted" }, active[use].base_model, " · run ", h("code", {}, active[use].run_id)),
          readOnly
            ? null
            : h("button", { class: "btn btn-sm", type: "button", onclick: async () => { await api(`/api/finetune/active/${use}/clear`, { method: "POST" }); await refreshOverview(); } }, "Deactivate"),
        ),
      ),
    );
  }

  async function refreshOverview() {
    S.overview = await api("/api/finetune/overview");
    renderDatasets();
    renderActive();
  }

  function fillTasks() {
    const form = document.getElementById("ft-new");
    const kind = form.elements.kind.value;
    const tasks = S.overview.tasks.filter((t) => t.kind === kind);
    fill(form.elements.task, ...tasks.map((t) => h("option", { value: t.task }, t.title)));
    const note = () => {
      const t = tasks.find((x) => x.task === form.elements.task.value);
      document.getElementById("ft-task-note").textContent = t ? t.description : "";
      form.elements.name.placeholder = t ? t.title : "";
    };
    form.elements.task.onchange = note;
    note();
  }

  document.getElementById("ft-new").addEventListener("change", (event) => {
    if (event.target.name === "kind") fillTasks();
  });
  document.getElementById("ft-new").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.target;
    const button = form.querySelector("button");
    await busy(button, async () => {
      try {
        const dataset = await api("/api/finetune/datasets", {
          method: "POST",
          body: { kind: form.elements.kind.value, task: form.elements.task.value, name: form.elements.name.value },
        });
        form.elements.name.value = "";
        await refreshOverview();
        await selectDataset(dataset.dataset_id);
      } catch (error) {
        say(document.getElementById("ft-task-note"), error.message, true);
      }
    });
  });

  // -- dataset ---------------------------------------------------------------
  async function selectDataset(id) {
    Object.values(S.timers).forEach(clearTimeout);
    S.timers = {};
    S.datasetId = id;
    S.selected.clear();
    S.runId = "";
    history.replaceState(null, "", `/fine-tuning?dataset=${encodeURIComponent(id)}`);
    try {
      S.dataset = await api(`/api/finetune/datasets/${id}`);
    } catch (error) {
      fill(document.getElementById("ft-main"), h("div", { class: "empty-state" }, h("strong", {}, "Dataset not found"), h("p", {}, error.message)));
      return;
    }
    renderDatasets();
    renderMain();
  }

  async function reloadDataset() {
    S.dataset = await api(`/api/finetune/datasets/${S.datasetId}`);
    renderHead();
  }

  function renderMain() {
    const main = document.getElementById("ft-main");
    const tabs = [
      ["data", "Data"],
      ["task", "Task"],
      ["format", "Format"],
      ["train", "Train"],
      ["results", "Results"],
    ];
    fill(main,
      h("section", { class: "ft-panel", id: "ft-head" }),
      h(
        "nav",
        { class: "page-tabs ft-tabs", "aria-label": "Dataset views" },
        tabs.map(([key, label]) =>
          h("button", { type: "button", class: "page-tab", "aria-current": S.tab === key ? "page" : null, onclick: () => { S.tab = key; renderMain(); } }, label),
        ),
      ),
      h("div", { id: "ft-tab" }),
    );
    renderHead();
    ({ data: renderData, task: renderTask, format: renderFormat, train: renderTrain, results: renderResults })[S.tab]();
  }

  function renderHead() {
    const d = S.dataset;
    const head = document.getElementById("ft-head");
    if (!head) return;
    const c = d.counts;
    fill(head,
      h(
        "div",
        { class: "ft-head" },
        h(
          "div",
          {},
          h("h2", {}, d.name),
          h(
            "div",
            { class: "ft-stats" },
            h("span", {}, KIND[d.kind], " · ", d.task.replace("_", " ")),
            h("span", {}, h("b", {}, c.total), " examples"),
            h("span", {}, h("b", {}, c.status.accepted || 0), " accepted"),
            h("span", {}, h("b", {}, c.status.pending || 0), " pending"),
            h("span", {}, "train ", h("b", {}, c.split.train || 0), " · validation ", h("b", {}, c.split.validation || 0), " · test ", h("b", {}, c.split.test || 0)),
          ),
        ),
        readOnly
          ? null
          : h("button", { class: "btn btn-sm btn-danger", type: "button", onclick: async () => {
              if (!confirm(`Delete "${d.name}" with its ${c.total} examples and run records?`)) return;
              await api(`/api/finetune/datasets/${d.dataset_id}`, { method: "DELETE" });
              S.datasetId = "";
              history.replaceState(null, "", "/fine-tuning");
              await refreshOverview();
              fill(document.getElementById("ft-main"), h("div", { class: "empty-state" }, h("strong", {}, "Dataset deleted")));
            } }, "Delete dataset"),
      ),
    );
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
      h("h3", {}, "Label balance"),
      h(
        "div",
        { class: "ft-bars" },
        Object.entries(labels).map(([key, v]) => {
          const accepted = v.accepted || 0;
          const pending = v.pending || 0;
          return h(
            "div",
            { class: "ft-bar", title: `${accepted} accepted, ${pending} pending, ${v.rejected || 0} rejected` },
            h("span", {}, key),
            h("span", { class: "ft-bar-track" }, h("span", { class: "ft-bar-fill", style: `width:${(accepted / max) * 100}%` }), h("span", { class: "ft-bar-fill ft-bar-fill--pending", style: `width:${(pending / max) * 100}%` })),
            h("span", { class: "ft-bar-num" }, `${accepted}${pending ? "+" + pending : ""}`),
          );
        }),
      ),
      h("p", { class: "ft-muted" }, "Solid: accepted · light: pending review."),
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
    const generateForm = h(
      "form",
      {
        onsubmit: async (event) => {
          event.preventDefault();
          const f = event.target.elements;
          const body = {
            labels: f.labels ? [...f.labels.selectedOptions].map((o) => o.value) : null,
            count: Number(f.count.value),
            teacher: f.teacher.value.trim() || null,
            verifier: verifier.value === "other" ? otherVerifier.value.trim() || null : verifier.value || null,
            context: f.context ? f.context.value : "",
            topic: f.topic ? f.topic.value : "",
            from_memory: f.from_memory ? f.from_memory.checked : false,
          };
          const button = event.target.querySelector("button");
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
              label.textContent = `Added ${r.added} of ${r.written ?? r.passages ?? "?"} written` + (rejected ? ` · rejected: ${rejected}` : "") + (r.short && r.short.length ? ` · still short: ${r.short.join(", ")}` : "") + (cost.length ? ` · cost $${cost.reduce((a, b) => a + b, 0).toFixed(4)}` : "") + `. Review them below.`;
              await reloadDataset();
              renderData();
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
        h("label", {}, "Examples", h("input", { type: "number", name: "count", min: 1, max: 2000, value: retrievalLike ? 24 : 60 })),
        h("label", { class: "ft-wide" }, "Teacher (provider/model)", h("input", { name: "teacher", value: teacher })),
        h("label", { class: "ft-wide" }, "Blind check", verifier, otherVerifier),
      ),
      retrievalLike
        ? h(
            "div",
            { class: "ft-row" },
            h("label", { class: "ft-wide" }, "Topic of generated memories", h("input", { name: "topic", placeholder: "e.g. a product team's projects, people and decisions" })),
            h("label", {}, h("span", {}, h("input", { type: "checkbox", name: "from_memory" }), " Use my stored memories instead")),
          )
        : h(
            "div",
            { class: "ft-row" },
            h("label", { class: "ft-wide" }, "Setting (optional)", h("input", { name: "context", placeholder: "e.g. a support assistant for a SaaS billing product" })),
            h("label", { class: "ft-wide", title: "Leave empty for every option. Pick the ones a model gets wrong to add examples where it needs them." }, "Only these options (optional)", h("select", { name: "labels", multiple: true, size: 3 }, optionKeys(question()).map((k) => h("option", { value: k }, k)))),
          ),
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
                loadExamples();
              } catch (error) {
                say(importNote, error.message, true);
              }
            }) }, "Import memories"), importNote),
          )
        : null;

    fill(tab,
      h("div", { class: "ft-grid2" },
        h("section", { class: "ft-panel" }, h("h2", {}, "Generate synthetic data"), h("p", { class: "ft-muted" }, retrievalLike ? "The teacher writes a question each memory answers and a hard negative that looks related but does not; a blind check keeps only pairs it confirms." : "The teacher writes states for each option in turn (standard, negation, boundary and hard-negative cases); a blind check keeps only examples it labels the same way. Labels that come up short are topped up."), generateForm),
        h("section", { class: "ft-panel" }, labelBars() || h("div", {}, h("h3", {}, "Sources"), h("p", { class: "ft-muted" }, Object.entries(d.sources || {}).map(([k, v]) => `${k} ${v}`).join(" · ") || "No examples yet.")), importBox, addExampleForm()),
      ),
      h("section", { class: "ft-panel", id: "ft-examples" }),
    );
    loadExamples();
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
      h("h3", {}, "Add an example"),
      h("textarea", { name: "state", rows: 3, required: true, placeholder: "A piece of state, e.g. a message or tool result", style: "width:100%" }),
      h("div", { class: "ft-row" }, h("label", { class: "ft-wide" }, "Correct option", h("select", { name: "label" }, optionKeys(q).map((k) => h("option", { value: k }, k)))), h("button", { class: "btn", type: "submit", disabled: readOnly }, "Add"), note));
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
    h("h3", {}, "Add an example"),
    h("input", { name: "query", required: true, placeholder: "Query", style: "width:100%;margin-bottom:.4rem" }),
    h("textarea", { name: "positive", rows: 2, required: true, placeholder: "The memory that answers it", style: "width:100%;margin-bottom:.4rem" }),
    h("textarea", { name: "negatives", rows: 2, placeholder: "Hard negatives, one per line (optional)", style: "width:100%" }),
    h("div", { class: "ft-actions" }, h("button", { class: "btn", type: "submit", disabled: readOnly }, "Add"), note));
  }

  async function loadExamples() {
    const box = document.getElementById("ft-examples");
    if (!box) return;
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
    const q = question();
    const isDecision = S.dataset.kind === "decision";
    const keys = optionKeys(q);
    const note = h("span", { class: "ft-muted" });
    const pick = (name, options) => {
      const select = h("select", { onchange: () => { S.filter[name] = select.value; loadExamples(); } }, options.map(([v, l]) => h("option", { value: v }, l)));
      select.value = S.filter[name];
      return select;
    };
    const bulk = async (status) => {
      if (!S.selected.size) return say(note, "Select examples first.");
      await api("/api/finetune/examples/review", { method: "POST", body: { ids: [...S.selected], status } });
      S.selected.clear();
      await reloadDataset();
      if (S.tab === "data") { renderHead(); renderData(); }
    };
    const rows = S.examples.map((e) => {
      const check = h("input", { type: "checkbox", "aria-label": "Select example", onchange: (ev) => (ev.target.checked ? S.selected.add(e.example_id) : S.selected.delete(e.example_id)) });
      check.checked = S.selected.has(e.example_id);
      const split = h("select", { disabled: readOnly, onchange: async () => { await api(`/api/finetune/examples/${e.example_id}`, { method: "PATCH", body: { split: split.value } }); reloadDataset(); } }, ["train", "validation", "test"].map((s) => h("option", { value: s }, s)));
      split.value = e.split;
      let label = "";
      if (isDecision && q) {
        const current = goldKey(q, e.label);
        const relabel = h("select", { disabled: readOnly, onchange: async () => { await api(`/api/finetune/examples/${e.example_id}`, { method: "PATCH", body: { label_key: relabel.value } }); reloadDataset(); } }, keys.map((k) => h("option", { value: k }, k)));
        relabel.value = current;
        label = relabel;
      }
      const probs = e.meta && e.meta.verifier_probabilities;
      const top = probs ? Object.entries(probs).sort((a, b) => b[1] - a[1])[0] : null;
      return h(
        "tr",
        {},
        h("td", {}, check),
        h("td", {}, statusPill(e.status)),
        isDecision ? h("td", {}, label) : null,
        h("td", { class: "ft-text" }, exampleText(e)),
        h("td", {}, e.verified ? h("span", { title: top ? `blind check: ${top[0]} ${(top[1] * 100).toFixed(0)}%` : "blind check agreed" }, "✓", top ? ` ${(top[1] * 100).toFixed(0)}%` : "") : ""),
        h("td", {}, split),
        h("td", { class: "ft-muted" }, [e.source, e.case_type].filter(Boolean).join(" · ")),
      );
    });
    fill(box,
      h("div", { class: "ft-panel-head" }, h("h2", {}, `Examples (${S.total})`), h("span", { class: "ft-muted" }, "Accepted examples are trained on; test and validation ones are held out for scoring.")),
      h(
        "div",
        { class: "ft-actions" },
        pick("status", [["", "All statuses"], ["pending", "Pending"], ["accepted", "Accepted"], ["rejected", "Rejected"]]),
        pick("split", [["", "All splits"], ["train", "Train"], ["validation", "Validation"], ["test", "Test"]]),
        h("button", { class: "btn btn-sm btn-primary", type: "button", disabled: readOnly, onclick: async () => { const r = await api(`/api/finetune/datasets/${S.datasetId}/accept-verified`, { method: "POST" }); say(note, `Accepted ${r.accepted} checked examples.`); await reloadDataset(); renderData(); } }, "Accept all checked (✓)"),
        h("button", { class: "btn btn-sm", type: "button", disabled: readOnly, onclick: () => bulk("accepted") }, "Accept selected"),
        h("button", { class: "btn btn-sm", type: "button", disabled: readOnly, onclick: () => bulk("rejected") }, "Reject selected"),
        h("button", { class: "btn btn-sm btn-danger", type: "button", disabled: readOnly, onclick: async () => { if (!S.selected.size) return say(note, "Select examples first."); if (!confirm(`Delete ${S.selected.size} examples?`)) return; await api("/api/finetune/examples/delete", { method: "POST", body: { ids: [...S.selected] } }); S.selected.clear(); await reloadDataset(); renderData(); } }, "Delete selected"),
        note,
      ),
      S.examples.length
        ? h("div", { class: "ft-table-wrap" }, h("table", { class: "ft-table" }, h("thead", {}, h("tr", {}, h("th", {}, ""), h("th", {}, "Status"), isDecision ? h("th", {}, "Label") : null, h("th", {}, isDecision ? "State" : "Query · memory · negatives"), h("th", {}, "Checked"), h("th", {}, "Split"), h("th", {}, "Source"))), h("tbody", {}, rows)))
        : h("p", { class: "ft-muted" }, "No examples match."),
      S.total > S.examples.length ? h("button", { class: "btn btn-sm", type: "button", onclick: () => { S.limit += 100; loadExamples(); } }, `Show more (${S.total - S.examples.length} left)`) : null,
    );
  }

  // -- task tab ----------------------------------------------------------------
  function renderTask() {
    const tab = document.getElementById("ft-tab");
    const d = S.dataset;
    const note = h("span", { class: "ft-muted" });
    if (d.kind !== "decision") {
      const input = h("textarea", { class: "ft-instructions", value: d.spec.instruction || "" });
      fill(tab, h("section", { class: "ft-panel" }, h("h2", {}, "Query instruction"), h("p", { class: "ft-muted" }, "Instruction-tuned embedders (Qwen3-Embedding) read this before each query. EmbeddingGemma uses its own trained search prompt."), input, h("div", { class: "ft-actions" }, h("button", { class: "btn btn-primary", type: "button", disabled: readOnly, onclick: async () => { try { await api(`/api/finetune/datasets/${d.dataset_id}`, { method: "PATCH", body: { spec: { ...d.spec, instruction: input.value } } }); say(note, "Saved."); await reloadDataset(); } catch (error) { say(note, error.message, true); } } }, "Save"), note)));
      return;
    }
    const q = question();
    const instructions = h("textarea", { class: "ft-instructions", value: q.instructions || "" });
    const editor = h("div", { class: "ft-options-editor" });
    const addRow = (key = "", desc = "") => {
      const row = h("div", { class: "ft-option" }, h("input", { value: key, placeholder: "option key", "aria-label": "Option key" }), h("textarea", { value: desc, placeholder: "When to choose it", "aria-label": "Option description" }), h("button", { class: "btn btn-sm", type: "button", onclick: () => row.remove() }, "Remove"));
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

  // -- format tab ----------------------------------------------------------------
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

  // -- train tab -----------------------------------------------------------------
  function lineChart(series, { width = 560, height = 190, yLabel = "", yMin = null, yMax = null } = {}) {
    const pad = { l: 46, r: 12, t: 12, b: 26 };
    const all = series.flatMap((s) => s.points);
    const svg = svgNode("svg", { viewBox: `0 0 ${width} ${height}`, class: "ft-chart", role: "img", "aria-label": yLabel });
    if (!all.length) {
      const t = svgNode("text", { x: pad.l, y: height / 2 });
      t.textContent = "Waiting for the first steps…";
      svg.append(t);
      return svg;
    }
    const xs = all.map((p) => p[0]);
    const ys = all.map((p) => p[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs, x0 + 1);
    const y0 = yMin ?? Math.min(...ys), y1 = yMax ?? Math.max(...ys, y0 + 1e-6);
    const X = (x) => pad.l + ((x - x0) / (x1 - x0)) * (width - pad.l - pad.r);
    const Y = (y) => height - pad.b - ((y - y0) / (y1 - y0 || 1)) * (height - pad.t - pad.b);
    for (let i = 0; i <= 3; i++) {
      const v = y0 + ((y1 - y0) * i) / 3;
      svg.append(svgNode("line", { class: "grid", x1: pad.l, x2: width - pad.r, y1: Y(v), y2: Y(v) }));
      const t = svgNode("text", { x: 4, y: Y(v) + 4 });
      t.textContent = fmt(v, 2);
      svg.append(t);
    }
    svg.append(svgNode("line", { class: "axis", x1: pad.l, x2: width - pad.r, y1: height - pad.b, y2: height - pad.b }));
    const xt = svgNode("text", { x: width - pad.r - 80, y: height - 6 });
    xt.textContent = `step ${x1}`;
    svg.append(xt);
    for (const s of series) {
      if (!s.points.length) continue;
      svg.append(svgNode("polyline", { class: `line ${s.cls || ""}`, points: s.points.map(([x, y]) => `${X(x)},${Y(y)}`).join(" ") }));
      if (s.dots) for (const [x, y] of s.points) svg.append(svgNode("circle", { class: "dot", cx: X(x), cy: Y(y), r: 3.5 }));
    }
    return svg;
  }

  async function renderTrain() {
    const tab = document.getElementById("ft-tab");
    const d = S.dataset;
    const base = baseSelect(d.kind);
    let allRuns = [];
    try {
      allRuns = (await api("/api/finetune/runs")).runs.filter((r) => r.kind === d.kind && r.status === "succeeded");
    } catch (error) {
      allRuns = [];
    }
    const names = Object.fromEntries((S.overview.datasets || []).map((x) => [x.dataset_id, x.name]));
    const mainMetric = d.kind === "decision" ? "accuracy" : "ndcg@10";
    const startFrom = h("select", { name: "from_run" }, h("option", { value: "" }, "The base model (a fresh start)"), allRuns.map((r) => h("option", { value: r.run_id }, `Continue run ${r.run_id} · ${names[r.dataset_id] || r.dataset_id} · ${mainMetric} ${fmt(r.metrics_after[mainMetric])}`)));
    startFrom.onchange = () => { document.getElementById("ft-base-label").hidden = !!startFrom.value; };
    const replaySelect = h("select", { name: "replay" }, h("option", { value: "" }, "None"), (S.overview.datasets || []).filter((x) => x.kind === d.kind && x.dataset_id !== d.dataset_id).map((x) => h("option", { value: x.dataset_id }, `${x.name} (${x.counts.split.train || 0} train)`)));
    const note = h("span", { class: "ft-muted" });
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
          try {
            const run = await api(`/api/finetune/datasets/${d.dataset_id}/runs`, { method: "POST", body: { base_model: base.value, params, backend: f.backend.value, from_run: startFrom.value || null, replay } });
            S.runId = run.run_id;
            S.events = [];
            S.after = 0;
            await renderTrain();
          } catch (error) {
            say(note, error.message, true);
          }
        },
      },
      h("div", { class: "ft-row" }, h("label", { class: "ft-wide" }, "Start from", startFrom), h("label", { class: "ft-wide", id: "ft-base-label" }, "Base model", base), h("label", {}, "Trainer", h("select", { name: "backend" }, h("option", { value: "auto" }, "Automatic"), h("option", { value: "local" }, "This machine (PEFT)"), h("option", { value: "unsloth", disabled: !S.overview.environment.unsloth.available }, "Unsloth (NVIDIA)")))),
      h("div", { class: "ft-row" }, h("label", { class: "ft-wide", title: "Mix in training examples from an earlier dataset so the model keeps what it learned there." }, "Replay earlier data", replaySelect), h("label", {}, "Share", h("input", { type: "number", name: "replay_fraction", min: 0.1, max: 1, step: 0.1, value: 0.5 }))),
      h(
        "div",
        { class: "ft-row" },
        h("label", {}, "Epochs", h("input", { type: "number", name: "epochs", min: 1, max: 20, step: 1, value: 3 })),
        h("label", {}, "Batch size", h("input", { type: "number", name: "batch_size", min: 1, max: 256, value: 8 })),
        h("label", {}, "Learning rate", h("input", { type: "number", name: "learning_rate", step: "0.00001", min: "0.000001", value: d.kind === "reranker" ? 0.00002 : 0.0002 })),
        embedOrDecision ? h("label", {}, "LoRA rank", h("input", { type: "number", name: "lora_r", min: 4, max: 128, value: 16 })) : null,
        h("label", {}, "Max tokens", h("input", { type: "number", name: "max_length", min: 64, max: 4096, step: 32, value: 256 })),
      ),
      h("div", { class: "ft-actions" }, h("button", { class: "btn btn-primary", type: "submit", disabled: readOnly }, "Start training"), note),
      h("p", { class: "ft-muted" }, "Trains on accepted train examples, checks validation each epoch, calibrates, and scores the held-out test examples before and after. The run continues if you close this page."),
    );
    const runsBox = h("div", { class: "ft-run-list" });
    const live = h("section", { class: "ft-panel", id: "ft-live" });
    fill(tab, h("div", { class: "ft-grid2" }, h("section", { class: "ft-panel" }, h("h2", {}, "Train a model"), form), h("section", { class: "ft-panel" }, h("h2", {}, "Runs"), runsBox)), live);
    try {
      S.runs = (await api(`/api/finetune/runs?dataset_id=${d.dataset_id}`)).runs;
    } catch (error) {
      fill(runsBox, h("p", { class: "ft-down" }, error.message));
      return;
    }
    if (!S.runId && S.runs.length) S.runId = S.runs[0].run_id;
    const main = d.kind === "decision" ? "accuracy" : "ndcg@10";
    fill(runsBox,
      ...(S.runs.length
        ? S.runs.map((r) =>
            h(
              "button",
              { class: "ft-run", type: "button", "aria-current": r.run_id === S.runId ? "true" : "false", onclick: () => { S.runId = r.run_id; S.events = []; S.after = 0; renderTrain(); } },
              h("span", {}, h("span", {}, r.base_model.split("/").pop(), r.parent_run_id ? ` ← ${r.parent_run_id}` : ""), h("br"), h("small", {}, `${r.run_id} · ${r.backbone} · ${(r.created_at || "").slice(0, 16).replace("T", " ")}${r.progress && r.progress.replay ? " · replay " + r.progress.replay.map((x) => x.examples).join("+") : ""}`)),
              h("span", {}, statusPill(r.status), h("br"), h("small", {}, `${main} ${fmt(r.metrics_before[main])} → ${fmt(r.metrics_after[main])}`)),
            ),
          )
        : [h("p", { class: "ft-muted" }, "No runs yet.")]),
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
    const steps = S.events.filter((e) => e.type === "step").map((e) => e.data);
    const evals = S.events.filter((e) => e.type === "eval" && e.data.phase === "validation");
    const metricKey = run.kind === "decision" ? "accuracy" : Object.keys((evals[0] || { data: { metrics: {} } }).data.metrics).find((k) => k.includes("ndcg")) || "accuracy";
    const p = run.progress || {};
    const pct = p.total ? Math.round((p.step / p.total) * 100) : 0;
    const eta = p.eta_seconds ? `${Math.round(p.eta_seconds / 60)}m ${Math.round(p.eta_seconds % 60)}s left` : "";
    fill(live,
      h("div", { class: "ft-panel-head" }, h("h2", {}, `Run ${run.run_id} · ${run.base_model}`), h("span", {}, statusPill(run.status), " ", h("span", { class: "ft-muted" }, `${run.backbone} · ${run.backend}${run.device ? " · " + run.device : ""}`))),
      h("div", { class: "ft-progress" }, h("span", {}, p.step ? `Step ${p.step} of ${p.total} · loss ${fmt(p.loss, 4)} ${eta ? "· " + eta : ""}` : p.message || "Starting…"), h("div", { class: "ft-progress-track" }, h("div", { class: "ft-progress-fill", style: `width:${run.status === "succeeded" ? 100 : pct}%` }))),
      run.error ? h("p", { class: "ft-down" }, run.error) : null,
      h("div", { class: "ft-grid2" },
        h("div", {}, h("h3", {}, "Training loss"), lineChart([{ points: steps.map((s) => [s.step, s.loss]) }], { yLabel: "loss" })),
        h("div", {}, h("h3", {}, `Validation ${metricKey}`), lineChart([{ points: evals.map((e) => [e.data.step || 0, Number(e.data.metrics[metricKey] ?? 0)]), cls: "line--b", dots: true }], { yLabel: metricKey, yMin: 0, yMax: 1 }))),
      h("h3", {}, "Log"),
      h("div", { class: "ft-log" }, S.events.filter((e) => e.type !== "step").map((e) => h("div", {}, `${e.timestamp.slice(11, 19)}  ${e.type === "eval" ? `${e.data.phase}: ` + Object.entries(e.data.metrics || {}).filter(([, v]) => typeof v === "number").slice(0, 5).map(([k, v]) => `${k} ${fmt(v)}`).join(", ") : e.data.message || e.data.status || ""}`))),
      run.status === "running" && !readOnly ? h("div", { class: "ft-actions" }, h("button", { class: "btn btn-sm btn-danger", type: "button", onclick: async () => { await api(`/api/finetune/runs/${run.run_id}/cancel`, { method: "POST" }); followRun(); } }, "Cancel run")) : null,
      run.status === "succeeded" ? h("div", { class: "ft-actions" }, h("button", { class: "btn btn-sm btn-primary", type: "button", onclick: () => { S.tab = "results"; renderMain(); } }, "See results")) : null,
    );
    if (run.status === "running" || run.status === "queued") S.timers.run = setTimeout(followRun, 2000);
    else if (S.runs.find((r) => r.run_id === run.run_id && r.status !== run.status)) renderTrain();
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
    if (!runs.length) {
      fill(tab, h("div", { class: "empty-state" }, h("strong", {}, "No finished runs yet"), h("p", {}, "Train a model in the Train tab; its scores before and after appear here.")));
      return;
    }
    if (!runs.find((r) => r.run_id === S.runId)) S.runId = runs[0].run_id;
    const run = runs.find((r) => r.run_id === S.runId);
    const select = h("select", { onchange: () => { S.runId = select.value; renderResults(); } }, runs.map((r) => h("option", { value: r.run_id }, `${r.run_id} · ${r.base_model}`)));
    select.value = S.runId;
    const before = run.metrics_before || {};
    const after = run.metrics_after || {};
    const lowerBetter = new Set(["nll", "brier", "ece", "order_flip_rate"]);
    const keys = ["accuracy", "macro_f1", "nll", "brier", "ece", "order_flip_rate", "mrr@10", "ndcg@10", "recall@1", "recall@5", "recall@10", "map"].filter((k) => k in before || k in after);
    const metrics = h("table", { class: "ft-metrics" }, h("thead", {}, h("tr", {}, h("th", {}, "Held-out metric"), h("th", {}, "Base model"), h("th", {}, "Fine-tuned"), h("th", {}, "Change"))), h("tbody", {}, keys.map((k) => {
      const a = before[k], b = after[k];
      const delta = typeof a === "number" && typeof b === "number" ? b - a : null;
      const good = delta !== null && (lowerBetter.has(k) ? delta < 0 : delta > 0);
      return h("tr", {}, h("td", {}, k), h("td", {}, fmt(a)), h("td", {}, fmt(b)), h("td", { class: delta === null || delta === 0 ? "" : good ? "ft-up" : "ft-down" }, delta === null ? "–" : (delta > 0 ? "+" : "") + delta.toFixed(3)));
    })));
    const conf = after.confusion;
    let confusion = null;
    if (conf) {
      const max = Math.max(1, ...conf.matrix.flat());
      confusion = h("div", { class: "ft-table-wrap" }, h("h3", {}, "Confusion on held-out examples (rows: true, columns: predicted)"), h("table", { class: "ft-confusion" }, h("thead", {}, h("tr", {}, h("th", {}, ""), conf.labels.map((l) => h("th", { class: "rot", title: l }, l)))), h("tbody", {}, conf.matrix.map((row, i) => h("tr", {}, h("th", { style: "text-align:right" }, conf.labels[i]), row.map((n, j) => h("td", { style: n ? `background: color-mix(in srgb, var(${i === j ? "--accent-secondary" : "--accent-danger"}) ${Math.round((n / max) * 70) + 10}%, transparent)` : "", title: `${conf.labels[i]} → ${conf.labels[j]}: ${n}` }, n || "")))))));
    }
    const uses = d.kind === "decision" ? (d.task === "relevance" ? ["relevance"] : ["memory_routing"]) : [d.kind];
    const actNote = h("span", { class: "ft-muted" });
    fill(tab,
      h("section", { class: "ft-panel" },
        h("div", { class: "ft-panel-head" }, h("h2", {}, "Before and after"), h("label", {}, "Run ", select)),
        h("p", { class: "ft-muted" }, `Scored on ${before.count ?? before.queries ?? "the"} held-out ${d.kind === "decision" ? "examples" : "queries"}${run.progress && run.progress.counts ? ` · trained on ${run.progress.counts.train}` : ""}. Small test sets swing a lot; generate or import more data for a firmer comparison.`),
        metrics,
        confusion),
      d.kind === "decision" ? historyPanel(run) : null,
      h("div", { class: "ft-grid2" },
        h("section", { class: "ft-panel" }, h("h2", {}, "Try it"), tryIt(run)),
        h("section", { class: "ft-panel" },
          h("h2", {}, "Use it"),
          h("p", { class: "ft-muted" }, "Model folder on this machine:"), h("pre", { class: "ft-code" }, run.progress && run.progress.model_dir || run.output_dir + "/model"),
          readOnly ? null : h("div", { class: "ft-actions" }, uses.map((use) => h("button", { class: "btn btn-primary", type: "button", onclick: async () => { try { await api(`/api/finetune/runs/${run.run_id}/activate`, { method: "POST", body: { use } }); say(actNote, `Active for ${use.replace("_", " ")}.`); await refreshOverview(); } catch (e) { say(actNote, e.message, true); } } }, `Activate for ${use.replace("_", " ")}`)), actNote),
          h("p", { class: "ft-muted" }, d.kind === "decision" ? "SDK: FineTuning(provider).route(text) · CLI: memorizz finetune route \"…\" · MCP: memorizz_finetune_decide" : d.kind === "embedding" ? "Use for memory retrieval: MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER=huggingface and MEMORIZZ_DEFAULT_EMBEDDING_MODEL=<model folder>. Stored memories must be re-embedded." : "Compare it in Evalground: a cross_encoder reranker with the model folder as its model."),
        )),
    );
  }

  function historyPanel(run) {
    const panel = h("section", { class: "ft-panel" });
    const decisionSets = (S.overview.datasets || []).filter((x) => x.kind === "decision");
    const datasetPick = h("select", {}, decisionSets.map((x) => h("option", { value: x.dataset_id }, x.name)));
    datasetPick.value = S.datasetId;
    const split = h("select", {}, [["test", "Test"], ["held_out", "Test + validation"], ["all", "All accepted"]].map(([v, l]) => h("option", { value: v }, l)));
    const modelPick = h(
      "select",
      {},
      h("option", { value: `run:${run.run_id}` }, `This run (${run.run_id})`),
      run.parent_run_id ? h("option", { value: `run:${run.parent_run_id}` }, `Its parent (${run.parent_run_id})`) : null,
      h("option", { value: `base:${run.base_model}` }, `${run.base_model}, untrained`),
      h("option", { value: "jev:jev-latest", disabled: !S.overview.jev_available }, "Jev, zero-shot (TypeSafe)"),
    );
    const note = h("span", { class: "ft-muted" });
    const table = h("div", { class: "ft-table-wrap" });
    const load = async () => {
      const { evaluations } = await api("/api/finetune/evaluations");
      const rows = evaluations.filter((e) => decisionSets.some((x) => x.dataset_id === e.dataset_id));
      if (!rows.length) {
        fill(table, h("p", { class: "ft-muted" }, "No evaluations yet. Score this run, its base model and Jev on the same frozen sets to see what training changed and what it kept."));
        return;
      }
      const columns = [...new Map(rows.map((e) => [`${e.dataset_id}|${e.split}`, `${e.dataset_name} · ${e.split}`])).entries()];
      const models = [...new Map(rows.map((e) => [e.model.label, e.model])).entries()];
      const latest = {};
      for (const e of rows) latest[`${e.model.label}|${e.dataset_id}|${e.split}`] = e;
      fill(table, h("table", { class: "ft-metrics" },
        h("thead", {}, h("tr", {}, h("th", {}, "Model"), columns.map(([, label]) => h("th", {}, label)))),
        h("tbody", {}, models.map(([label]) => h("tr", {}, h("td", {}, label), columns.map(([key]) => {
          const e = latest[`${label}|${key}`];
          return h("td", { title: e ? `macro-F1 ${fmt(e.metrics.macro_f1)} · ECE ${fmt(e.metrics.ece)} · n=${e.count}` : "" }, e ? `${(e.metrics.accuracy * 100).toFixed(1)}%` : "–");
        }))))),
        h("p", { class: "ft-muted" }, "Accuracy on each frozen set (hover for macro-F1, calibration error and size). Read a row left to right for what a version can do; read a column top to bottom for what training changed."));
    };
    fill(panel,
      h("h2", {}, "Across versions"),
      h("p", { class: "ft-muted" }, "Continual learning is judged on fixed sets: does a new version gain where it was trained without losing what earlier versions could do?"),
      h("div", { class: "ft-row" }, h("label", { class: "ft-wide" }, "Frozen set", datasetPick), h("label", {}, "Examples", split), h("label", { class: "ft-wide" }, "Model", modelPick),
        h("button", { class: "btn", type: "button", disabled: readOnly, onclick: async (event) => {
          const [type, value] = modelPick.value.split(/:(.*)/s);
          const body = { dataset_id: datasetPick.value, split: split.value, run_id: type === "run" ? value : null, base_model: type === "base" ? value : null, jev_model: type === "jev" ? value : null };
          event.target.disabled = true;
          say(note, "Scoring…");
          try {
            const { job_id } = await api("/api/finetune/evaluate", { method: "POST", body });
            const poll = async () => {
              const job = await api(`/api/finetune/jobs/${job_id}`);
              if (job.status === "running") return setTimeout(poll, 1500);
              event.target.disabled = readOnly;
              if (job.status === "failed") return say(note, job.error, true);
              say(note, `${job.report.model.label}: ${(job.report.metrics.accuracy * 100).toFixed(1)}% on ${job.report.count} examples (${job.report.seconds}s).`);
              load();
            };
            poll();
          } catch (error) {
            event.target.disabled = readOnly;
            say(note, error.message, true);
          }
        } }, "Score"), note),
      table,
    );
    load().catch((error) => say(note, error.message, true));
    return panel;
  }

  function tryIt(run) {
    const out = h("div", { class: "ft-probs" });
    const isPointwiseDecision = run.kind === "decision" && run.task !== "relevance";
    if (isPointwiseDecision) {
      const state = h("textarea", { rows: 3, placeholder: "e.g. Remember that Jordan from finance always wants reports as PDF.", style: "width:100%" });
      return h("div", {}, state, h("div", { class: "ft-actions" }, h("button", { class: "btn", type: "button", onclick: (event) => busy(event.target, async () => {
        try {
          const { answers } = await api(`/api/finetune/runs/${run.run_id}/decide`, { method: "POST", body: { state: state.value } });
          const answer = Object.values(answers)[0];
          const probs = Object.entries(answer.probabilities || { false: 1 - answer.noul, true: answer.noul }).sort((a, b) => b[1] - a[1]);
          fill(out, h("p", {}, h("strong", {}, answer.choice ?? String(answer.answer ?? answer.level)), " ", h("span", { class: "ft-muted" }, `confidence ${fmt(answer.confidence ?? answer.noul)}`)), ...probs.slice(0, 6).map(([k, p]) => h("div", { class: "ft-bar" }, h("span", {}, k), h("span", { class: "ft-bar-track" }, h("span", { class: "ft-bar-fill", style: `width:${p * 100}%` })), h("span", { class: "ft-bar-num" }, `${(p * 100).toFixed(1)}%`))));
        } catch (error) {
          say(out, error.message, true);
        }
      }) }, "Decide")), out);
    }
    const query = h("input", { placeholder: "A query", style: "width:100%;margin-bottom:.4rem" });
    const passages = h("textarea", { rows: 4, placeholder: "Candidate memories, one per line", style: "width:100%" });
    return h("div", {}, query, passages, h("div", { class: "ft-actions" }, h("button", { class: "btn", type: "button", onclick: (event) => busy(event.target, async () => {
      try {
        const { ranking } = await api(`/api/finetune/runs/${run.run_id}/rank`, { method: "POST", body: { query: query.value, passages: passages.value.split("\n").map((s) => s.trim()).filter(Boolean) } });
        fill(out, ...ranking.map((r, i) => h("div", { class: "ft-bar" }, h("span", { title: r.passage }, `${i + 1}. ${r.passage}`), h("span", { class: "ft-bar-track" }, h("span", { class: "ft-bar-fill", style: `width:${Math.max(2, Math.min(100, r.score * 100))}%` })), h("span", { class: "ft-bar-num" }, fmt(r.score)))));
      } catch (error) {
        say(out, error.message, true);
      }
    }) }, "Rank")), out);
  }

  loadOverview();
})();

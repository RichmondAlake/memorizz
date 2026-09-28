/* Evalground comparisons. Provider outputs are rendered as text, never HTML. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const catalog = JSON.parse($("benchmark-catalog").textContent);
  const readers = [
    "ollama",
    "openai",
    "anthropic",
    "azure",
    "huggingface",
    "mlx",
  ];
  const rerankers = [
    "none",
    "heuristic",
    "llm",
    "cross_encoder",
    "cohere",
    "voyage",
    "jev",
  ];
  const labels = {
    openai: "OpenAI",
    anthropic: "Anthropic",
    ollama: "Ollama (local)",
    azure: "Azure OpenAI",
    huggingface: "Hugging Face",
    mlx: "MLX (local)",
    none: "Original candidate order",
    heuristic: "Token overlap",
    llm: "LLM relevance scores",
    cross_encoder: "Cross-encoder",
    cohere: "Cohere",
    jev: "Jev",
    voyage: "Voyage",
  };
  const activeStatuses = new Set(["queued", "running", "cancelling"]);
  let currentId, currentResult, timer;
  let submitting = false,
    setupStep = 0,
    resultTab = "overview";
  function showSetup(show) {
    $("comparison-form").hidden = !show;
    $("saved-configuration").hidden = true;
    $("toggle-setup").textContent = show ? "Hide setup" : "View configuration";
    $("toggle-setup").setAttribute("aria-expanded", String(show));
  }
  function showResultTab(name) {
    resultTab = name;
    document
      .querySelectorAll("[data-result-panel]")
      .forEach((p) => (p.hidden = p.dataset.resultPanel !== name));
    document.querySelectorAll("[data-result-tab]").forEach((b) => {
      if (b.dataset.resultTab === name) b.setAttribute("aria-current", "page");
      else b.removeAttribute("aria-current");
    });
    $("configuration-inspector").hidden = name === "overview";
  }
  function validateStep(index) {
    const panel = document.querySelector(`[data-panel="${index}"]`);
    const invalid = [...panel.querySelectorAll("input,select")].find(
      (i) => !i.disabled && !i.checkValidity(),
    );
    if (invalid) {
      goStep(index, false);
      let parent = invalid.parentElement;
      while (parent !== panel) {
        if (parent.tagName === "DETAILS") parent.open = true;
        parent = parent.parentElement;
      }
      invalid.reportValidity();
      return false;
    }
    if (index === 1) {
      const rows = [...panel.querySelectorAll(".model-row")].filter(
        (r) =>
          !(
            $("experiment-type").value === "reader" &&
            r.dataset.kind === "reranker"
          ),
      );
      if (rows.some((r) => r.dataset.loading === "true")) {
        $("form-error").textContent =
          "Model lists are still loading. Wait a moment, then continue.";
        return false;
      }
      if (
        rows.some((r) => {
          const get = (k) => r.querySelector(`[data-key=${k}]`).value;
          return (
            !["none", "heuristic"].includes(get("provider")) &&
            !(get("model") === "__custom__"
              ? get("custom_model").trim()
              : get("model"))
          );
        })
      ) {
        $("form-error").textContent =
          "Select a model for each provider, or enter its deployment name.";
        return false;
      }
    }
    return true;
  }
  function goStep(index, validate = true) {
    if (validate && index > setupStep) {
      for (let i = 0; i < index; i++) if (!validateStep(i)) return;
    }
    const changed = setupStep !== index;
    setupStep = index;
    document
      .querySelectorAll("[data-panel]")
      .forEach((p) => (p.hidden = Number(p.dataset.panel) !== index));
    document.querySelectorAll("[data-step]").forEach((b) => {
      if (Number(b.dataset.step) === index)
        b.setAttribute("aria-current", "step");
      else b.removeAttribute("aria-current");
    });
    $("setup-back").hidden = index === 0;
    $("setup-next").hidden = index === 2;
    $("setup-next").textContent =
      index === 0 ? "Continue to models →" : "Review comparison →";
    $("setup-position").textContent = `Step ${index + 1} of 3`;
    $("form-error").textContent = "";
    if (changed)
      document.querySelector(".setup-steps").scrollIntoView({ block: "start" });
  }
  document
    .querySelectorAll("[data-step]")
    .forEach((b) =>
      b.addEventListener("click", () => goStep(Number(b.dataset.step))),
    );
  document
    .querySelectorAll("[data-result-tab]")
    .forEach((b) =>
      b.addEventListener("click", () => showResultTab(b.dataset.resultTab)),
    );
  $("setup-next").addEventListener("click", () => goStep(setupStep + 1));
  $("setup-back").addEventListener("click", () => goStep(setupStep - 1, false));
  $("toggle-setup").addEventListener("click", () => {
    if (currentResult && !$("experiment-results").hidden) {
      const c = currentResult.config,
        open = $("saved-configuration").hidden;
      const fields = [
        ["Comparison", c.experiment_type],
        ["Dataset", currentResult.dataset_label || c.dataset],
        ["Questions per configuration", c.limit],
        ["Evidence / candidate pool", `${c.top_k} / ${c.candidate_pool_size}`],
        ["Answer models", c.readers.map(analysis.modelName).join(", ")],
        ["Rerankers", c.rerankers.map(analysis.modelName).join(", ")],
        ["Repeats", c.repeats],
        [
          "Time / spend thresholds",
          `${c.max_seconds / 60} minutes / ${c.max_cost_usd == null ? "no spend threshold" : "$" + c.max_cost_usd}`,
        ],
      ];
      $("saved-configuration-summary").replaceChildren(
        ...fields.map(([label, value]) => {
          const item = node("div");
          item.append(node("span", label), node("strong", String(value)));
          return item;
        }),
      );
      $("saved-configuration-json").textContent = JSON.stringify(c, null, 2);
      $("saved-configuration").hidden = !open;
      $("toggle-setup").textContent = open
        ? "Hide configuration"
        : "View configuration";
      $("toggle-setup").setAttribute("aria-expanded", String(open));
    } else showSetup($("comparison-form").hidden);
  });
  function node(tag, text, cls) {
    const n = document.createElement(tag);
    if (text !== undefined) n.textContent = text;
    if (cls) n.className = cls;
    return n;
  }
  function field(parent, title, key, value, choices) {
    const label = node("label", title),
      input = node(choices ? "select" : "input");
    input.dataset.key = key;
    if (choices)
      choices.forEach((p) => {
        const o = node("option", labels[p] || p);
        o.value = p;
        input.append(o);
      });
    input.value = value ?? "";
    label.append(input);
    parent.append(label);
    return input;
  }
  function addModel(container, spec = {}, kind = "reader") {
    const row = node("div", undefined, "model-row"),
      grid = node("div", undefined, "compare-grid");
    row.dataset.kind = kind;
    row.dataset.label = spec.label || "";
    row.append(node("p", "", "model-role"));
    const provider = field(
      grid,
      "Provider",
      "provider",
      spec.provider || (kind === "reranker" ? "none" : "ollama"),
      kind === "reranker" ? rerankers : readers,
    );
    const model = field(grid, "Model", "model", "", []);
    const customModel = field(
      grid,
      "Custom model / deployment name",
      "custom_model",
      spec.model || "",
    );
    customModel.parentElement.hidden = true;
    const discovery = node("p", "Loading models…", "model-discovery muted");
    discovery.setAttribute("role", "status");
    const refresh = node("button", "Refresh model list", "btn btn-secondary");
    refresh.type = "button";
    const llmProvider = field(
      grid,
      "LLM reranker provider",
      "llm_provider",
      spec.llm_provider || "ollama",
      readers,
    );
    const jevMethod = field(
      grid,
      "Jev recipe",
      "jev_method",
      spec.jev_method || "noul",
      ["noul", "score", "choice"],
    );
    row.append(grid);
    row.append(discovery, refresh);
    const details = node("details"),
      summary = node("summary", "Inference settings & pricing");
    details.append(summary);
    const settings = node("div", undefined, "compare-grid");
    const stream = field(
      settings,
      "Measure time to first text token",
      "stream",
      "",
      null,
    );
    stream.type = "checkbox";
    stream.checked = !!spec.stream;
    const pricing = spec.pricing || {};
    for (const [key, title] of [
      ["input", "Input $ / million tokens"],
      ["cached_input", "Cached input $ / million"],
      ["output", "Output $ / million tokens"],
      ["cache_write", "Cache write $ / million (optional)"],
      ["cache_write_1h", "1-hour cache write $ / million (optional)"],
    ]) {
      const input = field(settings, title, key, pricing[key]);
      input.type = "number";
      input.min = "0";
      input.step = "any";
      input.placeholder = "Automatic if known";
    }
    const unit = field(
      settings,
      "Cohere $ / search unit",
      "search_unit_usd",
      spec.search_unit_usd,
    );
    unit.type = "number";
    unit.min = "0";
    unit.step = "any";
    const options = field(
      settings,
      "Advanced inference options (JSON)",
      "options",
      JSON.stringify(spec.options || {}),
    );
    options.placeholder = '{"num_predict":256,"temperature":0}';
    details.append(
      settings,
      node(
        "p",
        "Unknown prices stay unknown. Explicit rates override the registry. Local API charges are $0; optional hardware estimates are separate.",
        "muted",
      ),
    );
    row.append(details);
    if (kind !== "judge") {
      const remove = node("button", "Remove", "btn btn-secondary remove-model");
      remove.type = "button";
      remove.addEventListener("click", () => {
        row.remove();
        plan();
      });
      row.append(remove);
    }
    function update() {
      const simple = ["none", "heuristic"].includes(provider.value);
      model.required = !simple;
      model.parentElement.hidden = simple;
      customModel.parentElement.hidden = simple || model.value !== "__custom__";
      customModel.required = !simple && model.value === "__custom__";
      discovery.hidden = simple;
      refresh.hidden = simple;
      llmProvider.parentElement.hidden = provider.value !== "llm";
      jevMethod.parentElement.hidden = provider.value !== "jev";
      details.hidden = simple;
      unit.parentElement.hidden = provider.value !== "cohere";
      stream.parentElement.hidden =
        kind === "reranker" && provider.value !== "llm";
    }
    let loadVersion = 0,
      available = [];
    function describePrice() {
      update();
      const selected = available.find((m) => m.id === model.value);
      const old = row.querySelector(".model-price");
      if (old) old.remove();
      if (selected) {
        const p = selected.pricing;
        row.append(
          node(
            "p",
            p
              ? `Automatic standard rate: $${p.input} input / $${p.cached_input} cached / $${p.output} output per million tokens${p.as_of ? " · verified " + p.as_of : ""}.`
              : "Price not in the registry. Add your rates under inference settings to calculate cost.",
            "model-price muted",
          ),
        );
      }
    }
    async function loadModels(preferred = "", force = false) {
      const version = ++loadVersion;
      const selectedProvider =
        provider.value === "llm" ? llmProvider.value : provider.value;
      if (["none", "heuristic"].includes(selectedProvider)) {
        row.dataset.loading = "false";
        update();
        plan();
        return;
      }
      row.dataset.loading = "true";
      row.querySelector(".model-price")?.remove();
      plan();
      model.replaceChildren(new Option("Loading models…", ""));
      discovery.textContent = "Fetching models from the provider…";
      try {
        const params = new URLSearchParams({
          provider: selectedProvider,
          host: $("ollama-host").value,
          refresh: String(force),
        });
        const data = await request("/evalground/model-catalog?" + params);
        if (version !== loadVersion || !row.isConnected) return;
        available = data.models;
        model.replaceChildren();
        available.forEach((m) =>
          model.append(
            new Option(m.name === m.id ? m.id : `${m.name} · ${m.id}`, m.id),
          ),
        );
        if (preferred && !available.some((m) => m.id === preferred))
          model.append(new Option(`${preferred} (saved/custom)`, preferred));
        model.append(new Option("Enter a custom model…", "__custom__"));
        model.value = preferred || available[0]?.id || "__custom__";
        discovery.textContent = `${available.length} models · ${data.message}`;
        describePrice();
        row.dataset.loading = "false";
        plan();
      } catch (e) {
        if (version !== loadVersion) return;
        model.replaceChildren(
          new Option("Enter a custom model…", "__custom__"),
        );
        discovery.textContent = e.message;
        row.dataset.loading = "false";
        update();
        plan();
      }
    }
    model.addEventListener("change", () => {
      row.dataset.label = "";
      describePrice();
      plan();
    });
    customModel.addEventListener("input", plan);
    refresh.addEventListener("click", () =>
      loadModels(
        model.value === "__custom__" ? customModel.value : model.value,
        true,
      ),
    );
    llmProvider.addEventListener("change", () => {
      row.dataset.label = "";
      loadModels();
    });
    jevMethod.addEventListener("change", () => {
      row.dataset.label = "";
      plan();
    });
    provider.addEventListener("change", () => {
      row.dataset.label = "";
      settings
        .querySelectorAll("input[type=number]")
        .forEach((i) => (i.value = ""));
      options.value = "{}";
      update();
      plan();
      loadModels();
    });
    update();
    $(container).append(row);
    plan();
    loadModels(spec.model || "");
  }
  function specs(id) {
    return [...$(id).children].map((row) => {
      const get = (k) => row.querySelector(`[data-key="${k}"]`),
        provider = get("provider").value;
      const spec = {
        provider,
        label: row.dataset.label || "",
        jev_method: get("jev_method").value,
        model: (get("model").value === "__custom__"
          ? get("custom_model").value
          : get("model").value
        ).trim(),
        llm_provider: get("llm_provider").value,
        stream: get("stream").checked,
      };
      if (["none", "heuristic"].includes(provider)) return { provider };
      try {
        spec.options = JSON.parse(get("options").value || "{}");
      } catch {
        throw Error("Inference options must be valid JSON.");
      }
      const rates = ["input", "cached_input", "output"];
      if (rates.some((k) => get(k).value !== "")) {
        if (!rates.every((k) => get(k).value !== ""))
          throw Error(
            "Enter all three token prices, including zero where appropriate.",
          );
        spec.pricing = Object.fromEntries(
          rates.map((k) => [k, Number(get(k).value)]),
        );
      }
      for (const key of ["cache_write", "cache_write_1h"]) {
        if (get(key).value !== "") {
          if (!spec.pricing)
            throw Error(
              "Enter the three base token prices before overriding cache-write prices.",
            );
          spec.pricing[key] = Number(get(key).value);
        }
      }
      if (provider === "cohere" && get("search_unit_usd").value !== "")
        spec.search_unit_usd = Number(get("search_unit_usd").value);
      return spec;
    });
  }
  function plan() {
    const type = $("experiment-type").value;
    $("reranker-section").hidden = type === "reader";
    $("rerankers")
      .querySelectorAll("input,select")
      .forEach((input) => (input.disabled = type === "reader"));
    $("add-reader").hidden = type === "reranker";
    $("start-comparison").disabled =
      submitting ||
      activeStatuses.has(currentResult?.status) ||
      [...document.querySelectorAll(".model-row")].some(
        (row) =>
          row.dataset.loading === "true" &&
          !(type === "reader" && row.dataset.kind === "reranker"),
      );
    const count =
      $("readers").children.length *
      (type === "reader" ? 1 : $("rerankers").children.length) *
      Number($("repeats").value);
    $("experiment-purpose").textContent = {
      reader:
        "Change the answer model. Keep retrieved evidence fixed so differences reflect the model’s response.",
      reranker:
        "Change the ranking method. Keep the initial candidates and answer model fixed.",
      pipeline:
        "Test each answer model with each reranker on the same initial candidates.",
    }[type];
    $("reader-section-title").textContent =
      type === "reranker" ? "Fixed answer model" : "Answer models to compare";
    for (const [id, kind] of [
      ["readers", "answer model"],
      ["rerankers", "reranker"],
      ["judge", "judge"],
    ]) {
      [...$(id).children].forEach(
        (row, i) =>
          (row.querySelector(".model-role").textContent =
            kind === "judge"
              ? "Fixed evaluation model"
              : type === "reranker" && kind === "answer model"
                ? "Fixed across all methods"
                : i === 0
                  ? `Baseline ${kind}`
                  : `Alternative ${i} · ${kind}`),
      );
    }
    const snapshot = !!$("candidate-snapshot").value.trim();
    $("snapshot-note").textContent = snapshot
      ? "Saved pool selected. Initial retrieval will not run again."
      : "";
    for (const id of ["embedding", "ollama-host"]) $(id).disabled = snapshot;
    $("retrieval-settings-note").textContent = snapshot
      ? "Saved candidates supply the evidence. Ollama embedding settings are unused in this replay."
      : "Ollama supplies query and document embeddings. Initial retrieval is shared across configurations; local hardware estimates exclude ingestion and idle time.";
    $("run-plan").textContent =
      `${count} configuration runs × ${$("limit").value || 0} questions = up to ${count * Number($("limit").value || 0)} evaluated answers. This starts live inference. Provider charges apply.`;
    const modelList = (id) =>
      [...$(id).children]
        .map((r) => {
          const input = r.querySelector("[data-key=model]");
          const model =
            input?.value === "__custom__"
              ? r.querySelector("[data-key=custom_model]").value
              : input?.value || r.querySelector("[data-key=provider]").value;
          return window.MemorizzComparisonAnalysis.modelName({
            model,
            provider: r.querySelector("[data-key=provider]").value,
            label: r.dataset.label,
            jev_method: r.querySelector("[data-key=jev_method]").value,
          });
        })
        .join(", ");
    const fields = [
      [
        "Comparison",
        {
          reader: "Answer models",
          reranker: "Rerankers",
          pipeline: "Memory pipelines",
        }[type],
      ],
      [
        "Dataset",
        snapshot
          ? "Saved candidate snapshot"
          : $("dataset").selectedOptions[0]?.textContent,
      ],
      ["Answer models", modelList("readers")],
      [
        "Rerankers",
        type === "reader" ? "Original candidate order" : modelList("rerankers"),
      ],
      [
        "Retrieval",
        snapshot
          ? "Fixed evidence · original retrieval excluded"
          : `Ollama · ${$("embedding").value}`,
      ],
      [
        "API spend threshold",
        $("max-cost").value
          ? Number($("max-cost").value).toLocaleString("en-US", {
              style: "currency",
              currency: "USD",
              maximumFractionDigits: 6,
            })
          : "Not set",
      ],
    ];
    $("review-summary").replaceChildren(
      ...fields.map(([label, value]) => {
        const item = node("div");
        item.append(
          node("span", label),
          node("strong", value || "Select a model"),
        );
        return item;
      }),
    );
  }
  function datasetChanged() {
    const dataset = $("dataset").value,
      spec = catalog.find((s) => s.benchmark_id === dataset);
    const builtin = ["demo", "memory_checks"].includes(dataset);
    $("path-field").hidden = builtin;
    $("data-path").required = !builtin;
    $("variant-field").hidden = !spec;
    $("variant").replaceChildren();
    if (spec) {
      spec.variants.forEach((v) => {
        const option = node("option", v);
        option.value = v;
        $("variant").append(option);
      });
      $("variant").value = spec.default_variant;
      if (!$("data-path").value) $("data-path").value = spec.dataset_path || "";
    }
    $("dataset-help").textContent = builtin
      ? "Sample data checks that the integration works. Use a held-out dataset from your application before making a model-selection decision."
      : "Use labeled questions with expected answers and relevant source IDs. Evalground reads the server file and saves the selected cases with the run.";
    plan();
  }
  function config() {
    const type = $("experiment-type").value;
    return {
      name: $("experiment-name").value,
      experiment_type: type,
      dataset: $("dataset").value,
      data_path: $("data-path").value,
      candidate_snapshot_path: $("candidate-snapshot").value,
      variant: $("variant-field").hidden ? null : $("variant").value,
      limit: Number($("limit").value),
      readers: specs("readers"),
      rerankers:
        type === "reader" ? [{ provider: "none" }] : specs("rerankers"),
      judge: specs("judge")[0],
      repeats: Number($("repeats").value),
      seed: Number($("seed").value),
      top_k: Number($("top-k").value),
      candidate_pool_size: Number($("pool").value),
      embedding_model: $("embedding").value,
      ollama_host: $("ollama-host").value,
      oracle_reader: $("oracle").checked,
      max_seconds: Number($("minutes").value) * 60,
      max_cost_usd:
        $("max-cost").value === "" ? null : Number($("max-cost").value),
      local_hourly_usd: Number($("local-cost").value),
    };
  }
  async function request(url, options) {
    const r = await fetch(url, options);
    const data = await r.json();
    if (!r.ok)
      throw Error(
        typeof data.detail === "string"
          ? data.detail
          : `Request failed (${r.status})`,
      );
    return data;
  }
  const num = (n, d = 2) => (n == null ? "—" : Number(n).toFixed(d)),
    pct = (n) => (n == null ? "—" : `${num(n * 100, 1)}%`),
    money = (n) => (n == null ? "Unknown" : `$${num(n, 6)}`),
    seconds = (n) => (n == null ? "—" : `${num(n)}s`);
  const analysis = window.MemorizzComparisonAnalysis;
  const modelName = analysis.modelName;
  function table(headers, rows) {
    const wrap = node("div", undefined, "compare-table-wrap"),
      t = node("table"),
      thead = node("thead"),
      tr = node("tr");
    headers.forEach((h) => tr.append(node("th", h)));
    thead.append(tr);
    t.append(thead);
    const body = node("tbody");
    rows.forEach((values) => {
      const r = node("tr");
      values.forEach((v) => r.append(node("td", String(v ?? "—"))));
      body.append(r);
    });
    t.append(body);
    wrap.append(t);
    return wrap;
  }
  function renderDetails() {
    const run = currentResult?.runs.find(
      (r) => r.id === $("inspect-run").value,
    );
    $("run-details").replaceChildren();
    if (!run) {window.MemorizzComparisonEvidence?.render(currentResult,null);return;}
    window.MemorizzComparisonEvidence?.render(currentResult, run);
    const s = run.summary || {},
      cards = node("div", undefined, "metric-cards");
    const rankingCards =
      currentResult?.config?.experiment_type === "reranker"
        ? [
            [`nDCG@${currentResult.config.top_k}`, num(s.ndcg_at_k, 3)],
            [`MRR@${currentResult.config.top_k}`, num(s.mrr, 3)],
            [`Recall@${currentResult.config.top_k}`, num(s.recall_at_k, 3)],
            [
              `Precision@${currentResult.config.top_k}`,
              num(s.precision_at_k, 3),
            ],
            [
              "Reranker p50 / p95",
              `${seconds(analysis.economics(run, "reranker").p50)} / ${seconds(analysis.economics(run, "reranker").p95)}`,
            ],
            [
              "Reranker API $ / question",
              money(analysis.economics(run, "reranker").perQuestion),
            ],
          ]
        : [];
    [
      ...rankingCards,
      [currentResult?.accuracy_label || "Accuracy", pct(s.accuracy)],
      [
        "Full pipeline p50 / p95",
        `${seconds(s.latency_p50_seconds)} / ${seconds(s.latency_p95_seconds)}`,
      ],
      ["Serving API $ / question", money(s.cost_per_question_usd)],
      ["Total API cost", money(s.total_cost_usd)],
      ["Evaluation API cost", money(s.evaluation_cost_usd)],
      ["Score", num(s.score, 3)],
      ["Reader-call TTFT p50", seconds(s.ttft_p50_seconds)],
      ["Output tokens / sec", num(s.output_tokens_per_second)],
      ["Reranker p95", seconds(s.reranker_p95_seconds)],
      [
        "API cost / correct",
        s.samples && s.correct === 0
          ? "— (0 correct)"
          : money(s.cost_per_correct_usd),
      ],
      [
        "Local compute estimate",
        s.local_compute_estimate_usd == null
          ? "Not configured"
          : money(s.local_compute_estimate_usd),
      ],
      ["Unpriced calls", s.unpriced_calls ?? 0],
      ["Token reporting", s.usage_complete ? "Complete" : "Partial"],
    ].forEach(([label, value]) => {
      const c = node("div", undefined, "metric-card");
      c.append(node("strong", String(value)), node("span", label));
      cards.append(c);
    });
    $("run-details").append(cards);
    if (s.accuracy_ci95)
      $("run-details").append(
        node(
          "p",
          `${currentResult.accuracy_label || "Accuracy"}: ${pct(s.accuracy)} (${s.correct} / ${s.samples}); Wilson 95% interval ${pct(s.accuracy_ci95.lower)} to ${pct(s.accuracy_ci95.upper)}. These questions are not a random sample of production traffic.`,
          "notice",
        ),
      );
    if (run.error) $("run-details").append(node("p", run.error));
    $("run-details").append(
      table(
        [
          "Call lane",
          "Calls",
          "Input tokens",
          "Cached input¹",
          "Output tokens",
          "Reasoning¹",
          "Cache writes¹",
          "1h writes²",
          "Cache hit rate³",
          "API cost",
        ],
        Object.entries(s.lanes || {}).map(([lane, v]) => [
          lane,
          v.calls,
          v.prompt_tokens,
          v.cached_tokens,
          v.completion_tokens,
          v.reasoning_tokens,
          v.cache_write_tokens ?? 0,
          v.cache_write_1h_tokens ?? 0,
          v.cache_reported_calls
            ? `${((100 * v.cache_hit_calls) / v.cache_reported_calls).toFixed(1)}% (${v.cache_hit_calls}/${v.cache_reported_calls})`
            : "Unknown",
          money(v.cost_usd),
        ]),
      ),
      node(
        "p",
        "¹ Cache reads and writes are subsets of input; reasoning is included in output. ² 1-hour writes are a subset of cache writes. ³ Calls that read from the provider prompt cache, among calls that reported cache usage. Throughput includes prefill. Unreported optional counts display zero; inspect provider call records for availability.",
        "muted",
      ),
    );
    const ci = s.score_ci95;
    if (ci)
      $("run-details").append(
        node(
          "p",
          `Mean score ${num(ci.mean, 3)}; bootstrap 95% interval [${num(ci.lower, 3)}, ${num(ci.upper, 3)}] over ${ci.samples} cases. Small datasets cannot establish a reliable winner.`,
          "muted",
        ),
      );
    const caseTitle = node("h3", "Per-question answers & evidence");
    $("run-details").append(caseTitle);
    (run.cases || []).forEach((c) => {
      const detail = node("details", undefined, "case-detail");
      detail.append(
        node(
          "summary",
          `${c.correct ? "✓" : "✗"} ${c.question} · score ${num(c.score, 3)}`,
        ),
      );
      detail.append(
        node("p", `Expected: ${(c.answers || []).join(" / ")}`),
        node("p", `Answer: ${c.prediction}`),
        node(
          "p",
          `Retrieved: ${(c.retrieved_source_ids || []).join(", ")} | Relevant: ${(c.relevant_source_ids || []).join(", ")}`,
        ),
      );
      (c.evidence || []).forEach((memory) => {
        const item = node("div", undefined, "evidence-card");
        item.append(
          node("strong", memory.source_id),
          node("p", memory.content),
        );
        detail.append(item);
      });
      const pre = node(
        "pre",
        JSON.stringify(
          {
            candidate_source_ids: c.candidate_source_ids,
            retrieval: c.retrieval,
            retrieval_trace: c.retrieval_trace,
            reader_output: c.reader_output,
            measurements: c.measurements,
          },
          null,
          2,
        ),
      );
      const raw = node("details");
      raw.append(node("summary", "Raw measurements and retrieval trace"), pre);
      detail.append(raw);
      $("run-details").append(detail);
    });
    const calls = node("details");
    calls.append(
      node("summary", "All measured calls · prices, usage, timing & failures"),
      node("pre", JSON.stringify(run.calls || [], null, 2)),
    );
    $("run-details").append(calls);
  }
  function render(data) {
    currentResult = data;
    $("experiment-results").hidden = false;
    $("toggle-setup").hidden = false;
    $("toggle-setup").disabled = !data.config;
    $("result-name").textContent = analysis.experimentName(data.name);
    $("result-name").title = data.name || "";
    $("comparison-page-title").textContent = "Evaluation results";
    $("comparison-page-description").textContent =
      "Compare outcomes, inspect evidence and trace each measurement to its source.";
    $("result-status").textContent = data.status;
    const active = activeStatuses.has(data.status);
    $("stop-comparison").hidden = !active;
    $("start-comparison").disabled = active;
    $("result-error").textContent = data.error || data.warning || "";
    $("measurement-scope").textContent = data.config?.candidate_snapshot_path
      ? "Frozen Oracle + Voyage candidates, live model calls. Ranking quality uses source labels; lexical answer checks require separate factual review. Full-pipeline measurements here include reranking and reading. Original embedding/retrieval work is excluded and retained in JSON source_provenance. Model loading is separate."
      : "Full-pipeline timing includes shared initial retrieval, reranking and answer generation. API costs exclude ingestion and local hardware. Judge/oracle calls are evaluation overhead. Unknown billing is never treated as zero.";
    window.MemorizzComparisonCharts?.render(data, (id) => {
      $("inspect-run").value = id;
      renderDetails();
      showResultTab("evidence");
      $("configuration-inspector").scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    });
    const runs = data.runs || [],
      cfg = data.config;
    const planned = cfg
      ? cfg.readers.length * cfg.rerankers.length * cfg.repeats
      : "?";
    $("result-progress").textContent =
      `${runs.filter((r) => r.status === "completed").length} / ${planned} runs completed · ${runs.reduce((n, r) => n + (r.cases || []).length, 0)} answers evaluated`;
    $("result-context").textContent =
      `${data.dataset_label || ""}${data.dataset_fingerprint ? " · Dataset SHA-256: " + data.dataset_fingerprint : ""}`;
    $("result-context").style.overflowWrap = "anywhere";
    $("export-json").href =
      `/evalground/comparisons/${currentId}/export?format=json`;
    $("export-csv").href =
      `/evalground/comparisons/${currentId}/export?format=csv`;
    const ranking = cfg?.experiment_type === "reranker",
      k = cfg?.top_k || "?";
    $("metric-explanations").replaceChildren(
      ...[
        [
          `nDCG@${k}`,
          "Rewards relevant evidence near the top, with stronger gains for higher relevance grades. DCG uses (2^grade − 1) / log2(rank + 1), divided by the ideal ordering of all gold grades. 1 is ideal; no positive labels means undefined.",
        ],
        [
          `MRR@${k}`,
          `Mean of 1 / rank of the first relevant result, counting only the first ${k} positions. No relevant result in that window contributes 0.`,
        ],
        [
          `Recall@${k}`,
          "Relevant records retrieved / all labeled relevant records, including any missed by the initial pool. The pool recall ceiling shows how much was available to the reranker.",
        ],
        [
          `Precision@${k}`,
          `Relevant records retrieved / ${k} requested slots. If a question has only one relevant memory and k = 3, perfect retrieval still has precision 1/3. Dataset values average per-question ratios.`,
        ],
        [
          data.accuracy_label || "Answer accuracy",
          "Fraction of answers passing the configured scorer. A lexical pass only checks expected words; it does not verify every claim, citation or contradiction.",
        ],
        [
          "Cost and latency",
          "p50 is the median; p95 is the interpolated 95th percentile of measured questions. Cost per question is selected-stage ledger spend / completed questions. Unknown prices stay unknown. Tiny samples and development-machine timings do not establish production performance.",
        ],
      ].map(([name, description]) => {
        const p = node("p");
        p.append(
          node("strong", name + ": "),
          document.createTextNode(description),
        );
        return p;
      }),
    );
    $("results-table").replaceChildren(
      table(
        ranking
          ? [
              "Reranking method",
              "Cases · status",
              `nDCG@${k}`,
              `MRR@${k}`,
              `Recall@${k}`,
              `Precision@${k}`,
              "Reranker p50 / p95",
              "Reranker API $ / question",
            ]
          : [
              "Configuration",
              "Cases · status",
              data.accuracy_label || "Accuracy",
              "Full pipeline p50 / p95",
              "Serving API $ / question",
              "Total API $",
            ],
        runs.map((r) => {
          const s = r.summary || {},
            e = analysis.economics(r, ranking ? "reranker" : "pipeline");
          const prefix = [
            analysis.runName(r, data),
            `${s.samples || 0} · ${r.status}`,
          ];
          return ranking
            ? [
                ...prefix,
                num(s.ndcg_at_k, 3),
                num(s.mrr, 3),
                num(s.recall_at_k, 3),
                num(s.precision_at_k, 3),
                `${seconds(e.p50)} / ${seconds(e.p95)}`,
                money(e.perQuestion),
              ]
            : [
                ...prefix,
                pct(s.accuracy),
                `${seconds(e.p50)} / ${seconds(e.p95)}`,
                money(e.perQuestion),
                money(s.total_cost_usd),
              ];
        }),
      ),
    );
    $("paired-results").replaceChildren();
    const pairDetails = node("details");
    pairDetails.append(
      node(
        "summary",
        ranking
          ? `Changes against the baseline · paired nDCG@${k}`
          : "Answer-score changes against the baseline",
      ),
    );
    $("paired-results").append(pairDetails);
    if (ranking)
      runs.slice(1).forEach((r) => {
        const pair = analysis.pairedMetric(r, runs[0], "ndcg_at_k");
        pairDetails.append(
          node(
            "p",
            `${analysis.runName(r, data)} vs ${analysis.runName(runs[0], data)}: nDCG@${k} ${num(pair.before, 3)} → ${num(pair.after, 3)} (${pair.delta >= 0 ? "+" : ""}${num(pair.delta, 3)}); ${pair.n} matched questions. Improved ${pair.improved.length}, regressed ${pair.regressed.length}. Point estimates only; no significance claim.`,
            "comparison-delta",
          ),
        );
        [
          ...pair.improved.map((p) => ["Improved", p]),
          ...pair.regressed.map((p) => ["Regressed", p]),
        ].forEach(([status, p]) =>
          pairDetails.append(
            node(
              "p",
              `${status}: ${p.question} (${num(p.before, 3)} → ${num(p.after, 3)})`,
              "muted",
            ),
          ),
        );
      });
    else
      (data.comparisons || []).forEach((c) => {
        const ci = c.score_delta_ci95;
        const name = (id) => {
          const r = runs.find((r) => r.id === id);
          return r ? analysis.runName(r, data) : "Unavailable configuration";
        };
        pairDetails.append(
          node(
            "p",
            `${name(c.run_id)} vs ${name(c.baseline_id)}: answer-score Δ ${num(ci.mean, 3)} (paired 95% interval ${num(ci.lower, 3)} to ${num(ci.upper, 3)}; ${c.paired_cases} cases). Improved: ${c.improved.length}. Regressed: ${c.regressed.length}. A zero-width interval on identical lexical checks does not prove equal factual quality.`,
            "comparison-delta",
          ),
        );
      });
    const selected = $("inspect-run").value;
    $("inspect-run").replaceChildren();
    runs.forEach((r) => {
      const o = node("option", analysis.runName(r, data));
      o.value = r.id;
      $("inspect-run").append(o);
    });
    if (runs.some((r) => r.id === selected)) $("inspect-run").value = selected;
    renderDetails();
    $("result-warnings").replaceChildren(
      ...(data.warnings || []).map((w) => node("li", w)),
    );
    return active;
  }
  async function openExperiment(id) {
    showSetup(false);
    showResultTab("overview");
    clearTimeout(timer);
    currentId = id;
    history.replaceState(null, "", `?experiment=${id}`);
    async function poll() {
      try {
        const data = await request(`/evalground/comparisons/${id}`);
        if (id !== currentId) return;
        if (render(data)) timer = setTimeout(poll, 2000);
        else await loadHistory();
      } catch (e) {
        $("result-error").textContent = e.message;
        timer = setTimeout(poll, 5000);
      }
    }
    await poll();
  }
  async function loadHistory() {
    try {
      const data = await request("/evalground/comparisons");
      $("comparison-history").replaceChildren();
      if (!data.experiments.length)
        $("comparison-history").textContent =
          "No comparisons yet. Start with the synthetic demo above.";
      data.experiments.forEach((e) => {
        const item = node("div", undefined, "history-item"),
          button = node("button", e.name, "btn btn-secondary");
        button.type = "button";
        button.addEventListener("click", () => openExperiment(e.id));
        item.append(
          button,
          node(
            "span",
            `${e.status} · ${new Date(e.created_at * 1000).toLocaleString()}`,
          ),
        );
        $("comparison-history").append(item);
      });
    } catch (e) {
      $("comparison-history").textContent = e.message;
    }
  }
  $("comparison-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (setupStep < 2) {
      goStep(setupStep + 1);
      return;
    }
    for (let i = 0; i < 3; i++) if (!validateStep(i)) return;
    submitting = true;
    $("form-error").textContent = "";
    $("start-comparison").disabled = true;
    try {
      const result = await request("/evalground/comparisons", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config()),
      });
      await openExperiment(result.id);
      $("experiment-results").scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
      await loadHistory();
    } catch (e) {
      $("form-error").textContent = e.message;
      $("start-comparison").disabled = false;
    } finally {
      submitting = false;
      plan();
    }
  });
  $("stop-comparison").addEventListener("click", async () => {
    try {
      await request(`/evalground/comparisons/${currentId}/stop`, {
        method: "POST",
      });
      await openExperiment(currentId);
    } catch (e) {
      $("result-error").textContent = e.message;
    }
  });
  function applyConfig(c) {
    if (!c) return;
    if (!activeStatuses.has(currentResult?.status)) {
      clearTimeout(timer);
      currentId = null;
      currentResult = null;
      $("experiment-results").hidden = true;
      $("toggle-setup").hidden = true;
      history.replaceState(null, "", `?type=${c.experiment_type}`);
      $("comparison-page-title").textContent = "New comparison";
      $("comparison-page-description").textContent =
        "Review this configuration, then run a new comparison.";
    }
    showSetup(true);
    goStep(0, false);
    const values = {
      "experiment-name": analysis.experimentName(c.name),
      "experiment-type": c.experiment_type,
      dataset: c.dataset,
      "data-path": c.data_path,
      "candidate-snapshot": c.candidate_snapshot_path || "",
      limit: c.limit,
      repeats: c.repeats ?? 1,
      seed: c.seed ?? 0,
      "top-k": c.top_k,
      pool: c.candidate_pool_size,
      embedding: c.embedding_model,
      "ollama-host": c.ollama_host || "http://localhost:11434",
      minutes: c.max_seconds / 60,
      "max-cost": c.max_cost_usd ?? "",
      "local-cost": c.local_hourly_usd ?? 0,
    };
    Object.entries(values).forEach(([id, v]) => ($(id).value = v));
    datasetChanged();
    if (c.variant) $("variant").value = c.variant;
    $("oracle").checked = c.oracle_reader;
    ["readers", "rerankers", "judge"].forEach((id) => $(id).replaceChildren());
    c.readers.forEach((s) => addModel("readers", s));
    c.rerankers.forEach((s) => addModel("rerankers", s, "reranker"));
    addModel("judge", c.judge, "judge");
    plan();
    $("comparison-form").scrollIntoView({ behavior: "smooth" });
  }
  $("reuse-config").addEventListener("click", () =>
    applyConfig(currentResult?.config),
  );
  for (const [id, kind] of [
    ["load-system-one", "reranking"],
    ["load-system-one-readers", "readers"],
  ]) {
    $(id).addEventListener("click", async () => {
      try {
        applyConfig(
          await request(`/evalground/system-one-preset?kind=${kind}`),
        );
        $("preset-note").textContent =
          kind === "readers"
            ? "Opus 5.5, GPT-6 Sol, and GPT-6 Luna: six questions, five frozen evidence items, medium effort, 4,096 output tokens. Effort labels do not imply identical compute across providers."
            : "Seven methods, six questions, twenty identical Oracle candidates, top three evidence items, one fixed OpenAI reader. No model response is replayed.";
      } catch (e) {
        $("preset-note").textContent = e.message;
      }
    });
  }
  $("add-reader").addEventListener("click", () =>
    addModel("readers", { provider: "openai", model: "" }),
  );
  $("add-reranker").addEventListener("click", () =>
    addModel(
      "rerankers",
      { provider: "voyage", model: "rerank-2.5" },
      "reranker",
    ),
  );
  $("experiment-type").addEventListener("change", () => {
    if ($("experiment-type").value === "reranker") {
      while ($("readers").children.length > 1)
        $("readers").lastElementChild.remove();
    }
    plan();
  });
  $("dataset").addEventListener("change", datasetChanged);
  [
    "repeats",
    "limit",
    "max-cost",
    "candidate-snapshot",
    "embedding",
    "top-k",
    "pool",
  ].forEach((id) => $(id).addEventListener("input", plan));
  $("inspect-run").addEventListener("change", renderDetails);
  document.querySelectorAll("[data-show-setup]").forEach((a) =>
    a.addEventListener("click", () => {
      showSetup(true);
      goStep(0, false);
      $("sample-presets").open = true;
    }),
  );
  if (location.hash === "#experiment-library")
    $("experiment-library").open = true;
  const entryType = new URLSearchParams(location.search).get("type");
  if (["reader", "reranker", "pipeline"].includes(entryType))
    $("experiment-type").value = entryType;
  if (entryType)
    $("experiment-name").value =
      {
        reader: "Answer-model comparison",
        reranker: "Reranking comparison",
        pipeline: "Memory pipeline comparison",
      }[entryType] || "Model comparison";
  addModel("readers", { provider: "openai", model: "gpt-4.1-mini" });
  if (entryType !== "reranker")
    addModel("readers", { provider: "openai", model: "gpt-6-luna" });
  addModel("rerankers", { provider: "none" }, "reranker");
  if (entryType === "reranker")
    addModel(
      "rerankers",
      { provider: "voyage", model: "rerank-2.5" },
      "reranker",
    );
  addModel("judge", { provider: "openai", model: "gpt-4.1-mini" }, "judge");
  datasetChanged();
  plan();
  goStep(0, false);
  loadHistory();
  const saved = new URLSearchParams(location.search).get("experiment");
  if (saved) openExperiment(saved);
})();

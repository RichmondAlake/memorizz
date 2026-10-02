/* Evalground: run a harness on Terminal-Bench 4.0.
   Checks what this machine has, keeps the form's fields to the chosen
   harness, picks tasks, estimates the cost, and follows the run's log. */
(function () {
  const form = document.getElementById("tb-form");
  if (!form) return;
  const data = JSON.parse(document.getElementById("tb-data")?.textContent || "{}");
  const $ = (id) => document.getElementById(id);
  const harness = $("tb-harness");
  const model = $("tb-model");
  const status = $("tb-status");
  const start = $("tb-start");
  const stop = $("tb-stop");
  const liveSection = $("live-run-output-section");
  const liveOutput = $("live-run-output");
  let machine = null;
  let activeRun = null;

  const MODELS = {
    codex: ["openai/gpt-5.6-terra", "openai/gpt-5.6-sol", "openai/gpt-6-astra", "openai/gpt-5.5"],
    "claude-code": ["anthropic/claude-opus-5-5", "anthropic/claude-sonnet-5-5", "anthropic/claude-haiku-4-5"],
    memagent: ["openai/gpt-5.6-terra", "openai/gpt-5.6-sol", "openai/gpt-5.5"],
  };
  // Rough USD per task with frontier models: agentic runs vary a lot by task.
  const PER_TASK = { codex: [0.5, 2], "claude-code": [1, 3], memagent: [0.5, 2], oracle: [0, 0] };

  const tasks = () => [...form.querySelectorAll('input[name="tasks"]')];
  const chosen = () => tasks().filter((box) => box.checked);

  function setCheck(name, ok, text) {
    const item = document.querySelector(`#tb-checks [data-check="${name}"]`);
    if (!item) return;
    item.dataset.state = ok === null ? "idle" : ok ? "ok" : "bad";
    item.querySelector("small").textContent = text;
  }

  function renderChecks() {
    if (!machine) return;
    setCheck("harbor", machine.harbor.ok, machine.harbor.detail);
    const docker = machine.docker;
    setCheck(
      "docker",
      docker.ok,
      docker.ok
        ? `${docker.detail}${docker.cpus ? ` · ${docker.cpus} CPUs` : ""}${docker.memory_gb ? ` · ${docker.memory_gb} GB` : ""}`
        : docker.detail,
    );
    const name = harness.value;
    const auth = $("tb-codex-auth").value;
    const keys = machine.keys;
    const need = {
      codex: auth === "chatgpt" ? ["codex_chatgpt", "~/.codex/auth.json"] : ["openai", "OPENAI_API_KEY"],
      "claude-code": ["anthropic", "ANTHROPIC_API_KEY"],
      memagent: ["openai", "OPENAI_API_KEY"],
    }[name];
    if (!need) setCheck("key", true, "Not needed for reference solutions");
    else setCheck("key", !!keys[need[0]], keys[need[0]] ? `${need[1]} found` : `${need[1]} missing`);
    const memoryOn = $("tb-memory").value === "on" && name !== "oracle";
    if (!memoryOn) setCheck("memory", null, "Off");
    else setCheck("memory", !!machine.memory_root, machine.memory_root ? "Connected FileSystem store" : "Needs a FileSystem store");
    const box = $("tb-concurrent-hint");
    if (box && docker.ok && docker.memory_gb) {
      const fits = Math.max(1, Math.floor(docker.memory_gb / 4));
      box.textContent = `Docker has ${docker.memory_gb} GB; most tasks need 4 GB each, so about ${fits} at once.`;
    }
  }

  function syncHarness() {
    const name = harness.value;
    form.querySelectorAll("[data-tb-for]").forEach((field) => {
      field.hidden = !field.dataset.tbFor.split(" ").includes(name);
    });
    const memoryOn = $("tb-memory").value === "on";
    form.querySelectorAll("[data-tb-memory]").forEach((field) => {
      if (field.dataset.tbFor.split(" ").includes(name)) field.hidden = !memoryOn;
    });
    const list = $("tb-models");
    list.replaceChildren(
      ...(MODELS[name] || []).map((value) => Object.assign(document.createElement("option"), { value })),
    );
    const fallback = (data.default_models || {})[name] || "";
    if (!model.value || !(MODELS[name] || []).some((m) => m.split("/")[0] === model.value.split("/")[0])) {
      model.value = fallback;
    }
    model.placeholder = fallback;
    $("tb-model-hint").textContent =
      name === "memagent" ? "OpenAI models only; the MemAgent paces itself to the time limit." : "Write it as provider/model.";
    $("tb-codex-warning").hidden = !(name === "codex" && $("tb-codex-auth").value === "chatgpt");
    renderChecks();
    estimate();
  }

  function estimate() {
    const count = chosen().length;
    const attempts = Math.max(1, Number($("tb-attempts").value) || 1);
    $("tb-task-count").textContent = count ? `${count} selected` : "none selected";
    const line = $("tb-estimate");
    const name = harness.value;
    const trials = count * attempts;
    const minutes = Math.round(Number($("tb-timeout").value) * 480);
    if (!count) {
      line.textContent = "Choose at least one task.";
      return;
    }
    if (name === "oracle") {
      line.textContent = `${trials} trial${trials === 1 ? "" : "s"} of the reference solutions: no model calls, so no API cost. Checks Docker and grading.`;
      return;
    }
    const [low, high] = PER_TASK[name];
    const chatgpt = name === "codex" && $("tb-codex-auth").value === "chatgpt";
    const money = chatgpt
      ? "billed to your ChatGPT plan, not per token"
      : `roughly $${(low * trials).toFixed(0)}–$${(high * trials).toFixed(0)} in API cost`;
    const cap = name === "memagent" ? `, never more than $${(Number($("tb-cost").value) * trials).toFixed(0)}` : "";
    line.textContent = `${trials} trial${trials === 1 ? "" : "s"} of up to ${minutes} minutes each: ${money}${cap}. Costs vary a lot by task.`;
  }

  function pick(which) {
    const smoke = new Set(data.smoke || []);
    tasks().forEach((box) => {
      box.checked = which === "smoke" ? smoke.has(box.value) : false;
    });
    estimate();
  }

  function filterTasks() {
    const query = $("tb-task-filter").value.trim().toLowerCase();
    form.querySelectorAll("[data-tb-group]").forEach((group) => {
      let shown = 0;
      group.querySelectorAll(".tb-task").forEach((label) => {
        const match = !query || label.dataset.name.toLowerCase().includes(query);
        label.hidden = !match;
        shown += match ? 1 : 0;
      });
      group.hidden = shown === 0;
    });
  }

  function running(runId) {
    activeRun = runId;
    start.disabled = !!runId;
    start.textContent = runId ? "Running…" : "Run Terminal-Bench";
    stop.hidden = !runId;
    stop.disabled = false;
  }

  async function follow(runId) {
    let cursor = 0;
    running(runId);
    if (liveSection) liveSection.hidden = false;
    for (;;) {
      const response = await fetch(`/evalground/runs/${encodeURIComponent(runId)}?after=${cursor}`, {
        headers: { Accept: "application/json" },
      });
      if (!response.ok) throw new Error("Lost connection to the run.");
      const payload = await response.json();
      const lines = Array.isArray(payload.logs) ? payload.logs : [];
      if (window.memorizzEvalMonitor) window.memorizzEvalMonitor.syncRunPayload(runId, payload);
      if (lines.length && liveOutput) {
        liveOutput.textContent += (liveOutput.textContent ? "\n" : "") + lines.join("\n");
        liveOutput.scrollTop = liveOutput.scrollHeight;
      }
      cursor = typeof payload.next_index === "number" ? payload.next_index : cursor + lines.length;
      if (["queued", "running", "canceling"].includes(payload.status)) {
        status.textContent =
          payload.status === "canceling"
            ? "Stopping: Harbor is taking the task containers down…"
            : "Running. The first trial downloads its task image, which can take a few minutes.";
        await new Promise((resolve) => setTimeout(resolve, 1500));
        continue;
      }
      if (payload.status === "completed") {
        status.textContent = "Finished. Loading results…";
        window.location.href = `/evalground?run_id=${encodeURIComponent(runId)}`;
        return;
      }
      status.textContent = payload.status === "canceled" ? "Run stopped." : payload.error || "The run failed.";
      running(null);
      return;
    }
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!chosen().length) {
      status.textContent = "Choose at least one task.";
      return;
    }
    start.disabled = true;
    status.textContent = "Starting…";
    if (liveOutput) liveOutput.textContent = "";
    try {
      const response = await fetch("/evalground/terminal-bench/runs", {
        method: "POST",
        body: new FormData(form),
        headers: { Accept: "application/json" },
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok || !payload.run_id) throw new Error(payload.error || payload.detail || "Could not start the run.");
      if (window.memorizzEvalMonitor) {
        window.memorizzEvalMonitor.trackRun(payload.run_id, { status: payload.status || "queued", keepLogs: false });
      }
      await follow(payload.run_id);
    } catch (error) {
      status.textContent = error?.message || String(error);
      running(null);
    }
  });

  stop.addEventListener("click", async () => {
    if (!activeRun) return;
    stop.disabled = true;
    status.textContent = "Stopping: Harbor is taking the task containers down…";
    try {
      const response = await fetch(`/evalground/runs/${encodeURIComponent(activeRun)}/stop`, {
        method: "POST",
        headers: { Accept: "application/json" },
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error || "Could not stop the run.");
      }
    } catch (error) {
      status.textContent = error?.message || String(error);
      stop.disabled = false;
    }
  });

  form.addEventListener("change", (event) => {
    if (event.target.matches('input[name="tasks"]')) estimate();
    else syncHarness();
  });
  form.addEventListener("input", (event) => {
    if (event.target.id === "tb-task-filter") filterTasks();
    else if (event.target.matches("#tb-attempts, #tb-cost")) estimate();
  });
  form.querySelectorAll("[data-tb-pick]").forEach((button) =>
    button.addEventListener("click", () => pick(button.dataset.tbPick)),
  );

  // Reopen a run's settings when its results are shown.
  const run = data.run;
  if (run) {
    if (run.harness) harness.value = run.harness;
    if (run.model && run.harness !== "oracle") model.value = run.model;
    if (run.codex_auth) $("tb-codex-auth").value = run.codex_auth;
    if (run.attempts) $("tb-attempts").value = run.attempts;
    if (run.n_concurrent) $("tb-concurrent").value = run.n_concurrent;
    if (run.agent_timeout_multiplier) $("tb-timeout").value = String(run.agent_timeout_multiplier);
    if (run.memory_id) {
      $("tb-memory").value = "on";
      $("tb-memory-id").value = run.memory_id;
    }
    if (["queued", "running", "canceling"].includes(run.status)) {
      follow(run.run_id).catch((error) => {
        status.textContent = error?.message || String(error);
        running(null);
      });
    }
  }
  syncHarness();
  fetch("/evalground/terminal-bench/status", { headers: { Accept: "application/json" } })
    .then((response) => (response.ok ? response.json() : Promise.reject()))
    .then((payload) => {
      machine = payload;
      renderChecks();
    })
    .catch(() => {
      ["harbor", "docker", "key", "memory"].forEach((name) => setCheck(name, false, "Could not check"));
    });
})();

// What a swap touches, shown before it happens: the memory store, the agents
// in it (and those that keep their own choice), and the other MemoRizz
// processes using the same store, which may need a restart.
(() => {
  const h = (tag, attrs = {}, ...kids) => {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (value === null || value === undefined || value === false) continue;
      if (key === "class") node.className = value;
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value === true ? "" : value);
    }
    for (const kid of kids.flat(Infinity)) {
      if (kid === null || kid === undefined || kid === false) continue;
      node.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    }
    return node;
  };
  const JOB = { memory_routing: "memory routing", embedding: "memory embeddings", reranking: "memory reranking" };
  const STORE = { oracle: "Oracle", filesystem: "Filesystem", mongodb: "MongoDB" };
  const processName = (p) => ({ ui: `UI${p.port ? " on port " + p.port : ""}`, mcp: "MCP server", cli: "CLI", run: "CLI run", chat: "CLI chat", serve: "server" }[p.kind] || p.kind) + ` (process ${p.pid})`;

  // Resolves false (cancelled), true, or for an embedding swap-in
  // { dimensions } (the vector size chosen from the version's plan).
  async function confirmSwap({ card, action = "in", label = "", runId = "" }) {
    let impact;
    let plan = null;
    try {
      const response = await fetch(`/api/finetune/models/${card}/impact`);
      impact = await response.json();
      if (!response.ok) throw new Error(impact.detail || "Could not check what the swap affects");
      if (card === "embedding" && action === "in" && runId) {
        const planned = await fetch(`/api/finetune/runs/${runId}/embedding-plan`);
        plan = await planned.json();
        if (!planned.ok) throw new Error(plan.detail || "Could not check the vector size");
      }
    } catch (error) {
      return window.confirm(`${error.message}. Swap anyway?`);
    }
    const a = impact.agents;
    const own = a.own_choice || [];
    const restart = impact.restart_needed;
    const others = impact.processes || [];
    const unknown = impact.unregistered || [];
    const title = action === "in" ? `Swap in ${label || "this version"} for ${JOB[card]}?` : `Swap ${JOB[card]} back?`;
    const dialog = h("dialog", { class: "sw-dialog", "aria-labelledby": "sw-title" });
    const done = (value) => {
      dialog.close();
      dialog.remove();
      resolve(value);
    };
    let resolve;
    const answer = new Promise((r) => { resolve = r; });
    const rows = [
      h("dt", {}, "Memory store"), h("dd", {}, `${STORE[impact.store.type] || impact.store.type} · ${impact.store.id}`),
      h("dt", {}, "Agents"),
      h("dd", {},
        card === "embedding"
          ? `All ${a.total} agents in this store: every memory is one model's vectors, so embeddings cannot differ by agent.`
          : own.length
            ? `${a.follow_store} of ${a.total} agents. ${own.length} keep their own choice: ${own.map((x) => x.name).join(", ")}.`
            : `All ${a.total} agents in this store (none has its own choice).`,
        a.names && a.names.length ? h("details", {}, h("summary", {}, "Show them"), h("p", { class: "sw-names" }, a.names.join(" · "))) : null),
      h("dt", {}, "Saved in"),
      h("dd", {}, card === "memory_routing" ? "The memory store: every process sees it on its next write." : `${impact.home}/.env, so it holds after a restart.`),
    ];
    let chosen = plan ? plan.default : null;
    if (plan) {
      const sizes = plan.options.length > 1
        ? h("div", { class: "sw-sizes", role: "radiogroup", "aria-label": "Vector size" },
            plan.options.map((o) => {
              const input = h("input", { type: "radio", name: "sw-size", value: o.dimensions });
              input.checked = o.dimensions === plan.default;
              input.addEventListener("change", () => { chosen = o.dimensions; });
              return h("label", { class: "sw-size" }, input,
                h("span", {}, o.text, o.dimensions === plan.default ? h("small", {}, " · scored on the card") : null,
                  o.kind === "resize" ? h("small", { class: "sw-size-note" }, `Changes every vector column to ${o.dimensions}; processes still on the old size fail to write until they restart.`) : null));
            }))
        : h("span", {}, plan.options.length ? plan.options[0].text : plan.reason);
      rows.push(h("dt", {}, "Vector size"), h("dd", {}, plan.store ? h("span", { class: "sw-muted" }, `This store holds ${plan.store}-dimension vectors; the model makes ${plan.native}. `) : null, sizes));
    }
    if (impact.reembeds) {
      rows.push(h("dt", {}, "Memories"), h("dd", {},
        !impact.staged
          ? "Every stored memory is re-embedded with the new model, in place. This can take a few minutes; nothing changes if it fails."
          : action === "in"
            ? "Every stored memory is re-embedded beside the live vectors, so search keeps working on the current ones until a single switch at the end. Nothing changes if it fails. The vectors it replaces are kept, so swapping back re-embeds only what changed since."
            : impact.kept && impact.kept.rows
              ? `Swapping back reuses the ${impact.kept.rows} vectors kept at the swap and re-embeds only memories written or edited since, then switches at once.`
              : "Every stored memory is re-embedded with the earlier model beside the live vectors, then switched at once."));
    }
    const warn = [];
    if (others.length) {
      warn.push(h("div", { class: restart ? "sw-warn" : "sw-note" },
        h("strong", {}, restart ? `Restart ${others.length === 1 ? "this process" : `these ${others.length} processes`} after the swap` : `${others.length} other process${others.length === 1 ? "" : "es"} use this store`),
        h("ul", {}, others.map((p) => h("li", {}, processName(p), p.embedding && p.embedding.model ? ` · embedding ${p.embedding.model.split("/").pop()}` : ""))),
        restart
          ? h("p", {}, card === "embedding"
              ? "Until they restart they keep embedding queries with the old model and search the re-embedded memories with it: nothing fails, the results are just wrong."
              : "Until they restart they keep the old reranker.")
          : h("p", {}, "They pick up the new router on their next write.")));
    } else {
      warn.push(h("p", { class: "sw-ok" }, "No other MemoRizz process on this machine reports using this store."));
    }
    const close = unknown.filter((p) => p.same_server);
    const rest = unknown.filter((p) => !p.same_server);
    if (close.length) {
      warn.push(h("div", { class: "sw-warn" },
        h("strong", {}, `${close.length} older MemoRizz process${close.length === 1 ? " is" : "es are"} connected to this database server (${impact.server})`),
        h("ul", {}, close.map((p) => h("li", {}, processName(p)))),
        h("p", {}, card === "embedding"
          ? "If it uses this store, it keeps searching with its own embedding model: check which store it is connected to, and stop it or point its MemoRizz home at the new model before swapping. A restart alone is not enough when its home names another model."
          : card === "memory_routing"
            ? "Nothing to do: the router is read from the store on every write."
            : "If it uses this store, restart it after the swap (until then it keeps the old reranker).")));
    }
    unknown.splice(0, unknown.length, ...rest);
    if (unknown.length) {
      warn.push(h("div", { class: "sw-note" },
        h("strong", {}, `${unknown.length} other MemoRizz process${unknown.length === 1 ? "" : "es"} could not say which store ${unknown.length === 1 ? "it uses" : "they use"}`),
        h("p", {}, card === "memory_routing"
          ? "They run an older MemoRizz. The router is read from the store, so they need nothing."
          : "They run an older MemoRizz. If any use this store, restart them after the swap" + (card === "embedding" ? " (they would search with the old embedding model)." : ".")),
        h("details", {}, h("summary", {}, "Show them"), h("ul", {}, unknown.map((p) => h("li", {}, processName(p)))))));
    }
    dialog.append(
      h("h2", { id: "sw-title" }, title),
      h("dl", { class: "sw-facts" }, rows),
      ...warn,
      h("div", { class: "sw-actions" },
        h("button", { class: "btn btn-primary", type: "button", disabled: plan && !plan.options.length, onclick: () => done(plan ? { dimensions: chosen } : true) }, action === "in" ? (card === "embedding" ? "Swap in and re-embed" : "Swap in") : "Swap back"),
        h("button", { class: "btn", type: "button", onclick: () => done(false) }, "Cancel")),
    );
    dialog.addEventListener("cancel", (event) => { event.preventDefault(); done(false); });
    document.body.append(dialog);
    dialog.showModal();
    return answer;
  }

  window.MemorizzSwap = { confirm: confirmSwap };
})();

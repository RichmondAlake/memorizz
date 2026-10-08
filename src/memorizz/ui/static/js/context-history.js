/* Saved turns contain actual fitted requests; the current preview is an estimate. */
(() => {
  "use strict";
  const root = document.querySelector("[data-context-navigator]");
  if (!root) return;
  const panel = root.closest('[role="tabpanel"]'),
    preview = panel.querySelector("[data-context-preview]"),
    body = panel.querySelector("[data-context-snapshot]"),
    heading = panel.querySelector("[data-context-heading]"),
    count = document.getElementById("context-window-count");
  const prev = root.querySelector("[data-context-previous]"),
    next = root.querySelector("[data-context-next]"),
    position = root.querySelector("[data-context-position]");
  let turns = [],
    index = 0,
    memory = "",
    sequence = 0,
    detailSequence = 0,
    olderCursors = {},
    loadingOlder = false,
    previewCount = count.textContent;
  const linkParams = new URLSearchParams(location.search);
  const requestedTurn = (linkParams.get("turn_id") || "").slice(0, 256);
  let requestedTurnOpened = false;
  function scopedPath(path) {
    const url = new URL(path, location.origin);
    ["user_id", "application_id"].forEach((key) => {
      if (linkParams.has(key)) url.searchParams.set(key, linkParams.get(key));
    });
    return url.pathname + url.search;
  }
  function controls() {
    prev.disabled =
      loadingOlder ||
      (index <= 0 && !olderCursors.snapshots && !olderCursors.traces);
    next.disabled = loadingOlder || index >= turns.length;
    position.textContent =
      index === turns.length
        ? "Current context · estimate"
        : `Turn ${index + 1} of ${turns.length}`;
  }
  function element(tag, text, className) {
    const node = document.createElement(tag);
    node.textContent = text;
    if (className) node.className = className;
    return node;
  }
  function showPreview() {
    ++detailSequence;
    preview.hidden = false;
    body.hidden = true;
    heading.textContent = "Current context · estimate";
    count.textContent = previewCount;
    controls();
  }
  async function showCall(snapshotId) {
    const request = ++detailSequence,
      activeMemory = memory;
    body.replaceChildren(element("p", "Loading captured request…"));
    count.textContent = "";
    try {
      const response = await fetch(
        scopedPath(
          `/agents/${encodeURIComponent(root.dataset.agentId)}/playground/context-history/${encodeURIComponent(snapshotId)}?memory_id=${encodeURIComponent(activeMemory)}`,
        ),
        { cache: "no-store" },
      );
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = await response.json();
      if (request !== detailSequence || activeMemory !== memory) return;
      const snapshot = payload.snapshot;
      count.textContent = `${snapshot.message_count} messages`;
      body.replaceChildren();
      const calls = turns[index]?.calls || [];
      if (calls.length > 1) {
        const label = element("label", "Model request "),
          select = document.createElement("select");
        select.setAttribute("aria-label", "Model request in this turn");
        calls.forEach((call, i) => {
          const option = element(
            "option",
            `Call ${i + 1} · ${call.stage || "model"}`,
          );
          option.value = call.record_id;
          option.selected = call.record_id === snapshotId;
          select.appendChild(option);
        });
        select.addEventListener("change", () => showCall(select.value));
        label.appendChild(select);
        body.appendChild(label);
      }
      const previous = turns
        .slice(0, index)
        .reverse()
        .find((turn) => turn.calls.length)
        ?.calls.at(-1);
      const delta = previous
        ? Number(snapshot.estimated_tokens) - Number(previous.estimated_tokens)
        : null;
      body.appendChild(
        element(
          "p",
          `${Number(snapshot.estimated_tokens).toLocaleString()} estimated tokens · ${snapshot.tool_count} tools · ${snapshot.window_tokens || "unknown"} token window${delta !== null ? " · " + (delta >= 0 ? "+" : "") + delta + " since the previous captured turn" : ""}.`,
          "pg-panel-intro",
        ),
      );
      body.appendChild(
        element(
          "p",
          `${snapshot.model || "Model"} · ${snapshot.timestamp || ""}. Captured after prompt fitting, before provider-specific transformations.`,
          "pg-panel-intro",
        ),
      );
      if (payload.content_mode === "metadata") {
        body.appendChild(
          element("p", "Request content is hidden by the trace access policy."),
        );
        return;
      }
      (snapshot.messages || []).forEach((message, i) => {
        const details = document.createElement("details");
        details.className = "context-block context-block-collapsible";
        details.appendChild(
          element(
            "summary",
            `${i + 1}. ${message.role || "unknown"}`,
            "context-block-title",
          ),
        );
        details.appendChild(
          element(
            "pre",
            typeof message.content === "string"
              ? message.content
              : JSON.stringify(message, null, 2),
            "context-log-text",
          ),
        );
        if (message.tool_calls)
          details.appendChild(
            element(
              "pre",
              JSON.stringify(message.tool_calls, null, 2),
              "context-log-text",
            ),
          );
        body.appendChild(details);
      });
      const tools = document.createElement("details");
      tools.className = "context-block context-block-collapsible";
      tools.appendChild(
        element(
          "summary",
          `Tools sent (${snapshot.tool_count})`,
          "context-block-title",
        ),
      );
      tools.appendChild(
        element(
          "pre",
          JSON.stringify(snapshot.tools || [], null, 2),
          "context-log-text",
        ),
      );
      body.appendChild(tools);
    } catch (error) {
      if (request === detailSequence)
        body.replaceChildren(
          element("p", "Could not load captured request: " + error.message),
        );
    }
  }
  function show() {
    controls();
    if (index === turns.length) {
      showPreview();
      return;
    }
    preview.hidden = true;
    body.hidden = false;
    heading.textContent = "Sent to the model";
    const turn = turns[index];
    if (!turn.calls.length && olderCursors.snapshots && !loadingOlder) {
      ++detailSequence;
      body.replaceChildren(
        element("p", "Loading this turn’s captured requests…"),
      );
      ensureTurnRequest(turn.turn_id);
      return;
    }
    if (!turn.calls.length) {
      ++detailSequence;
      count.textContent =
        turn.status === "no_model_request" ? "0 requests" : "Unavailable";
      body.replaceChildren(
        element(
          "p",
          turn.reason || "No captured request is available for this turn.",
        ),
      );
      return;
    }
    showCall(turn.calls.at(-1).record_id);
  }
  async function fetchOlder(selected, request, snapshotsOnly = false) {
    const activeMemory = memory;
    const params = new URLSearchParams({ memory_id: activeMemory });
    if (olderCursors.snapshots)
      params.set("snapshot_cursor", olderCursors.snapshots);
    else params.set("skip_snapshots", "true");
    if (!snapshotsOnly && olderCursors.traces)
      params.set("trace_cursor", olderCursors.traces);
    else params.set("skip_traces", "true");
    const response = await fetch(
      scopedPath(
        `/agents/${encodeURIComponent(root.dataset.agentId)}/playground/context-history?${params}`,
      ),
      { cache: "no-store" },
    );
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    if (request !== sequence || memory !== activeMemory) return false;
    const merged = new Map(turns.map((turn) => [turn.turn_id, turn]));
    (payload.turns || []).forEach((turn) => {
      const existing = merged.get(turn.turn_id);
      if (existing) {
        const calls = [
          ...new Map(
            [...turn.calls, ...existing.calls].map((call) => [
              call.record_id,
              call,
            ]),
          ).values(),
        ].sort((a, b) =>
          String(a.timestamp).localeCompare(String(b.timestamp)),
        );
        merged.set(turn.turn_id, {
          ...turn,
          ...existing,
          calls,
          status: calls.length ? "recorded" : existing.status,
        });
      } else merged.set(turn.turn_id, turn);
    });
    turns = [...merged.values()].sort(
      (a, b) =>
        String(a.timestamp).localeCompare(String(b.timestamp)) ||
        a.turn_id.localeCompare(b.turn_id),
    );
    olderCursors = {
      snapshots: payload.next_cursors?.snapshots,
      traces: snapshotsOnly
        ? olderCursors.traces
        : payload.next_cursors?.traces,
    };
    index = Math.max(
      0,
      turns.findIndex((turn) => turn.turn_id === selected),
    );
    return true;
  }
  async function ensureTurnRequest(selected) {
    const request = sequence;
    loadingOlder = true;
    controls();
    let failed = false;
    try {
      // Trace pages can cover more turns than request pages when tools loop.
      // Exhaust relevant request pages before calling a turn unavailable.
      while (olderCursors.snapshots && !turns[index]?.calls.length) {
        if (!(await fetchOlder(selected, request, true))) return;
      }
    } catch (error) {
      failed = true;
      if (request === sequence)
        body.replaceChildren(
          element("p", "Could not load older requests: " + error.message),
        );
    } finally {
      if (request === sequence) {
        loadingOlder = false;
        controls();
        if (!failed) show();
      }
    }
  }
  prev.addEventListener("click", async () => {
    if (loadingOlder) return;
    if (index === 0 && (olderCursors.snapshots || olderCursors.traces)) {
      const request = sequence,
        selected = turns[index]?.turn_id;
      loadingOlder = true;
      controls();
      let errorText = "";
      try {
        if (!(await fetchOlder(selected, request))) return;
      } catch (error) {
        errorText = "Could not load older turns: " + error.message;
        return;
      } finally {
        if (request === sequence) {
          loadingOlder = false;
          controls();
          if (errorText) position.textContent = errorText;
        }
      }
    }
    if (index > 0) {
      index--;
      show();
    }
  });
  next.addEventListener("click", () => {
    if (index < turns.length) {
      index++;
      show();
    }
  });
  window.MemorizzContextHistory = {
    async refresh(memoryId) {
      const request = ++sequence,
        wasPreview = index === turns.length,
        selected = turns[index]?.turn_id,
        changed = memory !== memoryId;
      memory = memoryId;
      loadingOlder = false;
      previewCount = count.textContent;
      if (changed) {
        turns = [];
        olderCursors = {};
        index = 0;
        showPreview();
      }
      if (!memoryId) return;
      try {
        const response = await fetch(
          scopedPath(
            `/agents/${encodeURIComponent(root.dataset.agentId)}/playground/context-history?memory_id=${encodeURIComponent(memoryId)}`,
          ),
          { cache: "no-store" },
        );
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const payload = await response.json();
        if (request !== sequence) return;
        turns = payload.turns || [];
        olderCursors = payload.next_cursors || {};
        if (
          requestedTurn &&
          !requestedTurnOpened &&
          (!linkParams.get("memory_id") ||
            linkParams.get("memory_id") === memoryId)
        ) {
          loadingOlder = true;
          controls();
          position.textContent = "Finding the requested turn…";
          while (
            !turns.some((turn) => turn.turn_id === requestedTurn) &&
            (olderCursors.snapshots || olderCursors.traces)
          ) {
            const before = JSON.stringify(olderCursors);
            if (!(await fetchOlder(requestedTurn, request))) return;
            if (before === JSON.stringify(olderCursors)) break;
          }
          if (request !== sequence) return;
          loadingOlder = false;
          requestedTurnOpened = true;
          const selectedIndex = turns.findIndex(
            (turn) => turn.turn_id === requestedTurn,
          );
          if (selectedIndex >= 0) {
            index = selectedIndex;
            show();
          } else {
            ++detailSequence;
            preview.hidden = true;
            body.hidden = false;
            heading.textContent = "Requested turn context";
            body.replaceChildren(
              element(
                "p",
                "No captured request is available for this turn in this scope. Use the arrows to inspect other turns.",
              ),
            );
            index = turns.length;
            controls();
            position.textContent = "Requested turn unavailable";
            count.textContent = "Unavailable";
          }
          return;
        }
        index =
          wasPreview || changed
            ? turns.length
            : Math.max(
                0,
                turns.findIndex((turn) => turn.turn_id === selected),
              );
        show();
      } catch (error) {
        if (request === sequence) {
          loadingOlder = false;
          position.textContent = "History unavailable: " + error.message;
          prev.disabled = true;
          next.disabled = true;
        }
      }
    },
  };
  const current = document.getElementById("pg-memory-id")?.value?.trim();
  if (current) window.MemorizzContextHistory.refresh(current);
})();

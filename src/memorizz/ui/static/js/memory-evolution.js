/* One renderer for playground, observability and harness comparisons. */
(() => {
  "use strict";
  // base.html's escapeHtml; the inline copy keeps the module working standalone.
  const esc = typeof escapeHtml === "function" ? escapeHtml : (value) =>
    String(value ?? "").replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );
  const dateOf = (value) => {
    if (!value) return null;
    const date = new Date(value);
    return Number.isFinite(date.getTime()) ? date : null;
  };
  const friendly = (value) =>
    String(value || "")
      .replace(/^MemoryType\./, "")
      .toLowerCase()
      .replace(/_/g, " ");
  const elapsed = (milliseconds) => {
    const seconds = Math.round(milliseconds / 1000);
    if (seconds < 1) return "<1s";
    if (seconds < 60) return seconds + "s";
    const minutes = Math.floor(seconds / 60),
      remainder = seconds % 60;
    if (minutes < 60)
      return minutes + "m" + (remainder ? " " + remainder + "s" : "");
    const hours = Math.floor(minutes / 60),
      minutePart = minutes % 60;
    if (hours < 24)
      return hours + "h" + (minutePart ? " " + minutePart + "m" : "");
    const days = Math.floor(hours / 24),
      hourPart = hours % 24;
    return days + "d" + (hourPart ? " " + hourPart + "h" : "");
  };
  class MemoryEvolution {
    constructor(root) {
      this.root = root;
      this.events = [];
      this.existing = [];
      this.cursor = null;
      this.sequence = 0;
      this.selected = null;
      this.writers = {};
      this.valueSequence = 0;
      this.focused = false;
      this.compact = root.hasAttribute("data-me-compact");
      this.step = this.compact ? 238 : 288;
      this.cardWidth = this.compact ? 216 : 256;
      this.laneHeight = 188;
      this.gutter = this.compact ? 0 : 132;
      this.axisHeight = 64;
      this.q = (name) => root.querySelector(`[data-me-${name}]`);
      if (
        this.q("group") &&
        new URLSearchParams(location.search).get("group") === "writer"
      )
        this.q("group").value = "writer";
      this.localZone =
        Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
      this.q("timezone").options[0].textContent = this.localZone;
      this.q("timezone").addEventListener("change", () => {
        this.render();
        if (this.selected) this.select(this.selected);
        this.pendingFocus = true;
        this.focusLatest();
      });
      ["type", "action", "actor", "execution", "existing"].forEach((name) =>
        this.q(name).addEventListener("change", () => {
          this.focused = false;
          this.render();
          this.pendingFocus = true;
          this.focusLatest();
        }),
      );
      this.q("search").addEventListener("input", () => {
        this.focused = false;
        this.render();
        this.pendingFocus = true;
        this.focusLatest();
      });
      this.q("group")?.addEventListener("change", () => {
        this.render();
        this.pendingFocus = true;
        this.focusLatest();
      });
      this.q("related")?.addEventListener("click", () => {
        this.focused = !this.focused;
        this.render();
        if (this.selected) this.select(this.selected);
        this.pendingFocus = true;
        this.focusLatest();
      });
      this.q("latest")?.addEventListener("click", () => {
        const latest = this.visible?.at(-1);
        if (latest) this.reveal(latest.record_id);
      });
      this.q("detail").addEventListener("click", (event) => {
        if (event.target.closest("[data-me-close]")) {
          const selected = this.selected;
          this.clearSelection();
          this.highlight();
          this.q("nodes")
            .querySelector(`[data-event-id="${CSS.escape(selected || "")}"]`)
            ?.focus({ preventScroll: true });
        }
        if (event.target.closest("[data-me-current]")) this.loadCurrent();
        if (event.target.closest("[data-me-source-page]") && this.cursor)
          this.load(true);
        const step = event.target.closest("[data-me-step]");
        if (step) this.stepChange(Number(step.dataset.meStep));
        const link = event.target.closest("[data-me-reveal]");
        if (link) this.reveal(link.dataset.meReveal);
      });
      this.q("nodes").addEventListener("keydown", (event) => {
        const node = event.target.closest("[data-event-id]");
        if (
          !node ||
          !["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)
        )
          return;
        event.preventDefault();
        const index = this.visible.findIndex(
          (e) => e.record_id === node.dataset.eventId,
        );
        const next =
          event.key === "Home"
            ? 0
            : event.key === "End"
              ? this.visible.length - 1
              : Math.max(
                  0,
                  Math.min(
                    this.visible.length - 1,
                    index + (event.key === "ArrowRight" ? 1 : -1),
                  ),
                );
        this.reveal(this.visible[next].record_id);
        this.q("nodes")
          .querySelector(`[data-event-id="${CSS.escape(this.selected)}"]`)
          ?.focus({ preventScroll: true });
      });
      this.q("left").addEventListener("click", () => this.move(-1));
      this.q("right").addEventListener("click", () => this.move(1));
      this.q("refresh").addEventListener("click", () => this.load());
      this.q("older").addEventListener("click", () => this.load(true));
      this.q("scroll").addEventListener("scroll", () => this.scrollControls(), {
        passive: true,
      });
      this.resizeObserver = new ResizeObserver(() => this.focusLatest());
      this.resizeObserver.observe(this.q("scroll"));
      this.q("scroll").addEventListener(
        "wheel",
        (event) => {
          if (event.shiftKey && event.deltaY && !event.deltaX) {
            event.preventDefault();
            this.q("scroll").scrollLeft += event.deltaY;
          }
        },
        { passive: false },
      );
      this.q("scroll").addEventListener("keydown", (event) => {
        if (event.target !== this.q("scroll")) return;
        if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
          event.preventDefault();
          this.move(event.key === "ArrowLeft" ? -1 : 1);
        }
      });
      this.q("nodes").addEventListener("click", (event) => {
        const node = event.target.closest("[data-event-id]");
        if (node) {
          this.reveal(node.dataset.eventId);
          if (this.compact || innerWidth < 1200)
            this.q("detail").scrollIntoView({ block: "nearest" });
        }
      });
      this.load();
    }
    scrollControls() {
      const scroll = this.q("scroll");
      this.q("left").disabled = scroll.scrollLeft <= 1;
      this.q("right").disabled =
        scroll.scrollLeft + scroll.clientWidth >= scroll.scrollWidth - 1;
      this.q("axis")
        .querySelectorAll(".me-axis-tick")
        .forEach((tick) => {
          const covered =
            parseFloat(tick.style.left) - scroll.scrollLeft < this.gutter + 4;
          tick.style.visibility = covered ? "hidden" : "visible";
          tick.setAttribute("aria-hidden", String(covered));
        });
    }
    focusLatest() {
      const scroll = this.q("scroll");
      if (!scroll.clientWidth) return;
      if (this.pendingFocus) {
        const latest = this.visible?.at(-1);
        if (latest) {
          scroll.scrollLeft = Math.max(
            0,
            latest._x + this.cardWidth + 20 - scroll.clientWidth,
          );
          scroll.scrollTop = this.compact
            ? 0
            : Math.max(0, latest._y - this.axisHeight - 32);
        }
        this.pendingFocus = false;
      }
      this.scrollControls();
    }
    move(direction) {
      this.q("scroll").scrollBy({
        left:
          direction * Math.max(this.step, this.q("scroll").clientWidth * 0.75),
        behavior: matchMedia("(prefers-reduced-motion: reduce)").matches
          ? "auto"
          : "smooth",
      });
    }
    async load(older = false) {
      const sequence = ++this.sequence;
      this.root.classList.add("is-loading");
      const params = new URLSearchParams({
        limit: "200",
        include_existing: older ? "false" : "true",
      });
      ["agentId", "memoryId", "runId"].forEach((k) => {
        if (this.root.dataset[k])
          params.set(
            k.replace(/[A-Z]/g, (c) => "_" + c.toLowerCase()),
            this.root.dataset[k],
          );
      });
      // Preserve authenticated trace scope from the URL on embedded/page views.
      const current = new URLSearchParams(location.search);
      ["user_id", "application_id"].forEach((k) => {
        if (current.has(k)) params.set(k, current.get(k));
      });
      if (older && this.cursor) params.set("cursor", this.cursor);
      try {
        const response = await fetch("/traces/memory-history?" + params, {
          cache: "no-store",
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const page = await response.json();
        if (sequence !== this.sequence) return;
        this.events = older ? [...page.events, ...this.events] : page.events;
        this.events = [
          ...new Map(this.events.map((e) => [e.record_id, e])).values(),
        ];
        if (!older) this.existing = page.existing || [];
        this.writers = older
          ? { ...this.writers, ...page.writers }
          : page.writers || {};
        this.navigation = page.navigation || {};
        this.playgroundAvailable = page.playground_available !== false;
        this.cursor = page.next_cursor;
        this.q("older").hidden = !this.cursor;
        this.coverage = page.coverage;
        this.filters();
        this.render();
        this.renderNavigation();
        if (!older) {
          this.pendingFocus = true;
          this.focusLatest();
        }
      } catch (error) {
        if (sequence === this.sequence)
          this.q("status").textContent =
            "Could not load memory history: " + error.message;
      } finally {
        if (sequence === this.sequence)
          this.root.classList.remove("is-loading");
      }
    }
    filters() {
      const all = [...this.events, ...this.existing];
      [
        ["type", "memory_type"],
        ["actor", "actor"],
      ].forEach(([name, key]) => {
        const select = this.q(name),
          selected = select.value;
        select.innerHTML =
          `<option value="">All ${name === "type" ? "types" : "actors"}</option>` +
          [...new Set(all.map((e) => e[key]).filter(Boolean))]
            .sort()
            .map(
              (v) =>
                `<option value="${esc(v)}">${esc(name === "type" ? friendly(v) : this.writerLabel(v))}</option>`,
            )
            .join("");
        select.value = selected;
      });
      const execution = this.q("execution"),
        selected = execution.value;
      const identities = [
        ...new Set(
          this.events.map((e) => this.executionKey(e)).filter(Boolean),
        ),
      ];
      execution.innerHTML =
        '<option value="">All loaded executions</option>' +
        identities
          .map((key) => {
            const [kind, id] = key.split(/:(.*)/s);
            const events = this.events.filter(
              (e) => this.executionKey(e) === key,
            );
            return `<option value="${esc(key)}">${esc(friendly(kind))} ${esc(id.slice(0, 8))} · ${events.length} changes</option>`;
          })
          .join("") +
        '<option value="none">No execution ID recorded</option>';
      execution.value = selected;
    }
    executionKey(event) {
      return event.run_id
        ? "run:" + event.run_id
        : event.turn_id
          ? "turn:" + event.turn_id
          : "";
    }
    render() {
      const recorded = new Set(
        this.events.map((e) => e.memory_type + ":" + e.target_record_id),
      );
      const legacy = this.q("existing").checked
        ? this.existing.filter(
            (e) => !recorded.has(e.memory_type + ":" + e.target_record_id),
          )
        : [];
      const all = [
        ...legacy,
        ...this.events
          .slice()
          .sort(
            (a, b) =>
              (dateOf(a.timestamp)?.getTime() ?? -Infinity) -
                (dateOf(b.timestamp)?.getTime() ?? -Infinity) || 0,
          ),
      ];
      this.connections = this.edgesFor(all);
      const search = this.q("search").value.trim().toLowerCase();
      let visible = all.filter(
        (e) =>
          (!this.q("type").value || e.memory_type === this.q("type").value) &&
          (!this.q("action").value || e.action === this.q("action").value) &&
          (!this.q("actor").value || e.actor === this.q("actor").value) &&
          (!this.q("execution").value ||
            (this.q("execution").value === "none"
              ? !this.executionKey(e)
              : this.executionKey(e) === this.q("execution").value)) &&
          (!search ||
            [
              this.nodeLabel(e),
              this.writerLabel(e.actor),
              e.actor,
              e.source,
              e.source_ref,
              friendly(e.memory_type),
              e.target_record_id,
              e.run_id,
              ...(e.changed_fields || []),
            ]
              .join(" ")
              .toLowerCase()
              .includes(search)),
      );
      if (
        !this.focused &&
        this.selected &&
        !visible.some((e) => e.record_id === this.selected)
      )
        this.clearSelection();
      if (this.focused && this.selected) {
        const connected = this.relatedIds(this.selected);
        // Connected versions and sources can fall outside display filters.
        // Keep the controls set so leaving this view restores the selection.
        visible = all.filter((e) => connected.has(e.record_id));
      }
      this.visible = visible;
      const byWriter = this.q("group")?.value === "writer";
      const laneKey = (event) =>
        byWriter ? event.actor || "unknown" : event.memory_type;
      const lanes = [...new Set(visible.map(laneKey))];
      const gaps = [];
      let x = this.gutter + 20;
      visible.forEach((event, index) => {
        const previous = visible[index - 1];
        const from =
          previous?.action !== "observed" && dateOf(previous?.timestamp);
        const to = event.action !== "observed" && dateOf(event.timestamp);
        const milliseconds = from && to ? to - from : 0;
        if (milliseconds >= 60_000) {
          gaps.push({ x: x - 16, milliseconds });
          x += 88;
        }
        event._x = x;
        event._y =
          this.axisHeight +
          (this.compact
            ? 16
            : lanes.indexOf(laneKey(event)) * this.laneHeight + 32);
        x += this.step;
      });
      const canvas = this.q("canvas");
      canvas.style.width =
        Math.max(
          (visible.at(-1)?._x ?? this.gutter + 20) + this.cardWidth + 20,
          this.q("scroll").clientWidth,
        ) + "px";
      canvas.style.height = this.compact
        ? this.axisHeight + 156 + "px"
        : this.axisHeight +
          Math.max(lanes.length * this.laneHeight + 24, 240) +
          "px";
      this.renderAxis(visible, gaps);
      this.q("lanes").innerHTML = this.compact
        ? ""
        : lanes
            .map(
              (lane, i) =>
                `<div class="me-lane" style="top:${i * this.laneHeight}px" title="${esc(lane)}"><span>${byWriter ? this.writerLink(lane) : esc(friendly(lane))}</span><small>${visible.filter((e) => laneKey(e) === lane).length} ${visible.filter((e) => laneKey(e) === lane).length === 1 ? "change" : "changes"}</small></div>`,
            )
            .join("");
      this.q("nodes").innerHTML = visible
        .map((e) => {
          const label = this.nodeLabel(e);
          const actor = this.writerLabel(e.actor);
          const relation = e.parents?.length
            ? "Has explicit sources"
            : e.previous_event_id
              ? "Next recorded version"
              : "";
          return `<button type="button" class="me-node" data-action="${esc(e.action)}" data-event-id="${esc(e.record_id)}" aria-pressed="${e.record_id === this.selected}" style="left:${e._x}px;top:${e._y}px" title="${esc(label + " · " + actor + " · " + this.changeSummary(e))}"><span class="me-node-title"><span>${esc(e.action === "observed" ? "Existing · history unknown" : e.action)}</span><span title="${esc(relation)}">${e.parents?.length ? "↳ " + new Set(e.parents.map((p) => p.record_id)).size : e.previous_event_id ? "↻" : ""}</span></span><strong>${esc(label)}</strong><small class="me-node-type">${esc(friendly(e.memory_type))}</small><small title="${esc(e.actor)}">${esc(actor)}</small>${this.compact ? "" : `<small class="me-node-summary">${esc(this.changeSummary(e))}</small>`}</button>`;
        })
        .join("");
      if (!visible.length)
        this.q("nodes").innerHTML =
          `<p class="me-empty">${all.length ? "No changes match these filters." : "No recorded changes for this conversation yet."}</p>`;
      const visibleIds = new Set(visible.map((e) => e.record_id));
      const edges = this.connections.filter(
        ([a, b]) => visibleIds.has(a.record_id) && visibleIds.has(b.record_id),
      );
      this.q("links").innerHTML = edges
        .map(([a, b, branch]) => {
          const x = a._x + this.cardWidth,
            y = a._y + 68,
            tx = b._x,
            ty = b._y + 68,
            m = (x + tx) / 2;
          return `<path class="me-edge${branch ? " me-edge-branch" : ""}" data-from="${esc(a.record_id)}" data-to="${esc(b.record_id)}" d="M ${x} ${y} C ${m} ${y}, ${m} ${ty}, ${tx} ${ty}"/>`;
        })
        .join("");
      const recordedVisible = visible.filter(
        (e) => e.action !== "observed",
      ).length;
      const existingVisible = visible.length - recordedVisible;
      this.q("status").textContent =
        `${recordedVisible === this.events.length ? this.events.length : recordedVisible + " of " + this.events.length} changes${existingVisible ? " · " + existingVisible + " earlier records" : ""}` +
        (this.compact
          ? ""
          : ` · ${new Set(visible.map((e) => e.memory_type + ":" + e.target_record_id)).size} memories · ${new Set(visible.filter((e) => e.actor && e.actor !== "unknown").map((e) => e.actor)).size} known writers`);
      this.q("summary").innerHTML = ["created", "updated", "deleted"]
        .map(
          (action) =>
            `<span><strong>${visible.filter((e) => e.action === action).length}</strong> ${esc(action)}</span>`,
        )
        .join("");
      this.q("coverage-note").textContent =
        `Write history · does not establish memory use.${this.focused ? " Related changes includes connected versions and sources outside the display filters." : ""}${this.existing.length ? " " + this.existing.length + " current records have unknown history; use the checkbox to include them." : ""}${this.cursor ? " Older changes remain unloaded; counts and search cover loaded history." : ""}`;
      this.q("coverage").textContent =
        `${this.existing.length} existing records have unavailable history. ${this.coverage || ""}`;
      const filterCount = this.q("filter-count");
      if (filterCount) {
        const active =
          ["type", "action", "actor", "execution", "search"].filter(
            (name) => this.q(name).value,
          ).length + Number(this.q("existing").checked);
        filterCount.hidden = !active;
        filterCount.textContent = active;
      }
      if (this.q("related")) {
        this.q("related").disabled = !this.selected;
        this.q("related").setAttribute("aria-pressed", String(this.focused));
      }
      this.highlight();
      if (this.selected) this.select(this.selected);
      this.scrollControls();
    }
    clearSelection() {
      this.selected = null;
      ++this.valueSequence;
      this.focused = false;
      this.root.classList.remove("has-selection");
      this.q("detail").textContent =
        "Select a memory change to inspect its source, author, version and lineage.";
      this.q("detail").hidden = true;
    }
    edgesFor(events) {
      const byId = new Map(events.map((e) => [e.record_id, e])),
        byRecord = new Map(),
        edges = [],
        seen = new Set();
      events.forEach((event) => {
        const previous = byId.get(event.previous_event_id);
        if (previous) edges.push([previous, event, false]);
        (event.parents || []).forEach((parent) => {
          const source = byRecord.get(parent.record_id);
          const key = source?.record_id + ":" + event.record_id;
          if (source && !seen.has(key)) {
            seen.add(key);
            edges.push([source, event, true]);
          }
        });
        byRecord.set(event.target_record_id, event);
      });
      return edges;
    }
    relatedIds(id) {
      const related = new Set(id ? [id] : []),
        adjacency = new Map();
      (this.connections || []).forEach(([a, b]) => {
        for (const [from, to] of [
          [a, b],
          [b, a],
        ]) {
          if (!adjacency.has(from.record_id)) adjacency.set(from.record_id, []);
          adjacency.get(from.record_id).push(to.record_id);
        }
      });
      const pending = [...related];
      while (pending.length)
        for (const next of adjacency.get(pending.pop()) || []) {
          if (!related.has(next)) {
            related.add(next);
            pending.push(next);
          }
        }
      return related;
    }
    highlight() {
      const related = this.relatedIds(this.selected);
      this.q("nodes")
        .querySelectorAll("[data-event-id]")
        .forEach((node) => {
          node.classList.toggle(
            "me-muted",
            !!this.selected && !related.has(node.dataset.eventId),
          );
        });
      this.q("links")
        .querySelectorAll("path")
        .forEach((edge) => {
          edge.classList.toggle(
            "me-edge-focus",
            !!this.selected &&
              related.has(edge.dataset.from) &&
              related.has(edge.dataset.to),
          );
          edge.classList.toggle(
            "me-muted",
            !!this.selected && !related.has(edge.dataset.from),
          );
        });
    }
    writerLabel(identity) {
      if (!identity || identity === "unknown") return "Writer unknown";
      if (this.writers[identity]) return this.writers[identity];
      if (identity === this.root.dataset.agentId) return "This agent";
      return /^[a-f\d]{8}-[a-f\d-]{27}$/i.test(identity)
        ? "Agent " + identity.slice(0, 8)
        : identity;
    }
    scopeParams() {
      const current = new URLSearchParams(location.search),
        params = new URLSearchParams();
      ["user_id", "application_id"].forEach((key) => {
        if (current.has(key)) params.set(key, current.get(key));
      });
      return params;
    }
    trail() {
      try {
        const trail = JSON.parse(
          new URLSearchParams(location.search).get("memory_trail") || "[]",
        );
        if (!Array.isArray(trail)) return [];
        return trail
          .slice(-8)
          .filter((value) => value && typeof value === "object")
          .map((value) =>
            Object.fromEntries(
              ["agent_id", "memory_id", "run_id", "group"]
                .filter(
                  (key) =>
                    typeof value[key] === "string" && value[key].length <= 256,
                )
                .map((key) => [key, value[key]]),
            ),
          );
      } catch {
        return [];
      }
    }
    timelineHref(identity) {
      const params = this.scopeParams(),
        trail = this.trail();
      const previous = {
        agent_id: this.root.dataset.agentId || "",
        memory_id: this.root.dataset.memoryId || "",
        run_id: this.root.dataset.runId || "",
        group: this.q("group")?.value || "type",
      };
      if (previous.agent_id || previous.memory_id || previous.run_id)
        trail.push(previous);
      params.set("agent_id", identity);
      if (trail.length)
        params.set("memory_trail", JSON.stringify(trail.slice(-8)));
      return "/traces/memory-evolution?" + params;
    }
    writerLink(identity) {
      const label = esc(this.writerLabel(identity));
      return this.writers[identity] && identity !== this.root.dataset.agentId
        ? `<a class="me-agent-link" href="${esc(this.timelineHref(identity))}" data-me-agent-history="${esc(identity)}" title="Open this MemAgent's own memory history">${label} ↗</a>`
        : label;
    }
    renderNavigation() {
      const nav = this.q("agent-nav"),
        current = this.navigation?.agent,
        delegates = this.navigation?.delegates || [],
        trail = this.trail();
      let previous = "";
      if (trail.length) {
        const params = this.scopeParams(),
          scope = trail.pop();
        Object.entries(scope).forEach(([key, value]) => {
          if (value) params.set(key, value);
        });
        if (trail.length) params.set("memory_trail", JSON.stringify(trail));
        previous = `<a class="me-agent-link" data-me-back href="${esc("/traces/memory-evolution?" + params)}">← Previous timeline</a>`;
      }
      const ownHistory =
        current &&
        (this.compact || this.root.dataset.memoryId || this.root.dataset.runId)
          ? `<a class="me-agent-link" data-me-own-history href="${esc(this.timelineHref(current.agent_id))}">All agent history ↗</a>`
          : "";
      const inspectorParams = this.scopeParams();
      inspectorParams.set("inspector", "memory");
      const memoryTrail = this.trail();
      if (memoryTrail.length)
        inspectorParams.set("memory_trail", JSON.stringify(memoryTrail));
      const inspector =
        current && this.playgroundAvailable
          ? `<a class="me-agent-link" data-me-memory-inspector href="${esc("/agents/" + encodeURIComponent(current.agent_id) + "/playground?" + inspectorParams)}">Memory inspector ↗</a>`
          : "";
      nav.hidden = !current && !previous && !delegates.length;
      nav.innerHTML = `<div class="me-agent-nav-head"><div>${current ? `<small>MemAgent memory history</small><strong>${esc(current.name)}</strong>` : ""}</div><div class="me-agent-shortcuts">${previous}${ownHistory}${inspector}</div></div>${delegates.length ? `<div class="me-delegates"><span title="Delegates from the current saved configuration">Delegates</span>${delegates.map((agent) => `<a class="me-agent-link" data-me-delegate="${esc(agent.agent_id)}" href="${esc(this.timelineHref(agent.agent_id))}" title="Open this delegate's own memory history">${esc(agent.name)} →</a>`).join("")}</div>` : ""}`;
    }
    changeSummary(event) {
      if (event.action === "observed") return "Earlier changes unknown";
      if (event.action === "created") return "New memory";
      if (event.action === "deleted") return "Memory removed";
      const fields = event.changed_fields || [];
      return fields.length <= 2 && fields.length
        ? fields.map(friendly).join(" · ") + " updated"
        : fields.length
          ? fields.length + " fields updated"
          : "Memory updated";
    }
    stepChange(direction) {
      const index = this.visible.findIndex(
        (e) => e.record_id === this.selected,
      );
      const next = this.visible[index + direction];
      if (next) this.reveal(next.record_id);
    }
    reveal(id) {
      if (!this.visible.some((e) => e.record_id === id)) {
        ["type", "action", "actor", "execution", "search"].forEach(
          (name) => (this.q(name).value = ""),
        );
        this.focused = false;
        if (this.existing.some((e) => e.record_id === id))
          this.q("existing").checked = true;
        this.render();
      }
      const event = this.visible.find((e) => e.record_id === id);
      if (!event) return;
      this.select(id);
      const scroll = this.q("scroll");
      scroll.scrollLeft = Math.max(0, event._x - this.gutter - 20);
      scroll.scrollTop = this.compact
        ? 0
        : Math.max(0, event._y - this.axisHeight - 32);
      this.scrollControls();
    }
    stamp(value) {
      const date = dateOf(value);
      return date
        ? date.toLocaleString("en-GB", {
            timeZone:
              this.q("timezone").value === "UTC" ? "UTC" : this.localZone,
          })
        : "Time unknown";
    }
    renderAxis(events, gaps) {
      const zone = this.q("timezone").value === "UTC" ? "UTC" : this.localZone;
      const day = new Intl.DateTimeFormat("en-GB", {
        day: "2-digit",
        month: "short",
        year: "numeric",
        timeZone: zone,
      });
      const clock = new Intl.DateTimeFormat("en-GB", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hourCycle: "h23",
        timeZone: zone,
      });
      const known = events.filter(
        (e) => e.action !== "observed" && dateOf(e.timestamp),
      );
      const start = known.length && day.format(dateOf(known[0].timestamp));
      const end = known.length && day.format(dateOf(known.at(-1).timestamp));
      this.q("range").textContent = start
        ? start === end
          ? start
          : start + " – " + end
        : "Recorded change times";
      this.q("axis").innerHTML =
        (this.compact
          ? ""
          : '<span class="me-axis-corner">Recorded at</span>') +
        events
          .map((event) => {
            const date = event.action !== "observed" && dateOf(event.timestamp);
            const time = date
              ? `<time datetime="${esc(event.timestamp)}" title="${esc(this.stamp(event.timestamp) + " · " + zone)}"><strong>${esc(clock.format(date))}</strong><span>${esc(day.format(date))}</span></time>`
              : `<strong>${event.action === "observed" ? "History unknown" : "Time unknown"}</strong><span>${event.action === "observed" ? "Existing record" : "Timestamp unavailable"}</span>`;
            const execution = event.run_id
              ? `Run ${String(event.run_id).slice(0, 8)}`
              : event.turn_id
                ? `Turn ${String(event.turn_id).slice(0, 8)}`
                : "";
            return `<div class="me-axis-tick" style="left:${event._x}px" data-time-event-id="${esc(event.record_id)}">${time}${execution ? `<small title="${esc(event.run_id || event.turn_id)}">${esc(execution)}</small>` : ""}</div>`;
          })
          .join("") +
        gaps
          .map(
            (gap) =>
              `<span class="me-time-gap" style="left:${gap.x}px" title="${esc(elapsed(gap.milliseconds))} between recorded changes">↔ ${esc(elapsed(gap.milliseconds))}</span>`,
          )
          .join("");
    }
    nodeLabel(event) {
      if (event.memory_type === "conversation_memory") {
        const roles = {
          user: "User message",
          assistant: "Assistant reply",
          system: "System message",
          tool: "Tool result",
        };
        if (roles[event.label]) return roles[event.label];
      }
      if (event.label && event.label !== event.target_record_id)
        return event.label.startsWith("MemoryType.")
          ? friendly(event.label)
          : event.label;
      return (
        {
          agents: "Agent configuration",
          conversation_memory: "Conversation message",
          summaries: "Conversation summary",
          workflow_memory: "Workflow",
          tool_log: "Tool activity",
          semantic_cache: "Cached response",
          short_term_memory: "Working memory",
        }[event.memory_type] ||
        event.memory_type
          .replace(/_/g, " ")
          .replace(/^\w/, (c) => c.toUpperCase())
      );
    }
    select(id) {
      const event = this.visible.find((e) => e.record_id === id);
      if (!event) return;
      this.selected = id;
      ++this.valueSequence;
      this.root.classList.add("has-selection");
      this.q("nodes")
        .querySelectorAll("[data-event-id]")
        .forEach((node) =>
          node.setAttribute(
            "aria-pressed",
            String(node.dataset.eventId === id),
          ),
        );
      const pair = (key, value) =>
        `<dt>${esc(key)}</dt><dd>${esc(value ?? "Unavailable")}</dd>`;
      const record = pair("Record", event.target_record_id);
      const identifiers = `${pair("Run", event.run_id)}${pair("Turn", event.turn_id)}${pair("Before hash", event.before_hash)}${pair("After hash", event.after_hash)}${pair("Previous version", event.previous_event_id)}`;
      this.q("detail").hidden = false;
      const position = this.visible.findIndex((e) => e.record_id === id);
      const versionsLoaded = [...this.events]
        .filter(
          (e) =>
            e.memory_type === event.memory_type &&
            e.target_record_id === event.target_record_id,
        )
        .sort((a, b) => String(a.timestamp).localeCompare(String(b.timestamp)));
      const versionIndex = versionsLoaded.findIndex((e) => e.record_id === id);
      const traceParams = this.scopeParams();
      if (event.agent_id) traceParams.set("agent_id", event.agent_id);
      if (event.memory_id) traceParams.set("thread_memory_id", event.memory_id);
      ["run_id", "turn_id", "root_trace_id"].forEach((key) => {
        if (event[key]) traceParams.set(key, event[key]);
      });
      const hasExecution = event.run_id || event.turn_id || event.root_trace_id;
      const contextParams = this.scopeParams();
      contextParams.set("inspector", "context");
      if (event.memory_id) contextParams.set("memory_id", event.memory_id);
      if (event.turn_id) contextParams.set("turn_id", event.turn_id);
      const captured =
        event.agent_id &&
        event.memory_id &&
        event.turn_id &&
        this.writers[event.agent_id] &&
        this.playgroundAvailable;
      const ownership = event.source?.startsWith("shared-memory:")
        ? "Shared coordination"
        : event.agent_id
          ? "Agent-owned record"
          : "Ownership unknown";
      this.q("detail").innerHTML =
        `<div class="me-detail-controls"><span>Change ${position + 1} of ${this.visible.length} visible</span><button type="button" class="btn btn-sm" data-me-close aria-label="Close change details">×</button></div><p class="me-detail-eyebrow">${esc(event.action === "observed" ? "Existing record" : event.action)} · ${esc(friendly(event.memory_type))}</p><h3>${esc(this.nodeLabel(event))}</h3><div class="me-step-controls"><button type="button" class="btn btn-sm" data-me-step="-1" ${position === 0 ? "disabled" : ""}>← Previous change</button><button type="button" class="btn btn-sm" data-me-step="1" ${position === this.visible.length - 1 ? "disabled" : ""}>Next change →</button></div><dl>${pair(event.action === "observed" ? "Record timestamp" : "Recorded at", this.stamp(event.timestamp))}<dt>Writer</dt><dd>${this.writerLink(event.actor)}</dd><dt>Owner</dt><dd>${event.agent_id ? this.writerLink(event.agent_id) : "Unknown"}</dd>${pair("Initiated by", event.initiator ? this.writerLabel(event.initiator) : "Not recorded")}${pair("Storage", ownership)}${pair("Namespace", event.memory_id || "Not recorded")}${pair("Source", event.source)}${event.source_ref ? pair("Origin", event.source_ref) : ""}${pair("Execution", event.run_id ? "Run " + event.run_id.slice(0, 8) : event.turn_id ? "Turn " + event.turn_id.slice(0, 8) : "No execution ID recorded")}${versionIndex >= 0 ? pair("Loaded version", versionIndex + 1 + " of " + versionsLoaded.length) : ""}</dl><div class="me-evidence-links">${hasExecution ? `<a class="btn btn-sm" data-me-trace href="${esc("/traces?" + traceParams)}">Open execution trace ↗</a>` : ""}${captured ? `<a class="btn btn-sm" data-me-context href="${esc("/agents/" + encodeURIComponent(event.agent_id) + "/playground?" + contextParams)}">Captured turn context ↗</a>` : ""}</div>${event.changed_fields?.length ? `<div class="me-fields"><span>Changed fields</span>${event.changed_fields.map((field) => `<code title="${esc(field)}">${esc(friendly(field))}${event.field_changes?.[field] ? " · " + esc(event.field_changes[field]) : ""}</code>`).join("")}</div>` : ""}<div class="me-value-evidence"><strong>Value evidence</strong><p>Historical before/after values were not stored in this journal. Hashes establish a recorded change; they do not reconstruct its content.</p>${event.action !== "observed" ? '<button class="btn btn-sm" type="button" data-me-current>View current stored value</button><div data-me-values></div>' : "<p>This is a current-record observation, not a recorded change.</p>"}</div><details class="me-identifiers"><summary>Technical details</summary><dl>${record}${pair("Actor ID", event.actor)}${pair("Attribution", event.attribution || "Historical attribution unavailable")}${identifiers}</dl></details>${event.history_gap ? '<p class="me-gap-note">Earlier changes for this record are unavailable. No continuous version history is implied.</p>' : ""}`;
      const versions = (this.connections || []).filter(
        ([a, b, branch]) =>
          !branch && (a.record_id === id || b.record_id === id),
      );
      if (versions.length)
        this.q("detail").insertAdjacentHTML(
          "beforeend",
          `<div class="me-version-links">${versions.map(([a, b]) => `<button class="btn btn-sm" type="button" data-me-reveal="${esc(a.record_id === id ? b.record_id : a.record_id)}">${a.record_id === id ? "Next" : "Previous"} recorded version</button>`).join("")}</div>`,
        );
      if (event.parents?.length) {
        const heading = document.createElement("h4");
        heading.textContent = "Explicit lineage";
        this.q("detail").appendChild(heading);
        event.parents.forEach((parent) => {
          const source = (this.connections || []).find(
            ([a, b, branch]) =>
              branch &&
              b.record_id === id &&
              a.target_record_id === parent.record_id,
          )?.[0];
          const p = document.createElement("p");
          p.className = "me-lineage-source";
          p.innerHTML = source
            ? `${esc(friendly(parent.relation))} ← <button type="button" class="me-source-link" data-me-reveal="${esc(source.record_id)}">${esc(this.nodeLabel(source))}</button><small>${esc(this.writerLabel(source.actor))}</small>`
            : `${esc(friendly(parent.relation))} ← ${esc(parent.record_id)} <small>Source is outside the loaded scope</small>${this.cursor ? '<button class="btn btn-sm" type="button" data-me-source-page>Load older history</button>' : ""}`;
          this.q("detail").appendChild(p);
        });
      }
      const children = (this.connections || []).filter(
        ([a, b, branch]) => branch && a.record_id === id,
      );
      if (children.length)
        this.q("detail").insertAdjacentHTML(
          "beforeend",
          `<h4>Changes using this source</h4>${children.map(([, child]) => `<p><button type="button" class="me-source-link" data-me-reveal="${esc(child.record_id)}">${esc(this.nodeLabel(child))}</button><small> · ${esc(this.writerLabel(child.actor))}</small></p>`).join("")}`,
        );
      if (this.q("related")) this.q("related").disabled = false;
      this.highlight();
    }
    async loadCurrent() {
      const id = this.selected,
        sequence = ++this.valueSequence;
      const target = this.q("detail").querySelector("[data-me-values]");
      if (!target) return;
      target.textContent = "Loading current stored value…";
      const params = this.scopeParams();
      const event = this.visible.find((e) => e.record_id === id);
      if (event.agent_id) params.set("agent_id", event.agent_id);
      try {
        const response = await fetch(
          `/traces/memory-history/${encodeURIComponent(id)}/record?${params}`,
          { cache: "no-store" },
        );
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const value = await response.json();
        if (sequence !== this.valueSequence || id !== this.selected) return;
        const notices = {
          hidden: "Value content is hidden by the trace access policy.",
          configuration:
            "Agent configuration is available in that agent's Settings inspector.",
          deleted:
            "The target record has been deleted. Deleted values were not retained.",
          unavailable:
            "The current value is unavailable in this scope. Historical values were not retained.",
        };
        if (notices[value.state]) {
          target.textContent = notices[value.state];
          return;
        }
        target.innerHTML =
          `<p class="me-current-note">Current stored value · ${value.matches_selected_version ? "matches the selected version's hash" : "does not match the selected version; this is not a historical snapshot"}.</p>` +
          Object.entries(value.fields || {})
            .map(
              ([key, text]) =>
                `<details class="me-record-value" open><summary>${esc(friendly(key))}</summary><pre>${esc(text)}</pre></details>`,
            )
            .join("") +
          (value.truncated
            ? "<p>Value preview truncated to 16,000 characters.</p>"
            : "") +
          (!Object.keys(value.fields || {}).length
            ? "<p>No previewable content fields are available for this record.</p>"
            : "");
        target.scrollIntoView({ block: "nearest" });
      } catch (error) {
        if (sequence === this.valueSequence && id === this.selected)
          target.textContent = "Could not load current value: " + error.message;
      }
    }
    setScope(scope) {
      Object.entries(scope).forEach(([key, value]) => {
        this.root.dataset[key] = value || "";
      });
      this.clearSelection();
      this.load();
    }
  }
  window.MemorizzMemoryEvolution = {
    mount(root) {
      if (!root._memoryEvolution)
        root._memoryEvolution = new MemoryEvolution(root);
      return root._memoryEvolution;
    },
  };
  const init = () => {
    document
      .querySelectorAll("[data-memory-evolution]:not([data-me-lazy])")
      .forEach((root) => window.MemorizzMemoryEvolution.mount(root));
    const form = document.getElementById("memory-evolution-scope");
    if (form)
      form.addEventListener("submit", (event) => {
        event.preventDefault();
        const values = new FormData(form),
          params = new URLSearchParams();
        for (const [k, v] of values) if (v) params.set(k, v);
        location.search = params.toString();
      });
  };
  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", init);
  else init();
})();

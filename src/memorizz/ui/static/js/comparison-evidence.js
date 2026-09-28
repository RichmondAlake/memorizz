/* Before/after view of the recorded candidate pool and selected evidence. */
(() => {
  const A = window.MemorizzComparisonAnalysis;
  const el = (tag, text, cls) => {
    const n = document.createElement(tag);
    if (text !== undefined) n.textContent = text;
    if (cls) n.className = cls;
    return n;
  };
  let experiment,
    question,
    version = 0;
  const cache = new Map();
  function compareCard(
    row,
    position,
    beforeIds,
    gold,
    available,
    isAfter,
    trace = {},
  ) {
    const id = row.source_id,
      grade = gold[id] || 0,
      card = el(
        "article",
        undefined,
        "ranking-memory" + (available && grade > 0 ? " is-relevant" : ""),
      );
    const heading = el("div", undefined, "ranking-memory-heading");
    heading.append(
      el("strong", `#${position} · ${id}`),
      el(
        "span",
        available
          ? grade > 0
            ? `Relevant · grade ${grade}`
            : "Not labeled relevant"
          : "Relevance not labeled",
        "relevance-tag",
      ),
    );
    card.append(
      heading,
      el("p", row.content || "Source text was not retained for this record."),
    );
    if (isAfter) {
      const original = beforeIds.indexOf(id) + 1,
        delta = original - position;
      card.append(
        el(
          "span",
          original
            ? delta > 0
              ? `↑ Up ${delta} · originally #${original}`
              : delta < 0
                ? `↓ Down ${-delta} · originally #${original}`
                : "→ Same position"
            : "Original rank unavailable",
          "rank-movement",
        ),
      );
      const score = row._retrieval?.reranker_score ?? trace.reranker_score;
      if (A.finite(score))
        card.append(
          el("span", ` · reranker score ${A.fmt(score, 5)}`, "rank-score"),
        );
    }
    return card;
  }
  function showMetrics(target, c, pool, k) {
    const gold =
      pool.gold ??
      Object.fromEntries((pool.relevant_source_ids || []).map((id) => [id, 1]));
    const relevant = Object.keys(gold).filter((id) => gold[id] > 0),
      chosen = c.retrieved_source_ids || [];
    const hits = chosen.slice(0, k).filter((id) => gold[id] > 0).length;
    const candidateHits = (
      c.candidate_source_ids || pool.candidates.map((r) => r.source_id)
    ).filter((id) => gold[id] > 0).length;
    const first = chosen.slice(0, k).findIndex((id) => gold[id] > 0);
    target.append(
      el(
        "p",
        pool.labels_available
          ? `This question: ${hits} relevant records in ${k} requested slots → Precision@${k} = ${hits}/${k}. Retrieved ${hits} of ${relevant.length} labeled relevant records → Recall@${k} = ${relevant.length ? hits + "/" + relevant.length : "undefined (no positive labels)"}. ${first >= 0 ? `First relevant result is #${first + 1} → reciprocal rank 1/${first + 1}.` : `No relevant result in top ${k} → reciprocal rank 0.`} Initial pool contains ${candidateHits}/${relevant.length} relevant records.`
          : "Relevance labels are unavailable; ranking movements can be inspected but do not establish quality.",
        "notice",
      ),
    );
    return gold;
  }
  async function render(data, run) {
    const root = document.getElementById("ranking-explorer");
    if (!root) return;
    const ticket = ++version;
    root.replaceChildren();
    root.hidden = !run?.cases?.length;
    if (root.hidden) return;
    if (experiment !== data.id) {
      experiment = data.id;
      question = null;
    }
    if (!run.cases.some((c) => c.case_id === question))
      question =
        run.cases.find((c) =>
          c.retrieved_source_ids?.some(
            (id, i) => id !== c.candidate_source_ids?.[i],
          ),
        )?.case_id || run.cases[0].case_id;
    root.append(
      el("h3", "Evidence before → after"),
      el(
        "p",
        `Inspect ${A.runName(run, data)}. Green marks the dataset’s relevant memories; arrows show movement from the original pool.`,
        "muted",
      ),
    );
    const label = el("label", "Question"),
      select = el("select");
    select.id = "ranking-question";
    run.cases.forEach((c) => {
      const o = el("option", c.question);
      o.value = c.case_id;
      select.append(o);
    });
    select.value = question;
    select.onchange = () => {
      question = select.value;
      render(data, run);
    };
    label.append(select);
    root.append(label);
    const body = el("div");
    root.append(body);
    body.append(el("p", "Loading saved evidence…", "muted"));
    try {
      const key = data.id + "/" + question;
      if (!cache.has(key))
        cache.set(
          key,
          fetch(
            `/evalground/comparisons/${data.id}/evidence?case_id=${encodeURIComponent(question)}`,
          )
            .then(async (r) => {
              const p = await r.json();
              if (!r.ok) throw Error(p.detail || "Saved evidence unavailable");
              return p;
            })
            .catch((e) => {
              cache.delete(key);
              throw e;
            }),
        );
      const pool = await cache.get(key);
      if (ticket !== version) return;
      body.replaceChildren();
      const c = run.cases.find((c) => c.case_id === question),
        k = data.config.top_k;
      if (!pool.candidates.length) {
        body.append(
          el(
            "p",
            "The original candidate text was not saved for this run. Download the result to inspect recorded IDs.",
            "notice",
          ),
        );
        return;
      }
      const beforeIds = c.candidate_source_ids?.length
        ? c.candidate_source_ids
        : pool.candidates.map((r) => r.source_id);
      const documents = new Map(pool.candidates.map((r) => [r.source_id, r]));
      const gold = showMetrics(body, c, pool, k),
        grid = el("div", undefined, "ranking-columns");
      const before = el("section"),
        after = el("section");
      before.append(el("h4", `Before · original top ${k}`));
      after.append(el("h4", `After · evidence sent to the answer model`));
      beforeIds
        .slice(0, k)
        .forEach((id, i) =>
          before.append(
            compareCard(
              documents.get(id) || { source_id: id },
              i + 1,
              beforeIds,
              gold,
              pool.labels_available,
              false,
            ),
          ),
        );
      (c.retrieved_source_ids || []).forEach((id, i) => {
        const recorded = (c.evidence || []).find((r) => r.source_id === id) ||
          documents.get(id) || { source_id: id };
        const trace =
          (c.retrieval_trace || []).find((t) => t.source_id === id) ||
          (c.retrieval_trace || [])[i] ||
          {};
        after.append(
          compareCard(
            recorded,
            i + 1,
            beforeIds,
            gold,
            pool.labels_available,
            true,
            trace,
          ),
        );
      });
      grid.append(before, after);
      body.append(grid);
      const recipe =
        run.reranker?.provider === "jev" &&
        run.reranker?.jev_method === "choice"
          ? "Jev Choice scores are relative to this candidate pool. They are not absolute relevance probabilities or deletion thresholds."
          : "Scores use each method’s own scale. Compare labeled ranking metrics across methods, not raw score magnitudes.";
      body.append(
        el(
          "p",
          recipe +
            " Only the returned top-k final order was saved; unselected final positions are unknown.",
          "muted",
        ),
      );
      const answer = el("details");
      answer.append(
        el(
          "summary",
          `Answer and check · ${c.correct ? "passed" : "did not pass"}`,
        ),
        el("p", c.prediction || ""),
        el(
          "p",
          `Expected ${data.accuracy_label ? "keywords" : "answer(s)"}: ${(c.answers || []).join(" / ")}`,
        ),
      );
      body.append(answer);
      const full = el("details");
      full.append(el("summary", `All ${beforeIds.length} initial candidates`));
      beforeIds.forEach((id, i) =>
        full.append(
          compareCard(
            documents.get(id) || { source_id: id },
            i + 1,
            beforeIds,
            gold,
            pool.labels_available,
            false,
          ),
        ),
      );
      body.append(full);
    } catch (e) {
      if (ticket === version)
        body.replaceChildren(el("p", e.message, "notice"));
    }
  }
  window.MemorizzComparisonEvidence = { render };
})();

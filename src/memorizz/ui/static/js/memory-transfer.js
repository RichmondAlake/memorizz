(() => {
  const exportForm = document.getElementById("memory-export-form");
  const importForm = document.getElementById("memory-import-form");
  if (!exportForm || !importForm) return;
  const status = (id, text) => {
    document.getElementById(id).textContent = text;
  };
  const error = async (response) => {
    const result = await response.json();
    if (!response.ok)
      throw new Error(result.detail || "Archive request failed");
    return result;
  };
  exportForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = exportForm.querySelector("button");
    button.disabled = true;
    status("memory-export-status", "Preparing archive…");
    try {
      const form = new FormData(exportForm);
      const types = form.getAll("memory_type");
      if (!types.length) throw new Error("Select at least one memory type.");
      const query = new URLSearchParams({
        agent_id: form.get("agent_id"),
        memory_id: form.get("memory_id"),
        memory_types: types.join(","),
      });
      for (const key of [
        "include_delegates",
        "include_history",
        "include_context",
      ])
        query.set(key, String(form.has(key)));
      const response = await fetch("/api/memory-transfer/export?" + query);
      if (!response.ok) await error(response);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download =
        response.headers
          .get("Content-Disposition")
          ?.match(/filename="([^"]+)"/)?.[1] || "memory.memorizz.json";
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      status(
        "memory-export-status",
        `Downloaded ${response.headers.get("X-Memorizz-Records")} records.`,
      );
    } catch (e) {
      status("memory-export-status", e.message);
    } finally {
      button.disabled = false;
    }
  });
  let reviewed = null;
  const apply = document.getElementById("memory-import-apply");
  const preview = document.getElementById("memory-import-preview");
  importForm.addEventListener("change", () => {
    reviewed = null;
    apply.disabled = true;
    preview.hidden = true;
  });
  const render = (report) => {
    preview.replaceChildren();
    const heading = document.createElement("h3");
    heading.textContent = report.dry_run ? "Planned restore" : "Restore result";
    const totals = document.createElement("p");
    totals.textContent = `${report.planned} planned · ${report.imported} imported · ${report.skipped} skipped · ${report.conflicts.length} existing`;
    const table = document.createElement("table");
    table.setAttribute("aria-label", "Records by memory type");
    for (const [type, count] of Object.entries(report.counts)) {
      if (!count) continue;
      const row = table.insertRow();
      row.insertCell().textContent = type.replaceAll("_", " ");
      row.insertCell().textContent = String(count);
    }
    const notes = document.createElement("ul");
    for (const note of report.warnings) {
      const item = document.createElement("li");
      item.textContent = note;
      notes.append(item);
    }
    for (const failure of report.errors) {
      const item = document.createElement("li");
      item.textContent = `Restore stopped at ${failure.memory_type} / ${failure.id} (${failure.error_type}). Earlier writes may have committed.`;
      notes.append(item);
    }
    preview.append(heading, totals, table, notes);
    preview.hidden = false;
  };
  const send = async (payload) =>
    error(
      await fetch("/api/memory-transfer/import", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
    );
  importForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    reviewed = null;
    apply.disabled = true;
    const button = importForm.querySelector("button[type=submit]");
    button.disabled = true;
    status(
      "memory-import-status",
      "Validating archive and checking destination…",
    );
    try {
      const file = document.getElementById("memory-archive-file").files[0];
      if (!file || file.size > 50 * 1024 * 1024)
        throw new Error("Choose a MemoRizz JSON archive smaller than 50 MiB.");
      const form = new FormData(importForm);
      const payload = {
        archive: JSON.parse(await file.text()),
        conflict: form.get("conflict"),
        id_strategy: form.get("id_strategy"),
        dry_run: true,
      };
      const report = await send(payload);
      render(report);
      if (report.ok && report.planned > 0) {
        reviewed = payload;
        apply.disabled = false;
        status(
          "memory-import-status",
          "Preview ready. Review the records, then import.",
        );
      } else
        status(
          "memory-import-status",
          report.ok
            ? "No new records to import."
            : "Existing records prevent this restore. Choose another conflict policy or new IDs.",
        );
    } catch (e) {
      status("memory-import-status", e.message);
    } finally {
      button.disabled = false;
    }
  });
  apply.addEventListener("click", async () => {
    if (!reviewed) return;
    apply.disabled = true;
    status("memory-import-status", "Restoring reviewed archive…");
    try {
      const report = await send({ ...reviewed, dry_run: false });
      render(report);
      status(
        "memory-import-status",
        report.ok
          ? `Imported ${report.imported} records. Open Memory or Agents to inspect them.`
          : "Restore stopped. Review the committed count and error below.",
      );
      reviewed = null;
    } catch (e) {
      status("memory-import-status", e.message);
    }
  });
})();

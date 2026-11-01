/* Settings → Connected apps: make a pairing code and disconnect an app. The
   server renders the list; this sends the two changes and updates the page.
   The code is copied by the shared [data-copy] handler in app.js. */
(() => {
  const section = document.getElementById("connected-apps");
  if (!section) return;
  const errorLine = document.getElementById("apps-error");
  const showError = (message) => {
    if (!errorLine) return;
    errorLine.textContent = message || "";
    errorLine.hidden = !message;
  };
  const send = async (url, body) => {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(body || {}),
    });
    const result = await response.json().catch(() => ({}));
    if (!response.ok)
      throw new Error(
        typeof result.detail === "string"
          ? result.detail
          : `The request failed (HTTP ${response.status}).`,
      );
    return result;
  };

  const pair = document.getElementById("apps-pair");
  const box = document.getElementById("apps-code");
  const value = document.getElementById("apps-code-value");
  const copy = document.getElementById("apps-copy");
  const note = document.getElementById("apps-code-note");
  let expiry = null;
  const showCode = (result) => {
    const until = new Date(result.expires_at * 1000).toLocaleTimeString([], {
      hour: "numeric",
      minute: "2-digit",
    });
    const who = document.createElement("strong");
    who.textContent = result.shared ? "your local agents" : result.principal;
    value.textContent = result.code;
    copy.dataset.copy = result.code;
    copy.hidden = false;
    note.replaceChildren(
      "Connects one app to the memory of ",
      who,
      `${result.read_only ? " (read only)" : ""}. It works once, until ${until}.`,
    );
    box.classList.remove("is-expired");
    box.hidden = false;
    clearTimeout(expiry);
    expiry = setTimeout(
      () => {
        box.classList.add("is-expired");
        copy.hidden = true;
        note.textContent = "This code has expired. Make another.";
      },
      Math.max(0, result.expires_at * 1000 - Date.now()),
    );
  };
  if (pair)
    pair.addEventListener("click", async () => {
      pair.disabled = true;
      showError("");
      try {
        const readOnly = document.getElementById("apps-read-only").checked;
        const shared = document.getElementById("apps-shared").checked;
        showCode(
          await send("/api/connected-apps/pairing-code", {
            read_only: readOnly,
            shared,
          }),
        );
        pair.textContent = "Make another code";
      } catch (error) {
        showError(error.message);
      } finally {
        pair.disabled = false;
      }
    });

  const updateCount = () => {
    const count = section.querySelectorAll("tbody tr[data-grant-id]").length;
    document.getElementById("apps-count").textContent = `${count} connected`;
    const badge = document.querySelector(
      '[data-section-link="connected-apps"] .settings-count',
    );
    if (badge) {
      badge.textContent = String(count);
      badge.title = `${count} connected`;
      badge.classList.toggle("is-set", count > 0);
    }
    document.getElementById("apps-list").hidden = count === 0;
    document.getElementById("apps-empty").hidden = count > 0;
  };
  section.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-disconnect]");
    if (!button) return;
    const name = button.dataset.appName || "this app";
    const sure = await MemorizzDialogs.confirm(`Disconnect ${name}?`, {
      detail:
        "It stops using your memory now. To bring it back, approve it again with a new pairing code.",
      confirmLabel: "Disconnect",
      cancelLabel: "Keep",
    });
    if (!sure) return;
    button.disabled = true;
    try {
      await send(
        `/api/connected-apps/${encodeURIComponent(button.dataset.disconnect)}/disconnect`,
      );
      button.closest("tr").remove();
      updateCount();
      MemorizzDialogs.notify(`Disconnected ${name}.`, { tone: "info" });
    } catch (error) {
      button.disabled = false;
      MemorizzDialogs.notify(error.message);
    }
  });
})();

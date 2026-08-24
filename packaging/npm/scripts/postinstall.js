#!/usr/bin/env node
// Copyright (c) 2024 Richmond Alake. Licensed under PolyForm Noncommercial 1.0.0.
//
// Convenience bootstrapper: `npm i -g memorizz` installs the real Python CLI via
// `uv` (a standalone binary that manages its own Python — no pre-existing Python
// required). This is NOT the canonical install path; `uv tool install
// --python 3.12 memorizz` and `pipx install memorizz` are first-class. Fails
// soft: a bootstrap error
// never bricks the package — bin/memorizz.js retries via `uv tool run` on first
// use.
"use strict";

const { spawnSync } = require("child_process");
const os = require("os");
const path = require("path");
const fs = require("fs");

const VERSION = require("../package.json").version;
const PYTHON_VERSION = "3.12";
// The npm bridge represents the complete command-line product, including MCP
// client/server commands whose Python dependencies are optional for library-only
// installs.
const PACKAGE_SPEC = `memorizz[mcp]==${VERSION}`;

const log = (m) => process.stdout.write(`[memorizz] ${m}\n`);
const warn = (m) => process.stderr.write(`[memorizz] ${m}\n`);

function which(cmd) {
  const finder = process.platform === "win32" ? "where" : "which";
  const r = spawnSync(finder, [cmd], { encoding: "utf8" });
  if (r.status === 0 && r.stdout) return r.stdout.split(/\r?\n/)[0].trim();
  return null;
}

function uvBinDirs() {
  const home = os.homedir();
  return [
    path.join(home, ".local", "bin"),
    path.join(home, ".cargo", "bin"),
    path.join(home, ".local", "share", "uv", "bin"),
  ];
}

function findUv() {
  const onPath = which("uv");
  if (onPath) return onPath;
  const exe = process.platform === "win32" ? "uv.exe" : "uv";
  for (const dir of uvBinDirs()) {
    const candidate = path.join(dir, exe);
    if (fs.existsSync(candidate)) return candidate;
  }
  return null;
}

function installUv() {
  log("Installing uv (standalone, no Python required)…");
  if (process.platform === "win32") {
    spawnSync(
      "powershell",
      ["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "irm https://astral.sh/uv/install.ps1 | iex"],
      { stdio: "inherit" }
    );
  } else {
    spawnSync("sh", ["-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"], { stdio: "inherit" });
  }
  return findUv();
}

function main() {
  if (process.env.MEMORIZZ_SKIP_POSTINSTALL) {
    log("postinstall skipped (MEMORIZZ_SKIP_POSTINSTALL set)");
    return;
  }

  let uv = findUv() || installUv();
  if (uv) {
    log(`Installing ${PACKAGE_SPEC} via uv…`);
    const r = spawnSync(
      uv,
      ["tool", "install", "--python", PYTHON_VERSION, "--force", PACKAGE_SPEC],
      { stdio: "inherit" }
    );
    if (r.status === 0) {
      log("Done. Run:  memorizz");
      return;
    }
    warn("uv install failed — trying pipx fallback.");
  }

  const pipx = which("pipx");
  if (pipx) {
    const r = spawnSync(pipx, ["install", "--force", PACKAGE_SPEC], { stdio: "inherit" });
    if (r.status === 0) {
      log("Done (via pipx). Run:  memorizz");
      return;
    }
  }

  warn(
    [
      "Could not bootstrap memorizz automatically.",
      "Install it manually with either:",
      "  curl -LsSf https://astral.sh/uv/install.sh | sh",
      `  uv tool install --python ${PYTHON_VERSION} '${PACKAGE_SPEC}'`,
      `  pipx install '${PACKAGE_SPEC}'`,
      "The `memorizz` launcher will also retry via `uv tool run` on first use.",
    ].join("\n")
  );
  // Intentionally exit 0 (fail-soft).
}

main();

#!/usr/bin/env node
// Copyright (c) 2024 Richmond Alake. Licensed under PolyForm Noncommercial 1.0.0.
//
// Thin launcher: exec the uv-installed `memorizz`, forwarding argv/stdio/exit
// code. Falls back to `uv tool run memorizz` when the launcher isn't yet on PATH
// (e.g. a fresh shell after install, or a postinstall that failed soft).
"use strict";

const { spawnSync } = require("child_process");
const os = require("os");
const path = require("path");
const fs = require("fs");

function findExe(name) {
  const finder = process.platform === "win32" ? "where" : "which";
  const onPath = spawnSync(finder, [name], { encoding: "utf8" });
  if (onPath.status === 0 && onPath.stdout.trim()) {
    return onPath.stdout.split(/\r?\n/)[0].trim();
  }
  const home = os.homedir();
  const exe = process.platform === "win32" ? `${name}.exe` : name;
  for (const dir of [path.join(home, ".local", "bin"), path.join(home, ".cargo", "bin")]) {
    const candidate = path.join(dir, exe);
    if (fs.existsSync(candidate)) return candidate;
  }
  return null;
}

const args = process.argv.slice(2);
const exe = findExe("memorizz");

let res;
if (exe) {
  res = spawnSync(exe, args, { stdio: "inherit" });
} else {
  const uv = findExe("uv");
  if (!uv) {
    process.stderr.write(
      "[memorizz] CLI not installed and uv not found.\n" +
        "  Try: npm rebuild -g memorizz   (re-runs the bootstrapper)\n" +
        "  Or:  uv tool install memorizz  /  pipx install memorizz\n"
    );
    process.exit(127);
  }
  res = spawnSync(uv, ["tool", "run", "memorizz", ...args], { stdio: "inherit" });
}

process.exit(res.status == null ? 1 : res.status);

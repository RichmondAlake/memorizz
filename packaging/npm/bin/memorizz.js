#!/usr/bin/env node
// Copyright (c) 2024 Richmond Alake. Licensed under PolyForm Noncommercial 1.0.0.
//
// Thin launcher: exec the exact uv-installed `memorizz`, forwarding
// argv/stdio/exit code. Falls back to `uv tool run` with the pinned package
// specification when the managed binary is missing or stale.
"use strict";

const { spawnSync } = require("child_process");
const os = require("os");
const path = require("path");
const fs = require("fs");
const VERSION = require("../package.json").version;
const PYTHON_VERSION = "3.12";
const PACKAGE_SPEC = `memorizz[mcp]==${VERSION}`;
const EXPECTED_VERSION = `memorizz ${VERSION}`;

function realpath(value) {
  try {
    return fs.realpathSync(value);
  } catch (_) {
    return path.resolve(value);
  }
}

const LAUNCHER_PATH = realpath(__filename);

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

function uvToolBinDir(uv) {
  const result = spawnSync(uv, ["tool", "dir", "--bin"], {
    encoding: "utf8",
  });
  if (result.status !== 0 || !result.stdout.trim()) return null;
  return result.stdout.trim().split(/\r?\n/).pop();
}

function isExpectedCli(candidate) {
  if (!candidate || !fs.existsSync(candidate)) return false;
  if (realpath(candidate) === LAUNCHER_PATH) return false;
  const result = spawnSync(candidate, ["--version"], { encoding: "utf8" });
  return result.status === 0 && result.stdout.trim() === EXPECTED_VERSION;
}

function findUvManagedCli(uv) {
  const binDir = uvToolBinDir(uv);
  if (!binDir) return null;
  const executable = process.platform === "win32" ? "memorizz.exe" : "memorizz";
  const candidate = path.join(binDir, executable);
  return isExpectedCli(candidate) ? candidate : null;
}

function findCompatibleCliOnPath() {
  const executable = process.platform === "win32" ? "memorizz.exe" : "memorizz";
  for (const directory of (process.env.PATH || "").split(path.delimiter)) {
    if (!directory) continue;
    const candidate = path.join(directory, executable);
    if (isExpectedCli(candidate)) return candidate;
  }
  return null;
}

const args = process.argv.slice(2);
const uv = findExe("uv");
const exe = (uv && findUvManagedCli(uv)) || findCompatibleCliOnPath();

let res;
if (exe) {
  res = spawnSync(exe, args, { stdio: "inherit" });
} else {
  if (!uv) {
    process.stderr.write(
      `[memorizz] MemoRizz ${VERSION} is not installed and uv was not found.\n` +
        "  Try: npm rebuild -g memorizz   (re-runs the bootstrapper)\n" +
        `  Or:  uv tool install --python ${PYTHON_VERSION} '${PACKAGE_SPEC}'\n`
    );
    process.exit(127);
  }
  res = spawnSync(
    uv,
    ["tool", "run", "--python", PYTHON_VERSION, "--from", PACKAGE_SPEC, "memorizz", ...args],
    { stdio: "inherit" }
  );
}

process.exit(res.status == null ? 1 : res.status);

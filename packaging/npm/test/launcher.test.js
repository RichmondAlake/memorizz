"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { spawnSync } = require("node:child_process");

const packageRoot = path.resolve(__dirname, "..");
const launcher = path.join(packageRoot, "bin", "memorizz.js");
const postinstall = path.join(packageRoot, "scripts", "postinstall.js");
const version = require(path.join(packageRoot, "package.json")).version;

function executable(target, source) {
  fs.writeFileSync(target, `#!${process.execPath}\n${source}\n`, { mode: 0o755 });
}

function fixture(managedVersion) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "memorizz-npm-launcher-"));
  const commands = path.join(root, "commands");
  const managed = path.join(root, "managed");
  fs.mkdirSync(commands);
  fs.mkdirSync(managed);

  executable(
    path.join(commands, "uv"),
    `const args = process.argv.slice(2);\n` +
      `if (args.join(" ") === "tool dir --bin") { console.log(${JSON.stringify(
        managed
      )}); process.exit(0); }\n` +
      `if (args[0] === "tool" && args[1] === "run") { ` +
      `console.log("fallback:" + args.join(" ")); process.exit(0); }\n` +
      `if (args[0] === "tool" && args[1] === "install") { ` +
      `console.log("install:" + args.join(" ")); process.exit(0); }\n` +
      `process.exit(2);`
  );
  executable(
    path.join(managed, "memorizz"),
    `const args = process.argv.slice(2);\n` +
      `if (args[0] === "--version") { console.log("memorizz ${managedVersion}"); ` +
      `process.exit(0); }\n` +
      `console.log("managed:" + args.join(" "));`
  );
  return {
    env: {
      ...process.env,
      PATH: [commands, "/usr/bin", "/bin"].join(path.delimiter),
    },
    root,
  };
}

function check(name, callback) {
  try {
    callback();
    process.stdout.write(`ok - ${name}\n`);
  } catch (error) {
    process.stderr.write(`not ok - ${name}\n${error.stack || error}\n`);
    process.exitCode = 1;
  }
}

if (process.platform === "win32") {
  process.stdout.write("launcher tests skipped on Windows\n");
} else {
  check("delegates to the exact uv-managed CLI", () => {
    const setup = fixture(version);
    try {
      const result = spawnSync(
        process.execPath,
        [launcher, "capabilities", "--json"],
        {
          encoding: "utf8",
          env: setup.env,
        }
      );

      assert.equal(result.status, 0, result.stderr);
      assert.equal(result.stdout.trim(), "managed:capabilities --json");
    } finally {
      fs.rmSync(setup.root, { force: true, recursive: true });
    }
  });

  check("uses the pinned uv fallback when the managed CLI is stale", () => {
    const setup = fixture("0.2.2");
    try {
      const result = spawnSync(process.execPath, [launcher, "--version"], {
        encoding: "utf8",
        env: setup.env,
      });

      assert.equal(result.status, 0, result.stderr);
      assert.equal(
        result.stdout.trim(),
        `fallback:tool run --python 3.12 --from memorizz[mcp]==${version} memorizz --version`
      );
    } finally {
      fs.rmSync(setup.root, { force: true, recursive: true });
    }
  });

  check("bootstraps the pinned CLI with a managed Python", () => {
    const setup = fixture(version);
    try {
      const result = spawnSync(process.execPath, [postinstall], {
        encoding: "utf8",
        env: setup.env,
      });

      assert.equal(result.status, 0, result.stderr);
      assert.match(
        result.stdout,
        new RegExp(
          `install:tool install --python 3\\.12 --force memorizz\\[mcp\\]==${version.replace(
            /\\./g,
            "\\."
          )}`
        )
      );
    } finally {
      fs.rmSync(setup.root, { force: true, recursive: true });
    }
  });

  check("does not recurse into its own npm launcher", () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "memorizz-npm-self-"));
    try {
      fs.symlinkSync(launcher, path.join(root, "memorizz"));
      const result = spawnSync(process.execPath, [launcher, "--version"], {
        encoding: "utf8",
        env: { ...process.env, HOME: root, PATH: root },
      });

      assert.equal(result.status, 127);
      assert.match(result.stderr, new RegExp(`MemoRizz ${version} is not installed`));
    } finally {
      fs.rmSync(root, { force: true, recursive: true });
    }
  });
}

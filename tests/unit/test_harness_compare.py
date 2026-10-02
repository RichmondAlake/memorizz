"""Compare harness runs: the facts and timeline marks harness-compare.js draws."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

STATIC = Path(__file__).parents[2] / "src" / "memorizz" / "ui" / "static"

# Loads the page scripts into a bare context (no browser) and prints what the
# pure parts of MemorizzCompare make of a few runs.
SCRIPT = r"""
const vm = require('vm');
const fs = require('fs');
const ctx = {window: {}};
vm.createContext(ctx);
for (const file of process.argv.slice(1)) vm.runInContext(fs.readFileSync(file, 'utf8'), ctx);
const C = ctx.window.MemorizzCompare;
const at = (s) => new Date(Date.UTC(2026, 8, 30, 12, 0, 0) + s * 1000).toISOString();
let seq = 0;
const ev = (s, type, data) => ({run_id: 'r', sequence: ++seq, timestamp: at(s), type, data});

// Claude Code: waits 1 s for approval, starts a subagent whose Task call
// returns at once while its search runs on under it.
const claudeEvents = [
  ev(0, 'status', {status: 'pending_approval'}),
  ev(0, 'approval', {status: 'requested'}),
  ev(1, 'status', {status: 'running'}),
  ev(2, 'tool_call', {type: 'tool_use', id: 'T1', name: 'Task', subagent: true, input: {description: 'Find ORCL'}}),
  ev(2.01, 'tool_result', {message: {content: [{type: 'tool_result', tool_use_id: 'T1', content: 'launched'}]}}),
  ev(3, 'tool_call', {type: 'tool_use', id: 'W1', name: 'WebSearch', parent_id: 'T1', input: {query: 'ORCL'}}),
  ev(8, 'tool_result', {message: {content: [{type: 'tool_result', tool_use_id: 'W1', content: 'results'}]}}),
  ev(9, 'reasoning', {text: '', hidden: true, parent_id: 'T1'}),
  ev(12, 'message', {role: 'assistant', text: 'ORCL $137'}),
  ev(13, 'complete', {status: 'succeeded'}),
];
const claude = {run_id: 'claude', harness: 'claude-code', status: 'succeeded',
  task: {task: 'Price check'},
  result: {cost_usd: 0.42, latency_ms: 46000, final_response: 'ORCL $137',
    usage: {input_tokens: 4, cache_read_input_tokens: 10000, cache_creation_input_tokens: 1000, output_tokens: 500, model: 'claude-opus-5-5'}}};
const codex = {run_id: 'codex', harness: 'codex', status: 'succeeded', task: {task: 'Price check'},
  result: {cost_usd: 0.3, latency_ms: 52000, usage: {input_tokens: 1000, cached_input_tokens: 800, output_tokens: 100, model: 'gpt-6-astra', cost_basis: 'list_rate_estimate'}}};
const memagent = {run_id: 'mem', harness: 'memagent', status: 'succeeded', task: {task: 'Price check'},
  result: {latency_ms: 71000, usage: {input_tokens: 500, output_tokens: 50}}};
const failed = {run_id: 'bad', harness: 'pi', status: 'failed', task: {task: 'Price check'},
  result: {cost_usd: 0.01, latency_ms: 1000, error: 'no model'}};

const now = Date.parse(at(60));
const c = C.summarize(claude, claudeEvents, now);
const others = [C.summarize(codex, [], now), C.summarize(memagent, [], now), C.summarize(failed, [], now)];
const bars = [{start: 0, end: 5}, {start: 1, end: 3}, {start: 5, end: 6}, {start: 3.5, end: 4}];
const rows = C.pack(bars);
console.log(JSON.stringify({
  claude: {
    waitedMs: c.waitedMs, model: c.model, counts: c.counts, hidden: c.hiddenReasoning,
    tokens: c.tokens, actions: c.actions,
    bars: c.bars.map(b => [b.kind, b.depth, +b.start.toFixed(2), +b.end.toFixed(2)]),
    ticks: c.ticks.map(t => [t.kind, t.depth, +t.at.toFixed(2)]),
  },
  codex: {tokens: others[0].tokens, estimated: others[0].costEstimated, model: others[0].model},
  marks: C.highlights([c, ...others]),
  packed: {rows, placed: bars.map(b => b.row)},
}));
"""


def _run() -> dict:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed to run the page script")
    files = [
        str(STATIC / "js" / "harness-trajectory.js"),
        str(STATIC / "js" / "harness-compare.js"),
    ]
    done = subprocess.run(
        [node, "-e", SCRIPT, *files], capture_output=True, text=True, timeout=30
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_a_run_becomes_facts_and_timeline_marks():
    out = _run()
    claude = out["claude"]
    # Lanes start when the harness started, after the approval wait.
    assert claude["waitedMs"] == 1000
    assert claude["model"] == "claude-opus-5-5"
    assert claude["counts"]["subagent"] == 1 and claude["counts"]["tool"] == 1
    assert claude["hidden"] == 1 and claude["actions"] == 2
    # The subagent spans its search and reasoning, though its Task call
    # returned at once.
    assert ["subagent", 0, 1.0, 8.0] in claude["bars"]
    assert ["tool", 1, 2.0, 7.0] in claude["bars"]
    assert ["thinking", 1, 8.0] in claude["ticks"]
    assert ["message", 0, 11.0] in claude["ticks"]
    # Anthropic's cached input counts as input, and is flagged as one call's worth.
    assert claude["tokens"]["input"] == 11004 and claude["tokens"]["partial"] is True
    codex = out["codex"]
    assert codex["tokens"] == {
        "input": 1000,
        "output": 100,
        "cached": 800,
        "partial": False,
    }
    assert codex["estimated"] is True and codex["model"] == "gpt-6-astra"


def test_highlights_compare_effort_only_among_runs_that_succeeded():
    marks = _run()["marks"]
    assert marks["claude"] == {"duration": "Fastest"}
    assert marks["codex"]["cost"] == "Cheapest"
    # Claude Code's one-call token count is left out, and the failed run is
    # never the fastest or cheapest.
    assert marks["mem"] == {"actions": "Fewest actions", "tokens": "Fewest tokens"}
    assert "bad" not in marks


def test_overlapping_steps_go_on_separate_rows():
    packed = _run()["packed"]
    assert packed["rows"] == 2
    assert packed["placed"] == [0, 1, 0, 1]


def test_the_page_scripts_exist_with_the_public_api():
    compare = (STATIC / "js" / "harness-compare.js").read_text()
    assert (
        "window.MemorizzCompare = {open, summarize, highlights, pack, tokensOf}"
        in compare
    )
    assert (STATIC / "css" / "pages" / "harness-compare.css").exists()

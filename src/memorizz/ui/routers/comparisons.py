"""Durable Evalground comparisons with owned, cancellable worker processes."""

import csv
import io
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import ValidationError

from ...benchmarks.comparison import ComparisonConfig, atomic_json, summarize_run
from ...benchmarks.memory_suite import BENCHMARK_CATALOG
from ..helpers import _list_agents
from ..state import _state, templates

router = APIRouter(tags=["evalground"])
_workers = {}
_lock = threading.Lock()


def root() -> Path:
    from .evalground import _get_longmemeval_paths

    configured = os.environ.get("MEMORIZZ_COMPARISON_HOME")
    value = (
        Path(configured).expanduser()
        if configured
        else _get_longmemeval_paths()["results_dir"].parent / "comparisons"
    )
    value.mkdir(parents=True, exist_ok=True)
    return value


def directory(run_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", run_id):
        raise HTTPException(404, "Comparison not found")
    path = root() / run_id
    if not path.is_dir():
        raise HTTPException(404, "Comparison not found")
    return path


def read_result(run_id: str) -> dict:
    path = directory(run_id)
    result = path / "result.json"
    data = (
        json.loads(result.read_text())
        if result.exists()
        else {"status": "queued", "runs": []}
    )
    with _lock:
        worker = _workers.get(run_id)
    if data["status"] in {"queued", "running"}:
        if worker is None or worker.poll() is not None:
            data["status"] = (
                "cancelled" if (path / "cancel").exists() else "interrupted"
            )
            data[
                "warning"
            ] = "Worker stopped. Completed cases remain available; in-flight billing may be unknown."
            for row in data.get("runs", []):
                if row["status"] == "running":
                    row["status"] = data["status"]
                    row["summary"] = summarize_run(
                        row, data.get("config", {}).get("local_hourly_usd", 0)
                    )
                    row["summary"].update(
                        total_cost_usd=None,
                        serving_cost_usd=None,
                        cost_per_question_usd=None,
                        cost_per_correct_usd=None,
                        billing_incomplete=True,
                    )
        elif (path / "cancel").exists():
            data["status"] = "cancelling"
    data["id"] = run_id
    return data


def stop_comparison_workers():
    with _lock:
        workers = list(_workers.values())
    for process in workers:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


@router.get("/evalground/compare", response_class=HTMLResponse)
async def page(request: Request):
    if not _state["provider"]:
        return RedirectResponse("/connect", status_code=302)
    return templates.TemplateResponse(
        "comparisons.html",
        {
            "request": request,
            "active_page": "evalground",
            "provider_type": _state.get("provider_type"),
            "benchmark_catalog": [s.to_dict() for s in BENCHMARK_CATALOG.values()],
            "appbook_url": os.environ.get(
                "MEMORIZZ_APPBOOK_URL", "http://127.0.0.1:8878"
            ),
        },
    )


@router.get("/evalground/comparisons")
async def history():
    values = []
    for path in sorted(
        root().glob("*/config.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )[:100]:
        result = read_result(path.parent.name)
        config = json.loads(path.read_text())
        values.append(
            {
                "id": result["id"],
                "name": config["name"],
                "status": result["status"],
                "runs": len(result.get("runs", [])),
                "created_at": path.stat().st_mtime,
            }
        )
    return {"experiments": values}


@router.get("/evalground/model-catalog")
def model_catalog(
    provider: str, host: str = "http://localhost:11434", refresh: bool = False
):
    from ...benchmarks.measurement import READER_PROVIDERS
    from ...benchmarks.model_catalog import discover_models
    from ...benchmarks.rerankers import RERANKERS

    if provider not in READER_PROVIDERS | RERANKERS:
        raise HTTPException(422, "Unsupported provider")
    return discover_models(provider, host=host, refresh=refresh)


@router.get("/evalground/run-library")
def run_library():
    """One read-only index for saved comparisons and agent benchmark runs."""
    from .evalground import _build_eval_run_history_rows

    if not _state.get("provider"):
        raise HTTPException(409, "Connect a memory provider first")
    rows, warnings = [], []
    for config_path in sorted(
        root().glob("*/config.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )[:200]:
        try:
            config = json.loads(config_path.read_text())
            result = read_result(config_path.parent.name)
            runs = result.get("runs", [])
            costs = [r.get("summary", {}).get("total_cost_usd") for r in runs]
            priced = bool(costs) and all(
                type(v) in (float, int) and math.isfinite(v) for v in costs
            )
            kind = config.get("experiment_type", "pipeline")
            key = "ndcg_at_k" if kind == "reranker" else "accuracy"
            quality = [r.get("summary", {}).get(key) for r in runs]
            quality = [
                v for v in quality if type(v) in (float, int) and math.isfinite(v)
            ]
            created = result.get("started_at") or config_path.stat().st_mtime
            rows.append(
                {
                    "id": result["id"],
                    "name": config.get("name", "Model comparison"),
                    "kind": kind,
                    "status": result["status"],
                    "created_at": created,
                    "models": list(
                        dict.fromkeys(
                            s.get("label") or s.get("model") or s.get("provider", "")
                            for s in config.get("readers", [])
                            + config.get("rerankers", [])
                            if s.get("provider") != "none"
                        )
                    ),
                    "completed": sum(len(r.get("cases", [])) for r in runs),
                    "planned": len(config.get("readers", []))
                    * len(config.get("rerankers", []))
                    * config.get("repeats", 1)
                    * len(result.get("case_ids") or range(config.get("limit", 0))),
                    "progress_unit": "answers",
                    "cost_usd": sum(costs) if priced else None,
                    "quality_label": f"nDCG@{config.get('top_k', '?')}"
                    if kind == "reranker"
                    else result.get("accuracy_label", "Answer accuracy"),
                    "quality_min": min(quality) if quality else None,
                    "quality_max": max(quality) if quality else None,
                    "url": f"/evalground/compare?experiment={result['id']}",
                    "scope": result.get("dataset_label", config.get("dataset", "")),
                }
            )
        except (OSError, ValueError, TypeError, KeyError, HTTPException):
            warnings.append(
                "A saved comparison could not be read; its files were left unchanged."
            )
    try:
        agents = _list_agents()
    except Exception:
        agents = []
    for row in _build_eval_run_history_rows(agents=agents, limit=200):
        try:
            stamp = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
            created = stamp.replace(tzinfo=stamp.tzinfo or timezone.utc).timestamp()
        except (ValueError, TypeError, AttributeError):
            created = 0
        accuracy = row.get("overall_accuracy")
        quality = accuracy / 100 if accuracy is not None else None
        rows.append(
            {
                "id": row["run_id"],
                "name": f"{row.get('agent_name', 'Agent')} · {row.get('benchmark', 'evaluation')}",
                "kind": "agent",
                "status": row["status"],
                "created_at": created,
                "models": [
                    row.get("model") or row.get("agent_id") or "Standalone benchmark"
                ],
                "completed": row.get("evaluated_samples"),
                "planned": row.get("num_samples"),
                "progress_unit": "questions",
                "cost_usd": row.get("cost_usd"),
                "quality_label": "Benchmark accuracy",
                "quality_min": quality,
                "quality_max": quality,
                "url": f"/evalground?run_id={row['run_id']}",
                "scope": row.get("dataset_variant", ""),
            }
        )
    return {
        "runs": sorted(rows, key=lambda r: r["created_at"], reverse=True),
        "warnings": list(dict.fromkeys(warnings)),
        "scope": "Saved model comparisons on this machine and agent evaluations from this server session. Quality values use different scorers; compare within an experiment. API estimates total all configurations and evaluation overhead; local hardware is excluded.",
    }


@router.get("/evalground/system-one-preset")
def system_one_preset(kind: str = "reranking"):
    configured = os.environ.get("MEMORIZZ_SYSTEM_ONE_HOME")
    if not configured:
        raise HTTPException(
            409, "Set MEMORIZZ_SYSTEM_ONE_HOME to the course folder on the server"
        )
    path = (
        Path(configured).expanduser()
        / "artifacts"
        / "memorizz"
        / "reranking_config.json"
    )
    if not path.is_file():
        raise HTTPException(409, "Run notebook 01 and export_memorizz.py first")
    config = json.loads(path.read_text())
    if kind == "readers":
        config.update(
            name="Memory readers · Opus 5.5 / GPT-6 Sol / GPT-6 Luna",
            experiment_type="reader",
            rerankers=[{"provider": "none"}],
            max_cost_usd=5,
            top_k=5,
            readers=[
                {
                    "provider": "anthropic",
                    "model": "claude-opus-5-5",
                    "options": {
                        "max_tokens": 4096,
                        "temperature": None,
                        "effort": "medium",
                        "enable_prompt_caching": False,
                    },
                },
                *[
                    {
                        "provider": "openai",
                        "model": name,
                        "options": {
                            "max_completion_tokens": 4096,
                            "reasoning_effort": "medium",
                            "temperature": None,
                        },
                    }
                    for name in ["gpt-6-sol", "gpt-6-luna"]
                ],
            ],
        )
    elif kind != "reranking":
        raise HTTPException(422, "Choose reranking or readers")
    return ComparisonConfig.model_validate(config).model_dump()


@router.post("/evalground/comparisons")
async def start(request: Request):
    if not _state["provider"]:
        raise HTTPException(409, "Connect a memory provider first")
    if _state.get("read_only"):
        raise HTTPException(403, "Comparisons cannot be started in read-only mode")
    raw = await request.body()
    if len(raw) > 100_000:
        raise HTTPException(413, "Configuration is too large")
    try:
        config = ComparisonConfig.model_validate_json(raw)
    except ValidationError as exc:
        messages = [str(error["msg"]) for error in exc.errors(include_input=False)]
        raise HTTPException(422, "; ".join(messages)) from exc
    with _lock:
        if any(p.poll() is None for p in _workers.values()):
            raise HTTPException(
                409, "A comparison is already running. Stop it or wait for completion."
            )
        run_id = uuid.uuid4().hex
        path = root() / run_id
        atomic_json(path / "config.json", config.model_dump())
        env = dict(os.environ)
        source_root = str(Path(__file__).resolve().parents[3])
        env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
        with (path / "worker.log").open("w") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "memorizz.benchmarks.comparison",
                    str(path / "config.json"),
                ],
                env=env,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        _workers[run_id] = process
    return JSONResponse({"id": run_id, "status": "queued"}, status_code=202)


@router.get("/evalground/comparisons/{run_id}")
async def detail(run_id: str):
    return read_result(run_id)


@router.get("/evalground/comparisons/{run_id}/evidence")
async def evidence(run_id: str, case_id: str):
    """Inspect the saved candidate pool without fetching or reranking memories."""
    path = directory(run_id)
    cases_path = path / "cases.json"
    cases = json.loads(cases_path.read_text()) if cases_path.exists() else []
    case = next((c for c in cases if c.get("case_id") == case_id), None)
    if case is None:
        raise HTTPException(404, "Saved question not found")
    candidates = case.get("candidates")
    if candidates is None:
        saved = path / "candidates.json"
        pools = json.loads(saved.read_text()) if saved.exists() else {}
        candidates = pools.get(case_id, {}).get("rows", [])
    gold = case.get("gold")
    return {
        "case_id": case_id,
        "candidates": candidates,
        "gold": gold,
        "relevant_source_ids": (
            [source for source, grade in gold.items() if grade > 0]
            if gold is not None
            else case.get("relevant_source_ids", [])
        ),
        "labels_available": gold is not None or bool(case.get("relevant_source_ids")),
        "scope": "Saved initial pool and recorded top-k output; unselected final ranks were not retained.",
    }


@router.post("/evalground/comparisons/{run_id}/stop")
async def stop(run_id: str):
    if _state.get("read_only"):
        raise HTTPException(403, "Read-only mode")
    path = directory(run_id)
    with _lock:
        process = _workers.get(run_id)
    if process and process.poll() is None:
        (path / "cancel").touch()
        # The worker's most recent completed cases are durable. Terminate the
        # whole process group so a blocked provider cannot hang cancellation.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    return {"id": run_id, "status": read_result(run_id)["status"]}


@router.get("/evalground/comparisons/{run_id}/export")
async def export(run_id: str, format: str = "json"):
    result = read_result(run_id)
    if format == "json":
        return JSONResponse(
            result,
            headers={
                "Content-Disposition": f'attachment; filename="comparison-{run_id}.json"'
            },
        )
    if format != "csv":
        raise HTTPException(400, "Choose json or csv")
    rows = [
        {
            "run": r["id"],
            "reader": r["reader"]["provider"] + ":" + r["reader"]["model"],
            "reranker": r["reranker"]["provider"]
            + ":"
            + r["reranker"].get("model", ""),
            "status": r["status"],
            **{
                k: v
                for k, v in r.get("summary", {}).items()
                if not isinstance(v, (dict, list))
            },
        }
        for r in result.get("runs", [])
    ]
    buffer = io.StringIO()
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["run", "status"]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        # Prevent spreadsheet formulas in user-entered labels/model names.
        writer.writerow(
            {
                k: "'" + v
                if isinstance(v, str) and v.startswith(("=", "+", "-", "@"))
                else v
                for k, v in row.items()
            }
        )
    return Response(
        buffer.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="comparison-{run_id}.csv"'
        },
    )

# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Evalground benchmark runner pages and APIs.

Extracted verbatim from ``ui/app.py``: GET /evalground, POST /evalground/runs,
GET /evalground/runs/active, GET /evalground/runs/{run_id},
POST /evalground/runs/{run_id}/stop, and POST /evalground/datasets, plus the
in-process run registry (run dicts, log capture, subprocess bookkeeping) and
the LongMemEval path helpers. Route paths, response classes, and behavior are
unchanged.

The run registry is process-global module state used only by these routes;
``ui/app.py``'s lifespan calls :func:`stop_active_eval_run_processes` on
shutdown to terminate any benchmark subprocesses that are still running.
"""

import importlib.util
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from ..helpers import _extract_agent_persona_name
from ..state import _state, templates

logger = logging.getLogger(__name__)

router = APIRouter(tags=["evalground"])

_eval_runs_lock = threading.Lock()
_eval_runs: Dict[str, Dict[str, Any]] = {}
_eval_run_processes: Dict[str, subprocess.Popen] = {}
_EVAL_RUN_MAX_LOG_LINES = 2000


# -----------------------------------------------------------------------------
# Run registry helpers
# -----------------------------------------------------------------------------


def _utcnow_iso() -> str:
    """Return an ISO timestamp in UTC."""
    return datetime.utcnow().isoformat() + "Z"


def _create_eval_run(payload: Dict[str, Any]) -> str:
    """Create a new Evalground run record and return its run_id."""
    run_id = uuid.uuid4().hex
    run = {
        "run_id": run_id,
        "status": "queued",
        "created_at": _utcnow_iso(),
        "started_at": None,
        "finished_at": None,
        "benchmark": payload.get("benchmark", "longmemeval"),
        "dataset_variant": payload.get("dataset_variant", "oracle"),
        "num_samples": payload.get("num_samples", 10),
        "agent_id": payload.get("agent_id", ""),
        "logs": [],
        "error": None,
        "eval_results": None,
        "eval_output_path": None,
        "cancel_requested": False,
    }
    with _eval_runs_lock:
        _eval_runs[run_id] = run
    return run_id


def _update_eval_run(run_id: str, **updates: Any) -> None:
    """Update fields on a stored run."""
    with _eval_runs_lock:
        run = _eval_runs.get(run_id)
        if not run:
            return
        run.update(updates)


def _append_eval_run_log(run_id: str, line: str) -> None:
    """Append a line of output to a run."""
    if not line:
        return
    with _eval_runs_lock:
        run = _eval_runs.get(run_id)
        if not run:
            return
        logs = run.setdefault("logs", [])
        logs.append(line)
        if len(logs) > _EVAL_RUN_MAX_LOG_LINES:
            run["logs"] = logs[-_EVAL_RUN_MAX_LOG_LINES:]


def _get_eval_run_snapshot(run_id: str) -> Optional[Dict[str, Any]]:
    """Return a copy of a run for read-only use."""
    with _eval_runs_lock:
        run = _eval_runs.get(run_id)
        if not run:
            return None
        snapshot = dict(run)
        snapshot["logs"] = list(run.get("logs", []))
        return snapshot


def _list_eval_runs(limit: int = 100) -> List[Dict[str, Any]]:
    """Return recent Evalground runs in reverse chronological order."""
    with _eval_runs_lock:
        rows = []
        for run in _eval_runs.values():
            snapshot = dict(run)
            snapshot["logs"] = list(run.get("logs", []))
            rows.append(snapshot)

    rows.sort(key=lambda row: row.get("created_at") or "", reverse=True)
    if limit > 0:
        rows = rows[:limit]
    return rows


def _build_eval_run_history_rows(
    agents: Optional[List[Any]] = None, limit: int = 100
) -> List[Dict[str, Any]]:
    """Build table rows for Evalground run history."""
    agents_by_id: Dict[str, Any] = {}
    for agent in agents or []:
        agent_id = getattr(agent, "agent_id", None)
        if agent_id:
            agents_by_id[agent_id] = agent

    history_rows: List[Dict[str, Any]] = []
    for run in _list_eval_runs(limit=limit):
        agent_id = str(run.get("agent_id") or "")
        agent_name = "Unknown"
        matched_agent = agents_by_id.get(agent_id)
        if matched_agent:
            agent_name = _extract_agent_persona_name(matched_agent)

        overall_accuracy = None
        eval_results_payload = run.get("eval_results")
        if isinstance(eval_results_payload, dict):
            value = eval_results_payload.get("overall_accuracy")
            if isinstance(value, (int, float)):
                overall_accuracy = round(float(value) * 100, 2)

        history_rows.append(
            {
                "run_id": run.get("run_id"),
                "created_at": run.get("created_at"),
                "started_at": run.get("started_at"),
                "finished_at": run.get("finished_at"),
                "status": run.get("status"),
                "benchmark": run.get("benchmark"),
                "dataset_variant": run.get("dataset_variant"),
                "num_samples": run.get("num_samples"),
                "agent_id": agent_id,
                "agent_name": agent_name,
                "overall_accuracy": overall_accuracy,
            }
        )

    return history_rows


def _get_latest_active_eval_run() -> Optional[Dict[str, Any]]:
    """Return the most recent run still in a non-terminal state."""
    active_statuses = {"queued", "running", "canceling"}
    for run in _list_eval_runs(limit=200):
        run_id = run.get("run_id")
        status = str(run.get("status") or "")
        if not run_id or status not in active_statuses:
            continue
        return {
            "run_id": run_id,
            "status": status,
            "created_at": run.get("created_at"),
            "started_at": run.get("started_at"),
            "agent_id": run.get("agent_id"),
            "benchmark": run.get("benchmark"),
            "dataset_variant": run.get("dataset_variant"),
            "num_samples": run.get("num_samples"),
        }
    return None


def _set_eval_run_process(run_id: str, process: subprocess.Popen) -> None:
    """Associate a running subprocess with an Evalground run."""
    with _eval_runs_lock:
        _eval_run_processes[run_id] = process


def _get_eval_run_process(run_id: str) -> Optional[subprocess.Popen]:
    """Get the process currently associated with a run."""
    with _eval_runs_lock:
        return _eval_run_processes.get(run_id)


def _pop_eval_run_process(run_id: str) -> Optional[subprocess.Popen]:
    """Remove and return the subprocess associated with a run."""
    with _eval_runs_lock:
        return _eval_run_processes.pop(run_id, None)


def _get_eval_run_delta(run_id: str, after: int = 0) -> Optional[Dict[str, Any]]:
    """Return incremental logs and status for polling."""
    with _eval_runs_lock:
        run = _eval_runs.get(run_id)
        if not run:
            return None

        logs = run.get("logs", [])
        safe_after = max(0, min(after, len(logs)))
        new_logs = logs[safe_after:]
        next_index = safe_after + len(new_logs)

        return {
            "run_id": run_id,
            "status": run.get("status"),
            "logs": new_logs,
            "next_index": next_index,
            "error": run.get("error"),
            "finished_at": run.get("finished_at"),
            "cancel_requested": bool(run.get("cancel_requested")),
        }


def stop_active_eval_run_processes() -> None:
    """Stop any active Evalground benchmark subprocesses (app shutdown)."""
    with _eval_runs_lock:
        active_processes = list(_eval_run_processes.items())
    for run_id, process in active_processes:
        if process.poll() is not None:
            continue
        try:
            os.killpg(process.pid, signal.SIGTERM)
            logger.info(
                "Stopped Evalground benchmark subprocess during shutdown (run_id=%s, pid=%s)",
                run_id,
                process.pid,
            )
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------


@router.get("/evalground", response_class=HTMLResponse)
async def evalground(request: Request):
    """Show the Evalground benchmark runner."""
    if not _state["provider"]:
        return RedirectResponse(url="/connect", status_code=302)

    agents = []
    try:
        agents = _state["provider"].list_memagents()
    except Exception as e:
        logger.error(f"Failed to list agents for Evalground: {e}")

    paths = _get_longmemeval_paths()
    dataset_files = _longmemeval_dataset_files(paths["dataset_dir"])
    evaluator_script = paths["evaluator_script"]
    dataset_status, missing_variants = _build_dataset_status(dataset_files)

    warnings = []
    if _state["provider_type"] != "oracle":
        warnings.append("Evalground currently supports the Oracle memory provider.")
    if not os.environ.get("OPENAI_API_KEY"):
        warnings.append("OPENAI_API_KEY is required to score LongMemEval responses.")
    if missing_variants:
        warnings.append(
            "LongMemEval datasets are missing. Use 'Download datasets' below."
        )
    if not evaluator_script.exists():
        warnings.append(
            "LongMemEval evaluator script is missing from this installation."
        )

    can_run = bool(agents)
    default_variant = "oracle"
    if _state["provider_type"] != "oracle":
        can_run = False
    if not os.environ.get("OPENAI_API_KEY"):
        can_run = False
    if not dataset_files[default_variant].exists():
        can_run = False
    if not evaluator_script.exists():
        can_run = False

    run_id = (request.query_params.get("run_id") or "").strip()

    selected_agent = None
    selected_agent_id = ""
    selected_run_status = None
    benchmark = "longmemeval"
    dataset_variant = default_variant
    num_samples = 10
    eval_results = None
    eval_output_path = None
    run_output = None
    error = None

    if run_id:
        run_state = _get_eval_run_snapshot(run_id)
        if run_state:
            selected_agent_id = run_state.get("agent_id") or ""
            benchmark = run_state.get("benchmark") or benchmark
            dataset_variant = run_state.get("dataset_variant") or dataset_variant
            num_samples = run_state.get("num_samples") or num_samples

            if selected_agent_id:
                try:
                    selected_agent = _state["provider"].retrieve_memagent(
                        selected_agent_id
                    )
                except Exception as exc:
                    logger.warning(
                        f"Failed to retrieve run agent {selected_agent_id}: {exc}"
                    )

            logs = run_state.get("logs") or []
            if logs:
                run_output = "\n".join(logs)

            status = run_state.get("status")
            selected_run_status = status
            if status == "completed":
                eval_results = run_state.get("eval_results")
                eval_output_path = run_state.get("eval_output_path")
                if not eval_results and eval_output_path:
                    result_file = paths["repo_root"] / eval_output_path
                    if result_file.exists():
                        try:
                            with open(result_file, "r", encoding="utf-8") as handle:
                                eval_results = json.load(handle)
                        except Exception as exc:
                            logger.warning(
                                "Failed to load eval output file %s: %s",
                                result_file,
                                exc,
                            )
            elif status == "failed":
                error = run_state.get("error") or "Evalground evaluation failed."
            elif status in {"canceling", "canceled"}:
                warnings.append(
                    "The selected Evalground run was canceled before completion."
                )
            elif status in {"queued", "running"}:
                warnings.append("The selected Evalground run is still in progress.")
        else:
            error = f"Evalground run '{run_id}' was not found."

    runs_history = _build_eval_run_history_rows(agents=agents, limit=200)

    return templates.TemplateResponse(
        "evalground.html",
        {
            "request": request,
            "provider_type": _state["provider_type"],
            "connection_info": _state["connection_info"],
            "agents": agents,
            "selected_agent": selected_agent,
            "selected_agent_id": selected_agent_id,
            "benchmark": benchmark,
            "dataset_variant": dataset_variant,
            "num_samples": num_samples,
            "eval_results": eval_results,
            "eval_output_path": eval_output_path,
            "run_output": run_output,
            "error": error,
            "warnings": warnings,
            "dataset_status": dataset_status,
            "missing_variants": missing_variants,
            "download_message": None,
            "download_error": None,
            "can_run": can_run,
            "eval_results_dir": str(paths["results_dir"]),
            "runs_history": runs_history,
            "selected_run_id": run_id,
            "selected_run_status": selected_run_status,
            "active_page": "evalground",
        },
    )


def _run_evalground_job(
    run_id: str,
    agent_id: str,
    dataset_variant: str,
    samples_value: int,
    paths: Dict[str, Any],
) -> None:
    """Execute an Evalground run in a background thread via subprocess."""
    current_run = _get_eval_run_snapshot(run_id) or {}
    if current_run.get("cancel_requested"):
        _update_eval_run(
            run_id,
            status="canceled",
            finished_at=_utcnow_iso(),
            error="Evaluation canceled before start.",
        )
        _append_eval_run_log(run_id, "Evaluation canceled before worker startup.")
        return

    _update_eval_run(
        run_id,
        status="running",
        started_at=_utcnow_iso(),
        error=None,
    )
    output_filename = f"evalground_{run_id}.json"
    output_path = paths["results_dir"] / output_filename
    eval_script = paths["evaluator_script"]

    _append_eval_run_log(
        run_id,
        (
            "Starting Evalground run "
            f"(agent_id={agent_id}, dataset_variant={dataset_variant}, samples={samples_value})"
        ),
    )

    try:
        secrets = _state.get("provider_secrets") or {}
        oracle_user = str(secrets.get("oracle_user") or "").strip()
        oracle_password = str(secrets.get("oracle_password") or "").strip()
        oracle_dsn = str(secrets.get("oracle_dsn") or "").strip()
        oracle_schema = str(secrets.get("oracle_schema") or "").strip()

        if not all([oracle_user, oracle_password, oracle_dsn]):
            raise RuntimeError(
                "Missing Oracle credentials for background Evalground execution. "
                "Reconnect to Oracle and try again."
            )
        if not eval_script.exists():
            raise FileNotFoundError(
                f"LongMemEval evaluator script not found at {eval_script}"
            )

        if output_path.exists():
            output_path.unlink()

        command = [
            sys.executable,
            "-u",
            str(eval_script),
            "--dataset_variant",
            dataset_variant,
            "--num_samples",
            str(samples_value),
            "--output_dir",
            str(paths["results_dir"]),
            "--output_filename",
            output_filename,
            "--agent_id",
            agent_id,
        ]

        env = dict(os.environ)
        env["ORACLE_USER"] = oracle_user
        env["ORACLE_PASSWORD"] = oracle_password
        env["ORACLE_DSN"] = oracle_dsn
        if oracle_schema:
            env["ORACLE_SCHEMA"] = oracle_schema
        if paths.get("mode") == "repo":
            repo_src = paths["repo_root"] / "src"
            if repo_src.exists():
                existing_pythonpath = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = (
                    f"{repo_src}{os.pathsep}{existing_pythonpath}"
                    if existing_pythonpath
                    else str(repo_src)
                )

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=True,
        )
        _set_eval_run_process(run_id, process)
        _update_eval_run(run_id, process_id=process.pid)
        _append_eval_run_log(
            run_id,
            f"Benchmark subprocess started with PID {process.pid}.",
        )

        try:
            if process.stdout:
                for raw_line in process.stdout:
                    line = raw_line.rstrip("\n")
                    if line:
                        _append_eval_run_log(run_id, line)
            return_code = process.wait()
        finally:
            _pop_eval_run_process(run_id)

        run_state = _get_eval_run_snapshot(run_id) or {}
        cancel_requested = bool(run_state.get("cancel_requested"))

        if cancel_requested:
            _update_eval_run(
                run_id,
                status="canceled",
                finished_at=_utcnow_iso(),
                error="Evaluation canceled by user.",
            )
            _append_eval_run_log(run_id, "Evaluation was canceled.")
            return

        if return_code != 0:
            error_text = f"Benchmark exited with non-zero code: {return_code}"
            _update_eval_run(
                run_id,
                status="failed",
                finished_at=_utcnow_iso(),
                error=error_text,
            )
            _append_eval_run_log(run_id, error_text)
            return

        if not output_path.exists():
            error_text = f"Expected results file was not created: {output_path}"
            _update_eval_run(
                run_id,
                status="failed",
                finished_at=_utcnow_iso(),
                error=error_text,
            )
            _append_eval_run_log(run_id, error_text)
            return

        with open(output_path, "r", encoding="utf-8") as handle:
            eval_results = json.load(handle)

        try:
            eval_output_path = str(output_path.relative_to(paths["repo_root"]))
        except ValueError:
            eval_output_path = str(output_path)

        _update_eval_run(
            run_id,
            status="completed",
            finished_at=_utcnow_iso(),
            eval_results=eval_results,
            eval_output_path=eval_output_path,
            error=None,
        )
        _append_eval_run_log(run_id, "Evalground run completed successfully.")
    except Exception as exc:
        _pop_eval_run_process(run_id)
        error_text = str(exc)
        _update_eval_run(
            run_id,
            status="failed",
            finished_at=_utcnow_iso(),
            error=error_text,
        )
        _append_eval_run_log(run_id, f"Evalground run failed: {error_text}")


@router.post("/evalground/runs")
async def evalground_start_run(
    agent_id: str = Form(""),
    benchmark: str = Form("longmemeval"),
    dataset_variant: str = Form("oracle"),
    num_samples: str = Form("10"),
):
    """Start an Evalground run and return a run ID for live polling."""
    if not _state["provider"]:
        return JSONResponse(
            status_code=400,
            content={"error": "Memory provider is not connected."},
        )

    paths = _get_longmemeval_paths()
    dataset_files = _longmemeval_dataset_files(paths["dataset_dir"])
    evaluator_script = paths["evaluator_script"]

    try:
        samples_value = int(num_samples)
        if samples_value < 1:
            raise ValueError("Samples must be positive.")
    except (TypeError, ValueError):
        return JSONResponse(
            status_code=400,
            content={"error": "Samples must be a positive integer."},
        )

    if _state["provider_type"] != "oracle":
        return JSONResponse(
            status_code=400,
            content={"error": "Evalground runs only with the Oracle memory provider."},
        )
    if not agent_id.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "Please select an agent to evaluate."},
        )
    if benchmark != "longmemeval":
        return JSONResponse(
            status_code=400,
            content={"error": "Unsupported benchmark selection."},
        )
    if not os.environ.get("OPENAI_API_KEY"):
        return JSONResponse(
            status_code=400,
            content={"error": "OPENAI_API_KEY is required to run LongMemEval."},
        )
    if dataset_variant not in dataset_files:
        return JSONResponse(
            status_code=400,
            content={"error": "Unknown dataset variant."},
        )
    if not evaluator_script.exists():
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    "LongMemEval evaluator script is missing from this installation."
                )
            },
        )
    secrets = _state.get("provider_secrets") or {}
    if not all(
        str(secrets.get(key) or "").strip()
        for key in ("oracle_user", "oracle_password", "oracle_dsn")
    ):
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    "Missing Oracle credentials for Evalground background execution. "
                    "Reconnect to Oracle and try again."
                )
            },
        )
    if not dataset_files[dataset_variant].exists():
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    f"Dataset file {dataset_files[dataset_variant].name} is missing. "
                    "Use the 'Download datasets' action in Evalground."
                )
            },
        )

    selected_agent = _state["provider"].retrieve_memagent(agent_id)
    if not selected_agent:
        return JSONResponse(
            status_code=404,
            content={"error": "Selected agent was not found."},
        )

    run_id = _create_eval_run(
        {
            "benchmark": benchmark,
            "dataset_variant": dataset_variant,
            "num_samples": samples_value,
            "agent_id": agent_id,
        }
    )

    worker = threading.Thread(
        target=_run_evalground_job,
        args=(run_id, agent_id, dataset_variant, samples_value, paths),
        daemon=True,
    )
    worker.start()

    return {"run_id": run_id, "status": "queued"}


@router.get("/evalground/runs/active")
async def evalground_active_run():
    """Return the most recent active Evalground run, if one exists."""
    return {"run": _get_latest_active_eval_run()}


@router.get("/evalground/runs/{run_id}")
async def evalground_run_status(run_id: str, after: int = 0):
    """Return incremental logs and status for an Evalground run."""
    snapshot = _get_eval_run_delta(run_id, after=after)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Run not found")
    return snapshot


@router.post("/evalground/runs/{run_id}/stop")
async def evalground_stop_run(run_id: str):
    """Request stop for an active Evalground run."""
    run_state = _get_eval_run_snapshot(run_id)
    if not run_state:
        raise HTTPException(status_code=404, detail="Run not found")

    status = run_state.get("status")
    if status not in {"queued", "running", "canceling"}:
        return JSONResponse(
            status_code=409,
            content={
                "error": f"Run is already in terminal state '{status}'.",
                "status": status,
            },
        )

    _update_eval_run(run_id, cancel_requested=True, status="canceling")
    _append_eval_run_log(run_id, "Stop requested by user.")

    process = _get_eval_run_process(run_id)
    if process and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            _append_eval_run_log(
                run_id,
                f"Sent SIGTERM to benchmark process group (pid={process.pid}).",
            )
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                _append_eval_run_log(
                    run_id,
                    (
                        "Benchmark process group did not stop after SIGTERM; "
                        f"sent SIGKILL (pid={process.pid})."
                    ),
                )
        except ProcessLookupError:
            pass
        except Exception as exc:
            _append_eval_run_log(
                run_id,
                f"Failed to terminate benchmark process group: {exc}",
            )
    elif status == "queued":
        _update_eval_run(
            run_id,
            status="canceled",
            finished_at=_utcnow_iso(),
            error="Evaluation canceled before start.",
        )
        _append_eval_run_log(
            run_id,
            "Evaluation canceled before subprocess startup.",
        )

    latest_state = _get_eval_run_snapshot(run_id) or {}
    return {"run_id": run_id, "status": latest_state.get("status") or "canceling"}


@router.post("/evalground/datasets", response_class=HTMLResponse)
async def evalground_download(request: Request):
    """Download LongMemEval datasets using the bundled script."""
    if not _state["provider"]:
        return RedirectResponse(url="/connect", status_code=302)

    agents = []
    try:
        agents = _state["provider"].list_memagents()
    except Exception as e:
        logger.error(f"Failed to list agents for Evalground: {e}")

    paths = _get_longmemeval_paths()
    dataset_files = _longmemeval_dataset_files(paths["dataset_dir"])
    evaluator_script = paths["evaluator_script"]
    dataset_status, missing_variants = _build_dataset_status(dataset_files)

    download_message = None
    download_error = None

    if missing_variants:
        success, message = _run_longmemeval_dataset_download()
        if success:
            download_message = message
        else:
            download_error = message
    else:
        download_message = "Datasets are already available."

    dataset_status, missing_variants = _build_dataset_status(dataset_files)

    warnings = []
    if _state["provider_type"] != "oracle":
        warnings.append("Evalground currently supports the Oracle memory provider.")
    if not os.environ.get("OPENAI_API_KEY"):
        warnings.append("OPENAI_API_KEY is required to score LongMemEval responses.")
    if missing_variants:
        warnings.append(
            "LongMemEval datasets are missing. Use 'Download datasets' below."
        )
    if not evaluator_script.exists():
        warnings.append(
            "LongMemEval evaluator script is missing from this installation."
        )

    default_variant = "oracle"
    can_run = bool(agents)
    if _state["provider_type"] != "oracle":
        can_run = False
    if not os.environ.get("OPENAI_API_KEY"):
        can_run = False
    if not dataset_files[default_variant].exists():
        can_run = False
    if not evaluator_script.exists():
        can_run = False

    runs_history = _build_eval_run_history_rows(agents=agents, limit=200)

    return templates.TemplateResponse(
        "evalground.html",
        {
            "request": request,
            "provider_type": _state["provider_type"],
            "connection_info": _state["connection_info"],
            "agents": agents,
            "selected_agent": None,
            "selected_agent_id": "",
            "benchmark": "longmemeval",
            "dataset_variant": default_variant,
            "num_samples": 10,
            "eval_results": None,
            "eval_output_path": None,
            "run_output": None,
            "error": None,
            "warnings": warnings,
            "dataset_status": dataset_status,
            "missing_variants": missing_variants,
            "download_message": download_message,
            "download_error": download_error,
            "can_run": can_run,
            "eval_results_dir": str(paths["results_dir"]),
            "runs_history": runs_history,
            "selected_run_id": None,
            "selected_run_status": None,
            "active_page": "evalground",
        },
    )


# -----------------------------------------------------------------------------
# LongMemEval path/dataset helpers and log capture
# -----------------------------------------------------------------------------


class _EvalgroundLogHandler(logging.Handler):
    """Capture a bounded set of log lines for Evalground run output."""

    def __init__(
        self,
        max_lines: int = 800,
        on_line: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(level=logging.INFO)
        self.max_lines = max_lines
        self._on_line = on_line
        self._lines: List[str] = []
        self.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
        except Exception:
            return
        if self._on_line:
            try:
                self._on_line(line)
            except Exception:
                pass
        self._lines.append(line)
        if len(self._lines) > self.max_lines:
            self._lines = self._lines[-self.max_lines :]

    def render(self) -> Optional[str]:
        if not self._lines:
            return None
        return "\n".join(self._lines)


@contextmanager
def _capture_logs(
    handler: logging.Handler, minimum_level: int = logging.INFO
) -> Iterator[None]:
    """Attach a temporary log handler and restore logger state afterwards."""
    root_logger = logging.getLogger()
    original_level = root_logger.level
    if original_level > minimum_level:
        root_logger.setLevel(minimum_level)
    root_logger.addHandler(handler)
    try:
        yield
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(original_level)


def _is_site_packages_path(path: Path) -> bool:
    """Return True when the path is inside a site/dist-packages directory."""
    parts = {part.lower() for part in path.parts}
    return "site-packages" in parts or "dist-packages" in parts


def _iter_candidate_roots() -> List[Path]:
    """Build candidate roots that may contain eval/longmemeval scripts."""
    candidates: List[Path] = []

    cwd = Path.cwd().resolve()
    candidates.append(cwd)
    candidates.extend(cwd.parents)

    module_path = Path(__file__).resolve()
    candidates.extend(module_path.parents)

    seen: Set[str] = set()
    unique: List[Path] = []
    for item in candidates:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _find_longmemeval_eval_dir() -> Tuple[Optional[Path], Optional[Path]]:
    """Find a root containing eval/longmemeval scripts."""
    cached = getattr(_find_longmemeval_eval_dir, "_cached", None)
    if cached is not None:
        return cached

    for root in _iter_candidate_roots():
        eval_dir = root / "eval" / "longmemeval"
        if (eval_dir / "evaluate_memorizz.py").exists():
            _find_longmemeval_eval_dir._cached = (root, eval_dir)
            return root, eval_dir

    _find_longmemeval_eval_dir._cached = (None, None)
    return None, None


def _get_longmemeval_paths() -> Dict[str, Any]:
    """Resolve LongMemEval script/data/results paths for repo and package installs."""
    root, eval_dir = _find_longmemeval_eval_dir()
    if root is None:
        root = Path.cwd().resolve()

    user_eval_home = (
        Path(os.environ.get("MEMORIZZ_EVALGROUND_HOME", ""))
        if os.environ.get("MEMORIZZ_EVALGROUND_HOME")
        else (Path.home() / ".memorizz" / "evalground" / "longmemeval")
    )

    if eval_dir and not _is_site_packages_path(eval_dir):
        dataset_dir = eval_dir / "data"
        results_dir = eval_dir / "results"
        mode = "repo"
    else:
        dataset_dir = user_eval_home / "data"
        results_dir = user_eval_home / "results"
        mode = "package"

    try:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        fallback_home = (
            Path(tempfile.gettempdir()) / "memorizz" / "evalground" / "longmemeval"
        )
        dataset_dir = fallback_home / "data"
        results_dir = fallback_home / "results"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

    evaluator_script = (
        eval_dir / "evaluate_memorizz.py"
        if eval_dir
        else (root / "eval" / "longmemeval" / "evaluate_memorizz.py")
    )
    download_script = (
        eval_dir / "download_dataset.py"
        if eval_dir
        else (root / "eval" / "longmemeval" / "download_dataset.py")
    )

    return {
        "repo_root": root,
        "eval_dir": eval_dir or (root / "eval" / "longmemeval"),
        "dataset_dir": dataset_dir,
        "results_dir": results_dir,
        "evaluator_script": evaluator_script,
        "download_script": download_script,
        "mode": mode,
    }


def _longmemeval_dataset_files(dataset_dir: Path) -> Dict[str, Path]:
    """Map LongMemEval dataset variants to local files."""
    return {
        "oracle": dataset_dir / "longmemeval_oracle.json",
        "s": dataset_dir / "longmemeval_s.json",
        "m": dataset_dir / "longmemeval_m.json",
    }


def _build_dataset_status(
    dataset_files: Dict[str, Path],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Describe dataset availability for the UI."""
    status = []
    missing = []
    for variant, path in dataset_files.items():
        exists = path.exists()
        status.append(
            {
                "variant": variant,
                "filename": path.name,
                "exists": exists,
            }
        )
        if not exists:
            missing.append(variant)
    return status, missing


def _run_longmemeval_dataset_download() -> Tuple[bool, str]:
    """Run the LongMemEval dataset download script."""
    paths = _get_longmemeval_paths()
    script_path = paths["download_script"]
    if not script_path.exists():
        return (
            False,
            "download_dataset.py was not found. Reinstall/upgrade memorizz with Evalground assets.",
        )

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--output_dir",
                str(paths["dataset_dir"]),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return False, f"Failed to run download script: {exc}"

    output = (result.stdout or "") + (result.stderr or "")
    if output:
        logger.info("LongMemEval download output:\n%s", output)

    dataset_files = _longmemeval_dataset_files(_get_longmemeval_paths()["dataset_dir"])
    _, missing = _build_dataset_status(dataset_files)
    if result.returncode != 0 or missing:
        # Surface the script's output so the user can see what went wrong.
        detail = output.strip()[-500:] if output.strip() else "No output captured."
        return False, f"Dataset download did not complete successfully.\n{detail}"

    return True, "Dataset download complete."


def _load_longmemeval_evaluator():
    """Load the LongMemEval evaluator class via importlib."""
    cached = getattr(_load_longmemeval_evaluator, "_cached", None)
    if cached:
        return cached

    evaluator_path = _get_longmemeval_paths()["evaluator_script"]
    if not evaluator_path.exists():
        raise FileNotFoundError(f"LongMemEval evaluator not found at {evaluator_path}")

    spec = importlib.util.spec_from_file_location(
        "memorizz_longmemeval", evaluator_path
    )
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load LongMemEval evaluator module.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evaluator_cls = getattr(module, "LongMemEvalEvaluator", None)
    if evaluator_cls is None:
        raise ImportError("LongMemEvalEvaluator is missing in evaluator script.")

    _load_longmemeval_evaluator._cached = evaluator_cls
    return evaluator_cls

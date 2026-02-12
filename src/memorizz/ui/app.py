"""FastAPI application for Memorizz Local UI."""

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
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)

# Global state for the connected memory provider
_state: Dict[str, Any] = {
    "provider": None,
    "provider_type": None,
    "connection_info": {},
    "provider_secrets": {},
}

_eval_runs_lock = threading.Lock()
_eval_runs: Dict[str, Dict[str, Any]] = {}
_eval_run_processes: Dict[str, subprocess.Popen] = {}
_EVAL_RUN_MAX_LOG_LINES = 2000

# Paths
UI_DIR = Path(__file__).parent
TEMPLATES_DIR = UI_DIR / "templates"
STATIC_DIR = UI_DIR / "static"
ROOT_DIR = UI_DIR.parent.parent.parent
ENV_FILE_PATH = ROOT_DIR / ".env"

LLM_MODEL_CATALOG: Dict[str, List[Dict[str, str]]] = {
    # Source: https://platform.openai.com/docs/models (Latest models section)
    "openai": [
        {"value": "gpt-5.2", "label": "GPT-5.2", "group": "Featured"},
        {"value": "gpt-5.1", "label": "GPT-5.1", "group": "Featured"},
        {"value": "gpt-5", "label": "GPT-5", "group": "GPT-5 Family"},
        {"value": "gpt-5-mini", "label": "GPT-5 Mini", "group": "GPT-5 Family"},
        {"value": "gpt-5-nano", "label": "GPT-5 Nano", "group": "GPT-5 Family"},
        {"value": "o3-pro", "label": "o3 Pro", "group": "Reasoning"},
        {"value": "o3", "label": "o3", "group": "Reasoning"},
        {"value": "o4-mini", "label": "o4 Mini", "group": "Reasoning"},
        {"value": "gpt-4.1", "label": "GPT-4.1", "group": "GPT-4.1 Family"},
        {
            "value": "gpt-4.1-mini",
            "label": "GPT-4.1 Mini",
            "group": "GPT-4.1 Family",
        },
        {
            "value": "gpt-4.1-nano",
            "label": "GPT-4.1 Nano",
            "group": "GPT-4.1 Family",
        },
        {"value": "gpt-4o", "label": "GPT-4o", "group": "Legacy"},
        {"value": "gpt-4o-mini", "label": "GPT-4o Mini", "group": "Legacy"},
        {"value": "gpt-4-turbo", "label": "GPT-4 Turbo", "group": "Legacy"},
    ],
    # Azure OpenAI uses deployment names. These IDs are convenient defaults.
    "azure": [
        {"value": "gpt-5.2", "label": "GPT-5.2 deployment", "group": "Current"},
        {"value": "gpt-5", "label": "GPT-5 deployment", "group": "Current"},
        {"value": "gpt-5-mini", "label": "GPT-5 Mini deployment", "group": "Current"},
        {"value": "o3", "label": "o3 deployment", "group": "Reasoning"},
        {"value": "o4-mini", "label": "o4 Mini deployment", "group": "Reasoning"},
        {"value": "gpt-4.1", "label": "GPT-4.1 deployment", "group": "Previous"},
        {"value": "gpt-4o", "label": "GPT-4o deployment", "group": "Previous"},
    ],
    "huggingface": [],
}

DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_LLM_MODEL_BY_PROVIDER = {
    "openai": "gpt-5.2",
    "azure": "gpt-5",
    "huggingface": "meta-llama/Meta-Llama-3-8B-Instruct",
}
DEFAULT_GRAALPY_INTERNET_ACCESS = "1"

SETTINGS_SECTIONS = [
    {
        "title": "LLM & Embeddings",
        "description": "Keys used by OpenAI, Azure OpenAI, Voyage AI, and Hugging Face integrations.",
        "fields": [
            {
                "env": "OPENAI_API_KEY",
                "label": "OpenAI API Key",
                "placeholder": "sk-...",
                "hint": "Required for OpenAI LLM and embedding providers.",
            },
            {
                "env": "AZURE_OPENAI_API_KEY",
                "label": "Azure OpenAI API Key",
                "placeholder": "azure-...",
                "hint": "Used by Azure OpenAI LLM and embedding providers.",
            },
            {
                "env": "VOYAGE_API_KEY",
                "label": "Voyage AI API Key",
                "placeholder": "pa-...",
                "hint": "Used by the Voyage AI embedding provider.",
            },
            {
                "env": "HF_TOKEN",
                "label": "Hugging Face Token",
                "placeholder": "hf_...",
                "hint": "Optional token for private Hugging Face models.",
            },
            {
                "env": "MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER",
                "label": "Default Embedding Provider",
                "hint": "Used by Oracle provider connections when embedding provider is not set explicitly.",
                "field_type": "select",
                "options": [
                    {"value": "", "label": "Not set"},
                    {"value": "openai", "label": "OpenAI"},
                    {"value": "azure", "label": "Azure OpenAI"},
                    {"value": "huggingface", "label": "Hugging Face"},
                    {"value": "voyageai", "label": "Voyage AI"},
                    {"value": "ollama", "label": "Ollama"},
                ],
            },
            {
                "env": "MEMORIZZ_DEFAULT_EMBEDDING_MODEL",
                "label": "Default Embedding Model",
                "placeholder": "text-embedding-3-small",
                "hint": "Optional model id used with the default embedding provider.",
            },
            {
                "env": "MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS",
                "label": "Default Embedding Dimensions",
                "placeholder": "1536",
                "hint": "Optional output dimension override (must match Oracle VECTOR column dimensions).",
            },
        ],
    },
    {
        "title": "Default Agent Model",
        "description": "Default provider/model used in Agent creation and Playground config when not explicitly set on an agent.",
        "fields": [
            {
                "env": "MEMORIZZ_DEFAULT_LLM_PROVIDER",
                "label": "Default LLM Provider",
                "hint": "Applies to newly-created agents unless overridden.",
                "field_type": "select",
                "default_value": DEFAULT_LLM_PROVIDER,
                "options": [
                    {"value": "openai", "label": "OpenAI"},
                    {"value": "azure", "label": "Azure OpenAI"},
                    {"value": "huggingface", "label": "HuggingFace"},
                ],
            },
            {
                "env": "MEMORIZZ_DEFAULT_LLM_MODEL",
                "label": "Default Model / Deployment",
                "hint": "Provider-aware dropdown populated from the latest supported model catalog.",
                "field_type": "select",
                "default_value": DEFAULT_LLM_MODEL_BY_PROVIDER[DEFAULT_LLM_PROVIDER],
                "options": [],
            },
        ],
    },
    {
        "title": "Internet Access",
        "description": "Keys for Tavily, Firecrawl, and other web browsing providers.",
        "fields": [
            {
                "env": "TAVILY_API_KEY",
                "label": "Tavily API Key",
                "placeholder": "tvly-...",
                "hint": "Preferred default for Memorizz internet access.",
            },
            {
                "env": "FIRECRAWL_API_KEY",
                "label": "Firecrawl API Key",
                "placeholder": "fc-...",
                "hint": "Fallback provider for web browsing workflows.",
            },
            {
                "env": "MEMORIZZ_DEFAULT_INTERNET_PROVIDER_API_KEY",
                "label": "Default Internet Provider API Key",
                "placeholder": "provider-key",
                "hint": "Used when MEMORIZZ_DEFAULT_INTERNET_PROVIDER is set.",
            },
        ],
    },
    {
        "title": "Skills Marketplace",
        "description": "API key for skillsmp.com marketplace search integration.",
        "fields": [
            {
                "env": "SKILLSMP_API_KEY",
                "label": "SkillsMP API Key",
                "placeholder": "sk_live_skillsmp_...",
                "hint": "Used for https://skillsmp.com/ API access in agent marketplace tools.",
            },
        ],
    },
    {
        "title": "Sandbox Providers",
        "description": "Configure sandbox code execution providers. The default provider applies globally to all agents unless overridden per-agent in the Playground.",
        "fields": [
            {
                "env": "MEMORIZZ_DEFAULT_SANDBOX_PROVIDER",
                "label": "Default Sandbox Provider",
                "placeholder": "e2b",
                "hint": "Global default sandbox provider: e2b, daytona, or graalpy.",
                "field_type": "select",
                "options": [
                    {"value": "", "label": "None (disabled)"},
                    {"value": "e2b", "label": "E2B (Cloud — Firecracker microVMs)"},
                    {
                        "value": "daytona",
                        "label": "Daytona (Cloud — Full dev environments)",
                    },
                    {
                        "value": "graalpy",
                        "label": "GraalPy (Local — No cloud required)",
                    },
                ],
            },
            {
                "env": "E2B_API_KEY",
                "label": "E2B API Key",
                "placeholder": "e2b_...",
                "hint": "Required for E2B sandbox. Get one at e2b.dev ($100 free credits).",
            },
            {
                "env": "DAYTONA_API_KEY",
                "label": "Daytona API Key",
                "placeholder": "daytona_...",
                "hint": "Required for Daytona sandbox. Get one at daytona.io ($200 free credits).",
            },
            {
                "env": "GRAALPY_PATH",
                "label": "GraalPy Executable Path",
                "placeholder": "/path/to/graalpy",
                "hint": "Optional absolute path to the graalpy executable (used when not on PATH).",
                "field_type": "text",
            },
            {
                "env": "MEMORIZZ_GRAALPY_INTERNET_ACCESS",
                "label": "GraalPy Internet Access",
                "hint": (
                    "When enabled, GraalPy runs in subprocess mode (full network access). "
                    "When disabled, GraalPy uses java_wrapper + UNTRUSTED policy and requires GRAALPY_JAVA_WRAPPER_JAR."
                ),
                "field_type": "checkbox",
                "default_value": DEFAULT_GRAALPY_INTERNET_ACCESS,
                "checkbox_label": "Allow outbound internet access from GraalPy sandbox",
                "visible_if_sandbox_provider": "graalpy",
            },
            {
                "env": "GRAALPY_JAVA_WRAPPER_JAR",
                "label": "GraalPy Java Wrapper JAR",
                "placeholder": "/path/to/graalpy-sandbox.jar",
                "hint": "Required when GraalPy internet access is disabled (java_wrapper mode).",
                "field_type": "text",
                "visible_if_sandbox_provider": "graalpy",
            },
        ],
    },
]
SETTINGS_FIELDS = [
    field for section in SETTINGS_SECTIONS for field in section["fields"]
]

AGENT_SORT_LAST_RUN = "last_run"
AGENT_SORT_LAST_CREATED = "last_created"
AGENT_SORT_OPTIONS = {AGENT_SORT_LAST_RUN, AGENT_SORT_LAST_CREATED}
RECENT_NAV_AGENT_LIMIT = 6


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    logger.info("Memorizz UI starting...")
    yield
    # Cleanup: stop any active Evalground benchmark subprocesses
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
    # Cleanup: close provider connection
    if _state["provider"]:
        try:
            _state["provider"].close()
        except Exception:
            pass
    logger.info("Memorizz UI shutting down...")


def _normalize_llm_provider(value: Any) -> str:
    """Normalize user-configured LLM provider names."""
    provider = _to_text(value).strip().lower()
    if provider in LLM_MODEL_CATALOG:
        return provider
    return DEFAULT_LLM_PROVIDER


def _get_default_llm_provider() -> str:
    """Resolve default provider from environment with safe fallback."""
    return _normalize_llm_provider(os.environ.get("MEMORIZZ_DEFAULT_LLM_PROVIDER", ""))


def _get_default_llm_model(provider: Optional[str] = None) -> str:
    """Resolve default model/deployment from environment with provider fallback."""
    env_value = _to_text(os.environ.get("MEMORIZZ_DEFAULT_LLM_MODEL", "")).strip()
    if env_value:
        return env_value

    normalized_provider = _normalize_llm_provider(
        provider or _get_default_llm_provider()
    )
    return DEFAULT_LLM_MODEL_BY_PROVIDER.get(
        normalized_provider, DEFAULT_LLM_MODEL_BY_PROVIDER[DEFAULT_LLM_PROVIDER]
    )


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


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Memorizz Local UI",
        description="Web interface for exploring Memorizz memory providers",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Mount static files
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Setup templates
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.globals["llm_model_catalog"] = LLM_MODEL_CATALOG
    templates.env.globals["default_llm_provider"] = DEFAULT_LLM_PROVIDER
    templates.env.globals[
        "default_llm_model_by_provider"
    ] = DEFAULT_LLM_MODEL_BY_PROVIDER

    # -------------------------------------------------------------------------
    # Page Routes
    # -------------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        """Redirect to connect page or dashboard based on connection state."""
        if _state["provider"]:
            return RedirectResponse(url="/dashboard", status_code=302)
        return RedirectResponse(url="/connect", status_code=302)

    @app.get("/connect", response_class=HTMLResponse)
    async def connect_page(request: Request):
        """Show the connection page."""
        return templates.TemplateResponse(
            "connect.html",
            {
                "request": request,
                "error": None,
                "provider_type": "oracle",  # Default to Oracle
            },
        )

    @app.post("/connect", response_class=HTMLResponse)
    async def connect_submit(
        request: Request,
        provider_type: str = Form(...),
        # Oracle fields
        oracle_user: Optional[str] = Form(None),
        oracle_password: Optional[str] = Form(None),
        oracle_dsn: Optional[str] = Form(None),
        oracle_schema: Optional[str] = Form(None),
        # MongoDB fields
        mongodb_uri: Optional[str] = Form(None),
        mongodb_db_name: Optional[str] = Form(None),
        # FileSystem fields
        filesystem_path: Optional[str] = Form(None),
    ):
        """Handle connection form submission."""
        error = None
        try:
            provider = None

            if provider_type == "oracle":
                if not all([oracle_user, oracle_password, oracle_dsn]):
                    raise ValueError("Oracle requires user, password, and DSN")
                from ..memory_provider.oracle import OracleConfig, OracleProvider

                config = OracleConfig(
                    user=oracle_user,
                    password=oracle_password,
                    dsn=oracle_dsn,
                    schema=oracle_schema or oracle_user,
                    lazy_vector_indexes=True,
                )
                provider = OracleProvider(config)
                _state["connection_info"] = {
                    "user": oracle_user,
                    "dsn": oracle_dsn,
                    "schema": oracle_schema or oracle_user,
                }
                _state["provider_secrets"] = {
                    "oracle_user": oracle_user,
                    "oracle_password": oracle_password,
                    "oracle_dsn": oracle_dsn,
                    "oracle_schema": oracle_schema or oracle_user,
                }

            elif provider_type == "mongodb":
                if not mongodb_uri:
                    raise ValueError("MongoDB requires a URI")
                from ..memory_provider.mongodb import MongoDBConfig, MongoDBProvider

                config = MongoDBConfig(
                    uri=mongodb_uri,
                    db_name=mongodb_db_name or "memorizz",
                    lazy_vector_indexes=True,
                )
                provider = MongoDBProvider(config)
                _state["connection_info"] = {
                    "uri": _mask_uri(mongodb_uri),
                    "db_name": mongodb_db_name or "memorizz",
                }
                _state["provider_secrets"] = {}

            elif provider_type == "filesystem":
                if not filesystem_path:
                    raise ValueError("FileSystem requires a path")
                from ..memory_provider.filesystem import (
                    FileSystemConfig,
                    FileSystemProvider,
                )

                config = FileSystemConfig(root_path=filesystem_path)
                provider = FileSystemProvider(config)
                _state["connection_info"] = {"path": filesystem_path}
                _state["provider_secrets"] = {}

            else:
                raise ValueError(f"Unknown provider type: {provider_type}")

            # Close existing provider if any
            if _state["provider"]:
                try:
                    _state["provider"].close()
                except Exception:
                    pass

            _state["provider"] = provider
            _state["provider_type"] = provider_type
            logger.info(f"Connected to {provider_type} provider")

            return RedirectResponse(url="/dashboard", status_code=302)

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            error = _format_connect_error(provider_type, e)
            return templates.TemplateResponse(
                "connect.html",
                {
                    "request": request,
                    "error": error,
                    "provider_type": provider_type,
                },
            )

    @app.get("/disconnect")
    async def disconnect():
        """Disconnect from the current provider."""
        if _state["provider"]:
            try:
                _state["provider"].close()
            except Exception:
                pass
        _state["provider"] = None
        _state["provider_type"] = None
        _state["connection_info"] = {}
        _state["provider_secrets"] = {}
        return RedirectResponse(url="/connect", status_code=302)

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard(request: Request):
        """Show the main dashboard."""
        if not _state["provider"]:
            return RedirectResponse(url="/connect", status_code=302)

        # Get counts for each memory type
        stats = _get_memory_stats()

        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "provider_type": _state["provider_type"],
                "connection_info": _state["connection_info"],
                "stats": stats,
                "active_page": "dashboard",
            },
        )

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request):
        """Show API key settings page."""
        if not _state["provider"]:
            return RedirectResponse(url="/connect", status_code=302)

        return templates.TemplateResponse(
            "settings.html",
            {
                "request": request,
                "provider_type": _state["provider_type"],
                "connection_info": _state["connection_info"],
                "settings_sections": _build_settings_sections(),
                "message": None,
                "error": None,
                "active_page": "settings",
            },
        )

    @app.post("/settings", response_class=HTMLResponse)
    async def settings_submit(request: Request):
        """Handle API key settings submission."""
        if not _state["provider"]:
            return RedirectResponse(url="/connect", status_code=302)

        form = await request.form()
        updates: Dict[str, str] = {}
        for field in SETTINGS_FIELDS:
            env_key = field["env"]
            raw_value = form.get(env_key)
            if raw_value is None:
                continue
            value = str(raw_value).strip()
            if value:
                updates[env_key] = value

        message = None
        error = None
        if updates:
            env_error = _apply_env_updates(updates)
            if env_error:
                message = f"Saved {len(updates)} setting(s) for this session."
                error = f"Failed to update .env: {env_error}"
            else:
                message = f"Saved {len(updates)} setting(s)."
        else:
            message = "No changes to save."

        sandbox_default_provider = _to_text(
            os.environ.get("MEMORIZZ_DEFAULT_SANDBOX_PROVIDER", "")
        ).strip()
        sandbox_validation_error = _validate_sandbox_provider_choice(
            sandbox_default_provider
        )
        if sandbox_validation_error:
            if error:
                error = f"{error} | Sandbox: {sandbox_validation_error}"
            else:
                error = sandbox_validation_error

        return templates.TemplateResponse(
            "settings.html",
            {
                "request": request,
                "provider_type": _state["provider_type"],
                "connection_info": _state["connection_info"],
                "settings_sections": _build_settings_sections(),
                "message": message,
                "error": error,
                "active_page": "settings",
            },
        )

    @app.get("/playground", response_class=HTMLResponse)
    async def playground_index(request: Request):
        """Show the playground page with agent selector."""
        if not _state["provider"]:
            return RedirectResponse(url="/connect", status_code=302)

        agents = []
        try:
            agents = _state["provider"].list_memagents()
        except Exception as e:
            logger.error(f"Failed to list agents for playground: {e}")

        return templates.TemplateResponse(
            "playground_select.html",
            {
                "request": request,
                "provider_type": _state["provider_type"],
                "connection_info": _state["connection_info"],
                "agents": agents,
                "active_page": "playground",
                "agents_nav": _build_agent_nav_items(),
            },
        )

    @app.get("/agents", response_class=HTMLResponse)
    async def agents_list(request: Request):
        """Show list of all agents."""
        if not _state["provider"]:
            return RedirectResponse(url="/connect", status_code=302)

        sort_by = _normalize_agent_sort_option(request.query_params.get("sort_by"))
        agents = []
        last_run_by_agent: Dict[str, float] = {}
        try:
            agents = _state["provider"].list_memagents()
            last_run_by_agent = _load_agent_last_run_map(agents)
            if sort_by == AGENT_SORT_LAST_CREATED:
                agents = _sort_agents_by_created_at_desc(agents)
            else:
                agents = _sort_agents_by_last_run_desc(
                    agents, last_run_by_agent=last_run_by_agent
                )
        except Exception as e:
            logger.error(f"Failed to list agents: {e}")

        agent_threads = _build_agent_threads_map(agents)
        agent_recent_messages, agent_default_memory_ids = _build_agent_recent_messages(
            agents,
            thread_rows_by_agent=agent_threads,
            per_agent_limit=5,
        )
        agent_tool_counts = _build_agent_tool_count_map(agents)

        return templates.TemplateResponse(
            "agents.html",
            {
                "request": request,
                "provider_type": _state["provider_type"],
                "connection_info": _state["connection_info"],
                "agents": agents,
                "agent_recent_messages": agent_recent_messages,
                "agent_default_memory_ids": agent_default_memory_ids,
                "agent_threads": agent_threads,
                "agent_tool_counts": agent_tool_counts,
                "sort_by": sort_by,
                "agents_nav": _build_agent_nav_items(
                    active_agent_id=None,
                    agents=agents,
                    last_run_by_agent=last_run_by_agent,
                ),
                "active_page": "agents",
            },
        )

    @app.post("/agents/{agent_id}/favorite")
    async def agent_toggle_favorite(
        agent_id: str,
        redirect_to: str = Form("/agents"),
        is_favorite: Optional[str] = Form(None),
    ):
        """Toggle or set an agent favorite flag."""
        if not _state["provider"]:
            return RedirectResponse(url="/connect", status_code=302)

        from ..memagent.models import MemAgentModel

        existing = _state["provider"].retrieve_memagent(agent_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Agent not found")

        if is_favorite is None:
            next_value = not bool(getattr(existing, "is_favorite", False))
        else:
            next_value = _parse_bool(is_favorite)

        updated = MemAgentModel(
            agent_id=agent_id,
            name=getattr(existing, "name", None),
            instruction=getattr(existing, "instruction", None),
            application_mode=getattr(existing, "application_mode", "assistant"),
            memory_types=getattr(existing, "memory_types", None),
            max_steps=getattr(existing, "max_steps", 20),
            tool_access=getattr(existing, "tool_access", "private"),
            semantic_cache=bool(getattr(existing, "semantic_cache", False)),
            memory_ids=getattr(existing, "memory_ids", None),
            persona=getattr(existing, "persona", None),
            llm_config=getattr(existing, "llm_config", None),
            tools=getattr(existing, "tools", None),
            delegates=getattr(existing, "delegates", None),
            semantic_cache_config=getattr(existing, "semantic_cache_config", None),
            context_window_tokens=getattr(existing, "context_window_tokens", None),
            is_favorite=next_value,
            internet_access_provider=getattr(
                existing, "internet_access_provider", None
            ),
            internet_access_config=getattr(existing, "internet_access_config", None),
            skills_marketplace_provider=getattr(
                existing, "skills_marketplace_provider", None
            ),
            skills_marketplace_config=getattr(
                existing, "skills_marketplace_config", None
            ),
            long_term_memory_ids=getattr(existing, "long_term_memory_ids", None),
            sandbox_provider=getattr(existing, "sandbox_provider", None),
            skill_paths=getattr(existing, "skill_paths", None),
            mcp_servers=getattr(existing, "mcp_servers", None),
        )

        try:
            _state["provider"].store_memagent(updated)
        except Exception as exc:
            logger.error("Failed to update favorite for agent %s: %s", agent_id, exc)
            raise HTTPException(status_code=500, detail="Failed to update favorite")

        redirect_target = _to_text(redirect_to).strip() or "/agents"
        if not redirect_target.startswith("/"):
            redirect_target = f"/agents/{agent_id}/playground"
        return RedirectResponse(url=redirect_target, status_code=302)

    @app.get("/agents/new", response_class=HTMLResponse)
    async def agent_create_page(request: Request):
        """Show form to create a new agent."""
        if not _state["provider"]:
            return RedirectResponse(url="/connect", status_code=302)

        from ..memagent.constants import DEFAULT_INSTRUCTION, DEFAULT_MAX_STEPS

        default_llm_provider = _get_default_llm_provider()
        default_llm_model = _get_default_llm_model(default_llm_provider)
        default_skills_marketplace_provider = (
            _normalize_skills_marketplace_provider_name(
                os.environ.get("MEMORIZZ_DEFAULT_SKILLS_MARKETPLACE_PROVIDER", "")
            )
            or ""
        )

        return templates.TemplateResponse(
            "agent_form.html",
            {
                "request": request,
                "provider_type": _state["provider_type"],
                "connection_info": _state["connection_info"],
                "active_page": "agents",
                "form_title": "Create Agent",
                "form_action": "/agents/new",
                "is_edit": False,
                "error": None,
                "agent_id": "",
                "agent_name": "",
                "instruction": DEFAULT_INSTRUCTION,
                "application_mode": "assistant",
                "max_steps": DEFAULT_MAX_STEPS,
                "tool_access": "private",
                "semantic_cache": False,
                "memory_ids_raw": "",
                "persona_name": "",
                "persona_role": "",
                "persona_goals": "",
                "persona_background": "",
                "llm_provider": default_llm_provider,
                "llm_model": default_llm_model,
                "llm_config_json": "",
                "sandbox_provider": os.environ.get(
                    "MEMORIZZ_DEFAULT_SANDBOX_PROVIDER", ""
                ),
                "internet_provider": os.environ.get(
                    "MEMORIZZ_DEFAULT_INTERNET_PROVIDER", ""
                ),
                "skills_marketplace_provider": default_skills_marketplace_provider,
                "enable_entity_memory": True,
                "enable_workflow_memory": False,
                "agent_tools": [],
            },
        )

    @app.post("/agents/new", response_class=HTMLResponse)
    async def agent_create_submit(
        request: Request,
        instruction: str = Form(""),
        application_mode: str = Form("assistant"),
        enable_entity_memory: Optional[str] = Form(None),
        enable_workflow_memory: Optional[str] = Form(None),
        max_steps: int = Form(20),
        tool_access: str = Form("private"),
        semantic_cache: Optional[str] = Form(None),
        memory_ids: str = Form(""),
        agent_name: str = Form(""),
        persona_name: str = Form(""),
        persona_role: str = Form(""),
        persona_goals: str = Form(""),
        persona_background: str = Form(""),
        llm_provider: str = Form(DEFAULT_LLM_PROVIDER),
        llm_model: str = Form(DEFAULT_LLM_MODEL_BY_PROVIDER[DEFAULT_LLM_PROVIDER]),
        llm_config_json: str = Form(""),
        sandbox_provider: str = Form(""),
        internet_provider: str = Form(""),
        skills_marketplace_provider: str = Form(""),
    ):
        """Create a new agent using the configured memory provider."""
        if not _state["provider"]:
            return RedirectResponse(url="/connect", status_code=302)

        from ..memagent.constants import DEFAULT_INSTRUCTION
        from ..memagent.models import MemAgentModel

        error = None
        llm_config, llm_error = _parse_llm_config(
            llm_provider, llm_model, llm_config_json
        )
        if llm_error:
            error = llm_error

        persona_payload = _build_persona_payload(
            persona_name, persona_role, persona_goals, persona_background
        )
        agent_name_value = _to_text(agent_name).strip() or None
        memory_id_list = _parse_memory_ids(memory_ids)
        semantic_cache_enabled = _parse_bool(semantic_cache)
        enable_entity_memory_value = _parse_bool(enable_entity_memory)
        enable_workflow_memory_value = _parse_bool(enable_workflow_memory)
        memory_types_value = _build_memory_types_for_agent(
            application_mode=application_mode or "assistant",
            enable_entity_memory=enable_entity_memory_value,
            enable_workflow_memory=enable_workflow_memory_value,
        )
        skills_marketplace_provider_value = (
            _normalize_skills_marketplace_provider_name(skills_marketplace_provider)
            or ""
        )

        if error:
            return templates.TemplateResponse(
                "agent_form.html",
                {
                    "request": request,
                    "provider_type": _state["provider_type"],
                    "connection_info": _state["connection_info"],
                    "active_page": "agents",
                    "form_title": "Create Agent",
                    "form_action": "/agents/new",
                    "is_edit": False,
                    "error": error,
                    "agent_id": "",
                    "agent_name": agent_name,
                    "instruction": instruction,
                    "application_mode": application_mode,
                    "max_steps": max_steps,
                    "tool_access": tool_access,
                    "semantic_cache": semantic_cache_enabled,
                    "memory_ids_raw": memory_ids,
                    "persona_name": persona_name,
                    "persona_role": persona_role,
                    "persona_goals": persona_goals,
                    "persona_background": persona_background,
                    "llm_provider": llm_provider,
                    "llm_model": llm_model,
                    "llm_config_json": llm_config_json,
                    "sandbox_provider": sandbox_provider,
                    "internet_provider": internet_provider,
                    "skills_marketplace_provider": skills_marketplace_provider_value,
                    "enable_entity_memory": enable_entity_memory_value,
                    "enable_workflow_memory": enable_workflow_memory_value,
                    "agent_tools": [],
                },
            )

        instruction_value = instruction.strip() if instruction else ""
        sandbox_value = sandbox_provider.strip() if sandbox_provider else None
        internet_value = _normalize_internet_provider_name(internet_provider) or None
        internet_config = _build_internet_provider_config(internet_value)
        skills_marketplace_value = skills_marketplace_provider_value or None
        skills_marketplace_config = _build_skills_marketplace_provider_config(
            skills_marketplace_value
        )
        sandbox_validation_error = _validate_sandbox_provider_choice(sandbox_value)
        internet_validation_error = _validate_internet_provider_choice(
            internet_value, internet_config
        )
        skills_marketplace_validation_error = (
            _validate_skills_marketplace_provider_choice(
                skills_marketplace_value,
                skills_marketplace_config,
            )
        )
        if (
            sandbox_validation_error
            or internet_validation_error
            or skills_marketplace_validation_error
        ):
            return templates.TemplateResponse(
                "agent_form.html",
                {
                    "request": request,
                    "provider_type": _state["provider_type"],
                    "connection_info": _state["connection_info"],
                    "active_page": "agents",
                    "form_title": "Create Agent",
                    "form_action": "/agents/new",
                    "is_edit": False,
                    "error": (
                        sandbox_validation_error
                        or internet_validation_error
                        or skills_marketplace_validation_error
                    ),
                    "agent_id": "",
                    "agent_name": agent_name,
                    "instruction": instruction,
                    "application_mode": application_mode,
                    "max_steps": max_steps,
                    "tool_access": tool_access,
                    "semantic_cache": semantic_cache_enabled,
                    "memory_ids_raw": memory_ids,
                    "persona_name": persona_name,
                    "persona_role": persona_role,
                    "persona_goals": persona_goals,
                    "persona_background": persona_background,
                    "llm_provider": llm_provider,
                    "llm_model": llm_model,
                    "llm_config_json": llm_config_json,
                    "sandbox_provider": sandbox_provider,
                    "internet_provider": internet_provider,
                    "skills_marketplace_provider": skills_marketplace_provider_value,
                    "enable_entity_memory": enable_entity_memory_value,
                    "enable_workflow_memory": enable_workflow_memory_value,
                    "agent_tools": [],
                },
            )

        memagent = MemAgentModel(
            name=agent_name_value,
            instruction=instruction_value or DEFAULT_INSTRUCTION,
            application_mode=application_mode or "assistant",
            memory_types=memory_types_value,
            max_steps=max_steps or 20,
            tool_access=tool_access or "private",
            semantic_cache=semantic_cache_enabled,
            is_favorite=False,
            memory_ids=memory_id_list or None,
            persona=persona_payload,
            llm_config=llm_config,
            sandbox_provider=sandbox_value,
            internet_access_provider=internet_value,
            internet_access_config=internet_config,
            skills_marketplace_provider=skills_marketplace_value,
            skills_marketplace_config=skills_marketplace_config,
        )

        try:
            result = _state["provider"].store_memagent(memagent)
            agent_id = _extract_agent_id(result, memagent.agent_id)
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            error = str(e)
            agent_id = None

        if error or not agent_id:
            return templates.TemplateResponse(
                "agent_form.html",
                {
                    "request": request,
                    "provider_type": _state["provider_type"],
                    "connection_info": _state["connection_info"],
                    "active_page": "agents",
                    "form_title": "Create Agent",
                    "form_action": "/agents/new",
                    "is_edit": False,
                    "error": error or "Failed to create agent.",
                    "agent_id": "",
                    "agent_name": agent_name,
                    "instruction": instruction,
                    "application_mode": application_mode,
                    "max_steps": max_steps,
                    "tool_access": tool_access,
                    "semantic_cache": semantic_cache_enabled,
                    "memory_ids_raw": memory_ids,
                    "persona_name": persona_name,
                    "persona_role": persona_role,
                    "persona_goals": persona_goals,
                    "persona_background": persona_background,
                    "llm_provider": llm_provider,
                    "llm_model": llm_model,
                    "llm_config_json": llm_config_json,
                    "sandbox_provider": sandbox_provider,
                    "internet_provider": internet_provider,
                    "skills_marketplace_provider": skills_marketplace_provider_value,
                    "enable_entity_memory": enable_entity_memory_value,
                    "enable_workflow_memory": enable_workflow_memory_value,
                    "agent_tools": [],
                },
            )

        return RedirectResponse(url=f"/agents/{agent_id}/playground", status_code=302)

    @app.get("/agents/{agent_id}/edit", response_class=HTMLResponse)
    async def agent_edit_page(request: Request, agent_id: str):
        """Show form to edit an agent."""
        if not _state["provider"]:
            return RedirectResponse(url="/connect", status_code=302)

        agent = _state["provider"].retrieve_memagent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        form_data = _build_agent_form_data(agent)

        return templates.TemplateResponse(
            "agent_form.html",
            {
                "request": request,
                "provider_type": _state["provider_type"],
                "connection_info": _state["connection_info"],
                "agents_nav": _build_agent_nav_items(active_agent_id=agent_id),
                "active_agent_id": agent_id,
                "active_page": "agents",
                "form_title": "Edit Agent",
                "form_action": f"/agents/{agent_id}/edit",
                "is_edit": True,
                "error": None,
                **form_data,
            },
        )

    @app.post("/agents/{agent_id}/edit", response_class=HTMLResponse)
    async def agent_edit_submit(
        request: Request,
        agent_id: str,
        instruction: str = Form(""),
        application_mode: str = Form("assistant"),
        enable_entity_memory: Optional[str] = Form(None),
        enable_workflow_memory: Optional[str] = Form(None),
        max_steps: int = Form(20),
        tool_access: str = Form("private"),
        semantic_cache: Optional[str] = Form(None),
        memory_ids: str = Form(""),
        agent_name: str = Form(""),
        persona_name: str = Form(""),
        persona_role: str = Form(""),
        persona_goals: str = Form(""),
        persona_background: str = Form(""),
        llm_provider: str = Form(DEFAULT_LLM_PROVIDER),
        llm_model: str = Form(DEFAULT_LLM_MODEL_BY_PROVIDER[DEFAULT_LLM_PROVIDER]),
        llm_config_json: str = Form(""),
        sandbox_provider: str = Form(""),
        internet_provider: str = Form(""),
        skills_marketplace_provider: Optional[str] = Form(None),
    ):
        """Update an existing agent."""
        if not _state["provider"]:
            return RedirectResponse(url="/connect", status_code=302)

        from ..memagent.constants import DEFAULT_INSTRUCTION
        from ..memagent.models import MemAgentModel

        existing = _state["provider"].retrieve_memagent(agent_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Agent not found")

        error = None
        llm_config, llm_error = _parse_llm_config(
            llm_provider, llm_model, llm_config_json
        )
        if llm_error:
            error = llm_error

        persona_payload = _build_persona_payload(
            persona_name, persona_role, persona_goals, persona_background
        )
        agent_name_value = _to_text(agent_name).strip() or None
        memory_id_list = _parse_memory_ids(memory_ids)
        semantic_cache_enabled = _parse_bool(semantic_cache)
        enable_entity_memory_value = _parse_bool(enable_entity_memory)
        enable_workflow_memory_value = _parse_bool(enable_workflow_memory)
        memory_types_value = _build_memory_types_for_agent(
            application_mode=application_mode
            or getattr(existing, "application_mode", "assistant"),
            enable_entity_memory=enable_entity_memory_value,
            enable_workflow_memory=enable_workflow_memory_value,
            existing_memory_types=getattr(existing, "memory_types", None),
        )
        if skills_marketplace_provider is None:
            skills_marketplace_provider_value = (
                _normalize_skills_marketplace_provider_name(
                    getattr(existing, "skills_marketplace_provider", None)
                )
                or ""
            )
        else:
            skills_marketplace_provider_value = (
                _normalize_skills_marketplace_provider_name(skills_marketplace_provider)
                or ""
            )

        skills_marketplace_value = skills_marketplace_provider_value or None
        existing_skills_provider = _normalize_skills_marketplace_provider_name(
            getattr(existing, "skills_marketplace_provider", None)
        )
        skills_marketplace_base_config = (
            getattr(existing, "skills_marketplace_config", None)
            if skills_marketplace_value
            and existing_skills_provider == skills_marketplace_value
            else None
        )
        skills_marketplace_config_value = _build_skills_marketplace_provider_config(
            skills_marketplace_value,
            skills_marketplace_base_config,
        )

        if error:
            form_data = _build_agent_form_data(existing)
            form_data.update(
                {
                    "instruction": instruction,
                    "application_mode": application_mode,
                    "max_steps": max_steps,
                    "tool_access": tool_access,
                    "semantic_cache": semantic_cache_enabled,
                    "memory_ids_raw": memory_ids,
                    "agent_name": agent_name,
                    "persona_name": persona_name,
                    "persona_role": persona_role,
                    "persona_goals": persona_goals,
                    "persona_background": persona_background,
                    "llm_provider": llm_provider,
                    "llm_model": llm_model,
                    "llm_config_json": llm_config_json,
                    "internet_provider": internet_provider,
                    "skills_marketplace_provider": skills_marketplace_provider_value,
                    "enable_entity_memory": enable_entity_memory_value,
                    "enable_workflow_memory": enable_workflow_memory_value,
                }
            )
            return templates.TemplateResponse(
                "agent_form.html",
                {
                    "request": request,
                    "provider_type": _state["provider_type"],
                    "connection_info": _state["connection_info"],
                    "agents_nav": _build_agent_nav_items(active_agent_id=agent_id),
                    "active_agent_id": agent_id,
                    "active_page": "agents",
                    "form_title": "Edit Agent",
                    "form_action": f"/agents/{agent_id}/edit",
                    "is_edit": True,
                    "error": error,
                    **form_data,
                },
            )

        instruction_value = instruction.strip() if instruction else ""
        internet_value = _normalize_internet_provider_name(internet_provider) or None
        internet_config = _build_internet_provider_config(internet_value)
        internet_validation_error = _validate_internet_provider_choice(
            internet_value, internet_config
        )
        if internet_validation_error:
            form_data = _build_agent_form_data(existing)
            form_data.update(
                {
                    "instruction": instruction,
                    "application_mode": application_mode,
                    "max_steps": max_steps,
                    "tool_access": tool_access,
                    "semantic_cache": semantic_cache_enabled,
                    "memory_ids_raw": memory_ids,
                    "agent_name": agent_name,
                    "persona_name": persona_name,
                    "persona_role": persona_role,
                    "persona_goals": persona_goals,
                    "persona_background": persona_background,
                    "llm_provider": llm_provider,
                    "llm_model": llm_model,
                    "llm_config_json": llm_config_json,
                    "sandbox_provider": sandbox_provider,
                    "internet_provider": internet_provider,
                    "skills_marketplace_provider": skills_marketplace_provider_value,
                    "enable_entity_memory": enable_entity_memory_value,
                    "enable_workflow_memory": enable_workflow_memory_value,
                }
            )
            return templates.TemplateResponse(
                "agent_form.html",
                {
                    "request": request,
                    "provider_type": _state["provider_type"],
                    "connection_info": _state["connection_info"],
                    "agents_nav": _build_agent_nav_items(active_agent_id=agent_id),
                    "active_agent_id": agent_id,
                    "active_page": "agents",
                    "form_title": "Edit Agent",
                    "form_action": f"/agents/{agent_id}/edit",
                    "is_edit": True,
                    "error": internet_validation_error,
                    **form_data,
                },
            )

        skills_marketplace_validation_error = (
            _validate_skills_marketplace_provider_choice(
                skills_marketplace_value,
                skills_marketplace_config_value,
            )
        )
        if skills_marketplace_validation_error:
            form_data = _build_agent_form_data(existing)
            form_data.update(
                {
                    "instruction": instruction,
                    "application_mode": application_mode,
                    "max_steps": max_steps,
                    "tool_access": tool_access,
                    "semantic_cache": semantic_cache_enabled,
                    "memory_ids_raw": memory_ids,
                    "agent_name": agent_name,
                    "persona_name": persona_name,
                    "persona_role": persona_role,
                    "persona_goals": persona_goals,
                    "persona_background": persona_background,
                    "llm_provider": llm_provider,
                    "llm_model": llm_model,
                    "llm_config_json": llm_config_json,
                    "sandbox_provider": sandbox_provider,
                    "internet_provider": internet_provider,
                    "skills_marketplace_provider": skills_marketplace_provider_value,
                    "enable_entity_memory": enable_entity_memory_value,
                    "enable_workflow_memory": enable_workflow_memory_value,
                }
            )
            return templates.TemplateResponse(
                "agent_form.html",
                {
                    "request": request,
                    "provider_type": _state["provider_type"],
                    "connection_info": _state["connection_info"],
                    "agents_nav": _build_agent_nav_items(active_agent_id=agent_id),
                    "active_agent_id": agent_id,
                    "active_page": "agents",
                    "form_title": "Edit Agent",
                    "form_action": f"/agents/{agent_id}/edit",
                    "is_edit": True,
                    "error": skills_marketplace_validation_error,
                    **form_data,
                },
            )

        sandbox_value = sandbox_provider.strip() if sandbox_provider else None
        if sandbox_value is None:
            sandbox_value = getattr(existing, "sandbox_provider", None)
        sandbox_validation_error = _validate_sandbox_provider_choice(sandbox_value)
        if sandbox_validation_error:
            form_data = _build_agent_form_data(existing)
            form_data.update(
                {
                    "instruction": instruction,
                    "application_mode": application_mode,
                    "max_steps": max_steps,
                    "tool_access": tool_access,
                    "semantic_cache": semantic_cache_enabled,
                    "memory_ids_raw": memory_ids,
                    "agent_name": agent_name,
                    "persona_name": persona_name,
                    "persona_role": persona_role,
                    "persona_goals": persona_goals,
                    "persona_background": persona_background,
                    "llm_provider": llm_provider,
                    "llm_model": llm_model,
                    "llm_config_json": llm_config_json,
                    "sandbox_provider": sandbox_provider,
                    "internet_provider": internet_provider,
                    "skills_marketplace_provider": skills_marketplace_provider_value,
                    "enable_entity_memory": enable_entity_memory_value,
                    "enable_workflow_memory": enable_workflow_memory_value,
                }
            )
            return templates.TemplateResponse(
                "agent_form.html",
                {
                    "request": request,
                    "provider_type": _state["provider_type"],
                    "connection_info": _state["connection_info"],
                    "agents_nav": _build_agent_nav_items(active_agent_id=agent_id),
                    "active_agent_id": agent_id,
                    "active_page": "agents",
                    "form_title": "Edit Agent",
                    "form_action": f"/agents/{agent_id}/edit",
                    "is_edit": True,
                    "error": sandbox_validation_error,
                    **form_data,
                },
            )

        updated = MemAgentModel(
            agent_id=agent_id,
            name=agent_name_value or getattr(existing, "name", None),
            instruction=instruction_value
            or getattr(existing, "instruction", None)
            or DEFAULT_INSTRUCTION,
            application_mode=application_mode
            or getattr(existing, "application_mode", "assistant"),
            memory_types=memory_types_value,
            max_steps=max_steps or getattr(existing, "max_steps", 20),
            tool_access=tool_access or getattr(existing, "tool_access", "private"),
            semantic_cache=semantic_cache_enabled,
            is_favorite=bool(getattr(existing, "is_favorite", False)),
            memory_ids=memory_id_list or getattr(existing, "memory_ids", None),
            persona=persona_payload,
            llm_config=llm_config or getattr(existing, "llm_config", None),
            tools=getattr(existing, "tools", None),
            delegates=getattr(existing, "delegates", None),
            semantic_cache_config=getattr(existing, "semantic_cache_config", None),
            context_window_tokens=getattr(existing, "context_window_tokens", None),
            internet_access_provider=internet_value,
            internet_access_config=internet_config,
            skills_marketplace_provider=skills_marketplace_value,
            skills_marketplace_config=skills_marketplace_config_value,
            long_term_memory_ids=getattr(existing, "long_term_memory_ids", None),
            sandbox_provider=sandbox_value,
            skill_paths=getattr(existing, "skill_paths", None),
            mcp_servers=getattr(existing, "mcp_servers", None),
        )

        try:
            _state["provider"].store_memagent(updated)
            _persist_mcp_configs_to_toolbox(
                agent_id=agent_id,
                mcp_servers=updated.mcp_servers or [],
                memory_ids=updated.memory_ids or [],
            )
        except Exception as e:
            logger.error(f"Failed to update agent {agent_id}: {e}")
            form_data = _build_agent_form_data(updated)
            return templates.TemplateResponse(
                "agent_form.html",
                {
                    "request": request,
                    "provider_type": _state["provider_type"],
                    "connection_info": _state["connection_info"],
                    "agents_nav": _build_agent_nav_items(active_agent_id=agent_id),
                    "active_agent_id": agent_id,
                    "active_page": "agents",
                    "form_title": "Edit Agent",
                    "form_action": f"/agents/{agent_id}/edit",
                    "is_edit": True,
                    "error": str(e),
                    **form_data,
                },
            )

        return RedirectResponse(url=f"/agents/{agent_id}", status_code=302)

    @app.post("/agents/{agent_id}/run", response_class=HTMLResponse)
    async def agent_run(
        request: Request,
        agent_id: str,
        query: str = Form(""),
        memory_id: str = Form(""),
    ):
        """Run an agent against a prompt."""
        if not _state["provider"]:
            return RedirectResponse(url="/connect", status_code=302)

        from ..memagent import MemAgent

        run_response = None
        run_error = None
        memory_id_value = memory_id.strip() if memory_id else None

        try:
            agent_instance = MemAgent.load(agent_id, memory_provider=_state["provider"])
            if not query.strip():
                raise ValueError("Please provide a prompt to run.")
            run_response = agent_instance.run(
                query.strip(), memory_id=memory_id_value or None
            )

            current_memory_id = getattr(agent_instance, "_current_memory_id", None)
            if current_memory_id and current_memory_id not in agent_instance.memory_ids:
                agent_instance.memory_ids.append(current_memory_id)
                if hasattr(_state["provider"], "update_memagent_memory_ids"):
                    _state["provider"].update_memagent_memory_ids(
                        agent_id, agent_instance.memory_ids
                    )

        except Exception as e:
            logger.error(f"Failed to run agent {agent_id}: {e}")
            run_error = str(e)

        agent_detail = _load_agent_detail(agent_id)

        return templates.TemplateResponse(
            "agent_detail.html",
            {
                "request": request,
                "provider_type": _state["provider_type"],
                "connection_info": _state["connection_info"],
                "agents_nav": _build_agent_nav_items(active_agent_id=agent_id),
                "active_agent_id": agent_id,
                "agent": agent_detail["agent"],
                "context_window": agent_detail["context_window"],
                "token_stats": agent_detail["token_stats"],
                "error": agent_detail["error"],
                "run_query": query,
                "run_response": run_response,
                "run_error": run_error,
                "active_page": "agents",
            },
        )

    @app.get("/agents/{agent_id}/playground", response_class=HTMLResponse)
    async def agent_playground(
        request: Request, agent_id: str, config_error: Optional[str] = None
    ):
        """Show the playground page for interacting with an agent."""
        if not _state["provider"]:
            return RedirectResponse(url="/connect", status_code=302)

        agent = None
        error = config_error
        context_window: List[Dict[str, Any]] = []
        toolbox_memory: List[Dict[str, Any]] = []
        workflow_memory: List[Dict[str, Any]] = []
        entity_memory: List[Dict[str, Any]] = []
        summary_memory: List[Dict[str, Any]] = []
        threads: List[Dict[str, Any]] = []
        default_memory_id = ""
        enable_entity_memory = False
        enable_workflow_memory = False
        entity_memory_status_error = ""

        try:
            agent = _state["provider"].retrieve_memagent(agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")

            threads = _build_agent_threads(agent)
            if threads:
                default_memory_id = str(threads[0].get("memory_id", "") or "")
            else:
                memory_ids = getattr(agent, "memory_ids", None) or []
                if isinstance(memory_ids, list) and memory_ids:
                    default_memory_id = _to_text(memory_ids[0]).strip()

            if default_memory_id:
                context_window = _load_thread_messages(default_memory_id, limit=None)
                toolbox_memory = _load_thread_toolbox_memory(
                    agent_id=agent_id,
                    memory_id=default_memory_id,
                    limit=None,
                )
                workflow_memory = _load_thread_workflow_memory(
                    agent_id=agent_id,
                    memory_id=default_memory_id,
                    limit=None,
                )
                entity_memory = _load_thread_entity_memory(
                    agent_id=agent_id,
                    memory_id=default_memory_id,
                    limit=None,
                )
                summary_memory = _load_thread_summary_memory(
                    agent_id=agent_id,
                    memory_id=default_memory_id,
                    limit=None,
                )
            enable_entity_memory = _agent_entity_memory_enabled(agent)
            enable_workflow_memory = _agent_workflow_memory_enabled(agent)
            entity_memory_status_error = _entity_memory_status_error(agent)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to load agent {agent_id}: {e}")
            error = str(e)

        # Build token stats for context panel
        token_stats = (
            _build_token_stats(
                agent,
                context_window,
                toolbox_memory=toolbox_memory,
                workflow_memory=workflow_memory,
                entity_memory=entity_memory,
                summary_memory=summary_memory,
            )
            if agent
            else None
        )

        # Extract editable config fields
        llm_config = getattr(agent, "llm_config", {}) or {} if agent else {}
        llm_provider = _normalize_llm_provider(llm_config.get("provider", "openai"))
        llm_model = (
            llm_config.get("model")
            or llm_config.get("deployment_name")
            or _get_default_llm_model(llm_provider)
        )
        instruction = getattr(agent, "instruction", "") or "" if agent else ""
        max_steps = getattr(agent, "max_steps", 20) if agent else 20

        persona = getattr(agent, "persona", None) if agent else None
        if isinstance(persona, dict):
            persona_name = persona.get("name", "")
            persona_role = persona.get("role", "")
            persona_goals = persona.get("goals", "")
            persona_background = persona.get("background", "")
        else:
            persona_name = getattr(persona, "name", "") if persona else ""
            persona_role = getattr(persona, "role", "") if persona else ""
            persona_goals = getattr(persona, "goals", "") if persona else ""
            persona_background = getattr(persona, "background", "") if persona else ""

        # Sandbox provider: per-agent override or global default
        sandbox_provider = getattr(agent, "sandbox_provider", None) if agent else None
        if isinstance(sandbox_provider, dict):
            sandbox_provider = sandbox_provider.get("provider", "")
        if not sandbox_provider:
            sandbox_provider = os.environ.get("MEMORIZZ_DEFAULT_SANDBOX_PROVIDER", "")
        sandbox_status_error = _validate_sandbox_provider_choice(sandbox_provider)

        internet_provider = (
            _normalize_internet_provider_name(
                getattr(agent, "internet_access_provider", None) if agent else None
            )
            or _normalize_internet_provider_name(
                os.environ.get("MEMORIZZ_DEFAULT_INTERNET_PROVIDER", "")
            )
            or ""
        )
        internet_provider_config = (
            getattr(agent, "internet_access_config", None) if agent else None
        )
        internet_status_error = _validate_internet_provider_choice(
            internet_provider,
            internet_provider_config,
        )

        skills_marketplace_provider = (
            _normalize_skills_marketplace_provider_name(
                getattr(agent, "skills_marketplace_provider", None) if agent else None
            )
            or _normalize_skills_marketplace_provider_name(
                os.environ.get("MEMORIZZ_DEFAULT_SKILLS_MARKETPLACE_PROVIDER", "")
            )
            or ""
        )
        skills_marketplace_provider_config = (
            getattr(agent, "skills_marketplace_config", None) if agent else None
        )
        skills_marketplace_status_error = _validate_skills_marketplace_provider_choice(
            skills_marketplace_provider,
            skills_marketplace_provider_config,
        )

        skill_paths = getattr(agent, "skill_paths", None) if agent else None
        if isinstance(skill_paths, str):
            skill_paths = [skill_paths]
        if not isinstance(skill_paths, list):
            skill_paths = []

        mcp_servers = getattr(agent, "mcp_servers", None) if agent else None
        if not isinstance(mcp_servers, list):
            mcp_servers = []

        return templates.TemplateResponse(
            "playground.html",
            {
                "request": request,
                "provider_type": _state["provider_type"],
                "connection_info": _state["connection_info"],
                "agents_nav": _build_agent_nav_items(active_agent_id=agent_id),
                "active_agent_id": agent_id,
                "agent": agent,
                "context_window": context_window,
                "toolbox_memory": toolbox_memory,
                "workflow_memory": workflow_memory,
                "entity_memory": entity_memory,
                "summary_memory": summary_memory,
                "token_stats": token_stats,
                "error": error,
                "active_page": "playground",
                # Config fields
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "instruction": instruction,
                "max_steps": max_steps,
                "sandbox_provider": sandbox_provider,
                "sandbox_status_error": sandbox_status_error,
                "internet_provider": internet_provider,
                "internet_status_error": internet_status_error,
                "skills_marketplace_provider": skills_marketplace_provider,
                "skills_marketplace_status_error": skills_marketplace_status_error,
                "agent_tools": _extract_agent_tools(agent),
                "persona_name": persona_name,
                "persona_role": persona_role,
                "persona_goals": persona_goals,
                "persona_background": persona_background,
                "enable_entity_memory": enable_entity_memory,
                "enable_workflow_memory": enable_workflow_memory,
                "entity_memory_status_error": entity_memory_status_error,
                "skill_paths": skill_paths,
                "mcp_servers": mcp_servers,
                "threads": threads,
                "default_memory_id": default_memory_id,
            },
        )

    @app.post("/agents/{agent_id}/playground/stream")
    async def agent_playground_stream(request: Request, agent_id: str):
        """SSE endpoint: stream agent responses as Server-Sent Events."""
        if not _state["provider"]:
            raise HTTPException(status_code=400, detail="Not connected")

        form = await request.form()
        query = str(form.get("query", "")).strip()
        memory_id = str(form.get("memory_id", "")).strip() or None
        # Allow runtime overrides from the playground config panel
        override_model = str(form.get("llm_model", "")).strip() or None
        override_instruction = str(form.get("instruction", "")).strip() or None

        if not query:
            raise HTTPException(status_code=400, detail="No query provided")

        from ..memagent import MemAgent

        def _event_stream():
            trace_queue: Queue = Queue()
            chunk_queue: Queue = Queue()
            stream_done = object()

            def _queue_trace_event(event: Dict[str, Any]) -> None:
                if not isinstance(event, dict):
                    return
                trace_queue.put(dict(event))

            def _drain_trace_events() -> Iterator[str]:
                while True:
                    try:
                        event = trace_queue.get_nowait()
                    except Empty:
                        break
                    if not isinstance(event, dict):
                        continue
                    event["type"] = "trace"
                    escaped_event = json.dumps(event)
                    yield f"data: {escaped_event}\n\n"

            agent_instance = None
            try:
                overrides: Dict[str, Any] = {"streaming": True}
                if override_instruction:
                    overrides["instruction"] = override_instruction
                agent_instance = MemAgent.load(
                    agent_id, memory_provider=_state["provider"], **overrides
                )
                if hasattr(agent_instance, "set_stream_event_callback"):
                    agent_instance.set_stream_event_callback(_queue_trace_event)
                # Apply runtime model override if user changed the model in the playground
                if override_model:
                    try:
                        from ..llms.llm_factory import create_llm_provider

                        llm_config = (
                            getattr(
                                _state["provider"].retrieve_memagent(agent_id),
                                "llm_config",
                                {},
                            )
                            or {}
                        )
                        llm_config["model"] = override_model
                        agent_instance.model = create_llm_provider(llm_config)
                    except Exception as exc:
                        logger.warning(f"Could not apply model override: {exc}")

                # Apply sandbox provider from agent config or global default
                stored_agent = _state["provider"].retrieve_memagent(agent_id)
                sandbox_cfg = (
                    getattr(stored_agent, "sandbox_provider", None)
                    if stored_agent
                    else None
                )
                if not sandbox_cfg:
                    sandbox_cfg = os.environ.get(
                        "MEMORIZZ_DEFAULT_SANDBOX_PROVIDER", ""
                    )
                resolved_sandbox_cfg = _resolve_sandbox_provider_config(sandbox_cfg)
                sandbox_apply_error = None
                if resolved_sandbox_cfg and not agent_instance.has_sandbox():
                    try:
                        validation_error = _validate_sandbox_provider_choice(
                            resolved_sandbox_cfg
                        )
                        if validation_error:
                            raise ValueError(validation_error)
                        agent_instance.with_sandbox_provider(resolved_sandbox_cfg)
                    except Exception as exc:
                        sandbox_apply_error = (
                            "Sandbox unavailable: " + _to_text(exc).strip()
                        )
                        logger.warning(f"Could not apply sandbox provider: {exc}")

                if sandbox_apply_error:
                    escaped_warning = json.dumps(
                        {
                            "type": "warning",
                            "message": sandbox_apply_error,
                            "scope": "sandbox",
                        }
                    )
                    yield f"data: {escaped_warning}\n\n"

                # Apply internet provider from agent config or global default.
                internet_apply_error = None
                internet_provider_name = _normalize_internet_provider_name(
                    getattr(stored_agent, "internet_access_provider", None)
                    if stored_agent
                    else None
                )
                internet_provider_config = (
                    getattr(stored_agent, "internet_access_config", None)
                    if stored_agent
                    else None
                )
                if not internet_provider_name:
                    internet_provider_name = _normalize_internet_provider_name(
                        os.environ.get("MEMORIZZ_DEFAULT_INTERNET_PROVIDER", "")
                    )
                    internet_provider_config = None

                if internet_provider_name and not agent_instance.has_internet_access():
                    try:
                        from ..internet_access import create_internet_access_provider

                        resolved_internet_config = _build_internet_provider_config(
                            internet_provider_name, internet_provider_config
                        )
                        provider_instance = create_internet_access_provider(
                            internet_provider_name,
                            resolved_internet_config or {},
                        )
                        if provider_instance:
                            agent_instance.with_internet_access_provider(
                                provider_instance
                            )
                        else:
                            internet_apply_error = f"Internet provider '{internet_provider_name}' is unknown."
                    except Exception as exc:
                        internet_apply_error = (
                            "Internet unavailable: " + _to_text(exc).strip()
                        )
                        logger.warning(f"Could not apply internet provider: {exc}")

                if internet_apply_error:
                    escaped_warning = json.dumps(
                        {
                            "type": "warning",
                            "message": internet_apply_error,
                            "scope": "internet",
                        }
                    )
                    yield f"data: {escaped_warning}\n\n"

                # Validate entity-memory runtime tool availability.
                entity_memory_warning = _entity_memory_status_error(stored_agent)
                if (
                    not entity_memory_warning
                    and _agent_entity_memory_enabled(stored_agent)
                    and getattr(agent_instance, "tool_manager", None) is not None
                ):
                    try:
                        runtime_tools = set(agent_instance.tool_manager.list_tools())
                    except Exception:
                        runtime_tools = set()
                    if (
                        "entity_memory_lookup" not in runtime_tools
                        or "entity_memory_upsert" not in runtime_tools
                    ):
                        entity_memory_warning = (
                            "Entity memory is enabled in config, but runtime entity-memory "
                            "tools are unavailable in this session."
                        )
                if entity_memory_warning:
                    escaped_warning = json.dumps(
                        {
                            "type": "warning",
                            "message": entity_memory_warning,
                            "scope": "entity_memory",
                        }
                    )
                    yield f"data: {escaped_warning}\n\n"

                def _run_agent_stream() -> None:
                    try:
                        for chunk in agent_instance.run_stream(
                            query, memory_id=memory_id
                        ):
                            chunk_queue.put(chunk)
                    except Exception as stream_exc:
                        chunk_queue.put({"__stream_error__": _to_text(stream_exc)})
                    finally:
                        chunk_queue.put(stream_done)

                stream_worker = threading.Thread(
                    target=_run_agent_stream,
                    name=f"memagent-stream-{agent_id}",
                    daemon=True,
                )
                stream_worker.start()

                while True:
                    for trace_payload in _drain_trace_events():
                        yield trace_payload

                    try:
                        stream_item = chunk_queue.get(timeout=0.1)
                    except Empty:
                        if stream_worker.is_alive():
                            continue
                        stream_item = stream_done

                    if stream_item is stream_done:
                        break

                    if (
                        isinstance(stream_item, dict)
                        and "__stream_error__" in stream_item
                    ):
                        raise RuntimeError(
                            _to_text(stream_item.get("__stream_error__")).strip()
                        )

                    escaped = json.dumps(stream_item)
                    yield f"data: {escaped}\n\n"

                    for trace_payload in _drain_trace_events():
                        yield trace_payload

                for trace_payload in _drain_trace_events():
                    yield trace_payload

                current_memory_id = getattr(agent_instance, "_current_memory_id", None)
                if current_memory_id:
                    current_memory_id = _to_text(current_memory_id).strip()
                if current_memory_id:
                    if current_memory_id not in (agent_instance.memory_ids or []):
                        agent_instance.memory_ids.append(current_memory_id)
                    if hasattr(_state["provider"], "update_memagent_memory_ids"):
                        try:
                            _state["provider"].update_memagent_memory_ids(
                                agent_id, agent_instance.memory_ids
                            )
                        except Exception as exc:
                            logger.warning(
                                "Failed to persist memory_ids for agent %s: %s",
                                agent_id,
                                exc,
                            )

                yield "data: [DONE]\n\n"
            except Exception as exc:
                logger.error(f"Streaming error for agent {agent_id}: {exc}")
                error_msg = json.dumps(f"Error: {exc}")
                yield f"data: {error_msg}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                if agent_instance is not None and hasattr(
                    agent_instance, "set_stream_event_callback"
                ):
                    try:
                        agent_instance.set_stream_event_callback(None)
                    except Exception:
                        pass

        return StreamingResponse(
            _event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/agents/{agent_id}/playground/thread")
    async def agent_playground_thread(agent_id: str, memory_id: str = ""):
        """Return conversation history for a specific thread memory_id."""
        if not _state["provider"]:
            raise HTTPException(status_code=400, detail="Not connected")

        agent = _state["provider"].retrieve_memagent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        requested_memory_id = _to_text(memory_id).strip()
        messages: List[Dict[str, Any]] = []
        if requested_memory_id:
            messages = _load_thread_messages(requested_memory_id, limit=None)
            messages = [
                msg
                for msg in messages
                if not _to_text(msg.get("agent_id")).strip()
                or _to_text(msg.get("agent_id")).strip() == agent_id
            ]

        serialized = [_serialize_thread_message(msg) for msg in messages]
        toolbox_memory = _load_thread_toolbox_memory(
            agent_id=agent_id,
            memory_id=requested_memory_id,
            limit=None,
        )
        workflow_memory = _load_thread_workflow_memory(
            agent_id=agent_id,
            memory_id=requested_memory_id,
            limit=None,
        )
        entity_memory = _load_thread_entity_memory(
            agent_id=agent_id,
            memory_id=requested_memory_id,
            limit=None,
        )
        summary_memory = _load_thread_summary_memory(
            agent_id=agent_id,
            memory_id=requested_memory_id,
            limit=None,
        )
        token_stats = _build_token_stats(
            agent,
            messages,
            toolbox_memory=toolbox_memory,
            workflow_memory=workflow_memory,
            entity_memory=entity_memory,
            summary_memory=summary_memory,
        )

        last_activity = "—"
        if messages:
            last_ts = _coerce_timestamp(messages[-1].get("timestamp"))
            if last_ts is not None:
                last_activity = datetime.fromtimestamp(last_ts).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

        return {
            "memory_id": requested_memory_id,
            "messages": serialized,
            "toolbox_memory": toolbox_memory,
            "workflow_memory": workflow_memory,
            "entity_memory": entity_memory,
            "summary_memory": summary_memory,
            "token_stats": token_stats,
            "message_count": len(serialized),
            "last_activity": last_activity,
        }

    @app.post("/agents/{agent_id}/playground/config")
    async def agent_playground_config_update(request: Request, agent_id: str):
        """Update agent config from the playground panel."""
        if not _state["provider"]:
            raise HTTPException(status_code=400, detail="Not connected")

        from ..memagent.models import MemAgentModel

        existing = _state["provider"].retrieve_memagent(agent_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Agent not found")

        form = await request.form()
        new_instruction = str(form.get("instruction", "")).strip()
        new_model = str(form.get("llm_model", "")).strip()
        new_provider = str(form.get("llm_provider", "")).strip()
        new_max_steps = form.get("max_steps")
        new_sandbox_provider = str(form.get("sandbox_provider", "")).strip()
        new_internet_provider = _normalize_internet_provider_name(
            form.get("internet_provider", "")
        )
        raw_skills_marketplace_provider = form.get("skills_marketplace_provider")
        if raw_skills_marketplace_provider is None:
            new_skills_marketplace_provider = (
                _normalize_skills_marketplace_provider_name(
                    getattr(existing, "skills_marketplace_provider", None)
                )
            )
        else:
            new_skills_marketplace_provider = (
                _normalize_skills_marketplace_provider_name(
                    raw_skills_marketplace_provider
                )
            )
        new_persona_name = str(form.get("persona_name", "")).strip()
        new_persona_role = str(form.get("persona_role", "")).strip()
        new_persona_goals = str(form.get("persona_goals", "")).strip()
        new_persona_background = str(form.get("persona_background", "")).strip()
        raw_skill_paths = form.get("skill_paths_json")
        raw_mcp_servers = form.get("mcp_servers_json")
        raw_enable_entity_memory = form.get("enable_entity_memory")
        raw_enable_workflow_memory = form.get("enable_workflow_memory")

        parsed_skill_paths = None
        parsed_mcp_servers = None
        parse_error = None

        if raw_skill_paths is not None:
            parsed_skill_paths, parse_error = _parse_skill_paths_json(
                str(raw_skill_paths)
            )
        if not parse_error and raw_mcp_servers is not None:
            parsed_mcp_servers, parse_error = _parse_mcp_servers_json(
                str(raw_mcp_servers)
            )
        if parse_error:
            from urllib.parse import quote

            return RedirectResponse(
                url=f"/agents/{agent_id}/playground?config_error={quote(parse_error[:200])}",
                status_code=302,
            )

        # Build updated fields
        llm_config = getattr(existing, "llm_config", {}) or {}
        if new_provider:
            llm_config["provider"] = new_provider
        if new_model:
            if new_provider == "azure":
                llm_config["deployment_name"] = new_model
            else:
                llm_config["model"] = new_model

        persona_payload = None
        existing_persona = getattr(existing, "persona", None)
        if new_persona_name or (
            existing_persona
            and (
                (isinstance(existing_persona, dict) and existing_persona.get("name"))
                or (hasattr(existing_persona, "name") and existing_persona.name)
            )
        ):
            persona_payload = {}
            if isinstance(existing_persona, dict):
                persona_payload = dict(existing_persona)
            elif existing_persona and hasattr(existing_persona, "__dict__"):
                persona_payload = dict(existing_persona.__dict__)
            if new_persona_name:
                persona_payload["name"] = new_persona_name
            if new_persona_role:
                persona_payload["role"] = new_persona_role
            # Always update goals and background (allow clearing them)
            persona_payload["goals"] = new_persona_goals
            persona_payload["background"] = new_persona_background

        max_steps_value = getattr(existing, "max_steps", 20)
        if new_max_steps:
            try:
                max_steps_value = int(new_max_steps)
            except (ValueError, TypeError):
                pass

        enable_entity_memory_value = _parse_bool(raw_enable_entity_memory)
        enable_workflow_memory_value = _parse_bool(raw_enable_workflow_memory)
        memory_types_value = _build_memory_types_for_agent(
            application_mode=getattr(existing, "application_mode", "assistant"),
            enable_entity_memory=enable_entity_memory_value,
            enable_workflow_memory=enable_workflow_memory_value,
            existing_memory_types=getattr(existing, "memory_types", None),
        )

        # Resolve sandbox provider: use form value, fall back to existing
        sandbox_value = new_sandbox_provider if new_sandbox_provider else None
        if sandbox_value is None:
            sandbox_value = getattr(existing, "sandbox_provider", None)
        sandbox_validation_error = _validate_sandbox_provider_choice(sandbox_value)
        if sandbox_validation_error:
            from urllib.parse import quote

            return RedirectResponse(
                url=(
                    f"/agents/{agent_id}/playground?config_error="
                    f"{quote(sandbox_validation_error[:220])}"
                ),
                status_code=302,
            )

        internet_value = new_internet_provider or None
        internet_config_value = _build_internet_provider_config(internet_value)
        internet_validation_error = _validate_internet_provider_choice(
            internet_value, internet_config_value
        )
        if internet_validation_error:
            from urllib.parse import quote

            return RedirectResponse(
                url=(
                    f"/agents/{agent_id}/playground?config_error="
                    f"{quote(internet_validation_error[:220])}"
                ),
                status_code=302,
            )

        skills_marketplace_value = new_skills_marketplace_provider or None
        existing_skills_provider = _normalize_skills_marketplace_provider_name(
            getattr(existing, "skills_marketplace_provider", None)
        )
        skills_marketplace_base_config = (
            getattr(existing, "skills_marketplace_config", None)
            if skills_marketplace_value
            and existing_skills_provider == skills_marketplace_value
            else None
        )
        skills_marketplace_config_value = _build_skills_marketplace_provider_config(
            skills_marketplace_value,
            skills_marketplace_base_config,
        )
        skills_marketplace_validation_error = (
            _validate_skills_marketplace_provider_choice(
                skills_marketplace_value,
                skills_marketplace_config_value,
            )
        )
        if skills_marketplace_validation_error:
            from urllib.parse import quote

            return RedirectResponse(
                url=(
                    f"/agents/{agent_id}/playground?config_error="
                    f"{quote(skills_marketplace_validation_error[:220])}"
                ),
                status_code=302,
            )

        updated = MemAgentModel(
            agent_id=agent_id,
            name=getattr(existing, "name", None),
            instruction=new_instruction or getattr(existing, "instruction", None),
            application_mode=getattr(existing, "application_mode", "assistant"),
            memory_types=memory_types_value,
            max_steps=max_steps_value,
            tool_access=getattr(existing, "tool_access", "private"),
            semantic_cache=bool(getattr(existing, "semantic_cache", False)),
            is_favorite=bool(getattr(existing, "is_favorite", False)),
            memory_ids=getattr(existing, "memory_ids", None),
            persona=persona_payload,
            llm_config=llm_config,
            tools=getattr(existing, "tools", None),
            delegates=getattr(existing, "delegates", None),
            semantic_cache_config=getattr(existing, "semantic_cache_config", None),
            context_window_tokens=getattr(existing, "context_window_tokens", None),
            internet_access_provider=internet_value,
            internet_access_config=internet_config_value,
            skills_marketplace_provider=skills_marketplace_value,
            skills_marketplace_config=skills_marketplace_config_value,
            long_term_memory_ids=getattr(existing, "long_term_memory_ids", None),
            sandbox_provider=sandbox_value,
            skill_paths=(
                parsed_skill_paths
                if parsed_skill_paths is not None
                else getattr(existing, "skill_paths", None)
            ),
            mcp_servers=(
                parsed_mcp_servers
                if parsed_mcp_servers is not None
                else getattr(existing, "mcp_servers", None)
            ),
        )

        try:
            _state["provider"].store_memagent(updated)
            _persist_mcp_configs_to_toolbox(
                agent_id=agent_id,
                mcp_servers=updated.mcp_servers or [],
                memory_ids=updated.memory_ids or [],
            )
        except Exception as e:
            logger.error(f"Failed to update agent config {agent_id}: {e}")
            # Redirect back to playground with error as query param
            from urllib.parse import quote

            return RedirectResponse(
                url=f"/agents/{agent_id}/playground?config_error={quote(str(e)[:200])}",
                status_code=302,
            )

        return RedirectResponse(url=f"/agents/{agent_id}/playground", status_code=302)

    @app.get("/agents/{agent_id}", response_class=HTMLResponse)
    async def agent_detail(request: Request, agent_id: str):
        """Show agent detail with context window."""
        if not _state["provider"]:
            return RedirectResponse(url="/connect", status_code=302)

        agent_detail = _load_agent_detail(agent_id)

        return templates.TemplateResponse(
            "agent_detail.html",
            {
                "request": request,
                "provider_type": _state["provider_type"],
                "connection_info": _state["connection_info"],
                "agents_nav": _build_agent_nav_items(active_agent_id=agent_id),
                "active_agent_id": agent_id,
                "agent": agent_detail["agent"],
                "context_window": agent_detail["context_window"],
                "token_stats": agent_detail["token_stats"],
                "error": agent_detail["error"],
                "run_query": None,
                "run_response": None,
                "run_error": None,
                "active_page": "agents",
            },
        )

    @app.get("/observability")
    async def observability_redirect(request: Request):
        """Backwards-compatible redirect to traces page."""
        query = request.url.query
        suffix = f"?{query}" if query else ""
        return RedirectResponse(url=f"/traces{suffix}", status_code=302)

    @app.get("/traces", response_class=HTMLResponse)
    async def traces_page(
        request: Request,
        agent_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        thread_memory_id: Optional[str] = None,
        q: Optional[str] = None,
    ):
        """Show traces dashboard with searchable agent table and per-agent timeline."""
        if not _state["provider"]:
            return RedirectResponse(url="/connect", status_code=302)

        agents: List[Any] = []
        try:
            agents = _state["provider"].list_memagents()
        except Exception as exc:
            logger.error("Failed to list agents for traces: %s", exc)

        agent_query = _to_text(q).strip()
        last_run_by_agent = _load_agent_last_run_map(agents)
        agents = _sort_agents_by_last_run_desc(
            agents, last_run_by_agent=last_run_by_agent
        )

        tool_counts = _build_agent_tool_count_map(agents)
        trace_metrics = _build_agent_trace_metrics(agents)
        thread_rows_by_agent = _build_agent_thread_rows(agents)
        agent_rows = _build_trace_agent_rows(
            agents=agents,
            tool_counts=tool_counts,
            trace_metrics=trace_metrics,
            thread_rows_by_agent=thread_rows_by_agent,
            search_query=agent_query,
        )

        selected_agent = None
        trace_events: List[Dict[str, Any]] = []
        trace_summary: Optional[Dict[str, Any]] = None
        selected_thread_row: Optional[Dict[str, Any]] = None
        error = None

        selected_agent_id = _to_text(agent_id).strip() or None
        selected_thread_id = _to_text(thread_id).strip() or None
        selected_thread_memory_id = _to_text(thread_memory_id).strip() or None
        if selected_agent_id:
            try:
                selected_agent = _state["provider"].retrieve_memagent(selected_agent_id)
                if not selected_agent:
                    raise HTTPException(status_code=404, detail="Agent not found")

                trace_events = _load_agent_trace_events(
                    selected_agent,
                    thread_id=selected_thread_id,
                    thread_memory_id=selected_thread_memory_id,
                )
                latest_timestamp = (
                    trace_events[-1].get("timestamp") if trace_events else None
                )
                trace_summary = {
                    "event_count": len(trace_events),
                    "latest_timestamp": _format_trace_timestamp(latest_timestamp),
                }
                for thread_row in thread_rows_by_agent.get(selected_agent_id, []):
                    row_thread_id = _to_text(thread_row.get("thread_id")).strip()
                    row_memory_id = _to_text(thread_row.get("memory_id")).strip()
                    if selected_thread_memory_id:
                        if row_memory_id != selected_thread_memory_id:
                            continue
                    elif selected_thread_id:
                        if (
                            row_thread_id != selected_thread_id
                            and row_memory_id != selected_thread_id
                        ):
                            continue
                    else:
                        continue
                    selected_thread_row = thread_row
                    break
            except HTTPException:
                raise
            except Exception as exc:
                logger.error(
                    "Failed to load traces for agent %s: %s",
                    selected_agent_id,
                    exc,
                )
                error = str(exc)

        return templates.TemplateResponse(
            "observability.html",
            {
                "request": request,
                "provider_type": _state["provider_type"],
                "connection_info": _state["connection_info"],
                "agents_nav": _build_agent_nav_items(
                    active_agent_id=selected_agent_id,
                    agents=agents,
                    last_run_by_agent=last_run_by_agent,
                ),
                "active_agent_id": selected_agent_id,
                "active_page": "traces",
                "agent_query": agent_query,
                "agent_rows": agent_rows,
                "selected_agent": selected_agent,
                "selected_agent_id": selected_agent_id,
                "selected_thread_id": selected_thread_id,
                "selected_thread_memory_id": selected_thread_memory_id,
                "selected_thread_row": selected_thread_row,
                "trace_events": trace_events,
                "trace_summary": trace_summary,
                "error": error,
            },
        )

    @app.get("/evalground", response_class=HTMLResponse)
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
            warnings.append(
                "OPENAI_API_KEY is required to score LongMemEval responses."
            )
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

    @app.post("/evalground/runs")
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
                content={
                    "error": "Evalground runs only with the Oracle memory provider."
                },
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

    @app.get("/evalground/runs/active")
    async def evalground_active_run():
        """Return the most recent active Evalground run, if one exists."""
        return {"run": _get_latest_active_eval_run()}

    @app.get("/evalground/runs/{run_id}")
    async def evalground_run_status(run_id: str, after: int = 0):
        """Return incremental logs and status for an Evalground run."""
        snapshot = _get_eval_run_delta(run_id, after=after)
        if not snapshot:
            raise HTTPException(status_code=404, detail="Run not found")
        return snapshot

    @app.post("/evalground/runs/{run_id}/stop")
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

    @app.post("/evalground", response_class=HTMLResponse)
    async def evalground_run(
        request: Request,
        agent_id: str = Form(""),
        benchmark: str = Form("longmemeval"),
        dataset_variant: str = Form("oracle"),
        num_samples: str = Form("10"),
    ):
        """Run a benchmark evaluation for a selected agent."""
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
            warnings.append(
                "OPENAI_API_KEY is required to score LongMemEval responses."
            )
        if missing_variants:
            warnings.append(
                "LongMemEval datasets are missing. Use 'Download datasets' below."
            )
        if not evaluator_script.exists():
            warnings.append(
                "LongMemEval evaluator script is missing from this installation."
            )

        error = None
        eval_results = None
        eval_output_path = None
        run_output = None
        selected_agent = None

        samples_value = None
        try:
            samples_value = int(num_samples)
            if samples_value < 1:
                raise ValueError("Samples must be positive.")
        except (TypeError, ValueError):
            error = "Samples must be a positive integer."
            samples_value = 10

        if not error and _state["provider_type"] != "oracle":
            error = "Evalground runs only with the Oracle memory provider."
        if not error and not agent_id.strip():
            error = "Please select an agent to evaluate."
        if not error and benchmark != "longmemeval":
            error = "Unsupported benchmark selection."
        if not error and not os.environ.get("OPENAI_API_KEY"):
            error = "OPENAI_API_KEY is required to run LongMemEval."
        if not error and dataset_variant not in dataset_files:
            error = "Unknown dataset variant."
        if not error and not evaluator_script.exists():
            error = "LongMemEval evaluator script is missing from this installation."
        if not error and not dataset_files[dataset_variant].exists():
            error = (
                f"Dataset file {dataset_files[dataset_variant].name} is missing. "
                "Use the 'Download datasets' action in Evalground."
            )

        if not error:
            log_capture = _EvalgroundLogHandler(max_lines=1200)
            with _capture_logs(log_capture):
                logger.info(
                    "Evalground run started (agent_id=%s, dataset_variant=%s, samples=%s)",
                    agent_id,
                    dataset_variant,
                    samples_value,
                )
                try:
                    selected_agent = _state["provider"].retrieve_memagent(agent_id)
                    if not selected_agent:
                        raise HTTPException(status_code=404, detail="Agent not found")

                    evaluator_cls = _load_longmemeval_evaluator()
                    evaluator = evaluator_cls(
                        dataset_variant=dataset_variant,
                        application_mode=None,
                        output_dir=str(paths["results_dir"]),
                        verbose=False,
                        memory_provider=_state["provider"],
                        agent_template=selected_agent,
                        dataset_dir=str(paths["dataset_dir"]),
                    )

                    eval_results = evaluator.evaluate(num_samples=samples_value)
                    output_path = evaluator.save_results(eval_results)
                    try:
                        eval_output_path = str(
                            output_path.relative_to(paths["repo_root"])
                        )
                    except ValueError:
                        eval_output_path = str(output_path)
                    logger.info("Evalground run finished successfully")
                except HTTPException as exc:
                    error = exc.detail
                    logger.error(f"Evalground evaluation failed: {error}")
                except Exception as exc:
                    logger.error(f"Evalground evaluation failed: {exc}")
                    error = str(exc)
            run_output = log_capture.render()
            if error and not run_output:
                run_output = f"Evalground evaluation failed: {error}"

        can_run = bool(agents)
        if _state["provider_type"] != "oracle":
            can_run = False
        if not os.environ.get("OPENAI_API_KEY"):
            can_run = False
        if (
            dataset_variant in dataset_files
            and not dataset_files[dataset_variant].exists()
        ):
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
                "selected_agent": selected_agent,
                "selected_agent_id": agent_id,
                "benchmark": benchmark,
                "dataset_variant": dataset_variant,
                "num_samples": samples_value,
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
                "selected_run_id": None,
                "selected_run_status": None,
                "active_page": "evalground",
            },
        )

    @app.post("/evalground/datasets", response_class=HTMLResponse)
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
            warnings.append(
                "OPENAI_API_KEY is required to score LongMemEval responses."
            )
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

    @app.post("/agents/{agent_id}/delete")
    async def delete_agent(agent_id: str, cascade: bool = False):
        """Delete an agent."""
        if not _state["provider"]:
            raise HTTPException(status_code=400, detail="Not connected")

        try:
            success = _state["provider"].delete_memagent(agent_id, cascade=cascade)
            if not success:
                raise HTTPException(status_code=404, detail="Agent not found")
        except Exception as e:
            logger.error(f"Failed to delete agent {agent_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        return RedirectResponse(url="/agents", status_code=302)

    @app.get("/memory/{memory_type}", response_class=HTMLResponse)
    async def memory_list(request: Request, memory_type: str):
        """Show list of items for a memory type."""
        if not _state["provider"]:
            return RedirectResponse(url="/connect", status_code=302)

        from ..enums.memory_type import MemoryType

        # Map URL path to MemoryType
        type_mapping = {
            "personas": MemoryType.PERSONAS,
            "toolbox": MemoryType.TOOLBOX,
            "conversations": MemoryType.CONVERSATION_MEMORY,
            "workflows": MemoryType.WORKFLOW_MEMORY,
            "long-term": MemoryType.LONG_TERM_MEMORY,
            "short-term": MemoryType.SHORT_TERM_MEMORY,
            "entity": MemoryType.ENTITY_MEMORY,
            "summaries": MemoryType.SUMMARIES,
            "shared": MemoryType.SHARED_MEMORY,
            "cache": MemoryType.SEMANTIC_CACHE,
        }

        mem_type = type_mapping.get(memory_type)
        if not mem_type:
            raise HTTPException(status_code=404, detail="Unknown memory type")

        items = []
        try:
            items = _state["provider"].list_all(mem_type)
        except Exception as e:
            logger.error(f"Failed to list {memory_type}: {e}")

        return templates.TemplateResponse(
            "memory_list.html",
            {
                "request": request,
                "provider_type": _state["provider_type"],
                "connection_info": _state["connection_info"],
                "memory_type": memory_type,
                "memory_type_display": memory_type.replace("-", " ").title(),
                "items": items[:100],  # Limit to 100 items
                "active_page": memory_type,
            },
        )

    # -------------------------------------------------------------------------
    # API Routes (for AJAX/HTMX)
    # -------------------------------------------------------------------------

    @app.get("/api/status")
    async def api_status():
        """Get current connection status."""
        return {
            "connected": _state["provider"] is not None,
            "provider_type": _state["provider_type"],
            "connection_info": _state["connection_info"],
        }

    @app.get("/api/agents")
    async def api_list_agents():
        """API endpoint to list all agents."""
        if not _state["provider"]:
            raise HTTPException(status_code=400, detail="Not connected")

        agents = []
        try:
            raw_agents = _state["provider"].list_memagents()
            for agent in raw_agents:
                agents.append(_serialize_agent(agent))
        except Exception as e:
            logger.error(f"Failed to list agents: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        return {"agents": agents, "count": len(agents)}

    @app.get("/api/agents/{agent_id}")
    async def api_get_agent(agent_id: str):
        """API endpoint to get agent details."""
        if not _state["provider"]:
            raise HTTPException(status_code=400, detail="Not connected")

        try:
            agent = _state["provider"].retrieve_memagent(agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")
            return _serialize_agent(agent)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get agent {agent_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def _mask_uri(uri: str) -> str:
    """Mask sensitive parts of a connection URI."""
    if "@" in uri:
        # Hide credentials
        parts = uri.split("@")
        return f"***@{parts[-1]}"
    return uri


def _format_connect_error(provider_type: str, exc: Exception) -> str:
    """Return a user-facing connection error message."""
    message = str(exc).strip()
    if provider_type == "oracle":
        return _format_oracle_connect_error(message)
    return message or "Connection failed."


def _format_oracle_connect_error(message: str) -> str:
    """Map common Oracle connection failures to helpful messages."""
    lowered = message.lower()
    if "oracledb package is required" in lowered or "pip install oracledb" in lowered:
        return (
            "Oracle driver (oracledb) is not installed. "
            "Install with: pip install memorizz[oracle] (or pip install oracledb)."
        )
    if "dpi-1047" in lowered or "oracle client library" in lowered:
        return (
            "Oracle client libraries are missing. Install Oracle Instant Client or "
            f"configure python-oracledb to use thin mode. Details: {message}"
        )
    connection_markers = (
        "ora-12541",
        "ora-12514",
        "ora-12170",
        "ora-12545",
        "ora-12543",
        "no listener",
        "connection refused",
        "econnrefused",
        "timed out",
        "dpy-6005",
    )
    if any(marker in lowered for marker in connection_markers):
        return (
            "Cannot reach Oracle at the DSN. Ensure the database is running and "
            "the host/port/service are correct. If using Docker, start the "
            f"container and confirm port 1521 is mapped. Details: {message}"
        )
    return message or "Oracle connection failed."


def _read_lob_value(value: Any) -> Any:
    """Read Oracle LOB values into plain Python types when possible."""
    reader = getattr(value, "read", None)
    if callable(reader):
        try:
            return reader()
        except Exception:
            return value
    return value


def _to_text(value: Any) -> str:
    """Coerce values (including LOB/bytes) into safe UI strings."""
    value = _read_lob_value(value)
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.decode("utf-8", errors="ignore")
    return str(value)


def _extract_agent_identifier(agent: Any) -> Optional[str]:
    """Extract a string agent identifier from dict/model values."""
    if isinstance(agent, dict):
        raw_id = agent.get("agent_id") or agent.get("agentId") or agent.get("_id")
    else:
        raw_id = (
            getattr(agent, "agent_id", None)
            or getattr(agent, "agentId", None)
            or getattr(agent, "_id", None)
        )
    if raw_id is None:
        return None
    text = _to_text(raw_id).strip()
    return text or None


def _extract_agent_memory_ids(agent: Any) -> List[str]:
    """Extract normalized memory IDs from a memagent object/dict."""
    if isinstance(agent, dict):
        raw_memory_ids = agent.get("memory_ids") or agent.get("memoryIds") or []
    else:
        raw_memory_ids = getattr(agent, "memory_ids", None) or []

    memory_ids: List[str] = []
    for memory_id in raw_memory_ids:
        text = _to_text(memory_id).strip()
        if text:
            memory_ids.append(text)
    return memory_ids


def _extract_agent_persona_name(agent: Any) -> str:
    """Extract display name for an agent."""
    if isinstance(agent, dict):
        explicit_name = _to_text(agent.get("name")).strip()
    else:
        explicit_name = _to_text(getattr(agent, "name", None)).strip()
    if explicit_name:
        return explicit_name

    if isinstance(agent, dict):
        persona = agent.get("persona")
    else:
        persona = getattr(agent, "persona", None)

    if persona:
        if isinstance(persona, dict):
            name = _to_text(persona.get("name")).strip()
        else:
            name = _to_text(getattr(persona, "name", None)).strip()
        if name:
            return name
    return "Agent"


def _normalize_agent_sort_option(sort_by: Optional[str]) -> str:
    """Normalize sort options for the agent list page."""
    value = _to_text(sort_by).strip().lower()
    if value in AGENT_SORT_OPTIONS:
        return value
    return AGENT_SORT_LAST_RUN


def _parse_object_id_timestamp(value: Any) -> Optional[float]:
    """Parse Mongo ObjectId timestamps from hex/ObjectId(...) strings."""
    raw = _to_text(value).strip()
    if not raw:
        return None

    if raw.startswith("ObjectId(") and raw.endswith(")"):
        raw = raw[len("ObjectId(") : -1].strip().strip("'").strip('"')

    if len(raw) != 24:
        return None

    try:
        int(raw, 16)
    except ValueError:
        return None

    return float(int(raw[:8], 16))


def _coerce_timestamp(value: Any) -> Optional[float]:
    """Best-effort conversion of mixed timestamp formats into epoch seconds."""
    value = _read_lob_value(value)
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if hasattr(value, "timestamp") and callable(getattr(value, "timestamp")):
        try:
            return float(value.timestamp())
        except Exception:
            pass

    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except Exception:
            value = value.decode("utf-8", errors="ignore")

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None

        iso_value = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
        try:
            return float(datetime.fromisoformat(iso_value).timestamp())
        except ValueError:
            pass

        try:
            return float(raw)
        except ValueError:
            return None

    return None


def _load_memagent_created_at_map() -> Dict[str, float]:
    """Build a creation-time lookup map keyed by agent_id."""
    provider = _state.get("provider")
    if not provider:
        return {}

    try:
        from ..enums.memory_type import MemoryType

        docs = provider.list_all(MemoryType.MEMAGENT) or []
    except Exception as exc:
        logger.debug("Failed to load memagent docs for sorting: %s", exc)
        return {}

    created_by_agent: Dict[str, float] = {}
    for doc in docs:
        if not isinstance(doc, dict):
            continue

        raw_id = doc.get("agent_id") or doc.get("agentId") or doc.get("_id")
        agent_id = _to_text(raw_id).strip() if raw_id is not None else ""
        if not agent_id:
            continue

        created_at = doc.get("created_at")
        if created_at is None:
            created_at = doc.get("createdAt")

        timestamp = _coerce_timestamp(created_at)
        if timestamp is None:
            timestamp = _parse_object_id_timestamp(raw_id)

        if timestamp is not None:
            created_by_agent[agent_id] = timestamp

    return created_by_agent


def _agent_created_timestamp(
    agent: Any, created_by_agent: Optional[Dict[str, float]] = None
) -> float:
    """Resolve an agent's creation timestamp for deterministic sorting."""
    agent_id = _extract_agent_identifier(agent)

    if agent_id and created_by_agent and agent_id in created_by_agent:
        return created_by_agent[agent_id]

    if isinstance(agent, dict):
        created_at = agent.get("created_at")
        if created_at is None:
            created_at = agent.get("createdAt")
    else:
        created_at = getattr(agent, "created_at", None)
        if created_at is None:
            created_at = getattr(agent, "createdAt", None)

    timestamp = _coerce_timestamp(created_at)
    if timestamp is not None:
        return timestamp

    if agent_id:
        object_id_timestamp = _parse_object_id_timestamp(agent_id)
        if object_id_timestamp is not None:
            return object_id_timestamp

    return 0.0


def _sort_agents_by_created_at_desc(agents: List[Any]) -> List[Any]:
    """Sort agents by creation time (newest first), preserving ties."""
    if not agents:
        return []

    created_by_agent = _load_memagent_created_at_map()
    indexed = [
        (_agent_created_timestamp(agent, created_by_agent), idx, agent)
        for idx, agent in enumerate(agents)
    ]
    indexed.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in indexed]


def _extract_message_timestamp(message: Any) -> Optional[float]:
    """Extract a comparable timestamp from conversation message payloads."""
    if not isinstance(message, dict):
        return None

    timestamp = _coerce_timestamp(message.get("timestamp"))
    if timestamp is not None:
        return timestamp
    timestamp = _coerce_timestamp(message.get("created_at"))
    if timestamp is not None:
        return timestamp
    return _coerce_timestamp(message.get("createdAt"))


def _retrieve_conversation_history(
    memory_id: str, limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Retrieve conversation history with provider fallback and deterministic ordering."""
    provider = _state.get("provider")
    memory_id_value = _to_text(memory_id).strip()
    if not provider or not memory_id_value:
        return []

    history: List[Dict[str, Any]] = []

    # Preferred path: provider-level API (supported by all official providers).
    try:
        rows = (
            provider.retrieve_conversation_history_ordered_by_timestamp(
                memory_id=memory_id_value, limit=None
            )
            or []
        )
        history = [row for row in rows if isinstance(row, dict)]
    except Exception as exc:
        logger.debug(
            "Provider history API unavailable for memory '%s': %s",
            memory_id_value,
            exc,
        )

    # Compatibility fallback for custom providers exposing stores only.
    if not history:
        try:
            from ..enums.memory_type import MemoryType

            stores = getattr(provider, "stores", None) or {}
            conversation_store = stores.get(
                MemoryType.CONVERSATION_MEMORY
            ) or stores.get(MemoryType.CONVERSATION_MEMORY.value)
            if conversation_store is not None:
                rows = (
                    conversation_store.retrieve_conversation_history_ordered_by_timestamp(
                        memory_id=memory_id_value, limit=None
                    )
                    or []
                )
                history = [row for row in rows if isinstance(row, dict)]
        except Exception as exc:
            logger.debug(
                "Conversation-store fallback failed for memory '%s': %s",
                memory_id_value,
                exc,
            )

    history.sort(key=lambda row: _extract_message_timestamp(row) or 0.0)
    if limit and limit > 0:
        return history[-limit:]
    return history


def _load_last_run_map_from_conversation_docs(
    agent_ids: Optional[Set[str]] = None,
) -> Dict[str, float]:
    """Fallback last-run map built directly from conversation documents."""
    provider = _state.get("provider")
    if not provider:
        return {}

    try:
        from ..enums.memory_type import MemoryType

        docs = provider.list_all(MemoryType.CONVERSATION_MEMORY) or []
    except Exception as exc:
        logger.debug("Failed to load conversation docs for last-run fallback: %s", exc)
        return {}

    last_run_by_agent: Dict[str, float] = {}
    for doc in docs:
        if not isinstance(doc, dict):
            continue

        agent_id = _to_text(doc.get("agent_id") or doc.get("agentId")).strip()
        if not agent_id:
            continue
        if agent_ids is not None and agent_id not in agent_ids:
            continue

        timestamp = _extract_message_timestamp(doc)
        if timestamp is None:
            continue
        if timestamp > last_run_by_agent.get(agent_id, 0.0):
            last_run_by_agent[agent_id] = timestamp

    return last_run_by_agent


def _load_agent_last_run_map(agents: Optional[List[Any]] = None) -> Dict[str, float]:
    """Build a map of last run timestamp per agent."""
    provider = _state.get("provider")
    if not provider:
        return {}

    if agents is None:
        try:
            agents = provider.list_memagents()
        except Exception as exc:
            logger.debug("Failed to list agents while loading last-run map: %s", exc)
            return {}

    known_agent_ids: Set[str] = set()
    last_run_by_agent: Dict[str, float] = {}
    for agent in agents or []:
        agent_id = _extract_agent_identifier(agent)
        if not agent_id:
            continue
        known_agent_ids.add(agent_id)

        latest_timestamp = 0.0
        for memory_id in _extract_agent_memory_ids(agent):
            history = _retrieve_conversation_history(memory_id=memory_id, limit=None)

            for message in history:
                timestamp = _extract_message_timestamp(message)
                if timestamp and timestamp > latest_timestamp:
                    latest_timestamp = timestamp

        if latest_timestamp > 0:
            last_run_by_agent[agent_id] = latest_timestamp

    # Historical fallback: if memory_ids are sparse/missing, infer from conversation docs.
    if known_agent_ids and len(last_run_by_agent) < len(known_agent_ids):
        fallback_map = _load_last_run_map_from_conversation_docs(known_agent_ids)
        for agent_id, timestamp in fallback_map.items():
            if timestamp > last_run_by_agent.get(agent_id, 0.0):
                last_run_by_agent[agent_id] = timestamp

    return last_run_by_agent


def _build_agent_threads_map(agents: List[Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Build per-agent thread metadata for agents page UI controls."""
    thread_rows_by_agent: Dict[str, List[Dict[str, Any]]] = {}
    for agent in agents or []:
        agent_id = _extract_agent_identifier(agent)
        if not agent_id:
            continue
        thread_rows_by_agent[agent_id] = _build_agent_threads(agent)
    return thread_rows_by_agent


def _build_agent_recent_messages(
    agents: List[Any],
    thread_rows_by_agent: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    per_agent_limit: int = 5,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, str]]:
    """Collect latest-thread chat previews per agent for the agents page."""
    provider = _state.get("provider")
    if not provider:
        return {}, {}

    previews: Dict[str, List[Dict[str, Any]]] = {}
    default_memory_by_agent: Dict[str, str] = {}
    for agent in agents or []:
        agent_id = _extract_agent_identifier(agent)
        if not agent_id:
            continue

        selected_memory_id = ""
        if thread_rows_by_agent:
            rows = thread_rows_by_agent.get(agent_id, [])
            if rows:
                selected_memory_id = _to_text(rows[0].get("memory_id")).strip()

        if not selected_memory_id:
            memory_ids = _extract_agent_memory_ids(agent)
            if memory_ids:
                selected_memory_id = memory_ids[0]

        if not selected_memory_id:
            continue

        default_memory_by_agent[agent_id] = selected_memory_id
        history = _retrieve_conversation_history(
            memory_id=selected_memory_id, limit=120
        )
        if not history:
            continue

        recent: List[Dict[str, Any]] = []
        for message in history:
            role = _to_text(message.get("role")).strip().lower()
            if role not in {"user", "assistant"}:
                continue
            content = _to_text(
                message.get("content") or message.get("text", "")
            ).strip()
            if not content:
                continue
            if len(content) > 280:
                content = f"{content[:277]}..."
            recent.append({"role": role, "content": content})

        if recent:
            previews[agent_id] = recent[-max(1, per_agent_limit) :]

    return previews, default_memory_by_agent


def _toolbox_doc_key(doc: Dict[str, Any]) -> Optional[str]:
    """Build a stable dedupe key for toolbox tool docs."""
    if not isinstance(doc, dict):
        return None

    tool_id = _to_text(
        doc.get("tool_id") or doc.get("toolId") or doc.get("_id")
    ).strip()
    name = _to_text(doc.get("name")).strip()
    signature = _to_text(doc.get("signature")).strip()

    if tool_id:
        return f"id:{tool_id}"
    if name or signature:
        return f"{name}|{signature}"
    return None


def _is_toolbox_tool_doc(doc: Dict[str, Any]) -> bool:
    """Return True when a toolbox document represents an executable tool."""
    if not isinstance(doc, dict):
        return False

    tool_type = _to_text(doc.get("tool_type") or doc.get("type")).strip().lower()
    name = _to_text(doc.get("name")).strip()

    if tool_type == "mcp_server_config":
        return False
    if name.startswith("mcp::"):
        return False
    return bool(_toolbox_doc_key(doc))


def _load_toolbox_documents() -> List[Dict[str, Any]]:
    """Load toolbox documents for tool counting."""
    provider = _state.get("provider")
    if not provider:
        return []

    try:
        from ..enums.memory_type import MemoryType

        docs = provider.list_all(MemoryType.TOOLBOX) or []
    except Exception as exc:
        logger.debug("Failed to list toolbox docs: %s", exc)
        return []

    return [doc for doc in docs if isinstance(doc, dict)]


def _count_runtime_tools_for_agent(agent_id: str) -> int:
    """Best-effort runtime tool count by loading the agent tool manager."""
    provider = _state.get("provider")
    agent_id_value = _to_text(agent_id).strip()
    if not provider or not agent_id_value:
        return 0

    try:
        from ..memagent import MemAgent

        agent_instance = MemAgent.load(agent_id_value, memory_provider=provider)
        tool_manager = getattr(agent_instance, "tool_manager", None)
        if not tool_manager:
            return 0
        tool_names = tool_manager.list_tools() or []
        return len(tool_names)
    except Exception as exc:
        logger.debug(
            "Could not compute runtime tool count for %s: %s",
            agent_id_value,
            exc,
        )
        return 0


def _build_agent_tool_count_map(agents: List[Any]) -> Dict[str, int]:
    """Build per-agent tool counts using embedded tool metadata plus toolbox docs."""
    if not agents:
        return {}

    provider = _state.get("provider")
    toolbox_docs = [
        doc for doc in _load_toolbox_documents() if _is_toolbox_tool_doc(doc)
    ]

    docs_by_agent: Dict[str, List[Dict[str, Any]]] = {}
    docs_by_memory: Dict[str, List[Dict[str, Any]]] = {}
    for doc in toolbox_docs:
        agent_id = _to_text(doc.get("agent_id") or doc.get("agentId")).strip()
        memory_id = _to_text(doc.get("memory_id") or doc.get("memoryId")).strip()

        if agent_id:
            docs_by_agent.setdefault(agent_id, []).append(doc)
        if memory_id:
            docs_by_memory.setdefault(memory_id, []).append(doc)

    counts: Dict[str, int] = {}
    for agent in agents:
        agent_id = _extract_agent_identifier(agent)
        if not agent_id:
            continue

        seen_keys: Set[str] = set()
        memory_ids = _extract_agent_memory_ids(agent)

        # Embedded tools (already present on the memagent document).
        for tool_meta in _extract_agent_tools(agent):
            tool_name = _to_text(tool_meta.get("name")).strip()
            if tool_name:
                seen_keys.add(f"name:{tool_name}")

        # Provider-specific retrieval (Oracle) can surface tool associations
        # that may not be obvious from raw toolbox docs.
        if provider and hasattr(provider, "retrieve_tools_for_agent"):
            try:
                if isinstance(agent, dict):
                    tool_access = (
                        _to_text(
                            agent.get("tool_access") or agent.get("toolAccess")
                        ).strip()
                        or "private"
                    )
                else:
                    tool_access = (
                        _to_text(getattr(agent, "tool_access", None)).strip()
                        or "private"
                    )
                provider_tools = (
                    provider.retrieve_tools_for_agent(
                        agent_id=agent_id,
                        tool_access=tool_access,
                        query=None,
                        top_k=200,
                    )
                    or []
                )
                for doc in provider_tools:
                    if isinstance(doc, dict) and _is_toolbox_tool_doc(doc):
                        key = _toolbox_doc_key(doc)
                        if key:
                            seen_keys.add(key)
            except Exception as exc:
                logger.debug(
                    "Provider-specific tool retrieval failed for %s: %s", agent_id, exc
                )

        # Toolbox docs linked directly by agent_id.
        for doc in docs_by_agent.get(agent_id, []):
            key = _toolbox_doc_key(doc)
            if key:
                seen_keys.add(key)

        # Toolbox docs linked through memory IDs.
        for memory_id in memory_ids:
            for doc in docs_by_memory.get(memory_id, []):
                key = _toolbox_doc_key(doc)
                if key:
                    seen_keys.add(key)

        runtime_count = _count_runtime_tools_for_agent(agent_id)
        counts[agent_id] = max(len(seen_keys), runtime_count)

    return counts


def _agent_last_run_timestamp(
    agent: Any, last_run_by_agent: Optional[Dict[str, float]] = None
) -> float:
    """Resolve an agent's latest run timestamp."""
    agent_id = _extract_agent_identifier(agent)
    if agent_id and last_run_by_agent and agent_id in last_run_by_agent:
        return last_run_by_agent[agent_id]
    return 0.0


def _sort_agents_by_last_run_desc(
    agents: List[Any], last_run_by_agent: Optional[Dict[str, float]] = None
) -> List[Any]:
    """Sort agents by last run (newest first), then by creation time."""
    if not agents:
        return []

    last_run_map = last_run_by_agent or _load_agent_last_run_map(agents)
    created_by_agent = _load_memagent_created_at_map()
    indexed = [
        (
            _agent_last_run_timestamp(agent, last_run_map),
            _agent_created_timestamp(agent, created_by_agent),
            idx,
            agent,
        )
        for idx, agent in enumerate(agents)
    ]
    indexed.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [item[3] for item in indexed]


def _build_agent_nav_items(
    active_agent_id: Optional[str] = None,
    agents: Optional[List[Any]] = None,
    last_run_by_agent: Optional[Dict[str, float]] = None,
    limit: int = RECENT_NAV_AGENT_LIMIT,
) -> List[Dict[str, str]]:
    """Build recent agent list for sidebar navigation."""
    if not _state["provider"]:
        return []

    if agents is None:
        try:
            agents = _state["provider"].list_memagents()
        except Exception as exc:
            logger.error(f"Failed to list agents for navigation: {exc}")
            return []

    ordered_agents = _sort_agents_by_last_run_desc(
        agents, last_run_by_agent=last_run_by_agent
    )
    if limit > 0:
        ordered_agents = ordered_agents[:limit]

    items: List[Dict[str, str]] = []
    for agent in ordered_agents:
        agent_id = _extract_agent_identifier(agent)
        if not agent_id:
            continue
        items.append(
            {
                "agent_id": agent_id,
                "name": _extract_agent_persona_name(agent),
                "short_id": agent_id[:8],
                "is_active": agent_id == active_agent_id,
            }
        )

    return items


def _build_settings_sections() -> List[Dict[str, Any]]:
    """Attach env status to settings sections for rendering."""
    sections: List[Dict[str, Any]] = []
    for section in SETTINGS_SECTIONS:
        fields = []
        for field in section["fields"]:
            env_key = field["env"]
            current_value = _to_text(os.environ.get(env_key, "")).strip()
            if not current_value:
                if env_key == "MEMORIZZ_DEFAULT_LLM_PROVIDER":
                    current_value = _get_default_llm_provider()
                elif env_key == "MEMORIZZ_DEFAULT_LLM_MODEL":
                    current_value = _get_default_llm_model(_get_default_llm_provider())
                else:
                    current_value = _to_text(field.get("default_value", "")).strip()
            is_set = bool(os.environ.get(env_key))
            if field.get("field_type") == "checkbox":
                checked = _parse_bool(current_value)
                current_value = "1" if checked else "0"
                is_set = True
            enriched = {
                **field,
                "is_set": is_set,
                "current_value": current_value,
            }
            if field.get("field_type") == "checkbox":
                enriched["checkbox_checked"] = current_value == "1"
            fields.append(enriched)
        enriched_section = {**section, "fields": fields}
        if section.get("title") == "Sandbox Providers":
            enriched_section["provider_checks"] = _build_sandbox_provider_checks()
        sections.append(enriched_section)
    return sections


def _normalize_internet_provider_name(value: Any) -> str:
    """Normalize internet provider values to a stable lowercase name."""
    if isinstance(value, dict):
        value = value.get("provider") or value.get("name")
    return _to_text(value).strip().lower()


def _build_internet_provider_config(
    provider_name: Optional[str], base_config: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """Build internet provider config with optional default API key fallback."""
    normalized_provider = _normalize_internet_provider_name(provider_name)
    if not normalized_provider:
        return None

    config: Dict[str, Any] = {}
    if isinstance(base_config, dict):
        for key, value in base_config.items():
            if key == "provider":
                continue
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            config[key] = value

    if "api_key" not in config:
        default_key = _to_text(
            os.environ.get("MEMORIZZ_DEFAULT_INTERNET_PROVIDER_API_KEY", "")
        ).strip()
        if default_key:
            config["api_key"] = default_key

    return config or None


def _validate_internet_provider_choice(
    internet_provider: Any,
    internet_config: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Validate internet provider configuration and runtime prerequisites.

    Returns:
        ``None`` when valid, otherwise an actionable error message.
    """
    provider_name = _normalize_internet_provider_name(internet_provider)
    if not provider_name:
        return None

    try:
        from ..internet_access import create_internet_access_provider

        config = _build_internet_provider_config(provider_name, internet_config) or {}
        provider = create_internet_access_provider(provider_name, config)
        if not provider:
            return (
                f"Unknown internet provider '{provider_name}'. "
                "Supported providers: tavily, firecrawl, offline."
            )
        try:
            provider.close()
        except Exception:
            pass
        return None
    except Exception as exc:
        message = _to_text(exc).strip()
        if message:
            return message
        return f"Failed to initialize internet provider '{provider_name}'."


def _normalize_skills_marketplace_provider_name(value: Any) -> str:
    """Normalize skills marketplace provider values to a stable lowercase name."""
    if isinstance(value, dict):
        value = value.get("provider") or value.get("name")
    return _to_text(value).strip().lower()


def _build_skills_marketplace_provider_config(
    provider_name: Optional[str], base_config: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """Build skills marketplace config with API key fallback from Settings env."""
    normalized_provider = _normalize_skills_marketplace_provider_name(provider_name)
    if not normalized_provider:
        return None

    config: Dict[str, Any] = {}
    if isinstance(base_config, dict):
        for key, value in base_config.items():
            if key == "provider":
                continue
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            config[key] = value

    if "api_key" not in config:
        default_key = _to_text(os.environ.get("SKILLSMP_API_KEY", "")).strip()
        if default_key:
            config["api_key"] = default_key

    if "base_url" not in config:
        config["base_url"] = "https://skillsmp.com"

    return config or None


def _validate_skills_marketplace_provider_choice(
    skills_marketplace_provider: Any,
    skills_marketplace_config: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Validate skills marketplace provider configuration.

    Returns:
        ``None`` when valid, otherwise an actionable error message.
    """
    provider_name = _normalize_skills_marketplace_provider_name(
        skills_marketplace_provider
    )
    if not provider_name:
        return None

    if provider_name != "skillsmp":
        return (
            f"Unknown skills marketplace provider '{provider_name}'. "
            "Supported providers: skillsmp."
        )

    config = (
        _build_skills_marketplace_provider_config(
            provider_name, skills_marketplace_config
        )
        or {}
    )
    api_key = _to_text(config.get("api_key", "")).strip()
    if not api_key:
        return "Skills Marketplace requires an API key. Add `SKILLSMP_API_KEY` in Settings."

    return None


def _validate_sandbox_provider_choice(
    sandbox_provider: Any,
) -> Optional[str]:
    """
    Validate sandbox provider configuration and runtime prerequisites.

    Returns:
        ``None`` when valid, otherwise an actionable error message.
    """
    if not sandbox_provider:
        return None

    resolved_provider = _resolve_sandbox_provider_config(sandbox_provider)
    provider_name = ""
    provider_config: Dict[str, Any] = {}

    if isinstance(resolved_provider, dict):
        provider_name = _to_text(resolved_provider.get("provider")).strip().lower()
        provider_config = {
            key: value for key, value in resolved_provider.items() if key != "provider"
        }
    else:
        provider_name = _to_text(resolved_provider).strip().lower()

    if not provider_name:
        return None

    if provider_name == "graalpy":
        mode = _to_text(provider_config.get("mode", "")).strip().lower()
        wrapper_jar = _to_text(provider_config.get("java_wrapper_jar", "")).strip()
        if mode == "java_wrapper" and not wrapper_jar:
            return (
                "GraalPy internet access is disabled, but `GRAALPY_JAVA_WRAPPER_JAR` "
                "is not set. Add the wrapper JAR path in Settings or re-enable internet."
            )

    try:
        from ..sandbox.base import create_sandbox_provider

        provider = create_sandbox_provider(provider_name, provider_config)
        if not provider:
            return (
                f"Unknown sandbox provider '{provider_name}'. "
                "Supported providers: e2b, daytona, graalpy."
            )
        return None
    except Exception as exc:
        message = _to_text(exc).strip()
        if message:
            return message
        return f"Failed to initialize sandbox provider '{provider_name}'."


def _build_sandbox_provider_checks() -> List[Dict[str, Any]]:
    """Build readiness checks for sandbox providers shown in Settings."""
    checks: List[Dict[str, Any]] = []

    graalpy_error = _validate_sandbox_provider_choice("graalpy")
    checks.append(
        {
            "provider": "graalpy",
            "label": "GraalPy (Local)",
            "ready": graalpy_error is None,
            "message": (
                "GraalPy executable detected and ready."
                if graalpy_error is None
                else graalpy_error
            ),
        }
    )

    return checks


def _graalpy_internet_access_enabled() -> bool:
    """Return whether GraalPy should run with internet access."""
    raw_value = _to_text(os.environ.get("MEMORIZZ_GRAALPY_INTERNET_ACCESS", "")).strip()
    if not raw_value:
        return True
    return _parse_bool(raw_value)


def _build_graalpy_default_sandbox_config() -> Dict[str, Any]:
    """
    Build GraalPy provider config from UI settings.

    - Internet enabled  -> subprocess mode
    - Internet disabled -> java_wrapper mode with UNTRUSTED policy
    """
    config: Dict[str, Any] = {"provider": "graalpy"}

    graalpy_path = _to_text(os.environ.get("GRAALPY_PATH", "")).strip()
    if graalpy_path:
        config["graalpy_path"] = graalpy_path

    if _graalpy_internet_access_enabled():
        config["mode"] = "subprocess"
        return config

    config["mode"] = "java_wrapper"
    config["sandbox_policy"] = "UNTRUSTED"
    wrapper_jar = _to_text(os.environ.get("GRAALPY_JAVA_WRAPPER_JAR", "")).strip()
    if wrapper_jar:
        config["java_wrapper_jar"] = wrapper_jar
    return config


def _resolve_sandbox_provider_config(sandbox_provider: Any) -> Any:
    """
    Resolve sandbox provider input into the effective provider config.

    For GraalPy string inputs, this applies Settings-driven defaults so runtime
    behavior matches the Settings page toggles.
    """
    if not sandbox_provider:
        return None
    if isinstance(sandbox_provider, dict):
        return sandbox_provider

    provider_name = _to_text(sandbox_provider).strip().lower()
    if provider_name != "graalpy":
        return provider_name
    return _build_graalpy_default_sandbox_config()


def _apply_env_updates(updates: Dict[str, str]) -> Optional[str]:
    """Apply updates to process env and persist to .env when possible."""
    for key, value in updates.items():
        os.environ[key] = value

    try:
        _update_env_file(ENV_FILE_PATH, updates)
    except Exception as exc:
        logger.warning(f"Failed to update .env: {exc}")
        return str(exc)
    return None


def _update_env_file(env_path: Path, updates: Dict[str, str]) -> None:
    """Update or append environment variables in a .env file."""
    if env_path.exists():
        lines = env_path.read_text().splitlines()
    else:
        lines = []

    updated_lines: List[str] = []
    seen = set()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            updated_lines.append(line)
            continue
        key, _value = line.split("=", 1)
        key = key.strip()
        if key in updates:
            updated_lines.append(f"{key}={_format_env_value(updates[key])}")
            seen.add(key)
        else:
            updated_lines.append(line)

    for key, value in updates.items():
        if key in seen:
            continue
        if updated_lines and updated_lines[-1].strip():
            updated_lines.append("")
        updated_lines.append(f"{key}={_format_env_value(value)}")

    env_path.write_text("\n".join(updated_lines) + "\n")


def _format_env_value(value: str) -> str:
    """Format env var values for .env files."""
    if value == "":
        return ""
    if any(ch.isspace() for ch in value) or "#" in value or "=" in value:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def _get_memory_stats() -> Dict[str, int]:
    """Get counts for each memory type."""
    from ..enums.memory_type import MemoryType

    stats = {}
    if not _state["provider"]:
        return stats

    for mem_type in MemoryType:
        try:
            items = _state["provider"].list_all(mem_type)
            stats[mem_type.value] = len(items) if items else 0
        except Exception:
            stats[mem_type.value] = 0

    return stats


def _serialize_agent(agent) -> Dict[str, Any]:
    """Convert agent object to serializable dict."""
    if hasattr(agent, "model_dump"):
        data = agent.model_dump()
    elif hasattr(agent, "__dict__"):
        data = dict(agent.__dict__)
    else:
        data = {"agent_id": str(agent)}

    # Handle persona serialization
    if "persona" in data and data["persona"] and hasattr(data["persona"], "to_dict"):
        data["persona"] = data["persona"].to_dict()

    # Remove non-serializable items
    data.pop("memory_provider", None)
    data.pop("model", None)

    return data


def _load_agent_detail(agent_id: str) -> Dict[str, Any]:
    """Load agent detail information for rendering."""
    agent = None
    context_window: List[Dict[str, Any]] = []
    error = None
    token_stats = None

    try:
        agent = _state["provider"].retrieve_memagent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        memory_ids = getattr(agent, "memory_ids", []) or []
        for memory_id in memory_ids[:3]:
            try:
                history = _state[
                    "provider"
                ].retrieve_conversation_history_ordered_by_timestamp(
                    memory_id=memory_id, limit=50
                )
                context_window.extend(history or [])
            except Exception as e:
                logger.warning(f"Failed to get history for {memory_id}: {e}")

        context_window.sort(
            key=lambda x: x.get("timestamp", 0)
            if isinstance(x.get("timestamp"), (int, float))
            else 0
        )

        token_stats = _build_token_stats(agent, context_window)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load agent {agent_id}: {e}")
        error = str(e)

    return {
        "agent": agent,
        "context_window": context_window[-30:],
        "token_stats": token_stats,
        "error": error,
    }


def _load_thread_messages(
    memory_id: str, limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load and sort conversation messages for a single thread memory_id."""
    provider = _state.get("provider")
    normalized_memory_id = _to_text(memory_id).strip()
    if not provider or not normalized_memory_id:
        return []

    try:
        history = (
            provider.retrieve_conversation_history_ordered_by_timestamp(
                memory_id=normalized_memory_id, limit=limit
            )
            or []
        )
    except Exception as exc:
        logger.debug(
            "Failed to load conversation history for memory_id %s: %s",
            normalized_memory_id,
            exc,
        )
        return []

    history.sort(key=lambda item: _coerce_timestamp(item.get("timestamp")) or 0.0)
    return history


def _serialize_toolbox_memory_item(document: Dict[str, Any]) -> Dict[str, str]:
    """Convert a toolbox document into a compact, UI-safe payload."""
    tool_identifier = _to_text(
        document.get("tool_id")
        or document.get("toolId")
        or document.get("_id")
        or document.get("id")
    ).strip()
    tool_name = _to_text(
        document.get("name")
        or document.get("tool_name")
        or tool_identifier
        or "unnamed_toolbox_item"
    ).strip()
    tool_type = _to_text(document.get("tool_type") or document.get("type")).strip()
    description = _to_text(
        document.get("description") or document.get("docstring") or ""
    ).strip()
    signature = _to_text(document.get("signature") or "").strip()
    timestamp = _to_text(
        document.get("timestamp")
        or document.get("updated_at")
        or document.get("created_at")
        or document.get("createdAt")
        or ""
    ).strip()

    value_preview = ""
    if "parameters" in document and document.get("parameters") is not None:
        try:
            value_preview = json.dumps(document.get("parameters"), ensure_ascii=False)
        except Exception:
            value_preview = _to_text(document.get("parameters"))
    elif document.get("content") is not None:
        value_preview = _to_text(document.get("content"))

    return {
        "id": tool_identifier,
        "name": tool_name or "unnamed_toolbox_item",
        "tool_type": tool_type or "toolbox_item",
        "description": description,
        "signature": signature,
        "value_preview": value_preview,
        "timestamp": timestamp,
        "memory_id": _to_text(
            document.get("memory_id") or document.get("memoryId")
        ).strip(),
    }


def _normalize_workflow_steps(steps: Any) -> Dict[str, Any]:
    """Return workflow steps as a dictionary when possible."""
    if isinstance(steps, dict):
        return steps
    if isinstance(steps, list):
        normalized: Dict[str, Any] = {}
        for index, entry in enumerate(steps, start=1):
            key = f"Step {index}"
            normalized[key] = entry
        return normalized
    if isinstance(steps, str):
        text = steps.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except Exception:
            return {}
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            normalized = {}
            for index, entry in enumerate(parsed, start=1):
                normalized[f"Step {index}"] = entry
            return normalized
    return {}


def _serialize_workflow_memory_item(document: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a workflow document into a compact, UI-safe payload."""
    workflow_identifier = _to_text(
        document.get("workflow_id")
        or document.get("workflowId")
        or document.get("_id")
        or document.get("id")
    ).strip()
    workflow_name = _to_text(
        document.get("name")
        or document.get("title")
        or workflow_identifier
        or "unnamed_workflow"
    ).strip()
    description = _to_text(document.get("description") or "").strip()
    user_query = _to_text(
        document.get("user_query") or document.get("userQuery") or ""
    ).strip()
    status = _to_text(document.get("status") or "").strip()
    outcome = _to_text(document.get("outcome") or "").strip()
    timestamp = _to_text(
        document.get("updated_at")
        or document.get("updatedAt")
        or document.get("created_at")
        or document.get("createdAt")
        or document.get("timestamp")
        or ""
    ).strip()

    steps = _normalize_workflow_steps(document.get("steps"))
    step_names = list(steps.keys())
    step_count = len(step_names)
    step_names_preview = step_names[:3]

    return {
        "id": workflow_identifier,
        "name": workflow_name or "unnamed_workflow",
        "description": description,
        "user_query": user_query,
        "status": status,
        "outcome": outcome,
        "timestamp": timestamp,
        "step_count": step_count,
        "step_names_preview": step_names_preview,
        "memory_id": _to_text(
            document.get("memory_id") or document.get("memoryId")
        ).strip(),
    }


def _load_thread_toolbox_memory(
    agent_id: str, memory_id: str, limit: Optional[int] = None
) -> List[Dict[str, str]]:
    """Load toolbox-memory rows relevant to the active agent thread."""
    provider = _state.get("provider")
    normalized_agent_id = _to_text(agent_id).strip()
    normalized_memory_id = _to_text(memory_id).strip()
    if not provider or not normalized_agent_id:
        return []

    try:
        from ..enums.memory_type import MemoryType

        documents = provider.list_all(MemoryType.TOOLBOX) or []
    except Exception as exc:
        logger.debug(
            "Failed to load toolbox memory for agent %s thread %s: %s",
            normalized_agent_id,
            normalized_memory_id,
            exc,
        )
        return []

    filtered: List[Dict[str, str]] = []
    for doc in documents:
        if not isinstance(doc, dict):
            continue

        doc_agent_id = _to_text(doc.get("agent_id") or doc.get("agentId")).strip()
        if doc_agent_id and doc_agent_id != normalized_agent_id:
            continue

        doc_memory_id = _to_text(doc.get("memory_id") or doc.get("memoryId")).strip()
        if not doc_agent_id and not doc_memory_id:
            continue
        if normalized_memory_id:
            if doc_memory_id and doc_memory_id != normalized_memory_id:
                continue
        elif doc_memory_id:
            continue

        filtered.append(_serialize_toolbox_memory_item(doc))

    filtered.sort(
        key=lambda item: _coerce_timestamp(item.get("timestamp")) or 0.0, reverse=True
    )
    if limit and limit > 0:
        return filtered[:limit]
    return filtered


def _load_thread_workflow_memory(
    agent_id: str, memory_id: str, limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load workflow-memory rows relevant to the active agent thread."""
    provider = _state.get("provider")
    normalized_agent_id = _to_text(agent_id).strip()
    normalized_memory_id = _to_text(memory_id).strip()
    if not provider or not normalized_agent_id:
        return []

    try:
        from ..enums.memory_type import MemoryType

        documents = provider.list_all(MemoryType.WORKFLOW_MEMORY) or []
    except Exception as exc:
        logger.debug(
            "Failed to load workflow memory for agent %s thread %s: %s",
            normalized_agent_id,
            normalized_memory_id,
            exc,
        )
        return []

    filtered: List[Dict[str, Any]] = []
    for doc in documents:
        if not isinstance(doc, dict):
            continue

        doc_agent_id = _to_text(doc.get("agent_id") or doc.get("agentId")).strip()
        if doc_agent_id and doc_agent_id != normalized_agent_id:
            continue

        doc_memory_id = _to_text(doc.get("memory_id") or doc.get("memoryId")).strip()
        if not doc_agent_id and not doc_memory_id:
            continue
        if normalized_memory_id:
            if doc_memory_id and doc_memory_id != normalized_memory_id:
                continue
            if not doc_memory_id:
                continue
        elif doc_memory_id:
            continue

        filtered.append(_serialize_workflow_memory_item(doc))

    filtered.sort(
        key=lambda item: _coerce_timestamp(item.get("timestamp")) or 0.0, reverse=True
    )
    if limit and limit > 0:
        return filtered[:limit]
    return filtered


def _serialize_entity_memory_item(document: Dict[str, Any]) -> Dict[str, str]:
    """Convert an entity-memory document into a compact, UI-safe payload."""
    entity_identifier = _to_text(
        document.get("entity_id")
        or document.get("entityId")
        or document.get("_id")
        or document.get("id")
    ).strip()
    entity_name = _to_text(
        document.get("name")
        or document.get("entity_name")
        or entity_identifier
        or "unnamed_entity"
    ).strip()
    entity_type = _to_text(
        document.get("entity_type") or document.get("entityType") or ""
    ).strip()
    timestamp = _to_text(
        document.get("updated_at")
        or document.get("updatedAt")
        or document.get("created_at")
        or document.get("createdAt")
        or ""
    ).strip()

    raw_attributes = document.get("attributes")
    attributes_preview = ""
    attribute_pairs: List[str] = []
    attributes_obj = raw_attributes
    if isinstance(attributes_obj, str):
        try:
            parsed = json.loads(attributes_obj)
        except Exception:
            parsed = attributes_obj
        attributes_obj = parsed

    if isinstance(attributes_obj, dict):
        for key, value in attributes_obj.items():
            key_text = _to_text(key).strip()
            if not key_text:
                continue
            value_text = _to_text(value).strip()
            attribute_pairs.append(f"{key_text}: {value_text}")
    elif isinstance(attributes_obj, list):
        for entry in attributes_obj:
            if not isinstance(entry, dict):
                continue
            key_text = _to_text(entry.get("name") or entry.get("key")).strip()
            if not key_text:
                continue
            value_text = _to_text(entry.get("value")).strip()
            attribute_pairs.append(f"{key_text}: {value_text}")

    if attribute_pairs:
        attributes_preview = ", ".join(attribute_pairs[:6])
    if len(attributes_preview) > 260:
        attributes_preview = f"{attributes_preview[:257]}..."

    return {
        "id": entity_identifier,
        "name": entity_name or "unnamed_entity",
        "entity_type": entity_type,
        "timestamp": timestamp,
        "attributes_preview": attributes_preview,
        "memory_id": _to_text(
            document.get("memory_id") or document.get("memoryId")
        ).strip(),
    }


def _load_thread_entity_memory(
    agent_id: str, memory_id: str, limit: Optional[int] = None
) -> List[Dict[str, str]]:
    """Load entity-memory rows relevant to the active agent thread."""
    provider = _state.get("provider")
    normalized_agent_id = _to_text(agent_id).strip()
    normalized_memory_id = _to_text(memory_id).strip()
    if not provider or not normalized_agent_id:
        return []

    try:
        from ..enums.memory_type import MemoryType

        documents = provider.list_all(MemoryType.ENTITY_MEMORY) or []
    except Exception as exc:
        logger.debug(
            "Failed to load entity memory for agent %s thread %s: %s",
            normalized_agent_id,
            normalized_memory_id,
            exc,
        )
        return []

    scoped: List[Dict[str, str]] = []
    fallback_agent_scoped: List[Dict[str, str]] = []
    for doc in documents:
        if not isinstance(doc, dict):
            continue

        doc_agent_id = _to_text(doc.get("agent_id") or doc.get("agentId")).strip()
        if doc_agent_id and doc_agent_id != normalized_agent_id:
            continue

        doc_memory_id = _to_text(doc.get("memory_id") or doc.get("memoryId")).strip()
        if not doc_agent_id and not doc_memory_id:
            continue

        serialized = _serialize_entity_memory_item(doc)
        if normalized_memory_id:
            if doc_memory_id == normalized_memory_id:
                scoped.append(serialized)
                continue
            if not doc_memory_id and doc_agent_id == normalized_agent_id:
                fallback_agent_scoped.append(serialized)
            continue

        # No explicit thread selected: show all agent-scoped entity rows.
        if doc_memory_id or doc_agent_id == normalized_agent_id:
            scoped.append(serialized)

    filtered = scoped if scoped else fallback_agent_scoped

    filtered.sort(
        key=lambda item: _coerce_timestamp(item.get("timestamp")) or 0.0, reverse=True
    )
    if limit and limit > 0:
        return filtered[:limit]
    return filtered


def _serialize_summary_memory_item(document: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a summaries-memory document into a compact, UI-safe payload."""
    summary_identifier = _to_text(
        document.get("summary_id")
        or document.get("summaryId")
        or document.get("_id")
        or document.get("id")
    ).strip()
    summary_type = _to_text(
        document.get("summary_type") or document.get("summaryType") or "summary"
    ).strip()
    timestamp = _to_text(
        document.get("created_at")
        or document.get("createdAt")
        or document.get("updated_at")
        or document.get("updatedAt")
        or ""
    ).strip()
    content = _to_text(document.get("content") or "").strip()
    if len(content) > 520:
        content = f"{content[:517]}..."

    return {
        "id": summary_identifier,
        "summary_type": summary_type or "summary",
        "timestamp": timestamp,
        "content": content,
        "memory_id": _to_text(
            document.get("memory_id") or document.get("memoryId")
        ).strip(),
    }


def _load_thread_summary_memory(
    agent_id: str, memory_id: str, limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load summary-memory rows relevant to the active agent thread."""
    provider = _state.get("provider")
    normalized_agent_id = _to_text(agent_id).strip()
    normalized_memory_id = _to_text(memory_id).strip()
    if not provider or not normalized_agent_id:
        return []

    try:
        from ..enums.memory_type import MemoryType

        documents = provider.list_all(MemoryType.SUMMARIES) or []
    except Exception as exc:
        logger.debug(
            "Failed to load summary memory for agent %s thread %s: %s",
            normalized_agent_id,
            normalized_memory_id,
            exc,
        )
        return []

    filtered: List[Dict[str, Any]] = []
    for doc in documents:
        if not isinstance(doc, dict):
            continue

        doc_agent_id = _to_text(doc.get("agent_id") or doc.get("agentId")).strip()
        if doc_agent_id and doc_agent_id != normalized_agent_id:
            continue

        doc_memory_id = _to_text(doc.get("memory_id") or doc.get("memoryId")).strip()
        if not doc_agent_id and not doc_memory_id:
            continue
        if normalized_memory_id:
            if doc_memory_id and doc_memory_id != normalized_memory_id:
                continue
            if not doc_memory_id:
                continue
        elif doc_memory_id:
            continue

        filtered.append(_serialize_summary_memory_item(doc))

    filtered.sort(
        key=lambda item: _coerce_timestamp(item.get("timestamp")) or 0.0, reverse=True
    )
    if limit and limit > 0:
        return filtered[:limit]
    return filtered


def _serialize_thread_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Convert thread message payload into JSON-serializable structure."""
    if not isinstance(message, dict):
        return {
            "role": "system",
            "content": _to_text(message),
            "timestamp": "",
            "memory_id": "",
        }

    timestamp_value = message.get("timestamp")
    timestamp_text = _to_text(timestamp_value).strip() if timestamp_value else ""

    role = _to_text(message.get("role")).strip().lower() or "system"
    content = _to_text(message.get("content") or message.get("text", ""))
    serialized = {
        "role": role,
        "content": content,
        "timestamp": timestamp_text,
        "memory_id": _to_text(message.get("memory_id")).strip(),
    }

    if role == "tool":
        trace_events = _parse_trace_bundle_payload(content)
        if trace_events is not None:
            serialized["message_type"] = "trace_bundle"
            serialized["trace_events"] = trace_events
            serialized["content"] = f"Trace events ({len(trace_events)})"

    return serialized


def _parse_trace_bundle_payload(raw_content: Any) -> Optional[List[Dict[str, str]]]:
    """Decode persisted trace-bundle content from tool messages."""
    content = _to_text(raw_content).strip()
    if not content:
        return None

    try:
        payload = json.loads(content)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None
    if _to_text(payload.get("type")).strip().lower() != "trace_bundle":
        return None

    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        return []

    events: List[Dict[str, str]] = []
    for event in raw_events:
        if not isinstance(event, dict):
            continue

        trace_kind = _to_text(
            event.get("trace_kind") or event.get("kind") or "trace"
        ).strip()
        title = _to_text(event.get("title") or "Trace").strip()
        event_content = _to_text(
            event.get("content") or event.get("message") or ""
        ).strip()
        trace_id = _to_text(event.get("trace_id")).strip()

        if not title and not event_content:
            continue

        serialized_event = {
            "trace_kind": trace_kind.lower() or "trace",
            "title": title or "Trace",
            "content": event_content,
        }
        if trace_id:
            serialized_event["trace_id"] = trace_id
        events.append(serialized_event)

    return events


def _build_agent_threads(agent: Any) -> List[Dict[str, Any]]:
    """Build thread metadata from agent memory_ids for playground display."""
    if not _state.get("provider") or not agent:
        return []

    memory_ids = getattr(agent, "memory_ids", None) or []
    if not isinstance(memory_ids, list):
        return []

    seen: Set[str] = set()
    thread_rows: List[Dict[str, Any]] = []

    for raw_memory_id in memory_ids:
        memory_id = _to_text(raw_memory_id).strip()
        if not memory_id or memory_id in seen:
            continue
        seen.add(memory_id)

        history = _load_thread_messages(memory_id, limit=200)

        message_count = len(history)
        last_msg = history[-1] if history else {}
        last_role = _to_text(last_msg.get("role")).strip().lower() if last_msg else ""
        last_content = _to_text(last_msg.get("content") or last_msg.get("text", ""))
        if len(last_content) > 120:
            last_content = f"{last_content[:117]}..."

        last_ts_raw = last_msg.get("timestamp") if last_msg else None
        last_ts = _coerce_timestamp(last_ts_raw)
        if last_ts is not None:
            last_activity = datetime.fromtimestamp(last_ts).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        else:
            last_activity = _to_text(last_ts_raw).strip() if last_ts_raw else "—"

        thread_rows.append(
            {
                "memory_id": memory_id,
                "message_count": message_count,
                "last_role": last_role or "—",
                "last_content": last_content or "—",
                "last_activity": last_activity,
                "last_ts": last_ts or 0.0,
            }
        )

    thread_rows.sort(key=lambda item: item.get("last_ts", 0.0), reverse=True)
    return thread_rows


def _extract_agent_tools(agent) -> List[Dict[str, str]]:
    """Extract a simplified tools list from an agent for display in the UI."""
    if not agent:
        return []
    raw_tools = getattr(agent, "tools", None) or []
    result = []
    seen_names: Set[str] = set()
    for tool in raw_tools:
        if isinstance(tool, dict):
            name = _to_text(
                tool.get("name") or tool.get("function", {}).get("name", "Unknown")
            )
            desc = _to_text(
                tool.get("description")
                or tool.get("docstring", "")
                or tool.get("function", {}).get("description", "")
            )
            normalized_name = (name or "Unknown").strip()
            seen_names.add(normalized_name)
            result.append(
                {
                    "name": normalized_name,
                    "description": desc[:120] if desc else "",
                }
            )

    default_context_tools = [
        (
            "context_window_stats_tool",
            "Show the latest context-window token usage metrics.",
        ),
        (
            "list_summary_registry_tool",
            "List summaries generated automatically or on demand.",
        ),
        (
            "fetch_summary_tool",
            "Fetch a stored summary by summary_id.",
        ),
        (
            "autosummarize_conversation",
            "Generate conversation summaries now.",
        ),
    ]
    for tool_name, description in default_context_tools:
        if tool_name in seen_names:
            continue
        result.append({"name": tool_name, "description": description})
        seen_names.add(tool_name)

    sandbox_provider = getattr(agent, "sandbox_provider", None)
    if isinstance(sandbox_provider, dict):
        sandbox_provider = sandbox_provider.get("provider")
    sandbox_provider = _to_text(sandbox_provider).strip()
    if sandbox_provider:
        sandbox_tools = [
            (
                "execute_code",
                f"Execute code in the configured {sandbox_provider} sandbox.",
            ),
            ("sandbox_write_file", "Write a file inside the sandbox filesystem."),
            ("sandbox_read_file", "Read a file from the sandbox filesystem."),
        ]
        for tool_name, description in sandbox_tools:
            if tool_name in seen_names:
                continue
            result.append({"name": tool_name, "description": description})
            seen_names.add(tool_name)

    internet_provider = _normalize_internet_provider_name(
        getattr(agent, "internet_access_provider", None)
    )
    if internet_provider:
        internet_tools = [
            (
                "internet_search",
                f"Search the live internet via {internet_provider}.",
            ),
            (
                "open_web_page",
                "Fetch and summarize webpage content from a URL.",
            ),
        ]
        for tool_name, description in internet_tools:
            if tool_name in seen_names:
                continue
            result.append({"name": tool_name, "description": description})
            seen_names.add(tool_name)

    skills_marketplace_provider = _normalize_skills_marketplace_provider_name(
        getattr(agent, "skills_marketplace_provider", None)
    )
    if skills_marketplace_provider:
        tool_name = "skills_marketplace_search"
        if tool_name not in seen_names:
            result.append(
                {
                    "name": tool_name,
                    "description": (
                        "Search skills from the configured marketplace via sandbox execution."
                    ),
                }
            )
            seen_names.add(tool_name)

    if _agent_entity_memory_enabled(agent) and _provider_supports_entity_memory(
        _state.get("provider")
    ):
        entity_tools = [
            (
                "entity_memory_lookup",
                "Look up structured entity facts for the active thread.",
            ),
            (
                "entity_memory_upsert",
                "Create or update structured entity facts for the active thread.",
            ),
        ]
        for tool_name, description in entity_tools:
            if tool_name in seen_names:
                continue
            result.append({"name": tool_name, "description": description})
            seen_names.add(tool_name)
    return result


def _build_token_stats(
    agent,
    context_window: List[Dict[str, Any]],
    toolbox_memory: Optional[List[Dict[str, Any]]] = None,
    workflow_memory: Optional[List[Dict[str, Any]]] = None,
    entity_memory: Optional[List[Dict[str, Any]]] = None,
    summary_memory: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Estimate token usage based on current agent data and loaded context data."""

    def estimate_tokens(text: str) -> int:
        return len(text.split()) if text else 0

    def estimate_collection_tokens(items: Optional[List[Dict[str, Any]]]) -> int:
        if not items:
            return 0
        total = 0
        for item in items:
            if isinstance(item, dict):
                parts: List[str] = []
                for value in item.values():
                    if value is None:
                        continue
                    if isinstance(value, (dict, list)):
                        try:
                            parts.append(json.dumps(value, ensure_ascii=False))
                        except Exception:
                            parts.append(_to_text(value))
                    else:
                        parts.append(_to_text(value))
                total += estimate_tokens(" ".join(parts))
            elif isinstance(item, list):
                total += estimate_tokens(" ".join(_to_text(v) for v in item))
            else:
                total += estimate_tokens(_to_text(item))
        return total

    instruction_tokens = estimate_tokens(str(getattr(agent, "instruction", "") or ""))

    persona = getattr(agent, "persona", None)
    persona_text = ""
    if persona:
        if isinstance(persona, dict):
            persona_text = (
                f"{persona.get('name', '')} {persona.get('role', '')} "
                f"{persona.get('background', '')} {persona.get('goals', '')}"
            )
        else:
            persona_text = (
                f"{getattr(persona, 'name', '')} {getattr(persona, 'role', '')} "
                f"{getattr(persona, 'background', '')} {getattr(persona, 'goals', '')}"
            )
    persona_tokens = estimate_tokens(persona_text)

    tools = getattr(agent, "tools", None) or []
    tools_text = " ".join([str(t) for t in tools]) if tools else ""
    tools_tokens = estimate_tokens(tools_text)

    context_window_size = getattr(agent, "context_window_tokens", None) or 128000

    history_for_estimate = list(context_window or [])
    if agent and history_for_estimate and hasattr(agent, "_prepare_history_messages"):
        try:
            system_prompt = ""
            if hasattr(agent, "_build_system_prompt"):
                system_prompt = _to_text(agent._build_system_prompt())
            prepared = agent._prepare_history_messages(
                history_for_estimate, system_prompt, ""
            )
            if isinstance(prepared, list) and prepared:
                history_for_estimate = prepared
        except Exception:
            history_for_estimate = list(context_window or [])
    elif history_for_estimate:
        # Keep token estimates aligned with runtime prompt construction even when
        # we only have the persisted MemAgentModel (not a live MemAgent instance).
        history_limit = 60
        if context_window_size >= 200000:
            history_limit = 120
        elif context_window_size >= 100000:
            history_limit = 100
        elif context_window_size >= 64000:
            history_limit = 80
        elif context_window_size >= 16000:
            history_limit = 40
        else:
            history_limit = 24
        history_for_estimate = history_for_estimate[-history_limit:]

    history_tokens = 0
    for msg in history_for_estimate:
        content = msg.get("content") or msg.get("text") or ""
        history_tokens += estimate_tokens(str(content))

    toolbox_tokens = estimate_collection_tokens(toolbox_memory)
    workflow_tokens = estimate_collection_tokens(workflow_memory)
    entity_tokens = estimate_collection_tokens(entity_memory)
    summary_tokens = estimate_collection_tokens(summary_memory)

    total_tokens = (
        instruction_tokens
        + persona_tokens
        + tools_tokens
        + toolbox_tokens
        + workflow_tokens
        + entity_tokens
        + summary_tokens
        + history_tokens
    )
    percentage_used = (
        (total_tokens / context_window_size) * 100 if context_window_size > 0 else 0
    )

    return {
        "total_tokens": total_tokens,
        "context_window_tokens": context_window_size,
        "percentage_used": round(percentage_used, 2),
        "message_count": len(history_for_estimate),
        "thread_message_count": len(context_window or []),
        "composition": {
            "instruction": instruction_tokens,
            "persona": persona_tokens,
            "tools": tools_tokens,
            "toolbox_memory": toolbox_tokens,
            "workflow_memory": workflow_tokens,
            "entity_memory": entity_tokens,
            "summary_memory": summary_tokens,
            "history": history_tokens,
        },
    }


def _format_trace_timestamp(value: Any) -> str:
    """Format mixed timestamp values into a compact display string."""
    timestamp = _coerce_timestamp(value)
    if timestamp is not None:
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    text = _to_text(value).strip()
    return text or "—"


def _extract_trace_memory_id(payload: Dict[str, Any], fallback: str = "") -> str:
    """Extract memory ID from trace payloads."""
    if not isinstance(payload, dict):
        return _to_text(fallback).strip()
    memory_id = _to_text(
        payload.get("memory_id") or payload.get("memoryId") or fallback
    ).strip()
    return memory_id or "—"


def _extract_trace_thread_id(payload: Dict[str, Any], fallback: str = "") -> str:
    """Extract thread/conversation ID from trace payloads."""
    if not isinstance(payload, dict):
        return _to_text(fallback).strip() or "—"

    for key in ("thread_id", "threadId", "conversation_id", "conversationId"):
        value = _to_text(payload.get(key)).strip()
        if value:
            return value

    memory_fallback = _extract_trace_memory_id(payload, fallback=fallback)
    return memory_fallback or "—"


def _thread_row_key(thread_id: str, memory_id: str) -> str:
    """Build a stable key for thread rows."""
    _ = thread_id
    return memory_id


def _build_agent_thread_rows(agents: List[Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Build per-agent thread rows for traces table child rows."""
    provider = _state.get("provider")
    if not provider or not agents:
        return {}

    agent_ids = {
        _extract_agent_identifier(agent)
        for agent in agents
        if _extract_agent_identifier(agent)
    }

    aggregate: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # Fast path: aggregate directly from conversation documents.
    try:
        from ..enums.memory_type import MemoryType

        docs = provider.list_all(MemoryType.CONVERSATION_MEMORY) or []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            agent_id = _to_text(doc.get("agent_id") or doc.get("agentId")).strip()
            if not agent_id or agent_id not in agent_ids:
                continue

            memory_id = _extract_trace_memory_id(doc)
            key = _thread_row_key(memory_id, memory_id)
            ts = _extract_message_timestamp(doc) or 0.0

            by_thread = aggregate.setdefault(agent_id, {})
            entry = by_thread.setdefault(
                key,
                {
                    "thread_id": memory_id,
                    "memory_id": memory_id,
                    "event_count": 0,
                    "last_ts": 0.0,
                },
            )
            entry["event_count"] += 1
            if ts > entry["last_ts"]:
                entry["last_ts"] = ts
    except Exception as exc:
        logger.debug("Failed thread aggregation from docs: %s", exc)

    # Fallback for agents missing doc-level association.
    for agent in agents:
        agent_id = _extract_agent_identifier(agent)
        if not agent_id:
            continue
        if agent_id in aggregate and aggregate[agent_id]:
            continue

        by_thread: Dict[str, Dict[str, Any]] = {}
        for memory_id in _extract_agent_memory_ids(agent):
            history = _retrieve_conversation_history(memory_id=memory_id, limit=None)
            for msg in history:
                key = _thread_row_key(memory_id, memory_id)
                ts = _extract_message_timestamp(msg) or 0.0
                entry = by_thread.setdefault(
                    key,
                    {
                        "thread_id": memory_id,
                        "memory_id": memory_id,
                        "event_count": 0,
                        "last_ts": 0.0,
                    },
                )
                entry["event_count"] += 1
                if ts > entry["last_ts"]:
                    entry["last_ts"] = ts
        if by_thread:
            aggregate[agent_id] = by_thread

    thread_rows_by_agent: Dict[str, List[Dict[str, Any]]] = {}
    for agent_id, by_thread in aggregate.items():
        rows = []
        for entry in by_thread.values():
            rows.append(
                {
                    "thread_id": entry["thread_id"],
                    "memory_id": entry["memory_id"],
                    "event_count": int(entry["event_count"]),
                    "last_ts": float(entry["last_ts"]),
                    "last_activity": (
                        _format_trace_timestamp(entry["last_ts"])
                        if entry["last_ts"] > 0
                        else "—"
                    ),
                }
            )
        rows.sort(key=lambda item: item.get("last_ts", 0.0), reverse=True)
        thread_rows_by_agent[agent_id] = rows

    return thread_rows_by_agent


def _build_agent_trace_metrics(agents: List[Any]) -> Dict[str, Dict[str, Any]]:
    """Build per-agent trace metrics for the traces table."""
    provider = _state.get("provider")
    if not provider or not agents:
        return {}

    agent_ids = {
        _extract_agent_identifier(agent)
        for agent in agents
        if _extract_agent_identifier(agent)
    }

    metrics: Dict[str, Dict[str, Any]] = {}

    # Fast path: aggregate from conversation docs in one pass.
    try:
        from ..enums.memory_type import MemoryType

        docs = provider.list_all(MemoryType.CONVERSATION_MEMORY) or []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            agent_id = _to_text(doc.get("agent_id") or doc.get("agentId")).strip()
            if not agent_id or agent_id not in agent_ids:
                continue
            ts = _extract_message_timestamp(doc) or 0.0
            entry = metrics.setdefault(agent_id, {"event_count": 0, "last_ts": 0.0})
            entry["event_count"] += 1
            if ts > entry["last_ts"]:
                entry["last_ts"] = ts
    except Exception as exc:
        logger.debug("Failed to aggregate traces from conversation docs: %s", exc)

    # Fallback for agents not covered by doc-level metadata.
    for agent in agents:
        agent_id = _extract_agent_identifier(agent)
        if not agent_id:
            continue
        if agent_id in metrics and metrics[agent_id].get("event_count", 0) > 0:
            continue

        event_count = 0
        last_ts = 0.0
        for memory_id in _extract_agent_memory_ids(agent):
            history = _retrieve_conversation_history(memory_id=memory_id, limit=None)
            event_count += len(history)
            for message in history:
                ts = _extract_message_timestamp(message) or 0.0
                if ts > last_ts:
                    last_ts = ts
        metrics[agent_id] = {"event_count": event_count, "last_ts": last_ts}

    return metrics


def _build_trace_agent_rows(
    agents: List[Any],
    tool_counts: Dict[str, int],
    trace_metrics: Dict[str, Dict[str, Any]],
    thread_rows_by_agent: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    search_query: str = "",
) -> List[Dict[str, Any]]:
    """Create filtered rows for the traces agent table."""
    rows: List[Dict[str, Any]] = []
    query = _to_text(search_query).strip().lower()

    for agent in agents:
        agent_id = _extract_agent_identifier(agent)
        if not agent_id:
            continue

        metrics = trace_metrics.get(agent_id, {})
        last_ts = float(metrics.get("last_ts") or 0.0)
        row = {
            "agent_id": agent_id,
            "name": _extract_agent_persona_name(agent),
            "mode": _to_text(
                getattr(agent, "application_mode", None)
                if not isinstance(agent, dict)
                else agent.get("application_mode")
            ).strip()
            or "assistant",
            "memory_count": len(_extract_agent_memory_ids(agent)),
            "tool_count": int(tool_counts.get(agent_id, 0)),
            "event_count": int(metrics.get("event_count") or 0),
            "last_ts": last_ts,
            "last_activity": _format_trace_timestamp(last_ts) if last_ts > 0 else "—",
            "thread_rows": (
                thread_rows_by_agent.get(agent_id, []) if thread_rows_by_agent else []
            ),
        }

        if query:
            searchable = " ".join(
                [
                    row["name"],
                    row["agent_id"],
                    row["mode"],
                ]
            ).lower()
            if query not in searchable:
                continue

        rows.append(row)

    rows.sort(key=lambda item: item.get("last_ts", 0.0), reverse=True)
    return rows


def _load_agent_trace_events(
    agent: Any,
    per_memory_limit: int = 100,
    total_limit: int = 300,
    thread_id: Optional[str] = None,
    thread_memory_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load trace events for a selected agent."""
    events: List[Dict[str, Any]] = []
    agent_id = _extract_agent_identifier(agent)
    memory_ids = _extract_agent_memory_ids(agent)
    selected_thread_id = _to_text(thread_id).strip()
    selected_thread_memory_id = _to_text(thread_memory_id).strip()

    for memory_id in memory_ids:
        history = _retrieve_conversation_history(memory_id=memory_id, limit=None)
        if per_memory_limit > 0 and len(history) > per_memory_limit:
            history = history[-per_memory_limit:]
        for msg in history:
            event_thread_id = _extract_trace_thread_id(msg, fallback=memory_id)
            if (
                selected_thread_id
                and event_thread_id != selected_thread_id
                and memory_id != selected_thread_id
            ):
                continue
            if selected_thread_memory_id and memory_id != selected_thread_memory_id:
                continue
            events.append(
                {
                    "memory_id": memory_id,
                    "thread_id": event_thread_id,
                    "role": _to_text(msg.get("role")).strip().lower() or "system",
                    "content": _to_text(msg.get("content") or msg.get("text")),
                    "timestamp": _format_trace_timestamp(msg.get("timestamp")),
                }
            )

    # Fallback for agents without memory_ids: load directly by agent_id.
    if not events and agent_id:
        provider = _state.get("provider")
        if provider:
            try:
                from ..enums.memory_type import MemoryType

                docs = provider.list_all(MemoryType.CONVERSATION_MEMORY) or []
                for doc in docs:
                    if not isinstance(doc, dict):
                        continue
                    doc_agent_id = _to_text(
                        doc.get("agent_id") or doc.get("agentId")
                    ).strip()
                    if doc_agent_id != agent_id:
                        continue
                    event_memory_id = _extract_trace_memory_id(doc, fallback="—")
                    event_thread_id = _extract_trace_thread_id(
                        doc, fallback=event_memory_id
                    )
                    if (
                        selected_thread_id
                        and event_thread_id != selected_thread_id
                        and event_memory_id != selected_thread_id
                    ):
                        continue
                    if (
                        selected_thread_memory_id
                        and event_memory_id != selected_thread_memory_id
                    ):
                        continue
                    events.append(
                        {
                            "memory_id": event_memory_id,
                            "thread_id": event_thread_id,
                            "role": _to_text(doc.get("role")).strip().lower()
                            or "system",
                            "content": _to_text(doc.get("content") or doc.get("text")),
                            "timestamp": _format_trace_timestamp(doc.get("timestamp")),
                        }
                    )
            except Exception as exc:
                logger.debug(
                    "Failed direct trace fallback for agent %s: %s",
                    agent_id,
                    exc,
                )

    events.sort(key=_trace_sort_key)
    if total_limit > 0 and len(events) > total_limit:
        return events[-total_limit:]
    return events


def _trace_sort_key(event: Dict[str, Any]):
    """Sort trace events by timestamp when available."""
    timestamp = _coerce_timestamp(event.get("timestamp"))
    if timestamp is not None:
        return (0, timestamp)
    raw = _to_text(event.get("timestamp")).strip()
    return (1, raw)


def _normalize_memory_type_values(
    memory_types: Optional[List[Any]],
) -> List[str]:
    """Normalize memory type enums/strings into stable string values."""
    if not memory_types:
        return []

    from ..enums.memory_type import MemoryType

    valid_values = {memory_type.value for memory_type in MemoryType}
    normalized: List[str] = []
    seen = set()
    for item in memory_types:
        value = _to_text(getattr(item, "value", item)).strip().lower()
        if not value or value not in valid_values or value in seen:
            continue
        normalized.append(value)
        seen.add(value)
    return normalized


def _default_memory_types_for_mode(application_mode: Optional[str]) -> List[str]:
    """Return application-mode default memory types as normalized values."""
    from ..enums import ApplicationMode, ApplicationModeConfig

    normalized_mode = _to_text(application_mode).strip().lower()
    try:
        mode = (
            ApplicationModeConfig.validate_mode(normalized_mode)
            if normalized_mode
            else ApplicationMode.DEFAULT
        )
    except ValueError:
        mode = ApplicationMode.DEFAULT
    return _normalize_memory_type_values(ApplicationModeConfig.get_memory_types(mode))


def _build_memory_types_for_agent(
    application_mode: Optional[str],
    enable_entity_memory: bool,
    enable_workflow_memory: bool,
    existing_memory_types: Optional[List[Any]] = None,
) -> List[str]:
    """Build memory type configuration from mode defaults + explicit memory toggles."""
    from ..enums.memory_type import MemoryType

    default_values = _default_memory_types_for_mode(application_mode)
    existing_values = _normalize_memory_type_values(existing_memory_types)

    ordered_values: List[str] = []
    seen = set()
    for memory_type in existing_values + default_values:
        if memory_type in seen:
            continue
        ordered_values.append(memory_type)
        seen.add(memory_type)

    entity_value = MemoryType.ENTITY_MEMORY.value
    if enable_entity_memory:
        if entity_value not in seen:
            ordered_values.append(entity_value)
            seen.add(entity_value)
    else:
        ordered_values = [value for value in ordered_values if value != entity_value]

    workflow_value = MemoryType.WORKFLOW_MEMORY.value
    if enable_workflow_memory:
        if workflow_value not in seen:
            ordered_values.append(workflow_value)
            seen.add(workflow_value)
    else:
        ordered_values = [value for value in ordered_values if value != workflow_value]

    summaries_value = MemoryType.SUMMARIES.value
    if summaries_value not in seen:
        ordered_values.append(summaries_value)

    return ordered_values


def _agent_entity_memory_enabled(agent: Any) -> bool:
    """Return whether entity memory is enabled for an agent config."""
    from ..enums.memory_type import MemoryType

    configured_types = _normalize_memory_type_values(
        getattr(agent, "memory_types", None)
    )
    if not configured_types:
        configured_types = _default_memory_types_for_mode(
            getattr(agent, "application_mode", None)
        )
    return MemoryType.ENTITY_MEMORY.value in configured_types


def _agent_workflow_memory_enabled(agent: Any) -> bool:
    """Return whether workflow memory is enabled for an agent config."""
    from ..enums.memory_type import MemoryType

    configured_types = _normalize_memory_type_values(
        getattr(agent, "memory_types", None)
    )
    if not configured_types:
        configured_types = _default_memory_types_for_mode(
            getattr(agent, "application_mode", None)
        )
    return MemoryType.WORKFLOW_MEMORY.value in configured_types


def _provider_supports_entity_memory(provider: Any) -> bool:
    """Best-effort detection for provider entity-memory support."""
    if provider is None:
        return False

    support_fn = getattr(provider, "supports_entity_memory", None)
    if callable(support_fn):
        try:
            return bool(support_fn())
        except Exception:
            return False
    return bool(getattr(provider, "entity_memory_collection", None))


def _entity_memory_status_error(agent: Any) -> str:
    """Return a user-facing warning when entity memory is configured but unavailable."""
    if not agent or not _agent_entity_memory_enabled(agent):
        return ""
    provider = _state.get("provider")
    if _provider_supports_entity_memory(provider):
        return ""
    return (
        "Entity memory is enabled in this agent configuration, but the active memory "
        "provider does not support ENTITY_MEMORY. "
        "The entity_memory_lookup/entity_memory_upsert tools will not be available."
    )


def _parse_bool(value: Optional[str]) -> bool:
    """Parse checkbox-like values into bools."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "on", "yes"}


def _parse_memory_ids(value: Optional[str]) -> List[str]:
    """Parse a comma- or newline-delimited memory ID list."""
    if not value:
        return []
    raw_parts = value.replace("\n", ",").split(",")
    return [part.strip() for part in raw_parts if part.strip()]


def _parse_skill_paths_json(
    value: Optional[str],
) -> Tuple[Optional[List[str]], Optional[str]]:
    """Parse skill path JSON payload from Playground config."""
    if value is None:
        return None, None
    raw = str(value).strip()
    if not raw:
        return [], None
    try:
        parsed = json.loads(raw)
    except Exception as exc:
        return None, f"Invalid skill paths JSON: {exc}"
    if not isinstance(parsed, list):
        return None, "Skill paths JSON must be an array."

    paths: List[str] = []
    seen = set()
    for item in parsed:
        path = str(item or "").strip()
        if not path or path in seen:
            continue
        paths.append(path)
        seen.add(path)
    return paths, None


def _parse_mcp_servers_json(
    value: Optional[str],
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Parse MCP server JSON payload from Playground config."""
    if value is None:
        return None, None
    raw = str(value).strip()
    if not raw:
        return [], None
    try:
        parsed = json.loads(raw)
    except Exception as exc:
        return None, f"Invalid MCP servers JSON: {exc}"
    if not isinstance(parsed, list):
        return None, "MCP servers JSON must be an array."

    normalized: List[Dict[str, Any]] = []
    seen_names = set()
    for index, item in enumerate(parsed):
        if not isinstance(item, dict):
            return None, f"MCP server entry {index + 1} must be an object."

        name = str(item.get("name", "")).strip()
        if not name:
            return None, f"MCP server entry {index + 1} is missing 'name'."
        if name in seen_names:
            return None, f"MCP server name '{name}' is duplicated."

        transport = str(item.get("transport", "stdio")).strip().lower() or "stdio"
        if transport not in {"stdio", "http"}:
            return None, (
                f"MCP server '{name}' has unsupported transport '{transport}'. "
                "Use 'stdio' or 'http'."
            )

        server: Dict[str, Any] = {"name": name, "transport": transport}
        timeout = item.get("timeout")
        if timeout not in (None, ""):
            try:
                timeout_value = int(timeout)
            except Exception:
                return None, f"MCP server '{name}' has invalid timeout value."
            server["timeout"] = max(1, min(timeout_value, 300))
        else:
            server["timeout"] = 30

        if transport == "stdio":
            command = str(item.get("command", "")).strip()
            if not command:
                return (
                    None,
                    f"MCP server '{name}' requires a command for stdio transport.",
                )
            server["command"] = command
            args = item.get("args", [])
            if args is None:
                args = []
            if not isinstance(args, list):
                return None, f"MCP server '{name}' field 'args' must be an array."
            server["args"] = [str(arg) for arg in args if str(arg).strip()]
            env = item.get("env", {})
            if env is None:
                env = {}
            if not isinstance(env, dict):
                return None, f"MCP server '{name}' field 'env' must be an object."
            server["env"] = {
                str(key): str(val) for key, val in env.items() if str(key).strip()
            }
            cwd = str(item.get("cwd", "")).strip()
            if cwd:
                server["cwd"] = cwd
        else:
            url = str(item.get("url", "")).strip()
            if not url:
                return None, f"MCP server '{name}' requires 'url' for http transport."
            server["url"] = url

        normalized.append(server)
        seen_names.add(name)

    return normalized, None


def _parse_llm_config(
    provider: str, model: str, raw_json: str
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Build an LLM config dict from form values."""
    config: Optional[Dict[str, Any]] = None
    error = None

    provider_value = (provider or "").strip().lower()
    model_value = (model or "").strip()

    if provider_value or model_value:
        config = {}
        if model_value and not provider_value:
            provider_value = DEFAULT_LLM_PROVIDER
        if provider_value:
            config["provider"] = provider_value
        if model_value:
            if provider_value == "azure":
                config["deployment_name"] = model_value
            else:
                config["model"] = model_value

    if raw_json and raw_json.strip():
        try:
            extra = json.loads(raw_json)
            if not isinstance(extra, dict):
                error = "LLM config JSON must be an object."
            else:
                if config is None:
                    config = extra
                else:
                    config.update(extra)
        except Exception as exc:
            error = f"Invalid LLM config JSON: {exc}"

    return config, error


def _build_persona_payload(
    name: str, role: str, goals: str, background: str
) -> Optional[Dict[str, Any]]:
    """Build a minimal persona payload for persistence."""
    name_value = (name or "").strip()
    if not name_value:
        return None
    return {
        "name": name_value,
        "role": (role or "").strip() or "general",
        "goals": (goals or "").strip(),
        "background": (background or "").strip(),
    }


def _extract_agent_id(result: Any, fallback: Optional[str]) -> Optional[str]:
    """Extract agent_id from provider return values."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return (
            result.get("agent_id")
            or result.get("_id")
            or result.get("agentId")
            or fallback
        )
    if hasattr(result, "agent_id"):
        return getattr(result, "agent_id")
    return fallback


def _persist_mcp_configs_to_toolbox(
    agent_id: str, mcp_servers: List[Dict[str, Any]], memory_ids: List[str]
) -> None:
    """Persist MCP server JSON payloads into toolbox memory."""
    if not _state["provider"] or not mcp_servers:
        return

    from ..enums.memory_type import MemoryType

    memory_id = memory_ids[0] if memory_ids else None

    for server in mcp_servers:
        server_name = str(server.get("name", "")).strip()
        if not server_name:
            continue
        payload = {
            "_id": f"{agent_id}:mcp:{server_name}",
            "tool_id": f"{agent_id}:mcp:{server_name}",
            "name": f"mcp::{server_name}",
            "description": f"MCP server config for {server_name}",
            "signature": "mcp_server_config(server_json)",
            "docstring": "Stored MCP server configuration JSON.",
            "tool_type": "mcp_server_config",
            "type": "mcp_server_config",
            "parameters": server,
            "agent_id": agent_id,
            "memory_id": memory_id,
        }
        try:
            _state["provider"].store(payload, memory_store_type=MemoryType.TOOLBOX)
        except Exception as exc:
            logger.warning(
                "Failed to persist MCP server config '%s' for agent %s: %s",
                server_name,
                agent_id,
                exc,
            )


def _build_agent_form_data(agent: Any) -> Dict[str, Any]:
    """Prepare agent data for edit form rendering."""
    from ..memagent.constants import DEFAULT_INSTRUCTION, DEFAULT_MAX_STEPS

    persona = getattr(agent, "persona", None)
    if isinstance(persona, dict):
        persona_name = persona.get("name", "")
        persona_role = persona.get("role", "")
        persona_goals = persona.get("goals", "")
        persona_background = persona.get("background", "")
    else:
        persona_name = getattr(persona, "name", "") if persona else ""
        persona_role = getattr(persona, "role", "") if persona else ""
        persona_goals = getattr(persona, "goals", "") if persona else ""
        persona_background = getattr(persona, "background", "") if persona else ""

    llm_config = getattr(agent, "llm_config", None) or {}
    llm_provider = _normalize_llm_provider(llm_config.get("provider", "openai"))
    llm_model = (
        llm_config.get("model")
        or llm_config.get("deployment_name")
        or _get_default_llm_model(llm_provider)
    )
    extra_config = {
        key: value
        for key, value in llm_config.items()
        if key not in {"provider", "model", "deployment_name"}
    }
    llm_config_json = json.dumps(extra_config, indent=2) if extra_config else ""

    memory_ids = getattr(agent, "memory_ids", None) or []

    # Sandbox provider
    sandbox_val = getattr(agent, "sandbox_provider", None) or ""
    if isinstance(sandbox_val, dict):
        sandbox_val = sandbox_val.get("provider", "")
    internet_val = _normalize_internet_provider_name(
        getattr(agent, "internet_access_provider", None)
    )
    skills_marketplace_val = _normalize_skills_marketplace_provider_name(
        getattr(agent, "skills_marketplace_provider", None)
    )
    memory_types = _normalize_memory_type_values(getattr(agent, "memory_types", None))

    return {
        "agent_id": getattr(agent, "agent_id", ""),
        "agent_name": _to_text(getattr(agent, "name", "")).strip(),
        "instruction": getattr(agent, "instruction", None) or DEFAULT_INSTRUCTION,
        "application_mode": getattr(agent, "application_mode", None) or "assistant",
        "memory_types": memory_types,
        "enable_entity_memory": _agent_entity_memory_enabled(agent),
        "enable_workflow_memory": _agent_workflow_memory_enabled(agent),
        "max_steps": getattr(agent, "max_steps", None) or DEFAULT_MAX_STEPS,
        "tool_access": getattr(agent, "tool_access", None) or "private",
        "semantic_cache": bool(getattr(agent, "semantic_cache", False)),
        "is_favorite": bool(getattr(agent, "is_favorite", False)),
        "memory_ids_raw": ", ".join(memory_ids),
        "persona_name": persona_name,
        "persona_role": persona_role,
        "persona_goals": persona_goals,
        "persona_background": persona_background,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "llm_config_json": llm_config_json,
        "sandbox_provider": sandbox_val,
        "internet_provider": internet_val,
        "skills_marketplace_provider": skills_marketplace_val,
        "skill_paths": getattr(agent, "skill_paths", None) or [],
        "mcp_servers": getattr(agent, "mcp_servers", None) or [],
        "agent_tools": _extract_agent_tools(agent),
    }


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


def _get_repo_root() -> Path:
    """Locate the best available root directory for Evalground."""
    root, _ = _find_longmemeval_eval_dir()
    if root:
        return root
    return Path.cwd().resolve()


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


def run_server(host: str = "127.0.0.1", port: int = 8765):
    """Run the Memorizz UI server."""
    import uvicorn

    app = create_app()
    print(f"\n🧠 Memorizz Local UI")
    print(f"   Running at: http://{host}:{port}")
    print(f"   Press Ctrl+C to stop\n")
    uvicorn.run(app, host=host, port=port, log_level="info")

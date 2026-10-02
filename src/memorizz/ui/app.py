# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""FastAPI application for Memorizz Local UI."""

import logging
import os
import threading
import uuid
from contextlib import asynccontextmanager

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from .._env_io import apply_env_updates as _shared_apply_env_updates
from .._env_io import env_override_warnings
from .._env_io import load_layered_env as _load_layered_env
from .._env_io import resolve_env_file as _resolve_env_file
from .._env_io import (
    resolve_oracle_in_database_embedding_from_env as _resolve_oracle_embedding_mode,
)
from .._env_io import validate_env_updates
from .helpers import (
    DEFAULT_LLM_MODEL_BY_PROVIDER,
    DEFAULT_LLM_PROVIDER,
    LLM_MODEL_CATALOG,
    _build_agent_nav_items,
    _extract_agent_identifier,
    _extract_agent_persona_name,
    _get_default_llm_model,
    _get_default_llm_provider,
    _list_agents,
    _parse_bool,
    _to_text,
    _validate_sandbox_provider_choice,
)
from .integrations_view import settings_summary
from .routers.agents_api import router as agents_api_router
from .routers.agents_crud import router as agents_crud_router
from .routers.capabilities import router as capabilities_router
from .routers.comparisons import router as comparisons_router
from .routers.comparisons import stop_comparison_workers
from .routers.continual_learning import router as continual_learning_router
from .routers.evalground import router as evalground_router
from .routers.evalground import stop_active_eval_run_processes
from .routers.harnesses import router as harnesses_router
from .routers.huggingface import router as huggingface_router
from .routers.knowledge_base import router as knowledge_base_router
from .routers.learning_control_plane import router as learning_control_plane_router
from .routers.mcp import router as mcp_router
from .routers.memory_pages import router as memory_pages_router
from .routers.ollama import router as ollama_router
from .routers.oracle_docker import router as oracle_docker_router
from .routers.playground import router as playground_router
from .routers.traces import router as traces_router
from .routers.vercel_skills import router as vercel_skills_router
from .routers.whatsapp import router as whatsapp_router
from .security import ReadOnlyProviderProxy, UIAccessController, ui_read_only
from .state import STATIC_DIR, _state, close_meta_harness, templates

logger = logging.getLogger(__name__)


DEFAULT_GRAALPY_INTERNET_ACCESS = "0"

SETTINGS_SECTIONS = [
    {
        "title": "Memory provider & Notion",
        "description": "Saved defaults do not reconnect the current UI or migrate memory. Use Connect to switch this UI's provider. For guided Notion setup, run memorizz notion connect in a terminal.",
        "fields": [
            {
                "env": "MEMORIZZ_BACKEND",
                "label": "Default memory provider",
                "field_type": "select",
                "options": [{"value": "", "label": "Keep unchanged"}]
                + [
                    {"value": name, "label": name.title()}
                    for name in ("filesystem", "mongodb", "oracle", "notion")
                ],
                "hint": "Memory storage, not the LLM provider. Applies to new CLI / MCP processes.",
            },
            {
                "env": "NOTION_TOKEN",
                "label": "Notion connection token",
                "placeholder": "Leave blank to keep the configured token",
                "hint": "Share the exact parent page or library with this Notion connection. The token is never prefilled.",
            },
            {
                "env": "MEMORIZZ_NOTION_DATA_SOURCE_ID",
                "label": "Notion memory data source ID",
                "field_type": "text",
                "placeholder": "Data-source UUID (not the enclosing database ID)",
                "hint": "memorizz notion connect resolves database URLs and checks the managed memory columns.",
            },
            {
                "env": "MEMORIZZ_NOTION_SEMANTIC_BACKEND",
                "label": "Notion semantic / vector backend",
                "field_type": "select",
                "options": [
                    {"value": "", "label": "Keep unchanged (default: filesystem)"}
                ]
                + [
                    {"value": name, "label": name.title()}
                    for name in ("filesystem", "mongodb", "oracle", "none")
                ],
                "hint": "Notion stores documents; this provider stores vectors. None disables semantic search. Configure its credentials separately using the CLI wizard.",
            },
            {
                "env": "MEMORIZZ_NOTION_VECTOR_PATH",
                "label": "Notion filesystem vector directory",
                "field_type": "text",
                "placeholder": "Automatic separate directory per library when unset",
                "hint": "Requires the filesystem extra and a configured embedding provider/model.",
            },
        ],
    },
    {
        "title": "LLM & embeddings",
        "description": "Keys used by OpenAI, Anthropic, Azure OpenAI, Ollama, Voyage AI, and Hugging Face integrations.",
        "fields": [
            {
                "env": "OPENAI_API_KEY",
                "label": "OpenAI API key",
                "placeholder": "sk-...",
                "hint": "Required for OpenAI LLM and embedding providers.",
            },
            {
                "env": "ANTHROPIC_API_KEY",
                "label": "Anthropic API key",
                "placeholder": "sk-ant-...",
                "hint": "Required for Anthropic Claude LLM provider.",
            },
            {
                "env": "AZURE_OPENAI_API_KEY",
                "label": "Azure OpenAI API key",
                "placeholder": "azure-...",
                "hint": "Used by Azure OpenAI LLM and embedding providers.",
            },
            {
                "env": "OLLAMA_HOST",
                "label": "Ollama host URL",
                "placeholder": "http://localhost:11434",
                "hint": "Ollama server URL. Defaults to http://localhost:11434.",
            },
            {
                "env": "VOYAGE_API_KEY",
                "label": "Voyage AI API key",
                "placeholder": "pa-...",
                "hint": "Used by the Voyage AI embedding provider.",
            },
            {
                "env": "HF_TOKEN",
                "label": "Hugging Face token",
                "placeholder": "hf_...",
                "hint": "Optional token for private Hugging Face models.",
            },
            {
                "env": "MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER",
                "label": "Default embedding provider",
                "hint": "Default for filesystem, MongoDB, Oracle external embeddings, and Notion's separate vector store when not explicitly overridden.",
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
                "label": "Default embedding model",
                "field_type": "text",
                "placeholder": "text-embedding-3-small",
                "hint": "Optional model id used with the default embedding provider.",
            },
            {
                "env": "MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS",
                "label": "Default embedding dimensions",
                "field_type": "text",
                "placeholder": "1536",
                "hint": "Optional output dimensions; must match the selected model and vector index/schema. Changing dimensions may require a new index and sync.",
            },
            {
                "env": "MEMORIZZ_ORACLE_IN_DATABASE_EMBEDDING",
                "label": "Oracle embedding mode",
                "hint": "Use Oracle's ONNX model, or use the external embedding provider configured above.",
                "field_type": "select",
                "default_value": "true",
                "options": [
                    {"value": "true", "label": "Oracle in-database ONNX"},
                    {"value": "false", "label": "Configured external provider"},
                ],
            },
        ],
    },
    {
        "title": "Default agent model",
        "description": "Default provider/model used in Agent creation and Playground config when not explicitly set on an agent.",
        "fields": [
            {
                "env": "MEMORIZZ_DEFAULT_LLM_PROVIDER",
                "label": "Default LLM provider",
                "hint": "Applies to newly-created agents unless overridden.",
                "field_type": "select",
                "default_value": DEFAULT_LLM_PROVIDER,
                "options": [
                    {"value": "openai", "label": "OpenAI"},
                    {"value": "anthropic", "label": "Anthropic"},
                    {"value": "azure", "label": "Azure OpenAI"},
                    {"value": "ollama", "label": "Ollama (Local)"},
                    {"value": "huggingface", "label": "HuggingFace"},
                    {"value": "mlx", "label": "MLX (Apple Silicon)"},
                    {
                        "value": "local-openai",
                        "label": "Local OpenAI-compatible (llama.cpp / LM Studio)",
                    },
                ],
            },
            {
                "env": "MEMORIZZ_DEFAULT_LLM_MODEL",
                "label": "Default model / deployment",
                "hint": "Provider-aware dropdown populated from the latest supported model catalog.",
                "field_type": "select",
                "default_value": DEFAULT_LLM_MODEL_BY_PROVIDER[DEFAULT_LLM_PROVIDER],
                "options": [],
            },
        ],
    },
    {
        "title": "Internet access",
        "description": "Keys for Tavily, Firecrawl, and other web browsing providers.",
        "fields": [
            {
                "env": "TAVILY_API_KEY",
                "label": "Tavily API key",
                "placeholder": "tvly-...",
                "hint": "Preferred default for Memorizz internet access.",
            },
            {
                "env": "FIRECRAWL_API_KEY",
                "label": "Firecrawl API key",
                "placeholder": "fc-...",
                "hint": "Fallback provider for web browsing workflows.",
            },
            {
                "env": "MEMORIZZ_DEFAULT_INTERNET_PROVIDER_API_KEY",
                "label": "Default internet provider API key",
                "placeholder": "provider-key",
                "hint": "Used when MEMORIZZ_DEFAULT_INTERNET_PROVIDER is set.",
            },
        ],
    },
    {
        "title": "Skills marketplace",
        "description": "API keys for skillsmp.com and Vercel Agent Skills (skills.sh) marketplace integrations.",
        "fields": [
            {
                "env": "SKILLSMP_API_KEY",
                "label": "SkillsMP API key",
                "placeholder": "sk_live_skillsmp_...",
                "hint": "Used for https://skillsmp.com/ API access in agent marketplace tools.",
            },
            {
                "env": "GITHUB_TOKEN",
                "label": "GitHub personal access token",
                "placeholder": "ghp_...",
                "hint": (
                    "Required for Vercel Agent Skills search (uses GitHub's code search API, which "
                    "requires authentication). Also raises GitHub API rate limits from 60 to 5,000 "
                    "requests/hour. Memorizz only reads public skills data, so "
                    "<strong>no scopes are required</strong> — create a classic token with every box "
                    "unchecked at "
                    '<a href="https://github.com/settings/tokens/new?description=Memorizz%20Vercel%20Skills" '
                    'target="_blank" rel="noopener noreferrer">github.com/settings/tokens</a>. '
                    "(If you need to tick something, <code>public_repo</code> is the safe minimum — "
                    "do <em>not</em> grant <code>repo</code>, which includes private-repo write access.) "
                    "See the "
                    '<a href="https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens" '
                    'target="_blank" rel="noopener noreferrer">GitHub docs</a> for step-by-step instructions.'
                ),
            },
        ],
    },
    {
        "title": "Browser control",
        "description": (
            "Configure the isolated Browser Use provider. Browser tasks are "
            "always side-effecting and require durable operator approval."
        ),
        "fields": [
            {
                "env": "MEMORIZZ_BROWSER_CONTROL_PROVIDER",
                "label": "Default browser control provider",
                "hint": "Browser automation is opt-in; choose Browser Use or leave disabled.",
                "field_type": "select",
                "options": [
                    {"value": "", "label": "None (disabled)"},
                    {"value": "browseruse", "label": "Browser Use (isolated tool)"},
                ],
            },
            {
                "env": "BROWSER_USE_API_KEY",
                "label": "Browser Use API key",
                "placeholder": "bu_...",
                "hint": "Only required when the Browser Use model provider is selected.",
            },
            {
                "env": "MEMORIZZ_BROWSER_USE_COMMAND",
                "label": "Browser Use entry point",
                "placeholder": "browser-use",
                "hint": "Entry point from an isolated Python 3.11+ Browser Use installation.",
                "field_type": "text",
            },
            {
                "env": "MEMORIZZ_BROWSER_USE_PYTHON_COMMAND",
                "label": "Browser Use Python path",
                "placeholder": "/absolute/path/to/browser-use/bin/python",
                "hint": "Optional override when the entry point does not expose an absolute interpreter shebang.",
                "field_type": "text",
            },
            {
                "env": "MEMORIZZ_BROWSER_USE_LLM_PROVIDER",
                "label": "Browser Use LLM provider",
                "field_type": "select",
                "options": [
                    {"value": "openai", "label": "OpenAI"},
                    {"value": "anthropic", "label": "Anthropic"},
                    {"value": "google", "label": "Google"},
                    {"value": "browseruse", "label": "Browser Use"},
                ],
                "default_value": "openai",
            },
            {
                "env": "MEMORIZZ_BROWSER_USE_MODEL",
                "label": "Browser Use model",
                "placeholder": "gpt-4.1-mini",
                "hint": "Optional model override for the selected provider.",
                "field_type": "text",
            },
            {
                "env": "MEMORIZZ_BROWSER_USE_ALLOWED_DOMAINS",
                "label": "Allowed domains",
                "placeholder": "calendar.google.com, *.notion.so",
                "hint": "Comma-separated navigation allowlist. Empty permits any public domain.",
                "field_type": "text",
            },
            {
                "env": "MEMORIZZ_BROWSER_USE_PROHIBITED_DOMAINS",
                "label": "Prohibited domains",
                "placeholder": "bank.example, admin.example",
                "hint": "Comma-separated denylist applied in addition to the allowlist.",
                "field_type": "text",
            },
            {
                "env": "MEMORIZZ_BROWSER_USE_HEADLESS",
                "label": "Headless browser",
                "field_type": "checkbox",
                "default_value": "1",
                "checkbox_label": "Run Chromium without a visible window",
            },
            {
                "env": "MEMORIZZ_BROWSER_USE_BLOCK_IP_ADDRESSES",
                "label": "Block direct IP navigation",
                "field_type": "checkbox",
                "default_value": "1",
                "checkbox_label": "Reject direct IP-address navigation (recommended)",
            },
            {
                "env": "MEMORIZZ_BROWSER_USE_MAX_STEPS",
                "label": "Maximum browser steps",
                "placeholder": "25",
                "hint": "Hard upper bound per task (1-100).",
                "field_type": "text",
            },
            {
                "env": "MEMORIZZ_BROWSER_USE_TASK_TIMEOUT",
                "label": "Browser task timeout (seconds)",
                "placeholder": "600",
                "hint": "Hard wall-clock limit per task (10-3600 seconds).",
                "field_type": "text",
            },
        ],
    },
    {
        "title": "Sandbox providers",
        "description": "Configure sandbox code execution providers. The default provider applies globally to all agents unless overridden per-agent in the Playground.",
        "fields": [
            {
                "env": "MEMORIZZ_DEFAULT_SANDBOX_PROVIDER",
                "label": "Default sandbox provider",
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
                "label": "E2B API key",
                "placeholder": "e2b_...",
                "hint": "Required for the E2B sandbox provider. Manage it at e2b.dev.",
            },
            {
                "env": "DAYTONA_API_KEY",
                "label": "Daytona API key",
                "placeholder": "daytona_...",
                "hint": "Required for the Daytona sandbox provider. Manage it at daytona.io.",
            },
            {
                "env": "GRAALPY_PATH",
                "label": "GraalPy executable path",
                "placeholder": "/path/to/graalpy",
                "hint": "Optional absolute path to the graalpy executable (used when not on PATH).",
                "field_type": "text",
            },
            {
                "env": "MEMORIZZ_GRAALPY_MODE",
                "label": "GraalPy execution mode",
                "hint": (
                    "Subprocess is bounded trusted-code execution, not a security "
                    "sandbox. Java wrapper is the validated UNTRUSTED boundary."
                ),
                "field_type": "select",
                "default_value": "subprocess",
                "options": [
                    {
                        "value": "subprocess",
                        "label": "Subprocess (trusted code only)",
                    },
                    {
                        "value": "java_wrapper",
                        "label": "Java UNTRUSTED wrapper",
                    },
                ],
                "visible_if_sandbox_provider": "graalpy",
            },
            {
                "env": "MEMORIZZ_GRAALPY_INTERNET_ACCESS",
                "label": "GraalPy internet access",
                "hint": (
                    "Explicit egress opt-in for trusted subprocess mode. Network is "
                    "denied by default and Java UNTRUSTED mode always blocks guest IO."
                ),
                "field_type": "checkbox",
                "default_value": DEFAULT_GRAALPY_INTERNET_ACCESS,
                "checkbox_label": "Allow outbound internet access from GraalPy sandbox",
                "visible_if_sandbox_provider": "graalpy",
            },
            {
                "env": "GRAALPY_JAVA_WRAPPER_JAR",
                "label": "GraalPy Java wrapper JAR",
                "placeholder": "/path/to/graalpy-sandbox.jar",
                "hint": "Required only when GraalPy Execution Mode is Java UNTRUSTED wrapper.",
                "field_type": "text",
                "visible_if_sandbox_provider": "graalpy",
            },
        ],
    },
    {
        "title": "Automations",
        "description": "Settings for scheduled automations and message delivery.",
        "fields": [
            {
                "env": "MEMORIZZ_DEFAULT_TIMEZONE",
                "label": "Default timezone",
                "placeholder": "America/New_York",
                "hint": "Default IANA timezone used by automation tools/UI when timezone is omitted.",
                "field_type": "timezone",
            },
            {
                "env": "MEMORIZZ_AUTOMATIONS_UI_WORKER",
                "label": "UI automations worker",
                "hint": "When enabled, the UI process also runs an automations worker loop.",
                "field_type": "checkbox",
                "default_value": "0",
                "checkbox_label": "Run automations worker inside UI process",
            },
            {
                "env": "MEMORIZZ_AUTOMATIONS_POLL_INTERVAL_S",
                "label": "Automations poll interval (s)",
                "placeholder": "5",
                "hint": "Worker poll interval in seconds.",
            },
            {
                "env": "MEMORIZZ_AUTOMATIONS_LEASE_SECONDS",
                "label": "Automations lease seconds",
                "placeholder": "120",
                "hint": "Job lease duration in seconds to prevent duplicate runs across workers.",
            },
            {
                "env": "MEMORIZZ_AUTOMATIONS_CONCURRENCY",
                "label": "Automations concurrency",
                "placeholder": "2",
                "hint": "Max concurrent job executions per worker.",
            },
            {
                "env": "TWILIO_ACCOUNT_SID",
                "label": "Twilio account SID",
                "placeholder": "AC...",
                "hint": "Required for WhatsApp delivery via Twilio.",
            },
            {
                "env": "TWILIO_AUTH_TOKEN",
                "label": "Twilio auth token",
                "placeholder": "••••••••",
                "hint": "Required for WhatsApp delivery via Twilio.",
            },
            {
                "env": "TWILIO_WHATSAPP_FROM",
                "label": "Twilio WhatsApp sender",
                "placeholder": "whatsapp:+1415...",
                "hint": "WhatsApp-enabled Twilio sender. Format: whatsapp:+E164.",
            },
        ],
    },
]
SETTINGS_FIELDS = [
    field for section in SETTINGS_SECTIONS for field in section["fields"]
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    logger.info("Memorizz UI starting...")
    automations_stop_event = threading.Event()
    automations_thread: Optional[threading.Thread] = None

    if str(os.environ.get("MEMORIZZ_AUTOMATIONS_UI_WORKER", "")).strip() == "1":
        # Run a background worker in the UI process (off by default).
        def _automations_worker_loop() -> None:
            import time

            try:
                from ..automation.store.factory import get_automation_store
                from ..automation.worker import run_worker
            except Exception as exc:
                logger.error("Automations UI worker imports failed: %s", exc)
                return

            poll_interval = int(
                str(os.environ.get("MEMORIZZ_AUTOMATIONS_POLL_INTERVAL_S", "5") or "5")
            )
            lease_seconds = int(
                str(
                    os.environ.get("MEMORIZZ_AUTOMATIONS_LEASE_SECONDS", "120") or "120"
                )
            )
            concurrency = int(
                str(os.environ.get("MEMORIZZ_AUTOMATIONS_CONCURRENCY", "2") or "2")
            )

            while not automations_stop_event.is_set():
                provider = _state.get("provider")
                if not provider:
                    time.sleep(1)
                    continue

                store = None
                try:
                    store = get_automation_store(provider)
                except Exception:
                    store = None

                if store is None:
                    time.sleep(5)
                    continue

                try:
                    run_worker(
                        store=store,
                        memory_provider=provider,
                        poll_interval_s=poll_interval,
                        lease_seconds=lease_seconds,
                        max_concurrency=concurrency,
                        stop_event=automations_stop_event,
                    )
                except Exception as exc:
                    # If provider disconnects or the worker errors, retry after a pause.
                    logger.error("Automations UI worker crashed: %s", exc)
                    time.sleep(5)

        automations_thread = threading.Thread(
            target=_automations_worker_loop,
            name="memorizz-automations-ui-worker",
            daemon=True,
        )
        automations_thread.start()

    # Start WhatsApp worker if Twilio is configured
    if all(
        [
            os.environ.get("TWILIO_ACCOUNT_SID"),
            os.environ.get("TWILIO_AUTH_TOKEN"),
            os.environ.get("TWILIO_WHATSAPP_FROM"),
        ]
    ):
        try:
            from ..channels.whatsapp.worker import run_whatsapp_worker

            stop_event = threading.Event()
            worker_thread = threading.Thread(
                target=run_whatsapp_worker,
                kwargs={
                    "memory_provider": _state["provider"],
                    "stop_event": stop_event,
                },
                daemon=True,
                name="whatsapp-worker",
            )
            worker_thread.start()
            _state["whatsapp_worker_thread"] = worker_thread
            _state["whatsapp_worker_stop_event"] = stop_event
            logger.info("WhatsApp worker started")
        except Exception as e:
            logger.error(f"Failed to start WhatsApp worker: {e}", exc_info=True)

    yield

    # Stop WhatsApp worker on shutdown
    if _state.get("whatsapp_worker_stop_event"):
        _state["whatsapp_worker_stop_event"].set()
        if _state.get("whatsapp_worker_thread"):
            try:
                _state["whatsapp_worker_thread"].join(timeout=5)
                logger.info("WhatsApp worker stopped")
            except Exception as e:
                logger.error(f"Error stopping WhatsApp worker: {e}")

    if automations_thread is not None:
        automations_stop_event.set()
        try:
            automations_thread.join(timeout=8)
        except Exception:
            pass

    # Cleanup: stop any active Evalground benchmark subprocesses
    stop_active_eval_run_processes()
    stop_comparison_workers()
    close_meta_harness()
    # Cleanup: close provider connection
    if _state["provider"]:
        try:
            _state["provider"].close()
        except Exception:
            pass
    logger.info("Memorizz UI shutting down...")


def create_app(
    *,
    identity_resolver=None,
    artifact_resolver=None,
    resource_authorizer=None,
    usage_pricing=None,
    persona_evolution=None,
    persona_evolution_actions=False,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    # Load layered env (~/.memorizz/.env, then $CWD/.env) so the UI sees the
    # same keys the CLI configured. override=False keeps real env vars winning.
    _load_layered_env()
    app = FastAPI(
        title="Memorizz Local UI",
        description="Web interface for exploring Memorizz memory providers",
        version=__version__,
        lifespan=lifespan,
    )
    access = UIAccessController()
    if persona_evolution is not None and not access.enabled:
        raise ValueError(
            "Host-owned persona profiles require Memorizz UI authentication"
        )
    app.state.trace_identity_resolver = identity_resolver
    app.state.trace_artifact_resolver = artifact_resolver
    app.state.trace_resource_authorizer = resource_authorizer
    app.state.usage_pricing = usage_pricing
    app.state.persona_evolution = persona_evolution
    app.state.persona_evolution_actions = bool(
        persona_evolution and persona_evolution_actions
    )
    read_only_mode = ui_read_only()

    from fastapi.exception_handlers import request_validation_exception_handler
    from fastapi.exceptions import RequestValidationError

    @app.exception_handler(RequestValidationError)
    async def trace_validation_error(request, exc):
        if request.url.path.startswith("/traces"):
            # Never echo transient account addresses or rejected secret values.
            return JSONResponse(
                {"detail": "Invalid trace request"},
                status_code=422,
                headers={"Cache-Control": "no-store"},
            )
        return await request_validation_exception_handler(request, exc)

    @app.middleware("http")
    async def secure_local_ui(request: Request, call_next):
        from .security import audit_trace_view
        from .trace_access import current_principal

        path = request.url.path
        public_path = path == "/login" or path.startswith("/static/")
        principal = access.principal_for_request(request)

        def denied(detail, status_code=403):
            audit_token = (
                current_principal.set(principal) if principal is not None else None
            )
            try:
                audit_trace_view(
                    request, "trace_access_denied", content_mode="metadata"
                )
            finally:
                if audit_token is not None:
                    current_principal.reset(audit_token)
            return JSONResponse(
                {"detail": detail},
                status_code=status_code,
                headers={"Cache-Control": "no-store, max-age=0"},
            )

        if (
            access.enabled
            and not public_path
            and not access.request_is_authenticated(request)
        ):
            accepts_html = "text/html" in request.headers.get("accept", "")
            if accepts_html:
                safe_next = (
                    path if path.startswith("/") and not path.startswith("//") else "/"
                )
                return RedirectResponse(url=f"/login?next={safe_next}", status_code=303)
            return JSONResponse(
                {"detail": "Memorizz UI authentication required"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        if (
            read_only_mode
            and request.method.upper() not in {"GET", "HEAD", "OPTIONS"}
            and path
            not in {"/login", "/connect", "/traces/account/resolve", "/traces/reveal"}
            and not (
                path == "/persona-evolution/action"
                and app.state.persona_evolution_actions
            )
        ):
            return denied("This Memorizz UI is running in read-only mode")
        principal = access.principal_for_request(request)
        if principal is not None and path.startswith("/traces"):
            from dataclasses import replace

            from ..observability.privacy import validate_opaque

            updates = {}
            for key in ("application_id", "user_id"):
                if key not in request.query_params:
                    continue
                value = request.query_params[key]
                try:
                    validate_opaque(value)
                    if not value or len(value) > 240:
                        raise ValueError()
                except ValueError:
                    return denied("Invalid trace scope", 400)
                if key in principal.filters() and principal.filters()[key] != value:
                    return denied("Requested trace scope is not authorized")
                updates[key] = value
                if key == "user_id":
                    updates["user_bound"] = True
            if updates:
                principal = replace(principal, **updates)
        if principal is not None and principal.restricted and not public_path:
            allowed_posts = {
                "/traces/account/resolve",
                "/traces/reveal",
                "/traces/replays",
            }
            if not path.startswith("/traces") or (
                request.method not in {"GET", "HEAD", "OPTIONS"}
                and path not in allowed_posts
            ):
                return denied("This account is restricted to trace inspection")
        # Prevent cross-origin form submissions to privileged trace actions.
        if (
            path.startswith("/traces") or path.startswith("/persona-evolution")
        ) and request.method not in {
            "GET",
            "HEAD",
            "OPTIONS",
        }:
            origin = request.headers.get("origin")
            if origin and origin != str(request.base_url).rstrip("/"):
                return denied("Cross-origin trace action denied")
        token = current_principal.set(principal) if principal is not None else None
        try:
            response = await call_next(request)
            if path.startswith("/traces") and response.status_code in {401, 403}:
                audit_trace_view(
                    request, "trace_access_denied", content_mode="metadata"
                )
        finally:
            if token is not None:
                current_principal.reset(token)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        if path.startswith("/traces"):
            response.headers["Cache-Control"] = "no-store, max-age=0"
        return response

    # Mount static files
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.include_router(ollama_router)
    app.include_router(huggingface_router)
    app.include_router(oracle_docker_router)
    app.include_router(agents_api_router)
    app.include_router(traces_router)
    from .routers.usage import router as usage_router

    app.include_router(usage_router)
    from .routers.trace_tools import router as trace_tools_router

    app.include_router(trace_tools_router)
    from .routers.persona_evolution import router as persona_evolution_router

    app.include_router(persona_evolution_router)
    app.include_router(continual_learning_router)
    app.include_router(learning_control_plane_router)
    app.include_router(harnesses_router)
    app.include_router(capabilities_router)
    app.include_router(memory_pages_router)
    app.include_router(mcp_router)
    app.include_router(vercel_skills_router)
    app.include_router(knowledge_base_router)
    app.include_router(whatsapp_router)
    app.include_router(evalground_router)
    app.include_router(comparisons_router)
    app.include_router(agents_crud_router)
    app.include_router(playground_router)

    # Configure the shared template engine (imported from ui.state)
    templates.env.globals["llm_model_catalog"] = LLM_MODEL_CATALOG
    templates.env.globals[
        "default_llm_model_by_provider"
    ] = DEFAULT_LLM_MODEL_BY_PROVIDER
    try:
        from zoneinfo import available_timezones

        timezone_options = sorted(
            tz for tz in available_timezones() if tz and "build/" not in tz
        )
    except Exception:
        timezone_options = []

    templates.env.globals["timezone_options"] = timezone_options

    # -------------------------------------------------------------------------
    # Page Routes
    # -------------------------------------------------------------------------

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request, next: str = "/"):
        if not access.enabled or access.request_is_authenticated(request):
            return RedirectResponse(url="/", status_code=303)
        next_path = next if next.startswith("/") and not next.startswith("//") else "/"
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": None, "next_path": next_path},
        )

    @app.post("/login", response_class=HTMLResponse)
    async def login_submit(request: Request):
        form = await request.form()
        candidate = str(form.get("access_token") or "")
        next_value = str(form.get("next") or "/")
        next_path = (
            next_value
            if next_value.startswith("/") and not next_value.startswith("//")
            else "/"
        )
        if not access.token_matches(candidate):
            return templates.TemplateResponse(
                "login.html",
                {
                    "request": request,
                    "error": "Invalid access token.",
                    "next_path": next_path,
                },
                status_code=401,
            )
        response = RedirectResponse(url=next_path, status_code=303)
        response.set_cookie(
            access.cookie_name,
            access.issue_session(token=candidate),
            max_age=access.ttl_seconds,
            httponly=True,
            secure=os.getenv("MEMORIZZ_UI_COOKIE_SECURE", "").strip().lower()
            in {"1", "true", "yes", "on"},
            samesite="strict",
            path="/",
        )
        return response

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        """Redirect to connect page or dashboard based on connection state."""
        if _state["provider"]:
            return RedirectResponse(url="/dashboard", status_code=302)
        return RedirectResponse(url="/connect", status_code=302)

    @app.get("/connect", response_class=HTMLResponse)
    async def connect_page(request: Request):
        """Show the connection page."""
        env_mongodb_uri = os.environ.get("MONGODB_URI", "")
        provider_type = os.environ.get("MEMORIZZ_BACKEND", "").strip()
        if not provider_type:
            provider_type = "mongodb" if env_mongodb_uri else "oracle"
        return templates.TemplateResponse(
            "connect.html",
            {
                "request": request,
                "error": None,
                "provider_type": provider_type,
                # Pre-fill from env vars so users don't have to re-type
                "env_oracle_user": os.environ.get("ORACLE_USER", ""),
                "has_env_oracle_password": bool(os.environ.get("ORACLE_PASSWORD", "")),
                "env_oracle_dsn": os.environ.get("ORACLE_DSN", ""),
                "env_oracle_schema": os.environ.get("ORACLE_SCHEMA", ""),
                "has_env_mongodb_uri": bool(env_mongodb_uri),
                "env_mongodb_db_name": os.environ.get("MONGODB_DB_NAME", ""),
                "env_notion_data_source_id": os.environ.get(
                    "MEMORIZZ_NOTION_DATA_SOURCE_ID", ""
                ),
                "env_filesystem_path": os.environ.get("MEMORIZZ_MEMORY_ROOT", ""),
                "has_env_notion_token": bool(os.environ.get("NOTION_TOKEN")),
                "env_notion_semantic_backend": os.environ.get(
                    "MEMORIZZ_NOTION_SEMANTIC_BACKEND", "filesystem"
                ),
                "ui_read_only": read_only_mode,
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
        # Notion records and independently selected semantic storage
        notion_token: Optional[str] = Form(None),
        notion_data_source_id: Optional[str] = Form(None),
        notion_semantic_backend: Optional[str] = Form(None),
    ):
        """Handle connection form submission."""
        error = None
        try:
            provider = None

            if provider_type == "oracle":
                resolved_oracle_password = oracle_password or os.environ.get(
                    "ORACLE_PASSWORD", ""
                )
                if not all([oracle_user, resolved_oracle_password, oracle_dsn]):
                    raise ValueError("Oracle requires user, password, and DSN")
                from ..memory_provider.oracle import OracleConfig, OracleProvider

                config = OracleConfig(
                    user=oracle_user,
                    password=resolved_oracle_password,
                    dsn=oracle_dsn,
                    schema=oracle_schema or oracle_user,
                    lazy_vector_indexes=True,
                    in_database_embedding=_resolve_oracle_embedding_mode(),
                )
                provider = OracleProvider(config)
                if config.in_database_embedding:
                    embedding_description = "Oracle in-database"
                elif config.embedding_provider:
                    dimensions = config.embedding_config.get("dimensions")
                    embedding_description = str(config.embedding_provider)
                    if dimensions:
                        embedding_description += f" ({dimensions} dimensions)"
                else:
                    embedding_description = "Disabled"
                _state["connection_info"] = {
                    "user": oracle_user,
                    "dsn": oracle_dsn,
                    "schema": oracle_schema or oracle_user,
                    "embedding": embedding_description,
                }
                _state["provider_secrets"] = {
                    "oracle_user": oracle_user,
                    "oracle_password": resolved_oracle_password,
                    "oracle_dsn": oracle_dsn,
                    "oracle_schema": oracle_schema or oracle_user,
                }

            elif provider_type == "mongodb":
                resolved_mongodb_uri = mongodb_uri or os.environ.get("MONGODB_URI", "")
                if not resolved_mongodb_uri:
                    raise ValueError("MongoDB requires a URI")
                from ..memory_provider.mongodb import MongoDBConfig, MongoDBProvider

                config = MongoDBConfig(
                    uri=resolved_mongodb_uri,
                    db_name=mongodb_db_name or "memorizz",
                    lazy_vector_indexes=True,
                    read_only=read_only_mode,
                )
                provider = MongoDBProvider(config)
                _state["connection_info"] = {
                    "uri": _mask_uri(resolved_mongodb_uri),
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

            elif provider_type == "notion":
                from starlette.concurrency import run_in_threadpool

                from ..memory_provider.notion.factory import (
                    create_notion_provider_from_env,
                )

                provider = await run_in_threadpool(
                    create_notion_provider_from_env,
                    token=notion_token,
                    data_source_id=notion_data_source_id,
                    semantic_backend=notion_semantic_backend,
                    read_only=read_only_mode,
                )
                _state["connection_info"] = {
                    "data_source_id": provider.config.data_source_id,
                    "semantic_backend": notion_semantic_backend
                    or os.getenv("MEMORIZZ_NOTION_SEMANTIC_BACKEND", "filesystem"),
                }
                _state["provider_secrets"] = {}

            else:
                raise ValueError(f"Unknown provider type: {provider_type}")

            if read_only_mode:
                provider = ReadOnlyProviderProxy(provider)
                _state["connection_info"]["access"] = "read-only"

            # Close existing provider if any
            if _state["provider"]:
                close_meta_harness()
                try:
                    _state["provider"].close()
                except Exception:
                    pass

            _state["provider"] = provider
            _state["provider_type"] = provider_type
            _state["read_only"] = read_only_mode
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
                    "env_oracle_user": os.environ.get("ORACLE_USER", ""),
                    "has_env_oracle_password": bool(
                        os.environ.get("ORACLE_PASSWORD", "")
                    ),
                    "env_oracle_dsn": os.environ.get("ORACLE_DSN", ""),
                    "env_oracle_schema": os.environ.get("ORACLE_SCHEMA", ""),
                    "has_env_mongodb_uri": bool(os.environ.get("MONGODB_URI", "")),
                    "env_mongodb_db_name": os.environ.get("MONGODB_DB_NAME", ""),
                    "env_notion_data_source_id": os.environ.get(
                        "MEMORIZZ_NOTION_DATA_SOURCE_ID", ""
                    ),
                    "has_env_notion_token": bool(os.environ.get("NOTION_TOKEN")),
                    "env_notion_semantic_backend": os.environ.get(
                        "MEMORIZZ_NOTION_SEMANTIC_BACKEND", "filesystem"
                    ),
                    "ui_read_only": read_only_mode,
                },
            )

    @app.get("/disconnect")
    async def disconnect():
        """Disconnect from the current provider."""
        close_meta_harness()
        if _state["provider"]:
            try:
                _state["provider"].close()
            except Exception:
                pass
        _state["provider"] = None
        _state["provider_type"] = None
        _state["connection_info"] = {}
        _state["provider_secrets"] = {}
        _state["read_only"] = False
        return RedirectResponse(url="/connect", status_code=302)

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard(request: Request, window: str = "7d", load: bool = False):
        """Show live run health for the platform, then its inventory."""
        if not _state["provider"]:
            return RedirectResponse(url="/connect", status_code=302)

        from starlette.concurrency import run_in_threadpool

        from . import dashboard as dash

        stats = await run_in_threadpool(_get_memory_stats)
        context = await run_in_threadpool(
            dash.build_dashboard_context,
            request,
            window,
            automation_store=_get_automation_store_for_ui(),
            load=load,
        )
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "provider_type": _state["provider_type"],
                "connection_info": _state["connection_info"],
                "stats": stats,
                "active_page": "dashboard",
                **context,
                **dash.template_helpers(),
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
                "settings_sections": (sections := _build_settings_sections()),
                "settings_summary": settings_summary(sections),
                "env_file_path": str(_resolve_env_file().expanduser().absolute()),
                "config_warnings": [],
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
        config_warnings = []
        if updates:
            try:
                validate_env_updates(updates)
                for key in ("MEMORIZZ_BACKEND", "MEMORIZZ_NOTION_SEMANTIC_BACKEND"):
                    if key in updates:
                        field = next(
                            field for field in SETTINGS_FIELDS if field["env"] == key
                        )
                        if updates[key] not in {
                            option["value"] for option in field["options"]
                        }:
                            raise ValueError(
                                "Choose a supported memory/vector backend from the dropdown."
                            )
                if "MEMORIZZ_NOTION_DATA_SOURCE_ID" in updates:
                    try:
                        updates["MEMORIZZ_NOTION_DATA_SOURCE_ID"] = str(
                            uuid.UUID(updates["MEMORIZZ_NOTION_DATA_SOURCE_ID"])
                        )
                    except ValueError:
                        raise ValueError(
                            "Enter a Notion data-source UUID; use memorizz notion connect to resolve database URLs."
                        ) from None
                config_warnings = env_override_warnings(updates)
            except ValueError as exc:
                error = str(exc)
            if error is None:
                env_error = _apply_env_updates(updates)
                if env_error:
                    message = f"Applied {len(updates)} setting(s) to this process, but NOT saved. Existing connections are unchanged."
                    error = f"Failed to update .env: {env_error}"
                else:
                    message = f"Saved {len(updates)} setting(s). Reconnect existing clients to use changed credentials; restart CLI/MCP processes to load saved defaults."
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
                "settings_sections": (sections := _build_settings_sections()),
                "settings_summary": settings_summary(sections),
                "env_file_path": str(_resolve_env_file().expanduser().absolute()),
                "config_warnings": config_warnings,
                "message": message,
                "error": error,
                "active_page": "settings",
            },
        )

    # ------------------------------------------------------------------
    # Automations UI
    # ------------------------------------------------------------------

    def _parse_whatsapp_recipients(value: Any) -> List[str]:
        text = _to_text(value)
        if not text.strip():
            return []
        import re

        raw_lines = text.replace(",", "\n").splitlines()
        recipients: List[str] = []
        seen = set()
        for raw in raw_lines:
            token = _to_text(raw).strip()
            if not token:
                continue
            if token.lower().startswith("whatsapp:"):
                token = token.split(":", 1)[1].strip()
            token = re.sub(r"[\s\-()]", "", token)
            if token and not token.startswith("+") and token.isdigit():
                token = f"+{token}"
            if not (token.startswith("+") and token[1:].isdigit()):
                continue
            normalized = f"whatsapp:{token}"
            if normalized in seen:
                continue
            seen.add(normalized)
            recipients.append(normalized)
        return recipients

    def _automation_agent_options() -> List[Dict[str, str]]:
        """Saved agents an automation can run, for the form's picker."""
        try:
            agents = _list_agents()
        except Exception:
            agents = []
        options = []
        for agent in agents:
            aid = _extract_agent_identifier(agent)
            if aid:
                options.append(
                    {"agent_id": aid, "label": _extract_agent_persona_name(agent)}
                )
        return options

    def _parse_automation_form(form: Any, *, require_agent: bool) -> Dict[str, Any]:
        """The automation form's values, the next run time, and the first
        validation error (None when valid)."""
        from ..automation.schedule import (
            compute_next_run_at,
            utcnow,
            validate_timezone_name,
        )

        values: Dict[str, Any] = {
            "agent_id": _to_text(form.get("agent_id")).strip(),
            "name": _to_text(form.get("name")).strip(),
            "enabled": _parse_bool(form.get("enabled")),
            "schedule_type": _to_text(form.get("schedule_type")).strip().lower(),
            "cron_expr": _to_text(form.get("cron_expr")).strip(),
            "interval_seconds_raw": _to_text(form.get("interval_seconds")).strip(),
            "interval_seconds": None,
            "timezone": _to_text(form.get("timezone")).strip(),
            "query_template": _to_text(form.get("query_template")),
            "memory_id": _to_text(form.get("memory_id")).strip(),
            "whatsapp_to": _parse_whatsapp_recipients(form.get("whatsapp_to")),
            "delivery_channel": _to_text(form.get("delivery_channel")).strip()
            or "in_chat",
            "now_utc": utcnow(),
            "next_run_at": None,
        }
        error: Optional[str] = None
        if require_agent and not values["agent_id"]:
            error = "agent_id is required."
        elif not values["name"]:
            error = "name is required."
        elif not values["timezone"]:
            error = "timezone is required."
        else:
            try:
                validate_timezone_name(values["timezone"])
            except Exception as exc:
                error = str(exc)

        if not error and values["schedule_type"] == "interval":
            try:
                values["interval_seconds"] = int(values["interval_seconds_raw"] or "0")
            except Exception:
                values["interval_seconds"] = 0
            if not values["interval_seconds"] or values["interval_seconds"] <= 0:
                error = "interval_seconds must be a positive integer."

        if not error and values["schedule_type"] == "cron" and not values["cron_expr"]:
            error = "cron_expr is required for cron schedules."

        if not error and not values["query_template"].strip():
            error = "query_template is required."

        if not error:
            try:
                values["next_run_at"] = compute_next_run_at(
                    schedule_type=values["schedule_type"],
                    cron_expr=values["cron_expr"] or None,
                    interval_seconds=values["interval_seconds"],
                    tz_name=values["timezone"],
                    after_utc=values["now_utc"],
                )
            except Exception as exc:
                error = str(exc)

        if not values["memory_id"]:
            values["memory_id"] = str(uuid.uuid4())
        values["error"] = error
        # Delivery goes to WhatsApp only with recipients; in-chat otherwise.
        whatsapp = (
            values["delivery_channel"] == "whatsapp_twilio" and values["whatsapp_to"]
        )
        values["delivery_type"] = (
            "whatsapp_twilio"
            if whatsapp
            else "in_chat"
            if values["delivery_channel"] == "in_chat"
            else None
        )
        values["delivery_config"] = (
            {"whatsapp_to": values["whatsapp_to"]} if whatsapp else {}
        )
        return values

    def _automation_form_error(
        request: Request,
        values: Dict[str, Any],
        *,
        agent_id: str,
        active_agent_id: Optional[str],
        job_id: str = "",
    ):
        """Show the automation form again with what was entered and why it
        was refused."""
        return templates.TemplateResponse(
            "automation_form.html",
            {
                "request": request,
                "provider_type": _state["provider_type"],
                "connection_info": _state["connection_info"],
                "agents_nav": _build_agent_nav_items(active_agent_id=active_agent_id),
                "active_agent_id": active_agent_id,
                "active_page": "automations",
                "form_title": "Edit Automation" if job_id else "Create Automation",
                "form_action": f"/automations/{job_id}/edit"
                if job_id
                else "/automations",
                "is_edit": bool(job_id),
                "error": values["error"],
                "job_id": job_id,
                "agent_options": _automation_agent_options(),
                "agent_id": agent_id,
                "name": values["name"],
                "enabled": values["enabled"],
                "schedule_type": values["schedule_type"],
                "cron_expr": values["cron_expr"],
                "interval_seconds": values["interval_seconds_raw"],
                "timezone": values["timezone"],
                "query_template": values["query_template"],
                "memory_id": values["memory_id"],
                "whatsapp_to": "\n".join(values["whatsapp_to"]),
            },
        )

    @app.get("/automations", response_class=HTMLResponse)
    async def automations_page(request: Request, agent_id: Optional[str] = None):
        """List automation jobs for the connected provider."""
        if not _state["provider"]:
            return RedirectResponse(url="/connect", status_code=302)

        from datetime import datetime, timezone

        from . import dashboard as dash
        from .execution_view import AUTOMATION_RECENT_RUNS, build_automation_monitor

        selected_agent_id = _to_text(agent_id).strip() or None
        store = _get_automation_store_for_ui()
        jobs: List[Dict[str, Any]] = []
        error: Optional[str] = None

        if store is None:
            error = "Automations are unavailable for the current provider."
        else:
            try:
                rows = store.list_jobs(agent_id=selected_agent_id, enabled=None)
                jobs = [job.model_dump() for job in rows]
                # Convert next_run_at from UTC to the job's configured timezone
                for job_dict in jobs:
                    try:
                        tz_name = job_dict.get("timezone")
                        nra = job_dict.get("next_run_at")
                        if tz_name and nra and ZoneInfo is not None:
                            tz = ZoneInfo(tz_name)
                            if hasattr(nra, "astimezone"):
                                job_dict["next_run_at"] = nra.astimezone(tz)
                    except Exception:
                        pass
                # Attach latest run to each job for inline preview
                for job_dict in jobs:
                    try:
                        runs = store.list_runs(
                            job_dict["job_id"], limit=AUTOMATION_RECENT_RUNS
                        )
                        job_dict["recent_runs"] = [
                            run.model_dump(mode="json") for run in runs
                        ]
                        if runs:
                            run_dict = runs[0].model_dump(mode="json")
                            # Convert run timestamps to the job's timezone
                            tz_name = job_dict.get("timezone")
                            if tz_name and ZoneInfo is not None:
                                try:
                                    tz = ZoneInfo(tz_name)
                                    for ts_field in (
                                        "finished_at",
                                        "started_at",
                                        "scheduled_for",
                                    ):
                                        val = getattr(runs[0], ts_field, None)
                                        if val and hasattr(val, "astimezone"):
                                            run_dict[ts_field] = val.astimezone(
                                                tz
                                            ).isoformat()
                                except Exception:
                                    pass
                            job_dict["latest_run"] = run_dict
                        else:
                            job_dict["latest_run"] = None
                    except Exception:
                        job_dict["latest_run"] = None
            except Exception as exc:
                message = str(exc)
                if "ORA-00942" in message and "AUTOMATION_JOBS" in message:
                    message = (
                        message
                        + " Hint: run `memorizz oracle setup-schema` "
                        + "to apply schema updates without dropping your user."
                    )
                error = message

        agent_options = _automation_agent_options()

        now = datetime.now(timezone.utc)
        return templates.TemplateResponse(
            "automations.html",
            {
                "request": request,
                "provider_type": _state["provider_type"],
                "connection_info": _state["connection_info"],
                "agents_nav": _build_agent_nav_items(active_agent_id=selected_agent_id),
                "active_agent_id": selected_agent_id,
                "active_page": "automations",
                "error": error,
                "jobs": jobs,
                "selected_agent_id": selected_agent_id or "",
                "agent_options": agent_options,
                "monitor": build_automation_monitor(
                    jobs,
                    agent_names={o["agent_id"]: o["label"] for o in agent_options},
                    now=now,
                    runs_per_job=AUTOMATION_RECENT_RUNS,
                ),
                "generated_at": now,
                **dash.template_helpers(),
            },
        )

    @app.get("/automations/new", response_class=HTMLResponse)
    async def automations_create_page(request: Request, agent_id: Optional[str] = None):
        """Render create automation form."""
        if not _state["provider"]:
            return RedirectResponse(url="/connect", status_code=302)

        store = _get_automation_store_for_ui()
        if store is None:
            return RedirectResponse(url="/automations", status_code=302)

        agent_options = _automation_agent_options()

        default_agent_id = _to_text(agent_id).strip() if agent_id else ""
        if default_agent_id and not any(
            opt["agent_id"] == default_agent_id for opt in agent_options
        ):
            default_agent_id = ""

        default_timezone = _to_text(
            os.environ.get("MEMORIZZ_DEFAULT_TIMEZONE", "")
        ).strip()

        return templates.TemplateResponse(
            "automation_form.html",
            {
                "request": request,
                "provider_type": _state["provider_type"],
                "connection_info": _state["connection_info"],
                "agents_nav": _build_agent_nav_items(
                    active_agent_id=default_agent_id or None
                ),
                "active_agent_id": default_agent_id or None,
                "active_page": "automations",
                "form_title": "Create Automation",
                "form_action": "/automations",
                "is_edit": False,
                "error": None,
                "job_id": "",
                "agent_options": agent_options,
                "agent_id": default_agent_id,
                "name": "",
                "enabled": True,
                "schedule_type": "cron",
                "cron_expr": "",
                "interval_seconds": "",
                "timezone": default_timezone,
                "query_template": "",
                "memory_id": "",
                "whatsapp_to": "",
                "delivery_channel": "in_chat",
            },
        )

    @app.post("/automations", response_class=HTMLResponse)
    async def automations_create_submit(request: Request):
        """Create an automation job."""
        if not _state["provider"]:
            return RedirectResponse(url="/connect", status_code=302)

        store = _get_automation_store_for_ui()
        if store is None:
            return RedirectResponse(url="/automations", status_code=302)

        from ..automation.models import AutomationJob

        values = _parse_automation_form(await request.form(), require_agent=True)
        if values["error"]:
            return _automation_form_error(
                request,
                values,
                agent_id=values["agent_id"],
                active_agent_id=values["agent_id"] or None,
            )

        job = AutomationJob(
            job_id=str(uuid.uuid4()),
            agent_id=values["agent_id"],
            name=values["name"],
            enabled=values["enabled"],
            schedule_type=values["schedule_type"],  # type: ignore[arg-type]
            cron_expr=values["cron_expr"] or None,
            interval_seconds=values["interval_seconds"],
            timezone=values["timezone"],
            start_at=values["now_utc"],
            next_run_at=values["next_run_at"],  # type: ignore[arg-type]
            action_type="agent_query",
            action_config={
                "query_template": values["query_template"],
                "memory_id": values["memory_id"],
            },
            delivery_type=values["delivery_type"],
            delivery_config=values["delivery_config"],
        )

        created = store.create_job(job)
        return RedirectResponse(url=f"/automations/{created.job_id}", status_code=302)

    @app.get("/automations/{job_id}/edit", response_class=HTMLResponse)
    async def automations_edit_page(request: Request, job_id: str):
        """Render edit automation form."""
        if not _state["provider"]:
            return RedirectResponse(url="/connect", status_code=302)

        store = _get_automation_store_for_ui()
        if store is None:
            return RedirectResponse(url="/automations", status_code=302)

        job = store.get_job(_to_text(job_id).strip())
        if not job:
            raise HTTPException(status_code=404, detail="Automation job not found")

        agent_options = _automation_agent_options()

        query_template_value = _to_text(job.action_config.get("query_template"))
        memory_id_value = _to_text(job.action_config.get("memory_id")).strip()
        whatsapp_list = job.delivery_config.get("whatsapp_to") or []
        if isinstance(whatsapp_list, str):
            whatsapp_lines = whatsapp_list
        elif isinstance(whatsapp_list, list):
            whatsapp_lines = "\n".join(
                _to_text(item).strip()
                for item in whatsapp_list
                if _to_text(item).strip()
            )
        else:
            whatsapp_lines = ""

        return templates.TemplateResponse(
            "automation_form.html",
            {
                "request": request,
                "provider_type": _state["provider_type"],
                "connection_info": _state["connection_info"],
                "agents_nav": _build_agent_nav_items(active_agent_id=job.agent_id),
                "active_agent_id": job.agent_id,
                "active_page": "automations",
                "form_title": "Edit Automation",
                "form_action": f"/automations/{job.job_id}/edit",
                "is_edit": True,
                "error": None,
                "job_id": job.job_id,
                "agent_options": agent_options,
                "agent_id": job.agent_id,
                "name": job.name,
                "enabled": bool(job.enabled),
                "schedule_type": job.schedule_type,
                "cron_expr": _to_text(job.cron_expr).strip(),
                "interval_seconds": _to_text(job.interval_seconds).strip(),
                "timezone": job.timezone,
                "query_template": query_template_value,
                "memory_id": memory_id_value,
                "whatsapp_to": whatsapp_lines,
                "delivery_channel": job.delivery_type
                if job.delivery_type in ("in_chat", "whatsapp_twilio")
                else "in_chat",
            },
        )

    @app.post("/automations/{job_id}/edit", response_class=HTMLResponse)
    async def automations_edit_submit(request: Request, job_id: str):
        """Update an automation job."""
        if not _state["provider"]:
            return RedirectResponse(url="/connect", status_code=302)

        store = _get_automation_store_for_ui()
        if store is None:
            return RedirectResponse(url="/automations", status_code=302)

        existing = store.get_job(_to_text(job_id).strip())
        if not existing:
            raise HTTPException(status_code=404, detail="Automation job not found")

        values = _parse_automation_form(await request.form(), require_agent=False)
        if values["error"]:
            return _automation_form_error(
                request,
                values,
                agent_id=existing.agent_id,
                active_agent_id=existing.agent_id,
                job_id=existing.job_id,
            )

        patch = {
            "name": values["name"],
            "enabled": values["enabled"],
            "schedule_type": values["schedule_type"],
            "cron_expr": values["cron_expr"] or None,
            "interval_seconds": values["interval_seconds"],
            "timezone": values["timezone"],
            "next_run_at": values["next_run_at"],
            "action_type": "agent_query",
            "action_config": {
                "query_template": values["query_template"],
                "memory_id": values["memory_id"],
            },
            "delivery_type": values["delivery_type"],
            "delivery_config": values["delivery_config"],
            "locked_by": None,
            "lock_expires_at": None,
        }
        store.update_job(existing.job_id, patch)
        return RedirectResponse(url=f"/automations/{existing.job_id}", status_code=302)

    @app.get("/automations/{job_id}", response_class=HTMLResponse)
    async def automations_detail_page(request: Request, job_id: str):
        """Show automation job detail + run history."""
        if not _state["provider"]:
            return RedirectResponse(url="/connect", status_code=302)

        store = _get_automation_store_for_ui()
        if store is None:
            return RedirectResponse(url="/automations", status_code=302)

        job = store.get_job(_to_text(job_id).strip())
        if not job:
            raise HTTPException(status_code=404, detail="Automation job not found")

        runs = []
        error = None
        try:
            runs = store.list_runs(job.job_id, limit=50)
        except Exception as exc:
            error = str(exc)
            runs = []

        error_param = _to_text(request.query_params.get("error")).strip().lower()
        if error_param == "confirm_required":
            error = error or "Confirmation is required to delete this automation."
        elif error_param == "job_locked":
            error = error or (
                "This automation is currently locked by another worker. "
                "If it's actively running, wait a moment and try Run now again."
            )

        return templates.TemplateResponse(
            "automation_detail.html",
            {
                "request": request,
                "provider_type": _state["provider_type"],
                "connection_info": _state["connection_info"],
                "agents_nav": _build_agent_nav_items(active_agent_id=job.agent_id),
                "active_agent_id": job.agent_id,
                "active_page": "automations",
                "error": error,
                "job": job.model_dump(),
                "runs": [run.model_dump() for run in runs],
            },
        )

    @app.post("/automations/{job_id}/pause")
    async def automations_pause(job_id: str):
        if not _state["provider"]:
            return RedirectResponse(url="/connect", status_code=302)
        store = _get_automation_store_for_ui()
        if store is None:
            return RedirectResponse(url="/automations", status_code=302)
        store.pause_job(_to_text(job_id).strip())
        return RedirectResponse(url=f"/automations/{job_id}", status_code=302)

    @app.post("/automations/{job_id}/resume")
    async def automations_resume(job_id: str):
        if not _state["provider"]:
            return RedirectResponse(url="/connect", status_code=302)
        store = _get_automation_store_for_ui()
        if store is None:
            return RedirectResponse(url="/automations", status_code=302)
        store.resume_job(_to_text(job_id).strip())
        return RedirectResponse(url=f"/automations/{job_id}", status_code=302)

    @app.post("/automations/{job_id}/run-now")
    async def automations_run_now(job_id: str):
        if not _state["provider"]:
            return RedirectResponse(url="/connect", status_code=302)
        store = _get_automation_store_for_ui()
        if store is None:
            return RedirectResponse(url="/automations", status_code=302)
        from ..automation.schedule import utcnow
        from ..automation.worker import execute_claimed_job

        safe_job_id = _to_text(job_id).strip()
        now_utc = utcnow()
        lease_seconds = int(
            str(os.environ.get("MEMORIZZ_AUTOMATIONS_LEASE_SECONDS", "120") or "120")
        )
        worker_id = f"ui-run-now:{os.getpid()}:{uuid.uuid4().hex[:8]}"

        if not hasattr(store, "claim_job"):
            # Fallback behavior: enqueue immediate execution for an external worker.
            store.update_job(
                safe_job_id,
                {
                    "next_run_at": now_utc,
                    "enabled": True,
                    "locked_by": None,
                    "lock_expires_at": None,
                },
            )
            return RedirectResponse(url=f"/automations/{safe_job_id}", status_code=302)

        job = store.claim_job(
            safe_job_id,
            worker_id=worker_id,
            now_utc=now_utc,
            lease_seconds=lease_seconds,
            force_enable=True,
        )
        if job is None:
            return RedirectResponse(
                url=f"/automations/{safe_job_id}?error=job_locked", status_code=302
            )

        run = store.start_run(job, now_utc, worker_id)

        def _run_in_background() -> None:
            execute_claimed_job(
                job,
                run,
                scheduled_for=now_utc,
                memory_provider=_state["provider"],
                store=store,
            )

        threading.Thread(
            target=_run_in_background,
            name=f"memorizz-automation-run-now:{safe_job_id[:8]}",
            daemon=True,
        ).start()

        return RedirectResponse(url=f"/automations/{safe_job_id}", status_code=302)

    @app.post("/automations/{job_id}/delete")
    async def automations_delete(request: Request, job_id: str):
        if not _state["provider"]:
            return RedirectResponse(url="/connect", status_code=302)
        store = _get_automation_store_for_ui()
        if store is None:
            return RedirectResponse(url="/automations", status_code=302)
        form = await request.form()
        confirm = _parse_bool(form.get("confirm"))
        if not confirm:
            return RedirectResponse(
                url=f"/automations/{job_id}?error=confirm_required", status_code=302
            )
        store.delete_job(_to_text(job_id).strip())
        return RedirectResponse(url="/automations", status_code=302)

    # -------------------------------------------------------------------------
    # API Routes (for AJAX/HTMX)
    # -------------------------------------------------------------------------

    @app.get("/api/persona-presets")
    async def api_persona_presets():
        """Return the built-in persona presets sourced from RoleType + PREDEFINED_INFO.

        Used by the agent config UI to populate the "Load built-in preset"
        dropdown. The shape is intentionally minimal so the frontend can
        fill the form fields directly.
        """
        from ..long_term.semantic.persona.role_type import PREDEFINED_INFO, RoleType

        presets = []
        for role_type in RoleType:
            info = PREDEFINED_INFO.get(role_type, {})
            presets.append(
                {
                    "key": role_type.name,
                    "role": role_type.value,
                    "goals": info.get("goals", ""),
                    "background": info.get("background", ""),
                }
            )
        return {"presets": presets}

    @app.get("/api/personas")
    async def api_list_personas():
        """List saved personas from the PERSONAS collection.

        Powers the "Load saved persona" dropdown in the agent config UI.
        Returns a compact serialization (no embeddings) that the form can
        use to pre-fill the persona fields and carry the ``persona_id``
        forward via a hidden input.
        """
        if not _state["provider"]:
            raise HTTPException(status_code=400, detail="Not connected")

        from ..enums.memory_type import MemoryType

        try:
            raw = _state["provider"].list_all(memory_store_type=MemoryType.PERSONAS)
        except Exception as exc:
            logger.error("Failed to list personas: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

        personas = []
        for doc in raw or []:
            if not isinstance(doc, dict):
                continue
            storage_id = doc.get("_id") or doc.get("id") or doc.get("storage_id")
            if storage_id is not None:
                storage_id = str(storage_id)
            history = doc.get("evolution_history") or []
            personas.append(
                {
                    "id": storage_id,
                    "persona_id": doc.get("persona_id"),
                    "name": doc.get("name", ""),
                    "role": doc.get("role", ""),
                    "goals": doc.get("goals", ""),
                    "background": doc.get("background", ""),
                    "version": int(doc.get("version") or 1),
                    "created_at": doc.get("created_at"),
                    "updated_at": doc.get("updated_at"),
                    "history_count": len(history) if isinstance(history, list) else 0,
                }
            )
        personas.sort(
            key=lambda p: (p.get("updated_at") or p.get("created_at") or ""),
            reverse=True,
        )
        return {"personas": personas, "count": len(personas)}

    @app.get("/api/automations/{job_id}/runs")
    async def api_automation_runs(job_id: str, limit: int = 50):
        """JSON endpoint for polling automation run history."""
        if not _state["provider"]:
            raise HTTPException(status_code=400, detail="Not connected")
        store = _get_automation_store_for_ui()
        if store is None:
            raise HTTPException(status_code=400, detail="Automation store unavailable")
        job = store.get_job(_to_text(job_id).strip())
        if not job:
            raise HTTPException(status_code=404, detail="Automation job not found")
        runs = store.list_runs(job.job_id, limit=min(limit, 50))
        return {
            "job": job.model_dump(mode="json"),
            "runs": [r.model_dump(mode="json") for r in runs],
        }

    @app.get("/api/agents/{agent_id}/automation-runs")
    async def api_agent_automation_runs(agent_id: str):
        """JSON endpoint returning recent automation runs and jobs for an agent."""
        if not _state["provider"]:
            raise HTTPException(status_code=400, detail="Not connected")
        store = _get_automation_store_for_ui()
        if store is None:
            return {"runs": [], "jobs": []}
        try:
            jobs = store.list_jobs(agent_id=_to_text(agent_id).strip(), enabled=None)
        except Exception:
            return {"runs": [], "jobs": []}
        result = []
        jobs_result = []
        for job in jobs[:10]:
            schedule_label = ""
            if job.schedule_type == "interval" and job.interval_seconds:
                schedule_label = f"interval {job.interval_seconds}s"
            elif job.schedule_type == "cron" and job.cron_expr:
                schedule_label = f"cron {job.cron_expr}"
            next_run_local = None
            if job.next_run_at:
                try:
                    if ZoneInfo is not None and job.timezone:
                        next_run_local = job.next_run_at.astimezone(
                            ZoneInfo(job.timezone)
                        ).isoformat()
                    else:
                        next_run_local = job.next_run_at.isoformat()
                except Exception:
                    next_run_local = job.next_run_at.isoformat()
            jobs_result.append(
                {
                    "job_id": job.job_id,
                    "name": job.name,
                    "enabled": job.enabled,
                    "schedule": schedule_label,
                    "next_run_at": next_run_local,
                }
            )
            try:
                runs = store.list_runs(job.job_id, limit=5)
            except Exception:
                continue
            for run in runs:
                payload = run.result_payload or {}
                result.append(
                    {
                        "job_id": job.job_id,
                        "job_name": job.name,
                        "run_id": run.run_id,
                        "status": run.status,
                        "scheduled_for": run.scheduled_for.isoformat()
                        if run.scheduled_for
                        else None,
                        "finished_at": run.finished_at.isoformat()
                        if run.finished_at
                        else None,
                        "memory_id": payload.get("memory_id", ""),
                        "response_snippet": (str(payload.get("response", ""))[:120])
                        if payload.get("response")
                        else "",
                    }
                )
        result.sort(key=lambda r: r.get("scheduled_for") or "", reverse=True)
        return {"runs": result[:20], "jobs": jobs_result}

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
                "current_value": current_value
                if field.get("field_type") in {"text", "timezone", "checkbox", "select"}
                else "",
            }
            if field.get("field_type") == "checkbox":
                enriched["checkbox_checked"] = current_value == "1"
            fields.append(enriched)
        enriched_section = {**section, "fields": fields}
        if str(section.get("title") or "").lower() == "sandbox providers":
            enriched_section["provider_checks"] = _build_sandbox_provider_checks()
        sections.append(enriched_section)
    return sections


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


def _apply_env_updates(updates: Dict[str, str]) -> Optional[str]:
    """Apply updates to process env and persist to the canonical .env.

    Delegates to memorizz._env_io so the UI and CLI write the exact same file.
    """
    err = _shared_apply_env_updates(updates)
    if err:
        logger.warning(f"Failed to update .env: {err}")
    return err


def _get_automation_store_for_ui():
    """Return the automation store for the connected provider, if supported."""
    if not _state["provider"]:
        return None
    try:
        from ..automation.store.factory import get_automation_store

        return get_automation_store(_state["provider"])
    except Exception:
        return None


def _get_memory_stats() -> Dict[str, int]:
    """Get counts for each memory type."""
    from ..enums.memory_type import MemoryType

    stats = {}
    if not _state["provider"]:
        return stats

    if _state.get("provider_type") == "notion":
        # Notion has no cheap COUNT endpoint. Display explicitly labelled local
        # journal counts, not thirteen full remote scans on every page load.
        return _state["provider"].index_status()["tracked_by_memory_type"]

    # Database providers count from collection metadata; loading every record
    # of every memory type on each page view is not safe on production stores.
    estimate = getattr(_state["provider"], "estimate_count", None)
    for mem_type in MemoryType:
        try:
            if callable(estimate):
                stats[mem_type.value] = int(estimate(mem_type) or 0)
                continue
            items = _state["provider"].list_all(mem_type)
            stats[mem_type.value] = len(items) if items else 0
        except Exception:
            stats[mem_type.value] = 0

    # Automation jobs use a separate store; count them independently.
    try:
        store = _get_automation_store_for_ui()
        if store is not None:
            stats["automations"] = len(store.list_jobs(enabled=None))
        else:
            stats["automations"] = 0
    except Exception:
        stats["automations"] = 0

    return stats


def run_server(host: str = "127.0.0.1", port: int = 8765):
    """Run the Memorizz UI server."""
    import uvicorn

    app = create_app()
    print("\n🧠 Memorizz Local UI")
    print(f"   Running at: http://{host}:{port}")
    print("   Press Ctrl+C to stop\n")
    uvicorn.run(app, host=host, port=port, log_level="info")

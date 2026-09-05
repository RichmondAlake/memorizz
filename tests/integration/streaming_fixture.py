"""Synthetic, barrier-controlled application. Never loads credentials/models."""

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from memorizz.memagent import MemAgent
from memorizz.streaming import check_cancelled

ROOT = Path(os.environ["MEMORIZZ_STREAM_FIXTURE_DIR"])
ROOT.mkdir(parents=True, exist_ok=True)


class Provider:
    model = "synthetic-stream-barrier"

    def get_config(self):
        return {"provider": "openai", "model": self.model}

    def generate_stream(self, messages, tools=None):
        try:
            yield {"type": "content", "content": "Hello "}
            (ROOT / "provider_waiting").touch()
            deadline = time.monotonic() + 30
            while not (ROOT / "release").exists():
                check_cancelled()
                if time.monotonic() > deadline:
                    raise RuntimeError("Fixture barrier timed out")
                time.sleep(0.01)
            if (ROOT / "fail").exists():
                raise RuntimeError("PRIVATE fixture provider failure")
            yield {"type": "content", "content": "世界 \n"}
            (ROOT / "provider_complete").touch()
            yield {"type": "done", "content": "Hello 世界 \n"}
        finally:
            (ROOT / "provider_closed").touch()

    def get_last_usage(self):
        return None

    def get_context_window_tokens(self):
        return 8192


def agent():
    instance = MemAgent(
        model=Provider(), instruction="Synthetic fixture.", auto_register=False
    )
    instance.agent_id = "stream-fixture"
    instance._build_context = lambda *a, **k: {}
    instance._build_system_prompt = lambda: "Synthetic fixture."
    instance._build_llm_tools = lambda *a, **k: []
    instance._init_workflow_capture = lambda *a, **k: None

    def persist(*a, **k):
        (ROOT / "persist_started").touch()
        while (ROOT / "hold_persistence").exists() and not (
            ROOT / "release_persistence"
        ).exists():
            check_cancelled()
            time.sleep(0.01)

    instance._record_interaction = persist
    return instance


def main():
    mode = sys.argv[1]
    os.environ["MEMORIZZ_HOME"] = str(ROOT)
    if mode == "cli":
        import importlib

        cli = importlib.import_module("memorizz.cli.app")
        cli._load_env = lambda: None
        cli._build_or_wizard = lambda *a, **k: SimpleNamespace(
            agent=agent(),
            memory_id=None,
            thread_id=None,
            user_id=None,
        )
        cli.app(args=sys.argv[2:])
    elif mode == "repl":
        from rich.console import Console

        from memorizz.cli.repl import _stream_turn

        _stream_turn(
            SimpleNamespace(
                agent=agent(),
                memory_id=None,
                thread_id=None,
                user_id=None,
                console=Console(force_terminal=True, color_system=None),
            ),
            "hello",
        )
    elif mode.startswith("mcp"):
        from memorizz.approval import SQLiteApprovalStore
        from memorizz.mcp_server import (
            MemorizzMCPServerConfig,
            MemorizzRuntime,
            create_memorizz_mcp_server,
        )
        from memorizz.mcp_server.config import StaticAPIKeyGrant
        from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8792
        http = mode == "mcp-http"
        config = MemorizzMCPServerConfig(
            transport="streamable-http" if http else "stdio",
            host="127.0.0.1",
            port=port,
            public_url=f"http://127.0.0.1:{port}",
            allow_writes=True,
            allow_agent_execution=True,
            exposed_agent_ids={"stream-fixture"},
            api_key_grants=[
                StaticAPIKeyGrant(who, who + "-synthetic-stream-token-long-enough")
                for who in ("alice", "bob")
            ]
            if http
            else [],
        )
        runtime = MemorizzRuntime(
            config,
            provider=FileSystemProvider(
                FileSystemConfig(
                    root_path=ROOT / "memory",
                    embedding_provider=None,
                    lazy_vector_indexes=True,
                )
            ),
            approval_store=SQLiteApprovalStore(ROOT / "approvals.db"),
        )
        runtime._agents["stream-fixture"] = agent()
        server = create_memorizz_mcp_server(config, runtime=runtime)
        if http:
            import uvicorn

            uvicorn.run(
                server.streamable_http_app(),
                host="127.0.0.1",
                port=port,
                log_level="warning",
            )
        else:
            server.run(transport="stdio")
    elif mode == "ui":
        import uvicorn

        from memorizz.llms import llm_factory
        from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
        from memorizz.ui.app import create_app
        from memorizz.ui.state import _state

        llm_factory.create_llm_provider = lambda *a, **k: Provider()
        os.environ.update(
            MEMORIZZ_UI_AUTH_TOKEN="stream-fixture-token",
            MEMORIZZ_UI_AUTH_ACCOUNTS="{}",
            MEMORIZZ_UI_READ_ONLY="false",
        )
        provider = FileSystemProvider(
            FileSystemConfig(
                root_path=ROOT / "memory",
                embedding_provider=None,
                lazy_vector_indexes=True,
            )
        )
        instance = agent()
        # Store minimal normal agent metadata for rendering the real template.
        instance.memory_provider = provider
        instance.save()
        instance.memory_provider = None
        MemAgent.load = classmethod(lambda cls, *a, **k: agent())
        app = create_app()
        _state["provider"] = provider
        _state["provider_type"] = "filesystem"
        uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[2]), log_level="warning")
    else:
        raise ValueError(mode)


if __name__ == "__main__":
    main()

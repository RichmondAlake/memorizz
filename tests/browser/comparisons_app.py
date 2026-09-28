"""Isolated comparison UI fixture. Live tests use installed local Ollama models."""

import os
from tempfile import TemporaryDirectory

import uvicorn

from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.ui import state
from memorizz.ui.app import create_app
from memorizz.ui.routers.comparisons import stop_comparison_workers


def run():
    with TemporaryDirectory(prefix="memorizz-comparison-browser-") as root:
        os.environ.update(
            MEMORIZZ_UI_AUTH_TOKEN="memorizz-browser-fixture-token",
            MEMORIZZ_UI_AUTH_ACCOUNTS="{}",
            MEMORIZZ_UI_READ_ONLY="false",
            MEMORIZZ_UI_AUDIT_LOG=root + "/audit.jsonl",
        )
        os.environ.setdefault("MEMORIZZ_COMPARISON_HOME", root + "/comparisons")
        provider = FileSystemProvider(
            FileSystemConfig(root_path=root + "/memory", lazy_vector_indexes=True)
        )
        state._state.update(
            provider=provider, provider_type="filesystem", connection_info={}
        )
        try:
            uvicorn.run(
                create_app(),
                host="127.0.0.1",
                port=int(os.getenv("MEMORIZZ_BROWSER_TEST_PORT", "8785")),
                lifespan="off",
            )
        finally:
            stop_comparison_workers()
            provider.close()


if __name__ == "__main__":
    run()

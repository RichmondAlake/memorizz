#!/usr/bin/env python3
"""End-to-end smoke test: MemAgent + OllamaLLM(gemma4) + FileSystemProvider.

Run with the memorizz_local conda env active:
    conda activate memorizz_local
    python scripts/test_gemma4_e2e.py
"""

from __future__ import annotations

import shutil
import sys
import time
import traceback
from pathlib import Path

from memorizz import FileSystemConfig, FileSystemProvider, MemAgent
from memorizz.llms import OllamaLLM

MODEL_NAME = "gemma4"
MEM_DIR = Path("/tmp/memagent_gemma4_e2e")


def main() -> int:
    if MEM_DIR.exists():
        shutil.rmtree(MEM_DIR)
    MEM_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[setup] memory dir: {MEM_DIR}")
    memory_provider = FileSystemProvider(FileSystemConfig(root_path=MEM_DIR))

    print(f"[setup] instantiating OllamaLLM(model={MODEL_NAME!r})")
    llm = OllamaLLM(
        model=MODEL_NAME,
        host="http://localhost:11434",
        num_predict=256,
        timeout=120.0,
    )

    print("[setup] building MemAgent")
    agent = MemAgent(
        model=llm,
        memory_provider=memory_provider,
        instruction="You are a concise assistant. Answer in one sentence.",
    )

    prompt = "What is the capital of France? Answer in one short sentence."
    print(f"[run]  prompt: {prompt}")
    t0 = time.time()
    try:
        response = agent.run(prompt)
    except Exception:
        print("[fail] agent.run raised:")
        traceback.print_exc()
        return 1
    elapsed = time.time() - t0

    print(f"[run]  elapsed: {elapsed:.2f}s")
    print(f"[run]  response: {response!r}")

    if not response or not isinstance(response, str):
        print(f"[fail] expected non-empty string, got: {response!r}")
        return 1
    if "paris" not in response.lower():
        print(
            "[warn] response doesn't mention Paris — model may have hallucinated, but it did respond."
        )

    print("[ok]   end-to-end test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

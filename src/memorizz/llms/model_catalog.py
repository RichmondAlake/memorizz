# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""The models the local UI and the CLI offer by provider, and their defaults.

Plain data, so the CLI can read it without the UI. ``model_lists`` asks a
provider for its newest models live; this is the list to fall back on.
"""

from __future__ import annotations

from typing import Dict, List

LLM_MODEL_CATALOG: Dict[str, List[Dict[str, str]]] = {
    # Source: https://developers.openai.com/api/docs/models/all (2026-10-21).
    # Deprecated models are left out; any model ID still works as a custom
    # model. Pro models need the Responses API and are left out too.
    "openai": [
        {"value": "gpt-6-astra", "label": "GPT-6 Astra", "group": "Latest"},
        {"value": "gpt-6.1-sol", "label": "GPT-6.1 Sol", "group": "Latest"},
        {"value": "gpt-6-luna", "label": "GPT-6 Luna", "group": "Latest"},
        {"value": "gpt-5.6-sol", "label": "GPT-5.6 Sol", "group": "GPT-5.6"},
        {"value": "gpt-5.6-terra", "label": "GPT-5.6 Terra", "group": "GPT-5.6"},
        {"value": "gpt-5.6-luna", "label": "GPT-5.6 Luna", "group": "GPT-5.6"},
        {"value": "gpt-6-sol", "label": "GPT-6 Sol", "group": "Previous"},
        {"value": "gpt-5.5", "label": "GPT-5.5", "group": "Previous"},
        {"value": "gpt-5.4", "label": "GPT-5.4", "group": "Previous"},
        {"value": "gpt-5.4-mini", "label": "GPT-5.4 Mini", "group": "Previous"},
        {"value": "gpt-5.2", "label": "GPT-5.2", "group": "Previous"},
        {"value": "gpt-5", "label": "GPT-5", "group": "Previous"},
        {"value": "gpt-5-mini", "label": "GPT-5 Mini", "group": "Previous"},
        {"value": "gpt-5-nano", "label": "GPT-5 Nano", "group": "Previous"},
        {"value": "o3", "label": "o3", "group": "Reasoning"},
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
    # HuggingFace Hub repo IDs. Custom input still accepts arbitrary repos;
    # this list is just a convenience starting point of widely-used instruct
    # models. Availability check at /api/huggingface/installed reflects what
    # the user has actually downloaded into their HF cache.
    "huggingface": [
        # Meta Llama
        {
            "value": "meta-llama/Llama-3.3-70B-Instruct",
            "label": "Llama 3.3 70B Instruct",
            "group": "Meta",
        },
        {
            "value": "meta-llama/Llama-3.2-3B-Instruct",
            "label": "Llama 3.2 3B Instruct",
            "group": "Meta",
        },
        {
            "value": "meta-llama/Llama-3.2-1B-Instruct",
            "label": "Llama 3.2 1B Instruct",
            "group": "Meta",
        },
        {
            "value": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "label": "Llama 3.1 8B Instruct",
            "group": "Meta",
        },
        {
            "value": "meta-llama/Meta-Llama-3-8B-Instruct",
            "label": "Llama 3 8B Instruct",
            "group": "Meta",
        },
        # Alibaba Qwen
        {"value": "Qwen/Qwen3-32B", "label": "Qwen 3 32B", "group": "Alibaba"},
        {"value": "Qwen/Qwen3-8B", "label": "Qwen 3 8B", "group": "Alibaba"},
        {"value": "Qwen/Qwen3-4B", "label": "Qwen 3 4B", "group": "Alibaba"},
        {
            "value": "Qwen/Qwen2.5-7B-Instruct",
            "label": "Qwen 2.5 7B Instruct",
            "group": "Alibaba",
        },
        # Google Gemma 4 (April 2026, Apache 2.0, gated — accept license once on HF)
        {
            "value": "google/gemma-4-E2B-it",
            "label": "Gemma 4 E2B Instruct (2.3B eff)",
            "group": "Google",
        },
        {
            "value": "google/gemma-4-E4B-it",
            "label": "Gemma 4 E4B Instruct (4.5B eff)",
            "group": "Google",
        },
        {
            "value": "google/gemma-4-26B-A4B-it",
            "label": "Gemma 4 26B A4B Instruct (MoE)",
            "group": "Google",
        },
        {
            "value": "google/gemma-4-31B-it",
            "label": "Gemma 4 31B Instruct (Dense)",
            "group": "Google",
        },
        # Google Gemma 3
        {"value": "google/gemma-3-27b-it", "label": "Gemma 3 27B", "group": "Google"},
        {"value": "google/gemma-3-4b-it", "label": "Gemma 3 4B", "group": "Google"},
        {"value": "google/gemma-2-9b-it", "label": "Gemma 2 9B", "group": "Google"},
        # Mistral
        {
            "value": "mistralai/Mistral-7B-Instruct-v0.3",
            "label": "Mistral 7B Instruct v0.3",
            "group": "Mistral AI",
        },
        {
            "value": "mistralai/Mistral-Small-Instruct-2409",
            "label": "Mistral Small Instruct (24B)",
            "group": "Mistral AI",
        },
        # Microsoft Phi
        {"value": "microsoft/phi-4", "label": "Phi-4 (14B)", "group": "Microsoft"},
        {
            "value": "microsoft/Phi-4-mini-instruct",
            "label": "Phi-4 mini Instruct",
            "group": "Microsoft",
        },
        # DeepSeek R1 distills
        {
            "value": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "label": "DeepSeek R1 Distill Llama 8B",
            "group": "DeepSeek",
        },
    ],
    # Anthropic Claude models. Source: the Anthropic Models API (2026-10-21).
    "anthropic": [
        {"value": "claude-opus-5-5", "label": "Claude Opus 5.5", "group": "Latest"},
        {"value": "claude-sonnet-5-5", "label": "Claude Sonnet 5.5", "group": "Latest"},
        {"value": "claude-haiku-5-5", "label": "Claude Haiku 5.5", "group": "Latest"},
        {"value": "claude-fable-5-1", "label": "Claude Fable 5.1", "group": "Latest"},
        {"value": "claude-opus-5", "label": "Claude Opus 5", "group": "Previous"},
        {"value": "claude-sonnet-5", "label": "Claude Sonnet 5", "group": "Previous"},
        {"value": "claude-fable-5", "label": "Claude Fable 5", "group": "Previous"},
        {"value": "claude-opus-4-8", "label": "Claude Opus 4.8", "group": "Previous"},
        {"value": "claude-opus-4-7", "label": "Claude Opus 4.7", "group": "Older"},
        {"value": "claude-sonnet-4-6", "label": "Claude Sonnet 4.6", "group": "Older"},
        {"value": "claude-opus-4-6", "label": "Claude Opus 4.6", "group": "Older"},
        {
            "value": "claude-haiku-4-5-20251001",
            "label": "Claude Haiku 4.5",
            "group": "Older",
        },
        {
            "value": "claude-sonnet-4-5-20250929",
            "label": "Claude Sonnet 4.5",
            "group": "Older",
        },
    ],
    # Ollama local models. Values are the exact Ollama tags users `ollama pull`,
    # so the dropdown can never offer a model that doesn't exist on the registry
    # (the older `gemma4` / `qwen3.6` entries here resulted in 404s at chat time).
    "ollama": [
        # Meta Llama
        {"value": "llama3.3:70b", "label": "Llama 3.3 (70B)", "group": "Meta"},
        {"value": "llama3.2:3b", "label": "Llama 3.2 (3B)", "group": "Meta"},
        {"value": "llama3.2:1b", "label": "Llama 3.2 (1B)", "group": "Meta"},
        {"value": "llama3.1:8b", "label": "Llama 3.1 (8B)", "group": "Meta"},
        # Google Gemma 4 — needs Ollama daemon >= 0.22.1 (April 28 2026 release).
        # E2B/E4B are the on-device "effective-N" sizes, the 26B MoE and 31B
        # dense are the workstation tier. Tags via `ollama pull gemma4:<size>`.
        # `gemma4:latest` aliases to e4b — surfaced as its own entry because
        # that's what `ollama pull gemma4` (no tag) writes to disk.
        {
            "value": "gemma4:latest",
            "label": "Gemma 4 (latest, = E4B)",
            "group": "Google",
        },
        {"value": "gemma4:e2b", "label": "Gemma 4 (E2B, on-device)", "group": "Google"},
        {"value": "gemma4:e4b", "label": "Gemma 4 (E4B, on-device)", "group": "Google"},
        {"value": "gemma4:26b", "label": "Gemma 4 (26B A4B, MoE)", "group": "Google"},
        {"value": "gemma4:31b", "label": "Gemma 4 (31B Dense)", "group": "Google"},
        # Google Gemma 3 — multiple sizes; gemma3n is the multimodal variant.
        {"value": "gemma3:27b", "label": "Gemma 3 (27B)", "group": "Google"},
        {"value": "gemma3:12b", "label": "Gemma 3 (12B)", "group": "Google"},
        {"value": "gemma3:4b", "label": "Gemma 3 (4B)", "group": "Google"},
        {"value": "gemma3:1b", "label": "Gemma 3 (1B)", "group": "Google"},
        {
            "value": "gemma3n:e4b",
            "label": "Gemma 3n (E4B, multimodal)",
            "group": "Google",
        },
        # Alibaba Qwen 3 — chat sizes plus the MoE and coder-specialized tags.
        {"value": "qwen3:32b", "label": "Qwen 3 (32B)", "group": "Alibaba"},
        {"value": "qwen3:14b", "label": "Qwen 3 (14B)", "group": "Alibaba"},
        {"value": "qwen3:8b", "label": "Qwen 3 (8B)", "group": "Alibaba"},
        {"value": "qwen3:4b", "label": "Qwen 3 (4B)", "group": "Alibaba"},
        {"value": "qwen3:1.7b", "label": "Qwen 3 (1.7B)", "group": "Alibaba"},
        {"value": "qwen3:0.6b", "label": "Qwen 3 (0.6B)", "group": "Alibaba"},
        {
            "value": "qwen3:30b-a3b",
            "label": "Qwen 3 (30B-A3B, MoE)",
            "group": "Alibaba",
        },
        {"value": "qwen3-coder:30b", "label": "Qwen 3 Coder (30B)", "group": "Alibaba"},
        # Mistral
        {
            "value": "mistral-small:24b",
            "label": "Mistral Small (24B)",
            "group": "Mistral AI",
        },
        {
            "value": "mistral-nemo:12b",
            "label": "Mistral Nemo (12B)",
            "group": "Mistral AI",
        },
        {"value": "mistral:7b", "label": "Mistral (7B)", "group": "Mistral AI"},
        # DeepSeek R1 reasoning
        {"value": "deepseek-r1:14b", "label": "DeepSeek R1 (14B)", "group": "DeepSeek"},
        {"value": "deepseek-r1:7b", "label": "DeepSeek R1 (7B)", "group": "DeepSeek"},
        {
            "value": "deepseek-r1:1.5b",
            "label": "DeepSeek R1 (1.5B)",
            "group": "DeepSeek",
        },
        # Microsoft
        {"value": "phi4:14b", "label": "Phi-4 (14B)", "group": "Microsoft"},
    ],
    # MLX (Apple Silicon native). mlx-community/* repos ship pre-quantized
    # weights — Google explicitly recommends MLX for Gemma 4 on Macs.
    # Requires a native arm64 Python — `pip install memorizz[mlx]` will
    # fail under Rosetta. Custom repo IDs are still accepted.
    "mlx": [
        # Google Gemma 4 (4-bit MLX quants — fast on Apple Silicon)
        {
            "value": "mlx-community/gemma-4-E2B-it-4bit",
            "label": "Gemma 4 E2B 4-bit",
            "group": "Google",
        },
        {
            "value": "mlx-community/gemma-4-E4B-it-4bit",
            "label": "Gemma 4 E4B 4-bit",
            "group": "Google",
        },
        {
            "value": "mlx-community/gemma-4-26b-a4b-it-4bit",
            "label": "Gemma 4 26B A4B 4-bit (MoE)",
            "group": "Google",
        },
        # Llama
        {
            "value": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "label": "Llama 3.2 3B 4-bit",
            "group": "Meta",
        },
        {
            "value": "mlx-community/Llama-3.2-1B-Instruct-4bit",
            "label": "Llama 3.2 1B 4-bit",
            "group": "Meta",
        },
        # Qwen
        {
            "value": "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "label": "Qwen 2.5 7B 4-bit",
            "group": "Alibaba",
        },
        {
            "value": "mlx-community/Qwen2.5-3B-Instruct-4bit",
            "label": "Qwen 2.5 3B 4-bit",
            "group": "Alibaba",
        },
        # Mistral / Microsoft
        {
            "value": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
            "label": "Mistral 7B Instruct 4-bit",
            "group": "Mistral AI",
        },
        {
            "value": "mlx-community/Phi-3.5-mini-instruct-4bit",
            "label": "Phi-3.5 mini 4-bit",
            "group": "Microsoft",
        },
    ],
    # OpenAI-compatible local servers (llama.cpp's `llama-server`, LM Studio,
    # vLLM, mlx_lm.server, etc.). The model name is whatever the local server
    # exposes — we forward it verbatim. Most servers accept any string; some
    # (LM Studio) echo the loaded model id back via /v1/models.
    #
    # Two groups of entries: GGUF repos for `llama-server -hf <repo>` and
    # `mlx-community/*` repos for `python -m mlx_lm.server --model <repo>`.
    # The agent-form JS detects which family the user picked and shows the
    # right startup command in the hint.
    "local-openai": [
        # GGUF — for llama.cpp's `llama-server`, LM Studio, llamafile
        {
            "value": "ggml-org/gemma-4-E2B-it-GGUF",
            "label": "Gemma 4 E2B (GGUF, llama.cpp)",
            "group": "GGUF — llama.cpp / LM Studio",
        },
        {
            "value": "ggml-org/gemma-4-E4B-it-GGUF",
            "label": "Gemma 4 E4B (GGUF, llama.cpp)",
            "group": "GGUF — llama.cpp / LM Studio",
        },
        {
            "value": "bartowski/gemma-3-1b-it-GGUF",
            "label": "Gemma 3 1B (GGUF, llama.cpp)",
            "group": "GGUF — llama.cpp / LM Studio",
        },
        {
            "value": "bartowski/Llama-3.2-3B-Instruct-GGUF",
            "label": "Llama 3.2 3B (GGUF, llama.cpp)",
            "group": "GGUF — llama.cpp / LM Studio",
        },
        {
            "value": "bartowski/Qwen2.5-7B-Instruct-GGUF",
            "label": "Qwen 2.5 7B (GGUF, llama.cpp)",
            "group": "GGUF — llama.cpp / LM Studio",
        },
        # MLX — for `mlx_lm.server` running in a native arm64 sidecar venv.
        # Use this when memorizz itself is on an x86_64/Rosetta env: the MLX
        # process lives in a separate native arm64 Python and memorizz talks
        # to it over OpenAI-compatible HTTP. Pairs with the in-process
        # "MLX (Apple Silicon)" provider, which only works when memorizz's
        # own Python is arm64.
        {
            "value": "mlx-community/gemma-4-E2B-it-4bit",
            "label": "Gemma 4 E2B 4-bit (MLX server)",
            "group": "MLX — mlx_lm.server",
        },
        {
            "value": "mlx-community/gemma-4-E4B-it-4bit",
            "label": "Gemma 4 E4B 4-bit (MLX server)",
            "group": "MLX — mlx_lm.server",
        },
        {
            "value": "mlx-community/gemma-4-26b-a4b-it-4bit",
            "label": "Gemma 4 26B A4B 4-bit (MLX server)",
            "group": "MLX — mlx_lm.server",
        },
        {
            "value": "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "label": "Qwen 2.5 7B 4-bit (MLX server)",
            "group": "MLX — mlx_lm.server",
        },
        {
            "value": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "label": "Llama 3.2 3B 4-bit (MLX server)",
            "group": "MLX — mlx_lm.server",
        },
    ],
}

DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_LLM_MODEL_BY_PROVIDER = {
    "openai": "gpt-6.1-sol",
    "azure": "gpt-5",
    "huggingface": "meta-llama/Meta-Llama-3-8B-Instruct",
    "anthropic": "claude-opus-5-5",
    "ollama": "llama3.1:8b",
    "mlx": "mlx-community/gemma-4-E2B-it-4bit",
    "local-openai": "ggml-org/gemma-4-E2B-it-GGUF",
}


__all__ = [
    "DEFAULT_LLM_MODEL_BY_PROVIDER",
    "DEFAULT_LLM_PROVIDER",
    "LLM_MODEL_CATALOG",
]

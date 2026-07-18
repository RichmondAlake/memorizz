# Memorizz Evaluation Framework

This directory contains evaluation scripts and benchmarks for testing Memorizz's memory capabilities across various tasks and scenarios.

## Structure

```
eval/
├── README.md              # This file
├── longmemeval/           # LongMemEval benchmark evaluation
│   ├── evaluate_memorizz.py  # Main evaluation script
│   └── README.md          # LongMemEval specific documentation
└── [future benchmarks]/  # Additional evaluation frameworks
```

## Overview

The evaluation framework is designed to assess Memorizz's performance on various memory-related tasks, providing objective metrics to track improvements and compare against other agent memory systems.

## Available Benchmarks

### LongMemEval
LongMemEval is a comprehensive benchmark for evaluating long-term memory capabilities of chat assistants. It tests five core memory abilities:

1. **Information Extraction** - Recalling specific information from extensive histories
2. **Multi-Session Reasoning** - Synthesizing information across multiple conversation sessions
3. **Knowledge Updates** - Recognizing and updating changed user information over time
4. **Temporal Reasoning** - Understanding time-aware aspects of information
5. **Abstention** - Knowing when to refuse answering based on insufficient information

## Quick Start

1. Install dependencies:
```bash
pip install datasets transformers openai
```

2. Set up environment variables:
```bash
export OPENAI_API_KEY="your_openai_api_key"
export MONGODB_URI="your_mongodb_uri"
```

3. Run LongMemEval evaluation:
```bash
cd eval/longmemeval
python evaluate_memorizz.py
```

## Adding New Benchmarks

To add a new evaluation benchmark:

1. Create a new directory under `eval/`
2. Implement an evaluation script that follows the pattern in `longmemeval/evaluate_memorizz.py`
3. Update this README with documentation for your benchmark
4. Add any necessary dependencies to the project requirements

## Results

Evaluation results will be saved in JSON format with timestamps, allowing for easy tracking of performance improvements over time.

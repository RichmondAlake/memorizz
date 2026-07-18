#!/usr/bin/env python3
"""
LongMemEval Evaluation Script for Memorizz with Oracle AI Database

This script evaluates Memorizz's long-term memory capabilities using the LongMemEval benchmark.
It loads the dataset, creates Memorizz agents with Oracle AI Database as the memory provider,
processes conversations, and measures performance across five core memory abilities.

Prerequisites:
- Oracle Database 23ai or higher with AI Vector Search
- Oracle user and schema set up (use examples/setup_oracle_user.py)
- OpenAI API key for LLM and embeddings

Environment Variables:
- OPENAI_API_KEY: Required for LLM and embeddings
- ORACLE_USER: Oracle database user (default: memorizz_user)
- ORACLE_PASSWORD: Oracle database password (default: SecurePass123!)
- ORACLE_DSN: Oracle connection string (default: localhost:1521/FREEPDB1)
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the project root to Python path for local execution
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from memorizz.llms.openai import OpenAI
    from memorizz.memagent.builders import MemAgentBuilder
    from memorizz.memory_provider.oracle import OracleConfig, OracleProvider
except ImportError as e:
    try:
        from src.memorizz.llms.openai import OpenAI
        from src.memorizz.memagent.builders import MemAgentBuilder
        from src.memorizz.memory_provider.oracle import OracleConfig, OracleProvider
    except ImportError as exc:
        raise ImportError(
            "Make sure you're running from the project root and Memorizz is properly installed."
        ) from exc

logger = logging.getLogger(__name__)


class LongMemEvalEvaluator:
    """Evaluator for Memorizz using LongMemEval benchmark with Oracle AI Database."""

    def __init__(
        self,
        dataset_variant: str = "oracle",
        application_mode: Optional[str] = "assistant",
        output_dir: Optional[str] = "./results",
        verbose: bool = False,
        memory_provider: Optional[Any] = None,
        agent_template: Optional[Any] = None,
        dataset_dir: Optional[str] = None,
        evaluation_model: Optional[Any] = None,
        ingest_mode: str = "direct",
        judge_model: str = "gpt-4o",
        config_label: str = "",
        context_window_tokens: Optional[int] = None,
        disable_auto_summaries: bool = False,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initialize the evaluator with Oracle AI Database as the memory provider.

        Args:
            dataset_variant: LongMemEval variant ("oracle", "s", "m")
            application_mode: Memorizz application mode to use
            output_dir: Directory to save results
            verbose: Enable verbose logging
            ingest_mode: How haystack history enters agent memory:
                - "direct" (benchmark-correct): the dataset's user AND
                  assistant turns are written to conversation memory as-is
                  (with embeddings), exactly as LongMemEval intends. Fast —
                  no LLM calls during ingestion.
                - "run": legacy behaviour; each user turn is replayed
                  through ``agent.run()`` (the dataset's assistant replies
                  are discarded and regenerated, which loses the evidence
                  for single-session-assistant questions and costs one LLM
                  call per turn).
            judge_model: OpenAI model used for LLM-as-judge scoring.
            config_label: Free-form label recorded in results metadata —
                use it to tag A/B arms (e.g. "baseline-HEAD" vs
                "candidate-context-efficiency").
        """
        self.dataset_variant = dataset_variant
        self.application_mode = application_mode
        self.output_dir = Path(output_dir) if output_dir else Path("./results")
        self.verbose = verbose
        self.agent_template = agent_template
        self.ingest_mode = ingest_mode
        self.config_label = config_label
        self.context_window_tokens = context_window_tokens
        self.disable_auto_summaries = disable_auto_summaries
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.dataset_dir = (
            Path(dataset_dir) if dataset_dir else Path(__file__).parent / "data"
        )

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize memory provider
        self.memory_provider = memory_provider or self._init_memory_provider()

        # Initialize evaluation model for scoring
        self.eval_model = evaluation_model or OpenAI(model=judge_model)

        # Load dataset
        self.dataset = self._load_dataset()

        # Category mapping - updated to match actual LongMemEval question types
        self.categories = {
            "single-session-user": "SSU",
            "single-session-assistant": "SSA",
            "single-session-preference": "SSP",
            "multi-session": "MS",
            "temporal-reasoning": "TR",
            "knowledge-update": "KU",
        }

        if not self.application_mode:
            self.application_mode = (
                self._get_template_value("application_mode") or "assistant"
            )

        logger.info(
            "Initialized LongMemEval evaluator with variant: %s", dataset_variant
        )

    def _get_template_value(self, key: str) -> Any:
        """Fetch a value from the agent template if available."""
        if not self.agent_template:
            return None
        if isinstance(self.agent_template, dict):
            return self.agent_template.get(key)
        return getattr(self.agent_template, key, None)

    def _sanitize_template_tools(self, tools: Any) -> List[Any]:
        """Keep only executable tools from a template payload."""
        if not tools:
            return []

        if isinstance(tools, list):
            raw_tools = tools
        else:
            raw_tools = [tools]

        sanitized: List[Any] = []
        for tool in raw_tools:
            if callable(tool):
                sanitized.append(tool)
                continue
            if not isinstance(tool, dict):
                continue

            tool_type = (
                str(
                    tool.get("tool_type")
                    or tool.get("toolType")
                    or tool.get("type")
                    or ""
                )
                .strip()
                .lower()
            )
            name = str(
                tool.get("name") or (tool.get("function", {}) or {}).get("name") or ""
            ).strip()

            if tool_type == "mcp_server_config":
                continue
            if not name or name.lower() in {"unknown", "unknown_tool"}:
                continue

            sanitized.append(tool)

        return sanitized

    def _init_memory_provider(self):
        """Initialize Oracle memory provider."""
        # Get Oracle connection details from environment variables
        oracle_user = os.environ.get("ORACLE_USER", "memorizz_user")
        oracle_password = os.environ.get("ORACLE_PASSWORD", "SecurePass123!")
        oracle_dsn = os.environ.get("ORACLE_DSN", "localhost:1521/FREEPDB1")
        oracle_schema = os.environ.get("ORACLE_SCHEMA", oracle_user)
        openai_api_key = os.environ.get("OPENAI_API_KEY")

        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        try:
            config = OracleConfig(
                user=oracle_user,
                password=oracle_password,
                dsn=oracle_dsn,
                schema=oracle_schema,
                lazy_vector_indexes=False,
                embedding_provider="openai",
                embedding_config={
                    "model": "text-embedding-3-small",
                    "api_key": openai_api_key,
                },
            )
            return OracleProvider(config)
        except Exception as e:
            logger.error(f"Failed to initialize Oracle provider: {e}")
            raise

    def _load_dataset(self):
        """Load LongMemEval dataset from local files."""
        try:
            # First, try to load from local data directory
            data_dir = self.dataset_dir

            # Map dataset variants to filenames
            filename_map = {
                "oracle": "longmemeval_oracle.json",
                "s": "longmemeval_s.json",
                "m": "longmemeval_m.json",
            }

            if self.dataset_variant not in filename_map:
                raise ValueError(f"Unknown dataset variant: {self.dataset_variant}")

            filename = filename_map[self.dataset_variant]
            filepath = data_dir / filename

            if filepath.exists():
                logger.info(f"Loading dataset from local file: {filepath}")
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Return the raw data directly - no need for HuggingFace datasets
                logger.info(
                    f"Loaded LongMemEval-{self.dataset_variant.upper()} dataset with {len(data)} samples"
                )
                return data
            else:
                # If local file doesn't exist, provide instructions for downloading
                logger.error(f"Dataset file not found: {filepath}")
                logger.error("Please download the LongMemEval dataset by running:")
                logger.error("python download_dataset.py")
                logger.error(f"This will download and extract the files to: {data_dir}")
                raise FileNotFoundError(f"Dataset file not found: {filepath}")

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def _create_fresh_agent(self):
        """Create a fresh Memorizz agent for evaluation using MemAgentBuilder."""
        logger.info(
            "Creating fresh agent with specified memory provider and application mode"
        )

        default_instruction = (
            "You are a helpful assistant with excellent memory. "
            "Pay close attention to all conversations and remember important details "
            "about users and their preferences."
        )

        instruction = self._get_template_value("instruction") or default_instruction
        application_mode = self.application_mode or "assistant"

        llm_config = self._get_template_value("llm_config")
        if isinstance(llm_config, dict):
            llm_config = dict(llm_config)
            provider_name = llm_config.get("provider", "openai").lower()
            if provider_name == "openai" and "api_key" not in llm_config:
                api_key = os.environ.get("OPENAI_API_KEY")
                if api_key:
                    llm_config["api_key"] = api_key
        else:
            llm_config = None

        if llm_config is None:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required")
            llm_config = {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": openai_api_key,
            }

        builder = (
            MemAgentBuilder()
            .with_instruction(instruction)
            .with_memory_provider(self.memory_provider)
            .with_llm_config(llm_config)
            .with_application_mode(application_mode)
        )

        max_steps = self._get_template_value("max_steps")
        if max_steps:
            builder.with_max_steps(max_steps)

        tool_access = self._get_template_value("tool_access")
        if tool_access:
            builder.with_tool_access(tool_access)

        persona = self._get_template_value("persona")
        if persona:
            builder.with_persona(persona=persona)

        tools = self._get_template_value("tools")
        if tools:
            sanitized_tools = self._sanitize_template_tools(tools)
            if sanitized_tools:
                builder.with_tools(sanitized_tools)
            else:
                logger.warning(
                    "Template tools were present but none were executable; continuing without template tools."
                )

        semantic_cache = self._get_template_value("semantic_cache")
        if semantic_cache:
            cache_config = self._get_template_value("semantic_cache_config") or {}
            threshold = cache_config.get("similarity_threshold", 0.85)
            scope = cache_config.get("scope", "local")
            builder.with_semantic_cache(enabled=True, threshold=threshold, scope=scope)

        context_window_tokens = self.context_window_tokens or self._get_template_value(
            "context_window_tokens"
        )
        if context_window_tokens:
            builder.config.context_window_tokens = context_window_tokens

        agent = builder.build()

        if self.disable_auto_summaries:
            # Push the auto-summary trigger out of reach. With a constrained
            # window, usage sits above the default 80% trigger permanently;
            # on Oracle the summary-marking column doesn't exist, so at HEAD
            # (synchronous summarization + never-marked messages) this loops
            # forever re-summarizing the same units. Disabling it in BOTH
            # arms keeps the A/B focused on assembly+retrieval accuracy.
            try:
                agent._context_summary_trigger = 1_000_000.0
            except Exception:
                pass

        # Save the agent to Oracle
        agent.save()
        return agent

    def _process_conversation_history(
        self, agent, history: List[List[Dict[str, Any]]]
    ) -> None:
        """
        Process conversation history session by session to build up agent memory.

        Legacy ("run") ingestion: each USER turn is replayed through
        ``agent.run()``; the dataset's assistant replies are regenerated.

        Args:
            agent: The Memorizz agent
            history: List of conversation sessions, where each session is a list of messages
        """
        for session_idx, session in enumerate(history):
            if self.verbose:
                logger.info(f"Processing session {session_idx + 1}/{len(history)}")

            # Each session is directly a list of messages
            for message in session:
                role = message.get("role", "")
                content = message.get("content", "")

                if role == "user":
                    # Process user message through the agent to build memory
                    try:
                        response = agent.run(content)
                        if self.verbose:
                            logger.debug(f"User: {content[:100]}...")
                            logger.debug(f"Agent: {response[:100]}...")
                    except Exception as e:
                        logger.warning(f"Error processing message: {e}")
                        continue

    @staticmethod
    def _parse_session_date(raw: Any):
        """Parse a LongMemEval haystack date like '2023/04/10 (Mon) 17:50'."""
        if not raw:
            return None
        text = str(raw).strip()
        # Drop the parenthesised weekday.
        import re

        text = re.sub(r"\s*\([^)]*\)\s*", " ", text).strip()
        for fmt in ("%Y/%m/%d %H:%M", "%Y/%m/%d", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        return None

    def _ingest_history_direct(
        self,
        agent,
        history: List[List[Dict[str, Any]]],
        session_dates: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Write haystack sessions straight into conversation memory.

        Benchmark-correct ingestion: both the user AND assistant turns from
        the dataset are stored verbatim (LongMemEval questions frequently
        probe what the ASSISTANT said, which replay-based ingestion loses).
        Rows are stored with their embedding computed inline so episodic
        vector recall is ready before the question is asked — no reliance
        on background backfill timing, and identical behaviour on old and
        new memorizz code (only public/stable APIs are used).

        Session timestamps come from ``haystack_dates`` (fallback: now),
        with a per-message second offset to preserve intra-session order.

        Returns {"memory_id", "thread_id"} for the follow-up question turn.
        """
        from datetime import timedelta

        from memorizz.embeddings import get_embedding
        from memorizz.enums import Role

        memory_id, thread_id = agent._resolve_execution_state(None, None)
        session_dates = session_dates or []

        for session_idx, session in enumerate(history):
            base_ts = None
            if session_idx < len(session_dates):
                base_ts = self._parse_session_date(session_dates[session_idx])
            if base_ts is None:
                base_ts = datetime.now()

            for msg_idx, message in enumerate(session):
                role_raw = str(message.get("role", "")).strip().lower()
                content = str(message.get("content", "") or "").strip()
                if not content or role_raw not in ("user", "assistant"):
                    continue
                role = Role.USER if role_raw == "user" else Role.ASSISTANT

                unit = agent.memory_manager.create_conversation_memory_unit(
                    role=role,
                    content=content,
                    thread_id=thread_id,
                    memory_id=memory_id,
                    timestamp=base_ts + timedelta(seconds=msg_idx),
                    agent_id=agent.agent_id,
                )
                # Embed BEFORE storing (single write, provider-agnostic) so
                # episodic vector recall is queryable immediately.
                try:
                    unit.embedding = get_embedding(content)
                except Exception as exc:
                    logger.debug("Inline embedding failed: %s", exc)
                agent.memory_manager.save_memory_unit(unit, memory_id)

            if self.verbose:
                logger.info(
                    "Ingested session %d/%d (%d messages)",
                    session_idx + 1,
                    len(history),
                    len(session),
                )

        # The direct writes populated the provider behind the in-process
        # conversation cache; clear it so the question turn reloads fresh.
        try:
            agent.memory_manager.clear_conversation_cache(memory_id)
        except Exception:
            pass

        return {"memory_id": memory_id, "thread_id": thread_id}

    def _evaluate_response(
        self, question: str, agent_response: str, ground_truth: str, category: str
    ) -> Dict[str, Any]:
        """
        Evaluate agent response using GPT-4 as a judge.

        Args:
            question: The evaluation question
            agent_response: Agent's response
            ground_truth: Expected answer
            category: Question category

        Returns:
            Dictionary with evaluation results
        """
        evaluation_prompt = f"""
You are evaluating a chat assistant's response to a question about information from a long conversation history.

Question: {question}

Agent's Response: {agent_response}

Ground Truth Answer: {ground_truth}

Category: {category}

Please evaluate the agent's response on the following criteria:
1. Correctness: Does the response correctly answer the question?
2. Completeness: Does it provide sufficient detail?
3. Relevance: Is the response relevant to the question?

For categories involving abstention (when the agent should say "I don't know"), consider:
- If the ground truth indicates the information is unknown, the agent should abstain
- If the agent abstains when it should know the answer, that's incorrect
- If the agent provides an answer when it should abstain, that's incorrect

Provide your evaluation as a JSON object with:
{{
    "correct": true/false,
    "score": 0.0-1.0,
    "reasoning": "explanation of your evaluation"
}}

Only respond with the JSON object.
"""

        try:
            # Use the evaluation model - fix method name to generate_text
            eval_response = self.eval_model.generate_text(evaluation_prompt)

            # Parse JSON response (tolerate markdown code fences)
            cleaned = str(eval_response).strip()
            if cleaned.startswith("```"):
                cleaned = "\n".join(
                    line
                    for line in cleaned.splitlines()
                    if not line.strip().startswith("```")
                ).strip()
            eval_result = json.loads(cleaned)

            return {
                "correct": eval_result.get("correct", False),
                "score": eval_result.get("score", 0.0),
                "reasoning": eval_result.get("reasoning", ""),
                "agent_response": agent_response,
                "ground_truth": ground_truth,
            }

        except Exception as e:
            logger.warning(f"Error in evaluation: {e}")
            return {
                "correct": False,
                "score": 0.0,
                "reasoning": f"Evaluation error: {e}",
                "agent_response": agent_response,
                "ground_truth": ground_truth,
            }

    def _cleanup_agent(self, agent) -> None:
        """Remove evaluation agent data from the memory provider."""
        if not agent or not self.memory_provider:
            return
        if hasattr(self.memory_provider, "delete_memagent"):
            try:
                self.memory_provider.delete_memagent(agent.agent_id, cascade=True)
            except Exception as e:
                logger.warning(f"Error cleaning up agent {agent.agent_id}: {e}")

    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single sample from the dataset.

        Args:
            sample: A single evaluation sample

        Returns:
            Dictionary with evaluation results
        """
        start_time = time.time()

        # Extract sample information with correct field names
        question = sample["question"]
        ground_truth = sample["answer"]
        category = sample["question_type"]  # Changed from "category" to "question_type"

        # Handle haystack_sessions which might be a string that needs parsing
        history_raw = sample["haystack_sessions"]
        if isinstance(history_raw, str):
            try:
                import ast

                history = ast.literal_eval(history_raw)
            except:
                # If parsing fails, try json.loads
                try:
                    history = json.loads(history_raw)
                except:
                    logger.warning(
                        f"Could not parse haystack_sessions: {history_raw[:100]}..."
                    )
                    history = []
        else:
            history = history_raw

        if self.verbose:
            logger.info(f"Evaluating question: {question[:100]}...")
            logger.info(f"Category: {category}")

        # Create fresh agent for this sample
        agent = self._create_fresh_agent()

        try:
            # Build up agent memory from the haystack history
            if self.ingest_mode == "direct":
                scope = self._ingest_history_direct(
                    agent, history, sample.get("haystack_dates")
                )
                agent_response = agent.run(
                    question,
                    memory_id=scope["memory_id"],
                    thread_id=scope["thread_id"],
                )
            else:
                self._process_conversation_history(agent, history)
                agent_response = agent.run(question)

            # Capture token/cache usage for the question turn (the metric
            # the context-efficiency work targets alongside accuracy).
            usage = {}
            try:
                usage = dict(agent.model.get_last_usage() or {})
            except Exception:
                usage = {}

            # Evaluate the response
            evaluation = self._evaluate_response(
                question, agent_response, ground_truth, category
            )

            # Calculate processing time
            processing_time = time.time() - start_time

            result = {
                "question_id": sample.get("question_id"),
                "question": question,
                "category": category,
                "agent_response": agent_response,
                "ground_truth": ground_truth,
                "evaluation": evaluation,
                "usage": usage,
                "processing_time": processing_time,
                "history_length": len(history) if history else 0,
            }

            # Clean up agent
            self._cleanup_agent(agent)

            return result

        except Exception as e:
            logger.error(f"Error evaluating sample: {e}")
            # Clean up agent on error
            self._cleanup_agent(agent)

            return {
                "question": question,
                "category": category,
                "agent_response": f"Error: {e}",
                "ground_truth": ground_truth,
                "evaluation": {
                    "correct": False,
                    "score": 0.0,
                    "reasoning": f"Evaluation error: {e}",
                },
                "processing_time": time.time() - start_time,
                "history_length": len(history) if history else 0,
            }

    def evaluate(self, num_samples: int = 50) -> Dict[str, Any]:
        """
        Run evaluation on specified number of samples.

        Args:
            num_samples: Number of samples to evaluate

        Returns:
            Dictionary with comprehensive evaluation results
        """
        logger.info(f"Starting evaluation on {num_samples} samples...")

        # Sample from dataset (self.dataset is now a list). Evenly-spaced
        # deterministic indices instead of the first N: the dataset is
        # grouped by question type, so a head-slice would over-represent one
        # category. Identical index set for every run => paired A/B.
        total_samples = len(self.dataset)
        num_samples = min(num_samples, total_samples)
        if num_samples >= total_samples:
            indices = list(range(total_samples))
        else:
            stride = total_samples / num_samples
            indices = sorted({int(i * stride) for i in range(num_samples)})
        samples = [self.dataset[i] for i in indices]

        results = []
        category_scores = {cat: [] for cat in self.categories.keys()}

        # Resume support: reload per-sample results checkpointed by a
        # previous (interrupted) run and skip those samples.
        done_ids = {}
        if self.checkpoint_path and self.checkpoint_path.exists():
            with open(self.checkpoint_path) as ckpt:
                for line in ckpt:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("question_id"):
                        done_ids[row["question_id"]] = row
            if done_ids:
                logger.info(
                    "Resuming: %d samples restored from checkpoint %s",
                    len(done_ids),
                    self.checkpoint_path,
                )

        for i, sample in enumerate(samples):
            cached_row = done_ids.get(sample.get("question_id"))
            if cached_row is not None:
                results.append(cached_row)
                category = cached_row["category"]
                if category in category_scores:
                    category_scores[category].append(cached_row["evaluation"]["score"])
                continue

            logger.info(f"Evaluating sample {i+1}/{len(samples)}")

            result = self.evaluate_sample(sample)
            results.append(result)

            if self.checkpoint_path:
                with open(self.checkpoint_path, "a") as ckpt:
                    ckpt.write(json.dumps(result, default=str) + "\n")

            # Track category performance
            category = result["category"]
            score = result["evaluation"]["score"]

            if category in category_scores:
                category_scores[category].append(score)

        # Calculate aggregate metrics
        overall_scores = [r["evaluation"]["score"] for r in results]
        overall_accuracy = sum(r["evaluation"]["correct"] for r in results) / len(
            results
        )
        overall_score = (
            sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
        )

        category_results = {}
        for category, scores in category_scores.items():
            if scores:
                category_results[category] = {
                    "accuracy": sum(1 for s in scores if s >= 0.5) / len(scores),
                    "average_score": sum(scores) / len(scores),
                    "num_samples": len(scores),
                }
            else:
                category_results[category] = {
                    "accuracy": 0.0,
                    "average_score": 0.0,
                    "num_samples": 0,
                }

        # Aggregate token/cache usage across question turns.
        total_prompt = sum(
            (r.get("usage") or {}).get("prompt_tokens") or 0 for r in results
        )
        total_cached = sum(
            (r.get("usage") or {}).get("cached_tokens") or 0 for r in results
        )

        # Record which memorizz build produced this run (A/B provenance).
        try:
            import memorizz as _memorizz_mod

            memorizz_path = str(getattr(_memorizz_mod, "__file__", "unknown"))
        except Exception:
            memorizz_path = "unknown"

        # Compile final results
        evaluation_results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "dataset_variant": self.dataset_variant,
                "application_mode": self.application_mode,
                "config_label": self.config_label,
                "ingest_mode": self.ingest_mode,
                "context_window_tokens": self.context_window_tokens,
                "disable_auto_summaries": self.disable_auto_summaries,
                "memorizz_path": memorizz_path,
                "num_samples": len(results),
                "total_processing_time": sum(r["processing_time"] for r in results),
                "total_prompt_tokens": total_prompt,
                "total_cached_tokens": total_cached,
                "cache_hit_ratio": (
                    total_cached / total_prompt if total_prompt else 0.0
                ),
            },
            "overall_accuracy": overall_accuracy,
            "overall_score": overall_score,
            "category_results": category_results,
            "detailed_results": results,
        }

        return evaluation_results

    def save_results(
        self, results: Dict[str, Any], filename: Optional[str] = None
    ) -> Path:
        """Save evaluation results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"longmemeval_{self.dataset_variant}_{self.application_mode}_{timestamp}.json"

        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to {filepath}")
        return filepath


def main():
    """Main evaluation function."""
    if "MEMORIZZ_LOG_LEVEL" not in os.environ:
        os.environ["MEMORIZZ_LOG_LEVEL"] = "WARNING"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("longmemeval_evaluation.log"),
            logging.StreamHandler(),
        ],
    )

    parser = argparse.ArgumentParser(
        description="Evaluate Memorizz using LongMemEval benchmark with Oracle AI Database",
        epilog="""
Environment Variables:
  OPENAI_API_KEY     Required: OpenAI API key for LLM and embeddings
  ORACLE_USER        Optional: Oracle database user (default: memorizz_user)
  ORACLE_PASSWORD    Optional: Oracle database password (default: SecurePass123!)
  ORACLE_DSN         Optional: Oracle connection string (default: localhost:1521/FREEPDB1)
        """,
    )

    parser.add_argument(
        "--dataset_variant",
        choices=["oracle", "s", "m"],
        default="oracle",
        help="LongMemEval dataset variant to use",
    )
    parser.add_argument(
        "--num_samples", type=int, default=50, help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--application_mode",
        type=str,
        default=None,
        help="Memorizz application mode to use (defaults to selected agent mode, then assistant)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results", help="Directory to save results"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default=None,
        help="Optional explicit output filename for JSON results.",
    )
    parser.add_argument(
        "--agent_id",
        type=str,
        default="",
        help="Optional existing MemAgent ID to use as the evaluation template.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--ingest_mode",
        choices=["direct", "run"],
        default="direct",
        help=(
            "History ingestion: 'direct' (benchmark-correct; dataset user+"
            "assistant turns stored verbatim with embeddings, no LLM calls) "
            "or 'run' (legacy replay of user turns through agent.run())."
        ),
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="gpt-4o",
        help="OpenAI model used for LLM-as-judge scoring.",
    )
    parser.add_argument(
        "--config_label",
        type=str,
        default="",
        help="Label recorded in results metadata (tag A/B arms).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a per-sample JSONL checkpoint. Each evaluated sample is "
            "appended immediately; an interrupted run restarted with the "
            "same checkpoint resumes where it left off."
        ),
    )
    parser.add_argument(
        "--disable_auto_summaries",
        action="store_true",
        help=(
            "Disable the automatic context-summary trigger in the agent "
            "(recommended with --context_window_tokens on Oracle: the "
            "conversation table has no summary_id column, so summarization "
            "can never mark progress and old builds loop on it)."
        ),
    )
    parser.add_argument(
        "--context_window_tokens",
        type=int,
        default=None,
        help=(
            "Constrain the agent's context window so haystack history "
            "genuinely overflows it — this is what makes the run a "
            "long-horizon MEMORY test (recall via retrieval/summaries) "
            "rather than an in-window reading test."
        ),
    )

    args = parser.parse_args()

    # Check for required environment variables
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is required")
        sys.exit(1)

    try:
        # Initialize evaluator
        evaluator = LongMemEvalEvaluator(
            dataset_variant=args.dataset_variant,
            application_mode=args.application_mode,
            output_dir=args.output_dir,
            verbose=args.verbose,
            ingest_mode=args.ingest_mode,
            judge_model=args.judge_model,
            config_label=args.config_label,
            context_window_tokens=args.context_window_tokens,
            disable_auto_summaries=args.disable_auto_summaries,
            checkpoint_path=args.checkpoint,
        )

        selected_agent_id = (args.agent_id or "").strip()
        if selected_agent_id:
            selected_agent = evaluator.memory_provider.retrieve_memagent(
                selected_agent_id
            )
            if not selected_agent:
                raise ValueError(f"Agent not found: {selected_agent_id}")
            evaluator.agent_template = selected_agent
            if not args.application_mode:
                template_mode = getattr(selected_agent, "application_mode", None)
                if template_mode:
                    evaluator.application_mode = template_mode
            logger.info(
                "Using agent template for evaluation: %s",
                selected_agent_id,
            )

        # Run evaluation
        results = evaluator.evaluate(num_samples=args.num_samples)

        # Save results
        output_file = evaluator.save_results(results, filename=args.output_filename)

        # Print summary
        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY - Memorizz with Oracle AI Database")
        print("=" * 50)
        print(f"Dataset Variant: {args.dataset_variant}")
        print(f"Application Mode: {evaluator.application_mode}")
        print(f"Memory Provider: Oracle AI Database")
        print(f"Config Label: {results['metadata'].get('config_label') or '(none)'}")
        print(f"Ingest Mode: {results['metadata'].get('ingest_mode')}")
        print(f"Memorizz Build: {results['metadata'].get('memorizz_path')}")
        print(f"Samples Evaluated: {results['metadata']['num_samples']}")
        print(f"Overall Accuracy: {results['overall_accuracy']:.3f}")
        print(f"Overall Score: {results['overall_score']:.3f}")
        print(f"Processing Time: {results['metadata']['total_processing_time']:.2f}s")
        print(
            "Question-turn tokens: prompt={p} cached={c} (hit ratio {r:.1%})".format(
                p=results["metadata"].get("total_prompt_tokens", 0),
                c=results["metadata"].get("total_cached_tokens", 0),
                r=results["metadata"].get("cache_hit_ratio", 0.0),
            )
        )
        print("\nCategory Performance:")
        for category, metrics in results["category_results"].items():
            print(
                f"  {category}: {metrics['accuracy']:.3f} ({metrics['num_samples']} samples)"
            )
        print(f"\nDetailed results saved to: {output_file}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

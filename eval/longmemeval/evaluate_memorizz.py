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
    ):
        """
        Initialize the evaluator with Oracle AI Database as the memory provider.

        Args:
            dataset_variant: LongMemEval variant ("oracle", "s", "m")
            application_mode: Memorizz application mode to use
            output_dir: Directory to save results
            verbose: Enable verbose logging
        """
        self.dataset_variant = dataset_variant
        self.application_mode = application_mode
        self.output_dir = Path(output_dir) if output_dir else Path("./results")
        self.verbose = verbose
        self.agent_template = agent_template
        self.dataset_dir = (
            Path(dataset_dir) if dataset_dir else Path(__file__).parent / "data"
        )

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize memory provider
        self.memory_provider = memory_provider or self._init_memory_provider()

        # Initialize evaluation model for scoring
        self.eval_model = evaluation_model or OpenAI(model="gpt-4")

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

        context_window_tokens = self._get_template_value("context_window_tokens")
        if context_window_tokens:
            builder.config.context_window_tokens = context_window_tokens

        agent = builder.build()

        # Save the agent to Oracle
        agent.save()
        return agent

    def _process_conversation_history(
        self, agent, history: List[List[Dict[str, Any]]]
    ) -> None:
        """
        Process conversation history session by session to build up agent memory.

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

            # Parse JSON response
            eval_result = json.loads(eval_response)

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
            # Process conversation history
            self._process_conversation_history(agent, history)

            # Ask the evaluation question
            agent_response = agent.run(question)

            # Evaluate the response
            evaluation = self._evaluate_response(
                question, agent_response, ground_truth, category
            )

            # Calculate processing time
            processing_time = time.time() - start_time

            result = {
                "question": question,
                "category": category,
                "agent_response": agent_response,
                "ground_truth": ground_truth,
                "evaluation": evaluation,
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

        # Sample from dataset (self.dataset is now a list)
        total_samples = len(self.dataset)
        num_samples = min(num_samples, total_samples)
        samples = self.dataset[:num_samples]

        results = []
        category_scores = {cat: [] for cat in self.categories.keys()}

        for i, sample in enumerate(samples):
            logger.info(f"Evaluating sample {i+1}/{len(samples)}")

            result = self.evaluate_sample(sample)
            results.append(result)

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

        # Compile final results
        evaluation_results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "dataset_variant": self.dataset_variant,
                "application_mode": self.application_mode,
                "num_samples": len(results),
                "total_processing_time": sum(r["processing_time"] for r in results),
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
        print(f"Samples Evaluated: {results['metadata']['num_samples']}")
        print(f"Overall Accuracy: {results['overall_accuracy']:.3f}")
        print(f"Overall Score: {results['overall_score']:.3f}")
        print(f"Processing Time: {results['metadata']['total_processing_time']:.2f}s")
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

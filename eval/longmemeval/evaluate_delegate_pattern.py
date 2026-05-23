#!/usr/bin/env python3
"""
LongMemEval Evaluation Script for Memorizz - Delegate Pattern Multi-Agent

This script evaluates Memorizz's long-term memory capabilities using the LongMemEval benchmark
with a delegate pattern multi-agent architecture.
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

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Place in environment variables
os.environ["MEMORIZZ_LOG_LEVEL"] = "WARNING"
os.environ["MONGODB_URI"] = ""
os.environ["OPENAI_API_KEY"] = ""
os.environ["VOYAGE_API_KEY"] = ""

try:
    from src.memorizz import MemAgent, MemoryProvider
    from src.memorizz.llms.openai import OpenAI
    from src.memorizz.long_term.semantic.persona.persona import Persona
    from src.memorizz.long_term.semantic.persona.role_type import RoleType
    from src.memorizz.memory_provider.mongodb.provider import (
        MongoDBConfig,
        MongoDBProvider,
    )
    from src.memorizz.multi_agent_orchestrator import MultiAgentOrchestrator
except ImportError as e:
    print(f"Error importing Memorizz: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("longmemeval_delegate_evaluation.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class LongMemEvalDelegateEvaluator:
    """Evaluator for Memorizz using LongMemEval benchmark with delegate pattern multi-agent."""

    def __init__(
        self,
        dataset_variant: str = "oracle",
        application_mode: str = "assistant",
        output_dir: str = "./results",
        verbose: bool = False,
    ):
        self.dataset_variant = dataset_variant
        self.application_mode = application_mode
        self.output_dir = Path(output_dir)
        self.verbose = verbose

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize memory provider
        self.memory_provider = self._init_memory_provider()

        # Initialize evaluation model for scoring
        self.eval_model = OpenAI(model="gpt-4")

        # Load dataset
        self.dataset = self._load_dataset()

        # Category mapping
        self.categories = {
            "single-session-user": "SSU",
            "single-session-assistant": "SSA",
            "single-session-preference": "SSP",
            "multi-session": "MS",
            "temporal-reasoning": "TR",
            "knowledge-update": "KU",
        }

        logger.info(
            f"Initialized LongMemEval delegate pattern evaluator with variant: {dataset_variant}"
        )

    def _init_memory_provider(self) -> MemoryProvider:
        """Initialize memory provider."""
        mongodb_uri = os.environ.get("MONGODB_URI")
        if not mongodb_uri:
            logger.warning("MONGODB_URI not found, using default memory provider")
            return MemoryProvider()

        try:
            config = MongoDBConfig(uri=mongodb_uri)
            return MongoDBProvider(config)
        except Exception as e:
            logger.warning(f"Failed to initialize MongoDB provider: {e}, using default")
            return MemoryProvider()

    def _load_dataset(self):
        """Load LongMemEval dataset from local files."""
        try:
            data_dir = Path(__file__).parent / "data"

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

                logger.info(
                    f"Loaded LongMemEval-{self.dataset_variant.upper()} dataset with {len(data)} samples"
                )
                return data
            else:
                logger.error(f"Dataset file not found: {filepath}")
                logger.error("Please download the LongMemEval dataset by running:")
                logger.error("python download_dataset.py")
                raise FileNotFoundError(f"Dataset file not found: {filepath}")

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def _create_delegate_agents(self) -> List[MemAgent]:
        """Create specialized delegate agents for different memory tasks."""
        delegates = []

        # Memory Specialist Agent
        memory_specialist = MemAgent(
            memory_provider=self.memory_provider,
            application_mode=self.application_mode,
            instruction="You are a memory specialist. Focus on retrieving, organizing, and recalling information from conversations. Remember user preferences, past interactions, and important contextual details.",
            persona=Persona(
                name="Memory Specialist",
                role=RoleType.TECHNICAL_EXPERT,
                goals="Specialize in memory retrieval, information organization, and context analysis with precise and detailed responses.",
                background="A technical expert focused on memory systems, information retrieval, and contextual analysis of conversational data.",
            ),
        )
        delegates.append(memory_specialist)

        # Temporal Reasoning Agent
        temporal_agent = MemAgent(
            memory_provider=self.memory_provider,
            application_mode=self.application_mode,
            instruction="You are a temporal reasoning specialist. Handle time-based queries, understand sequences of events, and reason about when things happened in conversations.",
            persona=Persona(
                name="Temporal Specialist",
                role=RoleType.RESEARCHER,
                goals="Focus on temporal reasoning, sequence analysis, and timeline construction with logical and chronological thinking.",
                background="A researcher specializing in time-based analysis, chronological sequencing, and temporal relationships in data.",
            ),
        )
        delegates.append(temporal_agent)

        # Context Integration Agent
        context_agent = MemAgent(
            memory_provider=self.memory_provider,
            application_mode=self.application_mode,
            instruction="You are a context integration specialist. Connect information across different conversation sessions and identify patterns in user behavior and preferences.",
            persona=Persona(
                name="Context Integrator",
                role=RoleType.RESEARCHER,
                goals="Perform cross-session analysis, pattern recognition, and information synthesis with comprehensive and analytical approaches.",
                background="A research specialist in cross-session data analysis, behavioral pattern identification, and comprehensive information integration.",
            ),
        )
        delegates.append(context_agent)

        # Save all delegate agents
        for agent in delegates:
            agent.save()

        return delegates

    def _create_fresh_orchestrator(self) -> MultiAgentOrchestrator:
        """Create a fresh multi-agent orchestrator for evaluation."""
        print(
            "Creating fresh delegate pattern orchestrator with root agent and specialized delegates"
        )

        # Create root agent
        root_agent = MemAgent(
            memory_provider=self.memory_provider,
            application_mode=self.application_mode,
            instruction="You are a coordinating agent that manages a team of memory specialists. Analyze complex memory queries and delegate tasks to appropriate specialists for optimal results.",
        )
        root_agent.save()

        # Create delegate agents
        delegates = self._create_delegate_agents()

        # Create orchestrator
        orchestrator = MultiAgentOrchestrator(
            root_agent=root_agent, delegates=delegates
        )

        return orchestrator

    def _process_conversation_history(
        self, orchestrator: MultiAgentOrchestrator, history: List[List[Dict[str, Any]]]
    ) -> None:
        """Process conversation history session by session to build up agent memory."""
        for session_idx, session in enumerate(history):
            if self.verbose:
                logger.info(f"Processing session {session_idx + 1}/{len(history)}")

            for message in session:
                role = message.get("role", "")
                content = message.get("content", "")

                if role == "user":
                    try:
                        response = orchestrator.execute_multi_agent_workflow(content)
                        if self.verbose:
                            logger.debug(f"User: {content[:100]}...")
                            logger.debug(f"Multi-Agent: {response[:100]}...")
                    except Exception as e:
                        logger.warning(f"Error processing message: {e}")

    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single sample from the dataset."""
        try:
            # Create fresh orchestrator for this sample
            orchestrator = self._create_fresh_orchestrator()

            # Extract conversation history and question
            history = sample.get("conversation", [])
            question = sample.get("question", "")
            ground_truth = sample.get("answer", "")
            category = sample.get("category", "unknown")

            if self.verbose:
                logger.info(f"Evaluating sample - Category: {category}")
                logger.info(f"Question: {question}")
                logger.info(f"History sessions: {len(history)}")

            # Process conversation history to build memory
            start_time = time.time()
            self._process_conversation_history(orchestrator, history)

            # Ask the test question
            agent_response = orchestrator.execute_multi_agent_workflow(question)
            end_time = time.time()

            if self.verbose:
                logger.info(f"Agent response: {agent_response}")
                logger.info(f"Ground truth: {ground_truth}")

            return {
                "question": question,
                "agent_response": agent_response,
                "ground_truth": ground_truth,
                "category": category,
                "response_time": end_time - start_time,
                "architecture": "delegate_pattern",
            }

        except Exception as e:
            logger.error(f"Error evaluating sample: {e}")
            return {
                "question": sample.get("question", ""),
                "agent_response": f"Error: {str(e)}",
                "ground_truth": sample.get("answer", ""),
                "category": sample.get("category", "unknown"),
                "response_time": 0,
                "architecture": "delegate_pattern",
                "error": str(e),
            }

    def evaluate(self, num_samples: int = 50) -> Dict[str, Any]:
        """Run evaluation on the dataset."""
        logger.info(
            f"Starting evaluation with delegate pattern on {num_samples} samples"
        )

        # Sample the dataset
        if num_samples < len(self.dataset):
            import random

            samples = random.sample(self.dataset, num_samples)
        else:
            samples = self.dataset[:num_samples]

        results = []

        for i, sample in enumerate(samples):
            logger.info(f"Evaluating sample {i+1}/{len(samples)}")

            result = self.evaluate_sample(sample)
            results.append(result)

        # Calculate aggregate metrics
        aggregate_results = {
            "architecture": "delegate_pattern",
            "dataset_variant": self.dataset_variant,
            "total_samples": len(results),
            "timestamp": datetime.now().isoformat(),
            "detailed_results": results,
        }

        logger.info("Evaluation completed")
        return aggregate_results

    def save_results(
        self, results: Dict[str, Any], filename: Optional[str] = None
    ) -> Path:
        """Save evaluation results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = (
                f"longmemeval_delegate_results_{self.dataset_variant}_{timestamp}.json"
            )

        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {filepath}")
        return filepath


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate Memorizz with delegate pattern on LongMemEval"
    )
    parser.add_argument(
        "--variant",
        choices=["oracle", "s", "m"],
        default="oracle",
        help="Dataset variant to use",
    )
    parser.add_argument(
        "--samples", type=int, default=50, help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--application-mode", default="assistant", help="Application mode to use"
    )
    parser.add_argument(
        "--output-dir", default="./results", help="Output directory for results"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = LongMemEvalDelegateEvaluator(
        dataset_variant=args.variant,
        application_mode=args.application_mode,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )

    # Run evaluation
    results = evaluator.evaluate(num_samples=args.samples)

    # Save results
    evaluator.save_results(results)

    # Print summary
    print(f"\n=== Delegate Pattern Evaluation Results ===")
    print(f"Architecture: {results['architecture']}")
    print(f"Dataset: LongMemEval-{args.variant.upper()}")
    print(f"Samples: {results['total_samples']}")


if __name__ == "__main__":
    main()

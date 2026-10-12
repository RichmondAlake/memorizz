"""Fine-tune decision models, embedding models and rerankers on MemoRizz data.

Datasets live in the connected memory provider; synthetic examples come
from a teacher LLM and are checked blind; training runs locally (Apple GPU,
CUDA or CPU via sentence-transformers and PEFT) or with Unsloth on an NVIDIA
GPU, and every run is scored before and after on held-out examples.
"""

from .inference import DecisionModel
from .models import Dataset, Example, ExampleStatus, ModelKind, RunStatus, TrainingRun
from .service import FineTuning, FineTuningError, environment
from .tasks import MEMORY_ROUTING_OPTIONS, TASKS

__all__ = [
    "Dataset",
    "DecisionModel",
    "Example",
    "ExampleStatus",
    "FineTuning",
    "FineTuningError",
    "MEMORY_ROUTING_OPTIONS",
    "ModelKind",
    "RunStatus",
    "TASKS",
    "TrainingRun",
    "environment",
]

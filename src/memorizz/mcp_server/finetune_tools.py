"""Fine-tuning through MCP: datasets in the memory provider, synthetic
data, training runs and their models.

Reads need the read scope; creating and reviewing data needs writes;
writing synthetic data calls a teacher LLM, so it also needs execution;
starting a training run needs ``allow_training`` (on by default only for a
local stdio server). Over HTTP each principal sees only its own datasets.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from .auth import RequestIdentity
from .config import EXECUTE_SCOPE, READ_SCOPE, WRITE_SCOPE

MAX_GENERATE = 200
MAX_ADD = 500


class FineTuneTools:
    def __init__(self, runtime: Any):
        self.runtime = runtime
        self._finetuning = None
        self._lock = threading.Lock()

    # -- helpers -------------------------------------------------------------

    def _error(self, code: str, message: str):
        from .runtime import MemorizzServerError

        return MemorizzServerError(code, message)

    def _ft(self):
        with self._lock:
            if self._finetuning is None:
                from ..finetune import FineTuning

                self._finetuning = FineTuning(self.runtime.provider)
            return self._finetuning

    def _call(self, function, *args, **kwargs):
        from ..finetune import FineTuningError

        try:
            return function(*args, **kwargs)
        except FineTuningError as exc:
            raise self._error("finetune_error", str(exc)) from exc
        except ImportError as exc:
            raise self._error(
                "finetune_unavailable",
                f'Fine-tuning needs its packages: pip install "memorizz[finetune]" ({exc.name})',
            ) from exc

    def _owner(self, identity: RequestIdentity) -> str:
        if self.runtime.config.transport == "stdio":
            return ""
        return str(identity.principal or "anonymous")

    def _dataset(self, dataset_id: str, identity: RequestIdentity) -> None:
        owner = self._owner(identity)
        try:
            found = self._ft().owner_of(dataset_id)
        except Exception:
            found = None
        if found is None or (owner and found != owner):
            raise self._error("not_found", f"No fine-tuning dataset {dataset_id}")

    def _run(self, run_id: str, identity: RequestIdentity) -> Dict[str, Any]:
        run = self._call(self._ft().run, run_id)
        self._dataset(run["dataset_id"], identity)
        return run

    # -- reads ---------------------------------------------------------------

    def list_datasets(
        self, identity: RequestIdentity, *, include_catalog: bool = False
    ) -> Dict[str, Any]:
        self.runtime._require_scope(identity, READ_SCOPE)
        from ..finetune import FineTuning

        ft = self._ft()
        out: Dict[str, Any] = {
            "ok": True,
            "datasets": self._call(ft.datasets, owner=self._owner(identity) or None),
            "active_models": self._call(ft.active),
        }
        if include_catalog:
            out["tasks"] = FineTuning.tasks()
            out["base_models"] = FineTuning.base_models()
            out["environment"] = FineTuning.environment()
            out["training_allowed"] = bool(self.runtime.config.allow_training)
        return out

    def get_dataset(
        self,
        dataset_id: str,
        identity: RequestIdentity,
        *,
        status: Optional[str] = None,
        split: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        self.runtime._require_scope(identity, READ_SCOPE)
        self._dataset(dataset_id, identity)
        ft = self._ft()
        return {
            "ok": True,
            "dataset": self._call(ft.dataset, dataset_id),
            "examples": self._call(
                ft.examples,
                dataset_id,
                status=status,
                split=split,
                limit=max(1, min(int(limit), 500)),
                offset=max(0, int(offset)),
            ),
            "runs": self._call(ft.runs, dataset_id),
            "evaluations": self._call(ft.evaluations, dataset_id),
        }

    def get_run(
        self, run_id: str, identity: RequestIdentity, *, after: int = 0
    ) -> Dict[str, Any]:
        self.runtime._require_scope(identity, READ_SCOPE)
        run = self._run(run_id, identity)
        events = self._call(
            self._ft().events, run_id, after=max(0, int(after)), limit=200
        )
        return {
            "ok": True,
            "run": run,
            "events": events,
            "next_after": events[-1]["sequence"] if events else after,
        }

    def decide(
        self,
        identity: RequestIdentity,
        state: Any,
        *,
        run_id: Optional[str] = None,
        use: str = "memory_routing",
    ) -> Dict[str, Any]:
        self.runtime._require_scope(identity, READ_SCOPE)
        ft = self._ft()
        if not run_id:
            record = self._call(ft.active).get(use)
            if not record:
                raise self._error(
                    "not_found", f"No active {use} model; train one and activate it"
                )
            run_id = record["run_id"]
        run = self._run(run_id, identity)
        if run["kind"] != "decision":
            raise self._error(
                "invalid_request", "Only decision models answer decisions"
            )
        return {
            "ok": True,
            "run_id": run_id,
            "answers": self._call(ft.decide, run_id, state),
        }

    # -- writes --------------------------------------------------------------

    def create_dataset(
        self,
        identity: RequestIdentity,
        *,
        kind: str,
        task: str,
        name: str = "",
        spec: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.runtime._require_scope(identity, WRITE_SCOPE)
        dataset = self._call(
            self._ft().create_dataset,
            kind,
            task,
            name=name,
            spec=spec,
            owner=self._owner(identity),
        )
        return {"ok": True, "dataset": dataset.to_dict()}

    def add_examples(
        self, dataset_id: str, examples: List[Dict[str, Any]], identity: RequestIdentity
    ) -> Dict[str, Any]:
        self.runtime._require_scope(identity, WRITE_SCOPE)
        self._dataset(dataset_id, identity)
        if len(examples) > MAX_ADD:
            raise self._error(
                "invalid_request", f"Add at most {MAX_ADD} examples per call"
            )
        ft, added = self._ft(), []
        for item in examples:
            if not isinstance(item, dict) or not isinstance(item.get("input"), dict):
                raise self._error(
                    "invalid_request", "Each example needs an input object"
                )
            example = self._call(
                ft.add_example,
                dataset_id,
                input=item["input"],
                label=item.get("label") or {},
                split=item.get("split"),
                status=item.get("status") or "accepted",
                source="mcp",
            )
            added.append(example.example_id)
        return {"ok": True, "added": len(added), "example_ids": added}

    def review_examples(
        self,
        dataset_id: str,
        identity: RequestIdentity,
        *,
        accept: Optional[List[str]] = None,
        reject: Optional[List[str]] = None,
        accept_verified: bool = False,
    ) -> Dict[str, Any]:
        self.runtime._require_scope(identity, WRITE_SCOPE)
        self._dataset(dataset_id, identity)
        ft = self._ft()
        mine = {e["example_id"] for e in self._call(ft.examples, dataset_id)}
        stray = [i for i in [*(accept or []), *(reject or [])] if i not in mine]
        if stray:
            raise self._error(
                "invalid_request",
                f"Examples not in this dataset: {', '.join(stray[:5])}",
            )
        out = {
            "accepted": self._call(ft.review, accept or [], "accepted"),
            "rejected": self._call(ft.review, reject or [], "rejected"),
        }
        if accept_verified:
            out["accepted_verified"] = self._call(ft.accept_verified, dataset_id)
        return {
            "ok": True,
            **out,
            "counts": self._call(ft.dataset, dataset_id)["counts"],
        }

    # -- execution -----------------------------------------------------------

    def generate(
        self,
        dataset_id: str,
        identity: RequestIdentity,
        *,
        count: int = 40,
        teacher: Optional[str] = None,
        verifier: Optional[str] = None,
        context: str = "",
        topic: str = "",
        from_memory: bool = False,
    ) -> Dict[str, Any]:
        self.runtime._require_scope(identity, EXECUTE_SCOPE)
        self.runtime._require_scope(identity, WRITE_SCOPE)
        self._dataset(dataset_id, identity)
        count = int(count)
        if not 1 <= count <= MAX_GENERATE:
            raise self._error("invalid_request", f"count must be 1-{MAX_GENERATE}")

        def spec(value: Optional[str]) -> Optional[Dict[str, str]]:
            if not value:
                return None
            if "/" not in value:
                raise self._error(
                    "invalid_request", "teacher and verifier are provider/model"
                )
            provider, model = value.split("/", 1)
            return {"provider": provider, "model": model}

        report = self._call(
            self._ft().generate,
            dataset_id,
            count=count,
            teacher=spec(teacher),
            verifier=spec(verifier),
            context=context,
            topic=topic,
            from_memory=from_memory,
        )
        return {"ok": True, "report": report}

    def start_training(
        self,
        dataset_id: str,
        identity: RequestIdentity,
        *,
        base_model: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        from_run: Optional[str] = None,
        replay: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        self.runtime._require_scope(identity, EXECUTE_SCOPE)
        if not self.runtime.config.allow_training:
            raise self._error(
                "training_disabled",
                "Fine-tuning runs are disabled by the server operator",
            )
        self._dataset(dataset_id, identity)
        if from_run:
            self._run(from_run, identity)
        for item in replay or []:
            self._dataset(str(item.get("dataset_id")), identity)
        run = self._call(
            self._ft().start_run,
            dataset_id,
            base_model=base_model,
            params=params or {},
            from_run=from_run,
            replay=replay,
        )
        return {"ok": True, "run": run.to_dict()}


__all__ = ["FineTuneTools"]

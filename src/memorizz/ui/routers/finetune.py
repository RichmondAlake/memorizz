"""Fine-tuning: datasets in the connected memory provider, synthetic data,
training runs with live metrics, and trained models to try, compare and
swap in."""

from __future__ import annotations

import io
import logging
import os
import threading
import uuid
import zipfile
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ..security import ui_read_only
from ..state import _state, templates
from ..trace_access import current_principal

logger = logging.getLogger(__name__)
router = APIRouter(tags=["fine-tuning"])

_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_services: Dict[int, Any] = {}


def _provider(*, write: bool = False):
    if current_principal.get().restricted:
        raise HTTPException(
            403, "Fine-tuning requires unrestricted administrator access"
        )
    provider = _state.get("provider")
    if not provider:
        raise HTTPException(400, "Connect a memory provider first")
    if write and (ui_read_only() or _state.get("read_only")):
        raise HTTPException(403, "Fine-tuning changes are disabled in read-only mode")
    return provider


def _ft(*, write: bool = False):
    from ...finetune import FineTuning

    provider = _provider(write=write)
    service = _services.get(id(provider))
    if service is None:
        _services.clear()
        service = _services[id(provider)] = FineTuning(provider)
    return service


async def _run(function, *args, **kwargs):
    from ...finetune import FineTuningError

    try:
        return await run_in_threadpool(function, *args, **kwargs)
    except FineTuningError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(
            503,
            f'Fine-tuning needs its packages: pip install "memorizz[finetune]" ({exc.name})',
        ) from exc


@router.get("/fine-tuning", response_class=HTMLResponse)
async def finetune_page(
    request: Request, dataset: str = "", tab: str = "", run: str = ""
):
    if not _state.get("provider"):
        return RedirectResponse("/connect", status_code=302)
    _provider()
    return templates.TemplateResponse(
        "finetune.html",
        {
            "request": request,
            "active_page": "fine-tuning",
            "dataset_id": dataset,
            # Deep links (the model registry): open a dataset's step and run.
            "tab": tab
            if tab in {"data", "train", "results", "scores", "task", "format"}
            else "",
            "run_id": run,
            "provider_type": _state.get("provider_type") or "",
            "read_only": ui_read_only() or _state.get("read_only", False),
        },
        headers={"Cache-Control": "no-store"},
    )


@router.get("/fine-tuning/models", response_class=HTMLResponse)
async def model_registry_page(request: Request):
    if not _state.get("provider"):
        return RedirectResponse("/connect", status_code=302)
    _provider()
    return templates.TemplateResponse(
        "finetune_models.html",
        {
            "request": request,
            "active_page": "model-registry",
            "provider_type": _state.get("provider_type") or "",
            "read_only": ui_read_only() or _state.get("read_only", False),
        },
        headers={"Cache-Control": "no-store"},
    )


@router.get("/api/finetune/registry")
async def model_registry():
    """Every model per job: in use, fine-tuned versions, base models."""
    return await _run(_ft().registry)


@router.get("/api/finetune/overview")
async def overview():
    from ...finetune import FineTuning
    from ...finetune.synthetic import default_teacher

    ft = _ft()
    datasets = await _run(ft.datasets)
    active = await _run(ft.active)
    return {
        "environment": await run_in_threadpool(FineTuning.environment),
        "tasks": FineTuning.tasks(),
        "base_models": FineTuning.base_models(),
        "datasets": datasets,
        "active": active,
        "teacher": default_teacher(),
        "jev_available": bool(os.environ.get("TYPESAFE_API_KEY")),
        "provider_type": _state.get("provider_type") or "",
        "read_only": ui_read_only() or _state.get("read_only", False),
    }


class DatasetIn(BaseModel):
    kind: str
    task: str
    name: str = ""
    spec: Optional[Dict[str, Any]] = None


@router.post("/api/finetune/datasets")
async def create_dataset(body: DatasetIn):
    ft = _ft(write=True)
    dataset = await _run(
        ft.create_dataset, body.kind, body.task, name=body.name, spec=body.spec
    )
    return dataset.to_dict()


@router.get("/api/finetune/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    return await _run(_ft().dataset, dataset_id)


class DatasetPatch(BaseModel):
    name: Optional[str] = None
    spec: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    test_set: Optional[bool] = None


@router.patch("/api/finetune/datasets/{dataset_id}")
async def update_dataset(dataset_id: str, body: DatasetPatch):
    ft = _ft(write=True)
    dataset = await _run(
        ft.update_dataset,
        dataset_id,
        name=body.name,
        spec=body.spec,
        description=body.description,
        test_set=body.test_set,
    )
    return dataset.to_dict()


@router.delete("/api/finetune/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    return {"deleted": await _run(_ft(write=True).delete_dataset, dataset_id)}


@router.get("/api/finetune/datasets/{dataset_id}/examples")
async def list_examples(
    dataset_id: str,
    status: str = "",
    split: str = "",
    limit: int = 100,
    offset: int = 0,
):
    ft = _ft()
    rows = await _run(
        ft.examples, dataset_id, status=status or None, split=split or None
    )
    return {
        "total": len(rows),
        "examples": rows[offset : offset + max(1, min(limit, 500))],
    }


class ExampleIn(BaseModel):
    input: Dict[str, Any]
    label: Dict[str, Any] = {}
    split: Optional[str] = None


@router.post("/api/finetune/datasets/{dataset_id}/examples")
async def add_example(dataset_id: str, body: ExampleIn):
    ft = _ft(write=True)
    example = await _run(
        ft.add_example, dataset_id, input=body.input, label=body.label, split=body.split
    )
    return example.to_dict()


class ReviewIn(BaseModel):
    ids: List[str]
    status: str


@router.post("/api/finetune/examples/review")
async def review(body: ReviewIn):
    return {"changed": await _run(_ft(write=True).review, body.ids, body.status)}


@router.post("/api/finetune/datasets/{dataset_id}/accept-verified")
async def accept_verified(dataset_id: str):
    return {"accepted": await _run(_ft(write=True).accept_verified, dataset_id)}


class ExamplePatch(BaseModel):
    label_key: Optional[str] = None
    split: Optional[str] = None
    status: Optional[str] = None


@router.patch("/api/finetune/examples/{example_id}")
async def update_example(example_id: str, body: ExamplePatch):
    ft = _ft(write=True)
    if body.label_key is not None:
        await _run(ft.relabel, example_id, body.label_key)
    if body.split is not None or body.status is not None:
        await _run(ft.update_example, example_id, split=body.split, status=body.status)
    example = await run_in_threadpool(ft.store.get_example, example_id)
    return example.to_dict() if example else {}


class DeleteIn(BaseModel):
    ids: List[str]


@router.post("/api/finetune/examples/delete")
async def delete_examples(body: DeleteIn):
    return {"deleted": await _run(_ft(write=True).delete_examples, body.ids)}


class GenerateIn(BaseModel):
    count: int = 40
    teacher: Optional[str] = None
    verifier: Optional[str] = None
    context: str = ""
    topic: str = ""
    from_memory: bool = False
    labels: Optional[List[str]] = None


def _spec(value: Optional[str]) -> Optional[Dict[str, str]]:
    if not value:
        return None
    if "/" not in value:
        raise HTTPException(400, "Use provider/model, for example ollama/qwen2.5:7b")
    provider, model = value.split("/", 1)
    return {"provider": provider, "model": model}


@router.post("/api/finetune/datasets/{dataset_id}/generate")
async def generate(dataset_id: str, body: GenerateIn):
    ft = _ft(write=True)
    if not 1 <= body.count <= 2000:
        raise HTTPException(400, "count must be 1-2000")
    teacher, verifier = _spec(body.teacher), _spec(body.verifier)
    return _start_job(
        dataset_id,
        lambda progress: ft.generate(
            dataset_id,
            count=body.count,
            teacher=teacher,
            verifier=verifier,
            context=body.context,
            topic=body.topic,
            from_memory=body.from_memory,
            labels=body.labels or None,
            progress=progress,
        ),
    )


def _start_job(dataset_id: str, work) -> Dict[str, str]:
    """Run ``work(progress)`` on a thread; the page polls /api/finetune/jobs."""
    job_id = uuid.uuid4().hex[:10]
    job = {
        "job_id": job_id,
        "dataset_id": dataset_id,
        "status": "running",
        "done": 0,
        "total": 1,
        "message": "Starting",
        "report": None,
        "error": "",
    }
    with _jobs_lock:
        _jobs[job_id] = job

    def progress(done: int, total: int, message: str) -> None:
        job.update(done=done, total=max(1, total), message=message)

    def run() -> None:
        from ...finetune import FineTuningError

        try:
            job["report"] = work(progress)
            job["status"] = "succeeded"
        except (FineTuningError, ValueError, ImportError) as exc:
            logger.warning("Fine-tuning job %s failed: %s", job_id, exc)
            job.update(status="failed", error=str(exc))
        except Exception as exc:  # shown in the page, not raised
            logger.exception("Fine-tuning job %s failed", job_id)
            job.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:500])

    threading.Thread(target=run, name=f"finetune-job-{job_id}", daemon=True).start()
    return {"job_id": job_id}


class EvaluateIn(BaseModel):
    dataset_id: str
    run_id: Optional[str] = None
    base_model: Optional[str] = None
    jev_model: Optional[str] = None
    in_use: bool = False
    split: str = "test"
    max_length: Optional[int] = None  # tokens an untrained base model reads


@router.post("/api/finetune/evaluate")
async def evaluate(body: EvaluateIn):
    ft = _ft(write=True)
    return _start_job(
        body.dataset_id,
        lambda progress: ft.evaluate(
            body.dataset_id,
            run_id=body.run_id,
            base_model=body.base_model,
            jev_model=body.jev_model,
            in_use=body.in_use,
            split=body.split,
            max_length=body.max_length,
        ),
    )


@router.get("/api/finetune/evaluations")
async def evaluations(dataset_id: str = ""):
    return {"evaluations": await _run(_ft().evaluations, dataset_id or None)}


@router.get("/api/finetune/jobs/{job_id}")
async def get_job(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job (the server may have restarted)")
    return job


class ImportIn(BaseModel):
    stores: Optional[List[str]] = None
    limit: int = 500


@router.post("/api/finetune/datasets/{dataset_id}/import-memories")
async def import_memories(dataset_id: str, body: ImportIn):
    ft = _ft(write=True)
    return await _run(
        ft.import_memories, dataset_id, memory_types=body.stores, limit=body.limit
    )


@router.get("/api/finetune/datasets/{dataset_id}/preview")
async def preview(
    dataset_id: str, base_model: str = "", format: str = "", limit: int = 3
):
    return await _run(
        _ft().preview,
        dataset_id,
        base_model=base_model or None,
        fmt=format or None,
        limit=limit,
    )


@router.get("/api/finetune/datasets/{dataset_id}/export")
async def export(
    dataset_id: str,
    format: str = "memorizz",
    base_model: str = "",
    status: str = "accepted",
):
    text = await _run(
        _ft().export,
        dataset_id,
        fmt=format,
        base_model=base_model or None,
        status=status or None,
    )
    return Response(
        text,
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": f'attachment; filename="finetune-{dataset_id}-{format}.jsonl"'
        },
    )


@router.get("/api/finetune/datasets/{dataset_id}/unsloth-bundle")
async def unsloth_bundle(dataset_id: str, base_model: str = "unsloth/Qwen3.5-4B"):
    files = await _run(_ft().unsloth_bundle, dataset_id, base_model=base_model)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, text in files.items():
            archive.writestr(f"memorizz-unsloth-{dataset_id}/{name}", text)
    return Response(
        buffer.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="memorizz-unsloth-{dataset_id}.zip"'
        },
    )


class RunIn(BaseModel):
    base_model: Optional[str] = None
    params: Dict[str, Any] = {}
    backend: str = "auto"
    name: str = ""
    label: str = ""
    from_run: Optional[str] = None
    replay: Optional[List[Dict[str, Any]]] = None


@router.post("/api/finetune/datasets/{dataset_id}/runs")
async def start_run(dataset_id: str, body: RunIn):
    ft = _ft(write=True)
    run = await _run(
        ft.start_run,
        dataset_id,
        base_model=body.base_model,
        params=body.params,
        backend=body.backend,
        name=body.name,
        from_run=body.from_run or None,
        replay=body.replay or None,
        label=body.label,
    )
    return await _run(ft.run, run.run_id)


@router.get("/api/finetune/runs")
async def list_runs(dataset_id: str = ""):
    return {"runs": await _run(_ft().runs, dataset_id or None)}


@router.get("/api/finetune/runs/{run_id}")
async def get_run(run_id: str):
    return await _run(_ft().run, run_id)


class RunPatch(BaseModel):
    label: str = ""


@router.patch("/api/finetune/runs/{run_id}")
async def rename_run(run_id: str, body: RunPatch):
    return await _run(_ft(write=True).rename_run, run_id, body.label)


@router.get("/api/finetune/runs/{run_id}/curves")
async def run_curves(run_id: str):
    """Loss by step, validation per epoch, before/after and convergence."""
    return await _run(_ft().curves, run_id)


@router.get("/api/finetune/runs/{run_id}/data")
async def run_data(
    run_id: str, split: str = "", source: str = "", offset: int = 0, limit: int = 25
):
    """The examples the run trained and was scored on (its own snapshot)."""
    return await _run(
        _ft().run_data,
        run_id,
        split=split or None,
        source=source or None,
        offset=offset,
        limit=limit,
    )


@router.post("/api/finetune/runs/{run_id}/speed")
async def run_speed(run_id: str):
    """Time the version, what is in use now and its base (a job)."""
    ft = _ft(write=True)
    return _start_job("", lambda progress: ft.speed(run_id, progress=progress))


@router.get("/api/finetune/runs/{run_id}/verdict")
async def run_verdict(run_id: str):
    """The comparison and verdict the run's Models-in-use card would show."""
    return await _run(_ft().verdict, run_id)


@router.get("/api/finetune/runs/{run_id}/events")
async def run_events(run_id: str, after: int = 0):
    events = await _run(_ft().events, run_id, after=after)
    return {"events": events, "next": events[-1]["sequence"] if events else after}


@router.post("/api/finetune/runs/{run_id}/cancel")
async def cancel_run(run_id: str):
    return await _run(_ft(write=True).cancel_run, run_id)


@router.delete("/api/finetune/runs/{run_id}")
async def delete_run(run_id: str):
    return {"deleted": await _run(_ft(write=True).delete_run, run_id)}


class DecideIn(BaseModel):
    state: Any


@router.post("/api/finetune/runs/{run_id}/decide")
async def decide(run_id: str, body: DecideIn):
    return {"answers": await _run(_ft().decide, run_id, body.state)}


class RankIn(BaseModel):
    query: str
    passages: List[str]


@router.post("/api/finetune/runs/{run_id}/rank")
async def rank(run_id: str, body: RankIn):
    return {"ranking": await _run(_ft().rank, run_id, body.query, body.passages)}


# -- models in use: compare and swap -------------------------------------------


@router.get("/api/finetune/models")
async def models():
    return {"models": await _run(_ft().overview)}


class SwapIn(BaseModel):
    run_id: str
    # Embeddings: the vector size to store (from the run's embedding plan).
    dimensions: Optional[int] = None


def _job_progress(progress):
    return lambda message: progress(0, 1, message)


@router.post("/api/finetune/models/{card}/swap")
async def swap(card: str, body: SwapIn):
    ft = _ft(write=True)
    return _start_job(
        "",
        lambda progress: ft.swap(
            card,
            body.run_id,
            progress=_job_progress(progress),
            dimensions=body.dimensions,
        ),
    )


@router.post("/api/finetune/models/{card}/swap-back")
async def swap_back(card: str):
    ft = _ft(write=True)
    return _start_job(
        "", lambda progress: ft.swap_back(card, progress=_job_progress(progress))
    )


@router.get("/api/finetune/runs/{run_id}/embedding-plan")
async def embedding_plan(run_id: str):
    """The vector sizes an embedding version can be swapped in at."""
    return await _run(_ft().embedding_plan, run_id)


@router.get("/api/finetune/models/{card}/impact")
async def swap_impact(card: str):
    """What a swap for ``card`` touches: store, agents, other processes."""
    return await _run(_ft().swap_impact, card)


@router.get("/api/finetune/agents")
async def agent_models():
    """Each agent's own memory router and reranker, if it has one."""
    return {"agents": await _run(_ft().agent_models)}


class AgentModelIn(BaseModel):
    choice: Optional[str] = None  # a run id, "off", or "default"


@router.put("/api/finetune/agents/{agent_id}/models/{job}")
async def set_agent_model(agent_id: str, job: str, body: AgentModelIn):
    return await _run(_ft(write=True).set_agent_model, agent_id, job, body.choice)


@router.post("/api/finetune/models/{card}/compare")
async def compare(card: str, body: SwapIn):
    """Score the fine-tuned model, its untrained base and (for embeddings and
    reranking) the model in use now on the run's test examples."""
    ft = _ft(write=True)
    return _start_job("", lambda progress: ft.compare(body.run_id, progress=progress))


__all__ = ["router"]

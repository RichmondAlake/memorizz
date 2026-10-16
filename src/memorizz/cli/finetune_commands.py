"""``memorizz finetune``: data, training and use of decision models,
embedding models and rerankers. Datasets live in the memory provider."""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, List, Optional

import typer

from .._env_io import env_text as _env
from . import config as cfg

finetune_app = typer.Typer(
    help="Fine-tune decision models, embedding models and rerankers on your memory.",
    no_args_is_help=True,
)

JSON_OUT = typer.Option(False, "--json", help="Emit compact JSON.")
BACKEND = typer.Option(
    None, "--backend", help="filesystem, mongodb or oracle (default: MEMORIZZ_BACKEND)."
)


@contextmanager
def _finetuning(backend: Optional[str] = None) -> Iterator[Any]:
    cfg.load_layered_env()
    resolved = str(backend or _env("MEMORIZZ_BACKEND") or "filesystem").lower()
    if resolved == "oracle":
        from ..memory_provider.oracle import OracleProvider

        provider = OracleProvider.from_env(index_policy="lazy")
    elif resolved == "mongodb":
        from ..memory_provider.mongodb import MongoDBConfig, MongoDBProvider

        uri = _env("MONGODB_URI")
        if not uri:
            raise typer.BadParameter("MONGODB_URI is required for --backend mongodb")
        provider = MongoDBProvider(
            MongoDBConfig(uri=uri, db_name=_env("MONGODB_DB_NAME") or "memorizz")
        )
    elif resolved == "filesystem":
        from ..memory_provider import FileSystemConfig, FileSystemProvider

        provider = FileSystemProvider(
            FileSystemConfig(root_path=cfg.memory_root(), lazy_vector_indexes=True)
        )
    else:
        raise typer.BadParameter("backend must be filesystem, mongodb, or oracle")
    from ..finetune import FineTuning, FineTuningError

    try:
        yield FineTuning(provider)
    except FineTuningError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise typer.Exit(1)
    finally:
        closer = getattr(provider, "close", None)
        if callable(closer):
            closer()


def _print(value: Any, raw_json: bool) -> None:
    print(
        json.dumps(value, indent=None if raw_json else 2, sort_keys=True, default=str)
    )


def _progress(done: int, total: int, message: str) -> None:
    print(f"[{done}/{total}] {message}", file=sys.stderr, flush=True)


def _teacher(value: Optional[str]) -> Optional[dict]:
    if not value:
        return None
    if "/" not in value:
        raise typer.BadParameter("--teacher is provider/model, e.g. ollama/qwen2.5:7b")
    provider, model = value.split("/", 1)
    return {"provider": provider, "model": model}


@finetune_app.command("doctor")
def doctor(raw_json: bool = JSON_OUT):
    """What this machine can train with (device, packages, Unsloth)."""
    from ..finetune import environment

    _print(environment(), raw_json)


@finetune_app.command("tasks")
def tasks(
    kind: Optional[str] = typer.Option(None, "--kind"), raw_json: bool = JSON_OUT
):
    """Tasks and base models for each kind of model."""
    from ..finetune import FineTuning

    rows = [t for t in FineTuning.tasks() if not kind or t["kind"] == kind]
    if raw_json:
        _print({"tasks": rows, "base_models": FineTuning.base_models(kind)}, True)
        return
    for t in rows:
        print(f"{t['kind']}/{t['task']:<16} {t['title']} — {t['description']}")
    print()
    for m in FineTuning.base_models(kind):
        print(
            f"{m['model_id']:<42} {'/'.join(m['kinds']):<20} {m['params']:<16} {m['notes']}"
        )


@finetune_app.command("datasets")
def datasets(backend: Optional[str] = BACKEND, raw_json: bool = JSON_OUT):
    """Datasets in the memory provider."""
    with _finetuning(backend) as ft:
        rows = ft.datasets()
    if raw_json:
        _print(rows, True)
        return
    for d in rows:
        c = d["counts"]
        print(
            f"{d['dataset_id']}  {d['kind']}/{d['task']:<15} {d['name']:<32} {c['total']} examples ({c['status'].get('accepted', 0)} accepted)"
        )


@finetune_app.command("create")
def create(
    kind: str = typer.Argument(..., help="decision, embedding or reranker"),
    task: str = typer.Argument(
        ..., help="e.g. memory_routing, relevance, custom, retrieval"
    ),
    name: str = typer.Option("", "--name"),
    spec_file: Optional[Path] = typer.Option(
        None, "--spec", help="JSON file with the task definition (questions/options)."
    ),
    backend: Optional[str] = BACKEND,
    raw_json: bool = JSON_OUT,
):
    """Create a dataset for a task."""
    spec = json.loads(spec_file.read_text()) if spec_file else None
    with _finetuning(backend) as ft:
        dataset = ft.create_dataset(kind, task, name=name, spec=spec)
    _print(dataset.to_dict(), raw_json)


@finetune_app.command("show")
def show(dataset_id: str, backend: Optional[str] = BACKEND, raw_json: bool = JSON_OUT):
    """A dataset's definition, counts and label balance."""
    with _finetuning(backend) as ft:
        _print(ft.dataset(dataset_id), raw_json)


@finetune_app.command("generate")
def generate(
    dataset_id: str,
    count: int = typer.Option(40, "--count", help="About how many examples to write."),
    teacher: Optional[str] = typer.Option(
        None,
        "--teacher",
        help="provider/model (default: MEMORIZZ_FINETUNE_TEACHER or your default LLM).",
    ),
    context: str = typer.Option(
        "", "--context", help="The setting the examples come from."
    ),
    topic: str = typer.Option(
        "", "--topic", help="Retrieval tasks: what the generated memories are about."
    ),
    from_memory: bool = typer.Option(
        False,
        "--from-memory",
        help="Retrieval tasks: write questions for your stored memories.",
    ),
    label: Optional[List[str]] = typer.Option(
        None,
        "--label",
        help="Decision tasks: only these options (repeatable), e.g. the ones a model gets wrong.",
    ),
    verify: bool = typer.Option(
        True,
        "--verify/--no-verify",
        help="Check every example with a blind second call.",
    ),
    verifier: Optional[str] = typer.Option(
        None,
        "--verifier",
        help="provider/model for the blind check, e.g. typesafe/jev-latest or openai/gpt-6.1-mini (default: the teacher).",
    ),
    backend: Optional[str] = BACKEND,
    raw_json: bool = JSON_OUT,
):
    """Write synthetic examples with a teacher LLM (left pending for review)."""
    with _finetuning(backend) as ft:
        report = ft.generate(
            dataset_id,
            count=count,
            teacher=_teacher(teacher),
            context=context,
            topic=topic,
            from_memory=from_memory,
            labels=label or None,
            verify=verify,
            verifier=_teacher(verifier),
            progress=_progress,
        )
    _print(report, raw_json)


@finetune_app.command("import-memories")
def import_memories(
    dataset_id: str,
    store: Optional[List[str]] = typer.Option(
        None, "--store", help="Memory stores to import (repeatable)."
    ),
    limit: int = typer.Option(500, "--limit"),
    backend: Optional[str] = BACKEND,
    raw_json: bool = JSON_OUT,
):
    """Stored memories as routing examples, labelled by the store they live in."""
    with _finetuning(backend) as ft:
        _print(
            ft.import_memories(dataset_id, memory_types=store, limit=limit), raw_json
        )


@finetune_app.command("import")
def import_file(
    dataset_id: str,
    path: Path = typer.Argument(..., help="JSONL exported with --format memorizz."),
    status: Optional[str] = typer.Option(
        None, "--status", help="Override every example's status (e.g. pending)."
    ),
    backend: Optional[str] = BACKEND,
    raw_json: bool = JSON_OUT,
):
    """Add exported examples to a dataset (e.g. to move one between providers)."""
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with _finetuning(backend) as ft:
        _print(ft.import_examples(dataset_id, rows, status=status), raw_json)


@finetune_app.command("examples")
def examples(
    dataset_id: str,
    status: Optional[str] = typer.Option(None, "--status"),
    split: Optional[str] = typer.Option(None, "--split"),
    limit: int = typer.Option(20, "--limit"),
    backend: Optional[str] = BACKEND,
    raw_json: bool = JSON_OUT,
):
    """List a dataset's examples."""
    with _finetuning(backend) as ft:
        rows = ft.examples(dataset_id, status=status, split=split, limit=limit)
    if raw_json:
        _print(rows, True)
        return
    for e in rows:
        label = json.dumps(e["label"]) if e["label"] else ""
        text = e["input"].get("state") or e["input"].get("query")
        text = text if isinstance(text, str) else json.dumps(text)
        mark = "✓" if e["verified"] else " "
        print(
            f"{e['example_id']} {e['status']:<8} {e['split']:<10} {mark} {label[:60]:<60} {text[:80]}"
        )


@finetune_app.command("review")
def review(
    dataset_id: str,
    accept: Optional[List[str]] = typer.Option(
        None, "--accept", help="Example ids to accept."
    ),
    reject: Optional[List[str]] = typer.Option(
        None, "--reject", help="Example ids to reject."
    ),
    accept_verified: bool = typer.Option(
        False,
        "--accept-verified",
        help="Accept every pending example a blind check agreed with.",
    ),
    backend: Optional[str] = BACKEND,
    raw_json: bool = JSON_OUT,
):
    """Mark examples for training (accepted) or leave them out (rejected)."""
    with _finetuning(backend) as ft:
        out = {
            "accepted": ft.review(accept or [], "accepted"),
            "rejected": ft.review(reject or [], "rejected"),
        }
        if accept_verified:
            out["accepted_verified"] = ft.accept_verified(dataset_id)
        out["counts"] = ft.dataset(dataset_id)["counts"]
    _print(out, raw_json)


@finetune_app.command("add")
def add(
    dataset_id: str,
    state: Optional[str] = typer.Option(
        None, "--state", help="Decision: the state text."
    ),
    label: Optional[str] = typer.Option(
        None, "--label", help="Decision: the correct option key."
    ),
    query: Optional[str] = typer.Option(None, "--query"),
    positive: Optional[str] = typer.Option(None, "--positive"),
    negative: Optional[List[str]] = typer.Option(None, "--negative"),
    split: Optional[str] = typer.Option(None, "--split"),
    backend: Optional[str] = BACKEND,
    raw_json: bool = JSON_OUT,
):
    """Add one example by hand (accepted)."""
    from ..finetune.tasks import answer_for, only_question

    with _finetuning(backend) as ft:
        dataset = ft.dataset(dataset_id)
        if dataset["kind"] == "decision":
            if not state or not label:
                raise typer.BadParameter("A decision example needs --state and --label")
            qid, question = only_question(dataset["spec"])
            example = ft.add_example(
                dataset_id,
                input={"state": state},
                label={qid: answer_for(question, label)},
                split=split,
            )
        else:
            if not query or not positive:
                raise typer.BadParameter(
                    "A retrieval example needs --query and --positive"
                )
            example = ft.add_example(
                dataset_id,
                input={
                    "query": query,
                    "positive": positive,
                    "negatives": list(negative or []),
                },
                split=split,
            )
    _print(example.to_dict(), raw_json)


@finetune_app.command("preview")
def preview(
    dataset_id: str,
    base_model: Optional[str] = typer.Option(None, "--base"),
    fmt: Optional[str] = typer.Option(None, "--format"),
    limit: int = typer.Option(2, "--limit"),
    backend: Optional[str] = BACKEND,
):
    """Show examples written in the input the chosen base model expects."""
    with _finetuning(backend) as ft:
        _print(
            ft.preview(dataset_id, base_model=base_model, fmt=fmt, limit=limit), False
        )


@finetune_app.command("export")
def export(
    dataset_id: str,
    fmt: str = typer.Option(
        "memorizz",
        "--format",
        help="systemone, jevembed, chat, sentence-transformers, cross-encoder or memorizz",
    ),
    base_model: Optional[str] = typer.Option(None, "--base"),
    out: Optional[Path] = typer.Option(
        None, "--out", help="Write JSONL here instead of stdout."
    ),
    unsloth_bundle: Optional[Path] = typer.Option(
        None,
        "--unsloth-bundle",
        help="Write data + a training script for Unsloth on a GPU into this folder.",
    ),
    backend: Optional[str] = BACKEND,
):
    """Export accepted examples, or an Unsloth bundle to train on a GPU."""
    with _finetuning(backend) as ft:
        if unsloth_bundle:
            files = ft.unsloth_bundle(
                dataset_id, base_model=base_model or "unsloth/Qwen3.5-4B"
            )
            unsloth_bundle.mkdir(parents=True, exist_ok=True)
            for name, text in files.items():
                (unsloth_bundle / name).write_text(text, encoding="utf-8")
            print(
                f"Wrote {', '.join(sorted(files))} to {unsloth_bundle}", file=sys.stderr
            )
            return
        text = ft.export(dataset_id, fmt=fmt, base_model=base_model)
    if out:
        out.write_text(text, encoding="utf-8")
        print(f"Wrote {text.count(chr(10))} rows to {out}", file=sys.stderr)
    else:
        sys.stdout.write(text)


def _params(values: Optional[List[str]]) -> dict:
    params = {}
    for item in values or []:
        key, _, raw = item.partition("=")
        try:
            params[key.strip()] = json.loads(raw)
        except ValueError:
            params[key.strip()] = raw
    return params


def _describe_event(event: dict) -> Optional[str]:
    data, kind = event["data"], event["type"]
    if kind == "status":
        return data.get("message")
    if kind == "step":
        return f"step {data['step']}/{data['total']}  loss {data['loss']:.4f}"
    if kind == "eval":
        m = data.get("metrics") or {}
        keys = [
            k
            for k in ("accuracy", "macro_f1", "ece", "mrr@10", "ndcg@10", "recall@1")
            if k in m
        ]
        return f"{data.get('phase')}: " + ", ".join(f"{k} {m[k]}" for k in keys)
    if kind == "finished":
        return f"finished: {data.get('status')}" + (
            f" ({data['error']})" if data.get("error") else ""
        )
    return None


def _follow(ft: Any, run_id: str, every: int = 10) -> dict:
    after, shown_step = 0, 0
    while True:
        for event in ft.events(run_id, after=after):
            after = event["sequence"]
            if event["type"] == "step":
                step = event["data"]["step"]
                if (
                    step - shown_step < every
                    and step != event["data"]["total"]
                    and step != 1
                ):
                    continue
                shown_step = step
            line = _describe_event(event)
            if line:
                print(line, file=sys.stderr, flush=True)
        run = ft.run(run_id)
        if run["status"] not in ("running", "queued"):
            return run
        time.sleep(2)


@finetune_app.command("train")
def train(
    dataset_id: str,
    base_model: Optional[str] = typer.Option(
        None,
        "--base",
        help="Base model id or folder (default: the kind's recommended one).",
    ),
    epochs: Optional[float] = typer.Option(None, "--epochs"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size"),
    learning_rate: Optional[float] = typer.Option(None, "--lr"),
    param: Optional[List[str]] = typer.Option(
        None, "--param", help="Other settings as key=value (repeatable)."
    ),
    backend_name: str = typer.Option(
        "auto", "--trainer", help="auto, local or unsloth."
    ),
    from_run: Optional[str] = typer.Option(
        None, "--from-run", help="Continue from this run's model (continual training)."
    ),
    replay: Optional[List[str]] = typer.Option(
        None,
        "--replay",
        help="DATASET[:share] whose training examples are mixed in, e.g. abc123:0.5 (repeatable).",
    ),
    follow: bool = typer.Option(
        True, "--follow/--detach", help="Show progress until the run ends."
    ),
    backend: Optional[str] = BACKEND,
    raw_json: bool = JSON_OUT,
):
    """Train on the accepted examples; scored before and after on held-out ones."""
    params = _params(param)
    for key, value in (
        ("epochs", epochs),
        ("batch_size", batch_size),
        ("learning_rate", learning_rate),
    ):
        if value is not None:
            params[key] = value
    with _finetuning(backend) as ft:
        replays = []
        for item in replay or []:
            dataset, _, share = item.partition(":")
            replays.append({"dataset_id": dataset, "fraction": float(share or 0.5)})
        run = ft.start_run(
            dataset_id,
            base_model=base_model,
            params=params,
            backend=backend_name,
            from_run=from_run,
            replay=replays or None,
        )
        print(
            f"Run {run.run_id} started ({run.base_model}, {run.backend}); log: {run.output_dir}/worker.log",
            file=sys.stderr,
        )
        result = _follow(ft, run.run_id) if follow else run.to_dict()
    _print(result, raw_json)


@finetune_app.command("evaluate")
def evaluate(
    dataset_id: str,
    run_id: Optional[str] = typer.Option(None, "--run", help="A trained run."),
    base_model: Optional[str] = typer.Option(
        None, "--base", help="An untrained base model."
    ),
    jev: Optional[str] = typer.Option(
        None, "--jev", help="Jev model, e.g. jev-latest (needs TYPESAFE_API_KEY)."
    ),
    in_use: bool = typer.Option(
        False,
        "--in-use",
        help="The embedding model in use now (retrieval and reranking datasets).",
    ),
    split: str = typer.Option(
        "test", "--split", help="test, validation, held_out or all."
    ),
    backend: Optional[str] = BACKEND,
    raw_json: bool = JSON_OUT,
):
    """Score a model on a dataset's frozen examples; kept for the history."""
    if sum(bool(x) for x in (run_id, base_model, jev, in_use)) != 1:
        raise typer.BadParameter("Give exactly one of --run, --base, --jev or --in-use")
    with _finetuning(backend) as ft:
        result = ft.evaluate(
            dataset_id,
            run_id=run_id,
            base_model=base_model,
            jev_model=jev,
            in_use=in_use,
            split=split,
        )
    if raw_json:
        _print(result, True)
        return
    print(
        f"{result['model']['label']} on {result['dataset_name']} ({split}, n={result['count']}): {_metrics_line(result['metrics'])}"
    )


def _metrics_line(m: dict) -> str:
    keys = ("accuracy", "macro_f1", "ece", "recall@1", "mrr@10", "ndcg@10")
    return ", ".join(f"{k} {m[k]}" for k in keys if k in m)


@finetune_app.command("history")
def history(
    dataset_id: Optional[str] = typer.Option(None, "--dataset"),
    backend: Optional[str] = BACKEND,
    raw_json: bool = JSON_OUT,
):
    """Every evaluation kept: each model version on each frozen set."""
    with _finetuning(backend) as ft:
        rows = ft.evaluations(dataset_id)
    if raw_json:
        _print(rows, True)
        return
    for e in rows:
        m = e["metrics"]
        print(
            f"{e['created_at'][:16]}  {e['model']['label']:<44} {e['dataset_name'][:30]:<30} {e['split']:<9} n={e['count']:<4} acc {m.get('accuracy')}  F1 {m.get('macro_f1')}"
        )


@finetune_app.command("runs")
def runs(
    dataset_id: Optional[str] = typer.Option(None, "--dataset"),
    backend: Optional[str] = BACKEND,
    raw_json: bool = JSON_OUT,
):
    """Training runs and their before/after scores."""
    with _finetuning(backend) as ft:
        rows = ft.runs(dataset_id)
    if raw_json:
        _print(rows, True)
        return
    for r in rows:
        main = "accuracy" if r["kind"] == "decision" else "ndcg@10"
        before, after = r["metrics_before"].get(main), r["metrics_after"].get(main)
        print(
            f"{r['run_id']}  {r['status']:<11} {r['kind']:<9} {r['base_model']:<40} {main} {before} → {after}"
        )


@finetune_app.command("status")
def status(
    run_id: str,
    follow: bool = typer.Option(False, "--follow"),
    backend: Optional[str] = BACKEND,
    raw_json: bool = JSON_OUT,
):
    """One run's progress and metrics."""
    with _finetuning(backend) as ft:
        _print(_follow(ft, run_id) if follow else ft.run(run_id), raw_json)


@finetune_app.command("cancel")
def cancel(run_id: str, backend: Optional[str] = BACKEND, raw_json: bool = JSON_OUT):
    """Ask a running training job to stop."""
    with _finetuning(backend) as ft:
        _print(ft.cancel_run(run_id), raw_json)


@finetune_app.command("decide")
def decide(
    run_id: str, state: str, backend: Optional[str] = BACKEND, raw_json: bool = JSON_OUT
):
    """Ask a trained decision model about a state."""
    with _finetuning(backend) as ft:
        _print(ft.decide(run_id, state), raw_json)


@finetune_app.command("rank")
def rank(
    run_id: str,
    query: str = typer.Option(..., "--query"),
    passage: List[str] = typer.Option(
        ..., "--passage", help="Repeat for each candidate."
    ),
    backend: Optional[str] = BACKEND,
    raw_json: bool = JSON_OUT,
):
    """Rank passages for a query with a trained embedding model, reranker or relevance decision model."""
    with _finetuning(backend) as ft:
        _print(ft.rank(run_id, query, list(passage)), raw_json)


@finetune_app.command("activate")
def activate(
    run_id: str,
    use: str = typer.Argument(
        ..., help="memory_routing or relevance (use swap for embedding and reranking)"
    ),
    backend: Optional[str] = BACKEND,
    raw_json: bool = JSON_OUT,
):
    """Record a decision model for memory_routing or relevance.

    Embedding models and rerankers are put to work with ``swap``."""
    with _finetuning(backend) as ft:
        _print(ft.activate(run_id, use), raw_json)


@finetune_app.command("models")
def models(backend: Optional[str] = BACKEND, raw_json: bool = JSON_OUT):
    """What each job uses now, the best fine-tuned model for it, and whether
    fine-tuning helped (on the same held-out examples)."""
    with _finetuning(backend) as ft:
        cards = ft.overview()
    if raw_json:
        _print(cards, True)
        return
    for card in cards:
        print(f"{card['title']}\n  in use: {card['in_use']['label']}")
        c = card["candidate"]
        if not c:
            print("  no fine-tuned model yet\n")
            continue
        e = c["evidence"]

        def pct(v):
            return "-" if v is None else f"{v * 100:.1f}%"

        line = f"  {c['label']}: {e['metric']} {pct(e['fine_tuned'])} vs base {pct(e['base'])}"
        if e.get("in_use"):
            line += f" vs in use {pct(e['in_use']['value'])}"
        print(
            line
            + f"  ({e['count']} {'frozen' if e['frozen'] else 'held-out'} examples)"
        )
        test = e.get("significance")
        chance = (
            f" (p = {test['p']}, {test['differ']} of {test['n']} differ)"
            if test
            else ""
        )
        verdict = {
            "better": f"helped: {c['gain_points']:+} points over {c['compared_with']}, unlikely to be chance{chance}",
            "worse": f"worse than {c['compared_with']}: {c['gain_points']:+} points{chance}",
            "unclear": f"{c['gain_points']:+} points vs {c['compared_with']}, could be chance{chance}",
            "untested": f"{c['gain_points']:+} points vs {c['compared_with']}, not tested for chance (memorizz finetune compare {c['run_id']})",
        }.get(c.get("verdict"), "nothing to compare with yet")
        state = (
            "in use"
            if c["in_use"]
            else f"memorizz finetune swap {card['use']} {c['run_id']}"
        )
        print(f"  {verdict} · {state}\n")


@finetune_app.command("compare")
def compare(run_id: str, backend: Optional[str] = BACKEND, raw_json: bool = JSON_OUT):
    """Score a fine-tuned model, its untrained base and (embeddings and
    rerankers) the embedding model in use now on the run's test examples."""
    with _finetuning(backend) as ft:
        results = ft.compare(run_id)
    if raw_json:
        _print(results, True)
        return
    for r in results:
        print(f"{r['model']['label']} (n={r['count']}): {_metrics_line(r['metrics'])}")


@finetune_app.command("swap")
def swap(
    card: str = typer.Argument(..., help="memory_routing, embedding or reranking"),
    run_id: str = typer.Argument(...),
    dimensions: Optional[int] = typer.Option(
        None,
        "--dimensions",
        help="Embeddings: the vector size to store (see `memorizz finetune plan`); "
        "default keeps the store's size when the model can be cut to it",
    ),
    backend: Optional[str] = BACKEND,
    raw_json: bool = JSON_OUT,
):
    """Put a fine-tuned model to work. Embedding swaps re-embed the stored
    memories beside the live ones and switch at once; ``swap-back`` undoes
    it, reusing the vectors it kept."""
    with _finetuning(backend) as ft:
        result = ft.swap(
            card,
            run_id,
            progress=lambda m: print(m, file=sys.stderr, flush=True),
            dimensions=dimensions,
        )
    _print(result, raw_json)


@finetune_app.command("plan")
def plan(
    run_id: str = typer.Argument(..., help="An embedding run"),
    backend: Optional[str] = BACKEND,
    raw_json: bool = JSON_OUT,
):
    """The vector sizes an embedding version can be swapped in at."""
    with _finetuning(backend) as ft:
        _print(ft.embedding_plan(run_id), raw_json)


@finetune_app.command("swap-back")
def swap_back(
    card: str = typer.Argument(..., help="memory_routing, embedding or reranking"),
    backend: Optional[str] = BACKEND,
    raw_json: bool = JSON_OUT,
):
    """Return a job to the model it used before the first swap."""
    with _finetuning(backend) as ft:
        result = ft.swap_back(
            card, progress=lambda m: print(m, file=sys.stderr, flush=True)
        )
    _print(result, raw_json)


@finetune_app.command("route")
def route(text: str, backend: Optional[str] = BACKEND, raw_json: bool = JSON_OUT):
    """Which memory store the active routing model picks for ``text``."""
    with _finetuning(backend) as ft:
        _print(ft.route(text), raw_json)


@finetune_app.command("delete")
def delete(
    dataset_id: Optional[str] = typer.Option(None, "--dataset"),
    run_id: Optional[str] = typer.Option(None, "--run"),
    yes: bool = typer.Option(False, "--yes", help="Do not ask."),
    backend: Optional[str] = BACKEND,
):
    """Delete a dataset (with its examples and run records) or one run."""
    if not dataset_id and not run_id:
        raise typer.BadParameter("Give --dataset or --run")
    target = f"dataset {dataset_id}" if dataset_id else f"run {run_id}"
    if not yes and not typer.confirm(f"Delete {target}?"):
        raise typer.Exit(1)
    with _finetuning(backend) as ft:
        ok = ft.delete_dataset(dataset_id) if dataset_id else ft.delete_run(run_id)
    print("deleted" if ok else "not found", file=sys.stderr)

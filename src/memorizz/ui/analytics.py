"""Small server-rendered chart view models; no browser chart dependency."""

import math


def chart(rows, field, *, limit=12, rank=False):
    measured = []
    for row in rows:
        try:
            number = float(row.get(field))
        except (ValueError, TypeError):
            continue
        if math.isfinite(number) and number >= 0:
            measured.append({"label": str(row["label"]), "value": number})
    if rank:
        measured.sort(key=lambda row: row["value"], reverse=True)
    selected = measured[:limit]
    maximum = max((row["value"] for row in selected), default=0) or 1
    total = sum(row["value"] for row in measured)
    offset = 0
    for index, row in enumerate(selected):
        row.update(
            width=round(row["value"] / maximum * 100, 4),
            share=row["value"] / total * 100 if total else 0,
            offset=offset,
            x=20 + index * 560 / max(1, len(selected) - 1),
            y=180 - row["value"] / maximum * 150,
        )
        offset += row["share"]
    return {
        "rows": selected,
        "total": total,
        "omitted": len(measured) - len(selected),
        "points": " ".join(f"{row['x']:.2f},{row['y']:.2f}" for row in selected),
    }


def usage_charts(usage):
    return {
        "daily_tokens": chart(usage["daily"], "total_tokens", limit=90),
        "daily_cost": chart(usage["daily"], "cost_usd", limit=90),
        "agent_cost": chart(usage["agents"], "cost_usd", rank=True),
        "memory_tokens": chart(usage["memory"], "memory_tokens_estimate", rank=True),
        "memory_latency": chart(usage["memory"], "retrieval_mean_ms", rank=True),
    }


def eval_charts(results, history):
    results = results or {}
    categories = [
        {"label": key, "accuracy": value.get("accuracy")}
        for key, value in (results.get("category_results") or {}).items()
        if isinstance(value, dict)
    ]
    efficiency = results.get("efficiency") or {}
    latency = [
        {"label": label, "seconds": efficiency.get(field)}
        for label, field in (
            ("Retrieval", "average_retrieval_seconds"),
            ("Generation", "average_generation_seconds"),
            ("Semantic search", "average_semantic_search_seconds"),
            ("Lexical search", "average_lexical_search_seconds"),
            ("Reranking", "average_reranking_seconds"),
        )
    ]
    return {
        "categories": chart(categories, "accuracy"),
        "latency": chart(latency, "seconds"),
        "cost": chart(
            [{**row, "label": row.get("run_id") or "Unknown run"} for row in history],
            "cost_usd",
        ),
    }

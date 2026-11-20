from __future__ import annotations

from typing import Mapping, Sequence


def reciprocal_rank_at_k(
    query: Mapping[str, object], hits: Sequence[Mapping[str, object]], k: int = 5
) -> float:
    gold = {int(value) for value in query["gold_target_indices"]}  # type: ignore[union-attr]
    for rank, hit in enumerate(hits[:k], start=1):
        if int(hit["target_idx"]) in gold:
            return 1.0 / rank
    return 0.0


def _align_inputs(
    query_rows: Sequence[Mapping[str, object]],
    bm25_rows: Sequence[Mapping[str, object]],
    qwen_rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[list[dict[str, object]]], list[dict[str, object]]]:
    query_by_id = {str(row["query_id"]): dict(row) for row in query_rows}
    bm25_by_id = {str(row["query_id"]): list(row["hits"]) for row in bm25_rows}
    qwen_by_id = {str(row["query_id"]): dict(row) for row in qwen_rows}
    expected = set(query_by_id)
    if set(bm25_by_id) != expected or set(qwen_by_id) != expected:
        raise ValueError("Gate input query IDs differ")
    ordered_ids = sorted(expected)
    return (
        [query_by_id[query_id] for query_id in ordered_ids],
        [bm25_by_id[query_id] for query_id in ordered_ids],
        [qwen_by_id[query_id] for query_id in ordered_ids],
    )

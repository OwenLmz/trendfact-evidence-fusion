from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from retrieval.evidence import file_sha256  # noqa: E402
from retrieval.run_trendfact_bm25_pilot import retrieval_metrics  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/data_trendfact.yaml"
DEFAULT_RERANK_CONFIG = ROOT / "configs/llm_evidence_pointwise_trendfact_qwen25_7b.yaml"
FUSION_ALPHA_GRID = tuple(round(value * 0.05, 2) for value in range(21))
STANCE_PRIORITY = {"support": 0, "refute": 0, "context": 1, "unknown": 2}


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _hit_at_k(
    query: Mapping[str, object], hits: Sequence[Mapping[str, object]], k: int
) -> bool:
    gold = {int(value) for value in query["gold_target_indices"]}  # type: ignore[union-attr]
    retrieved = {int(hit["target_idx"]) for hit in hits[:k]}
    return bool(gold & retrieved)


def paired_bootstrap_difference(
    baseline: Sequence[float | bool],
    method: Sequence[float | bool],
    *,
    seed: int = 20260730,
    samples: int = 10000,
) -> dict[str, float]:
    if len(baseline) != len(method) or not baseline:
        raise ValueError("Paired outcomes must be non-empty and have equal length")
    differences = np.asarray(method, dtype=np.float32) - np.asarray(
        baseline, dtype=np.float32
    )
    generator = np.random.default_rng(seed)
    bootstrap = np.empty(samples, dtype=np.float32)
    for index in range(samples):
        positions = generator.integers(0, len(differences), size=len(differences))
        bootstrap[index] = float(differences[positions].mean())
    return {
        "difference": round(float(differences.mean()), 6),
        "ci95_low": round(float(np.quantile(bootstrap, 0.025)), 6),
        "ci95_high": round(float(np.quantile(bootstrap, 0.975)), 6),
        "bootstrap_samples": samples,
    }


def fuse_candidate_rankings(
    bm25_hits: Sequence[Mapping[str, object]],
    decisions: Sequence[Mapping[str, object]],
    *,
    alpha: float,
    top_k: int = 5,
) -> list[dict[str, object]]:
    """Fuse BM25 rank and constrained-LLM relevance without using gold labels."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    decision_by_rank = {
        int(decision["candidate_rank"]): decision for decision in decisions
    }
    fused: list[dict[str, object]] = []
    for hit in bm25_hits:
        bm25_rank = int(hit["rank"])
        decision = decision_by_rank.get(bm25_rank, {})
        relevance = max(0, int(decision.get("relevance", -1)))
        stance_priority = STANCE_PRIORITY.get(str(decision.get("stance", "unknown")), 3)
        directness = max(0.0, (2.0 - stance_priority) / 2.0)
        llm_score = relevance / 3.0 + 0.01 * directness
        bm25_rank_score = 1.0 / bm25_rank
        fusion_score = alpha * bm25_rank_score + (1.0 - alpha) * llm_score
        fused.append(
            {
                **dict(hit),
                "llm_relevance": relevance,
                "llm_stance": str(decision.get("stance", "unknown")),
                "fusion_alpha": alpha,
                "fusion_score": fusion_score,
            }
        )
    ordered = sorted(
        fused,
        key=lambda hit: (-float(hit["fusion_score"]), int(hit["rank"])),
    )[:top_k]
    for fused_rank, hit in enumerate(ordered, start=1):
        hit["fusion_rank"] = fused_rank
    return ordered


def _selection_key(metrics: Mapping[str, object], alpha: float) -> tuple[float, ...]:
    recall = metrics["any_gold_recall_at_k"]
    assert isinstance(recall, Mapping)
    return (
        float(recall["5"]),
        float(metrics["mrr_at_max_k"]),
        float(recall["3"]),
        alpha,
    )


def _stratified_folds(
    queries: Sequence[Mapping[str, object]], *, fold_count: int
) -> list[list[int]]:
    folds: list[list[int]] = [[] for _ in range(fold_count)]
    labels = sorted({str(query["label_name"]) for query in queries})
    for label in labels:
        indices = [
            index
            for index, query in enumerate(queries)
            if str(query["label_name"]) == label
        ]
        indices.sort(key=lambda index: str(queries[index]["query_id"]))
        for offset, index in enumerate(indices):
            folds[offset % fold_count].append(index)
    return folds


def cross_validated_static_fusion(
    queries: Sequence[Mapping[str, object]],
    bm25_hits: Sequence[Sequence[Mapping[str, object]]],
    decisions: Sequence[Sequence[Mapping[str, object]]],
    *,
    alphas: Sequence[float] = FUSION_ALPHA_GRID,
    fold_count: int = 5,
) -> dict[str, object]:
    if not (len(queries) == len(bm25_hits) == len(decisions)):
        raise ValueError("Fusion inputs must have equal query counts")
    folds = _stratified_folds(queries, fold_count=fold_count)
    held_out_hits: list[list[dict[str, object]] | None] = [None] * len(queries)
    fold_reports: list[dict[str, object]] = []
    report_at_k = [1, 3, 5]
    for fold_index, validation_indices in enumerate(folds):
        validation_set = set(validation_indices)
        training_indices = [
            index for index in range(len(queries)) if index not in validation_set
        ]
        candidates: list[tuple[float, dict[str, object]]] = []
        for alpha in alphas:
            training_hits = [
                fuse_candidate_rankings(bm25_hits[index], decisions[index], alpha=alpha)
                for index in training_indices
            ]
            metrics = retrieval_metrics(
                [queries[index] for index in training_indices],
                training_hits,
                report_at_k,
            )
            candidates.append((alpha, metrics))
        selected_alpha, training_metrics = max(
            candidates, key=lambda item: _selection_key(item[1], item[0])
        )
        validation_hits = [
            fuse_candidate_rankings(
                bm25_hits[index], decisions[index], alpha=selected_alpha
            )
            for index in validation_indices
        ]
        for index, hits in zip(validation_indices, validation_hits):
            held_out_hits[index] = hits
        validation_metrics = retrieval_metrics(
            [queries[index] for index in validation_indices],
            validation_hits,
            report_at_k,
        )
        fold_reports.append(
            {
                "fold": fold_index,
                "training_queries": len(training_indices),
                "validation_queries": len(validation_indices),
                "selected_alpha": selected_alpha,
                "training_metrics": training_metrics,
                "validation_metrics": validation_metrics,
            }
        )
    if any(hits is None for hits in held_out_hits):
        raise RuntimeError("Cross-validation did not produce every held-out ranking")
    combined_hits = [hits for hits in held_out_hits if hits is not None]
    return {
        "selection_objective": "lexicographic_recall_at_5_then_mrr_at_5_then_recall_at_3",
        "fold_count": fold_count,
        "alpha_grid": list(alphas),
        "folds": fold_reports,
        "metrics": retrieval_metrics(queries, combined_hits, report_at_k),
        "hits": combined_hits,
    }


def evaluate(
    data_config_path: Path = DEFAULT_CONFIG,
    rerank_config_path: Path = DEFAULT_RERANK_CONFIG,
) -> dict[str, object]:
    data_config_path = data_config_path.resolve()
    rerank_config_path = rerank_config_path.resolve()
    data_config = yaml.safe_load(data_config_path.read_text(encoding="utf-8"))
    rerank_config = yaml.safe_load(rerank_config_path.read_text(encoding="utf-8"))
    queries_path = _resolve(data_config["output"]["data_dir"]) / "pilot_dev_queries.jsonl"
    bm25_path = _resolve(rerank_config["input"]["retrieval_path"])
    rerank_path = _resolve(rerank_config["output"]["path"])
    queries = _read_jsonl(queries_path)
    bm25_rows = _read_jsonl(bm25_path)
    rerank_rows = _read_jsonl(rerank_path)
    query_by_id = {str(row["query_id"]): row for row in queries}
    bm25_by_id = {str(row["query_id"]): list(row["hits"]) for row in bm25_rows}
    rerank_by_id = {str(row["query_id"]): list(row["selected"]) for row in rerank_rows}
    decisions_by_id = {
        str(row["query_id"]): list(row["candidate_decisions"]) for row in rerank_rows
    }
    expected = set(query_by_id)
    if set(bm25_by_id) != expected or set(rerank_by_id) != expected:
        raise ValueError("Query IDs differ across gold, BM25 and reranking files")

    ordered_ids = sorted(expected)
    ordered_queries = [query_by_id[query_id] for query_id in ordered_ids]
    bm25_hits = [bm25_by_id[query_id] for query_id in ordered_ids]
    rerank_hits = [rerank_by_id[query_id] for query_id in ordered_ids]
    candidate_decisions = [decisions_by_id[query_id] for query_id in ordered_ids]
    report_at_k = [1, 3, 5]
    baseline_metrics = retrieval_metrics(ordered_queries, bm25_hits, report_at_k)
    rerank_metrics = retrieval_metrics(ordered_queries, rerank_hits, report_at_k)
    fusion_grid: dict[str, object] = {}
    for alpha in FUSION_ALPHA_GRID:
        fused_hits = [
            fuse_candidate_rankings(hits, decisions, alpha=alpha)
            for hits, decisions in zip(bm25_hits, candidate_decisions)
        ]
        fusion_grid[str(alpha)] = retrieval_metrics(
            ordered_queries, fused_hits, report_at_k
        )
    cross_validated_fusion = cross_validated_static_fusion(
        ordered_queries, bm25_hits, candidate_decisions
    )
    cross_validated_hits = cross_validated_fusion.pop("hits")
    cross_validated_paired: dict[str, object] = {}
    for k in report_at_k:
        baseline_outcomes = [
            _hit_at_k(query, hits, k) for query, hits in zip(ordered_queries, bm25_hits)
        ]
        fusion_outcomes = [
            _hit_at_k(query, hits, k)
            for query, hits in zip(ordered_queries, cross_validated_hits)
        ]
        cross_validated_paired[str(k)] = paired_bootstrap_difference(
            baseline_outcomes, fusion_outcomes
        )
    cross_validated_fusion["paired_vs_bm25"] = cross_validated_paired

    paired: dict[str, object] = {}
    for k in report_at_k:
        baseline_outcomes = [
            _hit_at_k(query, hits, k) for query, hits in zip(ordered_queries, bm25_hits)
        ]
        method_outcomes = [
            _hit_at_k(query, hits, k) for query, hits in zip(ordered_queries, rerank_hits)
        ]
        outcome_counts = Counter(
            ("baseline_hit" if before else "baseline_miss")
            + "__"
            + ("rerank_hit" if after else "rerank_miss")
            for before, after in zip(baseline_outcomes, method_outcomes)
        )
        paired[str(k)] = {
            "outcomes": dict(sorted(outcome_counts.items())),
            "bootstrap": paired_bootstrap_difference(baseline_outcomes, method_outcomes),
        }

    per_label: dict[str, object] = {}
    for label in sorted({str(row["label_name"]) for row in ordered_queries}):
        indices = [
            index
            for index, query in enumerate(ordered_queries)
            if str(query["label_name"]) == label
        ]
        per_label[label] = {
            "queries": len(indices),
            "bm25": retrieval_metrics(
                [ordered_queries[index] for index in indices],
                [bm25_hits[index] for index in indices],
                report_at_k,
            ),
            "qwen": retrieval_metrics(
                [ordered_queries[index] for index in indices],
                [rerank_hits[index] for index in indices],
                report_at_k,
            ),
        }

    status_counts = Counter(str(row.get("status", "")) for row in rerank_rows)
    relevance_counts = Counter(
        int(decision.get("relevance", -1))
        for row in rerank_rows
        for decision in row.get("candidate_decisions", [])
    )
    report = {
        "experiment": "trendfact_qwen25_7b_constrained_rerank_evaluation_v1",
        "test_used": False,
        "queries": len(ordered_queries),
        "inputs": {
            "queries_sha256": file_sha256(queries_path),
            "bm25_sha256": file_sha256(bm25_path),
            "rerank_sha256": file_sha256(rerank_path),
            "rerank_config_sha256": file_sha256(rerank_config_path),
        },
        "bm25": baseline_metrics,
        "qwen_rerank": rerank_metrics,
        "static_fusion": {
            "score": "alpha*(1/bm25_rank)+(1-alpha)*(relevance/3+0.01*stance_directness)",
            "same_dev_grid_is_diagnostic_only": fusion_grid,
            "five_fold_cross_validated": cross_validated_fusion,
        },
        "candidate_recall_at_20": round(
            sum(_hit_at_k(query, hits, 20) for query, hits in zip(ordered_queries, bm25_hits))
            / len(ordered_queries),
            6,
        ),
        "paired": paired,
        "per_label": per_label,
        "rerank_status": dict(sorted(status_counts.items())),
        "candidate_relevance_scores": {
            str(score): count for score, count in sorted(relevance_counts.items())
        },
    }
    output_dir = _resolve(rerank_config["output"]["manifest_path"]).parent
    output_path = output_dir / "retrieval_evaluation.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TrendFact constrained LLM reranking.")
    parser.add_argument("--data-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rerank-config", type=Path, default=DEFAULT_RERANK_CONFIG)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(args.data_config, args.rerank_config), ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()

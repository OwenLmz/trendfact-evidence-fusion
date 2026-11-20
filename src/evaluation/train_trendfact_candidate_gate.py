from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evaluation.evaluate_trendfact_reranking import (  # noqa: E402
    _hit_at_k,
    paired_bootstrap_difference,
)
from evaluation.train_trendfact_reliability_gate import _align_inputs  # noqa: E402
from evaluation.train_trendfact_reliability_gate import reciprocal_rank_at_k  # noqa: E402
from retrieval.evidence import file_sha256  # noqa: E402
from retrieval.run_trendfact_bm25_pilot import retrieval_metrics  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/trendfact_candidate_reliability_gate.yaml"
FULL_FEATURE_NAMES = (
    "query_char_length",
    "candidate_char_length",
    "bm25_reciprocal_rank",
    "bm25_score",
    "bm25_score_over_top1",
    "bm25_score_zscore",
    "llm_relevance",
    "llm_relevance_high",
    "llm_direct_stance",
    "llm_support_stance",
    "llm_refute_stance",
    "llm_context_stance",
    "candidate_truncated",
    "excerpt_relative_start",
    "llm_relevance_x_bm25_ratio",
)
BM25_SCORE_FEATURES = (2, 3, 4, 5)
NON_LLM_FEATURES = tuple(range(6))
NO_LENGTH_FULL_FEATURES = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extract_candidate_features(
    query_text: str,
    bm25_hits: Sequence[Mapping[str, object]],
    qwen_row: Mapping[str, object],
    *,
    candidate_top_k: int = 20,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    candidates = [dict(hit) for hit in bm25_hits[:candidate_top_k]]
    decisions = {
        int(decision["candidate_rank"]): decision
        for decision in qwen_row["candidate_decisions"]  # type: ignore[union-attr]
    }
    if len(candidates) != candidate_top_k or len(decisions) < candidate_top_k:
        raise ValueError("Candidate gate requires complete BM25 and Qwen top-k inputs")
    bm25_scores = np.asarray(
        [
            float(candidate.get("bm25_score", candidate.get("score", 0.0)))
            for candidate in candidates
        ],
        dtype=np.float64,
    )
    score_mean = float(bm25_scores.mean())
    score_std = max(float(bm25_scores.std()), 1e-12)
    top1_score = max(float(bm25_scores[0]), 1e-12)
    feature_rows: list[tuple[float, ...]] = []
    for candidate in candidates:
        rank = int(candidate["rank"])
        decision = decisions[rank]
        relevance = max(0, int(decision.get("relevance", -1)))
        stance = str(decision.get("stance", "unknown"))
        candidate_chars = max(1, int(decision.get("candidate_text_chars", 0)))
        excerpt_start = max(0, int(decision.get("prompt_excerpt_start", 0)))
        score = float(candidate.get("bm25_score", candidate.get("score", 0.0)))
        score_ratio = score / top1_score
        feature_rows.append(
            (
                float(len(query_text)),
                float(candidate_chars),
                1.0 / rank,
                score,
                score_ratio,
                (score - score_mean) / score_std,
                float(relevance),
                float(relevance >= 2),
                float(stance in {"support", "refute"}),
                float(stance == "support"),
                float(stance == "refute"),
                float(stance == "context"),
                float(bool(decision.get("candidate_truncated", False))),
                excerpt_start / candidate_chars,
                relevance * score_ratio,
            )
        )
    return np.asarray(feature_rows, dtype=np.float64), candidates


def _fit(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    c: float,
    max_iter: int,
    seed: int,
) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(
        C=c,
        class_weight="balanced",
        max_iter=max_iter,
        random_state=seed,
        solver="liblinear",
    ).fit(scaler.transform(features), targets)
    return scaler, model


def _rank_candidates(
    candidates_by_query: Sequence[Sequence[Mapping[str, object]]],
    probabilities: np.ndarray,
    *,
    output_top_k: int,
) -> list[list[dict[str, object]]]:
    rankings: list[list[dict[str, object]]] = []
    offset = 0
    for candidates in candidates_by_query:
        count = len(candidates)
        query_probabilities = probabilities[offset : offset + count]
        offset += count
        scored = [
            {**dict(candidate), "candidate_reliability": float(probability)}
            for candidate, probability in zip(candidates, query_probabilities)
        ]
        ranked = sorted(
            scored,
            key=lambda candidate: (
                -float(candidate["candidate_reliability"]),
                int(candidate["rank"]),
            ),
        )[:output_top_k]
        for rank, candidate in enumerate(ranked, start=1):
            candidate["gate_rank"] = rank
        rankings.append(ranked)
    if offset != len(probabilities):
        raise ValueError("Unused candidate probabilities remain after ranking")
    return rankings


def _dataset_features(
    queries: Sequence[Mapping[str, object]],
    bm25_hits: Sequence[Sequence[Mapping[str, object]]],
    qwen_rows: Sequence[Mapping[str, object]],
    *,
    candidate_top_k: int,
) -> tuple[np.ndarray, list[list[dict[str, object]]]]:
    matrices: list[np.ndarray] = []
    candidates_by_query: list[list[dict[str, object]]] = []
    for query, hits, qwen in zip(queries, bm25_hits, qwen_rows):
        features, candidates = extract_candidate_features(
            str(query["query_text"]),
            hits,
            qwen,
            candidate_top_k=candidate_top_k,
        )
        matrices.append(features)
        candidates_by_query.append(candidates)
    return np.vstack(matrices), candidates_by_query


def _targets(
    queries: Sequence[Mapping[str, object]],
    candidates_by_query: Sequence[Sequence[Mapping[str, object]]],
) -> np.ndarray:
    labels: list[int] = []
    for query, candidates in zip(queries, candidates_by_query):
        gold = {int(value) for value in query["gold_target_indices"]}  # type: ignore[union-attr]
        labels.extend(int(int(candidate["target_idx"]) in gold) for candidate in candidates)
    return np.asarray(labels, dtype=np.int64)


def _paired_report(
    queries: Sequence[Mapping[str, object]],
    baseline_hits: Sequence[Sequence[Mapping[str, object]]],
    method_hits: Sequence[Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    report = {
        str(k): paired_bootstrap_difference(
            [_hit_at_k(query, hits, k) for query, hits in zip(queries, baseline_hits)],
            [_hit_at_k(query, hits, k) for query, hits in zip(queries, method_hits)],
        )
        for k in (1, 3, 5)
    }
    report["mrr_at_5"] = paired_bootstrap_difference(
        [reciprocal_rank_at_k(query, hits) for query, hits in zip(queries, baseline_hits)],
        [reciprocal_rank_at_k(query, hits) for query, hits in zip(queries, method_hits)],
    )
    return report


def _fit_and_rank_subset(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    dev_features: np.ndarray,
    dev_candidates: Sequence[Sequence[Mapping[str, object]]],
    feature_indices: Sequence[int],
    *,
    c: float,
    max_iter: int,
    seed: int,
    output_top_k: int,
) -> list[list[dict[str, object]]]:
    indices = tuple(feature_indices)
    scaler, model = _fit(
        train_features[:, indices],
        train_targets,
        c=c,
        max_iter=max_iter,
        seed=seed,
    )
    probability = model.predict_proba(
        scaler.transform(dev_features[:, indices])
    )[:, 1]
    return _rank_candidates(dev_candidates, probability, output_top_k=output_top_k)


def train_and_evaluate(config_path: Path = DEFAULT_CONFIG) -> dict[str, object]:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = {key: _resolve(value) for key, value in config["input"].items()}
    train_queries, train_bm25, train_qwen = _align_inputs(
        _read_jsonl(paths["train_queries"]),
        _read_jsonl(paths["train_bm25"]),
        _read_jsonl(paths["train_qwen"]),
    )
    dev_queries, dev_bm25, dev_qwen = _align_inputs(
        _read_jsonl(paths["dev_queries"]),
        _read_jsonl(paths["dev_bm25"]),
        _read_jsonl(paths["dev_qwen"]),
    )
    if {str(query["split"]) for query in train_queries} != {"train"}:
        raise ValueError("Candidate gate fitting must be train-only")
    if {str(query["split"]) for query in dev_queries} != {"dev"}:
        raise ValueError("Candidate gate evaluation must be dev-only")

    model_config = config["model"]
    candidate_top_k = int(model_config["candidate_top_k"])
    output_top_k = int(model_config["output_top_k"])
    train_features, train_candidates = _dataset_features(
        train_queries,
        train_bm25,
        train_qwen,
        candidate_top_k=candidate_top_k,
    )
    train_targets = _targets(train_queries, train_candidates)
    dev_features, dev_candidates = _dataset_features(
        dev_queries,
        dev_bm25,
        dev_qwen,
        candidate_top_k=candidate_top_k,
    )
    seed = int(config["seed"])

    full_scaler, full_model = _fit(
        train_features,
        train_targets,
        c=float(model_config["c"]),
        max_iter=int(model_config["max_iter"]),
        seed=seed,
    )
    full_probability = full_model.predict_proba(
        full_scaler.transform(dev_features)
    )[:, 1]
    full_rankings = _rank_candidates(
        dev_candidates, full_probability, output_top_k=output_top_k
    )

    bm25_score_rankings = _fit_and_rank_subset(
        train_features,
        train_targets,
        dev_features,
        dev_candidates,
        BM25_SCORE_FEATURES,
        c=float(model_config["c"]),
        max_iter=int(model_config["max_iter"]),
        seed=seed,
        output_top_k=output_top_k,
    )
    non_llm_rankings = _fit_and_rank_subset(
        train_features,
        train_targets,
        dev_features,
        dev_candidates,
        NON_LLM_FEATURES,
        c=float(model_config["c"]),
        max_iter=int(model_config["max_iter"]),
        seed=seed,
        output_top_k=output_top_k,
    )
    no_length_full_rankings = _fit_and_rank_subset(
        train_features,
        train_targets,
        dev_features,
        dev_candidates,
        NO_LENGTH_FULL_FEATURES,
        c=float(model_config["c"]),
        max_iter=int(model_config["max_iter"]),
        seed=seed,
        output_top_k=output_top_k,
    )

    qwen_rankings = [list(row["selected"]) for row in dev_qwen]
    report_at_k = [1, 3, 5]
    metrics = {
        "bm25": retrieval_metrics(dev_queries, dev_bm25, report_at_k),
        "qwen": retrieval_metrics(dev_queries, qwen_rankings, report_at_k),
        "learned_bm25_score_only": retrieval_metrics(
            dev_queries, bm25_score_rankings, report_at_k
        ),
        "learned_non_llm_with_length": retrieval_metrics(
            dev_queries, non_llm_rankings, report_at_k
        ),
        "candidate_gate_without_length_features": retrieval_metrics(
            dev_queries, no_length_full_rankings, report_at_k
        ),
        "candidate_reliability_gate": retrieval_metrics(
            dev_queries, full_rankings, report_at_k
        ),
    }

    per_label: dict[str, object] = {}
    for label in sorted({str(query["label_name"]) for query in dev_queries}):
        indices = [
            index
            for index, query in enumerate(dev_queries)
            if str(query["label_name"]) == label
        ]
        per_label[label] = {
            name: retrieval_metrics(
                [dev_queries[index] for index in indices],
                [rankings[index] for index in indices],
                report_at_k,
            )
            for name, rankings in (
                ("bm25", dev_bm25),
                ("qwen", qwen_rankings),
                ("candidate_reliability_gate", full_rankings),
            )
        }

    output_dir = _resolve(config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = output_dir / str(config["output"]["dev_rankings"])
    with ranking_path.open("w", encoding="utf-8", newline="\n") as handle:
        for query, rankings in zip(dev_queries, full_rankings):
            handle.write(
                json.dumps(
                    {
                        "query_id": query["query_id"],
                        "split": "dev",
                        "retrieval_method": "candidate_reliability_gate",
                        "hits": rankings,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    model_artifact = {
        "feature_names": list(FULL_FEATURE_NAMES),
        "scaler_mean": full_scaler.mean_.tolist(),
        "scaler_scale": full_scaler.scale_.tolist(),
        "logistic_coefficient": full_model.coef_[0].tolist(),
        "logistic_intercept": float(full_model.intercept_[0]),
        "train_candidate_pairs": len(train_targets),
        "train_positive_pairs": int(train_targets.sum()),
        "test_used": False,
    }
    model_path = output_dir / str(config["output"]["model"])
    model_path.write_text(
        json.dumps(model_artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = {
        "experiment": config["experiment"],
        "test_used": False,
        "inputs": {key: file_sha256(path) for key, path in paths.items()},
        "train": {
            "queries": len(train_queries),
            "candidate_pairs": len(train_targets),
            "positive_gold_pairs": int(train_targets.sum()),
            "negative_pairs": int(len(train_targets) - train_targets.sum()),
        },
        "dev": {
            "queries": len(dev_queries),
            "metrics": metrics,
            "per_label": per_label,
            "paired_candidate_gate_vs": {
                "bm25": _paired_report(dev_queries, dev_bm25, full_rankings),
                "qwen": _paired_report(dev_queries, qwen_rankings, full_rankings),
                "learned_bm25_score_only": _paired_report(
                    dev_queries, bm25_score_rankings, full_rankings
                ),
                "learned_non_llm_with_length": _paired_report(
                    dev_queries, non_llm_rankings, full_rankings
                ),
                "candidate_gate_without_length_features": _paired_report(
                    dev_queries, no_length_full_rankings, full_rankings
                ),
            },
        },
        "artifacts": {
            "model": model_path.relative_to(ROOT).as_posix(),
            "model_sha256": file_sha256(model_path),
            "dev_rankings": ranking_path.relative_to(ROOT).as_posix(),
            "dev_rankings_sha256": file_sha256(ranking_path),
        },
    }
    report_path = output_dir / str(config["output"]["report"])
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the TrendFact candidate reliability gate."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(train_and_evaluate(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

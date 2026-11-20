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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evaluation.evaluate_trendfact_reranking import (  # noqa: E402
    _hit_at_k,
    paired_bootstrap_difference,
)
from retrieval.evidence import file_sha256  # noqa: E402
from retrieval.run_trendfact_bm25_pilot import retrieval_metrics  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/trendfact_reliability_gate.yaml"
FEATURE_NAMES = (
    "query_char_length",
    "bm25_top1_score",
    "bm25_top1_top2_gap",
    "bm25_top1_top5_ratio",
    "llm_relevance_max",
    "llm_relevance_mean",
    "llm_relevance_std",
    "llm_relevance_margin",
    "llm_relevance_3_count",
    "llm_relevance_2plus_count",
    "llm_direct_stance_count",
    "qwen_top1_original_rank",
    "qwen_top1_agrees_bm25",
    "qwen_top5_bm25_overlap",
    "candidate_truncated_fraction",
)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def reciprocal_rank_at_k(
    query: Mapping[str, object], hits: Sequence[Mapping[str, object]], k: int = 5
) -> float:
    gold = {int(value) for value in query["gold_target_indices"]}  # type: ignore[union-attr]
    for rank, hit in enumerate(hits[:k], start=1):
        if int(hit["target_idx"]) in gold:
            return 1.0 / rank
    return 0.0


def extract_reliability_features(
    query_text: str,
    bm25_hits: Sequence[Mapping[str, object]],
    qwen_row: Mapping[str, object],
) -> np.ndarray:
    if len(bm25_hits) < 5:
        raise ValueError("Reliability features require at least five BM25 candidates")
    decisions = list(qwen_row["candidate_decisions"])  # type: ignore[arg-type]
    selected = list(qwen_row["selected"])  # type: ignore[arg-type]
    if not decisions or not selected:
        raise ValueError("Reliability features require Qwen decisions and selected evidence")
    scores = [
        float(hit.get("bm25_score", hit.get("score", 0.0))) for hit in bm25_hits
    ]
    relevance = np.asarray(
        [max(0, int(decision.get("relevance", -1))) for decision in decisions],
        dtype=np.float64,
    )
    sorted_relevance = sorted(relevance.tolist(), reverse=True)
    top1 = scores[0]
    top2 = scores[1]
    top5 = scores[4]
    qwen_ranks = [int(hit["rank"]) for hit in selected[:5]]
    direct_count = sum(
        str(decision.get("stance", "unknown")) in {"support", "refute"}
        for decision in decisions
    )
    truncated_fraction = sum(
        bool(decision.get("candidate_truncated", False)) for decision in decisions
    ) / len(decisions)
    values = (
        float(len(query_text)),
        top1,
        top1 - top2,
        top1 / max(top5, 1e-12),
        float(relevance.max()),
        float(relevance.mean()),
        float(relevance.std()),
        float(sorted_relevance[0] - sorted_relevance[1]),
        float(np.sum(relevance == 3)),
        float(np.sum(relevance >= 2)),
        float(direct_count),
        float(qwen_ranks[0]),
        float(qwen_ranks[0] == 1),
        float(len(set(qwen_ranks) & {1, 2, 3, 4, 5}) / 5.0),
        float(truncated_fraction),
    )
    return np.asarray(values, dtype=np.float64)


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


def _fit_model(
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


def _select_rankings(
    bm25_hits: Sequence[Sequence[Mapping[str, object]]],
    qwen_rows: Sequence[Mapping[str, object]],
    probabilities: Sequence[float],
    *,
    threshold: float,
) -> tuple[list[list[dict[str, object]]], list[str]]:
    selected_hits: list[list[dict[str, object]]] = []
    decisions: list[str] = []
    for bm25, qwen, probability in zip(bm25_hits, qwen_rows, probabilities):
        use_qwen = float(probability) >= threshold
        decisions.append("qwen" if use_qwen else "bm25")
        source = list(qwen["selected"]) if use_qwen else list(bm25[:5])  # type: ignore[arg-type]
        selected_hits.append([dict(hit) for hit in source])
    return selected_hits, decisions


def _metric_key(
    queries: Sequence[Mapping[str, object]],
    hits: Sequence[Sequence[Mapping[str, object]]],
    qwen_decisions: int,
) -> tuple[float, float, int]:
    metrics = retrieval_metrics(queries, hits, [1, 3, 5])
    recall = metrics["any_gold_recall_at_k"]
    assert isinstance(recall, Mapping)
    return (
        float(metrics["mrr_at_max_k"]),
        float(recall["5"]),
        -qwen_decisions,
    )


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
    if {str(row["split"]) for row in train_queries} != {"train"}:
        raise ValueError("Gate fitting input must be train-only")
    if {str(row["split"]) for row in dev_queries} != {"dev"}:
        raise ValueError("Gate evaluation input must be dev-only")

    train_features = np.vstack(
        [
            extract_reliability_features(str(query["query_text"]), bm25, qwen)
            for query, bm25, qwen in zip(train_queries, train_bm25, train_qwen)
        ]
    )
    train_delta = np.asarray(
        [
            reciprocal_rank_at_k(query, list(qwen["selected"]))
            - reciprocal_rank_at_k(query, bm25)
            for query, bm25, qwen in zip(train_queries, train_bm25, train_qwen)
        ]
    )
    informative = np.flatnonzero(train_delta != 0.0)
    targets = (train_delta[informative] > 0.0).astype(np.int64)
    if len(informative) < 20 or len(set(targets.tolist())) != 2:
        raise ValueError("Not enough non-tied train queries to fit a reliability gate")

    model_config = config["model"]
    threshold_config = config["threshold_selection"]
    seed = int(config["seed"])
    folds = int(threshold_config["folds"])
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof_probability = np.zeros(len(informative), dtype=np.float64)
    informative_features = train_features[informative]
    for fit_indices, validation_indices in splitter.split(informative_features, targets):
        scaler, model = _fit_model(
            informative_features[fit_indices],
            targets[fit_indices],
            c=float(model_config["c"]),
            max_iter=int(model_config["max_iter"]),
            seed=seed,
        )
        oof_probability[validation_indices] = model.predict_proba(
            scaler.transform(informative_features[validation_indices])
        )[:, 1]

    threshold_candidates: list[dict[str, object]] = []
    informative_queries = [train_queries[index] for index in informative]
    informative_bm25 = [train_bm25[index] for index in informative]
    informative_qwen = [train_qwen[index] for index in informative]
    for threshold in threshold_config["grid"]:
        hits, choices = _select_rankings(
            informative_bm25,
            informative_qwen,
            oof_probability,
            threshold=float(threshold),
        )
        metrics = retrieval_metrics(informative_queries, hits, [1, 3, 5])
        threshold_candidates.append(
            {
                "threshold": float(threshold),
                "qwen_decisions": choices.count("qwen"),
                "metrics": metrics,
                "selection_key": _metric_key(
                    informative_queries, hits, choices.count("qwen")
                ),
            }
        )
    selected_threshold_row = max(
        threshold_candidates, key=lambda row: tuple(row["selection_key"])
    )
    selected_threshold = float(selected_threshold_row["threshold"])

    scaler, model = _fit_model(
        informative_features,
        targets,
        c=float(model_config["c"]),
        max_iter=int(model_config["max_iter"]),
        seed=seed,
    )
    dev_features = np.vstack(
        [
            extract_reliability_features(str(query["query_text"]), bm25, qwen)
            for query, bm25, qwen in zip(dev_queries, dev_bm25, dev_qwen)
        ]
    )
    dev_probability = model.predict_proba(scaler.transform(dev_features))[:, 1]
    gate_hits, gate_choices = _select_rankings(
        dev_bm25, dev_qwen, dev_probability, threshold=selected_threshold
    )
    qwen_hits = [list(row["selected"]) for row in dev_qwen]
    report_at_k = [1, 3, 5]
    bm25_metrics = retrieval_metrics(dev_queries, dev_bm25, report_at_k)
    qwen_metrics = retrieval_metrics(dev_queries, qwen_hits, report_at_k)
    gate_metrics = retrieval_metrics(dev_queries, gate_hits, report_at_k)
    oracle_hits = [
        qwen if reciprocal_rank_at_k(query, qwen) > reciprocal_rank_at_k(query, bm25)
        else list(bm25[:5])
        for query, bm25, qwen in zip(dev_queries, dev_bm25, qwen_hits)
    ]

    paired: dict[str, object] = {}
    for method_name, comparison_hits in (("bm25", dev_bm25), ("qwen", qwen_hits)):
        paired[method_name] = {
            str(k): paired_bootstrap_difference(
                [_hit_at_k(query, hits, k) for query, hits in zip(dev_queries, comparison_hits)],
                [_hit_at_k(query, hits, k) for query, hits in zip(dev_queries, gate_hits)],
            )
            for k in report_at_k
        }

    output_dir = _resolve(config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = output_dir / str(config["output"]["dev_rankings"])
    with ranking_path.open("w", encoding="utf-8", newline="\n") as handle:
        for query, hits, probability, choice in zip(
            dev_queries, gate_hits, dev_probability, gate_choices
        ):
            handle.write(
                json.dumps(
                    {
                        "query_id": query["query_id"],
                        "split": "dev",
                        "qwen_reliability": round(float(probability), 8),
                        "selected_method": choice,
                        "hits": hits,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    model_artifact = {
        "feature_names": list(FEATURE_NAMES),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "logistic_coefficient": model.coef_[0].tolist(),
        "logistic_intercept": float(model.intercept_[0]),
        "threshold": selected_threshold,
        "seed": seed,
        "train_informative_queries": len(informative),
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
            "informative_non_ties": len(informative),
            "qwen_better": int(np.sum(targets == 1)),
            "bm25_better": int(np.sum(targets == 0)),
            "tied": int(len(train_queries) - len(informative)),
            "selected_threshold": selected_threshold,
            "threshold_cross_validation": threshold_candidates,
        },
        "dev": {
            "queries": len(dev_queries),
            "gate_choices": dict(sorted(Counter(gate_choices).items())),
            "bm25": bm25_metrics,
            "qwen": qwen_metrics,
            "reliability_gate": gate_metrics,
            "oracle_between_bm25_and_qwen": retrieval_metrics(
                dev_queries, oracle_hits, report_at_k
            ),
            "paired_gate_vs": paired,
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
    parser = argparse.ArgumentParser(description="Train and evaluate the TrendFact reliability gate.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(train_and_evaluate(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

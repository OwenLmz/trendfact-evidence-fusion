from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evaluation.train_trendfact_candidate_gate import (  # noqa: E402
    NON_LLM_FEATURES,
    _fit,
    _rank_candidates,
    extract_candidate_features,
)
from retrieval.evidence import file_sha256  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/trendfact_evidence_aggregator.yaml"
STANCES = ("support", "refute", "context", "unknown")


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def aggregate_ranked_signals(
    ranked_signals: Sequence[Mapping[str, object]], *, top_k: int = 5
) -> tuple[np.ndarray, list[str]]:
    signals = list(ranked_signals[:top_k])
    if len(signals) != top_k:
        raise ValueError(f"Expected {top_k} ranked evidence signals, got {len(signals)}")
    names: list[str] = []
    values: list[float] = []
    for position, signal in enumerate(signals, start=1):
        probability = float(signal["candidate_reliability"])
        relevance = float(signal["relevance"]) / 3.0
        names.extend(
            [
                f"rank{position}_candidate_reliability",
                f"rank{position}_llm_relevance",
                f"rank{position}_bm25_reciprocal_rank",
            ]
        )
        values.extend(
            [
                probability,
                relevance,
                1.0 / max(int(signal["bm25_rank"]), 1),
            ]
        )
        stance = str(signal.get("stance", "unknown"))
        for stance_name in STANCES:
            names.append(f"rank{position}_stance_{stance_name}")
            values.append(float(stance == stance_name))

    for stance_name in STANCES:
        stance_rows = [row for row in signals if str(row.get("stance", "unknown")) == stance_name]
        weighted = [
            float(row["candidate_reliability"]) * (float(row["relevance"]) + 1.0) / 4.0
            for row in stance_rows
        ]
        names.extend(
            [
                f"{stance_name}_weighted_sum",
                f"{stance_name}_weighted_max",
                f"{stance_name}_high_relevance_fraction",
            ]
        )
        values.extend(
            [
                float(sum(weighted)),
                float(max(weighted, default=0.0)),
                sum(float(row["relevance"]) >= 2 for row in stance_rows) / top_k,
            ]
        )

    probabilities = np.asarray(
        [float(row["candidate_reliability"]) for row in signals], dtype=np.float64
    )
    relevances = np.asarray([float(row["relevance"]) for row in signals], dtype=np.float64)
    names.extend(
        [
            "reliability_mean",
            "reliability_max",
            "reliability_top1_top2_margin",
            "relevance_mean",
            "relevance_max",
            "high_relevance_fraction",
        ]
    )
    values.extend(
        [
            float(probabilities.mean()),
            float(probabilities.max()),
            float(probabilities[0] - probabilities[1]),
            float(relevances.mean() / 3.0),
            float(relevances.max() / 3.0),
            float(np.mean(relevances >= 2)),
        ]
    )
    return np.asarray(values, dtype=np.float64), names


def _align(
    queries: Sequence[Mapping[str, object]],
    bm25_rows: Sequence[Mapping[str, object]],
    qwen_rows: Sequence[Mapping[str, object]],
    *,
    candidate_top_k: int,
    allow_test: bool = False,
) -> None:
    query_ids = [str(row["query_id"]) for row in queries]
    if query_ids != [str(row["query_id"]) for row in bm25_rows]:
        raise ValueError("BM25 rows do not align with queries")
    if query_ids != [str(row["query_id"]) for row in qwen_rows]:
        raise ValueError("Qwen rows do not align with queries")
    if any(len(row["hits"]) < candidate_top_k for row in bm25_rows):  # type: ignore[arg-type]
        raise ValueError("BM25 candidate pool is incomplete")
    if any(len(row["candidate_decisions"]) < candidate_top_k for row in qwen_rows):  # type: ignore[arg-type]
        raise ValueError("Qwen candidate decisions are incomplete")
    splits = {str(row["split"]) for row in queries}
    if len(splits) != 1:
        raise ValueError(f"Expected one split, got {sorted(splits)}")
    if "test" in splits and not allow_test:
        raise ValueError("Test evidence aggregation requires --allow-final-test")


def _candidate_dataset(
    queries: Sequence[Mapping[str, object]],
    bm25_rows: Sequence[Mapping[str, object]],
    qwen_rows: Sequence[Mapping[str, object]],
    *,
    candidate_top_k: int,
) -> tuple[list[np.ndarray], list[list[dict[str, object]]], list[np.ndarray]]:
    features_by_query: list[np.ndarray] = []
    candidates_by_query: list[list[dict[str, object]]] = []
    targets_by_query: list[np.ndarray] = []
    for query, bm25, qwen in zip(queries, bm25_rows, qwen_rows):
        features, candidates = extract_candidate_features(
            str(query["query_text"]),
            bm25["hits"],  # type: ignore[arg-type]
            qwen,
            candidate_top_k=candidate_top_k,
        )
        gold = {int(value) for value in query["gold_target_indices"]}  # type: ignore[union-attr]
        targets = np.asarray(
            [int(int(candidate["target_idx"]) in gold) for candidate in candidates],
            dtype=np.int64,
        )
        features_by_query.append(features)
        candidates_by_query.append(candidates)
        targets_by_query.append(targets)
    return features_by_query, candidates_by_query, targets_by_query


def _probabilities_from_frozen_model(
    features_by_query: Sequence[np.ndarray], model: Mapping[str, object]
) -> list[np.ndarray]:
    mean = np.asarray(model["scaler_mean"], dtype=np.float64)
    scale = np.asarray(model["scaler_scale"], dtype=np.float64)
    coefficient = np.asarray(model["logistic_coefficient"], dtype=np.float64)
    intercept = float(model["logistic_intercept"])
    probabilities = []
    for features in features_by_query:
        logits = ((features - mean) / scale) @ coefficient + intercept
        probabilities.append(1.0 / (1.0 + np.exp(-logits)))
    return probabilities


def _oof_gate_probabilities(
    features_by_query: Sequence[np.ndarray],
    targets_by_query: Sequence[np.ndarray],
    claim_labels: np.ndarray,
    *,
    folds: int,
    seed: int,
    c: float,
    max_iter: int,
    feature_indices: Sequence[int] | None = None,
) -> list[np.ndarray]:
    output = [np.zeros(len(features), dtype=np.float64) for features in features_by_query]
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    indices = np.arange(len(features_by_query))
    selected = tuple(feature_indices) if feature_indices is not None else None
    for train_indices, valid_indices in splitter.split(indices, claim_labels):
        train_features = np.vstack([features_by_query[index] for index in train_indices])
        train_targets = np.concatenate([targets_by_query[index] for index in train_indices])
        if selected is not None:
            train_features = train_features[:, selected]
        scaler, model = _fit(
            train_features, train_targets, c=c, max_iter=max_iter, seed=seed
        )
        for index in valid_indices:
            validation_features = features_by_query[index]
            if selected is not None:
                validation_features = validation_features[:, selected]
            output[index] = model.predict_proba(
                scaler.transform(validation_features)
            )[:, 1]
    return output


def _fit_gate_probabilities(
    train_features_by_query: Sequence[np.ndarray],
    train_targets_by_query: Sequence[np.ndarray],
    eval_features_by_query: Sequence[np.ndarray],
    *,
    c: float,
    max_iter: int,
    seed: int,
    feature_indices: Sequence[int] | None = None,
) -> list[np.ndarray]:
    selected = tuple(feature_indices) if feature_indices is not None else None
    train_features = np.vstack(train_features_by_query)
    eval_features = np.vstack(eval_features_by_query)
    if selected is not None:
        train_features = train_features[:, selected]
        eval_features = eval_features[:, selected]
    scaler, model = _fit(
        train_features,
        np.concatenate(train_targets_by_query),
        c=c,
        max_iter=max_iter,
        seed=seed,
    )
    flat = model.predict_proba(scaler.transform(eval_features))[:, 1]
    probabilities: list[np.ndarray] = []
    offset = 0
    for features in eval_features_by_query:
        probabilities.append(flat[offset : offset + len(features)])
        offset += len(features)
    if offset != len(flat):
        raise ValueError("Candidate probability split did not consume all rows")
    return probabilities


def _direct_qwen_probabilities(
    features_by_query: Sequence[np.ndarray],
) -> list[np.ndarray]:
    probabilities: list[np.ndarray] = []
    for features in features_by_query:
        relevance = features[:, 6]
        direct_stance = features[:, 8]
        probabilities.append((relevance + 0.01 * direct_stance) / 3.01)
    return probabilities


def _aggregation_matrix(
    candidates_by_query: Sequence[Sequence[Mapping[str, object]]],
    qwen_rows: Sequence[Mapping[str, object]],
    probabilities_by_query: Sequence[np.ndarray],
    *,
    top_k: int,
) -> tuple[np.ndarray, list[str]]:
    rows: list[np.ndarray] = []
    names: list[str] | None = None
    for candidates, qwen, probabilities in zip(
        candidates_by_query, qwen_rows, probabilities_by_query
    ):
        ranked = _rank_candidates(
            [candidates], probabilities, output_top_k=top_k
        )[0]
        decisions = {
            int(decision["candidate_rank"]): decision
            for decision in qwen["candidate_decisions"]  # type: ignore[union-attr]
        }
        signals = []
        for candidate in ranked:
            bm25_rank = int(candidate["rank"])
            decision = decisions[bm25_rank]
            signals.append(
                {
                    "candidate_reliability": float(candidate["candidate_reliability"]),
                    "relevance": max(0, int(decision.get("relevance", -1))),
                    "stance": str(decision.get("stance", "unknown")),
                    "bm25_rank": bm25_rank,
                }
            )
        row, row_names = aggregate_ranked_signals(signals, top_k=top_k)
        if names is None:
            names = row_names
        elif names != row_names:
            raise ValueError("Aggregation feature order changed between queries")
        rows.append(row)
    if names is None:
        raise ValueError("No aggregation features were generated")
    return np.vstack(rows), names


def _structural_aggregation_features(
    features: np.ndarray, feature_names: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    selected = [
        index
        for index, name in enumerate(feature_names)
        if "candidate_reliability" in name
        or "bm25_reciprocal_rank" in name
        or name.startswith("reliability_")
    ]
    return features[:, selected], [feature_names[index] for index in selected]


def _classification_metrics(
    labels: np.ndarray, predictions: np.ndarray, label_names: Sequence[str]
) -> dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=np.arange(len(label_names)),
        zero_division=0,
    )
    return {
        "samples": int(len(labels)),
        "accuracy": round(float(accuracy_score(labels, predictions)), 6),
        "macro_f1": round(float(np.mean(f1)), 6),
        "per_class": {
            name: {
                "precision": round(float(precision[index]), 6),
                "recall": round(float(recall[index]), 6),
                "f1": round(float(f1[index]), 6),
                "support": int(support[index]),
            }
            for index, name in enumerate(label_names)
        },
    }


def _paired_bootstrap(
    labels: np.ndarray,
    baseline: np.ndarray,
    method: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)

    def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(
            np.mean(
                precision_recall_fscore_support(
                    y_true,
                    y_pred,
                    labels=np.arange(3),
                    zero_division=0,
                )[2]
            )
        )

    differences = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.integers(0, len(labels), len(labels))
        differences[index] = macro_f1(labels[selected], method[selected]) - macro_f1(
            labels[selected], baseline[selected]
        )
    observed = macro_f1(labels, method) - macro_f1(labels, baseline)
    return {
        "metric": "macro_f1",
        "difference": round(observed, 6),
        "ci95": [
            round(float(np.percentile(differences, 2.5)), 6),
            round(float(np.percentile(differences, 97.5)), 6),
        ],
        "bootstrap_samples": samples,
    }


def _claim_pipeline(config: Mapping[str, object]) -> Pipeline:
    claim = config["claim_model"]  # type: ignore[index]
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=tuple(int(value) for value in claim["ngram_range"]),
                    min_df=int(claim["min_df"]),
                    max_features=int(claim["max_features"]),
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    C=float(claim["c"]),
                    class_weight="balanced",
                    random_state=int(config["seed"]),
                ),
            ),
        ]
    )


def _oof_claim_scores(
    texts: Sequence[str], labels: np.ndarray, config: Mapping[str, object]
) -> tuple[np.ndarray, Pipeline]:
    folds = int(config["training"]["folds"])  # type: ignore[index]
    splitter = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=int(config["seed"])
    )
    scores = np.zeros((len(texts), 3), dtype=np.float64)
    indices = np.arange(len(texts))
    base = _claim_pipeline(config)
    text_array = np.asarray(texts, dtype=object)
    for train_indices, valid_indices in splitter.split(indices, labels):
        model = clone(base).fit(text_array[train_indices], labels[train_indices])
        scores[valid_indices] = model.decision_function(text_array[valid_indices])
    final_model = clone(base).fit(text_array, labels)
    return scores, final_model


def _multiclass_logistic(seed: int, c: float, max_iter: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c,
                    class_weight="balanced",
                    max_iter=max_iter,
                    random_state=seed,
                ),
            ),
        ]
    )


def _rule_predictions(features: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    positions = {name: index for index, name in enumerate(feature_names)}
    support = features[:, positions["support_weighted_sum"]]
    refute = features[:, positions["refute_weighted_sum"]]
    context = features[:, positions["context_weighted_sum"]]
    high = features[:, positions["high_relevance_fraction"]]
    output = np.full(len(features), 2, dtype=np.int64)
    decisive = (high > 0) & (np.maximum(support, refute) >= context)
    output[decisive & (support >= refute)] = 0
    output[decisive & (refute > support)] = 1
    return output


def _dataset_inputs(config: Mapping[str, object], section: str) -> tuple[Path, Path, Path]:
    values = config["input"][section]  # type: ignore[index]
    return (
        _resolve(values["queries"]),
        _resolve(values["bm25"]),
        _resolve(values["qwen"]),
    )


def _load_dataset(
    paths: tuple[Path, Path, Path], *, candidate_top_k: int, allow_test: bool = False
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    queries, bm25, qwen = (_read_jsonl(path) for path in paths)
    _align(
        queries,
        bm25,
        qwen,
        candidate_top_k=candidate_top_k,
        allow_test=allow_test,
    )
    return queries, bm25, qwen


def run(
    config_path: Path = DEFAULT_CONFIG,
    *,
    evaluation_split: str,
    output_dir: Path,
    overwrite: bool = False,
    allow_test: bool = False,
) -> dict[str, object]:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if evaluation_split not in {"pilot", "heldout_dev", "test"}:
        raise ValueError("evaluation_split must be pilot, heldout_dev, or test")
    is_test = evaluation_split == "test"
    if is_test and not allow_test:
        raise ValueError("Test evidence aggregation requires --allow-final-test")
    if is_test and overwrite:
        raise ValueError("Final-test evidence output may not be overwritten")
    candidate_top_k = int(config["candidate_gate"]["candidate_top_k"])
    aggregate_top_k = int(config["aggregation"]["top_k"])
    train_paths = _dataset_inputs(config, "train")
    evaluation_paths = _dataset_inputs(config, evaluation_split)
    train_queries, train_bm25, train_qwen = _load_dataset(
        train_paths, candidate_top_k=candidate_top_k
    )
    eval_queries, eval_bm25, eval_qwen = _load_dataset(
        evaluation_paths,
        candidate_top_k=candidate_top_k,
        allow_test=is_test and allow_test,
    )
    train_labels = np.asarray([int(row["label_id"]) for row in train_queries], dtype=np.int64)
    eval_labels = np.asarray([int(row["label_id"]) for row in eval_queries], dtype=np.int64)
    gate_model_path = _resolve(config["candidate_gate"]["model_json"])
    gate_model = json.loads(gate_model_path.read_text(encoding="utf-8"))

    train_features, train_candidates, train_targets = _candidate_dataset(
        train_queries,
        train_bm25,
        train_qwen,
        candidate_top_k=candidate_top_k,
    )
    eval_features, eval_candidates, _ = _candidate_dataset(
        eval_queries,
        eval_bm25,
        eval_qwen,
        candidate_top_k=candidate_top_k,
    )
    training = config["training"]
    train_probabilities = _oof_gate_probabilities(
        train_features,
        train_targets,
        train_labels,
        folds=int(training["folds"]),
        seed=int(config["seed"]),
        c=float(config["candidate_gate"]["c"]),
        max_iter=int(config["candidate_gate"]["max_iter"]),
    )
    eval_probabilities = _probabilities_from_frozen_model(eval_features, gate_model)
    train_full_aggregate, feature_names = _aggregation_matrix(
        train_candidates,
        train_qwen,
        train_probabilities,
        top_k=aggregate_top_k,
    )
    eval_full_aggregate, eval_feature_names = _aggregation_matrix(
        eval_candidates,
        eval_qwen,
        eval_probabilities,
        top_k=aggregate_top_k,
    )
    if feature_names != eval_feature_names:
        raise ValueError("Train and evaluation aggregation features differ")

    seed = int(config["seed"])
    logistic_c = float(config["aggregation"]["c"])
    max_iter = int(config["aggregation"]["max_iter"])
    non_llm_train_probabilities = _oof_gate_probabilities(
        train_features,
        train_targets,
        train_labels,
        folds=int(training["folds"]),
        seed=seed,
        c=float(config["candidate_gate"]["c"]),
        max_iter=int(config["candidate_gate"]["max_iter"]),
        feature_indices=NON_LLM_FEATURES,
    )
    non_llm_eval_probabilities = _fit_gate_probabilities(
        train_features,
        train_targets,
        eval_features,
        c=float(config["candidate_gate"]["c"]),
        max_iter=int(config["candidate_gate"]["max_iter"]),
        seed=seed,
        feature_indices=NON_LLM_FEATURES,
    )
    train_non_llm_aggregate, non_llm_feature_names = _aggregation_matrix(
        train_candidates,
        train_qwen,
        non_llm_train_probabilities,
        top_k=aggregate_top_k,
    )
    eval_non_llm_aggregate, eval_non_llm_feature_names = _aggregation_matrix(
        eval_candidates,
        eval_qwen,
        non_llm_eval_probabilities,
        top_k=aggregate_top_k,
    )
    train_non_llm_aggregate, non_llm_feature_names = _structural_aggregation_features(
        train_non_llm_aggregate, non_llm_feature_names
    )
    eval_non_llm_aggregate, eval_non_llm_feature_names = _structural_aggregation_features(
        eval_non_llm_aggregate, eval_non_llm_feature_names
    )
    if non_llm_feature_names != eval_non_llm_feature_names:
        raise ValueError("Train and evaluation non-LLM features differ")

    train_direct_aggregate, direct_feature_names = _aggregation_matrix(
        train_candidates,
        train_qwen,
        _direct_qwen_probabilities(train_features),
        top_k=aggregate_top_k,
    )
    eval_direct_aggregate, eval_direct_feature_names = _aggregation_matrix(
        eval_candidates,
        eval_qwen,
        _direct_qwen_probabilities(eval_features),
        top_k=aggregate_top_k,
    )
    if direct_feature_names != eval_direct_feature_names:
        raise ValueError("Train and evaluation direct-Qwen features differ")

    aggregate_variants = {
        "non_llm_structural_aggregator": (
            train_non_llm_aggregate,
            eval_non_llm_aggregate,
        ),
        "direct_qwen_no_gate_aggregator": (
            train_direct_aggregate,
            eval_direct_aggregate,
        ),
        "evidence_only_calibrated_aggregator": (
            train_full_aggregate,
            eval_full_aggregate,
        ),
    }
    evidence_probabilities: dict[str, np.ndarray] = {}
    evidence_predictions: dict[str, np.ndarray] = {}
    for name, (train_aggregate, eval_aggregate) in aggregate_variants.items():
        model = _multiclass_logistic(seed, logistic_c, max_iter).fit(
            train_aggregate, train_labels
        )
        variant_probabilities = model.predict_proba(eval_aggregate)
        evidence_probabilities[name] = variant_probabilities
        evidence_predictions[name] = model.classes_[
            np.argmax(variant_probabilities, axis=1)
        ]
    train_texts = [str(row["query_text"]) for row in train_queries]
    eval_texts = [str(row["query_text"]) for row in eval_queries]
    train_claim_scores, claim_model = _oof_claim_scores(train_texts, train_labels, config)
    eval_claim_scores = claim_model.decision_function(eval_texts)
    claim_predictions = claim_model.predict(eval_texts)
    fusion_model = _multiclass_logistic(seed, logistic_c, max_iter).fit(
        np.hstack([train_claim_scores, train_full_aggregate]), train_labels
    )
    fusion_probabilities = fusion_model.predict_proba(
        np.hstack([eval_claim_scores, eval_full_aggregate])
    )
    fusion_predictions = fusion_model.classes_[np.argmax(fusion_probabilities, axis=1)]
    rule_predictions = _rule_predictions(eval_full_aggregate, feature_names)
    predictions = {
        "claim_only_tfidf_svm": np.asarray(claim_predictions, dtype=np.int64),
        "fixed_llm_stance_rule": rule_predictions,
        **{
            name: np.asarray(values, dtype=np.int64)
            for name, values in evidence_predictions.items()
        },
        "claim_evidence_late_fusion": np.asarray(fusion_predictions, dtype=np.int64),
    }
    probabilities = {
        **evidence_probabilities,
        "claim_evidence_late_fusion": fusion_probabilities,
    }
    label_names = [str(value) for value in config["labels"]]
    metrics = {
        name: _classification_metrics(eval_labels, values, label_names)
        for name, values in predictions.items()
    }
    comparisons = {
        name: _paired_bootstrap(
            eval_labels,
            predictions["claim_only_tfidf_svm"],
            values,
            samples=int(config["evaluation"]["bootstrap_samples"]),
            seed=seed,
        )
        for name, values in predictions.items()
        if name != "claim_only_tfidf_svm"
    }

    output_dir = output_dir.resolve()
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".trendfact-aggregator-", dir=output_dir.parent))
    with (staging / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for index, query in enumerate(eval_queries):
            row = {
                "query_id": query["query_id"],
                "split": query["split"],
                "gold_label_id": int(eval_labels[index]),
                "gold_label_name": label_names[int(eval_labels[index])],
                "predictions": {
                    name: {
                        "label_id": int(values[index]),
                        "label_name": label_names[int(values[index])],
                        **(
                            {
                                "probabilities": [
                                    float(value) for value in probabilities[name][index]
                                ]
                            }
                            if name in probabilities
                            else {}
                        ),
                    }
                    for name, values in predictions.items()
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    report = {
        "experiment": config["experiment"],
        "evaluation_split": evaluation_split,
        "train_queries": len(train_queries),
        "evaluation_queries": len(eval_queries),
        "train_label_distribution": {
            label_names[label]: int(np.sum(train_labels == label)) for label in range(3)
        },
        "evaluation_label_distribution": {
            label_names[label]: int(np.sum(eval_labels == label)) for label in range(3)
        },
        "aggregation_feature_names": feature_names,
        "ablation_feature_names": {
            "non_llm_structural_aggregator": non_llm_feature_names,
            "direct_qwen_no_gate_aggregator": direct_feature_names,
            "evidence_only_calibrated_aggregator": feature_names,
        },
        "metrics": metrics,
        "paired_vs_claim_only": comparisons,
        "inputs": {
            "train": {
                "queries_sha256": file_sha256(train_paths[0]),
                "bm25_sha256": file_sha256(train_paths[1]),
                "qwen_sha256": file_sha256(train_paths[2]),
            },
            evaluation_split: {
                "queries_sha256": file_sha256(evaluation_paths[0]),
                "bm25_sha256": file_sha256(evaluation_paths[1]),
                "qwen_sha256": file_sha256(evaluation_paths[2]),
            },
            "candidate_gate_model_sha256": file_sha256(gate_model_path),
        },
        "training_protocol": {
            "gate_train_features": "five_fold_query_level_out_of_fold",
            "claim_train_scores_for_fusion": "five_fold_out_of_fold",
            "non_llm_gate_train_features": "five_fold_query_level_out_of_fold",
            "direct_qwen_candidate_scores": "fixed_relevance_then_stance_no_gate",
            "evaluation_gate": "frozen_full_train_candidate_gate",
            "test_used": is_test,
        },
    }
    (staging / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    staging.replace(output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the TrendFact evidence aggregator")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--evaluation-split", choices=("pilot", "heldout_dev", "test"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-final-test", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.config,
                evaluation_split=args.evaluation_split,
                output_dir=args.output_dir,
                overwrite=args.overwrite,
                allow_test=args.allow_final_test,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

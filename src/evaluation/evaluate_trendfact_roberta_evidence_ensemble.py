from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


from evaluation.evaluate_trendfact_evidence_aggregator import (
    _classification_metrics,
    _read_jsonl,
    _resolve,
)


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def select_alpha(
    labels: np.ndarray,
    roberta_probabilities: np.ndarray,
    evidence_probabilities: np.ndarray,
    alpha_grid: Sequence[float],
    label_names: Sequence[str],
) -> tuple[float, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for alpha in alpha_grid:
        if alpha < 0 or alpha > 1:
            raise ValueError("Fusion alpha must be between 0 and 1")
        combined = (1.0 - alpha) * roberta_probabilities + alpha * evidence_probabilities
        predictions = np.argmax(combined, axis=1)
        metrics = _classification_metrics(labels, predictions, label_names)
        rows.append({"evidence_alpha": float(alpha), "metrics": metrics})
    best = max(
        rows,
        key=lambda row: (
            float(row["metrics"]["macro_f1"]),  # type: ignore[index]
            -float(row["evidence_alpha"]),
        ),
    )
    return float(best["evidence_alpha"]), rows


def _load_roberta(
    paths: Sequence[Path],
) -> tuple[list[str], np.ndarray, np.ndarray, list[np.ndarray]]:
    seed_probabilities: list[np.ndarray] = []
    reference_ids: list[str] | None = None
    reference_labels: np.ndarray | None = None
    reference_names: list[str] | None = None
    for path in paths:
        rows = _read_jsonl(path)
        identifiers = [str(row["query_id"]) for row in rows]
        labels = np.asarray([int(row["gold_label_id"]) for row in rows], dtype=np.int64)
        names = [str(row["gold_label_name"]) for row in rows]
        probabilities = softmax(
            np.asarray([row["logits"] for row in rows], dtype=np.float64)
        )
        if reference_ids is None:
            reference_ids, reference_labels, reference_names = identifiers, labels, names
        elif identifiers != reference_ids or not np.array_equal(labels, reference_labels):
            raise ValueError("RoBERTa seed prediction files do not align")
        seed_probabilities.append(probabilities)
    if reference_ids is None or reference_labels is None or reference_names is None:
        raise ValueError("At least one RoBERTa prediction file is required")
    return reference_ids, reference_labels, np.asarray(reference_names), seed_probabilities


def _load_evidence(
    path: Path, *, method: str
) -> tuple[list[str], np.ndarray, np.ndarray]:
    rows = _read_jsonl(path)
    identifiers = [str(row["query_id"]) for row in rows]
    labels = np.asarray([int(row["gold_label_id"]) for row in rows], dtype=np.int64)
    probabilities = np.asarray(
        [row["predictions"][method]["probabilities"] for row in rows],  # type: ignore[index]
        dtype=np.float64,
    )
    return identifiers, labels, probabilities


def _split_inputs(
    config: Mapping[str, object], split: str
) -> tuple[list[Path], Path]:
    values = config["input"][split]  # type: ignore[index]
    return (
        [_resolve(path) for path in values["roberta_predictions"]],  # type: ignore[index]
        _resolve(values["evidence_predictions"]),  # type: ignore[index]
    )

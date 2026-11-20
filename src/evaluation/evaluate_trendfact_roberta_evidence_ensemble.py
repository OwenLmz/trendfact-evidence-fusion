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


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evaluation.evaluate_trendfact_evidence_aggregator import (  # noqa: E402
    _classification_metrics,
    _paired_bootstrap,
    _read_jsonl,
    _resolve,
)
from retrieval.evidence import file_sha256  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/trendfact_roberta_evidence_ensemble.yaml"


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


def run(
    config_path: Path = DEFAULT_CONFIG, *, output_dir: Path, overwrite: bool = False
) -> dict[str, object]:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    label_names = [str(value) for value in config["labels"]]
    method = str(config["fusion"]["evidence_method"])
    data: dict[str, dict[str, object]] = {}
    input_hashes: dict[str, object] = {}
    for split in ("pilot", "heldout_dev"):
        roberta_paths, evidence_path = _split_inputs(config, split)
        identifiers, labels, names, seed_probabilities = _load_roberta(roberta_paths)
        evidence_ids, evidence_labels, evidence_probabilities = _load_evidence(
            evidence_path, method=method
        )
        if identifiers != evidence_ids or not np.array_equal(labels, evidence_labels):
            raise ValueError(f"RoBERTa and evidence predictions do not align for {split}")
        if any(label_names[int(label)] != name for label, name in zip(labels, names)):
            raise ValueError(f"Label names do not align for {split}")
        data[split] = {
            "ids": identifiers,
            "labels": labels,
            "seed_probabilities": seed_probabilities,
            "roberta_ensemble": np.mean(seed_probabilities, axis=0),
            "evidence": evidence_probabilities,
        }
        input_hashes[split] = {
            "roberta": [file_sha256(path) for path in roberta_paths],
            "evidence": file_sha256(evidence_path),
        }

    pilot = data["pilot"]
    alpha, alpha_selection = select_alpha(
        pilot["labels"],  # type: ignore[arg-type]
        pilot["roberta_ensemble"],  # type: ignore[arg-type]
        pilot["evidence"],  # type: ignore[arg-type]
        [float(value) for value in config["fusion"]["alpha_grid"]],
        label_names,
    )
    split_reports: dict[str, object] = {}
    combined_probabilities: dict[str, np.ndarray] = {}
    for split, values in data.items():
        labels = values["labels"]
        roberta = values["roberta_ensemble"]
        evidence = values["evidence"]
        combined = (1.0 - alpha) * roberta + alpha * evidence
        combined_probabilities[split] = combined
        roberta_predictions = np.argmax(roberta, axis=1)
        evidence_predictions = np.argmax(evidence, axis=1)
        combined_predictions = np.argmax(combined, axis=1)
        seed_metrics = [
            _classification_metrics(labels, np.argmax(probabilities, axis=1), label_names)
            for probabilities in values["seed_probabilities"]
        ]
        split_reports[split] = {
            "roberta_seeds": seed_metrics,
            "roberta_seed_macro_f1_mean": round(
                float(np.mean([row["macro_f1"] for row in seed_metrics])), 6
            ),
            "roberta_seed_macro_f1_std": round(
                float(np.std([row["macro_f1"] for row in seed_metrics], ddof=1)), 6
            ),
            "roberta_probability_ensemble": _classification_metrics(
                labels, roberta_predictions, label_names
            ),
            "evidence_only": _classification_metrics(
                labels, evidence_predictions, label_names
            ),
            "roberta_evidence_fusion": _classification_metrics(
                labels, combined_predictions, label_names
            ),
            "paired_fusion_vs_roberta_ensemble": _paired_bootstrap(
                labels,
                roberta_predictions,
                combined_predictions,
                samples=int(config["evaluation"]["bootstrap_samples"]),
                seed=int(config["seed"]),
            ),
        }

    output_dir = output_dir.resolve()
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".trendfact-ensemble-", dir=output_dir.parent))
    for split, values in data.items():
        combined = combined_probabilities[split]
        labels = values["labels"]
        with (staging / f"{split}_predictions.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            for identifier, gold, probabilities in zip(values["ids"], labels, combined):
                prediction = int(np.argmax(probabilities))
                row = {
                    "query_id": identifier,
                    "split": split,
                    "gold_label_id": int(gold),
                    "gold_label_name": label_names[int(gold)],
                    "prediction_label_id": prediction,
                    "prediction_label_name": label_names[prediction],
                    "probabilities": [float(value) for value in probabilities],
                    "evidence_alpha": alpha,
                }
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    report = {
        "experiment": config["experiment"],
        "selected_evidence_alpha": alpha,
        "alpha_selected_on": "frozen_pilot_dev_200",
        "alpha_selection": alpha_selection,
        "results": split_reports,
        "input_hashes": input_hashes,
        "heldout_used_for_parameter_selection": False,
        "test_used": False,
    }
    (staging / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    staging.replace(output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse TrendFact RoBERTa and evidence outputs")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.config, output_dir=args.output_dir, overwrite=args.overwrite),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
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
)
from evaluation.evaluate_trendfact_roberta_evidence_ensemble import (  # noqa: E402
    _load_evidence,
    _load_roberta,
)
from retrieval.evidence import file_sha256  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/final_test_trendfact_ablations.yaml"


def validate_preselected_alphas(
    report: Mapping[str, object],
    methods: Sequence[str],
    expected: Mapping[str, float],
) -> dict[str, float]:
    raw = report.get("selected_evidence_alphas")
    if not isinstance(raw, Mapping):
        raise ValueError("Dev selection report has no selected_evidence_alphas")
    selected = {str(key): float(value) for key, value in raw.items()}
    if set(selected) != set(methods):
        raise ValueError("Dev-selected methods do not match frozen test methods")
    frozen = {str(key): float(value) for key, value in expected.items()}
    if selected != frozen:
        raise ValueError("Dev-selected alphas do not match frozen expected alphas")
    return selected


def fuse_probabilities(
    roberta: np.ndarray, evidence: np.ndarray, *, alpha: float
) -> np.ndarray:
    if roberta.shape != evidence.shape:
        raise ValueError("RoBERTa and evidence probability shapes do not match")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    return (1.0 - alpha) * roberta + alpha * evidence


def run(config_path: Path, *, output_dir: Path) -> dict[str, object]:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    label_names = [str(value) for value in config["labels"]]
    methods = [str(value) for value in config["fusion"]["evidence_methods"]]
    primary_method = str(config["fusion"]["primary_evidence_method"])
    if primary_method not in methods or len(methods) != len(set(methods)):
        raise ValueError("Frozen evidence methods are invalid")

    selection_path = (ROOT / config["input"]["dev_selection_report"]).resolve()
    expected_selection_hash = str(
        config["input"]["dev_selection_report_sha256"]
    ).lower()
    if file_sha256(selection_path) != expected_selection_hash:
        raise ValueError("Frozen dev selection report SHA-256 mismatch")
    selection_report = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_alphas = validate_preselected_alphas(
        selection_report,
        methods,
        config["fusion"]["expected_preselected_alphas"],
    )

    roberta_paths = [
        (ROOT / value).resolve() for value in config["input"]["roberta_predictions"]
    ]
    evidence_path = (ROOT / config["input"]["evidence_predictions"]).resolve()
    identifiers, labels, names, seed_probabilities = _load_roberta(roberta_paths)
    if len(labels) != int(config["evaluation"]["samples"]):
        raise ValueError("Frozen test row count mismatch")
    if any(label_names[int(label)] != name for label, name in zip(labels, names)):
        raise ValueError("Frozen test label names do not align")

    roberta_probabilities = np.mean(seed_probabilities, axis=0)
    roberta_predictions = np.argmax(roberta_probabilities, axis=1)
    evidence_probabilities: dict[str, np.ndarray] = {}
    combined_probabilities: dict[str, np.ndarray] = {}
    method_reports: dict[str, object] = {}
    bootstrap_samples = int(config["evaluation"]["bootstrap_samples"])
    seed = int(config["seed"])
    for method in methods:
        evidence_ids, evidence_labels, probabilities = _load_evidence(
            evidence_path, method=method
        )
        if identifiers != evidence_ids or not np.array_equal(labels, evidence_labels):
            raise ValueError(f"Frozen {method} test predictions do not align")
        combined = fuse_probabilities(
            roberta_probabilities,
            probabilities,
            alpha=selected_alphas[method],
        )
        evidence_predictions = np.argmax(probabilities, axis=1)
        combined_predictions = np.argmax(combined, axis=1)
        evidence_probabilities[method] = probabilities
        combined_probabilities[method] = combined
        method_reports[method] = {
            "evidence_alpha": selected_alphas[method],
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
                samples=bootstrap_samples,
                seed=seed,
            ),
        }

    primary_predictions = np.argmax(combined_probabilities[primary_method], axis=1)
    paired_primary_vs_ablations = {
        method: _paired_bootstrap(
            labels,
            np.argmax(combined_probabilities[method], axis=1),
            primary_predictions,
            samples=bootstrap_samples,
            seed=seed,
        )
        for method in methods
        if method != primary_method
    }

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Frozen final ablation output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".trendfact-final-ablation-", dir=output_dir.parent))
    prediction_path = staging / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, identifier in enumerate(identifiers):
            row: dict[str, object] = {
                "query_id": identifier,
                "split": "test",
                "gold_label_id": int(labels[index]),
                "gold_label_name": label_names[int(labels[index])],
                "methods": {},
            }
            for method in methods:
                probabilities = combined_probabilities[method][index]
                prediction = int(np.argmax(probabilities))
                row["methods"][method] = {  # type: ignore[index]
                    "label_id": prediction,
                    "label_name": label_names[prediction],
                    "probabilities": [float(value) for value in probabilities],
                    "evidence_alpha": selected_alphas[method],
                }
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    report = {
        "experiment": config["experiment"],
        "freeze_config_sha256": file_sha256(config_path),
        "dev_selection_report_sha256": expected_selection_hash,
        "methods": methods,
        "primary_evidence_method": primary_method,
        "selected_evidence_alphas": selected_alphas,
        "alpha_selected_on": config["protocol"]["alphas_selected_on"],
        "roberta_probability_ensemble": _classification_metrics(
            labels, roberta_predictions, label_names
        ),
        "method_results": method_reports,
        "paired_primary_vs_ablations": paired_primary_vs_ablations,
        "inputs": {
            "roberta": [file_sha256(path) for path in roberta_paths],
            "evidence": file_sha256(evidence_path),
        },
        "predictions_sha256": file_sha256(prediction_path),
        "post_main_preselected_ablation_analysis": True,
        "test_used": True,
        "test_predictions_regenerated": False,
        "parameters_selected_on_test": False,
    }
    (staging / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    staging.replace(output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate preselected TrendFact component ablations on final test predictions"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.config, output_dir=args.output_dir),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

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


DEFAULT_CONFIG = ROOT / "configs/final_test_trendfact.yaml"


def run(
    config_path: Path,
    *,
    roberta_paths: list[Path],
    evidence_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    label_names = [str(value) for value in config["dataset"]["labels"]]
    alpha = float(config["decision_fusion"]["evidence_alpha"])
    method = str(config["evidence_aggregator"]["method"])
    identifiers, labels, names, seed_probabilities = _load_roberta(
        [path.resolve() for path in roberta_paths]
    )
    evidence_ids, evidence_labels, evidence_probabilities = _load_evidence(
        evidence_path.resolve(), method=method
    )
    if identifiers != evidence_ids or not np.array_equal(labels, evidence_labels):
        raise ValueError("Frozen RoBERTa and evidence test predictions do not align")
    if any(label_names[int(label)] != name for label, name in zip(labels, names)):
        raise ValueError("Frozen test label names do not align")
    if len(labels) != int(config["dataset"]["test_rows"]):
        raise ValueError("Frozen final-fusion row count mismatch")

    roberta_probabilities = np.mean(seed_probabilities, axis=0)
    final_probabilities = (
        (1.0 - alpha) * roberta_probabilities + alpha * evidence_probabilities
    )
    roberta_predictions = np.argmax(roberta_probabilities, axis=1)
    evidence_predictions = np.argmax(evidence_probabilities, axis=1)
    final_predictions = np.argmax(final_probabilities, axis=1)

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Frozen final-fusion output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".trendfact-final-fusion-", dir=output_dir.parent))
    prediction_path = staging / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, identifier in enumerate(identifiers):
            prediction = int(final_predictions[index])
            row = {
                "query_id": identifier,
                "split": "test",
                "gold_label_id": int(labels[index]),
                "gold_label_name": label_names[int(labels[index])],
                "prediction_label_id": prediction,
                "prediction_label_name": label_names[prediction],
                "probabilities": [float(value) for value in final_probabilities[index]],
                "evidence_alpha": alpha,
            }
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    report = {
        "experiment": config["experiment"],
        "freeze_config_sha256": file_sha256(config_path),
        "evidence_alpha": alpha,
        "alpha_selected_on": config["decision_fusion"]["selected_on"],
        "metrics": {
            "roberta_seeds": [
                _classification_metrics(labels, np.argmax(values, axis=1), label_names)
                for values in seed_probabilities
            ],
            "roberta_probability_ensemble": _classification_metrics(
                labels, roberta_predictions, label_names
            ),
            "evidence_only": _classification_metrics(
                labels, evidence_predictions, label_names
            ),
            "roberta_evidence_fusion": _classification_metrics(
                labels, final_predictions, label_names
            ),
        },
        "paired_fusion_vs_roberta_ensemble": _paired_bootstrap(
            labels,
            roberta_predictions,
            final_predictions,
            samples=2000,
            seed=20260730,
        ),
        "inputs": {
            "roberta": [file_sha256(path.resolve()) for path in roberta_paths],
            "evidence": file_sha256(evidence_path.resolve()),
        },
        "predictions_sha256": file_sha256(prediction_path),
        "test_used": True,
        "parameters_selected_on_test": False,
    }
    (staging / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    staging.replace(output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen TrendFact final fusion")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--roberta-path", type=Path, action="append", required=True)
    parser.add_argument("--evidence-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.config,
                roberta_paths=args.roberta_path,
                evidence_path=args.evidence_path,
                output_dir=args.output_dir,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

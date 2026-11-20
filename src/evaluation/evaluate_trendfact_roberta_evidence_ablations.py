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
from evaluation.evaluate_trendfact_roberta_evidence_ensemble import (  # noqa: E402
    _load_evidence,
    _load_roberta,
    _split_inputs,
    select_alpha,
)
from retrieval.evidence import file_sha256  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/trendfact_roberta_evidence_ablations.yaml"


def evaluate_fusion_method(
    labels: np.ndarray,
    roberta_probabilities: np.ndarray,
    evidence_probabilities: np.ndarray,
    *,
    alpha: float,
    label_names: Sequence[str],
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, object], np.ndarray]:
    combined = (1.0 - alpha) * roberta_probabilities + alpha * evidence_probabilities
    roberta_predictions = np.argmax(roberta_probabilities, axis=1)
    evidence_predictions = np.argmax(evidence_probabilities, axis=1)
    combined_predictions = np.argmax(combined, axis=1)
    return (
        {
            "evidence_alpha": alpha,
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
        },
        combined,
    )


def run(
    config_path: Path = DEFAULT_CONFIG, *, output_dir: Path, overwrite: bool = False
) -> dict[str, object]:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    label_names = [str(value) for value in config["labels"]]
    methods = [str(value) for value in config["fusion"]["evidence_methods"]]
    if len(methods) != len(set(methods)) or not methods:
        raise ValueError("evidence_methods must be non-empty and unique")
    primary_method = str(config["fusion"]["primary_evidence_method"])
    if primary_method not in methods:
        raise ValueError("primary_evidence_method must be listed in evidence_methods")

    data: dict[str, dict[str, object]] = {}
    input_hashes: dict[str, object] = {}
    for split in ("pilot", "heldout_dev"):
        roberta_paths, evidence_path = _split_inputs(config, split)
        identifiers, labels, names, seed_probabilities = _load_roberta(roberta_paths)
        evidence_by_method: dict[str, np.ndarray] = {}
        for method in methods:
            evidence_ids, evidence_labels, probabilities = _load_evidence(
                evidence_path, method=method
            )
            if identifiers != evidence_ids or not np.array_equal(labels, evidence_labels):
                raise ValueError(
                    f"RoBERTa and {method} predictions do not align for {split}"
                )
            evidence_by_method[method] = probabilities
        if any(label_names[int(label)] != name for label, name in zip(labels, names)):
            raise ValueError(f"Label names do not align for {split}")
        data[split] = {
            "ids": identifiers,
            "labels": labels,
            "roberta_ensemble": np.mean(seed_probabilities, axis=0),
            "evidence": evidence_by_method,
        }
        input_hashes[split] = {
            "roberta": [file_sha256(path) for path in roberta_paths],
            "evidence": file_sha256(evidence_path),
        }

    pilot = data["pilot"]
    alpha_grid = [float(value) for value in config["fusion"]["alpha_grid"]]
    selected_alphas: dict[str, float] = {}
    alpha_selection: dict[str, list[dict[str, object]]] = {}
    for method in methods:
        alpha, rows = select_alpha(
            pilot["labels"],  # type: ignore[arg-type]
            pilot["roberta_ensemble"],  # type: ignore[arg-type]
            pilot["evidence"][method],  # type: ignore[index]
            alpha_grid,
            label_names,
        )
        selected_alphas[method] = alpha
        alpha_selection[method] = rows

    bootstrap_samples = int(config["evaluation"]["bootstrap_samples"])
    seed = int(config["seed"])
    split_reports: dict[str, object] = {}
    combined_by_split: dict[str, dict[str, np.ndarray]] = {}
    for split, values in data.items():
        labels = values["labels"]
        roberta = values["roberta_ensemble"]
        roberta_predictions = np.argmax(roberta, axis=1)
        method_reports: dict[str, object] = {}
        method_probabilities: dict[str, np.ndarray] = {}
        for method in methods:
            report, combined = evaluate_fusion_method(
                labels,  # type: ignore[arg-type]
                roberta,  # type: ignore[arg-type]
                values["evidence"][method],  # type: ignore[index]
                alpha=selected_alphas[method],
                label_names=label_names,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
            method_reports[method] = report
            method_probabilities[method] = combined
        primary_predictions = np.argmax(method_probabilities[primary_method], axis=1)
        paired_primary_vs_ablations = {
            method: _paired_bootstrap(
                labels,  # type: ignore[arg-type]
                np.argmax(method_probabilities[method], axis=1),
                primary_predictions,
                samples=bootstrap_samples,
                seed=seed,
            )
            for method in methods
            if method != primary_method
        }
        split_reports[split] = {
            "roberta_probability_ensemble": _classification_metrics(
                labels, roberta_predictions, label_names  # type: ignore[arg-type]
            ),
            "methods": method_reports,
            "paired_primary_vs_ablations": paired_primary_vs_ablations,
        }
        combined_by_split[split] = method_probabilities

    output_dir = output_dir.resolve()
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".trendfact-ablation-", dir=output_dir.parent))
    for split, values in data.items():
        with (staging / f"{split}_predictions.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            for index, identifier in enumerate(values["ids"]):
                gold = int(values["labels"][index])  # type: ignore[index]
                row = {
                    "query_id": identifier,
                    "split": split,
                    "gold_label_id": gold,
                    "gold_label_name": label_names[gold],
                    "methods": {},
                }
                for method in methods:
                    probabilities = combined_by_split[split][method][index]
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
        "methods": methods,
        "primary_evidence_method": primary_method,
        "selected_evidence_alphas": selected_alphas,
        "alpha_selected_on": "frozen_pilot_dev_200",
        "alpha_selection": alpha_selection,
        "results": split_reports,
        "input_hashes": input_hashes,
        "post_hoc_after_component_heldout_results": True,
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
    parser = argparse.ArgumentParser(
        description="Ablate TrendFact RoBERTa evidence decision fusion"
    )
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

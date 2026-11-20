from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evaluation.evaluate_trendfact_evidence_aggregator import (  # noqa: E402
    _classification_metrics,
    _read_jsonl,
    _resolve,
)
from evaluation.evaluate_trendfact_roberta_evidence_ensemble import (  # noqa: E402
    _load_roberta,
)
from retrieval.evidence import file_sha256  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/trendfact_dev_error_analysis.yaml"


def outcome_counts(
    labels: np.ndarray, baseline: np.ndarray, method: np.ndarray
) -> dict[str, int]:
    baseline_correct = baseline == labels
    method_correct = method == labels
    return {
        "corrected": int(np.sum(~baseline_correct & method_correct)),
        "regressed": int(np.sum(baseline_correct & ~method_correct)),
        "unchanged_correct": int(np.sum(baseline_correct & method_correct)),
        "unchanged_wrong": int(np.sum(~baseline_correct & ~method_correct)),
    }


def claim_length_bin(length: int) -> str:
    if length <= 20:
        return "01_0_20"
    if length <= 40:
        return "02_21_40"
    if length <= 60:
        return "03_41_60"
    return "04_61_plus"


def gold_count_bin(count: int) -> str:
    return str(count) if count <= 3 else "4_plus"


def _group_report(
    values: Sequence[str], labels: np.ndarray, baseline: np.ndarray, method: np.ndarray
) -> dict[str, object]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(values):
        groups[value].append(index)
    report: dict[str, object] = {}
    for name, indices in sorted(groups.items()):
        selected = np.asarray(indices, dtype=np.int64)
        report[name] = {
            "samples": len(indices),
            "roberta_accuracy": round(
                float(np.mean(baseline[selected] == labels[selected])), 6
            ),
            "fusion_accuracy": round(
                float(np.mean(method[selected] == labels[selected])), 6
            ),
            **outcome_counts(labels[selected], baseline[selected], method[selected]),
        }
    return report


def run(
    config_path: Path = DEFAULT_CONFIG,
    *,
    output_dir: Path,
    overwrite: bool = False,
    allow_test: bool = False,
) -> dict[str, object]:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    label_names = [str(value) for value in config["labels"]]
    query_path = _resolve(config["input"]["queries"])
    roberta_paths = [
        _resolve(value) for value in config["input"]["roberta_predictions"]
    ]
    fusion_path = _resolve(config["input"]["fusion_predictions"])
    primary_method = str(config["input"]["primary_method"])

    queries = _read_jsonl(query_path)
    query_ids = [str(row["query_id"]) for row in queries]
    splits = {str(row["split"]) for row in queries}
    if len(splits) != 1:
        raise ValueError("Error analysis requires exactly one split")
    is_test = splits == {"test"}
    if is_test and not allow_test:
        raise ValueError("Test error analysis requires --allow-final-test")
    if is_test and overwrite:
        raise ValueError("Final-test error analysis may not be overwritten")
    roberta_ids, labels, _, seed_probabilities = _load_roberta(roberta_paths)
    fusion_rows = _read_jsonl(fusion_path)
    fusion_ids = [str(row["query_id"]) for row in fusion_rows]
    if query_ids != roberta_ids or query_ids != fusion_ids:
        raise ValueError("Query, RoBERTa, and fusion rows do not align")
    query_labels = np.asarray([int(row["label_id"]) for row in queries], dtype=np.int64)
    if not np.array_equal(labels, query_labels):
        raise ValueError("Query labels do not align with predictions")

    roberta_probabilities = np.mean(seed_probabilities, axis=0)
    roberta_predictions = np.argmax(roberta_probabilities, axis=1)
    prediction_format = str(config["input"].get("fusion_prediction_format", "methods"))
    if prediction_format == "methods":
        fusion_probabilities = np.asarray(
            [
                row["methods"][primary_method]["probabilities"]  # type: ignore[index]
                for row in fusion_rows
            ],
            dtype=np.float64,
        )
    elif prediction_format == "final":
        fusion_probabilities = np.asarray(
            [row["probabilities"] for row in fusion_rows], dtype=np.float64
        )
    else:
        raise ValueError(f"Unknown fusion_prediction_format: {prediction_format}")
    fusion_predictions = np.argmax(fusion_probabilities, axis=1)

    report = {
        "experiment": config["experiment"],
        "samples": len(queries),
        "primary_method": primary_method,
        "metrics": {
            "roberta_probability_ensemble": _classification_metrics(
                labels, roberta_predictions, label_names
            ),
            "roberta_evidence_fusion": _classification_metrics(
                labels, fusion_predictions, label_names
            ),
        },
        "confusion_matrices": {
            "label_order": label_names,
            "roberta_probability_ensemble": confusion_matrix(
                labels, roberta_predictions, labels=np.arange(len(label_names))
            ).tolist(),
            "roberta_evidence_fusion": confusion_matrix(
                labels, fusion_predictions, labels=np.arange(len(label_names))
            ).tolist(),
        },
        "paired_outcomes": outcome_counts(
            labels, roberta_predictions, fusion_predictions
        ),
        "by_label": _group_report(
            [str(row["label_name"]) for row in queries],
            labels,
            roberta_predictions,
            fusion_predictions,
        ),
        "by_domain": _group_report(
            [str(row.get("domain", "unknown")) for row in queries],
            labels,
            roberta_predictions,
            fusion_predictions,
        ),
        "by_claim_length": _group_report(
            [claim_length_bin(len(str(row["query_text"]))) for row in queries],
            labels,
            roberta_predictions,
            fusion_predictions,
        ),
        "by_gold_evidence_count": _group_report(
            [gold_count_bin(len(row["gold_target_indices"])) for row in queries],  # type: ignore[arg-type]
            labels,
            roberta_predictions,
            fusion_predictions,
        ),
        "input_hashes": {
            "queries": file_sha256(query_path),
            "roberta": [file_sha256(path) for path in roberta_paths],
            "fusion": file_sha256(fusion_path),
        },
        "descriptive_analysis_only": True,
        "heldout_used_for_parameter_selection": False,
        "test_used": is_test,
    }

    output_dir = output_dir.resolve()
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".trendfact-errors-", dir=output_dir.parent))
    cases: list[dict[str, object]] = []
    for index, query in enumerate(queries):
        baseline_correct = bool(roberta_predictions[index] == labels[index])
        method_correct = bool(fusion_predictions[index] == labels[index])
        if baseline_correct == method_correct:
            continue
        cases.append(
            {
                "query_id": query["query_id"],
                "query_text": query["query_text"],
                "domain": query.get("domain"),
                "gold_label": label_names[int(labels[index])],
                "outcome": "corrected" if method_correct else "regressed",
                "roberta_prediction": label_names[int(roberta_predictions[index])],
                "fusion_prediction": label_names[int(fusion_predictions[index])],
                "roberta_probabilities": [
                    float(value) for value in roberta_probabilities[index]
                ],
                "fusion_probabilities": [
                    float(value) for value in fusion_probabilities[index]
                ],
                "gold_evidence_count": len(query["gold_target_indices"]),  # type: ignore[arg-type]
            }
        )
    cases.sort(key=lambda row: (str(row["outcome"]), str(row["query_id"])))
    with (staging / "changed_cases.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in cases:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    (staging / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    staging.replace(output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze frozen TrendFact dev predictions")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-final-test", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.config,
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

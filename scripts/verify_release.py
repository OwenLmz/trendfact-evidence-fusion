"""Check the frozen release and optional local replay inputs."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as handle:
        sha = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def verify(relative_path, sha, size=None):
    path = ROOT / relative_path
    if not path.is_file() or (size is not None and path.stat().st_size != size):
        raise ValueError(f"Missing or unexpected file size: {relative_path}")
    if digest(path) != sha:
        raise ValueError(f"SHA-256 mismatch: {relative_path}")
    print(f"OK {relative_path}")


def verify_metrics(results, labels):
    matrices = json.loads((ROOT / "results/confusion_matrices.json").read_text(encoding="utf-8"))
    if matrices["labels"] != labels:
        raise ValueError("Confusion matrix label order differs from the frozen configuration")
    for name, matrix in matrices["matrices"].items():
        totals = [sum(row) for row in matrix]
        correct = [matrix[index][index] for index in range(len(labels))]
        predicted = [sum(column) for column in zip(*matrix)]
        f1 = [2 * tp / (gold + pred) for tp, gold, pred in zip(correct, totals, predicted)]
        accuracy = sum(correct) / sum(totals)
        macro_f1 = sum(f1) / len(labels)
        expected = results["metrics"][name]
        if sum(totals) != expected["samples"] or any(
            abs(computed - expected[field]) > 0.000001
            for field, computed in (("accuracy", accuracy), ("macro_f1", macro_f1))
        ):
            raise ValueError(f"Aggregate metrics mismatch: {name}")
        print(f"OK {name}: accuracy={accuracy:.6f}, macro_f1={macro_f1:.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", action="store_true", help="Check both upstream dataset files")
    parser.add_argument("--checkpoints", action="store_true", help="Check original RoBERTa weights")
    parser.add_argument("--splits", action="store_true", help="Check all six derived dataset files")
    parser.add_argument("--training-cache", action="store_true", help="Check original evidence training inputs")
    parser.add_argument("--rebuilt", action="store_true", help="Check locally rebuilt weights and cache presence")
    args = parser.parse_args()
    frozen_path = ROOT / "configs/final_test_trendfact.yaml"
    config = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    results = json.loads((ROOT / "results/frozen_test.json").read_text(encoding="utf-8"))
    verify(frozen_path.relative_to(ROOT), results["freeze_config_sha256"])
    for entry in config["local_artifacts"]:
        if entry["path"].startswith("configs/"):
            verify(entry["path"], entry["sha256"])
    verify("assets/candidate_gate.json", config["evidence"]["candidate_gate"]["model_sha256"])
    ablations = yaml.safe_load(
        (ROOT / "configs/final_test_trendfact_ablations.yaml").read_text(encoding="utf-8")
    )
    verify("assets/dev_ablation_selection.json", ablations["input"]["dev_selection_report_sha256"])
    inputs = json.loads((ROOT / "assets/frozen_inputs.json").read_text(encoding="utf-8"))
    for path, sha in inputs["scoring_source"].items():
        verify(path, sha)
    verify_metrics(results, config["dataset"]["labels"])
    if args.raw:
        upstream = yaml.safe_load((ROOT / "configs/data_trendfact.yaml").read_text(encoding="utf-8"))["input"]
        for key in ("dataset_json", "evidence_parquet"):
            verify(upstream[key], upstream[f"{key}_sha256"])
    if args.checkpoints:
        for entry in config["claim_model"]["checkpoints"]:
            verify(entry["path"], entry["sha256"], entry["bytes"])
    if args.splits:
        for path, sha in inputs["derived_data"].items():
            verify(path, sha)
    if args.training_cache:
        for path, sha in inputs["evidence_training"].items():
            verify(path, sha)
    if args.rebuilt:
        sys.path.insert(0, str(ROOT / "src"))
        from baselines.predict_trendfact_frozen_roberta import validate_checkpoint

        for path in inputs["evidence_training"]:
            if not (ROOT / path).is_file():
                raise FileNotFoundError(f"Missing rebuilt training input: {path}")
        for entry in config["claim_model"]["checkpoints"]:
            validate_checkpoint(config, int(entry["seed"]), ROOT / entry["path"], retrained=True)
            print(f"OK retrained {entry['path']}")


if __name__ == "__main__":
    main()

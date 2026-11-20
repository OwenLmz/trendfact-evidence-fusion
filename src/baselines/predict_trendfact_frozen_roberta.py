from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml


os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from baselines.trendfact_transformer_baseline import _texts_labels_ids  # noqa: E402
from evaluation.evaluate_trendfact_evidence_aggregator import (  # noqa: E402
    _classification_metrics,
    _read_jsonl,
    _resolve,
)
from retrieval.evidence import file_sha256  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/final_test_trendfact.yaml"


def select_checkpoint(
    checkpoints: Sequence[Mapping[str, object]], seed: int
) -> Mapping[str, object]:
    selected = [row for row in checkpoints if int(row["seed"]) == seed]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one frozen checkpoint for seed {seed}")
    return selected[0]


def validate_checkpoint(
    config: Mapping[str, object], seed: int, path: Path, *, retrained: bool
) -> tuple[int, str]:
    model = config["claim_model"]
    checkpoint = select_checkpoint(model["checkpoints"], seed)
    checkpoint_sha = file_sha256(path)
    if not retrained:
        if path.stat().st_size != int(checkpoint["bytes"]) or checkpoint_sha != checkpoint["sha256"]:
            raise ValueError("Frozen checkpoint size or SHA-256 mismatch")
        return int(checkpoint["best_epoch"]), checkpoint_sha

    report = json.loads(path.with_name("report.json").read_text(encoding="utf-8"))
    training_config = yaml.safe_load(
        (ROOT / f"configs/trendfact_claim_roberta_seed{seed}.yaml").read_text(encoding="utf-8")
    )
    expected_inputs = {
        "train_sha256": file_sha256(_resolve(training_config["input"]["train"])),
        "pilot_sha256": file_sha256(_resolve(training_config["input"]["pilot"])),
        "heldout_dev_sha256": file_sha256(_resolve(training_config["input"]["heldout_dev"])),
    }
    if (
        report["seed"] != seed
        or report["model"] != training_config["model"]
        or report["training"] != training_config["training"]
        or report["inputs"] != expected_inputs
        or report["checkpoint_sha256"] != checkpoint_sha
        or report["selection_split"] != "frozen_pilot_dev_200"
        or report["heldout_used_for_parameter_selection"] is not False
        or report["test_used"] is not False
    ):
        raise ValueError("Retrained checkpoint does not match the documented training run")
    return int(report["best_epoch"]), checkpoint_sha


def run(
    config_path: Path,
    *,
    seed: int,
    queries_path: Path,
    output_dir: Path,
    device_index: int = 0,
    retrained: bool = False,
) -> dict[str, object]:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["status"] != "frozen_not_run":
        raise ValueError("Final-test config is not in the frozen state")
    checkpoint = select_checkpoint(config["claim_model"]["checkpoints"], seed)
    checkpoint_path = _resolve(checkpoint["path"])
    best_epoch, checkpoint_sha = validate_checkpoint(config, seed, checkpoint_path, retrained=retrained)
    rows = _read_jsonl(queries_path.resolve())
    if {str(row["split"]) for row in rows} != {"test"}:
        raise ValueError("Frozen RoBERTa inference requires test queries only")
    if len(rows) != int(config["dataset"]["test_rows"]):
        raise ValueError("Frozen RoBERTa test query count mismatch")
    texts, labels, identifiers = _texts_labels_ids(rows, text_field="query_text")

    model_config = config["claim_model"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_config["name"]), revision=str(model_config["revision"])
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_config["name"]),
        revision=str(model_config["revision"]),
        num_labels=3,
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    device = torch.device(f"cuda:{device_index}")
    model.to(device).eval()

    class EncodedDataset(Dataset):
        def __init__(self, values: Sequence[str]) -> None:
            self.encodings = tokenizer(
                list(values),
                max_length=int(model_config["max_length"]),
                truncation=True,
                padding="max_length",
            )

        def __len__(self) -> int:
            return len(texts)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            return {
                key: torch.tensor(value[index], dtype=torch.long)
                for key, value in self.encodings.items()
            }

    loader = DataLoader(EncodedDataset(texts), batch_size=128, shuffle=False)
    logits: list[list[float]] = []
    with torch.inference_mode():
        for batch in loader:
            output = model(**{key: value.to(device) for key, value in batch.items()})
            logits.extend(output.logits.float().cpu().tolist())
    score_matrix = np.asarray(logits, dtype=np.float64)
    predictions = np.argmax(score_matrix, axis=1)
    label_array = np.asarray(labels, dtype=np.int64)
    label_names = [str(value) for value in config["dataset"]["labels"]]

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Frozen RoBERTa output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".trendfact-final-roberta-", dir=output_dir.parent))
    prediction_path = staging / "test_predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8", newline="\n") as handle:
        for identifier, gold, prediction, scores in zip(
            identifiers, label_array, predictions, score_matrix
        ):
            row = {
                "query_id": identifier,
                "split": "test",
                "gold_label_id": int(gold),
                "gold_label_name": label_names[int(gold)],
                "prediction_label_id": int(prediction),
                "prediction_label_name": label_names[int(prediction)],
                "logits": [float(value) for value in scores],
            }
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    report = {
        "experiment": config["experiment"],
        "seed": seed,
        "best_epoch": best_epoch,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_source": "retrained" if retrained else "original_frozen",
        "queries_sha256": file_sha256(queries_path.resolve()),
        "predictions_sha256": file_sha256(prediction_path),
        "metrics": _classification_metrics(label_array, predictions, label_names),
        "test_used": True,
        "parameters_selected_on_test": False,
    }
    (staging / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    staging.replace(output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a frozen TrendFact RoBERTa checkpoint")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--queries-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--retrained", action="store_true", help="Use a verified locally retrained checkpoint")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.config,
                seed=args.seed,
                queries_path=args.queries_path,
                output_dir=args.output_dir,
                device_index=args.device,
                retrained=args.retrained,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import random
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

from evaluation.evaluate_trendfact_evidence_aggregator import (  # noqa: E402
    _classification_metrics,
    _read_jsonl,
    _resolve,
)
from retrieval.evidence import file_sha256  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/trendfact_claim_roberta_seed13.yaml"


def balanced_class_weights(labels: Sequence[int], classes: int = 3) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(values, minlength=classes)
    if len(counts) != classes or np.any(counts == 0):
        raise ValueError(f"All {classes} labels must be present, got counts={counts.tolist()}")
    return len(values) / (classes * counts.astype(np.float64))


def _texts_labels_ids(
    rows: Sequence[Mapping[str, object]], *, text_field: str
) -> tuple[list[str], list[int], list[str]]:
    texts = [str(row[text_field]).strip() for row in rows]
    labels = [int(row["label_id"]) for row in rows]
    identifiers = [str(row.get("query_id", row.get("id", ""))) for row in rows]
    if any(not text for text in texts):
        raise ValueError(f"Empty text in field {text_field}")
    if any(not identifier for identifier in identifiers):
        raise ValueError("Every row must have a stable id or query_id")
    return texts, labels, identifiers


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run(
    config_path: Path = DEFAULT_CONFIG, *, output_dir: Path, overwrite: bool = False
) -> dict[str, object]:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers.optimization import get_linear_schedule_with_warmup

    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    _set_seed(seed)
    train_path = _resolve(config["input"]["train"])
    pilot_path = _resolve(config["input"]["pilot"])
    heldout_path = _resolve(config["input"]["heldout_dev"])
    train_rows = _read_jsonl(train_path)
    pilot_rows = _read_jsonl(pilot_path)
    heldout_rows = _read_jsonl(heldout_path)
    if {str(row["split"]) for row in train_rows} != {"train"}:
        raise ValueError("Transformer baseline training input must contain train rows only")
    if "test" in {str(row["split"]) for row in pilot_rows + heldout_rows}:
        raise ValueError("Test rows cannot be used by the Transformer baseline")
    train_texts, train_labels, train_ids = _texts_labels_ids(train_rows, text_field="claim")
    pilot_texts, pilot_labels, pilot_ids = _texts_labels_ids(
        pilot_rows, text_field="query_text"
    )
    heldout_texts, heldout_labels, heldout_ids = _texts_labels_ids(
        heldout_rows, text_field="query_text"
    )

    model_config = config["model"]
    training = config["training"]
    device = torch.device(f"cuda:{int(model_config['device'])}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_config["name"]), revision=str(model_config["revision"])
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_config["name"]),
        revision=str(model_config["revision"]),
        num_labels=3,
    ).to(device)

    class EncodedDataset(Dataset):
        def __init__(self, texts: Sequence[str], labels: Sequence[int]) -> None:
            self.encodings = tokenizer(
                list(texts),
                max_length=int(model_config["max_length"]),
                truncation=True,
                padding="max_length",
            )
            self.labels = list(labels)

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            row = {
                key: torch.tensor(value[index], dtype=torch.long)
                for key, value in self.encodings.items()
            }
            row["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
            return row

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        EncodedDataset(train_texts, train_labels),
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    pilot_loader = DataLoader(
        EncodedDataset(pilot_texts, pilot_labels),
        batch_size=int(training["eval_batch_size"]),
        shuffle=False,
    )
    heldout_loader = DataLoader(
        EncodedDataset(heldout_texts, heldout_labels),
        batch_size=int(training["eval_batch_size"]),
        shuffle=False,
    )

    weights = balanced_class_weights(train_labels)
    loss_function = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    epochs = int(training["epochs"])
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(training["warmup_ratio"])),
        num_training_steps=total_steps,
    )
    fp16 = bool(training["fp16"])
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)

    @torch.no_grad()
    def predict(loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        model.eval()
        labels: list[int] = []
        predictions: list[int] = []
        logits: list[list[float]] = []
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            batch_labels = batch.pop("labels")
            output = model(**batch).logits.float()
            labels.extend(batch_labels.cpu().tolist())
            predictions.extend(torch.argmax(output, dim=1).cpu().tolist())
            logits.extend(output.cpu().tolist())
        return (
            np.asarray(labels, dtype=np.int64),
            np.asarray(predictions, dtype=np.int64),
            np.asarray(logits, dtype=np.float64),
        )

    label_names = [str(value) for value in config["labels"]]
    history: list[dict[str, object]] = []
    best_epoch = 0
    best_macro_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=fp16):
                logits = model(**batch).logits
                loss = loss_function(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += float(loss.detach().cpu())
        pilot_gold, pilot_predictions, _ = predict(pilot_loader)
        pilot_metrics = _classification_metrics(pilot_gold, pilot_predictions, label_names)
        epoch_row = {
            "epoch": epoch,
            "train_loss": round(total_loss / max(len(train_loader), 1), 6),
            "pilot": pilot_metrics,
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row, ensure_ascii=False), flush=True)
        macro_f1 = float(pilot_metrics["macro_f1"])
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("No Transformer checkpoint was selected")
    model.load_state_dict(best_state)
    pilot_gold, pilot_predictions, pilot_logits = predict(pilot_loader)
    heldout_gold, heldout_predictions, heldout_logits = predict(heldout_loader)
    metrics = {
        "pilot": _classification_metrics(pilot_gold, pilot_predictions, label_names),
        "heldout_dev": _classification_metrics(
            heldout_gold, heldout_predictions, label_names
        ),
    }

    output_dir = output_dir.resolve()
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".trendfact-roberta-", dir=output_dir.parent))
    torch.save(best_state, staging / "best_state.pt")
    for split, ids, gold, predictions, logits in (
        ("pilot", pilot_ids, pilot_gold, pilot_predictions, pilot_logits),
        ("heldout_dev", heldout_ids, heldout_gold, heldout_predictions, heldout_logits),
    ):
        with (staging / f"{split}_predictions.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            for identifier, true_label, prediction, scores in zip(
                ids, gold, predictions, logits
            ):
                row = {
                    "query_id": identifier,
                    "split": split,
                    "gold_label_id": int(true_label),
                    "gold_label_name": label_names[int(true_label)],
                    "prediction_label_id": int(prediction),
                    "prediction_label_name": label_names[int(prediction)],
                    "logits": [float(value) for value in scores],
                }
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    report = {
        "experiment": config["experiment"],
        "seed": seed,
        "model": model_config,
        "training": training,
        "class_weights": [float(value) for value in weights],
        "train_samples": len(train_rows),
        "pilot_samples": len(pilot_rows),
        "heldout_dev_samples": len(heldout_rows),
        "best_epoch": best_epoch,
        "checkpoint_sha256": file_sha256(staging / "best_state.pt"),
        "history": history,
        "metrics": metrics,
        "inputs": {
            "train_sha256": file_sha256(train_path),
            "pilot_sha256": file_sha256(pilot_path),
            "heldout_dev_sha256": file_sha256(heldout_path),
        },
        "selection_split": "frozen_pilot_dev_200",
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
    parser = argparse.ArgumentParser(description="TrendFact claim-only RoBERTa baseline")
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

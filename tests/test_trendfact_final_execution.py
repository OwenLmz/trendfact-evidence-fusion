from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from baselines import predict_trendfact_frozen_roberta as predictor
from baselines.predict_trendfact_frozen_roberta import select_checkpoint
from evaluation.evaluate_trendfact_evidence_aggregator import _align
from evaluation.prepare_trendfact_final_test import build_test_queries
from evaluation.evaluate_trendfact_final_ablations import (
    fuse_probabilities,
    validate_preselected_alphas,
)
from retrieval.evidence import file_sha256


def test_retrained_checkpoint_is_explicit_and_bound_to_training_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(predictor, "ROOT", tmp_path)
    configs = tmp_path / "configs"
    configs.mkdir()
    inputs = {}
    hashes = {}
    for split in ("train", "pilot", "heldout_dev"):
        path = tmp_path / f"{split}.jsonl"
        path.write_bytes(b"{}\n")
        inputs[split] = str(path)
        hashes[f"{split}_sha256"] = file_sha256(path)
    model = {"name": "base-model"}
    training = {"epochs": 5}
    (configs / "trendfact_claim_roberta_seed13.yaml").write_text(
        yaml.safe_dump({"model": model, "training": training, "input": inputs}),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "best_state.pt"
    checkpoint.write_bytes(b"original checkpoint fixture")
    frozen = {"claim_model": {"checkpoints": [{
        "seed": 13, "bytes": checkpoint.stat().st_size,
        "sha256": file_sha256(checkpoint), "best_epoch": 5,
    }]}}
    assert predictor.validate_checkpoint(frozen, 13, checkpoint, retrained=False)[0] == 5
    checkpoint.write_bytes(b"retrained checkpoint fixture")
    with pytest.raises(ValueError, match="Frozen checkpoint"):
        predictor.validate_checkpoint(frozen, 13, checkpoint, retrained=False)

    report = {
        "seed": 13, "model": model, "training": training, "inputs": hashes,
        "best_epoch": 4, "checkpoint_sha256": file_sha256(checkpoint),
        "selection_split": "frozen_pilot_dev_200",
        "heldout_used_for_parameter_selection": False, "test_used": False,
    }
    checkpoint.with_name("report.json").write_text(json.dumps(report), encoding="utf-8")
    assert predictor.validate_checkpoint(frozen, 13, checkpoint, retrained=True) == (
        4, file_sha256(checkpoint)
    )
    checkpoint.write_bytes(b"unexpected replacement")
    with pytest.raises(ValueError, match="documented training run"):
        predictor.validate_checkpoint(frozen, 13, checkpoint, retrained=True)


def test_final_query_adapter_preserves_test_split_and_stable_id() -> None:
    rows = [
        {
            "id": "x",
            "split": "test",
            "claim": "待核查主张",
            "label_id": 0,
            "label_name": "support",
            "domain": "科学",
            "source_cohort": "single_evidence_schema",
            "gold_target_indices": [3],
            "gold_evidence": [{"target_idx": 3, "text": "证据"}],
        }
    ]

    queries = build_test_queries(rows)

    assert queries[0]["query_id"] == "x::claim"
    assert queries[0]["split"] == "test"
    assert queries[0]["gold_target_indices"] == [3]


def test_evidence_alignment_requires_explicit_final_test_permission() -> None:
    queries = [{"query_id": "q", "split": "test"}]
    bm25 = [{"query_id": "q", "hits": [{}] * 20}]
    qwen = [{"query_id": "q", "candidate_decisions": [{}] * 20}]

    with pytest.raises(ValueError, match="allow-final-test"):
        _align(queries, bm25, qwen, candidate_top_k=20)

    _align(queries, bm25, qwen, candidate_top_k=20, allow_test=True)


def test_frozen_checkpoint_selection_requires_exact_seed() -> None:
    checkpoints = [
        {"seed": 13, "path": "a"},
        {"seed": 42, "path": "b"},
    ]

    assert select_checkpoint(checkpoints, 42)["path"] == "b"
    with pytest.raises(ValueError, match="exactly one"):
        select_checkpoint(checkpoints, 7)


def test_fixed_fusion_formula_cannot_select_alpha() -> None:
    roberta = np.asarray([[0.6, 0.3, 0.1]])
    evidence = np.asarray([[0.2, 0.7, 0.1]])
    alpha = 0.5

    combined = (1.0 - alpha) * roberta + alpha * evidence

    np.testing.assert_allclose(combined, [[0.4, 0.5, 0.1]])


def test_final_ablation_requires_exact_preselected_alphas() -> None:
    report = {"selected_evidence_alphas": {"direct": 0.25, "gate": 0.5}}

    selected = validate_preselected_alphas(
        report, ["direct", "gate"], {"direct": 0.25, "gate": 0.5}
    )

    assert selected == {"direct": 0.25, "gate": 0.5}
    with pytest.raises(ValueError, match="frozen expected"):
        validate_preselected_alphas(
            report, ["direct", "gate"], {"direct": 0.5, "gate": 0.5}
        )


def test_final_ablation_fusion_rejects_invalid_inputs() -> None:
    roberta = np.asarray([[0.6, 0.3, 0.1]])
    evidence = np.asarray([[0.2, 0.7, 0.1]])

    np.testing.assert_allclose(
        fuse_probabilities(roberta, evidence, alpha=0.25),
        [[0.5, 0.4, 0.1]],
    )
    with pytest.raises(ValueError, match="shapes"):
        fuse_probabilities(roberta, evidence[:, :2], alpha=0.25)
    with pytest.raises(ValueError, match="between 0 and 1"):
        fuse_probabilities(roberta, evidence, alpha=1.5)

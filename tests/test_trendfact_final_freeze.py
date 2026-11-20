from __future__ import annotations

from pathlib import Path

import yaml

from evaluation.validate_trendfact_final_freeze import validate
from retrieval.evidence import file_sha256


def test_final_freeze_validator_checks_hashes_without_running_test(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.jsonl"
    artifact.write_text("{}\n{}\n", encoding="utf-8")
    config = {
        "experiment": "test",
        "status": "frozen_not_run",
        "dataset": {
            "test_path": str(artifact),
            "test_rows": 2,
        },
        "evidence": {
            "retrieval": {"candidate_top_k": 20},
            "qwen": {"selected_top_k": 5},
        },
        "decision_fusion": {"evidence_alpha": 0.5},
        "output": {"root": str(tmp_path / "not_created")},
        "protocol": {"configuration_frozen": True, "test_may_be_run_once": True},
        "local_artifacts": [
            {"path": str(artifact), "sha256": file_sha256(artifact)}
        ],
    }
    config_path = tmp_path / "freeze.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    report = validate(config_path)

    assert report["ready_for_explicit_one_shot_execution"] is True
    assert report["test_predictions_generated"] is False

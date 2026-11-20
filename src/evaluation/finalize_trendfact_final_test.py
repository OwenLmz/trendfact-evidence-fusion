from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from retrieval.evidence import file_sha256  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/final_test_trendfact.yaml"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def finalize(config_path: Path = DEFAULT_CONFIG) -> dict[str, object]:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = ROOT / str(config["output"]["root"])
    completion_path = output_root / "completion_manifest.json"
    if completion_path.exists():
        raise FileExistsError(f"Completion manifest already exists: {completion_path}")
    expected_rows = int(config["dataset"]["test_rows"])
    paths = {
        "queries": output_root / "data/test_queries.jsonl",
        "bm25": output_root / "retrieval_bm25/retrieval_candidates.jsonl",
        "bm25_metrics": output_root / "retrieval_bm25/metrics.json",
        "qwen": output_root / "qwen/test_top5.jsonl",
        "qwen_manifest": output_root / "qwen/manifest.json",
        "roberta_seed13": output_root / "roberta_seed13/test_predictions.jsonl",
        "roberta_seed42": output_root / "roberta_seed42/test_predictions.jsonl",
        "roberta_seed2026": output_root / "roberta_seed2026/test_predictions.jsonl",
        "evidence": output_root / "evidence_aggregator/predictions.jsonl",
        "evidence_report": output_root / "evidence_aggregator/report.json",
        "final": output_root / "final_fusion/predictions.jsonl",
        "final_report": output_root / "final_fusion/report.json",
        "error_analysis": output_root / "error_analysis/report.json",
        "changed_cases": output_root / "error_analysis/changed_cases.jsonl",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing final-test artifacts: {missing}")
    row_paths = [
        "queries",
        "bm25",
        "qwen",
        "roberta_seed13",
        "roberta_seed42",
        "roberta_seed2026",
        "evidence",
        "final",
    ]
    row_counts = {name: _line_count(paths[name]) for name in row_paths}
    if any(count != expected_rows for count in row_counts.values()):
        raise ValueError(f"Final-test row counts do not match {expected_rows}: {row_counts}")

    qwen = _read_json(paths["qwen_manifest"])
    evidence = _read_json(paths["evidence_report"])
    final = _read_json(paths["final_report"])
    error_analysis = _read_json(paths["error_analysis"])
    if int(qwen["rows"]) != expected_rows or int(qwen["pairs_processed_this_run"]) != 23000:
        raise ValueError("Qwen final-test manifest is incomplete")
    if not all(bool(report["test_used"]) for report in (qwen, final, error_analysis)):
        raise ValueError("One or more final reports do not record test_used=true")
    if not bool(evidence["training_protocol"]["test_used"]):
        raise ValueError("Evidence aggregator does not record test_used=true")
    config_hash = file_sha256(config_path)
    if str(final["freeze_config_sha256"]) != config_hash:
        raise ValueError("Final fusion used a different freeze config")

    hashes = {name: file_sha256(path) for name, path in paths.items()}
    checkpoint_sources = {
        str(seed): _read_json((output_root / f"roberta_seed{seed}/report.json")).get(
            "checkpoint_source", "original_frozen"
        )
        for seed in (13, 42, 2026)
    }
    report = {
        "experiment": config["experiment"],
        "status": "complete",
        "freeze_config_sha256": config_hash,
        "row_counts": row_counts,
        "qwen_pairs": int(qwen["pairs_processed_this_run"]),
        "qwen_parse_errors": int(qwen["pair_errors_this_run"]),
        "changed_cases": _line_count(paths["changed_cases"]),
        "metrics": final["metrics"],
        "paired_fusion_vs_roberta_ensemble": final[
            "paired_fusion_vs_roberta_ensemble"
        ],
        "artifact_sha256": hashes,
        "checkpoint_sources": checkpoint_sources,
        "test_used": True,
        "parameters_selected_on_test": False,
        "rerun_permitted": False,
    }
    completion_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize the one-shot TrendFact test")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(finalize(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

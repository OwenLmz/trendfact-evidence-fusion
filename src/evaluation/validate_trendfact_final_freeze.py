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


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def validate(config_path: Path = DEFAULT_CONFIG) -> dict[str, object]:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    checks: list[dict[str, object]] = []

    if config["status"] != "frozen_not_run":
        errors.append("Final-test status must remain frozen_not_run before execution")
    if not bool(config["protocol"]["configuration_frozen"]):
        errors.append("configuration_frozen must be true")
    if not bool(config["protocol"]["test_may_be_run_once"]):
        errors.append("test_may_be_run_once must be true")
    if float(config["decision_fusion"]["evidence_alpha"]) != 0.5:
        errors.append("Frozen evidence alpha must be 0.5")
    if int(config["evidence"]["retrieval"]["candidate_top_k"]) != 20:
        errors.append("Frozen BM25 candidate_top_k must be 20")
    if int(config["evidence"]["qwen"]["selected_top_k"]) != 5:
        errors.append("Frozen Qwen selected_top_k must be 5")

    for artifact in config["local_artifacts"]:
        path = _resolve(artifact["path"])
        expected = str(artifact["sha256"]).lower()
        if not path.exists():
            errors.append(f"Missing local artifact: {path}")
            continue
        actual = file_sha256(path)
        checks.append(
            {
                "path": _display_path(path),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "matched": actual == expected,
            }
        )
        if actual != expected:
            errors.append(f"SHA-256 mismatch: {path}")

    test_path = _resolve(config["dataset"]["test_path"])
    if test_path.exists():
        actual_rows = _line_count(test_path)
        checks.append(
            {
                "path": _display_path(test_path),
                "expected_rows": int(config["dataset"]["test_rows"]),
                "actual_rows": actual_rows,
                "matched": actual_rows == int(config["dataset"]["test_rows"]),
            }
        )
        if actual_rows != int(config["dataset"]["test_rows"]):
            errors.append("TrendFact test row count mismatch")

    output_root = _resolve(config["output"]["root"])
    if output_root.exists():
        errors.append(f"Final output already exists and may not be overwritten: {output_root}")

    report = {
        "experiment": config["experiment"],
        "config_path": _display_path(config_path),
        "config_sha256": file_sha256(config_path),
        "checks": checks,
        "errors": errors,
        "ready_for_explicit_one_shot_execution": not errors,
        "test_predictions_generated": False,
    }
    if errors:
        raise ValueError(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the frozen TrendFact final-test protocol")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args()
    report = validate(args.config)
    if args.report_path is not None:
        report_path = args.report_path.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

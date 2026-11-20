from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from data.prepare_trendfact import query_from_row  # noqa: E402
from evaluation.evaluate_trendfact_evidence_aggregator import _read_jsonl, _resolve  # noqa: E402
from evaluation.validate_trendfact_final_freeze import validate  # noqa: E402
from retrieval.evidence import file_sha256  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/final_test_trendfact.yaml"


def build_test_queries(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    if {str(row["split"]) for row in rows} != {"test"}:
        raise ValueError("Final-test query preparation requires test rows only")
    queries = [query_from_row(row, split="test") for row in rows]
    identifiers = [str(row["query_id"]) for row in queries]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Final-test query IDs are not unique")
    return queries


def prepare(config_path: Path = DEFAULT_CONFIG) -> dict[str, object]:
    config_path = config_path.resolve()
    validation = validate(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    test_path = _resolve(config["dataset"]["test_path"])
    output_root = _resolve(config["output"]["root"])
    if output_root.exists():
        raise FileExistsError(f"Final-test output already exists: {output_root}")
    rows = _read_jsonl(test_path)
    queries = build_test_queries(rows)
    if len(queries) != int(config["dataset"]["test_rows"]):
        raise ValueError("Final-test query count does not match the frozen config")

    data_dir = output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=False)
    query_path = data_dir / "test_queries.jsonl"
    with query_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in queries:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
    manifest = {
        "experiment": config["experiment"],
        "status": "started",
        "freeze_config": config_path.relative_to(ROOT).as_posix(),
        "freeze_config_sha256": file_sha256(config_path),
        "test_input_sha256": file_sha256(test_path),
        "test_queries": len(queries),
        "test_queries_path": query_path.relative_to(ROOT).as_posix(),
        "test_queries_sha256": file_sha256(query_path),
        "freeze_validation": validation,
        "test_predictions_generated": False,
    }
    (output_root / "execution_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the one-shot TrendFact final test")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(prepare(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

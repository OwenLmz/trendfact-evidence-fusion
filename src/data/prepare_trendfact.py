from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from retrieval.evidence import file_sha256, normalize_for_search, text_sha256


DEFAULT_CONFIG = ROOT / "configs/data_trendfact.yaml"
LABEL_NAMES = {0: "support", 1: "refute", 2: "nei"}
TOKEN_CHARACTER_RE = re.compile(r"[\u3400-\u9fffA-Za-z0-9]")


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def iter_gold_evidence(record: Mapping[str, object]) -> list[dict[str, object]]:
    raw = record.get("evidence")
    if not isinstance(raw, Mapping):
        raise ValueError("TrendFact record is missing an evidence object")
    if {"text", "url", "target_idx"}.issubset(raw):
        candidates = [raw]
    else:
        numeric_items = sorted(
            ((int(key), value) for key, value in raw.items() if str(key).isdigit()),
            key=lambda item: item[0],
        )
        candidates = [value for _, value in numeric_items]
    evidence: list[dict[str, object]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("TrendFact gold evidence entry must be an object")
        text = str(candidate.get("text", "")).strip()
        url = str(candidate.get("url", "")).strip()
        target_idx = candidate.get("target_idx")
        if not text and target_idx is None:
            continue
        if not text or not isinstance(target_idx, int) or target_idx < 0:
            raise ValueError("TrendFact gold evidence requires text and a non-negative target_idx")
        evidence.append({"text": text, "url": url, "target_idx": target_idx})
    if not evidence:
        raise ValueError("TrendFact record has no usable gold evidence")
    return evidence


def source_cohort(record: Mapping[str, object]) -> str:
    return "multi_evidence_schema" if "claimId" in record else "single_evidence_schema"


def _stable_record_id(raw_index: int, claim: str, record: Mapping[str, object]) -> str:
    source_key = str(record.get("claimId") or record.get("url") or raw_index)
    digest = text_sha256(f"{source_key}\0{normalize_for_search(claim)}")[:16]
    return f"trendfact_{raw_index:05d}_{digest}"


def adapt_record(raw_index: int, record: Mapping[str, object]) -> dict[str, object]:
    claim = str(record.get("claim", "")).strip()
    if not claim:
        raise ValueError(f"TrendFact row {raw_index} has an empty claim")
    label_id = record.get("label")
    if not isinstance(label_id, int) or label_id not in LABEL_NAMES:
        raise ValueError(f"TrendFact row {raw_index} has invalid label {label_id!r}")
    cohort = source_cohort(record)
    title = str(record.get("ori_title") or record.get("title") or "").strip()
    claim_date = str(record.get("date") or record.get("publish_date") or "").strip()
    gold_evidence = iter_gold_evidence(record)
    return {
        "id": _stable_record_id(raw_index, claim, record),
        "raw_index": raw_index,
        "dataset": "trendfact",
        "label_id": label_id,
        "label_name": LABEL_NAMES[label_id],
        "claim": claim,
        "normalized_claim": normalize_for_search(claim),
        "claim_sha256": text_sha256(normalize_for_search(claim)),
        "title": title,
        "domain": str(record.get("domain", "")).strip(),
        "source_cohort": cohort,
        "source": str(record.get("source") or cohort).strip(),
        "claim_url": str(record.get("url", "")).strip(),
        "claim_date": claim_date,
        "gold_evidence": gold_evidence,
        "gold_target_indices": sorted(
            {int(item["target_idx"]) for item in gold_evidence}
        ),
        "gold_explanation": str(record.get("explanation", "")).strip(),
        "hotspot_metadata": {
            "views": record.get("views"),
            "discussion": record.get("Discussion"),
            "engagement": record.get("Engagemen"),
            "post": record.get("Post"),
            "influence_score": record.get("influence_score"),
            "final_influence_score": record.get("final_influence_score"),
        },
    }


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    compact = "".join(char for char in text if TOKEN_CHARACTER_RE.fullmatch(char))
    return {compact[index : index + n] for index in range(max(0, len(compact) - n + 1))}


def build_groups(
    rows: list[dict[str, object]], *, threshold: float, neighbors: int
) -> tuple[list[list[dict[str, object]]], list[dict[str, object]], dict[str, int]]:
    union_find = UnionFind(len(rows))
    exact_owner: dict[str, int] = {}
    target_owner: dict[int, int] = {}
    exact_links = 0
    evidence_links = 0
    for index, row in enumerate(rows):
        normalized = str(row["normalized_claim"])
        if normalized in exact_owner:
            union_find.union(index, exact_owner[normalized])
            exact_links += 1
        else:
            exact_owner[normalized] = index
        for target_idx in row["gold_target_indices"]:  # type: ignore[union-attr]
            target_idx = int(target_idx)
            if target_idx in target_owner:
                union_find.union(index, target_owner[target_idx])
                evidence_links += 1
            else:
                target_owner[target_idx] = index

    texts = [str(row["normalized_claim"]) for row in rows]
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 3), binary=True, use_idf=False, norm="l2"
    )
    matrix = vectorizer.fit_transform(texts)
    model = NearestNeighbors(
        n_neighbors=min(neighbors, len(rows)), metric="cosine", algorithm="brute", n_jobs=1
    ).fit(matrix)
    _, candidate_indices = model.kneighbors(matrix)
    grams: dict[int, set[str]] = {}
    seen: set[tuple[int, int]] = set()
    near_pairs: list[dict[str, object]] = []
    for left_index, candidates in enumerate(candidate_indices):
        for raw_right_index in candidates:
            right_index = int(raw_right_index)
            if right_index == left_index:
                continue
            pair = tuple(sorted((left_index, right_index)))
            if pair in seen:
                continue
            seen.add(pair)
            grams.setdefault(left_index, _char_ngrams(texts[left_index]))
            grams.setdefault(right_index, _char_ngrams(texts[right_index]))
            union = grams[left_index] | grams[right_index]
            similarity = len(grams[left_index] & grams[right_index]) / len(union) if union else 0
            if similarity >= threshold:
                union_find.union(left_index, right_index)
                near_pairs.append(
                    {
                        "left_id": rows[left_index]["id"],
                        "right_id": rows[right_index]["id"],
                        "jaccard_char_3gram": round(similarity, 6),
                        "same_label": rows[left_index]["label_id"]
                        == rows[right_index]["label_id"],
                    }
                )

    components: dict[int, list[dict[str, object]]] = defaultdict(list)
    for index, row in enumerate(rows):
        components[union_find.find(index)].append(row)
    groups = sorted(
        components.values(), key=lambda group: min(str(row["id"]) for row in group)
    )
    return groups, sorted(near_pairs, key=lambda row: (row["left_id"], row["right_id"])), {
        "exact_claim_links": exact_links,
        "shared_gold_target_links": evidence_links,
    }


def stable_group_split(
    groups: Sequence[Sequence[dict[str, object]]],
    *,
    seed: int,
    train_ratio: float,
    dev_ratio: float,
) -> dict[str, list[dict[str, object]]]:
    ratios = {"train": train_ratio, "dev": dev_ratio, "test": 1 - train_ratio - dev_ratio}
    if any(value <= 0 for value in ratios.values()):
        raise ValueError("All split ratios must be positive")
    strata_total = Counter(
        (str(row["label_name"]), str(row["source_cohort"]))
        for group in groups
        for row in group
    )
    targets = {
        split: {stratum: count * ratio for stratum, count in strata_total.items()}
        for split, ratio in ratios.items()
    }
    assigned = {split: Counter() for split in ratios}
    output = {split: [] for split in ratios}
    ordered = sorted(
        groups,
        key=lambda group: (
            -len(group),
            hashlib.sha256(
                f"{seed}\0{min(str(row['id']) for row in group)}".encode("utf-8")
            ).hexdigest(),
        ),
    )
    split_priority = {"train": 2, "dev": 1, "test": 0}
    for group in ordered:
        counts = Counter(
            (str(row["label_name"]), str(row["source_cohort"])) for row in group
        )
        scores: dict[str, float] = {}
        for split in ratios:
            score = 0.0
            for stratum, count in counts.items():
                deficit = targets[split][stratum] - assigned[split][stratum]
                score += count * deficit / max(targets[split][stratum], 1.0)
            scores[split] = score
        chosen = max(scores, key=lambda split: (scores[split], split_priority[split]))
        output[chosen].extend(group)
        assigned[chosen].update(counts)
    for split, rows in output.items():
        for row in rows:
            row["split"] = split
        rows.sort(key=lambda row: str(row["id"]))
    return output


def select_balanced_queries(
    rows: Sequence[Mapping[str, object]],
    *,
    count: int,
    seed: int,
    split: str,
    purpose: str,
) -> list[dict[str, object]]:
    by_label: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if str(row.get("split", "")) != split:
            raise ValueError(f"Expected only {split} rows for {purpose}")
        by_label[str(row["label_name"])].append(row)
    labels = sorted(by_label)
    base, remainder = divmod(count, len(labels))
    selected: list[Mapping[str, object]] = []
    for label_index, label in enumerate(labels):
        quota = base + (1 if label_index < remainder else 0)
        ordered = sorted(
            by_label[label],
            key=lambda row: hashlib.sha256(
                f"{seed}\0{purpose}\0{row['id']}".encode("utf-8")
            ).hexdigest(),
        )
        if len(ordered) < quota:
            raise ValueError(f"Not enough {split} rows for {purpose} label {label}")
        selected.extend(ordered[:quota])
    selected.sort(key=lambda row: str(row["id"]))
    return [query_from_row(row, split=split) for row in selected]


def query_from_row(row: Mapping[str, object], *, split: str) -> dict[str, object]:
    if str(row.get("split", "")) != split:
        raise ValueError(f"Query row does not belong to requested split {split}")
    return {
        "query_id": f"{row['id']}::claim",
        "document_id": row["id"],
        "dataset": "trendfact",
        "split": split,
        "query_text": row["claim"],
        "label_id": row["label_id"],
        "label_name": row["label_name"],
        "domain": row["domain"],
        "source_cohort": row["source_cohort"],
        "gold_target_indices": row["gold_target_indices"],
        "gold_evidence_hashes": {
            str(item["target_idx"]): text_sha256(normalize_for_search(str(item["text"])))
            for item in row["gold_evidence"]  # type: ignore[union-attr]
        },
    }


def select_pilot_queries(
    dev_rows: Sequence[Mapping[str, object]], *, count: int, seed: int
) -> list[dict[str, object]]:
    return select_balanced_queries(
        dev_rows,
        count=count,
        seed=seed,
        split="dev",
        purpose="pilot",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )


def prepare(config_path: Path = DEFAULT_CONFIG, *, overwrite: bool = False) -> dict[str, object]:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_path = (ROOT / config["input"]["dataset_json"]).resolve()
    actual_hash = file_sha256(raw_path)
    expected_hash = str(config["input"]["dataset_json_sha256"]).lower()
    if actual_hash != expected_hash:
        raise ValueError(f"TrendFact JSON SHA-256 mismatch: {actual_hash} != {expected_hash}")
    raw_records = json.loads(raw_path.read_text(encoding="utf-8"))
    if not isinstance(raw_records, list):
        raise ValueError("TrendFact JSON root must be a list")
    if len(raw_records) != int(config["input"]["expected_dataset_rows"]):
        raise ValueError("TrendFact row count does not match the frozen config")

    excluded: Counter[str] = Counter()
    rows: list[dict[str, object]] = []
    for index, record in enumerate(raw_records):
        if not str(record.get("claim", "")).strip():
            excluded["empty_claim"] += 1
            continue
        rows.append(adapt_record(index, record))
    protocol = config["protocol"]
    groups, near_pairs, link_stats = build_groups(
        rows,
        threshold=float(protocol["near_duplicate_threshold"]),
        neighbors=int(protocol["nearest_neighbors"]),
    )
    splits = stable_group_split(
        groups,
        seed=int(config["seed"]),
        train_ratio=float(protocol["train_ratio"]),
        dev_ratio=float(protocol["dev_ratio"]),
    )
    pilot_queries = select_pilot_queries(
        splits["dev"], count=int(config["pilot"]["query_count"]), seed=int(config["seed"])
    )
    gate_config = config["gate_training"]
    gate_train_queries = select_balanced_queries(
        splits[str(gate_config["split"])],
        count=int(gate_config["query_count"]),
        seed=int(config["seed"]),
        split=str(gate_config["split"]),
        purpose=str(gate_config["purpose"]),
    )
    pilot_document_ids = {str(query["document_id"]) for query in pilot_queries}
    heldout_dev_queries = [
        query_from_row(row, split="dev")
        for row in splits["dev"]
        if str(row["id"]) not in pilot_document_ids
    ]
    expected_heldout = int(config["heldout_evaluation"]["expected_query_count"])
    if len(heldout_dev_queries) != expected_heldout:
        raise ValueError(
            f"Held-out dev query count mismatch: {len(heldout_dev_queries)} != {expected_heldout}"
        )

    output_dir = (ROOT / config["output"]["data_dir"]).resolve()
    audit_dir = (ROOT / config["output"]["audit_dir"]).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output exists: {output_dir}. Pass --overwrite to replace it.")
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    for split, split_rows in splits.items():
        _write_jsonl(output_dir / f"{split}.jsonl", split_rows)
    _write_jsonl(output_dir / "pilot_dev_queries.jsonl", pilot_queries)
    _write_jsonl(output_dir / "gate_train_queries.jsonl", gate_train_queries)
    _write_jsonl(output_dir / "heldout_dev_queries.jsonl", heldout_dev_queries)
    _write_jsonl(audit_dir / "near_duplicate_pairs.jsonl", near_pairs)

    split_by_id = {
        str(row["id"]): split for split, split_rows in splits.items() for row in split_rows
    }
    cross_group_errors = sum(
        len({split_by_id[str(row["id"])] for row in group}) > 1 for group in groups
    )
    manifest = {
        "experiment": config["experiment"],
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": file_sha256(config_path),
        "input": {
            "path": raw_path.relative_to(ROOT).as_posix(),
            "sha256": actual_hash,
            "raw_rows": len(raw_records),
            "usable_rows": len(rows),
            "excluded": dict(sorted(excluded.items())),
            "license": config["input"]["license"],
        },
        "protocol": {
            "official_split_available": False,
            "derived_split": True,
            "method": protocol["split_method"],
            "test_used_for_model_selection": False,
            "raw_structural_examples_viewed_before_split": int(
                protocol["raw_structural_examples_viewed_before_split"]
            ),
        },
        "schema": {
            "source_cohorts": dict(sorted(Counter(row["source_cohort"] for row in rows).items())),
            "gold_evidence_count": dict(
                sorted(Counter(len(row["gold_evidence"]) for row in rows).items())
            ),
            "max_gold_target_idx": max(
                target_idx for row in rows for target_idx in row["gold_target_indices"]
            ),
        },
        "grouping": {
            "groups": len(groups),
            "largest_group": max(len(group) for group in groups),
            "near_duplicate_pairs": len(near_pairs),
            "cross_split_group_errors": cross_group_errors,
            **link_stats,
        },
        "splits": {
            split: {
                "rows": len(split_rows),
                "labels": dict(sorted(Counter(row["label_name"] for row in split_rows).items())),
                "domains": dict(sorted(Counter(row["domain"] for row in split_rows).items())),
                "source_cohorts": dict(
                    sorted(Counter(row["source_cohort"] for row in split_rows).items())
                ),
                "sha256": file_sha256(output_dir / f"{split}.jsonl"),
            }
            for split, split_rows in splits.items()
        },
        "pilot": {
            "queries": len(pilot_queries),
            "labels": dict(sorted(Counter(row["label_name"] for row in pilot_queries).items())),
            "file": (output_dir / "pilot_dev_queries.jsonl").relative_to(ROOT).as_posix(),
            "sha256": file_sha256(output_dir / "pilot_dev_queries.jsonl"),
        },
        "gate_training": {
            "queries": len(gate_train_queries),
            "labels": dict(
                sorted(Counter(row["label_name"] for row in gate_train_queries).items())
            ),
            "file": (output_dir / "gate_train_queries.jsonl").relative_to(ROOT).as_posix(),
            "sha256": file_sha256(output_dir / "gate_train_queries.jsonl"),
            "test_used": False,
        },
        "heldout_evaluation": {
            "queries": len(heldout_dev_queries),
            "labels": dict(
                sorted(Counter(row["label_name"] for row in heldout_dev_queries).items())
            ),
            "file": (output_dir / "heldout_dev_queries.jsonl").relative_to(ROOT).as_posix(),
            "sha256": file_sha256(output_dir / "heldout_dev_queries.jsonl"),
            "overlap_with_pilot": len(
                pilot_document_ids
                & {str(query["document_id"]) for query in heldout_dev_queries}
            ),
            "test_used": False,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the derived TrendFact train/dev/test protocol.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.config, overwrite=args.overwrite), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

from data.prepare_trendfact import (
    adapt_record,
    iter_gold_evidence,
    select_pilot_queries,
    select_balanced_queries,
    query_from_row,
    stable_group_split,
)


def _single_record(label: int = 0) -> dict[str, object]:
    return {
        "claim": "甲地在2026年发布了一项可核验政策。",
        "label": label,
        "domain": "社会",
        "ori_title": "政策通报",
        "url": "https://example.test/claim",
        "date": "2026-01-01",
        "explanation": "依据证据判断。",
        "evidence": {
            "text": "甲地政府于2026年正式发布该政策。",
            "url": "https://example.test/evidence",
            "target_idx": 7,
        },
    }


def test_gold_evidence_adapter_supports_both_official_shapes() -> None:
    single = iter_gold_evidence(_single_record())
    multi_record = _single_record()
    multi_record["evidence"] = {
        "0": {"text": "第一条有效证据。", "url": "u0", "target_idx": 3},
        "1": {"text": "第二条有效证据。", "url": "u1", "target_idx": 4},
        "2": {"text": ""},
    }
    multi = iter_gold_evidence(multi_record)
    assert [row["target_idx"] for row in single] == [7]
    assert [row["target_idx"] for row in multi] == [3, 4]


def test_adapt_record_uses_official_label_mapping() -> None:
    row = adapt_record(2, _single_record(label=1))
    assert row["label_id"] == 1
    assert row["label_name"] == "refute"
    assert row["gold_target_indices"] == [7]
    assert row["source_cohort"] == "single_evidence_schema"


def test_group_split_is_deterministic_and_keeps_groups_intact() -> None:
    rows = []
    for index, label in enumerate((0, 0, 1, 1, 2, 2, 0, 1, 2)):
        row = adapt_record(index, _single_record(label=label))
        row["id"] = f"row-{index}"
        row["source_cohort"] = "single_evidence_schema"
        rows.append(row)
    groups = [[rows[0], rows[1]], *[[row] for row in rows[2:]]]
    first = stable_group_split(groups, seed=11, train_ratio=0.6, dev_ratio=0.2)
    second = stable_group_split(groups, seed=11, train_ratio=0.6, dev_ratio=0.2)
    first_split = {
        str(row["id"]): split for split, split_rows in first.items() for row in split_rows
    }
    second_split = {
        str(row["id"]): split for split, split_rows in second.items() for row in split_rows
    }
    assert first_split == second_split
    assert first_split["row-0"] == first_split["row-1"]


def test_pilot_query_selection_is_dev_only_and_label_balanced() -> None:
    rows = []
    for index, label in enumerate((0, 0, 1, 1, 2, 2)):
        row = adapt_record(index, _single_record(label=label))
        row["id"] = f"row-{index}"
        row["split"] = "dev"
        rows.append(row)
    queries = select_pilot_queries(rows, count=6, seed=3)
    assert {row["split"] for row in queries} == {"dev"}
    assert {row["label_name"] for row in queries} == {"support", "refute", "nei"}
    assert all(row["gold_target_indices"] == [7] for row in queries)
    assert all(set(row["gold_evidence_hashes"]) == {"7"} for row in queries)


def test_gate_query_selection_is_train_only_and_label_balanced() -> None:
    rows = []
    for index, label in enumerate((0, 0, 1, 1, 2, 2)):
        row = adapt_record(index, _single_record(label=label))
        row["id"] = f"train-row-{index}"
        row["split"] = "train"
        rows.append(row)

    queries = select_balanced_queries(
        rows,
        count=6,
        seed=3,
        split="train",
        purpose="train_gate",
    )

    assert {row["split"] for row in queries} == {"train"}
    assert {row["label_name"] for row in queries} == {"support", "refute", "nei"}


def test_query_adapter_preserves_requested_split_and_gold_pointer() -> None:
    row = adapt_record(0, _single_record(label=0))
    row["split"] = "dev"
    query = query_from_row(row, split="dev")
    assert query["split"] == "dev"
    assert query["gold_target_indices"] == [7]
    assert query["query_text"] == row["claim"]

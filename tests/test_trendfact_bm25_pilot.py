from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq

from retrieval.evidence import normalize_for_search, text_sha256
from retrieval.run_trendfact_bm25_pilot import (
    _first_pass,
    _query_vocabulary,
    _write_outputs,
    retrieval_metrics,
)


def test_trendfact_retrieval_metrics_handle_single_and_multi_gold() -> None:
    queries = [
        {"gold_target_indices": [1]},
        {"gold_target_indices": [3, 4]},
        {"gold_target_indices": [8]},
    ]
    hits = [
        [{"target_idx": 1}, {"target_idx": 9}],
        [{"target_idx": 0}, {"target_idx": 4}, {"target_idx": 3}],
        [{"target_idx": 7}, {"target_idx": 6}],
    ]
    metrics = retrieval_metrics(queries, hits, [1, 2, 3])
    assert metrics["any_gold_recall_at_k"] == {"1": 0.333333, "2": 0.666667, "3": 0.666667}
    assert metrics["mean_gold_fraction_at_k"]["2"] == 0.5
    assert metrics["mean_gold_fraction_at_k"]["3"] == 0.666667
    assert metrics["mrr_at_max_k"] == 0.5
    assert metrics["mrr_at_k"] == {"1": 0.333333, "2": 0.5, "3": 0.5}


def test_trendfact_retrieval_metrics_truncate_mrr_at_max_k() -> None:
    queries = [{"gold_target_indices": [4]}]
    hits = [
        [
            {"target_idx": 1},
            {"target_idx": 2},
            {"target_idx": 3},
            {"target_idx": 4},
        ]
    ]

    metrics = retrieval_metrics(queries, hits, [1, 3])

    assert metrics["any_gold_recall_at_k"] == {"1": 0.0, "3": 0.0}
    assert metrics["mrr_at_max_k"] == 0.0
    assert metrics["mrr_at_k"] == {"1": 0.0, "3": 0.0}
    assert metrics["max_k"] == 3


def test_trendfact_parquet_alignment_and_blind_candidate_output(tmp_path) -> None:
    evidence_texts = ["甲地政府发布了正式政策。", "乙机构否认了相关传言。"]
    parquet_path = tmp_path / "evidence.parquet"
    pq.write_table(
        pa.table(
            {
                "url": ["https://example.test/0", "https://example.test/1"],
                "text": evidence_texts,
                "publish_date": ["2026-01-01", "2026-01-02"],
            }
        ),
        parquet_path,
    )
    queries = [
        {
            "query_id": "q0",
            "document_id": "d0",
            "split": "dev",
            "query_text": "甲地发布政策",
            "label_name": "support",
            "gold_target_indices": [0],
            "gold_evidence_hashes": {
                "0": text_sha256(normalize_for_search(evidence_texts[0]))
            },
        }
    ]
    vocabulary, _ = _query_vocabulary(queries, "character_bigram")
    count, _, _, alignment = _first_pass(
        parquet_path=parquet_path,
        queries=queries,
        vocabulary=vocabulary,
        tokenizer="character_bigram",
        batch_size=1,
    )
    assert count == 2
    assert alignment["exact_normalized_text_matches"] == 1
    assert alignment["text_mismatches"] == 0

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_outputs(
        output_dir,
        queries,
        [[{"target_idx": 0, "text": evidence_texts[0], "rank": 1}]],
    )
    candidate = json.loads(
        (output_dir / "retrieval_candidates.jsonl").read_text(encoding="utf-8")
    )
    assert "label_name" not in candidate
    assert "gold_target_indices" not in candidate

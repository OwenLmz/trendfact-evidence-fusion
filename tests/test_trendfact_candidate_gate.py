from __future__ import annotations

import numpy as np

from evaluation.train_trendfact_candidate_gate import (
    BM25_SCORE_FEATURES,
    FULL_FEATURE_NAMES,
    NO_LENGTH_FULL_FEATURES,
    _rank_candidates,
    extract_candidate_features,
)


def _inputs() -> tuple[list[dict[str, object]], dict[str, object]]:
    hits = [
        {"rank": rank, "bm25_score": float(21 - rank), "target_idx": rank}
        for rank in range(1, 21)
    ]
    decisions = [
        {
            "candidate_rank": rank,
            "relevance": 3 if rank == 4 else 1,
            "stance": "refute" if rank == 4 else "context",
            "candidate_text_chars": 1000,
            "candidate_truncated": True,
            "prompt_excerpt_start": rank * 10,
        }
        for rank in range(1, 21)
    ]
    return hits, {"candidate_decisions": decisions}


def test_candidate_features_have_no_gold_or_claim_label_signal() -> None:
    hits, qwen = _inputs()
    features, candidates = extract_candidate_features("待核查主张", hits, qwen)

    assert features.shape == (20, len(FULL_FEATURE_NAMES))
    assert len(candidates) == 20
    assert features[3, FULL_FEATURE_NAMES.index("llm_relevance")] == 3.0
    assert features[0, FULL_FEATURE_NAMES.index("bm25_score")] == 20.0
    assert features[0, FULL_FEATURE_NAMES.index("bm25_score_over_top1")] == 1.0
    assert not any("gold" in name or "label" in name for name in FULL_FEATURE_NAMES)
    assert "candidate_char_length" not in {
        FULL_FEATURE_NAMES[index] for index in NO_LENGTH_FULL_FEATURES
    }
    assert {FULL_FEATURE_NAMES[index] for index in BM25_SCORE_FEATURES} == {
        "bm25_reciprocal_rank",
        "bm25_score",
        "bm25_score_over_top1",
        "bm25_score_zscore",
    }


def test_candidate_gate_ranks_probability_then_original_rank() -> None:
    hits, _ = _inputs()
    probability = np.zeros(20)
    probability[3] = 0.9
    probability[6] = 0.8

    rankings = _rank_candidates([hits], probability, output_top_k=3)

    assert [row["target_idx"] for row in rankings[0]] == [4, 7, 1]

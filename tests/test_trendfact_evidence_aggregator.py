from __future__ import annotations

import numpy as np

from evaluation.evaluate_trendfact_evidence_aggregator import (
    _classification_metrics,
    _direct_qwen_probabilities,
    _structural_aggregation_features,
    aggregate_ranked_signals,
)


def test_aggregate_ranked_signals_tracks_stance_and_reliability() -> None:
    signals = [
        {
            "candidate_reliability": 0.9 - index * 0.1,
            "relevance": 3 if index < 2 else 1,
            "stance": "support" if index < 2 else "context",
            "bm25_rank": index + 1,
        }
        for index in range(5)
    ]

    features, names = aggregate_ranked_signals(signals, top_k=5)
    values = dict(zip(names, features))

    assert len(features) == len(names)
    assert values["rank1_candidate_reliability"] == 0.9
    assert values["rank1_stance_support"] == 1.0
    assert values["support_weighted_sum"] > values["refute_weighted_sum"]
    assert values["high_relevance_fraction"] == 0.4


def test_three_class_metrics_report_macro_and_per_class_scores() -> None:
    labels = np.asarray([0, 1, 2, 2])
    predictions = np.asarray([0, 1, 1, 2])

    metrics = _classification_metrics(labels, predictions, ["support", "refute", "nei"])

    assert metrics["accuracy"] == 0.75
    assert set(metrics["per_class"]) == {"support", "refute", "nei"}
    assert metrics["per_class"]["nei"]["recall"] == 0.5


def test_direct_qwen_scores_prioritize_relevance_then_direct_stance() -> None:
    features = np.zeros((3, 15), dtype=np.float64)
    features[:, 6] = [2, 2, 1]
    features[:, 8] = [0, 1, 1]

    probabilities = _direct_qwen_probabilities([features])[0]

    assert probabilities[1] > probabilities[0] > probabilities[2]
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_structural_aggregation_excludes_llm_semantics() -> None:
    features = np.arange(8, dtype=np.float64).reshape(1, -1)
    names = [
        "rank1_candidate_reliability",
        "rank1_llm_relevance",
        "rank1_bm25_reciprocal_rank",
        "rank1_stance_support",
        "support_weighted_sum",
        "reliability_mean",
        "relevance_mean",
        "high_relevance_fraction",
    ]

    selected, selected_names = _structural_aggregation_features(features, names)

    assert selected_names == [
        "rank1_candidate_reliability",
        "rank1_bm25_reciprocal_rank",
        "reliability_mean",
    ]
    assert selected.tolist() == [[0.0, 2.0, 5.0]]

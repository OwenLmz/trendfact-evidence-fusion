from __future__ import annotations

import numpy as np

from evaluation.evaluate_trendfact_roberta_evidence_ablations import (
    evaluate_fusion_method,
)


def test_evaluate_fusion_method_reports_component_and_combined_metrics() -> None:
    labels = np.asarray([0, 1, 2])
    roberta = np.asarray(
        [[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.4, 0.4, 0.2]], dtype=np.float64
    )
    evidence = np.asarray(
        [[0.6, 0.3, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], dtype=np.float64
    )

    report, combined = evaluate_fusion_method(
        labels,
        roberta,
        evidence,
        alpha=0.5,
        label_names=["support", "refute", "nei"],
        bootstrap_samples=20,
        seed=7,
    )

    assert report["evidence_alpha"] == 0.5
    assert report["roberta_evidence_fusion"]["accuracy"] == 1.0
    assert combined.shape == (3, 3)

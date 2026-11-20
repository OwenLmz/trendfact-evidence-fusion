from __future__ import annotations

import numpy as np

from evaluation.evaluate_trendfact_roberta_evidence_ensemble import select_alpha, softmax


def test_softmax_rows_sum_to_one() -> None:
    probabilities = softmax(np.asarray([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0]]))

    assert np.allclose(probabilities.sum(axis=1), [1.0, 1.0])
    assert np.argmax(probabilities[0]) == 2


def test_alpha_selection_uses_pilot_macro_f1_and_conservative_tie_break() -> None:
    labels = np.asarray([0, 1, 2])
    roberta = np.eye(3)
    evidence = np.eye(3)

    alpha, rows = select_alpha(
        labels, roberta, evidence, [0.0, 0.5, 1.0], ["support", "refute", "nei"]
    )

    assert alpha == 0.0
    assert len(rows) == 3

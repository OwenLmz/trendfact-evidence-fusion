from __future__ import annotations

import numpy as np
import pytest

from baselines.trendfact_transformer_baseline import (
    _texts_labels_ids,
    balanced_class_weights,
)


def test_trendfact_balanced_class_weights() -> None:
    weights = balanced_class_weights([0, 0, 1, 2])

    assert weights.tolist() == pytest.approx([4 / 6, 4 / 3, 4 / 3])


def test_trendfact_transformer_adapter_uses_claim_and_stable_id() -> None:
    texts, labels, identifiers = _texts_labels_ids(
        [
            {"id": "a", "claim": "甲", "label_id": 0},
            {"id": "b", "claim": "乙", "label_id": 1},
            {"id": "c", "claim": "丙", "label_id": 2},
        ],
        text_field="claim",
    )

    assert texts == ["甲", "乙", "丙"]
    assert labels == [0, 1, 2]
    assert identifiers == ["a", "b", "c"]
    assert isinstance(balanced_class_weights(labels), np.ndarray)

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def _hit_at_k(
    query: Mapping[str, object], hits: Sequence[Mapping[str, object]], k: int
) -> bool:
    gold = {int(value) for value in query["gold_target_indices"]}  # type: ignore[union-attr]
    retrieved = {int(hit["target_idx"]) for hit in hits[:k]}
    return bool(gold & retrieved)


def paired_bootstrap_difference(
    baseline: Sequence[float | bool],
    method: Sequence[float | bool],
    *,
    seed: int = 20260730,
    samples: int = 10000,
) -> dict[str, float]:
    if len(baseline) != len(method) or not baseline:
        raise ValueError("Paired outcomes must be non-empty and have equal length")
    differences = np.asarray(method, dtype=np.float32) - np.asarray(
        baseline, dtype=np.float32
    )
    generator = np.random.default_rng(seed)
    bootstrap = np.empty(samples, dtype=np.float32)
    for index in range(samples):
        positions = generator.integers(0, len(differences), size=len(differences))
        bootstrap[index] = float(differences[positions].mean())
    return {
        "difference": round(float(differences.mean()), 6),
        "ci95_low": round(float(np.quantile(bootstrap, 0.025)), 6),
        "ci95_high": round(float(np.quantile(bootstrap, 0.975)), 6),
        "bootstrap_samples": samples,
    }

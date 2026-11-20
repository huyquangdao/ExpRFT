from __future__ import annotations

import math
from collections import defaultdict

SIGMA_FLOOR = 0.1


def standardize_per_context(
    rewards: list[float], context_ids: list[str], sigma_floor: float = SIGMA_FLOOR
) -> list[float]:
    if len(rewards) != len(context_ids):
        raise ValueError("rewards and context_ids must have the same length")
    groups: dict[str, list[int]] = defaultdict(list)
    for i, cid in enumerate(context_ids):
        groups[cid].append(i)
    out = [0.0] * len(rewards)
    for idx in groups.values():
        rs = [float(rewards[i]) for i in idx]
        mu = sum(rs) / len(rs)
        var = sum((r - mu) ** 2 for r in rs) / len(rs)      # population variance (ddof = 0)
        if var < 1e-24:                                      # constant rewards -> exactly 0
            continue
        sigma = max(math.sqrt(var), sigma_floor)
        for i in idx:
            out[i] = (float(rewards[i]) - mu) / sigma
    return out


def exprft_weights(
    rewards: list[float], context_ids: list[str], beta: float, sigma_floor: float = SIGMA_FLOOR
) -> list[float]:
    if beta <= 0:
        raise ValueError("beta must be positive")
    return [math.exp(r / beta) for r in standardize_per_context(rewards, context_ids, sigma_floor)]


def effective_sample_size(weights: list[float]) -> float:
    """(sum w)^2 / sum w^2 — a diagnostic for how peaked the weights are."""
    s1 = sum(weights)
    s2 = sum(w * w for w in weights)
    return s1 * s1 / s2 if s2 > 0 else 0.0

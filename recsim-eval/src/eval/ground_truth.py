"""Ground-truth evaluation: the true expected engagement of a recommender,
computed from the hidden latent factors (the oracle). This is the target a good
simulator must reproduce — never available to recommenders or simulators.
"""
from __future__ import annotations
import numpy as np
from ..data import Dataset


def true_click_prob(data: Dataset, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
    """Clean (noise-free) expected click prob for equal-length (user,item) arrays."""
    p = data.user_factors[user_ids]; q = data.item_factors[item_ids]
    dot = np.einsum("ij,ij->i", p, q)
    u = dot + float(data.cfg["quality_weight"]) * data.item_quality[item_ids] \
          + float(data.cfg["click_bias"])
    return 1.0 / (1.0 + np.exp(-np.clip(u, -30, 30)))


def true_engagement(data: Dataset, recs: np.ndarray, user_ids: np.ndarray) -> float:
    """Mean true expected click prob over a [U, N] recommendation matrix."""
    U, N = recs.shape
    uu = np.repeat(user_ids, N)
    ii = recs.reshape(-1)
    return float(true_click_prob(data, uu, ii).mean())

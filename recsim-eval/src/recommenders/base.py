"""Recommender interface. A recommender trains on observable logs only and
produces top-N lists. The evaluator (not the recommender) judges quality."""
from __future__ import annotations
import numpy as np
from ..data import Dataset


class Recommender:
    name: str = "base"

    def fit(self, data: Dataset) -> "Recommender":
        self.data = data
        self.seen = data.seen_sets()
        return self

    def scores(self, user_ids: np.ndarray) -> np.ndarray:
        """Return a [len(user_ids), n_items] score matrix. Override this."""
        raise NotImplementedError

    def recommend(self, user_ids: np.ndarray, N: int = 10,
                  exclude_seen: bool = True) -> np.ndarray:
        S = self.scores(user_ids).copy()
        if exclude_seen:
            for row, u in enumerate(user_ids):
                seen = self.seen[u]
                if seen:
                    S[row, list(seen)] = -np.inf
        # top-N via argpartition then sort
        N = min(N, S.shape[1])
        idx = np.argpartition(-S, N - 1, axis=1)[:, :N]
        order = np.argsort(-np.take_along_axis(S, idx, axis=1), axis=1)
        return np.take_along_axis(idx, order, axis=1)

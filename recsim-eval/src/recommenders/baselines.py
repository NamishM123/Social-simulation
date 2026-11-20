"""Non-personalized and lightly-personalized baselines."""
from __future__ import annotations
import numpy as np
from .base import Recommender
from ..data import Dataset


class RandomRec(Recommender):
    name = "Random"
    def fit(self, data: Dataset):
        super().fit(data); self.rng = np.random.default_rng(0); return self
    def scores(self, user_ids):
        return self.rng.random((len(user_ids), self.data.n_items))


class Popularity(Recommender):
    name = "Popularity"
    def fit(self, data: Dataset):
        super().fit(data)
        u, i = data.clicks()
        counts = np.bincount(i, minlength=data.n_items).astype(float)
        self.pop = np.log1p(counts)  # same for every user
        return self
    def scores(self, user_ids):
        return np.tile(self.pop, (len(user_ids), 1))


class GenreMatch(Recommender):
    """Recommend popular items in the user's most-clicked genre."""
    name = "GenreMatch"
    def fit(self, data: Dataset):
        super().fit(data)
        u, i = data.clicks()
        counts = np.bincount(i, minlength=data.n_items).astype(float)
        self.item_pop = np.log1p(counts)
        self.genre = data.item_genre
        G = int(data.cfg["n_genres"])
        # per-user click distribution over genres
        self.user_genre_pref = np.full((data.n_users, G), 0.1)
        for uu, ii in zip(u, i):
            self.user_genre_pref[uu, self.genre[ii]] += 1.0
        self.user_genre_pref /= self.user_genre_pref.sum(1, keepdims=True)
        return self
    def scores(self, user_ids):
        # score = popularity weighted by how much the user likes that item's genre
        genre_w = self.user_genre_pref[user_ids][:, self.genre]  # [U, n_items]
        return genre_w * self.item_pop[None, :]

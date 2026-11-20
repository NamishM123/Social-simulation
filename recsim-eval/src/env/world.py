"""Semi-synthetic recommendation world with known ground truth.

The world has latent user and item factors. We KNOW them, so for any
recommender we can compute its true expected engagement (the oracle target
a simulator must reproduce). Users are drawn from K latent behavioral
clusters — the ground-truth "personas" that grounded simulators try to recover.

Nothing here is random across runs unless you change the seed: the whole world
is a deterministic function of WorldConfig.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class WorldConfig:
    n_users: int = 2000
    n_items: int = 1000
    n_factors: int = 16          # latent preference dimensions
    n_clusters: int = 6          # ground-truth behavioral segments (personas)
    n_genres: int = 10           # observable item metadata
    cluster_spread: float = 0.45 # within-cluster user spread (lower = tighter segments)
    quality_weight: float = 0.8  # how much item "quality"/popularity drives clicks
    utility_noise: float = 0.5   # irreducible noise in a single click decision
    click_bias: float = -0.5     # global intercept on the click logit
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class World:
    """Latent-factor world. Call .true_click_prob(users, items) for ground truth."""

    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self.rng = rng
        D, K = cfg.n_factors, cfg.n_clusters

        # ---- ground-truth behavioral clusters (personas) ----
        # cluster centroids spread out in latent space
        self.cluster_centroids = rng.normal(0, 1.0, size=(K, D)) * 1.4
        # assign users to clusters (uneven sizes -> realistic traffic mix)
        cluster_probs = rng.dirichlet(np.ones(K) * 2.0)
        self.user_cluster = rng.choice(K, size=cfg.n_users, p=cluster_probs)
        self.cluster_probs = cluster_probs
        # user factor = its cluster centroid + within-cluster noise
        self.user_factors = (
            self.cluster_centroids[self.user_cluster]
            + rng.normal(0, cfg.cluster_spread, size=(cfg.n_users, D))
        )

        # ---- items: latent factors + observable metadata ----
        self.item_factors = rng.normal(0, 1.0, size=(cfg.n_items, D))
        # item quality (a popularity/appeal scalar) — heavy-tailed
        self.item_quality = rng.normal(0, 1.0, size=cfg.n_items)
        # observable genre: derived from latent factors so metadata is informative
        # (project items onto G random directions, take argmax as primary genre)
        genre_dirs = rng.normal(0, 1, size=(cfg.n_genres, D))
        genre_score = self.item_factors @ genre_dirs.T
        self.item_genre = genre_score.argmax(axis=1)

    # ---------- ground truth ----------
    def utility(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        """Latent utility of (user, item) pairs (elementwise over equal-length arrays)."""
        p = self.user_factors[user_idx]
        q = self.item_factors[item_idx]
        dot = np.einsum("ij,ij->i", p, q)
        return dot + self.cfg.quality_weight * self.item_quality[item_idx]

    def true_click_prob(self, user_idx: np.ndarray, item_idx: np.ndarray,
                        noisy: bool = False) -> np.ndarray:
        """Expected click probability under the true model.

        noisy=False gives the clean expected prob (used for ground-truth
        evaluation); noisy=True injects per-decision noise (used when *sampling*
        a click during data logging)."""
        u = self.utility(user_idx, item_idx) + self.cfg.click_bias
        if noisy:
            u = u + self.rng.normal(0, self.cfg.utility_noise, size=u.shape)
        return _sigmoid(u)

    def item_features(self) -> np.ndarray:
        """Observable item features a recommender/agent may use (one-hot genre
        + standardized quality). No latent factors leak here."""
        G = self.cfg.n_genres
        onehot = np.eye(G)[self.item_genre]
        q = (self.item_quality - self.item_quality.mean()) / (self.item_quality.std() + 1e-8)
        return np.concatenate([onehot, q[:, None]], axis=1)

    # convenience
    @property
    def n_users(self): return self.cfg.n_users
    @property
    def n_items(self): return self.cfg.n_items


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

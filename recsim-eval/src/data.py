"""Load a generated world + logs into a convenient container.

`factors`/`user_cluster` are the hidden ground truth — only the oracle evaluator
and the oracle agent may touch them. Everything else (logs, item_features) is
what a real recommender or simulator would see.
"""
from __future__ import annotations
from dataclasses import dataclass
import json, os
import numpy as np


@dataclass
class Dataset:
    cfg: dict
    # observable
    imp_user: np.ndarray
    imp_item: np.ndarray
    imp_click: np.ndarray
    imp_t: np.ndarray
    item_features: np.ndarray
    # hidden ground truth (eval/oracle only)
    user_factors: np.ndarray
    item_factors: np.ndarray
    item_quality: np.ndarray
    item_genre: np.ndarray
    user_cluster: np.ndarray

    @property
    def n_users(self) -> int: return int(self.cfg["n_users"])
    @property
    def n_items(self) -> int: return int(self.cfg["n_items"])

    # ---- observable derived views ----
    def seen_sets(self) -> list[set]:
        seen = [set() for _ in range(self.n_users)]
        for u, i in zip(self.imp_user, self.imp_item):
            seen[u].add(int(i))
        return seen

    def clicks(self):
        """Positive (user,item) clicks — the implicit-feedback training signal."""
        m = self.imp_click == 1
        return self.imp_user[m], self.imp_item[m]


def load(path: str) -> Dataset:
    w = np.load(os.path.join(path, "world.npz"))
    l = np.load(os.path.join(path, "logs.npz"))
    with open(os.path.join(path, "config.json")) as f:
        cfg = json.load(f)
    return Dataset(
        cfg=cfg,
        imp_user=l["imp_user"], imp_item=l["imp_item"],
        imp_click=l["imp_click"], imp_t=l["imp_t"],
        item_features=w["item_features"],
        user_factors=w["user_factors"], item_factors=w["item_factors"],
        item_quality=w["item_quality"], item_genre=w["item_genre"],
        user_cluster=w["user_cluster"],
    )

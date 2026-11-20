"""Matrix factorization trained with BPR (Bayesian Personalized Ranking) on
implicit click feedback. Personalized — should top the ground-truth ranking.
Capacity (n_factors) and training length control its quality, giving us a
spectrum of recommender strengths for the fidelity experiment.
"""
from __future__ import annotations
import numpy as np
import torch
from .base import Recommender
from ..data import Dataset


class MFBPR(Recommender):
    def __init__(self, n_factors: int = 32, epochs: int = 40, lr: float = 0.05,
                 reg: float = 1e-5, neg: int = 2, seed: int = 0, name: str | None = None):
        self.k = n_factors; self.epochs = epochs; self.lr = lr
        self.reg = reg; self.neg = neg; self.seed = seed
        self.name = name or f"MF-BPR(k={n_factors})"

    def fit(self, data: Dataset):
        super().fit(data)
        torch.manual_seed(self.seed)
        g = torch.Generator().manual_seed(self.seed)
        nU, nI = data.n_users, data.n_items
        u, i = data.clicks()
        u = torch.as_tensor(u, dtype=torch.long); i = torch.as_tensor(i, dtype=torch.long)

        U = torch.nn.Embedding(nU, self.k); V = torch.nn.Embedding(nI, self.k)
        torch.nn.init.normal_(U.weight, std=0.1); torch.nn.init.normal_(V.weight, std=0.1)
        opt = torch.optim.Adam([U.weight, V.weight], lr=self.lr, weight_decay=self.reg)

        n = u.shape[0]; bs = 4096
        for ep in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            for s in range(0, n, bs):
                b = perm[s:s + bs]
                ub, ib = u[b], i[b]
                for _ in range(self.neg):
                    jb = torch.randint(0, nI, (ub.shape[0],), generator=g)
                    pu = U(ub); pi = V(ib); pj = V(jb)
                    x = (pu * (pi - pj)).sum(1)
                    loss = -torch.nn.functional.logsigmoid(x).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
        self.U = U.weight.detach().numpy(); self.V = V.weight.detach().numpy()
        return self

    def scores(self, user_ids):
        return self.U[user_ids] @ self.V.T

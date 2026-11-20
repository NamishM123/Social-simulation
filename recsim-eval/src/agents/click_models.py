"""User-agent simulators. Each predicts P(click | user, item) from OBSERVABLE
data only (logs + item features), then estimates a recommender's engagement as
the mean predicted click over its recommendations.

Fidelity ladder:
  OracleAgent     - uses hidden true utility (upper bound).
  GlobalAgent     - one click model for everyone (no personalization).
  RandomPersona   - users split into G random groups, per-group click model.
  GroundedPersona - personas inferred by clustering observable behavior.
  LogisticMF      - learned per-user embeddings (strong personalized reference).

The headline test: does GroundedPersona reproduce the ground-truth model ranking
better than RandomPersona / GlobalAgent, approaching the oracle?
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from ..data import Dataset
from ..eval.ground_truth import true_click_prob


class Agent:
    name = "agent"
    def fit(self, data: Dataset): self.data = data; return self
    def click_prob(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class OracleAgent(Agent):
    name = "Oracle"
    def click_prob(self, user_ids, item_ids):
        return true_click_prob(self.data, user_ids, item_ids)


def _fit_group_lr(X, y):
    """Logistic click model for one group; robust to degenerate labels."""
    if len(np.unique(y)) < 2:
        rate = float(y.mean()) if len(y) else 0.5
        return ("const", np.clip(rate, 1e-3, 1 - 1e-3))
    lr = LogisticRegression(max_iter=200, C=1.0)
    lr.fit(X, y)
    return ("lr", lr)


def _group_prob(model, X):
    kind, m = model
    if kind == "const":
        return np.full(X.shape[0], m)
    return m.predict_proba(X)[:, 1]


class _PersonaAgent(Agent):
    """Shared machinery: assign each user to a group, fit a per-group click model
    on item features, predict via the user's group model."""
    def __init__(self, n_groups: int, seed: int = 0):
        self.G = n_groups; self.seed = seed; self.ari = None
    def _assign(self, data: Dataset) -> np.ndarray:
        raise NotImplementedError
    def fit(self, data: Dataset):
        super().fit(data)
        self.feat = data.item_features
        self.group = self._assign(data)                      # [n_users]
        imp_group = self.group[data.imp_user]
        X = self.feat[data.imp_item]; y = data.imp_click
        self.models = {}
        for g in range(self.G):
            m = imp_group == g
            self.models[g] = _fit_group_lr(X[m], y[m]) if m.any() else ("const", float(y.mean()))
        return self
    def click_prob(self, user_ids, item_ids):
        X = self.feat[item_ids]; g = self.group[user_ids]
        out = np.empty(len(item_ids))
        for grp in np.unique(g):
            mask = g == grp
            out[mask] = _group_prob(self.models[int(grp)], X[mask])
        return out


class GlobalAgent(_PersonaAgent):
    name = "Global (no persona)"
    def __init__(self): super().__init__(n_groups=1)
    def _assign(self, data): return np.zeros(data.n_users, dtype=int)


class RandomPersona(_PersonaAgent):
    def __init__(self, n_groups: int, seed: int = 0):
        super().__init__(n_groups, seed); self.name = f"Random personas (G={n_groups})"
    def _assign(self, data):
        return np.random.default_rng(self.seed).integers(0, self.G, size=data.n_users)


class GroundedPersona(_PersonaAgent):
    """Infer personas by clustering each user's OBSERVABLE behavior, then fit a
    per-persona click model. Two grounding representations:
      rep="genre"     - user's genre-click histogram (crude, cheap).
      rep="embedding" - learned behavioral embedding passed in via user_repr
                        (rich; recovers latent segments far better).
    The ablation 'does grounding quality (ARI) drive fidelity?' uses both."""
    def __init__(self, n_groups: int, rep: str = "genre",
                 user_repr: np.ndarray | None = None, seed: int = 0):
        super().__init__(n_groups, seed)
        self.rep = rep; self.user_repr = user_repr
        self.name = f"Grounded ({rep}, G={n_groups})"
    def _assign(self, data):
        if self.rep == "embedding":
            if self.user_repr is None:
                raise ValueError("rep='embedding' needs user_repr")
            prof = self.user_repr
        else:
            G = int(data.cfg["n_genres"])
            prof = np.full((data.n_users, G), 0.1)
            cu, ci = data.clicks()
            for u, i in zip(cu, ci):
                prof[u, data.item_genre[i]] += 1.0
            prof /= prof.sum(1, keepdims=True)
        km = KMeans(n_clusters=self.G, random_state=self.seed, n_init=10)
        lab = km.fit_predict(prof)
        self.ari = float(adjusted_rand_score(data.user_cluster, lab))  # recovery quality
        return lab


class LogisticMF(Agent):
    """Per-user + per-item logistic factorization fit on impressions — a strong
    personalized reference below the oracle."""
    name = "LogisticMF"
    def __init__(self, k: int = 16, epochs: int = 25, lr: float = 0.05, seed: int = 0):
        self.k = k; self.epochs = epochs; self.lr = lr; self.seed = seed
    def fit(self, data: Dataset):
        super().fit(data)
        import torch
        torch.manual_seed(self.seed); g = torch.Generator().manual_seed(self.seed)
        nU, nI = data.n_users, data.n_items
        u = torch.as_tensor(data.imp_user, dtype=torch.long)
        i = torch.as_tensor(data.imp_item, dtype=torch.long)
        y = torch.as_tensor(data.imp_click, dtype=torch.float32)
        U = torch.nn.Embedding(nU, self.k); V = torch.nn.Embedding(nI, self.k)
        ub = torch.nn.Parameter(torch.zeros(nU)); ib = torch.nn.Parameter(torch.zeros(nI))
        b = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(U.weight, std=0.1); torch.nn.init.normal_(V.weight, std=0.1)
        opt = torch.optim.Adam(list(U.parameters()) + list(V.parameters()) + [ub, ib, b], lr=self.lr, weight_decay=1e-5)
        n = u.shape[0]; bs = 8192; bce = torch.nn.BCEWithLogitsLoss()
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            for s in range(0, n, bs):
                bq = perm[s:s+bs]; uu, ii, yy = u[bq], i[bq], y[bq]
                logit = (U(uu) * V(ii)).sum(1) + ub[uu] + ib[ii] + b
                loss = bce(logit, yy)
                opt.zero_grad(); loss.backward(); opt.step()
        self.U = U.weight.detach().numpy(); self.V = V.weight.detach().numpy()
        self.ub = ub.detach().numpy(); self.ib = ib.detach().numpy(); self.b = float(b.detach())
        return self
    def click_prob(self, user_ids, item_ids):
        logit = (self.U[user_ids] * self.V[item_ids]).sum(1) + self.ub[user_ids] + self.ib[item_ids] + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))


def simulator_engagement(agent: Agent, recs: np.ndarray, user_ids: np.ndarray) -> float:
    U, N = recs.shape
    uu = np.repeat(user_ids, N); ii = recs.reshape(-1)
    return float(agent.click_prob(uu, ii).mean())

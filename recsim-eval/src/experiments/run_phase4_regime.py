"""Phase 4: WHEN does persona grounding help simulator fidelity?

Sweep `quality_weight` — how much clicks are driven by global item popularity
vs. individual user-item fit. Hypothesis:
  - high quality_weight: popularity dominates; a global simulator is already OK
    and grounding adds little.
  - low quality_weight: personal fit dominates; the simulator must model user
    preference, and grounded personas should clearly beat a global model.

Produces results/phase4_regime.{json,png}: fidelity (Kendall-tau) of each
simulator vs quality_weight, plus cluster-recovery ARI.

    python -m src.experiments.run_phase4_regime
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..env.world import World, WorldConfig
from ..env.generate import generate_logs, PRESETS
from ..data import Dataset
from ..recommenders.baselines import RandomRec, Popularity, GenreMatch
from ..recommenders.mf_bpr import MFBPR
from ..agents.click_models import (OracleAgent, GlobalAgent, GroundedPersona,
                                   LogisticMF, simulator_engagement)
from ..eval.ground_truth import true_engagement
from ..eval.fidelity import fidelity


def dataset_from_world(world: World, logs: dict) -> Dataset:
    return Dataset(
        cfg=world.cfg.to_dict(),
        imp_user=logs["imp_user"], imp_item=logs["imp_item"],
        imp_click=logs["imp_click"], imp_t=logs["imp_t"],
        item_features=world.item_features(),
        user_factors=world.user_factors, item_factors=world.item_factors,
        item_quality=world.item_quality, item_genre=world.item_genre,
        user_cluster=world.user_cluster,
    )


def suite():
    return [RandomRec(), Popularity(), GenreMatch(),
            MFBPR(8, 15, name="MF8"), MFBPR(16, 30, name="MF16"), MFBPR(32, 40, name="MF32")]


def run_world(qw: float, preset="small", seed=0):
    cfg = WorldConfig(seed=seed, quality_weight=qw, **PRESETS[preset])
    world = World(cfg)
    logs = generate_logs(world, impressions_per_user=30, seed=seed + 1)
    data = dataset_from_world(world, logs)
    users = np.arange(data.n_users)
    G = cfg.n_clusters

    recs_list, gt = [], []
    for rec in suite():
        rec.fit(data); r = rec.recommend(users, N=10)
        recs_list.append(r); gt.append(true_engagement(data, r, users))
    gt = np.array(gt)

    lmf = LogisticMF().fit(data)
    agents = {
        "Oracle": OracleAgent().fit(data),
        "Global": GlobalAgent().fit(data),
        "Grounded": GroundedPersona(G, rep="embedding", user_repr=lmf.U).fit(data),
        "LogisticMF": lmf,
    }
    out = {"quality_weight": qw, "ctr": float(data.imp_click.mean())}
    for name, ag in agents.items():
        sim = np.array([simulator_engagement(ag, r, users) for r in recs_list])
        fid = fidelity(sim, gt)
        out[name] = fid["kendall_tau"]
        if name == "Grounded":
            out["ari"] = ag.ari
    return out


def main():
    grid = [0.0, 0.2, 0.4, 0.8, 1.2, 1.8]
    seeds = [0, 1, 2, 3]
    rows = []
    keys = ["Global", "Grounded", "LogisticMF", "Oracle", "ari", "ctr"]
    print(f"(mean over {len(seeds)} seeds)")
    print(f"{'qw':>5s} {'CTR':>5s} {'Global':>7s} {'Grounded':>9s} {'LogMF':>7s} {'Oracle':>7s} {'ARI':>5s}")
    for qw in grid:
        runs = [run_world(qw, seed=s) for s in seeds]
        agg = {k: float(np.mean([r[k] for r in runs])) for k in keys}
        agg["quality_weight"] = qw
        agg.update({k + "_std": float(np.std([r[k] for r in runs])) for k in ["Global", "Grounded", "LogisticMF"]})
        rows.append(agg)
        print(f"{qw:5.1f} {agg['ctr']:5.2f} {agg['Global']:7.2f} {agg['Grounded']:9.2f} "
              f"{agg['LogisticMF']:7.2f} {agg['Oracle']:7.2f} {agg['ari']:5.2f}")

    os.makedirs("results", exist_ok=True)
    with open("results/phase4_regime.json", "w") as f:
        json.dump(rows, f, indent=2)

    qw = [r["quality_weight"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, style in [("Oracle", "k--"), ("LogisticMF", "o-"), ("Grounded", "s-"), ("Global", "^-")]:
        y = [r[name] for r in rows]
        err = [r.get(name + "_std", 0) for r in rows]
        ax[0].errorbar(qw, y, yerr=err, fmt=style, capsize=3, label=name)
    ax[0].set_xlabel("quality_weight  (popularity ← → personal fit)")
    ax[0].set_ylabel("Fidelity (Kendall-τ vs ground truth)")
    ax[0].set_title("When does the simulator rank recommenders correctly?")
    ax[0].axhline(0, color="gray", lw=0.6); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].invert_xaxis()
    ax[1].plot(qw, [r["Grounded"] for r in rows], "s-", label="Grounded fidelity (τ)")
    ax[1].plot(qw, [r["ari"] for r in rows], "d-", color="green", label="Cluster recovery (ARI)")
    ax[1].set_xlabel("quality_weight"); ax[1].set_title("Grounding: recovery vs fidelity")
    ax[1].legend(); ax[1].grid(alpha=.3); ax[1].invert_xaxis()
    fig.tight_layout(); fig.savefig("results/phase4_regime.png", dpi=130)
    print("\nsaved -> results/phase4_regime.json, results/phase4_regime.png")


if __name__ == "__main__":
    main()

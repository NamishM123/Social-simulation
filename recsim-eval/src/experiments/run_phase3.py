"""Phase 3: the headline experiment.

Rank the recommender suite with each agent simulator, and measure how well each
simulator's ranking matches the ground-truth ranking. Tests the hypothesis:
grounded personas > random personas > no persona, approaching the oracle.

    python -m src.experiments.run_phase3 --data data/world_small
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from ..data import load
from ..recommenders.baselines import RandomRec, Popularity, GenreMatch
from ..recommenders.mf_bpr import MFBPR
from ..agents.click_models import (OracleAgent, GlobalAgent, RandomPersona,
                                   GroundedPersona, LogisticMF, simulator_engagement)
from ..eval.ground_truth import true_engagement
from ..eval.fidelity import fidelity


def build_suite():
    return [RandomRec(), Popularity(), GenreMatch(),
            MFBPR(8, 15, name="MF-BPR(k=8,short)"),
            MFBPR(16, 40, name="MF-BPR(k=16)"),
            MFBPR(32, 60, name="MF-BPR(k=32)")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--out", default="results/phase3_fidelity.json")
    args = ap.parse_args()

    data = load(args.data)
    users = np.arange(data.n_users)
    G = int(data.cfg["n_clusters"])

    # ---- recommenders: recs + ground-truth engagement ----
    print("Training recommender suite...")
    model_names, recs_list, gt = [], [], []
    for rec in build_suite():
        rec.fit(data)
        r = rec.recommend(users, N=args.N)
        recs_list.append(r); model_names.append(rec.name)
        gt.append(true_engagement(data, r, users))
    gt = np.array(gt)

    # ---- agents (fidelity ladder) ----
    # a learned behavioral embedding (observable) for the rich grounding variant
    lmf = LogisticMF().fit(data)
    agents = [OracleAgent(), GlobalAgent(), RandomPersona(G),
              GroundedPersona(G, rep="genre"),
              GroundedPersona(G, rep="embedding", user_repr=lmf.U),
              lmf]
    print("Fitting agent simulators & scoring...\n")
    results = []
    for ag in agents:
        if ag is not lmf:
            ag.fit(data)
        sim = np.array([simulator_engagement(ag, r, users) for r in recs_list])
        fid = fidelity(sim, gt)
        row = {"agent": ag.name, **fid, "sim_scores": sim.tolist()}
        if getattr(ag, "ari", None) is not None:
            row["cluster_recovery_ari"] = ag.ari
        results.append(row)

    # ---- report ----
    gt_order = [model_names[i] for i in np.argsort(-gt)]
    print("GROUND-TRUTH ranking:", " > ".join(gt_order), "\n")
    print(f"{'simulator':24s} {'kendall_tau':>11s} {'spearman':>9s} {'pairwise':>9s} {'top1':>5s}  recovery")
    print("-" * 78)
    for r in sorted(results, key=lambda x: -x["kendall_tau"]):
        ari = f"  ARI={r['cluster_recovery_ari']:.2f}" if "cluster_recovery_ari" in r else ""
        print(f"{r['agent']:24s} {r['kendall_tau']:>11.3f} {r['spearman_rho']:>9.3f} "
              f"{r['pairwise_acc']:>9.2f} {r['top1_correct']:>5d}{ari}")

    out = args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"data": args.data, "N": args.N, "n_clusters": G,
                   "ground_truth_order": gt_order, "gt_scores": gt.tolist(),
                   "models": model_names, "results": results}, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()

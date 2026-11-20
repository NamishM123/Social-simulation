"""Phase 2: train the recommender suite and rank it by TRUE engagement.

This produces the ground-truth ranking of models — the thing the agent
simulators (Phase 3) will try to reproduce without the hidden factors.

    python -m src.experiments.run_phase2 --data data/world_small
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from ..data import load
from ..recommenders.baselines import RandomRec, Popularity, GenreMatch
from ..recommenders.mf_bpr import MFBPR
from ..eval.ground_truth import true_engagement


def build_suite():
    return [
        RandomRec(),
        Popularity(),
        GenreMatch(),
        MFBPR(n_factors=8,  epochs=15, name="MF-BPR(k=8,short)"),
        MFBPR(n_factors=16, epochs=40, name="MF-BPR(k=16)"),
        MFBPR(n_factors=32, epochs=60, name="MF-BPR(k=32)"),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = load(args.data)
    users = np.arange(data.n_users)
    rows = []
    for rec in build_suite():
        rec.fit(data)
        recs = rec.recommend(users, N=args.N)
        eng = true_engagement(data, recs, users)
        rows.append({"model": rec.name, "true_engagement": eng})
        print(f"{rec.name:22s}  true_engagement@{args.N} = {eng:.4f}")

    rows.sort(key=lambda r: -r["true_engagement"])
    print("\nGROUND-TRUTH RANKING (best → worst):")
    for rank, r in enumerate(rows, 1):
        print(f"  {rank}. {r['model']:22s} {r['true_engagement']:.4f}")

    out = args.out or os.path.join("results", "phase2_ground_truth.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"N": args.N, "data": args.data, "ranking": rows}, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()

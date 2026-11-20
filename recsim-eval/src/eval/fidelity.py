"""Simulator fidelity: how well does a simulator's model ranking match the
ground-truth ranking? Rank correlation over the recommender suite."""
from __future__ import annotations
import numpy as np
from scipy.stats import kendalltau, spearmanr


def fidelity(sim_scores: np.ndarray, gt_scores: np.ndarray) -> dict:
    tau, _ = kendalltau(sim_scores, gt_scores)
    rho, _ = spearmanr(sim_scores, gt_scores)
    # did the simulator pick the true best model?
    top1 = int(np.argmax(sim_scores) == np.argmax(gt_scores))
    # pairwise accuracy: fraction of model pairs ordered correctly
    n = len(gt_scores); correct = total = 0
    for a in range(n):
        for b in range(a + 1, n):
            total += 1
            if np.sign(sim_scores[a] - sim_scores[b]) == np.sign(gt_scores[a] - gt_scores[b]):
                correct += 1
    return {"kendall_tau": float(tau), "spearman_rho": float(rho),
            "top1_correct": top1, "pairwise_acc": correct / total}

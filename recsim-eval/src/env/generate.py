"""Generate a world + logged interaction data, and save to disk.

The logging policy is popularity-biased with a little exploration — like a real
production log, it over-represents already-popular items, so recommenders and
simulators must cope with selection bias. Only the logs are "observable"; the
latent factors are saved too but used solely by the oracle evaluator.

Usage:
    python -m src.env.generate --out data/world_small --preset small
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from .world import World, WorldConfig

PRESETS = {
    "tiny":  dict(n_users=500,  n_items=300,  n_clusters=4),
    "small": dict(n_users=2000, n_items=1000, n_clusters=6),
    "medium":dict(n_users=8000, n_items=4000, n_clusters=10),
}


def generate_logs(world: World, impressions_per_user: int = 30,
                  slate_size: int = 1, explore: float = 0.1, seed: int = 1):
    """Simulate a popularity-biased logging policy.

    For each user we draw `impressions_per_user` candidate items (mostly by
    popularity, `explore` fraction uniformly), then sample a click from the true
    noisy click model. Returns impression-level arrays (user, item, click, t).
    """
    cfg = world.cfg
    rng = np.random.default_rng(seed)
    # popularity sampling distribution over items (biased toward high quality)
    pop_logits = cfg.quality_weight * world.item_quality
    pop = np.exp(pop_logits - pop_logits.max()); pop /= pop.sum()
    uniform = np.full(cfg.n_items, 1.0 / cfg.n_items)
    sample_p = (1 - explore) * pop + explore * uniform
    sample_p /= sample_p.sum()

    users, items = [], []
    # vary engagement volume per user (heavy users vs light users)
    counts = rng.poisson(impressions_per_user, size=cfg.n_users).clip(5, None)
    for u in range(cfg.n_users):
        k = int(counts[u])
        cand = rng.choice(cfg.n_items, size=k, p=sample_p)
        users.append(np.full(k, u)); items.append(cand)
    user_arr = np.concatenate(users); item_arr = np.concatenate(items)
    # true (noisy) clicks
    click = (rng.random(user_arr.shape[0]) <
             world.true_click_prob(user_arr, item_arr, noisy=True)).astype(np.int8)
    # a time index per impression (global order; stable per user)
    t = np.arange(user_arr.shape[0])
    return dict(imp_user=user_arr, imp_item=item_arr, imp_click=click, imp_t=t)


def save(out_dir: str, world: World, logs: dict):
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, "world.npz"),
        user_factors=world.user_factors, item_factors=world.item_factors,
        item_quality=world.item_quality, item_genre=world.item_genre,
        user_cluster=world.user_cluster, cluster_centroids=world.cluster_centroids,
        item_features=world.item_features(),
    )
    np.savez_compressed(os.path.join(out_dir, "logs.npz"), **logs)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(world.cfg.to_dict(), f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="small", choices=list(PRESETS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--impressions", type=int, default=30)
    ap.add_argument("--explore", type=float, default=0.1)
    args = ap.parse_args()

    cfg = WorldConfig(seed=args.seed, **PRESETS[args.preset])
    world = World(cfg)
    logs = generate_logs(world, impressions_per_user=args.impressions,
                         explore=args.explore, seed=args.seed + 1)

    # quick stats
    ctr = logs["imp_click"].mean()
    n_imp = logs["imp_click"].shape[0]
    pos = int(logs["imp_click"].sum())
    print(f"world: {cfg.n_users} users, {cfg.n_items} items, {cfg.n_clusters} clusters")
    print(f"logs : {n_imp} impressions, {pos} clicks, CTR={ctr:.3f}")
    print(f"cluster sizes: {np.bincount(world.user_cluster)}")
    save(args.out, world, logs)
    print(f"saved -> {args.out}/")


if __name__ == "__main__":
    main()

# Generative-Agent User Simulators for Offline Recommender Evaluation

**Research question.** Can a population of generative *user agents* serve as a
reliable offline evaluator of recommender systems — i.e. do recommender models
ranked by an agent simulator rank the **same way** they would under real
held-out user data? And does **grounding** agent personas in real behavioral
clusters (vs. hand-written or no personas) close the sim-to-real gap?

This turns a practical question (can we A/B test recommenders *in simulation*
before shipping?) into a measurable scientific one, and tests a specific
hypothesis: **persona grounding improves simulator fidelity.**

## Why this setup

Studying *simulator fidelity* needs a world whose ground truth you know, so you
can measure exactly how wrong a simulator is. This is the approach of Google's
**RecSim** and Criteo's **RecoGym**: a controllable semi-synthetic environment
with latent user preferences. We adopt it, and add a **real-data adapter**
(MovieLens) you run where there's internet, for external validation.

## Method

1. **World** (`src/env`) — `N` users drawn from `K` latent behavioral clusters,
   `M` items with latent factors + metadata (genres). True utility
   `u(user,item) = ⟨p_user, q_item⟩ + quality`. We *know* these, so we can
   compute any recommender's true expected engagement.
2. **Logged data** — a logging policy generates interaction sequences (the only
   thing models and simulators get to see; the latent factors stay hidden).
3. **Recommenders** (`src/recommenders`) — Popularity, Matrix Factorization
   (BPR), and a sequential transformer (SASRec-lite). Trained on logged data.
4. **Ground-truth ranking** — rank recommenders by true expected engagement
   (the oracle). This is the target a good simulator must reproduce.
5. **Agent simulators** (`src/agents`) — predict engagement from logged data
   only, at increasing fidelity:
   - `no-persona` — one global click model.
   - `random-persona` — users split into random groups.
   - `grounded-persona` — personas inferred by clustering logged behavior.
   - `oracle` — true utility (upper bound on fidelity).
6. **Fidelity** (`src/eval`) — rank recommenders by each simulator, compare to
   the ground-truth ranking with Kendall-τ / Spearman-ρ. **Headline result:**
   does `grounded-persona` beat `random`/`no-persona`?

## Status

- [x] Phase 0 — scaffold, toolchain
- [ ] Phase 1 — semi-synthetic world + logged data  ← in progress
- [ ] Phase 2 — recommender baselines + ground-truth ranking
- [ ] Phase 3 — agent simulators + fidelity experiment
- [ ] Phase 4 — ablations (grounding, #agents, world params), sim-to-real gap
- [ ] Phase 5 — MovieLens adapter, figures, paper-style writeup

## Run

```bash
pip install -r requirements.txt
python -m src.env.generate --out data/world_small   # build a world + logs
```

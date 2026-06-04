# What Goes Wrong When You Train a World Model on Random Data

## A Diagnostic Study of Latent-World-Model Collapse in MiniGrid

### Abstract

Latent world models offer a principled approach to model-based reinforcement
learning: learn a compressed representation of observations, predict
transitions in that latent space, and plan via latent rollouts. In practice,
this pipeline is fragile. We train a series of increasingly sophisticated
world models (deterministic autoencoder, VICReg-regularized encoder, and a
full Recurrent State-Space Model) on random-policy data from MiniGrid
navigation tasks and evaluate a CEM planner that uses the learned model.

We document four distinct failure modes:
1. **Encoder collapse** — the latent representation collapses to a 0.01-
   diameter ball, making all states indistinguishable.
2. **Transition collapse** — the deterministic transition model learns
   $z_{t+1} \approx z_t$, achieving $10^{-6}$ MSE without learning dynamics.
3. **Reward-predictor collapse** — with sparse terminal rewards and a flat
   latent space, the reward predictor converges to a constant near zero.
4. **Planning distribution shift** — even after fixing the above collapses
   with an RSSM, the reward predictor (trained on posterior-encoded latents)
   assigns zero reward to prior-rollout latents at planning time.

Each collapse is diagnosed, fixed, and the residual bottleneck identified.
The final RSSM achieves stable 6-step open-loop rollouts (MSE 0.006--0.012)
but the CEM planner still fails to significantly outperform random
(30\% vs 23\% on Empty-5x5). A heuristic-reward ablation isolates the
world model's control-relevant prediction quality — not the reward
predictor — as the remaining bottleneck.

We conclude that pixel-level prediction quality is a poor proxy for
planning-relevant latent dynamics, and that data coverage (not architecture)
is the primary limiting factor at this scale. The full progression of
diagnoses, fixes, and residual failures constitutes a case study in the
fragility of latent-world-model pipelines.

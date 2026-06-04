## 1. Introduction

Latent world models represent a promising paradigm for model-based
reinforcement learning: compress high-dimensional observations into a
structured latent space, learn a dynamics model in that space, and use
it for planning or policy optimization. Dreamer \cite{hafner2020dreamer},
PlaNet \cite{hafner2019planet}, and TD-MPC \cite{hansen2022tdmpc}
have demonstrated impressive results across Atari, DM-Control, and
robotics domains.

The standard recipe is deceptively simple:
1. Collect experience (often from a random policy or replay buffer).
2. Train an encoder and decoder for observation compression.
3. Train a transition model to predict $z_{t+1} = f(z_t, a_t)$.
4. Train a reward predictor $\hat{r}_t = g(z_t)$.
5. Plan using CEM or MCTS with latent rollouts.

Step 5 rarely works on the first attempt. The research literature focuses
on the successes; the failure modes that dominate early iterations of
any world-model project are underdocumented.

**This paper documents a systematic progression of failure modes
encountered when building a latent-world-model planner from scratch on
MiniGrid navigation tasks.** The pipeline is textbook: CNN encoder, MLP
transition, CEM planner. It fails in four distinct ways, each with a
specific diagnosis and fix:

| Phase | Failure Mode | Symptom | Fix |
|-------|-------------|---------|-----|
| 1 | Transition collapse | $z_{t+1} \approx z_t$, MSE $10^{-6}$ | Residual transition |
| 2 | Encoder collapse | Latent ball diameter 0.01 | VICReg regularization |
| 3 | Reward collapse | Predictor outputs constant $\sim 0$ | Binary HER relabeling |
| 4 | Planning shift | Prior rollouts get zero reward | Heuristic diagnostic |

Only Phase 4 survives as the genuine residual challenge: even with a
working RSSM (stable 6-step rollouts, MSE 0.006--0.012), the planner
doesn't significantly beat random. A heuristic-reward ablation confirms
the world model's control-relevant precision -- not the reward predictor
-- is the bottleneck.

**Our contribution is not a new algorithm.** It is a detailed autopsy
of a well-known recipe, documenting exactly where it breaks, how each
break was diagnosed, and what remained. For researchers implementing
world models, this serves as a diagnostic checklist. For the field,
it reinforces a caution: pixel-level prediction quality is a poor proxy
for planning-relevant latent dynamics, and data coverage (not
architecture) is the primary limiting factor at practical scales.

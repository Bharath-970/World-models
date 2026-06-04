## 2. Related Work

### Latent World Models

The family of Recurrent State-Space Models (RSSMs) introduced in PlaNet
\cite{hafner2019planet} and extended in DreamerV1--V3 \cite{hafner2020dreamer,
hafner2021mastering, hafner2023dreamer} forms the dominant paradigm for
latent dynamics learning. These models maintain a deterministic recurrent
state $h_t$ (via GRU) and a stochastic latent $z_t \sim q(z_t \mid h_t, o_t)$,
with a learned prior $p(z_t \mid h_t)$ that enables open-loop rollouts.
Training minimizes an evidence lower bound (ELBO) combining reconstruction
loss, KL divergence between prior and posterior, and reward prediction.

TD-MPC \cite{hansen2022tdmpc} and TD-MPC2 \cite{hansen2023tdmpc2} replace
the decoder with temporal-difference learning in latent space, avoiding
pixel reconstruction entirely. While computationally efficient, their
success depends on rich task-reward feedback that may not exist in
goal-conditioned sparse-reward settings.

### Representation Collapse

Collapse is a well-known failure in self-supervised learning. VICReg
\cite{bardes2022vicreg} explicitly prevents it through variance and
covariance regularization. SimSiam \cite{chen2021simsiam} and BYOL
\cite{grill2020bootstrap} use architectural asymmetries (predictor,
stop-gradient, EMA) to the same end. In the world-model setting, collapse
appears as the encoder mapping all inputs to a small latent region,
allowing the transition model to trivially predict $z_{t+1} \approx z_t$.

### Planning with Learned Models

Cross-Entropy Method (CEM) planning \cite{rubinstein1999cross} and Model
Predictive Control (MPC) are standard for latent-planning. MuZero
\cite{schrittwieser2020mastering} learns a value-equivalent model and
plans via MCTS, avoiding pixel reconstruction entirely. A key challenge
is distribution shift: the planner evaluates actions using prior-rollout
latents, while the reward predictor was trained on posterior-encoded
latents from real observations. Dreamer addresses this by training a
value function $V(z)$ on prior-rollout data.

### Navigation in MiniGrid

MiniGrid \cite{chevalier2018minigrid} provides partially observable
grid-world tasks with discrete 7x7x3 image observations. The
observation space is simple (7 distinct pixel values) but the goal
structure (sparse terminal reward) makes it a challenging testbed for
world-model planning. Previous work has noted that random-policy data
on MiniGrid leads to poor dynamics learning \cite{sekani2021learning},
but a systematic decomposition of failure modes has been lacking.

# World Models Project — Full Explainer

## What Is This Project?

We tried to build a "world model" (a neural network that learns how a game works by watching it) and use it to plan actions in MiniGrid (a 5x5 grid where a triangle agent needs to reach a green goal square). The model learned to reconstruct images beautifully but was completely useless for planning. The project is about figuring out why.

## Basic Terms (Plain English)

**World model** — A neural net that predicts: given current observation + action, what happens next? Like learning the physics of a game by watching.

**Latent space** — Instead of predicting 147 pixels directly, we compress the image into a 64-number summary vector. The model works in this compressed space.

**RSSM (Recurrent State-Space Model)** — Dreamer's architecture. Two parallel representations: a deterministic one (GRU — like short-term memory) and a stochastic one (random sampling, handles uncertainty). A "prior" predicts without seeing observations. A "posterior" corrects using observations.

**CEM (Cross-Entropy Method)** — A planning algorithm. Sample 256 random action sequences, simulate them all, pick the best 16, refine, repeat 5 times.

**VICReg** — A training trick that forces the encoder to use all 64 latent dimensions instead of collapsing to just 1-2.

**HER (Hindsight Experience Replay)** — A relabeling trick: "You didn't reach the goal, but pretend that where you *were* was the goal, and learn from that."

**PPO** — An RL algorithm that learns a policy (what action to take in each state) through trial and error. We used it to collect better training data.

**Linear probe** — Train a single linear layer (simplest possible model) to predict something from latents. If it can't, the information isn't in the latents.

**R² = 0.04** — A measure from 0 to 1. 1 = perfect prediction. 0 = no better than guessing average. 0.04 = essentially zero.

---

## The Arc

### Phase 0: Preprocessing Bug
MiniGrid pixels are {0, 1, 2, 5, 8}, not {0..255} like photos. We divided by 255 by habit. The decoder outputs [0, 1], ground truth was [0, 0.031]. The loss measured the wrong thing. Fix: divide by 8 instead. Cost us a week.

### Phase 1: Simple Autoencoder — Encoder Collapse
CNN → 128-dim latent → decoder. MLP predicts next latent from current latent + action.

Result: MSE looked fine, but the encoder output a near-constant vector for all inputs. The decoder learned to render the average frame (mostly walls).

Why: Most pixels are walls (static). MSE is dominated by easy-to-predict regions. The encoder has no incentive to distinguish states.

Diagnostic: Pairwise distance between encoder outputs for different images = 0.0075 (on a ball of diameter ~0.01). Almost identical. This is "encoder collapse."

### Phase 1.5: Residual Transition
Even with a working encoder, the MLP learned z_{t+1} = z_t. MSE was 3e-7 — essentially zero. Multi-step error didn't grow with horizon (sounds good, but actually means the model outputs the same thing regardless).

Why: LayerNorm makes identity mapping a low-loss fixed point. The optimizer discovers that doing nothing is easier than learning dynamics.

Fix: z_{t+1} = z_t + 0.1 * delta(z_t, a_t). The model must learn a change, not the full state.

### Phase 2: VICReg Fixes the Encoder
VICReg adds two losses:
- Variance: penalizes if std < 1 (forces each dimension to vary)
- Covariance: penalizes correlated dimensions (forces independence)

Result: Pairwise distance went from 0.0075 to 12.7. Std from 0.0 to 0.84.

But: open-loop rollouts still drifted off after 2-3 steps. The deterministic model can't handle its own errors.

### Phase 3: RSSM (Dreamer Architecture)
Why RSSM: The deterministic model has no mechanism to handle uncertainty or correct errors. RSSM has:

- GRU: tracks history (short-term memory)
- Prior: predicts next latent from memory alone (used during planning)
- Posterior: predicts next latent from memory + observation (used during training)
- KL divergence: forces prior and posterior to agree. The posterior sees the observation, the prior doesn't. The KL loss forces the prior to learn to predict as well as the posterior.

Three-stage curriculum (full loss is unstable from scratch):
1. Stage A (AE): Train encoder + decoder + posterior. Weak KL to fixed N(0,I).
2. Stage B (Dynamics): Add GRU + learned prior. Stronger KL between prior and posterior.
3. Stage C (Overshoot): Add multi-step predictions at k=1, 3, 5 with weights 1.0, 0.5, 0.25.

Overshoot: A head that takes memory at time t and predicts the latent k steps ahead directly. Forces the model to maintain information across multiple steps.

Result: Recon 0.0092, KL 0.007, prior/post std ~1.0 (perfect unit Gaussian). Open-loop MSE flat at ~0.01 across 6 steps. RSSM works.

### Phase 4: The Planner Fails
CEM planner: sample 256 action sequences, simulate with prior, score with learned reward predictor, pick best 16, refine, repeat 5 times.

Reward predictor: Binary classifier. Takes [current_latent, goal_latent] as input, outputs probability of being at goal. Trained with 50/50 goal/non-goal pairs (HER).

Result: 30% success vs 23% random. Not statistically significant.

Heuristic reward test: Replace learned reward with pixel-space distance to goal (oracle signal). Result: 23% = same as random. The world model itself — not the reward predictor — is the bottleneck. Even with a perfect reward signal, the planner can't guide the agent.

### Phase 5: PPO Data Hypothesis
Hypothesis: Maybe we need better data. 1,000 random episodes, 39% success. Dreamer trains on millions.

What we did: Train PPO policy, then use it (70/30 mix with random) to collect 10,000 episodes. 281k transitions, 82.5% success. Retrain identical RSSM on this data.

Result: Better RSSM metrics (recon 0.0010, overshoot 0.43). Planner still at chance (10% learned, 30% heuristic vs 17-35% random). Data is not the answer.

This partially falsifies the data-coverage hypothesis.

### Phase 6: The Position Probe
What: Train a single linear layer to predict (x, y) agent position from RSSM latents.

Old-data RSSM: R² = 0.28 (modest position signal, but weak)
PPO-data RSSM: R² = 0.04 (essentially zero)

Why PPO data is worse: The PPO policy takes efficient center paths. Less spatial diversity in the data, so less incentive for the RSSM to encode position. Random data has more wandering = more positions seen = more position signal.

Why this is the most important result: Direct evidence for the root cause. Before this, we had speculation. After this, we had proof: a linear probe recovers almost no position information from latents.

---

## Key Decisions Explained

Q: Why MiniGrid and not Atari/DMC?
A: Simple 7x7 grids are cheap to train. Atari needs 10M frames. We wanted fast iteration.

Q: Why RSSM over a simpler deterministic model?
A: Deterministic drifts off after 2-3 steps. RSSM's KL mechanism forces the prior to stay close to the posterior.

Q: Why 3-stage curriculum?
A: Joint training with recon + KL + overshoot is too many gradients at once. Staged lets each component stabilize.

Q: Why VICReg over VAE/Beta-VAE?
A: VAE KL can still allow collapse with weak beta. VICReg directly penalizes collapse with a clear diagnostic (pairwise distance).

Q: Why binary HER over continuous HER?
A: Continuous (exp(-dist²/sigma²)) needs sigma tuning. Binary is simpler and achieves 98% accuracy.

Q: Why CEM over MCTS?
A: Standard in Dreamer/MBRL for continuous actions. MCTS is expensive. Random shooting is weaker. CEM is the standard.

Q: Why PPO 70/30 mix?
A: Pure PPO is too narrow (all successful paths look the same). Pure random wastes samples. 70/30 gives good coverage.

Q: Why linear probe instead of a neural network probe?
A: If a linear layer can't extract position, the info is not linearly accessible. Nonlinear probes could extract more but tell us less about representation quality.

Q: Why divide by 8 instead of 255?
A: MiniGrid pixels are {0,1,2,5,8}, not {0..255}. This was a bug that cost a week.

---

## The Final Narrative (2-Minute Version)

"We built a Dreamer-style world model for MiniGrid. It failed — not because it couldn't learn, but because it learned the wrong things.

We found 4 collapse modes: encoder collapse (all states map to the same latents), transition collapse (model learns z_{t+1} = z_t), reward collapse (classifier outputs constant zero), and planning shift (rollouts drift off). Each was fixed, and the final RSSM produced stable rollouts with MSE 0.009.

But the planner still performed at chance — even with an oracle reward signal. We tried collecting 10x more goal-directed data from PPO. No improvement.

The breakthrough was the position probe: train a linear layer to predict (x, y) from latents. R² = 0.04. The latent space simply doesn't encode where the agent is. It can render passable images, but can't track the one variable the planner needs.

The paper documents all 4 collapses, the data experiment, and the position probe — turning 'the planner didn't work' into 'the planner didn't work because the latent representation optimizes for reconstruction, not control.'"

## If You Get One Question

"So the RSSM was useless?"

No. The RSSM worked correctly. It learned to reconstruct images, maintain stable rollouts, and classify goals. It just didn't learn the information the planner needed. That's a specification problem, not a training failure. The reconstruction objective doesn't align with planning, and the position probe (R² = 0.04) proved it.

---

## One-Liners

Problem: Why does a world model that reconstructs images perfectly still fail at planning?

Answer: Because reconstruction and planning need different information. The model learns to predict pixels (mostly walls), not to track the agent's position. Position probe proves this: R² = 0.04.

Resume bullet: "Investigated model-based RL failure modes in MiniGrid, identifying representation collapse, reward collapse, transition drift, preprocessing errors, and control-state information loss (position probe R²=0.04)."

Most important number in the paper: R² = 0.04.

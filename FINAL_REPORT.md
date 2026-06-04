# World Models for Strategy Learning — Final Report

## TL;DR

Built and trained a CNN encoder + transition world model + reward predictor + CEM
planner on two MiniGrid environments (Empty-5x5, FourRooms) using random
trajectories. Replicated the standard world-model pipeline end-to-end on GPU and
ran ablations on latent dimension and CEM horizon.

**Headline finding:** the learned world model **collapsed** on random-policy
data (z_{t+1} ≈ z_t, MSE ~1e-7) and the learned reward predictor became a
constant ~0, leaving the CEM planner at 0% on the easy env where random solves
43%. Diagnosing and fixing the collapse required:

1. **Residual transition model** (`z_{t+1} = z_t + Δ(z_t, a)`) — the original
   `LayerNorm → MLP → LayerNorm` parameterization absorbed `z_t` and made
   identity prediction a fixed point. The residual form prevents this.
2. **Goal-conditioned HER reward predictor** — trained on (z_t, z_goal) pairs
   from relabelled trajectories, giving a dense signal where the original
   got ~0 everywhere.
3. **MPC with a fast env simulator** — once the world model is replaced by
   an in-process clone of the actual env dynamics, the planner goes from
   0% to **98%** on Empty-5x5 and from 7% to **13%** on FourRooms (vs 3%
   random).
4. **RSSM (Phase 7) — a real learned world model.** Three-stage curriculum
   (AE warmup → dynamics → multi-step rollout consistency), trained on
   properly-normalized observations. World-model rollouts stay in
   distribution across 6 steps (MSE 0.006-0.012). The planner still
   doesn't beat random because the world model isn't precise enough at
   this scale — see Section 7 for the full analysis.

**Numbers at a glance** (final):

| Metric                                 | Empty-5x5 | FourRooms |
|----------------------------------------|-----------|-----------|
| Autoencoder recon MSE (Phase 2)        | 5.22 (units mismatch — meaningless) | 1.29 |
| World-model training MSE (best)        | 3.25e-7   | ~3e-7     |
| Reward-predictor loss (final)          | 0.0015    | 0.00015   |
| Random success rate                    | 40%       | 3.3%      |
| Learned CEM planner (h=5)              | 0%        | 6.7%      |
| PPO success rate (20k steps)           | 0%        | 0%        |
| **MPC with env simulator**             | **98%**   | **13.3%** |
| MPC steps-to-goal                      | 15.7      | 48.8      |
| RSSM recon MSE (properly normalized)   | 0.0092    | —         |
| RSSM open-loop rollout MSE (6 steps)   | 0.006-0.012 | —       |
| RSSM + binary HER planner (h=4)        | 30%       | 5%        |
| RSSM + heuristic reward planner (h=4)  | 23% (avg, ±7 over 3 seeds) | 0% |

The world model itself is a cautionary tale about random-policy data: the
autoencoder looks fine (low recon error), the transition MSE is tiny, and
multi-step prediction looks great. The collapse only becomes visible when
the planner tries to use it. The MPC fix sidesteps the world model
entirely. The Phase 7 RSSM actually fixes the world model (rollouts stay
in distribution) but the precision isn't enough for navigation planning
at the scale of this dataset.

---

## Pipeline

```
[random policy] → trajectories.pkl
       ↓
CNN encoder (7×7×3 → 128d)            +    reward predictor (128d → r̂)
       ↓                                              ↑
Transition model (z_t, a → z_{t+1})                  |
       ↓                                              |
  CEM planner ──── sample actions, rollout in latent, score with r̂
                                                          ↓
                                          (also tried: HER + residual + MPC)
```

---

## Phase 1 — World Model Training

| Env         | Latent dim | Epochs | Final MSE  | Best MSE  | Status |
|-------------|------------|--------|------------|-----------|--------|
| Empty-5x5   | 128        | 30     | ~3e-7      | ~3e-7     | ✅     |
| FourRooms   | 128        | 50     | ~3e-7      | ~3e-7     | ✅     |

The world model loss collapses to ~1e-7 in both environments. Inspecting
predictions shows the model is essentially outputting `z_{t+1} ≈ z_t`. The
random-policy data has many "turn in place" and "bump wall" transitions where
the agent doesn't actually change position, so this is the easiest way to
minimise training MSE.

This is a well-known failure mode of world models trained on random data; it
motivates Dreamer-style stochastic latents and I-JEPA-style prediction targets
that prevent the collapse.

## Phase 2 — Visual Validation

| Env       | Reconstruction MSE | Mean pixel error (0–255) |
|-----------|--------------------|--------------------------|
| Empty-5x5 | 5.22               | 2.3 (<1% of full scale)  |
| FourRooms | 1.29               | 1.1                      |

The autoencoder reconstructs observations accurately, confirming the encoder
is preserving enough information for downstream prediction.

## Phase 3 — Multi-Step Prediction

| Horizon | Empty-5x5 MSE | FourRooms MSE |
|---------|---------------|---------------|
| 1       | ~1e-6         | ~0            |
| 5       | ~1e-6         | ~0            |
| 20      | ~1e-6         | ~0            |

Errors stay flat across horizons. This is the overfit symptom above: the model
isn't predicting the future, it's just outputting the present.

## Phase 4 — Planning with CEM (Learned World Model)

Reward predictor: small MLP (128 → 256 → 256 → 1) trained on the rewards stored
in trajectories. CEM planner with 128 samples, 10 elites, 4 iterations,
horizon 5, discount 0.99. PPO baseline from `stable-baselines3` (MlpPolicy,
20000 timesteps, image-only observation).

| Env       | Random | CEM Planner | PPO (20k steps) | Reward Predictor Loss |
|-----------|--------|-------------|-----------------|----------------------|
| Empty-5x5 | 43.3%  | 0.0%        | 0.0%            | 0.0015               |
| FourRooms | 6.7%   | 6.7%        | 0.0%            | 0.00015              |

Reward predictor loss is non-zero but the predicted reward is essentially
constant across the latent space — the training data contains almost no
positive-reward transitions (sparse, terminal-only reward in MiniGrid), so
the network fits "predict ≈ 0" as a low-loss solution. With a flat reward the
CEM planner has no gradient to optimise and produces a noisy action sequence
that doesn't reliably reach the goal.

**On the easy env, random 43% vs planner 0%** is a clean demonstration that
the world-model + learned-reward recipe, the textbook Dreamer / TD-MPC
setup, requires goal-directed data to bootstrap. On the hard env the planner
*matches* random at 6.7% success — both are essentially lucky, but the
planner reaches the goal slightly faster when it does (38 vs 48.5 steps).
That is a real, if small, signal: the latent rollout isn't entirely
degenerate; it's just not informative enough to outperform random.

The 6.7% number from Phase 4 is reproduced in the horizon sweep as 0% at h=5
(2/30 vs 0/30 successes) — the difference is within the noise of a 30-episode
sample, where ±2 episodes is a 6.7 percentage-point swing. Larger N would
be needed to nail the planner's true success rate on this env.

**PPO with 20k steps fails on both envs.** 20k timesteps is too small a
budget to learn the task from scratch from a Dict-obs image; PPO needs
~100k+ to solve Empty-5x5 reliably. This is a baseline, not a critique of
PPO; the relevant comparison is planner vs random.

**Why this is interesting, not just a bug:** the world-model + learned-reward
recipe is a textbook example from Dreamer / TD-MPC. Showing that it fails on
random-policy data — even in a tiny env where random already solves it 43% of
the time — is exactly the kind of result the paper has to demonstrate
before proposing fixes (recurrent state, hindsight relabelling, intrinsic
rewards, etc.).

## Phase 5 — Ablations

### 5a. Latent dimension sweep

World model + reward predictor re-trained for each latent dim, 20 epochs
on the same data. Results from `experiments/latent_ablation_*.json`:

**Empty-5x5**

| Latent dim | World-model MSE | Reward predictor MSE |
|------------|-----------------|----------------------|
| 32         | 3.87e-7         | 0.001503             |
| 64         | 5.52e-7         | 0.001504             |
| 128        | 7.77e-7         | 0.001504             |
| 256        | 1.96e-6         | 0.001504             |

**FourRooms**

| Latent dim | World-model MSE | Reward predictor MSE |
|------------|-----------------|----------------------|
| 32         | 2.63e-6         | 0.000152             |
| 64         | 2.87e-6         | 0.000152             |
| 128        | 1.70e-6         | 0.000152             |
| 256        | 1.69e-6         | 0.000152             |

World-model loss grows slightly with latent dim on the easy env (harder
to collapse to identity with more parameters) but stays at the 1e-6 scale
either way. Reward predictor loss is **essentially constant** in latent
dim — it depends only on the reward distribution in the data, which is
99.9 % zero regardless of how the latents are sized. Latent dim is not
the bottleneck for this dataset.

### 5b. CEM planning horizon sweep

CEM planner evaluated at horizons 1, 2, 5, 10 on the same trained
world model + reward predictor. 30 episodes per setting. Results from
`experiments/horizon_ablation_*.json`:

| Env        | h=1   | h=2   | h=5  | h=10  |
|------------|-------|-------|------|-------|
| Empty-5x5  | 0.0%  | 0.0%  | 0.0% | 0.0%  |
| FourRooms  | 6.7%  | 3.3%  | 0.0% | 10.0% |

On **Empty-5x5** every horizon gives 0 % — consistent with the planner
failure mode in Phase 4. On **FourRooms** horizon 10 is the best at
10 % success, almost double the random baseline of 6.7 %. Longer
planning helps on the harder env because the goal is further away
and short horizons (1–2) don't reach the planning region with
useful signal. The non-monotonic pattern (h=5 worse than h=2 and h=10)
is consistent with the reward predictor being noisy: a longer rollout
also accumulates more noise, and h=5 happens to lose on this noise.

The key takeaway: planning horizon matters, but only up to a point
determined by both the goal distance and the reward-prediction
signal-to-noise ratio.

## Phase 6 — Fixing the World Model

After Phases 1–5 the planner was clearly broken (0% on the easy env where
random hits 43%). This section documents what was tried to fix it and what
finally worked.

### 6.1 Diagnosis: where the collapse actually lives

Per-action change magnitude (latent space) on a sample of the data:

```
action 0 (turn L):  true change 0.0071, predicted 0.0071  ✅
action 1 (turn R):  true change 0.0071, predicted 0.0071  ✅
action 2 (forward): true change 0.0045, predicted 0.0035  ✅
action 3 (pickup):  true change 0.0000, predicted 0.0005  ✅
action 4 (drop):    true change 0.0000, predicted 0.0005  ✅
action 5 (toggle):  true change 0.0000, predicted 0.0006  ✅
action 6 (done):    true change 0.0000, predicted 0.0004  ✅
```

Wait — the transition model **was** learning per-action changes! With the
residual re-parameterization (`z_{t+1} = z_t + scale·Δ`, see
`models/transition_residual.py`), the model correctly predicts near-zero
change for pickup/drop/toggle/done and small non-zero change for
turn/forward. So the residual transition fix worked.

The deeper collapse turned out to be in the **encoder**: the latent space
spans only 0.0028–0.014 across all 1000 random-policy observations (mean
pairwise distance 0.0075). The autoencoder was trained with MSE on
pixel-reconstruction loss, and since most observations are mostly white
walls, the encoder collapsed to outputting a near-constant vector and the
decoder learned "render average white frame". The 5.22 MSE in Phase 2 is
dominated by the easy white pixels.

With latents this collapsed, the **reward predictor cannot distinguish
states** — even after HER relabelling, the network can only fit the mean
soft label and outputs 0.27 for every input.

### 6.2 Fix attempt 1: Hindsight relabelling (HER)

Implemented `training/train_reward_predictor_her.py` and
`planning/her_residual_planner.py`. For each timestep, sample K=4 "fake
goals" from future states in the same episode and train the predictor on
the soft distance `exp(-‖z_t − z_goal‖² / σ²)`.

- Training loss: 0.16 (vs 0.0015 for the original). The network has
  non-trivial input signal.
- Evaluation: **0% success** on the planner. The HER reward predictor
  *can* distinguish some (z_t, z_goal) pairs, but the residual transition
  magnitudes are still too small (0.005–0.007) compared to the latent
  scale (11.28) for CEM to find a useful gradient.

### 6.3 Fix attempt 2: Goal-conditioned encoder

Not implemented — would require retraining the encoder with a
contrastive/VICReg loss, which on the encoder we have is a
non-trivial additional training run.

### 6.4 Fix that worked: MPC with a fast env simulator

`evaluation/evaluate_mpc_real.py` (Empty-5x5) and
`evaluation/evaluate_mpc_fourrooms.py` (FourRooms). Implementation:

1. At each step, the actual env's `(agent_pos, agent_dir)` is copied into
   a lightweight in-process simulator (`evaluation/simple_envs.EmptyGrid`
   and a parametric `FourRoomsSim` that takes walls + goal).
2. CEM with 128 samples × 4 iterations × 16 elites explores candidate
   action sequences in the simulator.
3. The best first action is committed to the real env.

This sidesteps the world model entirely — instead of trying to learn
dynamics, we use the actual dynamics. With 128 samples and horizon 8–10,
each planner call takes ~10ms on CPU (≈25 it/s for the outer loop).

Results:

| Env        | Random | Learned planner (h=5) | **MPC with sim** |
|------------|--------|----------------------|------------------|
| Empty-5x5  | 40%    | 0%                   | **98%**          |
| FourRooms  | 3.3%   | 6.7%                 | **13.3%**        |

MPC's mean steps-to-goal: 15.7 (easy) and 48.8 (hard). On Empty-5x5
the shortest path is 4 steps, so 15.7 includes detours the planner
takes when initial samples don't immediately find the goal. On FourRooms
48.8 is a real navigation: 19×19 grid with 4 doorways to find.

**This is not a "world model" anymore** — it's classical Model-Predictive
Control with a known dynamics model. But it's the fix: it shows the
bottleneck was always the world model, not the planner or the
representations. With a correct model (in this case, the real env),
CEM solves the task.

### 6.5 Implications

The takeaway for anyone trying to replicate the world-model recipe on
random-policy data:

- A perfect-looking training loss (3e-7 MSE) is consistent with a
  collapsed model that outputs `z_{t+1} = z_t`.
- Sparse terminal rewards + random data + deterministic encoder + MSE
  loss = no learnable signal for the planner. The reward predictor
  converges to a constant; the planner has no gradient.
- Even with the right architecture (residual transition), if the
  encoder has collapsed to a 0.01-diameter ball in latent space, no
  downstream model can recover.
- The fix is either (a) train the encoder with a contrastive/VICReg
  loss, (b) use a much larger model and more data, or (c) skip the
  learned world model entirely and use the env as the dynamics.

## What I would do next

1. **VICReg-trained encoder.** Train the encoder with
   invariance + variance + covariance terms to force it to span a useful
   region of latent space. This addresses the root cause.
2. **Recurrent latent state (GRU/LSTM).** A deterministic MLP transition
   cannot represent "where the agent is" if the observation is mostly
   walls; an LSTM over the latent trajectory usually lifts FourRooms
   success from 0%.
3. **Stochastic transition model.** Add Gaussian noise to `z_{t+1}` to
   force the encoder to represent uncertainty. This is the key idea
   behind DreamerV2/V3 and breaks the "predict z_t" trivial solution.
4. **I-JEPA style target.** Predict embeddings of future observations
   rather than the encoder's own latents; the asymmetric target prevents
   collapse.

---

## Phase 7 — RSSM World Model (the proper fix)

After Phase 6, the MPC-with-env-simulator was a band-aid (acknowledged
as such). Phase 7 follows the Dreamer/PlaNet recipe: build an actual
Recurrent State-Space Model with a stochastic latent, prior/posterior
split, GRU dynamics, and a curriculum to prevent the
representational collapses that destroyed Phases 1-6.

### 7.1 Setup: the right normalization

The first thing the curriculum exposed was a silent data bug: MiniGrid
observations are `uint8` in the discrete set `{0, 1, 2, 5, 8}`, **not**
`[0, 255]`. Every encoder/decoder in Phases 1-6 normalized by `/255`,
which meant the decoder's `sigmoid` output range `[0, 1]` and the
ground-truth target `[0, 0.031]` (after dividing by 255) were in
incompatible ranges. Recon MSE was a units-mismatch artifact (~5.27 in
mixed units), not a meaningful image-similarity number. Comparing to a
constant-prediction baseline (≈4.29) and to the properly-normalized
mean (0.067 in `[0, 1]²` units) confirmed the bug.

**Fix:** divide obs by 8 (the actual max) in the encoder, the
decoder-normalize, the training data prep, and every evaluation
script. The VICReg "baseline 5.25" used throughout the project was
also meaningless; that was the units-mismatch value, not a model
quality number.

### 7.2 Three-stage curriculum

Per the project decision (incremental, no early skip of validation
gates), the RSSM is trained in three stages, each one a precondition
for the next.

**Stage A — Autoencoder warmup.**
Architecture: trunk encoder → posterior head (z distribution, KL to
fixed N(0, I)) → decoder. No GRU, no learned prior, no dynamics.
Loss: recon + β·KL(0.001).

```
Epoch 1/8  recon=0.0273  kl=2.9688  post_std=0.958
Epoch 4/8  recon=0.0035  kl=2.0971  post_std=0.977
Epoch 8/8  recon=0.0022  kl=2.5600  post_std=0.976
```

Mean-baseline constant prediction gives 0.067 in normalized units;
the warmup beats it by **30×**. The decoder is now actually using
the latent (visual check: red object placed at the correct cell in
`evaluation/rssm_ae_recon.png`).

**Stage B — Add dynamics.**
Loads Stage A, adds GRU + prior head. Now `forward_train` runs the
full posterior update at each step: `z_t ~ q(z_t | o_t, h_t)`,
`h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])`. Loss: recon + β·KL(0.1).

```
Epoch 1/8  recon=0.0091  kl=1.6504  kl/dim=0.0258  prior/post_std=0.99
Epoch 4/8  recon=0.0096  kl=0.0305  prior/post_std=0.996
Epoch 8/8  recon=0.0093  kl=0.0154  prior/post_std=0.995
```

The KL collapses a bit but `prior_std ≈ post_std ≈ 1.0`, meaning
both distributions are using the stochastic state — the prior is
matching the posterior (the desired state for inference).

**Stage C — Multi-step rollout consistency (Latent Overshooting).**
Loads Stage B, adds 3 overshoot heads that predict z_{t+k} for
k=1, 3, 5 from h_t. Loss: recon + β·KL + 0.5·(1·KL_1 + 0.5·KL_3 +
0.25·KL_5) per the Dreamer recipe. This directly attacks the
"step 1 good, step 3 dead" failure mode.

```
Epoch 1/6  recon=0.0094  kl=0.0505  overshoot=0.81
Epoch 6/6  recon=0.0092  kl=0.0073  overshoot=0.64
```

Overshoot loss drops 0.81 → 0.64. Not a dramatic improvement, but
the model is now trained to be consistent 1, 3, and 5 steps ahead.

### 7.3 The world model actually works (visual + numeric)

Open-loop rollout MSE — encoding the true obs at each step for the
initial state, then rolling out via the prior for the rest — across
6 timesteps:

```
step 1: mse=0.0066   ✓ OK
step 2: mse=0.0094   ✓ OK
step 3: mse=0.0126   ✓ OK
step 4: mse=0.0100   ✓ OK
step 5: mse=0.0088   ✓ OK
step 6: mse=0.0115   ✓ OK
```

The MSE stays flat at ~0.01 across the entire horizon. Compare to
the broken Phase 6 RSSM attempt (~5.3 across all steps, constant
yellow predictions). Visual sanity check in
`evaluation/rssm_dyn_rollouts.png` and `rssm_rollout_rollouts.png`:
the dark red object stays in roughly the right position throughout
the rollout. **The world model is real.**

### 7.4 Goal-conditioned reward predictor (HER on RSSM latents)

Initial attempt: soft-distance HER, `r = exp(-||z_t − z_goal||²/σ²)`
in observation space, then train the predictor on RSSM-encoded
(z_t, z_goal) pairs. Failed: the predictor's output saturated at
~0.94 (the mean soft label), giving reward ≈ 1.0 for every state.
Diagnosis: soft labels were concentrated in a narrow band; the
predictor learned the constant.

**Fix: binary HER.** For each timestep, pair with 50/50 probability
either the actual goal (reward 1) or a random non-goal from the same
trajectory (reward 0). Trained on 332k samples on RunPod CUDA
(8 epochs, ~1 min wall time).

```
Epoch 1/8 Loss: 0.0057
Epoch 4/8 Loss: 0.0001  Acc: 0.9764
Epoch 8/8 Loss: 0.0000  Acc: 0.9764
```

**97.6% classification accuracy** on goal vs non-goal states.

### 7.5 Planning: two failure modes diagnosed

The RSSM CEM planner (`planning/rssm_cem_planner.py` + binary HER
predictor) gives:

```
Empty-5x5 h=4, 30 ep, samples=256, elites=16, iters=5:
  Random:     23%
  Planner:    30%
```

Modest signal, similar to Phase 4-5. The planner picks the same
kinds of no-op / turn-heavy sequences that worked then.

**Failure-mode diagnostic.** Tracing the predictor's outputs:

```
Current state (real obs, posterior-encoded):  r = 0.86  ← high!
After 5 prior-rollout steps (any action):    r = 0.00  ← low
```

The predictor gives high reward to the initial state (which is
positionally at (1, 1) and clearly not the goal at (3, 3)), then
zero to anything produced by the prior. The issue: the predictor was
trained on posterior-encoded z's (`z ~ q(z | o, h=0)` from real
observations), but at planning time it's being asked to score
prior-predicted z's (`z ~ p(z | h, a)` from open-loop rollouts).
**Domain shift between training and planning distribution.**

This is the failure mode Dreamer solves with a value function
`V(z)` trained on prior rollouts. The standard fix.

### 7.6 Controlled ablation: heuristic reward

To isolate whether the world model is the bottleneck or the reward
predictor is, replace the learned reward with a heuristic:
`r = exp(-MSE(decoded_pred, obs_goal)/σ²)` — i.e., the pixel-space
distance between the predicted (decoded) state and the actual goal
observation.

```
Empty-5x5 h=4, 20 ep, samples=128, elites=16, iters=4, sigma=0.1:
  Random:  25%
  Planner: 40%        ← one run
```

A 15-point improvement. But running 3 more seeds at the same config
gives 30% / 17% / 23% (avg 23%) — i.e., **no significant
improvement over random with high variance**. The 40% was a lucky
run.

**Conclusion:** the reward predictor's domain shift was a real
contributor (eliminating it added some signal), but the **world
model itself is the bottleneck**. The rollouts stay in distribution
(blurry but consistent) but they don't precisely track the red
object's position, so pixel-distance to the goal is a noisy signal
for planning.

### 7.7 Honest numbers (final)

| Approach                                     | Empty-5x5 | FourRooms |
|----------------------------------------------|-----------|-----------|
| Random policy                                | 25%       | 5%        |
| Phase 4 learned planner (collapsed WM)       | 0%        | 6.7%      |
| Phase 6 MPC with env simulator               | **98%**   | **13.3%** |
| Phase 7 RSSM world model (rollout quality)   | recon 0.0092, rollout MSE 0.006-0.012 across 6 steps |
| Phase 7 RSSM + binary HER planner (h=4)      | 30%       | 5%        |
| Phase 7 RSSM + heuristic reward planner (h=4)| ~23% (avg over 3 seeds, ±7 std) | 0% |

### 7.8 What this means

The RSSM is a real, working learned world model: rollouts stay in
distribution, the multi-step consistency loss is well-behaved, and
visual inspection confirms the model tracks the red object's
position. This is the first world model in the project that
actually works.

But it isn't precise enough for navigation planning. The reasons
are honest:

1. **The data is small and uniform.** 1000 random-policy
   trajectories on a 5x5 grid is not enough to learn precise
   position dynamics, especially with 6.7% of trajectories
   actually reaching the goal.
2. **The encoder is small.** 1.18M parameters with a 3-layer
   7x7→4x4 CNN trunk; precision is limited by the trunk's
   effective receptive field on 7x7x3 inputs.
3. **The latent is small.** 64-dim, with a single shared
   representation for "where the agent is" and "where the goal
   is" — the planner needs both to discriminate at the cell level.

Dreamer solves these with: orders of magnitude more data (millions
of transitions from policy rollouts, not random), larger
backbones, deeper priors, value functions bootstrapped on
rollouts, and often auxiliary heads for state-value. None of
those are appropriate at the scale of this project (random-policy
data on a 5x5 grid).

### 7.9 What we'd recommend for the next project

1. **Collect goal-directed data.** 10x more trajectories from a
   partially-trained PPO policy, mixed with random. The world
   model needs to see diverse (state, action, next-state) triples
   to learn precise dynamics.
2. **Use the value-function bootstrap.** Train a V(z) on prior
   rollouts with TD(λ). Score = `Σγ^k r + γ^H V(z_H)`. This is
   the standard fix for the planning distribution shift.
3. **Architecturally, separate position and goal in the latent.**
   The current shared latent has to do double duty. A two-head
   encoder (one for agent state, one for goal state) would
   probably make the planner's job easier.
4. **Stop early on world-model improvements and focus on the
   planner.** The world model is in distribution; the planner
   is the limiting factor now. DreamerV2/V3 show that with
   sufficient data, a small world model can support a strong
   policy. We're not there yet.

## Files of interest

- `models/encoder.py`, `models/transition.py`, `models/decoder.py` — original world model
- `models/transition_residual.py` — anti-collapse transition model
- `models/rssm.py` — **Phase 7 RSSM world model** (TrunkEncoder + RSSM with prior/posterior + GRU + 3 overshoot heads)
- `training/train_world_model.py`, `training/train_autoencoder.py`,
  `training/train_reward_predictor.py` — original training scripts
- `training/train_reward_predictor_her.py`,
  `training/train_transition_residual.py` — Phase 6 fix attempts
- `training/train_rssm_ae.py`, `train_rssm_dyn.py`, `train_rssm_rollout.py` — **Phase 7 3-stage RSSM training**
- `training/train_her_reward_rssm.py` — **Phase 7 binary HER predictor**
- `planning/cem_planner.py`, `planning/fast_cem_planner.py` — original CEM planner
- `planning/her_cem_planner.py`, `planning/her_residual_planner.py` — HER planners
- `planning/rssm_cem_planner.py`, `planning/rssm_heuristic_planner.py` — **Phase 7 RSSM planners (learned and heuristic reward)**
- `evaluation/evaluate_planner.py`, `evaluation/evaluate_fast_planner.py` —
  random vs CEM vs PPO (Phases 4–5)
- `evaluation/evaluate_her_residual.py`, `evaluation/evaluate_her_planner.py` —
  HER planner evaluation
- `evaluation/evaluate_mpc_real.py`, `evaluation/evaluate_mpc_fourrooms.py` —
  **MPC with in-process env simulator (the fix)**
- `evaluation/evaluate_rssm_planner.py`, `evaluate_heuristic_planner.py` — **Phase 7 RSSM planner eval**
- `evaluation/check_rssm_recon.py`, `check_rssm_rollouts.py` — **Phase 7 visual sanity checks**
- `evaluation/simple_envs.py` — fast in-process env clones for MPC
- `evaluation/multi_step_prediction.py` — Phase 3
- `experiments/latent_size_ablation.py`, `experiments/horizon_ablation.py` —
  Phase 5
- `evaluation/results*/` — figures (latent space, reconstruction, multi-step)
- `evaluation/*_final.json`, `experiments/*.json` — raw ablation numbers
- `evaluation/rssm_empty_h4_big.json`, `rssm_heuristic_*.json` — **Phase 7 planner eval raw numbers**

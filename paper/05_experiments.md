## 4. Experiments

We present results chronologically, tracking the progression of
failure modes, diagnoses, and fixes. All experiments use the same
random-policy dataset (1,000 episodes, ~83k transitions for Empty-5x5)
unless otherwise noted.

### 4.1 Phase 1-3: Deterministic World Model

The initial autoencoder + MLP transition model achieves:

| Metric | Empty-5x5 | FourRooms |
|--------|-----------|-----------|
| Recon MSE | 5.22 (artifact) | 1.29 |
| Transition MSE | $3\times10^{-7}$ | $3\times10^{-7}$ |
| Multi-step MSE (h=1..20) | $\sim 10^{-6}$ | $\sim 0$ |

The reconstruction MSE of 5.22 on Empty-5x5 appears concerning, but
is entirely a units-mismatch artifact: observations are in $\{0,1,2,5,8\}$
but were divided by 255 at training time. After fixing normalization
(divide by 8), the properly-normalized baseline improves to MSE 0.067.
The apparent "good reconstruction" was the decoder learning to output
constant $\sim 0.015$ (5/255) in all pixels.

Multi-step prediction errors stay flat across all horizons, which
is the primary symptom of transition collapse: the model has learned
$z_{t+1} \approx z_t$.

### 4.2 Phase 4: Planning with Collapsed Model

We train a reward predictor on the sparse terminal rewards and evaluate
a CEM planner ($N=128$, $E=10$, $I=4$, $H=5$):

| Method | Empty-5x5 | FourRooms |
|--------|-----------|-----------|
| Random | 43.3\% | 6.7\% |
| CEM Planner | 0.0\% | 6.7\% |
| PPO (20k steps) | 0.0\% | 0.0\% |

The planner achieves 0\% on the easy environment where random succeeds
43\% of the time. This cleanly demonstrates the failure of the learned
world model to support planning.

### 4.3 Phase 5: Ablations

**Latent dimension sweep** (32, 64, 128, 256) shows world-model MSE
grows slightly with dimension (harder to collapse $z_{t+1}=z_t$) but
stays at $10^{-6}$ scale. Reward predictor MSE is **constant across
all dimensions** at 0.0015, depending only on the reward distribution
(99.9\% zero). Latent dimension is not the bottleneck.

**Planning horizon sweep** ($H=1,2,5,10$) on FourRooms shows the
planner's best at $H=10$ (10\%), worse than random at $H=5$ (0\%), and
non-monotonic. The noise is consistent with a reward predictor that
has near-zero gradient across most of the latent space.

### 4.4 Phase 6: Intervention

#### Residual Transition

Analyzing per-action latent changes reveals the transition model was
predicting action-appropriate changes after all -- turn/forward actions
produce small non-zero deltas while pickup/drop/toggle produce zero.
But the encoder collapse means these deltas operate in a latent ball
of diameter 0.01, and the reward predictor cannot distinguish states.

#### VICReg Encoder

VICReg expands the latent space from pairwise distance 0.0075 to 12.7
and per-dimension std from $\sim 0.0$ to 0.84. However, the deterministic
transition model still drifts out of distribution after 2-3 open-loop
steps, and the planner remains at 0\%.

#### MPC with Env Simulator

Replacing the learned world model with a fast in-process environment
simulator (the actual transition dynamics) gives the CEM planner the
accuracy it needs:

| Method | Empty-5x5 | FourRooms |
|--------|-----------|-----------|
| Random | 40.0\% | 3.3\% |
| MPC + Simulator | **98.0\%** | **13.3\%** |

This confirms the bottleneck was always the world model, not the
planner or the representations. With exact dynamics, CEM solves
the task.

### 4.5 Phase 7: RSSM Curriculum

Three-stage training on properly-normalized observations:

**Stage A (AE Warmup, 8 epochs):**
| Epoch | Recon MSE | KL | Post Std |
|-------|-----------|-----|----------|
| 1 | 0.0273 | 2.97 | 0.96 |
| 8 | **0.0022** | 2.56 | 0.98 |

Mean-baseline constant prediction is 0.067 in normalized units; the
warmup beats it by $30\times$. Visual reconstruction confirms the
decoder uses the latent (red object correctly positioned).

**Stage B (+Dynamics, 8 epochs):**
| Epoch | Recon MSE | KL | Prior/Post Std |
|-------|-----------|-----|----------------|
| 1 | 0.0091 | 1.65 | 0.99 |
| 8 | 0.0093 | **0.015** | 0.995 |

KL drops two orders of magnitude as the prior learns to match the
posterior. Both $\sigma \approx 1.0$, meaning the stochastic latent
is active -- this is the desired state for inference, not a collapse.

**Stage C (+Latent Overshooting, 6 epochs):**
| Epoch | Recon MSE | KL | Overshoot |
|-------|-----------|-----|-----------|
| 1 | 0.0094 | 0.05 | 0.81 |
| 6 | 0.0092 | 0.007 | **0.64** |

Overshoot loss drops from 0.81 to 0.64. Open-loop rollout quality
is stable across 6 steps:

| Step | Rollout MSE |
|------|-------------|
| 1 | 0.0066 |
| 2 | 0.0094 |
| 3 | 0.0126 |
| 4 | 0.0100 |
| 5 | 0.0088 |
| 6 | 0.0115 |

MSE stays flat at $\sim 0.01$ across the full horizon. Visual
inspection confirms the dark red object remains in roughly the
correct position throughout the rollout.

### 4.6 RSSM Planning

The binary HER reward predictor achieves 97.6\% classification
accuracy (loss $< 10^{-5}$) on posterior-encoded states. The RSSM
CEM planner ($N=256$, $E=16$, $I=5$, $H=4$) yields:

| Method | Empty-5x5 | FourRooms |
|--------|-----------|-----------|
| Random | 23\% | 5\% |
| RSSM Planner | **30\%** | 5\% |

A modest 7-point improvement on Empty-5x5. The planner picks
similar action sequences to what random produces.

#### Heuristic Reward Diagnostic

To isolate whether the world model or the reward predictor is the
bottleneck, we replace the learned reward with a heuristic:
$r = \exp(-\text{MSE}(\text{decoded}(\hat{z}_t), o_{\text{goal}}) / \sigma^2)$.
This evaluates the world model directly on pixel-space distance to
the goal.

| Seed | Random | Heuristic Planner |
|------|--------|-------------------|
| 1 | 25\% | 40\% |
| 2 | 25\% | 30\% |
| 3 | 25\% | 17\% |
| 4 | 25\% | 23\% |
| **Avg** | 25\% | **23\%** |

The heuristic planner averages 23\% across 3 seeds ($\sigma = \pm 7$),
not significantly different from random. This kills the hypothesis
that "reward domain shift" (posterior-trained predictor evaluated on
prior rollouts) was the primary bottleneck. The world model's
control-relevant prediction quality -- not the reward predictor --
is the limiting factor.

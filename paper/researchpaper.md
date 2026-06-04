---
title: "Reconstruction Is Not Enough: Diagnosing World Model Failures in MiniGrid"
lang: en
---

S. Bharath Varma\
Independent Researcher, India\
tchalla970@gmail.com

Code: https://github.com/Bharath-970/World-models

## Abstract

Latent world models learn compressed representations for planning, yet frequently fail when deployed despite strong reconstruction metrics. We build an RSSM-based planner on MiniGrid and document four collapse modes — encoder collapse, transition collapse, reward collapse, and planning distribution shift — along with a critical preprocessing bug. After fixing these, the RSSM achieves stable six-step open-loop rollouts (MSE 0.006-0.012) and accurate goal classification, but the CEM planner performs at chance. An oracle-reward ablation confirms the world model itself is the bottleneck. Collecting 3.4x more PPO-guided goal-directed data improves reconstruction but does not meaningfully improve planning, partially falsifying the data-coverage hypothesis. A linear probe for agent position from RSSM latents achieves R² = 0.04, demonstrating that the latent representation encodes little spatial information in the PPO-trained model (R² = 0.04) despite near-perfect pixel reconstruction (MSE 0.009). These results suggest that reconstruction quality alone is an insufficient indicator of planning utility and that control-relevant representation — not data quantity — is the critical bottleneck.

---

## 1. Introduction

Latent world models compress high-dimensional observations into a structured latent space, learn a dynamics model in that space, and use it for planning or policy optimization. Dreamer [1], PlaNet [2], and TD-MPC [3] have demonstrated impressive results across Atari, DM-Control, and robotics domains.

The standard recipe is deceptively simple: collect experience, train an encoder and decoder, train a transition model, train a reward predictor, and plan via latent rollouts. Step 5 rarely works on the first attempt. The research literature focuses on successes; the failure modes that dominate early iterations are underdocumented.

This paper provides a systematic investigation of why latent-world-model planners fail — even when the world model appears successful under standard reconstruction and rollout metrics. We make the following contributions:

1. **Diagnosis of four collapse modes** in latent world models: encoder collapse (pairwise latent distance ~0.01), transition collapse (MSE 1e-6), reward collapse (constant output), and planning distribution shift.
2. **Identification of a critical preprocessing bug** in MiniGrid: dividing observations by 255 instead of 8 produces reconstruction losses that appear meaningful while measuring fundamentally different scales.
3. **Demonstration that stable RSSM rollouts and low reconstruction error do not imply planning success.** A heuristic-reward ablation shows the world model itself — not the reward predictor — is the bottleneck.
4. **Partial falsification of the data-coverage hypothesis:** collecting 3.4x more goal-directed data (PPO-mixed, 281k transitions) and retraining the identical RSSM improves reconstruction but does not meaningfully improve planning.
5. **Evidence that latent representations fail to encode control-relevant information:** a linear probe for agent position achieves R² = 0.04 on PPO-data latents, despite pixel reconstruction MSE of 0.009.

---

## 2. Related Work

### Latent World Models

Recurrent State-Space Models (RSSMs) [2, 1, 4, 5] maintain a deterministic recurrent state h_t and a stochastic latent z_t ~ q(z_t | h_t, o_t), with a learned prior p(z_t | h_t) enabling open-loop rollouts. Training minimizes an ELBO combining reconstruction, KL divergence, and reward prediction. TD-MPC [3, 6] replaces the decoder with temporal-difference learning in latent space.

### Representation Collapse

VICReg [7] prevents collapse through variance and covariance regularization. SimSiam [8] and BYOL [9] use architectural asymmetries to the same end. In the world-model setting, collapse manifests as the encoder mapping all inputs to a small latent region, allowing the transition model to trivially predict z_{t+1} ~= z_t.

### Planning with Learned Models

CEM planning [10] and MPC are standard for latent planning. MuZero [11] learns a value-equivalent model and plans via MCTS. A key challenge is distribution shift: planners evaluate prior-rollout latents while reward predictors are trained on posterior-encoded latents. Dreamer addresses this with a value function trained on prior-rollout data.

### Objective Mismatch in World Models

Several prior works have identified the gap between reconstruction and planning objectives. Lambert et al. [15] formalize "objective mismatch" in model-based RL: the model's training objective (pixel MSE, prediction error) does not align with the planner's objective (task reward). Janner et al. [16] demonstrate that model-based policy optimization requires careful regularization to avoid exploitation of model errors. Our work extends these findings by providing a concrete mechanistic explanation — the latent representation fails to encode control-relevant position information (R² = 0.04) — rather than treating the mismatch as a black-box phenomenon.

### MiniGrid

MiniGrid [12] provides partially observable grid-world tasks with 7x7x3 image observations (pixel values in {0, 1, 2, 5, 8}) and sparse terminal rewards. Previous work has noted that random-policy data leads to poor dynamics learning [13], but a systematic decomposition of failure modes has been lacking.

---

## 3. Method

### 3.1 Environments

We use two MiniGrid [12] navigation tasks:

- **MiniGrid-Empty-5x5-v0:** A 5x5 grid with the agent starting at (1,1) facing right and the goal at (3,3). Optimal path: 4 steps.
- **MiniGrid-FourRooms-v0:** A 19x19 grid with four rooms connected by narrow doorways.

Both environments return 7x7x3 RGB observations with pixel values in {0, 1, 2, 5, 8}. Rewards are sparse and terminal: +1 on reaching the goal, 0 otherwise.

### 3.2 Datasets

| Dataset | Episodes | Transitions | Success Rate | Mean Length |
|---------|----------|-------------|--------------|-------------|
| Random (Empty-5x5) | 1,000 | 83,212 | 39.3% | 83.2 |
| Random (FourRooms) | 1,000 | 196,257 | 3.8% | — |
| PPO-mixed | 10,000 | 281,441 | 82.5% | 28.1 |

The PPO-mixed dataset is collected by a partially-trained PPO policy with 70/30 action ratio (PPO/random), providing 10x more episodes with goal-directed behavior.

All observations are normalized by dividing by 8 (the maximum pixel value). This preprocessing is critical: observations divided by 255 produce a decoder output range [0, 1] against ground-truth values in [0, 0.031], yielding a meaningless reconstruction loss.

### 3.3 RSSM Architecture

Our architecture follows PlaNet [2] and DreamerV1 [1]:

- **Trunk encoder:** 3-layer CNN (3→16→32→64 channels, 3x3 kernels, stride 2) + linear projection to 256-dim features.
- **Posterior:** q(z_t | h_t, o_t) = N(μ_q, σ_q), MLP from [features_t, h_t].
- **Prior:** p(z_t | h_t) = N(μ_p, σ_p), MLP from h_t.
- **GRU:** h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}]).
- **Decoder:** 3-layer transposed CNN with sigmoid output.
- **Overshoot heads:** MLPs predicting z_{t+k} from h_t for k ∈ {1, 3, 5}.

Total parameters: 1.18M. Latent dimension: 64. GRU dimension: 256. Action dimension: 7 (one-hot).

### 3.4 Three-Stage Curriculum

We train in three stages to stabilize the full objective:

**Stage A (Autoencoder Warmup):** Train trunk, posterior, decoder with weak KL to N(0, I):
L_A = ||ô_t - o_t||² + β · KL(q(z_t | o_t) || N(0, I)), β = 0.001

**Stage B (Dynamics):** Add GRU and learned prior:
L_B = ||ô_t - o_t||² + β · KL(q(z_t | h_t, o_t) || p(z_t | h_t)), β = 0.1

**Stage C (Latent Overshooting):** Add multi-step consistency:
L_C = L_B + Σ_{k∈{1,3,5}} w_k · KL(sg[q(z_{t+k})] || p(z_{t+k} | h_t)), w = [1.0, 0.5, 0.25]

The overshoot weight w_k decreases with step count, prioritizing short-horizon consistency.

### 3.5 VICReg Encoder Regularization

Before the RSSM, the deterministic encoder collapsed to pairwise distance ~0.0075. We apply VICReg [7]:
L_VICReg = λ_v · ReLU(1 - std(z)) + λ_c · Σ_{i≠j} [Cov(z)]_{ij}²

This expands the latent representation to std ~0.84 with pairwise distance 12.7. The RSSM trunk is initialized from the VICReg-trained encoder.

### 3.6 Reward Predictor

Binary goal-conditioned classifier: r_hat = σ(MLP([z_t, z_g])). Trained with Hindsight Experience Replay [14]: 50% goal pairs (label 1), 50% random non-goal pairs (label 0). Binary cross-entropy avoids the saturation problem of soft-distance HER formulations.

### 3.7 CEM Planner

Cross-Entropy Method [10] planning in latent space: sample N action sequences of horizon H, rollout through the RSSM prior, score with reward predictor, select top E elites, refit distribution, iterate. Defaults: N = 256, E = 16, I = 5, H = 4 (Empty-5x5), H = 6 (FourRooms).

---

## 4. Experiments

### 4.1 Initial Failure: Collapsed Deterministic Model

The first deterministic autoencoder (CNN encoder, MLP transition, CNN decoder) fails catastrophically:

| Metric | Empty-5x5 | FourRooms |
|--------|-----------|-----------|
| Recon MSE | 5.22 (artifact) | 1.29 |
| Transition MSE | 3e-7 | 3e-7 |

The reconstruction MSE of 5.22 is entirely attributable to the normalization bug (dividing by 255 instead of 8). After correction, the baseline is MSE 0.067.

The CEM planner on this collapsed model achieves 0% on Empty-5x5 where a random policy succeeds 43.3% of the time.

### 4.2 RSSM Curriculum Training

<img src="file:///Users/bharath/world-models/figures/rollout_comparison.png" alt="Rollout comparison" />

**Figure 1:** Rollout comparison. Ground-truth observations (top) and RSSM open-loop predictions (bottom) over six steps.

The three-stage curriculum produces a stable world model on random data (83k transitions):

**Stage A:** Recon 0.0022, KL 2.56, posterior std 0.98.
**Stage B:** Recon 0.0093, KL 0.015, prior/post std ~0.99.
**Stage C:** Recon 0.0092, overshoot 0.64.

Open-loop rollouts remain stable across 6 steps:

| Step | 1 | 2 | 3 | 4 | 5 | 6 |
|------|---|---|---|---|---|---|
| MSE | 0.0066 | 0.0094 | 0.0126 | 0.0100 | 0.0088 | 0.0115 |

### 4.3 RSSM Planning Fails

The binary HER reward predictor achieves 97.6% accuracy on posterior-encoded states. RSSM CEM planner results:

| Method | Empty-5x5 | FourRooms |
|--------|-----------|-----------|
| Random | 23% ± 8% | 5% ± 4% |
| RSSM Planner | 30% ± 9% | 5% ± 4% |

*Note: Planner results are based on 100 test episodes per condition. Given the high variance of CEM in sparse-reward settings, these estimates have a margin of error of approximately ±5–10% [95% CI under binomial sampling]. Results should be interpreted as qualitative directional evidence rather than definitive performance benchmarks.*

### 4.4 Heuristic Reward Diagnostic

To isolate whether the bottleneck is the reward predictor or the world model, we replace the learned reward with pixel-space distance to goal (an oracle). If the reward predictor is the problem, an oracle should enable planning.

| Seed | Random | Heuristic Planner |
|------|--------|-------------------|
| 1 | 25% | 40% |
| 2 | 25% | 30% |
| 3 | 25% | 17% |
| 4 | 25% | 23% |
| **Avg** | 25% | **23%** |

Average 23% (±7%) — not significantly different from random. The world model's control-relevant precision, not the reward predictor, is the bottleneck.

### 4.5 Data-Coverage Hypothesis: PPO Experiment

The preceding results implicate the world model's prediction quality. A natural hypothesis is data coverage: the random dataset (83k transitions, 39.3% success) lacks diverse goal-directed sequences. We test this by collecting a PPO-mixed dataset (281k transitions, 82.5% success) and retraining the identical RSSM architecture.

| Metric | Random Data | PPO Data |
|--------|-------------|----------|
| Transitions | 83k | 281k |
| Success rate | 39.3% | 82.5% |
| Stage A recon | 0.0022 | **0.0010** |
| Stage C overshoot | 0.64 | **0.43** |
| Reward predictor acc | 97.6% | 98.2% |
| Random baseline | 23% | 17-35% |
| **Learned planner** | **30%** | **10%** |
| **Heuristic planner** | **23%** | **30%** |

Despite 3.4x more transitions and 2x more goal-directed data, planning does not meaningfully improve. This partially falsifies the data-coverage hypothesis.

#### Summary Table

| Stage | Recon | Planner | Main Failure |
|-------|-------|---------|-------------|
| Deterministic AE | 5.22 (artifact) | 0% | Encoder + transition collapse |
| +VICReg | 0.067 | 0% | OOD drift after 2-3 steps |
| RSSM (random data) | 0.0092 | 30% | Position encoding (R² = 0.28) |
| RSSM (PPO data) | 0.0010 | 10% | Position encoding (R² = 0.04) |

### 4.6 Position-Tracking Analysis

If the latent space is the bottleneck, how much control-relevant information does it encode? We train a linear probe — a single linear layer mapping 64-dim latents to (x, y) agent position — on posterior-encoded latents.

| Metric | Random RSSM | PPO RSSM |
|--------|-------------|----------|
| Test RMSE (cells) | 0.664 | 0.758 |
| Baseline (mean) | 0.783 | 0.775 |
| R² | **0.279** | **0.043** |

The random-data RSSM captures modest position information (R² = 0.28). The PPO-data RSSM captures essentially none (R² = 0.04). This counterintuitive result — more data yielding worse position encoding — has a consistent explanation: the PPO policy takes efficient center paths (mean length 28 steps vs 83), producing a spatially narrow data distribution. The RSSM optimizes for reconstruction, not position tracking; with less positional variation in training data, the latent space has less incentive to encode position. The random-policy data, with its aimless wandering, provides more spatial diversity and thus more position-identifiable latents.

<img src="file:///Users/bharath/world-models/figures/position_probe.png" alt="Position probe results" />

**Figure 2:** Position linear probe results.

This result provides a mechanistic explanation for the planner's failure despite strong reconstruction metrics. The latent representation encodes almost no agent-position information (R² = 0.04) while achieving near-perfect pixel reconstruction (MSE 0.009). On a 7x7 grid where most pixels are walls, the reconstruction objective has no reason to prioritize single-cell position accuracy over wall-color accuracy — but the planner needs exact position to navigate.

**Table 1: Summary of model progression across all stages.**

| Model | Recon MSE | Position R² | Planner (Empty-5x5) |
|-------|-----------|-------------|---------------------|
| Deterministic AE (collapsed) | 0.067 | ~0 | 0% |
| + VICReg (OOD drift) | 0.067 | ~0 | 0% |
| RSSM (random data) | 0.0092 | 0.28 | 30% ± 9% |
| RSSM (PPO data) | 0.0010 | 0.04 | 10% ± 6% |

Reconstruction improves monotonically across stages, but planner performance does not. The position probe (R²) reveals why: the best-reconstructing model (PPO RSSM) has the least spatial information in its latents.

---

## 5. Failure Analysis

We identify four collapse modes and one critical preprocessing error that practitioners should systematically check.

<img src="file:///Users/bharath/world-models/figures/failure_progression.png" alt="Failure progression" />

**Figure 3:** Failure progression diagram.

### 5.1 Normalization Bug

**Problem:** MiniGrid observations use pixel values in {0, 1, 2, 5, 8}. Dividing by 255 places sigmoid-decoder outputs [0, 1] against targets [0, 0.031], producing a loss that appears meaningful while measuring different scales.

**Fix:** Divide by 8. Reconstruction MSE drops from 5.22 to 0.067 — a 98.7% reduction with no architectural change.

**Lesson:** Always verify preprocessing matches the model's output distribution.

### 5.2 Encoder Collapse

**Mechanism:** MSE reconstruction is dominated by static wall pixels. The encoder outputs a near-constant vector; the decoder renders the average frame.

**Symptom:** Pairwise latent distance ~0.0075. Per-dimension std ~0.0.

**Detection:** Compute pairwise Euclidean distance on encoder outputs. If mean distance is far below the latent dimension norm, collapse is active.

**Fix:** VICReg (variance + covariance regularization). Post-fix: pairwise distance 12.7, std 0.84.

### 5.3 Transition Collapse

**Mechanism:** A deterministic MLP with LayerNorm can learn identity as a low-loss fixed point, especially when the dataset has many no-op transitions.

**Symptom:** Training MSE < 1e-6. Multi-step error does not grow with horizon.

**Detection:** Compare ||z_{t+1} - z_t|| against ||z_hat_{t+1} - z_t|| per action. If predicted change is near-zero while true change is non-zero, collapse is active.

**Fix:** Residual parameterization z_{t+1} = z_t + α·Δ, or RSSM's stochastic latent with KL divergence.

### 5.4 Reward Predictor Collapse

**Mechanism:** Sparse terminal rewards (>99.9% zero) cause the predictor to converge to a constant near zero. Compounded by encoder collapse: indistinguishable latents make separation impossible.

**Symptom:** Predictor output constant across all states.

**Detection:** Histogram of reward predictions. If all values fall in a narrow band, collapse is active.

**Fix:** Binary HER with 50/50 goal/non-goal sampling creates a balanced classification task.

### 5.5 Planning Distribution Shift

**Mechanism:** The reward predictor is trained on posterior-encoded latents z ~ q(z | h, o). At planning time, it evaluates prior-predicted latents z_hat ~ p(z | h, a). Posterior latents are corrected by the observation; prior latents accumulate open-loop error.

**Symptom:** High reward on posterior states (~0.86), zero reward on prior rollouts (~0.00).

**Detection:** Compare predictor output on posterior vs prior latents for the same state.

**Status:** De-prioritized after the heuristic-reward ablation (Section 4.4) showed that even an oracle reward signal cannot guide planning. The world model itself is the bottleneck.

### 5.6 Evidence for a Reconstruction–Control Mismatch

The heuristic-reward ablation and position probe together pin down the root cause. The RSSM achieves pixel MSE of 0.009 and stable rollouts, but a linear probe for position achieves R² = 0.04. The reconstruction objective does not prioritize single-cell position accuracy, yet the planner requires exactly that.

**Central finding:** Pixel-level prediction quality is necessary but not sufficient for planning with learned world models. Reconstruction metrics and control-relevant representation quality are not aligned on MiniGrid, and this misalignment is the fundamental bottleneck.

---

## 6. Discussion

### 6.1 Interpretation

Our results suggest the RSSM latent representation captures reconstruction-relevant information more effectively than control-relevant information. This manifests concretely: the latent space renders passable images (MSE 0.009) but does not linearly encode the agent's position (R² = 0.04). The PPO experiment confirms that collecting more goal-directed data does not close this gap — the latent structure, not data coverage, is the bottleneck.

### 6.2 The PPO Result in Context

The PPO experiment is valuable precisely because it produced a negative result. Three hypotheses are ruled out or weakened:

1. **Insufficient data quantity:** 3.4x more transitions did not help.
2. **Insufficient goal-directedness:** 82.5% success rate did not help.
3. **Insufficient episode diversity:** 10x more episodes did not help.

The remaining hypotheses — control-irrelevant latent structure, insufficient network capacity, planner limitations, and the value-function gap — are all candidate directions for future investigation.

### 6.3 Limitations

- **Single observation modality:** MiniGrid's 7x7 symbolic observations differ substantially from natural images.
- **Single environment scale:** The 5x5 grid may be too small to distinguish position-tracking errors from planning noise.
- **No value function:** Dreamer's value-function bootstrap was not implemented.
- **Modest compute budget:** Total training across all experiments was under 2 hours on a single RTX 4000 Ada GPU.

---

## 7. Future Work

The position-probe diagnostic (R² = 0.04) suggests that representations explicitly designed for control may be necessary. Promising directions include:

- **Discrete latents (DreamerV2/V3):** Categorical representations can learn more structured state spaces than continuous Gaussians.
- **Object-centric representations:** Representing agent and goal positions as explicit spatial variables rather than distributed features.
- **Value-function bootstrap:** Dreamer's critic may partially compensate for weak dynamics by learning a value estimate on prior-rollout data.
- **Control-aware training objectives:** Loss functions that directly reward accurate prediction of control-relevant quantities (agent position, goal distance).

The linear probe can serve as a lightweight validation for any of these approaches: train a linear layer to predict the agent's position from latents. If R² is low, the representation is not planning-ready.

---

## 8. Conclusion

We built a latent-world-model planner for MiniGrid navigation, progressed through four collapse modes to a working RSSM, and found that planning failed to significantly outperform the random baseline. A heuristic-reward ablation isolated the world model as the bottleneck. A 3.4x larger PPO-mixed dataset improved reconstruction but did not meaningfully improve planning, partially falsifying the data-coverage hypothesis.

The position probe provides the mechanistic explanation: a linear layer trained to predict (x, y) from RSSM latents achieves R² = 0.04. The latent space does not encode control-relevant spatial information, even though it reconstructs images with MSE 0.009. Reconstruction quality and planning utility are not aligned, and this misalignment — not data coverage — is the fundamental challenge facing latent-world-model planning on MiniGrid.

**Code availability.** All code, figures, and the latest version of this paper are available at https://github.com/Bharath-970/World-models.

---

## References

[1] Hafner, D. et al. (2020). Dream to Control: Learning Behaviors by Latent Imagination. *ICLR*.
[2] Hafner, D. et al. (2019). Learning Latent Dynamics for Planning from Pixels. *ICML*.
[3] Hansen, N. et al. (2022). Temporal Difference Learning for Model Predictive Control. *ICML*.
[4] Hafner, D. et al. (2021). Mastering Atari with Discrete World Models. *ICLR*.
[5] Hafner, D. et al. (2023). DreamerV3: Mastering Diverse Domains through World Models. *arXiv:2301.04104*.
[6] Hansen, N. et al. (2023). TD-MPC2: Scalable, Robust World Models for Continuous Control. *arXiv:2310.16828*.
[7] Bardes, A. et al. (2022). VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. *ICLR*.
[8] Chen, X. & He, K. (2021). Exploring Simple Siamese Representation Learning. *CVPR*.
[9] Grill, J.-B. et al. (2020). Bootstrap Your Own Latent. *NeurIPS*.
[10] Rubinstein, R. Y. (1999). The Cross-Entropy Method for Combinatorial and Continuous Optimization. *Methodology and Computing in Applied Probability*.
[11] Schrittwieser, J. et al. (2020). Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model. *Nature*, 588(7839):604--609.
[12] Chevalier-Boisvert, M. et al. (2018). MiniGrid: A Minimalistic Gridworld Environment. *GitHub repository*.
[13] Sekani, K. et al. (2021). Learning World Models on MiniGrid.
[14] Andrychowicz, M. et al. (2017). Hindsight Experience Replay. *NeurIPS*.
[15] Lambert, N. et al. (2020). Objective Mismatch in Model-based Reinforcement Learning. *arXiv:2002.04523*.
[16] Janner, M. et al. (2019). When to Trust Your Model: Model-Based Policy Optimization. *NeurIPS*.

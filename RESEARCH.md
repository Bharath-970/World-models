# World Models Research Summary

## Paper Evolution

### 1. World Models (Ha & Schmidhuber, 2018)
**Key Innovation**: VAE + MDN-RNN + Evolution-based Controller

**Architecture**:
- **V (Vision)**: VAE compresses frames → 32-dim latent vector z
- **M (Memory)**: MDN-RNN predicts P(z_{t+1} | a_t, z_t, h_t) as mixture of Gaussians
- **C (Controller)**: Simple linear model mapping [z, h] → action

**Training**:
1. Collect 10,000 random rollouts
2. Train VAE unsupervised
3. Train MDN-RNN to predict next latent
4. Evolve controller with CMA-ES (867 parameters!)

**Results**:
- CarRacing: 906 ± 21 (SOTA at time)
- VizDoom: 1092 ± 556 (learned in dream environment)

**Critical Insight**: Temperature parameter τ controls dream uncertainty. Higher τ prevents exploitation but makes learning harder.

**Limitations**:
- Random policy data collection
- Evolution strategies don't scale well
- No gradient-based policy learning

---

### 2. PlaNet (Hafner et al., 2018)
**Key Innovation**: Latent overshooting + online planning

**Architecture**:
- Latent dynamics model with **deterministic + stochastic** components
- State: s_t = (h_t, z_t) where h is deterministic, z is stochastic
- RSSM (Recurrent State-Space Model)

**Training Objective**:
- Multi-step variational inference (latent overshooting)
- Predict rewards and observations

**Planning**:
- Cross-Entropy Method (CEM) in latent space
- Online planning at each step

**Results**:
- Solves contact dynamics, partial observability, sparse rewards
- More sample efficient than model-free methods

**Why It Matters**:
- First to show latent planning can solve hard control tasks
- Introduced RSSM architecture used in Dreamer series

---

### 3. Dreamer (Hafner et al., 2019)
**Key Innovation**: Gradient-based policy learning in latent space

**Architecture**:
- Same RSSM as PlaNet
- Actor-critic trained via backprop through imagined trajectories

**Training**:
1. Learn world model from real experience
2. Imagine trajectories in latent space
3. Train actor-critic on imagined data
4. Collect more real experience
5. Repeat

**Results**:
- Exceeds PlaNet performance
- More compute efficient (no online planning)
- 20 challenging visual control tasks

**Why It Matters**:
- Proved gradient-based RL works in latent space
- Much faster than CEM planning

---

### 4. DreamerV3 (Hafner et al., 2023)
**Key Innovation**: Single hyperparameter config across 150+ diverse tasks

**Architecture**:
- RSSM with categorical latents
- Symlog value prediction
- Free bits KL divergence
- Normalization everywhere

**Key Techniques**:
- **Categorical latents**: Discrete instead of continuous
- **Symlog**: log(|x| + 1) * sign(x) for value prediction
- **Free bits**: Prevents KL collapse
- **Unimix**: Categorical sampling trick

**Results**:
- First to collect diamonds in Minecraft from scratch
- Masters Atari, DMC, Crafter, Minecraft, etc.
- Fixed hyperparameters work everywhere

**Why It Matters**:
- Proved world models can be truly general
- Nature publication (2025)
- State-of-the-art model-based RL

**Implementation Notes**:
- Uses JAX (not PyTorch!)
- Complex but well-documented
- 3.3k GitHub stars

---

### 5. I-JEPA (Assran et al., 2023)
**Key Innovation**: Non-generative self-supervised learning

**Architecture**:
- Joint Embedding Predictive Architecture
- Predict target representations from context
- No pixel-level reconstruction

**Key Insight**:
- Masking strategy is crucial
- Large-scale semantic targets
- Spatially distributed context

**Results**:
- ViT-Huge/14 on ImageNet in 72 hours
- Strong downstream performance

**Why It Matters**:
- LeCun's vision for next-gen AI
- Avoids pixel prediction bottleneck
- Highly scalable

---

### 6. V-JEPA 2 (Meta, 2024)
**Key Innovation**: Vision + Action JEPA

**Architecture**:
- Extends I-JEPA to video and action
- Predicts future visual representations given actions
- World model for planning

**Why It Matters**:
- Most recent world model research
- Meta's frontier research
- Connects JEPA to planning

---

## Key Concepts

### Representation Collapse
**Problem**: Model learns constant latent (everything → same vector)
**Symptoms**: Loss looks good, but model learns nothing
**Solutions**:
- Contrastive learning
- VICReg (variance + covariance regularization)
- Free bits KL (DreamerV3)
- BYOL (exponential moving average)

### Latent Space Quality
**Metrics**:
- Prediction MSE
- Reconstruction quality
- Latent norm stability
- t-SNE/PCA clustering

### Planning in Latent Space
**Methods**:
1. **CEM** (PlaNet): Sample actions, keep best
2. **Gradient-based** (Dreamer): Backprop through imagined trajectories
3. **Value-guided** (V-JEPA): Score future states with value function

### Sample Efficiency
World models are more sample efficient than model-free because:
- Reuse experience via imagination
- Learn dynamics once, plan many times
- Better representations reduce sample complexity

---

## Common Pitfalls

### 1. Starting Too Complex
**Wrong**: Implement DreamerV3 from scratch
**Right**: Start with simple encoder + transition model

### 2. Ignoring Data Quality
**Wrong**: Assume model is bad
**Right**: Check trajectories, preprocessing, action encoding

### 3. No Visualization
**Wrong**: Only look at loss curves
**Right**: Visualize reconstructions, latent space, predictions

### 4. Representation Collapse
**Wrong**: Trust low loss
**Right**: Check latent variance, use t-SNE

### 5. Overfitting to Training Data
**Wrong**: Perfect training loss, bad generalization
**Right**: Hold-out validation set, monitor prediction error

### 6. Wrong Environment
**Wrong**: Start with Atari/Minecraft
**Right**: Start with MiniGrid/CartPole

### 7. No Baselines
**Wrong**: Only evaluate your model
**Right**: Compare to PPO, DQN, random policy

### 8. Ignoring Compute
**Wrong**: Train for weeks
**Right**: Know your budget, use small models first

---

## Implementation Strategy

### Phase 1: MVP (2 weeks)
**Goal**: Predict next latent state

**Components**:
1. MiniGrid environment
2. Trajectory collection (random policy)
3. CNN encoder (84x84 → 128-dim)
4. Transition model (z_t, a_t → z_{t+1})
5. MSE loss training
6. WandB logging

**Success Criteria**:
- Prediction loss < 0.1
- Latent space shows structure (t-SNE)
- Can predict 5 steps ahead

### Phase 2: Visual Validation (1 week)
**Goal**: Verify encoder quality

**Components**:
1. Add decoder
2. Reconstruct images
3. Side-by-side visualization

**Success Criteria**:
- Reconstructions capture key features
- Not blurry mess

### Phase 3: Multi-Step Prediction (1 week)
**Goal**: Predict future states

**Components**:
1. Roll out predictions: t+1, t+5, t+10
2. Measure error accumulation

**Success Criteria**:
- Reasonable predictions at t+5
- Understandable degradation

### Phase 4: Planning (2 weeks)
**Goal**: Use world model for decision making

**Components**:
1. CEM planner in latent space
2. Score futures with simple heuristic
3. Compare to random policy

**Success Criteria**:
- Beat random policy
- Beat PPO sample efficiency

### Phase 5: Research Experiments (2 weeks)
**Goal**: Answer research question

**Research Question**: How does latent prediction quality affect planning performance?

**Experiments**:
1. Vary latent size: 32, 64, 128, 256
2. Vary prediction horizon: 1, 5, 10, 20
3. Measure: reward, sample efficiency, convergence speed

**Baselines**:
- PPO (Stable-Baselines3)
- DQN (Stable-Baselines3)
- Random policy

### Phase 6: Memory (Optional, 2 weeks)
**Goal**: Handle partial observability

**Components**:
1. Add LSTM/GRU to encoder
2. Compare to memoryless version

**Success Criteria**:
- Better performance on POMDP tasks

---

## Tech Stack

```
Core:
- Python 3.11+
- PyTorch 2.x
- Gymnasium
- MiniGrid

RL:
- Stable-Baselines3 (baselines)

Experiment Tracking:
- Weights & Biases
- Hydra (config management)

Visualization:
- matplotlib
- scikit-learn (t-SNE, PCA)

Environment:
- conda
```

---

## Research Questions

### Primary
**How does latent prediction quality affect downstream planning performance?**

### Secondary
1. What latent dimensionality is optimal?
2. How does prediction horizon affect planning?
3. Does memory improve performance on POMDPs?
4. Can world models beat model-free baselines in sample efficiency?

---

## Paper Title Options

1. **Safe**: "Latent World Models for Planning in Dynamic Environments"
2. **Stronger**: "Evaluating Predictive World Models for Efficient Decision Making in Partially Observable Environments"
3. **Interesting**: "From Prediction to Planning: Investigating World Model Quality and Agent Performance"

---

## Timeline

| Week | Goal | Deliverable |
|------|------|-------------|
| 1-2 | MVP | Working world model, prediction loss curves |
| 3 | Visual validation | Reconstruction visualizations |
| 4 | Multi-step | Prediction horizon analysis |
| 5-6 | Planning | CEM planner, comparison to baselines |
| 7-8 | Research | Ablation studies, research question answered |
| 9-10 | Paper | Write-up, experiments documented |
| 11-12 | Polish | Code cleanup, GitHub repo, presentation |

---

## Success Metrics

### Technical
- [ ] World model predicts next latent (MSE < 0.1)
- [ ] Reconstructions capture key features
- [ ] Multi-step prediction works (t+5 reasonable)
- [ ] Planner beats random policy
- [ ] Beat PPO sample efficiency

### Research
- [ ] Answer primary research question
- [ ] Complete ablation study
- [ ] Compare to 2+ baselines
- [ ] Document all experiments

### Paper
- [ ] Clear research question
- [ ] Reproducible experiments
- [ ] Professional write-up
- [ ] GitHub repo with code

---

## Resources

### Papers (Read in Order)
1. World Models (2018): https://arxiv.org/abs/1803.10122
2. PlaNet (2018): https://arxiv.org/abs/1811.04551
3. Dreamer (2019): https://arxiv.org/abs/1912.01603
4. DreamerV3 (2023): https://arxiv.org/abs/2301.04104
5. I-JEPA (2023): https://arxiv.org/abs/2301.08243

### Code References
- DreamerV3: https://github.com/danijar/dreamerv3
- MiniGrid: https://minigrid.farama.org/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/

### Tutorials
- World Models interactive: https://worldmodels.github.io/
- DreamerV3 website: https://danijar.com/dreamerv3/

---

## Next Steps

1. **Read papers** (1 week)
   - World Models (2018)
   - DreamerV3 (2023)
   - Focus on architecture and training, skip math details

2. **Setup environment** (1 day)
   - Create conda env
   - Install dependencies
   - Test MiniGrid

3. **Build MVP** (2 weeks)
   - Follow Phase 1 plan
   - Log everything to WandB
   - Visualize early and often

4. **Iterate** (remaining weeks)
   - Add features incrementally
   - Run experiments
   - Document everything

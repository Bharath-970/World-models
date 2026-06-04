# Additional World Models Research

## Papers Discovered

### DreamerV2 (2020) - Mastering Atari with Discrete World Models
**Key Innovation**: First agent to achieve human-level performance on Atari using world models

**Architecture**:
- **Discrete latent representations** (categorical instead of continuous)
- RSSM with discrete states
- World model trained separately from policy
- Actor-critic trained in latent space

**Why Discrete?**:
- More stable training
- Better for Atari (discrete game states)
- Prevents representation collapse
- Easier to model discrete events

**Results**:
- Human-level on 55 Atari tasks
- Surpassed IQN and Rainbow with same compute
- Also works for continuous control (humanoid walking)

**Key Insight**: Discrete representations are more robust than continuous for world models

**Paper**: https://arxiv.org/abs/2010.02193

---

### TD-MPC2 (2023) - Scalable, Robust World Models
**Key Innovation**: Single 317M parameter agent across 80 tasks in 4 domains

**Architecture**:
- **Implicit world model** (no decoder!)
- TD-learning in latent space
- Local trajectory optimization (MPC)
- Multi-task, multi-domain training

**Key Features**:
- **Decoder-free**: Learns latent dynamics without reconstructing pixels
- **Scalable**: Performance improves with model size
- **Robust**: Single hyperparameter set across domains
- **Multi-task**: One agent, many environments

**Results**:
- 104 online RL tasks across 4 domains
- 317M parameter agent for 80 tasks
- Strong performance with fixed hyperparameters

**Why It Matters**:
- Shows world models scale like language models
- Decoder-free approach is more efficient
- Proves multi-domain world models work

**Paper**: https://arxiv.org/abs/2310.16828
**Code**: https://tdmpc2.com

---

### VICReg (2021) - Preventing Representation Collapse
**Key Innovation**: Explicit variance regularization to prevent collapse

**Problem**: Self-supervised models collapse to constant representations

**Solution**: Three regularization terms:
1. **Variance**: Ensure each dimension has sufficient variance
2. **Invariance**: Maximize similarity between augmented views
3. **Covariance**: Decorrelate dimensions (redundancy reduction)

**Loss Function**:
```
L = λ_var * Variance_Loss + λ_inv * Invariance_Loss + λ_cov * Covariance_Loss
```

**Variance Loss**:
```python
# Penalize low variance in each dimension
variance_loss = mean(relu(1 - std(z, dim=0)))
```

**Covariance Loss**:
```python
# Penalize correlation between dimensions
cov_matrix = cov(z)
covariance_loss = sum(cov_matrix ** 2) - sum(diag(cov_matrix) ** 2)
```

**Results**:
- Matches state-of-the-art on downstream tasks
- More stable training
- Clear theoretical justification

**Why It Matters for World Models**:
- World models often collapse to constant latents
- VICReg provides explicit collapse prevention
- Simpler than contrastive learning
- Can be added to any encoder

**Paper**: https://arxiv.org/abs/2105.04906
**Authors**: Bardes, Ponce, LeCun

---

### Hierarchical World Models (2024 Thesis)
**Key Innovation**: Multi-time-scale state space models

**Problem**: Standard SSMs can't capture hierarchical temporal structure

**Solution**: Two new formalisms:
1. **Hidden-Parameter SSMs**: Parameters change over time
2. **Multi-Time Scale SSMs**: Different dynamics at different time scales

**Key Ideas**:
- World has causal hierarchies (fast dynamics within slow dynamics)
- Single time-scale models miss this structure
- Hierarchical models can represent nonstationary dynamics

**Results**:
- Matches or exceeds transformer performance
- Better long-range predictions
- More interpretable

**Why It Matters**:
- Points to future of world models
- Connects to neuroscience (predictive processing)
- Suggests limitations of flat RSSM

**Paper**: https://arxiv.org/abs/2404.16078

---

## Environments

### Procgen Benchmark (OpenAI)
**16 procedurally-generated game-like environments**

**Why Procgen?**:
- Tests **generalization** (not just memorization)
- Fast (>4000 steps/sec per core)
- Randomized levels (can't memorize)
- Customizable (C++ source available)

**Environments**:
1. **bigfish** - Eat smaller fish, grow bigger
2. **bossfight** - Destroy boss starship
3. **caveflyer** - Navigate caves (Asteroids-like)
4. **chaser** - Collect orbs, avoid enemies (MsPacman-like)
5. **climber** - Platformer, climb and collect stars
6. **coinrun** - Simple platformer, collect coin
7. **dodgeball** - Dodge projectiles (Berzerk-like)
8. **fruitbot** - Scrolling game, collect fruit
9. **heist** - Steal gem, collect keys
10. **jumper** - Open world platformer
11. **leaper** - Cross lanes (Frogger-like)
12. **maze** - Navigate maze, find cheese
13. **miner** - Dig, collect diamonds (BoulderDash-like)
14. **ninja** - Platformer with throwing stars
15. **plunder** - Destroy enemy ships
16. **starpilot** - Side-scrolling shooter

**Observation**: 64x64x3 RGB images
**Action Space**: Discrete(15)
**Step Rate**: 15 Hz for humans

**Installation**:
```bash
pip install procgen
```

**Usage**:
```python
import gym
env = gym.make("procgen:procgen-coinrun-v0")
```

**Why Use Procgen?**:
- MiniGrid is too simple (7x7 observations)
- Atari is too slow and large
- Procgen is the sweet spot:
  - Fast enough for quick iteration
  - Complex enough to test generalization
  - Procedurally generated (no overfitting)

**Recommendation**:
- Start with MiniGrid (Phase 1-3)
- Move to Procgen for planning experiments (Phase 4-5)
- Use Procgen for final research results

**Paper**: https://arxiv.org/abs/1912.01588
**Code**: https://github.com/openai/procgen

---

## Architecture Deep Dive

### RSSM (Recurrent State-Space Model)
Used in PlaNet, Dreamer, DreamerV2, DreamerV3

**State**: s_t = (h_t, z_t)
- h_t: deterministic hidden state (GRU)
- z_t: stochastic latent state

**Dynamics**:
```
# Prior (prediction without observation)
p(z_t | h_t) = N(μ_prior, σ_prior)

# Posterior (with observation)
q(z_t | h_t, o_t) = N(μ_post, σ_post)

# Transition
h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})
```

**Training Objective**:
```
# Reconstruction loss
L_recon = -log p(o_t | h_t, z_t)

# KL divergence (prior vs posterior)
L_kl = KL(q(z_t | h_t, o_t) || p(z_t | h_t))

# Reward prediction
L_reward = -log p(r_t | h_t, z_t)

# Total loss
L = L_recon + β * L_kl + L_reward
```

**Free Bits** (DreamerV3):
- Don't penalize KL below threshold
- Prevents posterior collapse
- Allows some dimensions to be "free"

**Categorical Latents** (DreamerV3):
- Instead of Gaussian, use categorical distribution
- More stable for discrete environments
- Better for Atari, Minecraft

---

## Representation Collapse Solutions

### 1. Contrastive Learning (SimCLR, MoCo)
**Idea**: Maximize similarity between augmented views, minimize similarity with negatives

**Pros**: Well-studied, strong results
**Cons**: Requires negative samples, memory-intensive

### 2. BYOL (Bootstrap Your Own Latent)
**Idea**: Predict target network's representation from online network

**Pros**: No negatives needed
**Cons**: Mysterious why it works (EMA crucial)

### 3. VICReg
**Idea**: Explicit variance + covariance regularization

**Pros**: Clear justification, simple implementation
**Cons**: Newer, less battle-tested

### 4. Free Bits KL (DreamerV3)
**Idea**: Don't penalize KL below threshold

**Pros**: Simple, works in practice
**Cons**: Less theoretically grounded

### 5. Categorical Latents (DreamerV2/V3)
**Idea**: Use discrete instead of continuous

**Pros**: Naturally prevents collapse
**Cons**: Less flexible than continuous

**Recommendation for Your Project**:
- Start with simple MSE loss
- If collapse detected, add VICReg
- Consider categorical latents for discrete environments

---

## Practical Training Tips

### Data Collection
1. **Random policy first**: Simple, works for MiniGrid
2. **Iterative training**: Collect data with current policy, retrain
3. **Replay buffer**: Store transitions, sample randomly
4. **Prioritized replay**: Sample high-error transitions more often

### Hyperparameters
**Learning Rates**:
- Encoder: 1e-4 to 1e-3
- Transition model: 1e-4 to 1e-3
- Planner: 1e-4 to 3e-4

**Batch Size**: 64-256 (larger = more stable)

**Latent Dimension**: 64-256 (start with 128)

**KL Weight (β)**: 0.1-1.0 (DreamerV3 uses adaptive)

### Training Schedule
1. **Phase 1**: Train encoder + transition (50-100 epochs)
2. **Phase 2**: Add decoder, train autoencoder (30 epochs)
3. **Phase 3**: Freeze encoder, train planner

### Monitoring
**Must Track**:
- Prediction loss (MSE)
- Reconstruction loss (if decoder)
- Latent norm (should be stable)
- Latent variance (should not collapse to 0)
- KL divergence (if using VAE)

**Red Flags**:
- Loss decreases but latent variance → 0 (collapse!)
- Predictions look good but planning fails (representation issue)
- Training unstable (learning rate too high)

### Debugging Checklist
1. **Check data**: Visualize trajectories, check preprocessing
2. **Check encoder**: Can it reconstruct images?
3. **Check transition**: Does 1-step prediction work?
4. **Check multi-step**: How fast does error accumulate?
5. **Check planner**: Does it beat random policy?

---

## Compute Requirements

### MiniGrid (7x7 observations)
**Training**: 1-2 hours on single GPU
**Memory**: < 4GB VRAM
**Storage**: ~100MB for trajectories

### Procgen (64x64 observations)
**Training**: 4-8 hours on single GPU
**Memory**: 6-8GB VRAM
**Storage**: ~1GB for trajectories

### Atari (84x84 observations)
**Training**: 12-24 hours on single GPU
**Memory**: 8-12GB VRAM
**Storage**: ~5GB for trajectories

### Recommendations
**Cloud GPU Options**:
- **Lambda Labs**: $0.50/hr (A100)
- **RunPod**: $0.40/hr (A5000)
- **Vast.ai**: $0.30/hr (RTX 3090)
- **Google Colab Pro**: $10/month (A100)

**Budget Estimate**:
- Phase 1-3 (MiniGrid): $5-10
- Phase 4-5 (Procgen): $20-40
- Phase 6 (Optional): $10-20

**Total**: $35-70 for full project

---

## Open Source Implementations

### DreamerV3 (Official)
**Language**: JAX (not PyTorch!)
**Complexity**: High
**Quality**: Excellent
**Link**: https://github.com/danijar/dreamerv3

**Pros**:
- Official implementation
- Well-tested
- Reproduces paper results

**Cons**:
- JAX (learning curve if you know PyTorch)
- Complex codebase
- Hard to modify

### Tianshou
**Language**: PyTorch
**Complexity**: Medium
**Quality**: Good
**Link**: https://github.com/thu-ml/tianshou

**Pros**:
- PyTorch-based
- Clean code
- Multiple algorithms

**Cons**:
- Less focused on world models
- Smaller community

### TorchDreamer
**Language**: PyTorch
**Complexity**: Medium
**Quality**: Good
**Link**: https://github.com/EloiZer/TorchDreamer

**Pros**:
- PyTorch reimplementation of Dreamer
- Easier to understand
- Good for learning

**Cons**:
- Not official
- May not reproduce exact results

### Recommendation
**For Learning**: Read DreamerV3 code, implement simplified version in PyTorch
**For Research**: Build your own (as planned)
**For Production**: Use DreamerV3 (JAX) if possible

---

## Research Questions Refined

### Primary Question
**How does latent prediction quality affect downstream planning performance?**

**Operationalization**:
- **Independent variable**: Latent prediction MSE
- **Dependent variable**: Planning reward
- **Confounds**: Environment complexity, latent dimension, prediction horizon

### Secondary Questions

1. **Latent Dimensionality**
   - What is the optimal latent dimension for MiniGrid? Procgen?
   - Does larger latent always improve performance?
   - Is there a "sweet spot" where performance plateaus?

2. **Prediction Horizon**
   - How does multi-step prediction error affect planning?
   - What horizon is needed for different tasks?
   - Can we predict when planning will fail based on prediction error?

3. **Memory and Partial Observability**
   - Does memory (LSTM) improve performance on POMDPs?
   - How much memory is needed?
   - Does memory help with prediction quality?

4. **Representation Quality**
   - Does VICReg prevent collapse in world models?
   - Do categorical latents outperform continuous?
   - How does representation quality correlate with planning?

5. **Sample Efficiency**
   - How much data is needed to train a good world model?
   - Does iterative data collection help?
   - Can world models beat model-free baselines?

---

## Experimental Design

### Experiment 1: Latent Size Ablation
**Hypothesis**: Larger latent improves prediction but may overfit

**Setup**:
- Latent dims: [32, 64, 128, 256, 512]
- Environment: MiniGrid-Empty-5x5-v0
- Metric: Planning reward, prediction MSE

**Expected Result**: Performance plateaus around 128-256

### Experiment 2: Prediction Horizon
**Hypothesis**: Longer horizon improves planning but prediction error accumulates

**Setup**:
- Horizons: [1, 5, 10, 20, 50]
- Environment: MiniGrid-FourRooms-v0
- Metric: Planning reward, success rate

**Expected Result**: Optimal horizon around 10-20

### Experiment 3: Memory vs Memoryless
**Hypothesis**: Memory helps on POMDPs but not on MDPs

**Setup**:
- Models: [Memoryless, LSTM, GRU]
- Environments: [MiniGrid-Empty (MDP), MiniGrid-Memory (POMDP)]
- Metric: Planning reward, episode length

**Expected Result**: Memory helps on POMDP, neutral on MDP

### Experiment 4: Representation Collapse Prevention
**Hypothesis**: VICReg prevents collapse and improves planning

**Setup**:
- Methods: [None, VICReg, Free Bits, Categorical]
- Environment: MiniGrid-DynamicObstacles-v0
- Metric: Latent variance, prediction MSE, planning reward

**Expected Result**: VICReg or categorical prevents collapse

### Experiment 5: Sample Efficiency
**Hypothesis**: World models are more sample-efficient than PPO

**Setup**:
- Methods: [World Model, PPO, DQN, Random]
- Data budgets: [1000, 5000, 10000, 50000] transitions
- Environment: Procgen-coinrun
- Metric: Reward vs data size

**Expected Result**: World model wins at low data budgets

---

## Timeline Refined

| Week | Phase | Goal | Key Deliverable |
|------|-------|------|-----------------|
| 1 | 0-1 | Setup + Trajectories | Working environment, data collected |
| 2 | 1 | Train world model | Prediction loss < 0.1 |
| 3 | 2 | Visual validation | Reconstructions, t-SNE |
| 4 | 3 | Multi-step prediction | Horizon analysis |
| 5-6 | 4 | Planning | CEM planner, beat random |
| 7-8 | 5 | Research experiments | Ablation studies complete |
| 9-10 | 6 | Memory (optional) | POMDP experiments |
| 11-12 | 7 | Paper writing | Draft complete |
| 13 | 8 | Polish | Code cleanup, GitHub repo |

---

## Success Metrics (Updated)

### Technical
- [ ] World model predicts next latent (MSE < 0.1)
- [ ] Reconstructions capture key features
- [ ] Multi-step prediction works (t+5 MSE < 0.5)
- [ ] Planner beats random policy by 20%
- [ ] Beat PPO sample efficiency at 5000 transitions

### Research
- [ ] Answer primary research question with statistical significance
- [ ] Complete 3+ ablation studies
- [ ] Compare to 3+ baselines (PPO, DQN, Random)
- [ ] Document all experiments with plots and tables

### Paper
- [ ] Clear research question and hypothesis
- [ ] Reproducible experiments (code + data)
- [ ] Professional write-up (8-12 pages)
- [ ] GitHub repo with clean code

### Portfolio
- [ ] Public GitHub repo with README
- [ ] Demo video (2-3 minutes)
- [ ] Blog post explaining approach
- [ ] Presentation slides (15-20 slides)

---

## Next Steps

### Immediate (This Week)
1. **Read papers** (2-3 days)
   - World Models (2018) - Focus on architecture
   - DreamerV3 (2023) - Focus on techniques
   - VICReg (2021) - Understand collapse prevention

2. **Setup environment** (1 day)
   ```bash
   conda create -n world-models python=3.11
   conda activate world-models
   pip install torch torchvision gymnasium minigrid procgen
   pip install stable-baselines3 wandb hydra-core
   pip install matplotlib scikit-learn
   ```

3. **Test environments** (1 day)
   - Run MiniGrid example
   - Run Procgen example
   - Verify GPU access

### Week 2
1. **Collect trajectories** (2 days)
   - Random policy on MiniGrid
   - Store 1000 episodes
   - Verify data quality

2. **Build encoder** (2 days)
   - CNN encoder (7x7x3 → 128-dim)
   - Test forward pass
   - Visualize features

3. **Build transition model** (2 days)
   - MLP transition (z_t, a_t → z_{t+1})
   - Test forward pass
   - Check output dimensions

4. **Start training** (1 day)
   - Implement training loop
   - Setup WandB logging
   - Run first epoch

### Week 3
1. **Complete training** (3 days)
   - Train for 50 epochs
   - Monitor loss curves
   - Save checkpoints

2. **Evaluate** (2 days)
   - Compute prediction error
   - Visualize latent space (t-SNE)
   - Check for collapse

3. **Debug** (2 days)
   - Fix any issues
   - Tune hyperparameters
   - Retrain if needed

---

## Conclusion

This additional research strengthens the project plan:

**New Papers**:
- DreamerV2: Discrete latents work better for Atari
- TD-MPC2: World models scale, decoder-free approach
- VICReg: Explicit collapse prevention
- Hierarchical WMs: Future direction

**New Environments**:
- Procgen: Perfect for testing generalization
- 16 diverse environments
- Fast and customizable

**Refined Approach**:
- Start with MiniGrid (simple)
- Move to Procgen (generalization)
- Use VICReg if collapse detected
- Consider discrete latents for discrete environments

**Ready to Build**:
- Research complete
- Plan bulletproof
- Code templates ready
- Timeline realistic

**Next**: Start Phase 1 implementation!

## 6. Discussion

### 6.1 What Worked

The RSSM curriculum produced a real, working learned world model:
rollouts stay in distribution across 6 steps (MSE 0.006--0.012), the
KL balances prior and posterior at unit variance, and visual
inspection confirms the agent position is roughly tracked. This
is the first world model in the project that genuinely functions.

The binary HER reward predictor achieves 97.6\% accuracy on
posterior-encoded states, confirming that the RSSM latent has the
information needed to distinguish goal from non-goal states.

### 6.2 What Didn't Work

Planning nonetheless fails to significantly outperform random.
The heuristic-reward diagnostic pins this on the world model:
even exact pixel-space distance to the goal cannot guide the planner
reliably. The model's 6-step rollout MSE of 0.01 is impressive in
pixel space but insufficient for cell-level position tracking in
navigation.

### 6.3 The Data Hypothesis

Three pieces of evidence point to data coverage as the root cause:

1. **The dataset is small.** 1,000 trajectories with 83k transitions,
   of which only $\sim 40\%$ reach the goal (on the easy env).
   Dreamer-style models train on millions of transitions collected
   by an actively improving policy.

2. **The dataset is uniform.** Random actions produce mostly
   "turn in place" and "bump wall" transitions. Goal-directed
   action sequences -- the kind the planner actually needs to
   evaluate -- are vanishingly rare.

3. **The RNN has little to learn from.** At each step, the GRU
   receives a latent $z_{t-1}$ and action $a_{t-1}$ that mostly
   lead to no positional change. The model cannot learn precise
   dynamics from data where dynamics are absent.

### 6.4 Limitations

This study has several limitations:

- **Single observation modality.** MiniGrid's 7x7 symbolic
  observations are far simpler than the RGB images used in
  Dreamer benchmarks. Different failure modes may dominate in
  high-dimensional environments.
- **Random-policy data only.** The decision to use random
  trajectories was deliberate (to isolate the world model's
  data requirements), but iterative data collection with an
  improving policy is standard practice in MBRL.
- **No value function.** Dreamer addresses planning distribution
  shift with a value function trained on prior rollouts. We
  de-prioritized this fix after the heuristic ablation identified
  the world model itself as the bottleneck, but a value function
  may still help in conjunction with better data.

### 6.5 Recommendations

For practitioners building latent-world-model pipelines:

1. **Diagnose encoder collapse first.** Before training any dynamics
   model, check that the encoder produces a diverse latent space.
   Pairwise distance histograms are more informative than reconstruction
   loss.

2. **Test multi-step prediction with a horizon sweep.** Flat error
   across horizons indicates transition collapse. The error should
   grow with step count; if it doesn't, the model is cheating.

3. **Validate with heuristic rewards.** Before debugging a learned
   reward predictor, check whether the world model itself can
   support planning with an oracle reward signal. This isolates
   the bottleneck.

4. **Don't trust pixel MSE for control tasks.** A model that
   reconstructs well and rolls out stably may still lack the
   precision needed for planning. Task-specific metrics
   (e.g., position error in navigation) are more informative.

5. **Collect goal-directed data.** Random-policy data is convenient
   but may not contain the action sequences the planner needs to
   discover. A mix of random and policy-driven data is more robust.

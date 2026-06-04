## 5. Failure Analysis

The central contribution of this paper is a systematic decomposition
of why the latent-world-model recipe fails. We identify four distinct
collapse modes, each with its own mechanism, and each requiring a
different fix.

### 5.1 Failure 1: Encoder Collapse

**Mechanism:** The CNN encoder + MSE decoder is trained to reconstruct
7x7x3 observations. Since most pixels are walls (pixel value 1) and
the MSE loss is dominated by easy-to-predict static regions, the
encoder learns to output a near-constant vector. The decoder
reciprocates by learning to render the average frame.

**Symptom:** Pairwise latent distance across the dataset is $\sim 0.0075$.
Per-dimension standard deviation is $\sim 0.0$.

**Detection method:** Compute the pairwise Euclidean distance matrix
on a sample of encoder outputs. If the mean distance is orders of
magnitude smaller than the latent dimension norm (e.g., 0.0075 for a
128-dim latent), collapse is present.

**Fix:** VICReg regularization (variance hinge + covariance penalty)
forces the encoder to use its full representational capacity. After
fix: pairwise distance 12.7, std 0.84.

### 5.2 Failure 2: Transition Collapse

**Mechanism:** With a collapsed encoder, the transition model's
optimal strategy is $z_{t+1} = z_t$, which achieves near-zero MSE
because all latents are nearly identical. Even with a healthy encoder,
a deterministic MLP $f(z_t, a_t)$ can learn identity as a low-loss
fixed point when the dataset contains many "no-op" transitions
(turning in place, bumping into walls).

**Symptom:** Training MSE $< 10^{-6}$. Multi-step prediction errors
do not accumulate with horizon -- they stay flat at the single-step
level.

**Detection method:** Compare $\|z_{t+1} - z_t\|$ against
$\|\hat{z}_{t+1} - z_t\|$ for each action type. If the predicted
change is systematically near zero while the true change is non-zero
for certain actions, collapse is active. Also: multi-step error that
does not grow with step count is a strong indicator.

**Fix:** Residual parameterization $z_{t+1} = z_t + \alpha \Delta$
forces non-trivial predictions. The RSSM's stochastic latent with
KL divergence between prior and posterior also prevents this: even
if the mean predictions coincide, the distributional mismatch
produces a non-zero KL penalty.

### 5.3 Failure 3: Reward Predictor Collapse

**Mechanism:** Sparse terminal rewards ($+1$ on goal, $0$ otherwise)
mean that $>99.9\%$ of transitions have reward 0. The reward predictor
converges to outputting $\hat{r} \approx 0$ for all inputs, achieving
near-zero MSE.

This is compounded by encoder collapse: if all latents are
indistinguishable, even a dense reward function (e.g., HER with
soft distance) cannot separate goal states from non-goal states.
The soft-label mean (0.94) becomes the network's output for all
inputs.

**Symptom:** Reward predictor output is constant across all states.
CEM planner has no gradient to optimize.

**Detection method:** Histogram of reward predictions over a held-out
set of states. If all predictions fall in a narrow band (e.g.,
$0.27 \pm 0.01$ for soft-distance HER, or $0.00 \pm 0.01$ for sparse),
the predictor has collapsed.

**Fix:** Binary HER with 50/50 goal/non-goal sampling. This creates
a balanced classification task that forces the predictor to learn
distinctive features. Combined with the RSSM's structured latent,
this achieves 97.6\% accuracy.

### 5.4 Failure 4: Planning Distribution Shift

**Mechanism:** The reward predictor is trained on posterior-encoded
latents $z \sim q(z \mid h, o)$ -- the stochastic latent inferred
from an actual observation with the GRU at the correct time step.
At planning time, the predictor is evaluated on prior-predicted
latents $\hat{z} \sim p(z \mid h, a_1, \ldots, a_t)$ -- open-loop
rollouts that see only the initial state and a sequence of actions.
These two distributions differ systematically: posterior latents
are "corrected" by the observation; prior latents accumulate
open-loop error.

**Symptom:** The predictor assigns high reward ($\sim 0.86$) to
posterior-encoded states (even non-goal states) and zero reward
($\sim 0.00$) to any prior-rollout state, regardless of action
quality.

**Detection method:** Compare predictor output on posterior-encoded
$z$ vs prior-predicted $\hat{z}$ for the same underlying state. A
systematic gap indicates distribution shift.

**Status:** This failure was partially diagnosed and then
**de-prioritized** after the heuristic-reward ablation (Section 4.6).
Replacing the learned predictor with pixel-space distance to goal
did not improve planning, indicating that the world model's
control-relevant precision -- not the reward predictor's domain
shift -- is the primary bottleneck.

### 5.5 Root Cause: Pixel MSE vs. Control-Relevant Prediction

The heuristic-reward ablation is the most informative experiment in
this study. By removing the learned predictor entirely, we isolate
the world model itself as the bottleneck. The RSSM achieves excellent
pixel-level prediction (MSE 0.009) and stable open-loop rollouts,
but these metrics do not measure what the planner needs: accurate
tracking of agent and goal position at the single-cell level.

The decoder's sigmoid output in $[0, 1]$ makes pixel MSE a misleading
metric. On a 7x7 grid with 5 object types, a prediction that places
the agent one cell off from its true position might still achieve
low pixel MSE because most pixels are walls. The planner, however,
needs to distinguish position (1,1) from position (2,1) -- a
distinction that may account for only a few pixels in the image.

This is the central finding of the paper: **pixel-level prediction
quality is a necessary but not sufficient condition for planning
with learned world models.** The field has recognized this
repeatedly \cite{lambert2020objective, janner2019trust},
but the simplicity of the MSE objective makes it a persistent trap.

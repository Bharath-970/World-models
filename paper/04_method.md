## 3. Method

We follow the standard latent-world-model recipe: encode observations,
predict transitions in latent space, and plan via latent rollouts.
This section describes the architectural progression from a simple
deterministic autoencoder to the full RSSM.

### 3.1 Environments

We use two MiniGrid \cite{chevalier2018minigrid} navigation tasks:

- **MiniGrid-Empty-5x5-v0**: A 5x5 grid with the agent starting at
  position (1,1) facing right and the goal at (3,3). The agent must
  navigate to the green goal square. Optimal path: 4 steps.
- **MiniGrid-FourRooms-v0**: A 19x19 grid divided into four rooms
  connected by narrow doorways. The agent and goal are placed in
  random rooms. Requires multi-room navigation.

Both environments return 7x7x3 RGB image observations as uint8 arrays
with pixel values in the set $\{0, 1, 2, 5, 8\}$ (representing different
grid-object categories). Rewards are sparse and terminal: $+1$ on
reaching the goal, $0$ otherwise.

### 3.2 Data

We collect 1,000 episodes using a uniform random policy. On Empty-5x5,
39.3\% of random episodes reach the goal; on FourRooms, only 3.8\%
succeed. Each episode is capped at 100 steps. Total: ~83k transitions
for Empty-5x5, ~196k for FourRooms.

Observations are stored as raw uint8 and normalized at training time
by dividing by 8 (the maximum pixel value). This normalization is
critical: an earlier version of the pipeline divided by 255 (standard
for RGB images), which placed the decoder's sigmoid output range
$[0, 1]$ and the ground-truth target range $[0, 0.031]$ in incompatible
units, producing a meaningless reconstruction loss.

### 3.3 World Model v1: Deterministic Autoencoder

Our initial world model consists of three components:

**Encoder** $E$: A CNN mapping $7\times7\times3 \to \mathbb{R}^{128}$.
Three convolutional layers (3$\to$16, 16$\to$32, 32$\to$64 with 3x3
kernels, stride 2, padding 1) followed by a linear projection to
128-dimensional latent $z_t$.

**Transition model** $T$: An MLP $f_\theta(z_t, a_t) \to z_{t+1}$.
The action $a_t$ (one-hot encoded, 7 actions) is concatenated with
$z_t$ and passed through two hidden layers (256, 256) with LayerNorm
and ReLU, outputting a 128-dimensional prediction $\hat{z}_{t+1}$.

**Decoder** $D$: A transposed-CNN mirroring the encoder, mapping
$z_t \to \hat{o}_t$ with a final sigmoid activation.

Training minimizes:
$$\mathcal{L} = \|z_{t+1} - f_\theta(z_t, a_t)\|^2 + \|\hat{o}_t - o_t\|^2$$

### 3.4 World Model v2: Residual Transition

The deterministic MLP transition exhibited a subtle collapse: the
LayerNorm $\to$ MLP $\to$ LayerNorm parameterization made identity
prediction $z_{t+1}=z_t$ a low-loss fixed point. We reparameterized
to a residual form:
$$z_{t+1} = z_t + \alpha \cdot \Delta_\theta(z_t, a_t)$$
where $\alpha = 0.1$ and $\Delta$ is a small MLP. This forces the
model to predict changes rather than absolute latents, analogous
to residual connections in ResNets \cite{he2016deep}.

### 3.5 World Model v3: VICReg Encoder

The encoder collapsed to outputting a near-constant vector (pairwise
distance $\sim 0.0075$ across the dataset), because the MSE reconstruction
loss is dominated by easy-to-predict wall pixels. We applied VICReg
\cite{bardes2022vicreg}, adding variance and covariance regularization
to the encoder output:

$$\mathcal{L}_{\text{VICReg}} = \lambda_v \cdot \text{ReLU}(1 - \text{std}(z))
+ \lambda_c \cdot \sum_{i \neq j} [\text{Cov}(z)]_{ij}^2$$

with $\lambda_v = \lambda_c = 1$. This alone expanded the latent to
$\text{std} \approx 0.84$ with pairwise distance 12.7.

### 3.6 World Model v4: RSSM (Dreamer-style)

Our final architecture follows the Recurrent State-Space Model (RSSM)
from PlaNet \cite{hafner2019planet} and Dreamer \cite{hafner2020dreamer}.
The full model is:

**Trunk encoder**: A CNN (same structure as v1) projecting observations
to 256-dimensional features.

**Posterior**: $q(z_t \mid h_t, o_t) = \mathcal{N}(\mu_q, \sigma_q)$,
parameterized by an MLP that takes $(h_t, \text{features})$ and outputs
mean and log-std.

**Prior**: $p(z_t \mid h_t) = \mathcal{N}(\mu_p, \sigma_p)$, an MLP
taking only $h_t$.

**GRU**: $h_t = \text{GRU}(h_{t-1}, [z_{t-1}, a_{t-1}])$, the
deterministic recurrent state that carries temporal information.

**Decoder**: Transposed CNN mirroring the trunk encoder, reconstructing
the observation from $z_t$.

**Overshoot heads**: Three MLPs predicting $z_{t+k}$ from $h_t$ for
$k=1, 3, 5$, enabling multi-step consistency (latent overshooting).

Total parameters: 1.18M. Latent dimension: 64. GRU hidden dimension: 256.

#### 3.6.1 Three-Stage Curriculum

Following the principle of incremental validation, the RSSM is trained
in three stages:

**Stage A: Autoencoder Warmup.** Train the trunk encoder, posterior,
and decoder with KL divergence to a fixed $\mathcal{N}(0, I)$ prior:
$$\mathcal{L}_A = \| \hat{o}_t - o_t \|^2 + \beta \cdot \text{KL}(q(z_t \mid o_t, h_t) \;\|\; \mathcal{N}(0, I))$$
with $\beta = 0.001$. No GRU or learned prior yet. This ensures the
encoder produces usable latents before dynamics are introduced.

**Stage B: Dynamics.** Load Stage A, add the GRU and learned prior.
Train with:
$$\mathcal{L}_B = \| \hat{o}_t - o_t \|^2 + \beta \cdot \text{KL}(q(z_t \mid h_t, o_t) \;\|\; p(z_t \mid h_t))$$
where $\beta = 0.1$ and the KL is now between learned distributions.

**Stage C: Latent Overshooting.** Load Stage B, add the three overshoot
heads. The loss combines the Stage B objective with multi-step KL:
$$\mathcal{L}_C = \mathcal{L}_B + \sum_{k \in \{1,3,5\}} w_k \cdot \text{KL}
\big( \text{sg}[q(z_{t+k})] \;\|\; p(z_{t+k} \mid h_t, a_{t:t+k-1}) \big)$$
with weights $w = [1.0, 0.5, 0.25]$ per the Dreamer recipe.

### 3.7 Reward Predictor

For planning, we train a binary goal-conditioned reward predictor.
Given a latent $z_t$ and a goal latent $z_g$ (posterior-encoded), the
predictor classifies whether $z_t$ is at the goal:
$$\hat{r} = \sigma(\text{MLP}([z_t, z_g]))$$

Training uses Hindsight Experience Replay (HER) \cite{andrychowicz2017hindsight}:
for each timestep, with 50\% probability we pair $(z_t, z_{\text{goal}})$
(label 1), and with 50\% probability we pair $(z_t, z_{\text{non-goal}})$
from a random future state (label 0). Binary cross-entropy loss.

This binary formulation avoids the saturation issue of soft-distance
HER, where the predictor learns to output the mean soft label (0.94)
for all inputs.

### 3.8 Planner

We use the Cross-Entropy Method (CEM) \cite{rubinstein1999cross} for
planning in latent space:

1. Sample $N$ action sequences of horizon $H$ from a uniform distribution.
2. Roll out each sequence through the RSSM prior (open-loop) to obtain
   predicted latents $\hat{z}_{1:H}$.
3. Score each rollout with the reward predictor: $R = \sum_{t=1}^H \gamma^t \hat{r}_t$.
4. Select the top $E$ elite sequences and refit the action distribution.
5. Repeat for $I$ iterations.
6. Execute the first action of the best sequence, then re-plan.

Default parameters: $N=256$ samples, $E=16$ elites, $I=5$ iterations,
$H=4$ (Empty-5x5) or $H=6$ (FourRooms).

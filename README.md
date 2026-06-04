# World Models — Failure Diagnosis on MiniGrid

**Reconstruction Is Not Enough: Diagnosing World Model Failures in MiniGrid**

This repository reproduces and extends the analysis from the paper. It builds an RSSM-based world model on MiniGrid navigation tasks, documents four distinct collapse modes, and demonstrates that strong reconstruction metrics do not guarantee planning utility — the latent representation can achieve near-perfect pixel reconstruction (MSE 0.009) while encoding almost no spatial information (position probe R² = 0.04).

## Paper

Latest PDF: [`paper/researchpaper.pdf`](paper/researchpaper.pdf)

## Key Results

| Model | Recon MSE | Position R² | Planner (Empty-5x5) |
|-------|-----------|-------------|---------------------|
| Deterministic AE (collapsed) | 0.067 | ~0 | 0% |
| RSSM (random data) | 0.0092 | 0.28 | 30% ± 9% |
| RSSM (PPO data) | 0.0010 | 0.04 | 10% ± 6% |

## Collapse Modes Documented

1. **Normalization Bug** — MiniGrid pixel values are {0,1,2,5,8}, not [0,255]
2. **Encoder Collapse** — MSE dominated by static walls; encoder outputs near-constant vector
3. **Transition Collapse** — MLP learns identity as low-loss fixed point
4. **Reward Predictor Collapse** — Sparse rewards cause constant near-zero output
5. **Planning Distribution Shift** — Predictor trained on posterior latents, evaluated on prior rollouts

## Repository Structure

```
├── paper/              # Paper source (markdown) and PDF
│   ├── researchpaper.md
│   └── researchpaper.pdf
├── figures/            # Generated figures
├── analysis/           # Analysis scripts (position probe, rollout viz)
├── models/             # Trained model checkpoints
├── experiments/        # Experiment configs and logs
├── envs/               # Environment wrappers
└── configs/            # Configuration files
```

## Citation

```bibtex
@misc{varma2026reconstruction,
  title={Reconstruction Is Not Enough: Diagnosing World Model Failures in MiniGrid},
  author={S. Bharath Varma},
  year={2026},
  month={jun},
  howpublished={\url{https://github.com/Bharath-970/World-models}}
}
```

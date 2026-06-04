## 7. Conclusion

We built a latent-world-model planner for MiniGrid navigation,
starting from a textbook encoder-transition-decoder pipeline and
progressing through four collapse modes to a full RSSM with
curriculum training. Each collapse was diagnosed with specific
numerical tests and fixed with targeted interventions.

The final system -- RSSM with binary HER reward predictor and CEM
planner -- produces stable latent rollouts and accurate goal-state
classification, yet plans no better than random. A heuristic-reward
ablation isolates the world model's control-relevant precision as
the bottleneck: pixel-level prediction quality is necessary but not
sufficient for planning.

The evidence implicates data coverage. One thousand random-policy
trajectories on a 5x5 grid do not provide the diverse, goal-directed
transitions needed to learn dynamics precise enough for navigation
planning. This is not a limitation of the RSSM architecture but of
the data it was trained on -- a reminder that world models are
fundamentally data-driven, and that the data must match the task.

**Future work** should collect goal-directed data (e.g., from a
partially-trained PPO policy) and retrain the same RSSM architecture
with no modifications. If planning improves, the data hypothesis is
confirmed. If not, architectural changes (value function bootstrap,
discrete latents, separate position/goal representations) become
the next candidate.

The diagnostic framework developed here -- encoder variance checks,
multi-step horizon analysis, heuristic reward validation -- is
architecture-independent and should transfer to any latent-world-model
project.

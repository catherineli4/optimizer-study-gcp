# Gaussian weight-perturbation setup

How the perturbation study in this repo is defined, so it can be reproduced or
extended. Code: `new_utils/perturb_weights.py` (the math), `launch_jolmo/perturb.py`
(sweep construction), `launch_jolmo/training.py` (the `PerturbedModel` artifact),
`launch_jolmo/pretraining_matrix.py` (the wide sweep + evals).

## The noise

Per tensor, independently, isotropic Gaussian noise scaled to the weight's own
magnitude:

    s   = ||W||_F                      # Frobenius norm of the tensor
    std = gamma * s / sqrt(W.numel())  # = gamma * RMS(W)
    W'  = W + eps,   eps ~ N(0, std^2) elementwise

Dividing by `sqrt(numel)` is what makes gamma dimension-independent: the
per-entry std tracks the typical entry magnitude, so the *relative* whole-tensor
perturbation

    ||eps||_F / ||W||_F  ~=  gamma

holds for tensors of any shape. **gamma is therefore a relative noise level, not
an absolute std** — 0.02 means "2% of the weight's Frobenius norm", comparably
across layers and across model sizes.

Applied to **every float tensor** in the state dict (fp32/fp16/bf16). Non-float
tensors are copied through unchanged. `param_names` optionally restricts the
perturbation to a named allow-list (used by the per-matrix study,
`new_utils/evaluate_per_matrix_perturb.py`); the main sweep leaves it unset, so
all weights are perturbed.

One noise draw per (base, gamma): `PerturbedModel.seed` defaults to **64** and is
part of the artifact, so a rerun reproduces the same perturbation bit-for-bit.
The multi-seed variant (`build_multi_seed_perturbed_models`) instead writes N
directions under `seed_000/`, `seed_001/`, … for one (base, gamma), which is what
lets you separate direction variance from the gamma trend.

## Gamma grids

| constant | values | use |
|---|---|---|
| `PERTURB_WIDE_GAMMAS` | 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.13 | **the main sweep** (9 levels) |
| `LADDER_GAMMAS` | 0.002, 0.005, 0.007, 0.01, 0.02, 0.04, 0.08, 0.16 | multi-seed ladder |
| `DEFAULT_GAMMAS` | 0.02 | fallback when none given |

## Which models get perturbed

`perturb_wide_bases = tuned_bases_for(chins)` where `chins` is every chinchilla
with a tuned LR at the active size. So the sweep runs on the **best-LR (tuned)
pretrain of each (chinchilla, optimizer) cell** — one adamw and one muon base
per chinchilla — not on every LR in the sweep.

Both naming schemas appear among the bases (`MuonExpt3-…` and `PTSweep{SIZE}-…`);
they are the same configuration under two names, so parse the prefix off before
grouping or you will double-count.

## Naming

    PerturbedModel/{base}_perturbed_{gamma_tag}

`gamma_tag` comes from `get_perturbed_model_name`:

```python
f"{gamma:.2e}".replace("e-0", "e-").replace("e+0", "e+").replace(".", "_")
```

so 0.02 -> `2_00e-2`, 0.005 -> `5_00e-3`, 0.13 -> `1_30e-1`. A single-matrix run
appends `_param_{tag}`. Evals land at
`Optim-{SIZE}-tuning/ModelEvaluation/{perturbed_name}-eval.json`, with the same
`by_label` structure as any pretrain eval (`DCLM_heldout`, `C4_val`, `Books_val`,
`Wiki_val`, `Reddit_val`).

Checkpoint written to `PerturbedModel/{run_name}/final-unsharded/model.pt`,
matching where the script writes and where the existence check looks.

## Verified coverage (30M, as an example)

108 perturbed evals = **12 bases x 9 gammas**, exactly `PERTURB_WIDE_GAMMAS`:

    5_00e-3  1_00e-2  2_00e-2  3_00e-2  4_00e-2
    5_00e-2  7_00e-2  1_00e-1  1_30e-1

12 bases each.

## Reading the results

**Subtract the unperturbed baseline.** On raw post-perturbation loss muon looks
better simply because it starts lower; once degradation
(`loss(perturbed) - loss(base)`) is plotted instead, muon degrades *more*. Both
views are in the repo and they point opposite ways, so always say which one a
figure shows:

- `new_utils/plot_perturb_heatmap.py` — size x chinchilla heatmap per gamma,
  cell = muon minus adamw
- `new_utils/plot_perturb_chinchilla.py` — post-perturbed loss, and degradation,
  vs chinchilla per size
- `new_utils/plot_perturb_metrics.py` — loss / accuracy / CE against gamma

Colour scales are percentile-based; a full 0.005-0.13 range washes out the small
gammas, which is why the restricted 0.005-0.02 variants exist.

# h/kappa (critical-noise ratio) — data spec

Everything needed to download the results and plot them elsewhere. Produced by
`new_utils/margin_stats.py` (the computation), `new_utils/margin_stats_sweep.py`
(the sweep) and plotted by `new_utils/plot_margins.py`,
`new_utils/plot_margin_hist.py`.

## What the numbers are

From "An Implicit Closed Form for the Critical Weight Noise". At one validation
position with gold token `t` and logits `z_j`:

    m_j    = z_t - z_j        (j != t, the margins)
    h      = mean_{j!=t} m_j
    kappa  = sd_{j!=t} m_j    (population sd, denominator V-1)
    lambda = min_{j!=t} m_j

The result that makes this worth measuring:

    sigma* = f(h/kappa) * ||x|| / ||M||_F ,   f strictly increasing

`sigma*` is the weight-noise level at which the model breaks at that position,
and `f` is fixed by (vocab size, tolerance) alone. So **h/kappa carries the
entire model-dependent part of the brittleness ordering** — higher h/kappa means
larger sigma* means more robust — and two optimizers can be ranked without
perturbing anything. `sigma* := 0` where `lambda <= 0`.

Computed identities used (so you can re-derive, or check a reimplementation):

    h      = z_t - mean_{j!=t} z_j
    kappa  = sd_{j!=t} z_j        (the shift by z_t cancels)
    lambda = z_t - max_{j!=t} z_j

## Where the data is

    gs://cmu-gpucloud-catheri4/margins/
      summary/<SIZE>/<run_name>.json     one per config — start here
      summary/<SIZE>/<run_name>.npz      same summary, npz-wrapped
      per-token/<SIZE>/<run_name>.npz    full per-position arrays (~10 MB each)
      figures/                           png + pdf of everything below
      margins-h-kappa.csv                the flat table — easiest to plot from

`<SIZE>` in `{30M, 60M, 100M, 300M, 600M}`. Total 730 MiB, mostly `per-token/`.

    gsutil -m cp -r gs://cmu-gpucloud-catheri4/margins/summary .   # 616 KB
    gsutil cp gs://cmu-gpucloud-catheri4/margins/margins-h-kappa.csv .

## The flat table

`margins-h-kappa.csv`, 74 rows, one per config:

    model_size,chinchilla,optimizer,h_over_kappa_mean,h_over_kappa_median,
    p10,p90,h_mean,kappa_mean,top1_accuracy,frac_lambda_positive,n_positions

That is enough for every summary plot; the JSONs add only `n_lambda_positive`,
`n_ties_at_gold`, `sequence_length`, `instances` and the local input paths.

## Summary JSON keys

| key | meaning |
|---|---|
| `n_positions` | scored positions = instances x (sequence_length - 1) |
| `n_lambda_positive` / `frac_lambda_positive` | positions with `lambda > 0` |
| `top1_accuracy` | argmax correctness — see the tie caveat |
| `n_ties_at_gold` | positions where the gold logit ties the max |
| `h_over_kappa_{mean,median,p10,p90}` | over `lambda > 0` positions only |
| `h_mean`, `kappa_mean` | the two components, same subset |

## Per-token npz

Four `float32` arrays of length `n_positions`, flattened `(instances, L-1)` in
row-major order, plus a `summary` array (`uint8` JSON bytes):

    h, kappa, lam, h_over_kappa

`h_over_kappa` is **NaN wherever `lambda <= 0`** (~71% of positions), which is
deliberate — the closed form is stated only for `lambda > 0`. Filter with
`np.isfinite` before histogramming. `h`, `kappa`, `lam` are populated everywhere.

```python
import numpy as np, json
d = np.load("per-token/60M/<run_name>.npz")
meta = json.loads(bytes(d["summary"]).decode())
r = d["h_over_kappa"]; r = r[np.isfinite(r)]     # lambda > 0 subset
```

## Naming and parsing

Run names are the pretrained model's, so they carry both schemas:

    (MuonExpt3|PTSweep<SIZE>)-<mt>-chinchilla-<c>-<optimizer-part>[-wd0.1-bs1M]-wsd

```python
import re
m = re.search(r"chinchilla-([0-9.]+)-(adamw|muon)", name)   # all you need
```

**Both naming schemas appear and denote the same configuration.** Parse the
prefix off before grouping or you will double-count.

## Coverage

74 configs = the tuned (best-LR) base of every `chinchilla x optimizer` cell —
the same base set the perturbation, replay and EWC sweeps use, so rows line up
with those results.

| size | chinchillas | configs |
|---|---|---|
| 30M | 0.25 … 64 (9) | 18 |
| 60M | 0.25 … 128 (10) | 20 |
| 100M | 0.25 … 32 (8) | 16 |
| 300M | 0.25 … 8 (6) | 12 |
| 600M | 0.25 … 2 (4) | 8 |

Scored on the held-out DCLM shard `part-059/00004.npy` — never trained on by any
run, and distinct from the replay/Fisher shard `part-058`. 256 sequences x 4096
tokens = 1,048,320 positions per config.

## The result, for orientation

Median h/kappa: **muon is ahead at low token budget, adamw at high**, and the
crossover falls with model size — c=8 at 30M, c=2 at 60M, c=1 at 100M. At 300M
and 600M the switch reverts at a higher budget (labelled "not sustained" in the
figure); neither size reaches far enough into the data-rich regime to say
whether a sustained crossover exists there. muon is at least as accurate in all
37 paired cells, so this is not accuracy in disguise.

## Caveats that matter for plotting

- **256 instances, not the full 1024** (`DCLM_HELDOUT_INSTANCES`). Per-cell
  differences are ~0.1-0.3 against a p10-p90 spread of ~2.5. A 4x rerun is cheap
  (`--instances 1024`) and worth it before publication.
- **h/kappa is right-skewed**, so mean sits above median and the two can in
  principle order the optimizers differently. State which you plot.
- **`lambda > 0` is stricter than argmax-correct**, by exactly the tied
  positions. FlashAttention forces a bf16 forward (~3 significant digits), so
  ties are common (`n_ties_at_gold`, e.g. 3758 of ~1.05M) and this also caps the
  precision of h and kappa at roughly the 1% level. Fine for ranking; do not
  quote absolute values to 3 decimals.
- **The spread dwarfs the between-optimizer gap.** Any figure showing only
  medians should say so, or show the p10-p90 band, or show the histograms.

## Reproducing / extending

```bash
OPTIM_SIZE=60M python -m new_utils.margin_stats_sweep \
    --instances 1024 --save-per-token --out-dir /mnt/localssd/margins
python -m new_utils.plot_margins        # summary figures + csv
python -m new_utils.plot_margin_hist    # per-token histograms
```

Resumable: a config whose `.json` exists is skipped.

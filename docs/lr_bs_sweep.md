# `LrBatchSweep` — LR × batch-size cross product

Spec of a new object added to `optimizer-study-copy`, written so another agent can
extend, launch, or port it without reading the surrounding history.

---

## 1. What it is

`launch_jolmo/lr_bs_sweep.py` defines one frozen dataclass, `LrBatchSweep`, that
builds the full cross product of

```
main LR  ×  global-batch multiplier
```

for a single **(model size, optimizer, chinchilla)** cell.

Defaults: 8 LRs × 3 multipliers = **24 models**.

```python
SWEEP_LRS            = (5e-3, 7e-3, 1e-2, 1.4e-2, 2e-2, 2.8e-2, 4e-2, 5.6e-2)
SWEEP_BS_MULTIPLIERS = (1, 2, 4)          # × GLOBAL_BATCH_SIZE (1M tokens/step)
```

### Semantics per optimizer

| optimizer | what the sweep varies | adamw component |
| --- | --- | --- |
| `adamw` | `learning_rate` | n/a |
| `muon` | `muon_lr` | **pinned** to `PT_LR_BY_MODEL[model_type][scheduler]["adamw"][chinchilla]` |

A muon sweep whose tuned adamw cell is `None` builds **zero models** and prints
why. It never invents a component.

### Batch scaling

LRs are defined at the reference batch (1M tokens/step). A cell at multiplier `a`
trains at `sqrt(a) × lr`. At `a = 1` this is the identity.

For muon, **both legs scale** by `sqrt(a)` by default, preserving the muon:adamw
ratio across the batch axis. Set `scale_adamw_component=False` to hold the pinned
component fixed instead.

The **token budget is fixed** across the batch axis (`n_tokens` identical in every
cell); only the step count moves, so all 24 cells see the same data.

---

## 2. API

```python
from launch_jolmo.lr_bs_sweep import LrBatchSweep

sweep = LrBatchSweep(
    model_type="0.06B",        # key into training.MODEL_ARCHS
    optimizer="adamw",         # "adamw" | "muon"
    chinchilla=4,              # may be fractional (0.25, 0.5)
    # optional:
    lrs=SWEEP_LRS,
    bs_multipliers=(1, 2, 4),
    weight_decay=0.1,
    scheduler="wsd",
    num_processes=None,        # default: the $OPTIM_SIZE profile's value
    scale_adamw_component=True,
    validation_eval_interval=95,
)

sweep.models()   # ArtifactSet[JolmoModel]      — 24 cells
sweep.evals()    # ArtifactSet[ModelEvaluation] — one per cell
sweep.label      # "lrbs-0.06B-c4-adamw"        — the stage name
```

Read-only properties: `label`, `size_ok`, `n_tokens`, `gpus`, `tuned_adamw_lr`.

### Declaring a sweep

Append to `SWEEPS` at the bottom of the module; the launcher turns each entry into
two stages automatically.

```python
LRBS_60M_CHINCHILLAS: Tuple[float, ...] = (1, 2, 4, 8)

SWEEPS: Tuple[LrBatchSweep, ...] = tuple(
    LrBatchSweep(model_type="0.06B", optimizer=opt, chinchilla=c)
    for c in LRBS_60M_CHINCHILLAS
    for opt in ("adamw", "muon")
)
```

### Launching

```bash
OPTIM_SIZE=60M python -m launch_jolmo.launcher drylaunch lrbs-0.06B-c4-adamw
OPTIM_SIZE=60M python -m launch_jolmo.launcher launch    lrbs-0.06B-c4-adamw
OPTIM_SIZE=60M python -m launch_jolmo.launcher launch    lrbs-0.06B-c4-adamw-evals
```

---

## 3. Naming

```
<prefix>-<model_type>-chinchilla-<c>-<opt>-<lrtag>-<wdtag>-<bstag>-<scheduler>
```

```
PTSweep60M-0.06B-chinchilla-4-adamw-lr7.1e-3-wd0.1-bs2M-wsd
PTSweep60M-0.06B-chinchilla-4-muon-muonlr1.0e-2-adamwlr1.4e-2-wd0.1-bs4M-wsd
```

**The prefix must keep matching what is already on GCS.**
`DEFAULT_NAME_PREFIX = f"PTSweep{_SIZE}"` (`_SIZE` = the `$OPTIM_SIZE` key), so at
60M it is `PTSweep60M` — byte-identical to the names `pt_sweep_60m_chin4.py`
already wrote. This is not cosmetic: artifact existence is keyed on
`<project>/JolmoModel/<model_name>/final-unsharded/model.pt`, so a different
prefix silently retrains every cell that already exists. An earlier draft used
`LRBSSweep-` and would have re-run 39 finished 60M runs.

Override per instance with `name_prefix=` only when a separate tree is wanted.

* The LR tag carries the **scaled** LR (post-`sqrt(a)`), so the name states what
  the cell actually trained at. The pre-existing `pt_sweep_60m_chin4` grid scales
  the same way, which is why the tags line up.
* `_chin_tag` formats via `%g`, so `4` and `4.0` produce one name, and fractional
  chinchillas render as `chinchilla-0.25`.
* `models()` raises if two cells collide on a name (LRs rounding together under
  `_lr_tag`'s one decimal) rather than letting them share a GCS directory.

---

## 4. Guards

| condition | behavior |
| --- | --- |
| `model_type` ≠ active `$OPTIM_SIZE` profile | 0 models + message. The bucket is chosen by `$OPTIM_SIZE`, so this prevents writing into another study's prefix. |
| muon with no tuned adamw LR | 0 models + message naming the missing table cell. |
| multiplier that doesn't divide `n_tokens` | warns; that cell would train on a truncated budget. |
| duplicate run names | `ValueError`. |
| `optimizer` not in {adamw, muon} | `ValueError`. |

---

## 5. Files changed

### Added: `launch_jolmo/lr_bs_sweep.py`

The whole object. Imports its corpus definition from `pretraining_matrix`
(`TOKENIZER`, `GLOBAL_BATCH_SIZE`, `SEQUENCE_LENGTH`, `PT_LR_BY_MODEL`,
`_base_tokens_for`, `_dclm_chunks_for_tokens`, `_lr_tag`, `diversity_val_chunks`,
`dclm_heldout_val_chunks`, `DCLM_HELDOUT_INSTANCES`) so the data and val sets stay
single-sourced.

Naming constants: `DEFAULT_NAME_PREFIX` (size-derived, see §3) and the
`_bs_tag` / `_chin_tag` / `_wd_tag` helpers, which reproduce
`pt_sweep_60m_chin4`'s tag formats exactly.

`_shared_params()` mirrors `pt_sweep_60m_chin4._shared_params` — same optimizer
betas, `ddp`, bf16 params / fp32 reduce, `compile_model=True`, save intervals,
`unshard_checkpoint`, `upload` — with `rank_microbatch_size` derived from the
batch size (`max(SEQUENCE_LENGTH, gbs // gpus // 8)`) so the batch axis stays
within memory.

### Modified: `launch_jolmo/launcher.py`

```python
from launch_jolmo.lr_bs_sweep import SWEEPS as LRBS_SWEEPS
...
for _sweep in LRBS_SWEEPS:
    executor.stage(_sweep.label, _sweep.models())
    executor.stage(f"{_sweep.label}-evals", _sweep.evals())
```

### Added: this file

---

## 6. Implementation note worth preserving

`models()` is **memoised** in a module-level `_MODELS_CACHE`, keyed by the full
parameter tuple.

This is load-bearing, not an optimization. The executor resolves an evaluation's
dependency on its base model **by object identity**. `evals()` calls `models()`,
so without the cache the two stages hold distinct-but-equal `JolmoModel`
instances, and selecting both fails with:

```
ValueError: Artifact ModelEvaluation(...) depends on JolmoModel(...),
which is not in the artifact set. All dependencies must be explicitly included.
```

Any refactor that rebuilds models per call must preserve this.

---

## 7. Verified behavior

Declared sweeps (`SWEEPS`): 60M x chinchilla {1, 2, 4, 8} x {adamw, muon} =
8 stages x 24 cells = **192 models**.

Checked against the 210 `PTSweep60M-*` directories in
`gs://cmu-gpucloud-catheri4/Optim-60M-tuning/JolmoModel/`, using the artifact's
real `exists` check (`final-unsharded/model.pt`), not directory listing:

| stage | cells | already trained | to run |
| --- | --- | --- | --- |
| `lrbs-0.06B-c4-muon` | 24 | 24 | 0 |
| `lrbs-0.06B-c4-adamw` | 24 | 15 | 9 |
| `lrbs-0.06B-c1-*`, `c2-*`, `c8-*` (6 stages) | 24 each | 0 | 24 each |

**153 new runs, 39 reused.** The 9 outstanding c4-adamw cells are the top of the
LR range (2.8e-2 / 4e-2 / 5.6e-2 at bs1M, and their sqrt-scaled twins at 2M/4M),
which were not in the grid when that sweep last ran.

The muon match is the strongest signal that the pinning rule is right: the old
`SWEEP_LR_MUON` pairs used an adamw component of 7e-3, and the tuned adamw LR at
chinchilla-4 *is* 7e-3, so pinning reproduces the existing 24 runs exactly rather
than forking a near-duplicate set.

Scaling, from the built configs (60M chinchilla-4, `n_tokens` = 4,802,478,080):

| multiplier | gbs | steps | adamw lr (from 5e-3) | muon pair (from 5e-3, pin 7e-3) |
| --- | --- | --- | --- | --- |
| 1 | 1M | 4580 | 5.0e-3 | (5.0e-3, 7.0e-3) |
| 2 | 2M | 2290 | 7.1e-3 | (7.1e-3, 9.9e-3) |
| 4 | 4M | 1145 | 1.0e-2 | (1.0e-2, 1.4e-2) |

Budget truncation: chinchilla-1 at 2M/4M and chinchilla-2 at 4M do not divide
evenly and lose one partial step each — **0.087%** of the budget (~1.05M of
1.2B tokens). The guard warns; the effect is far below run-to-run noise.

Other checks:

```
drylaunch lrbs-0.06B-c4-adamw       -> Skipping 15 existing, Total tasks: 9
drylaunch lrbs-0.06B-c4-muon        -> Skipping 24 existing, no jobs
drylaunch lrbs-0.06B-c8-muon        -> Total tasks: 24, partition flame, gres=gpu:8
drylaunch lrbs-0.06B-c4-muon-evals  -> Total tasks: 24, ModelEvaluation
OPTIM_SIZE=300M ... LrBatchSweep("0.06B") -> 0 models (size guard fires)
```

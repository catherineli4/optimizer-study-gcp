# Seed replicates and WSD decay-floor variants — 60M, chinchilla 1 / 2 / 4, both optimizers

Two experiments over the same six tuned (best-LR) 60M base cells. Code:
`launch_jolmo/seed_replicates.py`; stages registered in `launch_jolmo/launcher.py`.

1. **Seed replicates** (`seed-rep-60m`): the reference configuration under two
   more training seeds (1 and 2; the reference is seed 0), so every cell has
   three seeds.
2. **Decay-floor variants** (`decaymin-60m`): the reference configuration
   (seed 0) with the WSD decay phase ending at 0.1 × max LR instead of 0.

Both then get the canonical pretrain eval, gsm8k finetunes with the `cpt`
recipe, and the `eval-cpt` evals.

## 1. Reference models (seed 0, already on GCS)

The tuned cell per budget is `PT_LR_BY_MODEL["0.06B"]["wsd"]` in
`pretraining_matrix.py`. `tuned_bases_for((1, 2, 4), ("adamw", "muon"))`
builds them with the same `SHARED_MODEL_PARAMS` / token budget / DCLM parts as
the main matrix and resolves each name to the schema that exists on GCS
(PTSweep preferred):

| budget | optimizer | reference run (`Optim-60M-tuning/JolmoModel/<run>`) | LR |
| --- | --- | --- | --- |
| c1 | adamw | `PTSweep60M-0.06B-chinchilla-1-adamw-lr2.0e-2-wd0.1-bs1M-wsd` | 2.0e-2 |
| c2 | adamw | `PTSweep60M-0.06B-chinchilla-2-adamw-lr2.0e-2-wd0.1-bs1M-wsd` | 2.0e-2 |
| c4 | adamw | `PTSweep60M-0.06B-chinchilla-4-adamw-lr1.0e-2-wd0.1-bs1M-wsd` | 1.0e-2 |
| c1 | muon | `PTSweep60M-0.06B-chinchilla-1-muon-muonlr1.4e-2-adamwlr2.0e-2-wd0.1-bs1M-wsd` | (1.4e-2, 2.0e-2) |
| c2 | muon | `PTSweep60M-0.06B-chinchilla-2-muon-muonlr1.0e-2-adamwlr2.0e-2-wd0.1-bs1M-wsd` | (1.0e-2, 2.0e-2) |
| c4 | muon | `PTSweep60M-0.06B-chinchilla-4-muon-muonlr1.4e-2-adamwlr1.0e-2-wd0.1-bs1M-wsd` | (1.4e-2, 1.0e-2) |

Shared recipe (read back from each run's `final-unsharded/config.json`): 0.06B
arch, OLMo-2 tokenizer, betas (0.9, 0.98), wd 0.1, grad clip 1.0, WSD (10 %
warmup, 10 % decay, floor 0), global batch 1,048,576 tokens, context 4096, DDP
bf16/fp32, compile on, diversity-v2 validation every 95 steps.

Randomness in the reference runs comes from three YAML fields:

| YAML field | value | controls |
| --- | --- | --- |
| `model.init_seed` | 0 (engine default, never written by the launcher) | parameter-init generator |
| `data_loader.seed` | 0 (literal) | instance shuffle |
| `init_seed` (top level) | 12536 | `seed_all()` on the process-global RNGs; does not reach parameter init |

## 2. New runs

### Seed replicates (12)

`<reference run>-seed<s>` for s ∈ {1, 2}. The generated YAML differs from the
reference only in `model.init_seed = s`, `data_loader.seed = s`, top-level
`init_seed = s`, `wandb.config.seed`, and the name-derived paths
(`save_folder`, `work_dir`, `wandb.name` / `wandb.id`).

### Decay-floor variants (6)

`<reference run>-decaymin0.1`. The YAML differs only in
`train_module.scheduler.decay_min_lr_fraction = 0.1` (plus name-derived paths).
The floor is relative **per optimizer param group**: Muon's own group decays to
0.1 × muon_lr and its AdamW component to 0.1 × adamw_lr. This needs the
`WSD.decay_min_lr_fraction` field added to JOLMo
(`src/olmo_core/optim/scheduler.py`); the engine's `decay_min_lr` is absolute
and shared across groups, so it could not express this for Muon.

### Why the suffixes are safe

Every discovery regex that rebuilds grid cells from the GCS listing
(`_discover_bases`, `_discover_every_base`, `_existing_bs1m_models`,
`_existing_bs_any_models`, `pt_sweep_60m_chin4`, `ft_sweep._canonical_base`)
anchors on `-wsd$` or `-wd0.1-bs1M-wsd`, so `-seed<s>` / `-decaymin0.1` runs
are never swept into CPT / perturbation / divergence stages as grid cells.

GPU count: the profile default (8 for 60M, `OPTIM_NUM_PROCESSES` overrides).
Some references trained on 2 GPUs; at a fixed 1M-token global batch only
`rank_microbatch_size` changes, not the optimisation. `--verify` reports this
as a gpu-count diff, never a failure.

## 3. Finetunes and evals

Per experiment, exactly the `cpt` / `eval-cpt` recipe (`cpt.py`):
gsm8k, 20M tokens, 1024 context, 64K-token batch, cosine; adamw base → adamw
FT over `CPT_LR_SWEEP["adamw"]` (8 LRs); muon base → muon FT (8 pairs, adamw
component = 0.25 × muon_lr) **and** adamw FT (8 LRs). Run names follow
`CPTModel.run_name` (`<base>-CPT-gsm8k-20M-<opt>-lr…-exp[-muonlr…]`), so the
base's suffix propagates. Evals score the gsm8k val set, diversity-v2, and
`DCLM_heldout` (1024 instances), like every other CPT eval.

| | models | base evals | gsm8k finetunes | finetune evals |
| --- | --- | --- | --- | --- |
| `seed-rep-60m` | 12 | 12 | 144 (6 adamw × 8 + 6 muon × 16) | 144 |
| `decaymin-60m` | 6 | 6 | 72 (3 adamw × 8 + 3 muon × 16) | 72 |

## 4. Stages

| stage | contents |
| --- | --- |
| `seed-rep-60m-models` / `-adamw-models` / `-muon-models` | the 12 / 6 / 6 JolmoModels |
| `seed-rep-60m-evals` | 12 pretrain evals |
| `seed-rep-60m-gsm8k` | 144 CPTModels |
| `seed-rep-60m-gsm8k-evals` | 144 CPT evals |
| `seed-rep-60m` | group: models + evals + gsm8k + gsm8k-evals |
| `decaymin-60m-*` / `decaymin-60m` | same shape, 6 / 6 / 72 / 72 |

Launch the **group** (or include the `-models` / `-gsm8k` stages in the same
selection as their evals): the executor resolves dependencies by object
identity and refuses an eval whose model is not in the selected set. All
stages resolve empty, with a printed reason, unless `OPTIM_SIZE=60M`.

## 5. Code changes

1. `JOLMo/src/olmo_core/optim/scheduler.py` — `WSD.decay_min_lr_fraction:
   Optional[float] = None`. When set, the floor used for a group is
   `fraction × initial_lr`; `None` keeps the absolute `decay_min_lr` (unchanged
   behaviour and existing tests). Applied to both JOLMo checkouts on babel
   (`optimizer-study/JOLMo`, the editable install, and
   `optimizer-study-copy/JOLMo`, the `jolmo_path`); must be committed to
   `catherineli4/JOLMo` for a GCP VM to pick it up.
2. `launch_jolmo/training.py` — `JolmoModel.seed: Optional[int] = None` and
   `JolmoModel.wsd_decay_min_lr_fraction: Optional[float] = None`;
   `_build_model_spec(..., init_seed)`, `_build_scheduler_spec(...,
   wsd_decay_min_lr_fraction)`, `data_loader.seed` / top-level `init_seed`
   follow `seed`; both recorded in `wandb.config`. With both `None` every
   existing artifact's YAML is byte-identical.
3. `launch_jolmo/seed_replicates.py` (new) — builds the models, evals,
   finetunes and finetune evals; `--verify` diffs each variant's generated YAML
   against its reference `config.json` on GCS.
4. `launch_jolmo/launcher.py` — registers the stages and groups; adds a
   `stage_group` fallback for `experiments` checkouts that predate it.

## 6. Verify and launch

```bash
conda activate optim-study
cd ~/optimizer-study-gcp
OPTIM_SIZE=60M python -m launch_jolmo.seed_replicates --verify     # YAML vs reference config.json
OPTIM_SIZE=60M python -m launch_jolmo.launcher drylaunch seed-rep-60m decaymin-60m
OPTIM_SIZE=60M python -m launch_jolmo.launcher launch    seed-rep-60m decaymin-60m
```

Outputs: `gs://cmu-gpucloud-catheri4/Optim-60M-tuning/JolmoModel/<run>/final-unsharded/`,
`CPTModel/gsm8k/<run>-CPT-gsm8k-…/final-unsharded/`, and
`ModelEvaluation/<run>-eval.json` for every model and finetune.

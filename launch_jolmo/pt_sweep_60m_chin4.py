"""60M 4-chinchilla PT Sweep — a staged hyperparameter sweep, separate from MuonExpt3.

Three sweeps, run IN ORDER, each feeding the next:

    1. pt60m4-lr-sweep   sweep the pretraining LR      (WD and batch at defaults)
    2. pt60m4-wd-sweep   sweep weight decay            (at the winning LR)
    3. pt60m4-bs-sweep   sweep global batch size       (at the winning LR + WD,
                                                        LR rescaled by sqrt(a))

Fill in the SWEEP VARIABLES block below. Stage 1's grid is yours to choose; after
it finishes, put the winner in BEST_LR_* and stage 2 becomes launchable, and so on.
A stage whose inputs are still blank resolves to ZERO models and says so, rather
than silently launching something arbitrary.

Isolation from the main study
-----------------------------
Every model here is named ``PTSweep60M-...`` instead of ``MuonExpt3-...``. Since
artifacts land in ``<project>/JolmoModel/<model_name>/``, the distinct prefix
already gives this sweep its own directory tree — nothing can collide with or
overwrite the MuonExpt3 runs.

Size is pinned
--------------
This module is 0.06B / chinchilla-4 by construction, NOT driven by $OPTIM_SIZE.
But the destination bucket IS chosen by $OPTIM_SIZE (run_local calls
Project.init on it), so running under the wrong size would write 60M models into
another study's bucket. The guard below makes the stages resolve empty unless
OPTIM_SIZE=60M:

    OPTIM_SIZE=60M python -m launch_jolmo.run_local list pt60m4-lr-sweep
    OPTIM_SIZE=60M python -m launch_jolmo.run_local queue pt60m4-lr-sweep --parallel 1 --nproc 8
"""

import math
import os
from typing import Any, Dict, List, Optional, Tuple

from experiments import ArtifactSet

from launch_jolmo.training import JolmoModel, ModelEvaluation
from launch_jolmo.sizes import active_profile

# Reused from the main matrix so the corpus definition has a single source of
# truth: if the DCLM layout or the validation set changes there, it changes here.
from launch_jolmo.pretraining_matrix import (
    TOKENIZER,
    _base_tokens_for,
    _dclm_chunks_for_tokens,
    diversity_val_chunks,
    dclm_heldout_val_chunks,
    DCLM_HELDOUT_INSTANCES,
    _lr_tag,
)
from launch_jolmo.cpt import build_cpt_models, build_cpt_model_evaluations


# ===========================================================================
# SWEEP VARIABLES — FILL THESE IN
# ===========================================================================

# --- Stage 1: PT learning rate -------------------------------------------
# AdamW: plain LRs.            e.g. [1.4e-2, 1e-2, 7e-3, 5e-3]
SWEEP_LR_ADAMW: List[float] = [7e-3, 1e-2, 1.4e-2, 2e-2]
# Muon: (muon_lr, adamw_component_lr) pairs.  e.g. [(1e-2, 7e-3), (7e-3, 7e-3)]
SWEEP_LR_MUON: List[Tuple[float, float]] = [(1e-2,7e-3), (1.4e-2,7e-3), (2e-2,7e-3), (2.8e-2,7e-3), (4e-2,7e-3), (5.6e-2,7e-3)]

# --- Stage 2: weight decay ------------------------------------------------
# e.g. [0.0, 0.05, 0.1, 0.2]
SWEEP_WD: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4]

# --- Stage 3: global batch size (TOKENS per step, must divide the budget) --
# Multiples a x DEFAULT_GLOBAL_BATCH_SIZE (1M). The LR is NOT held fixed here:
# each cell trains at sqrt(a) x the winning stage-1/2 LR — see _scaled_lr.
# Only some multiples divide the 4580-step budget: N_TOKENS/1M = 4580 = 2^2 x 5 x 229,
# so a in {1, 2, 4, 5, 10, 20} is exact and a = 8 or 16 would truncate the budget.
SWEEP_BATCH_SIZE: List[int] = [1_048_576, 2_097_152, 4_194_304]   # 1M, 2M, 4M

# --- Winners carried forward (fill in after each stage completes) ---------
# After stage 1, set the LR the sweep picked. Stage 2 and 3 need these.
BEST_LR_ADAMW: Optional[float] = 7e-3
BEST_LR_MUON: Optional[Tuple[float, float]] = (1e-2, 7e-3)
# After stage 2, set the winning weight decay. Stage 3 needs this.
BEST_WD: Optional[float] = 0.1

# Which optimizers to sweep. Drop one to halve the grid.
SWEEP_OPTIMIZERS: List[str] = ["adamw", "muon"]

# NOTE: finetuning is NOT configured here. It uses launch_jolmo/cpt.py's own
# pipeline verbatim — CPT_DATASETS, CPT_LR_SWEEP, CPT_TOKENS, CPT_SCHEDULER and
# the CPT batch/context. Change finetuning behaviour by editing cpt.py, exactly
# as it works for the MuonExpt3 cpt-* stages. See the CPT section at the bottom.

# ===========================================================================
# End of sweep variables
# ===========================================================================


# ---------------------------------------------------------------------------
# Fixed configuration for this sweep
# ---------------------------------------------------------------------------

NAME_PREFIX = "PTSweep60M"      # keeps every run distinct from MuonExpt3-*
MODEL_TYPE = "0.06B"
CHINCHILLA = 4
SCHEDULER = "wsd"

# --- Retargeting the same staged sweep at another (size, chinchilla) --------
# The module is 60M / chinchilla-4 by default, and stays byte-identical unless
# BOTH overrides are set. With them, the same naming scheme and the same three
# stages apply to another cell, e.g.
#
#   OPTIM_SIZE=30M OPTIM_PTSWEEP_SIZE=30M OPTIM_PTSWEEP_CHINCHILLA=1 \
#   OPTIM_PTSWEEP_WD=0,0.2,0.3 OPTIM_PTSWEEP_OPTIMIZERS=muon ...
#
# OPTIM_SIZE still picks the destination bucket and is still checked against
# MODEL_TYPE below, so a mismatched pair resolves to zero models rather than
# writing into another study's bucket.
_pts_size = os.environ.get("OPTIM_PTSWEEP_SIZE", "").strip()
_pts_chin = os.environ.get("OPTIM_PTSWEEP_CHINCHILLA", "").strip()
if _pts_size or _pts_chin:
    if not (_pts_size and _pts_chin):
        raise ValueError(
            "set OPTIM_PTSWEEP_SIZE and OPTIM_PTSWEEP_CHINCHILLA together; "
            f"got size={_pts_size!r} chinchilla={_pts_chin!r}")
    from launch_jolmo.sizes import PROFILES as _PTS_PROFILES
    if _pts_size not in _PTS_PROFILES:
        raise ValueError(f"OPTIM_PTSWEEP_SIZE={_pts_size!r}; "
                         f"available: {sorted(_PTS_PROFILES)}")
    NAME_PREFIX = f"PTSweep{_pts_size}"
    MODEL_TYPE = _PTS_PROFILES[_pts_size]["model_type"]
    CHINCHILLA = float(_pts_chin)
    # Integral budgets keep the historical integer naming (chinchilla-4, not
    # chinchilla-4.0); fractional ones print as 0.25 / 0.5 like everywhere else.
    if CHINCHILLA == int(CHINCHILLA):
        CHINCHILLA = int(CHINCHILLA)

SEQUENCE_LENGTH = 4096
DEFAULT_GLOBAL_BATCH_SIZE = 1_048_576   # 1M tokens/step — stages 1 & 2 hold this
DEFAULT_WEIGHT_DECAY = 0.1              # matches SHARED_MODEL_PARAMS in the main matrix

# Chinchilla-1 budget for this size, then x CHINCHILLA. Goes through the main
# matrix's helper instead of indexing the measured-budget dict directly: that
# dict holds the 0.06B entry only, so a retargeted size raised KeyError. 0.06B
# still resolves to the same 20 x 60,030,976 through the override inside it.
BASE_TOKENS = _base_tokens_for(MODEL_TYPE)
N_TOKENS = BASE_TOKENS * CHINCHILLA

_SIZE, _PROFILE = active_profile()

# Follow the size profile (60M -> 2 GPUs) rather than hardcoding, so Slurm asks
# for a plausible allocation on the general partition. get_requirements() turns
# this into gres=gpu:N, so a larger number means a longer queue wait.
# Override per-launch with OPTIM_NUM_PROCESSES=8.
NUM_PROCESSES = int(os.environ.get(
    "OPTIM_NUM_PROCESSES", str(_PROFILE.get("num_processes", 2))))

VALIDATION_EVAL_INTERVAL = 95

# Only pull the DCLM parts this budget actually needs (~4.8B tokens -> 1 part).
TRAIN_CHUNKS = _dclm_chunks_for_tokens(N_TOKENS)

_SIZE_OK = _PROFILE["model_type"] == MODEL_TYPE

_pts_wd = os.environ.get("OPTIM_PTSWEEP_WD", "").strip()
if _pts_wd:
    SWEEP_WD = [float(x) for x in _pts_wd.replace(",", " ").split()]

# OPTIM_PTSWEEP_NO_NORM_WD=1 exempts the RMSNorm gains from weight decay and
# appends -nonormwd to the run name, so these are distinct artifacts from the
# runs that decayed the gains (see JolmoModel.decay_norms).
DECAY_NORMS = os.environ.get("OPTIM_PTSWEEP_NO_NORM_WD", "").strip() in ("", "0")

_pts_opt = os.environ.get("OPTIM_PTSWEEP_OPTIMIZERS", "").strip()
if _pts_opt:
    SWEEP_OPTIMIZERS = [o.strip() for o in _pts_opt.replace(",", " ").split()]
    bad = [o for o in SWEEP_OPTIMIZERS if o not in ("adamw", "muon")]
    if bad:
        raise ValueError(f"OPTIM_PTSWEEP_OPTIMIZERS={bad}; use adamw and/or muon")

if _pts_size or _pts_chin:
    # BEST_LR_ADAMW / BEST_LR_MUON above are the 60M chinchilla-4 winners. On a
    # retargeted cell they would silently train the wrong LR, so take the tuned
    # LR for THIS (size, chinchilla) from the same table every other sweep reads.
    from launch_jolmo.pretraining_matrix import PT_LR as _PTS_PT_LR
    _pts_tuned = _PTS_PT_LR.get(SCHEDULER, {})
    BEST_LR_ADAMW = _pts_tuned.get("adamw", {}).get(CHINCHILLA)
    BEST_LR_MUON = _pts_tuned.get("muon", {}).get(CHINCHILLA)
    print(f"[ptsweep] retargeted to {NAME_PREFIX} {MODEL_TYPE} "
          f"chinchilla-{CHINCHILLA}: tuned adamw={BEST_LR_ADAMW} "
          f"muon={BEST_LR_MUON}", flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wd_tag(wd: float) -> str:
    """0.1 -> 'wd0.1', 0.0 -> 'wd0'. Trailing zeros trimmed so names stay short."""
    return "wd" + (f"{wd:g}")


def _bs_tag(gbs: int) -> str:
    """1048576 -> 'bs1M', 524288 -> 'bs512k'. Falls back to the raw count."""
    if gbs % (1024 * 1024) == 0:
        return f"bs{gbs // (1024 * 1024)}M"
    if gbs % 1024 == 0:
        return f"bs{gbs // 1024}k"
    return f"bs{gbs}"


def _schedule_for(global_batch_size: int) -> Dict[str, Any]:
    """Token budget is FIXED across the batch-size sweep; only the step count moves,
    so every cell sees the same data and stays comparable."""
    total_steps = N_TOKENS // global_batch_size
    return {
        "n_tokens": N_TOKENS,
        "warmup_steps": total_steps // 10,
    }


def _shared_params(global_batch_size: int, weight_decay: float) -> Dict[str, Any]:
    return {
        "model_type": MODEL_TYPE,
        "tokenizer": TOKENIZER,
        "sequence_length": SEQUENCE_LENGTH,
        # Optimizer
        # BOTH groups. _build_optimizer_spec routes muon_weight_decay to the
        # Muon group (the 2-D matrices, i.e. nearly every trainable weight) and
        # weight_decay to adamw_weight_decay (norms/biases; embeddings are
        # force-pinned to 0 by the group override). Setting only weight_decay
        # left the matrices decaying at the 0.1 default in every cell, so the
        # "wd sweep" varied decay on a small minority of parameters and the
        # wd=0 cell was not wd=0 at all.
        "weight_decay": weight_decay,
        "muon_weight_decay": weight_decay,
        "decay_norms": DECAY_NORMS,
        "betas": (0.9, 0.98),
        "max_grad_norm": 1.0,
        # Schedule
        "scheduler": SCHEDULER,
        "global_batch_size": global_batch_size,
        # Derived from the batch size, so the batch-size sweep stays within memory.
        "rank_microbatch_size": max(
            SEQUENCE_LENGTH, global_batch_size // NUM_PROCESSES // 8),
        "eval_rank_microbatch_size": max(
            SEQUENCE_LENGTH, global_batch_size // NUM_PROCESSES // 64),
        "validation_eval_interval": VALIDATION_EVAL_INTERVAL,
        # Parallelism & compilation
        "compile_model": True,
        "parallelism": "ddp",
        "num_processes": NUM_PROCESSES,
        "dp_param_dtype": "bfloat16",
        "dp_reduce_dtype": "float32",
        # Data
        "train_chunks": TRAIN_CHUNKS,
        "validation_chunks": diversity_val_chunks,
        # Checkpointing & export
        "save_interval": 1000,
        "ephemeral_save_interval": 500,
        "unshard_checkpoint": True,
        "convert_to_hf": False,
        "upload": True,
        # Experiment
        "experiment_name": _PROFILE["project"],
    }


def _model(opt: str, lr, weight_decay: float, global_batch_size: int) -> JolmoModel:
    """Build one cell. The name carries every swept axis, so a config appearing in
    two stages resolves to the SAME artifact and is reused rather than retrained."""
    if opt == "adamw":
        lr_tag = f"lr{_lr_tag(lr)}"
        extra = {"optimizer": "adamw", "learning_rate": lr}
    else:
        muon_lr, adamw_lr = lr
        lr_tag = f"muonlr{_lr_tag(muon_lr)}-adamwlr{_lr_tag(adamw_lr)}"
        extra = {"optimizer": "muon", "muon_lr": muon_lr, "learning_rate": adamw_lr}

    name = (
        f"{NAME_PREFIX}-{MODEL_TYPE}-chinchilla-{CHINCHILLA}-{opt}-{lr_tag}"
        f"-{_wd_tag(weight_decay)}-{_bs_tag(global_batch_size)}-{SCHEDULER}"
        + ("" if DECAY_NORMS else "-nonormwd")
    )
    return JolmoModel(
        model_name=name,
        **_shared_params(global_batch_size, weight_decay),
        **_schedule_for(global_batch_size),
        **extra,
    )


def _blocked(stage: str, why: str) -> ArtifactSet:
    print(f"[{stage}] 0 models — {why}", flush=True)
    return ArtifactSet([])


def _lrs_for(opt: str) -> list:
    return SWEEP_LR_ADAMW if opt == "adamw" else SWEEP_LR_MUON


def _best_lr_for(opt: str):
    return BEST_LR_ADAMW if opt == "adamw" else BEST_LR_MUON


def _scaled_lr(opt: str, lr, global_batch_size: int):
    """Square-root LR scaling for the batch-size sweep.

    Moving from the reference batch x (DEFAULT_GLOBAL_BATCH_SIZE) to a·x scales
    every LR by sqrt(a), so gradient-noise scale is held roughly constant instead
    of being confounded with the batch axis. At a = 1 this is the identity, so the
    reference cell is byte-identical to the stage-1/2 cell at the same WD and
    resolves to the SAME artifact (reused, not retrained).

    For muon BOTH legs move — the muon LR and its adamw component — so their ratio
    is preserved at every batch size and neither leg is silently left behind.
    """
    a = global_batch_size / DEFAULT_GLOBAL_BATCH_SIZE
    s = math.sqrt(a)
    if opt == "adamw":
        return lr * s
    muon_lr, adamw_lr = lr
    return (muon_lr * s, adamw_lr * s)


# ---------------------------------------------------------------------------
# Stage 1 — PT learning-rate sweep
# ---------------------------------------------------------------------------

def build_lr_sweep() -> ArtifactSet:
    if not _SIZE_OK:
        return _blocked("pt60m4-lr-sweep",
                        f"OPTIM_SIZE resolves to {_PROFILE['model_type']}, not {MODEL_TYPE}; "
                        f"rerun with OPTIM_SIZE=60M so artifacts land in the 60M bucket")
    models = []
    for opt in SWEEP_OPTIMIZERS:
        for lr in _lrs_for(opt):
            models.append(_model(opt, lr, DEFAULT_WEIGHT_DECAY, DEFAULT_GLOBAL_BATCH_SIZE))
    if not models:
        return _blocked("pt60m4-lr-sweep", "fill in SWEEP_LR_ADAMW / SWEEP_LR_MUON")
    return ArtifactSet(models)


# ---------------------------------------------------------------------------
# Stage 2 — weight-decay sweep, at the winning LR
# ---------------------------------------------------------------------------

def build_wd_sweep() -> ArtifactSet:
    if not _SIZE_OK:
        return _blocked("pt60m4-wd-sweep",
                        f"OPTIM_SIZE resolves to {_PROFILE['model_type']}, not {MODEL_TYPE}")
    if not SWEEP_WD:
        return _blocked("pt60m4-wd-sweep", "fill in SWEEP_WD")
    models = []
    for opt in SWEEP_OPTIMIZERS:
        best = _best_lr_for(opt)
        if best is None:
            print(f"[pt60m4-wd-sweep] skipping {opt} — BEST_LR not set yet", flush=True)
            continue
        for wd in SWEEP_WD:
            models.append(_model(opt, best, wd, DEFAULT_GLOBAL_BATCH_SIZE))
    if not models:
        return _blocked("pt60m4-wd-sweep",
                        "set BEST_LR_ADAMW / BEST_LR_MUON from the stage-1 results")
    return ArtifactSet(models)


# ---------------------------------------------------------------------------
# Stage 3 — batch-size sweep, at the winning LR + WD
# ---------------------------------------------------------------------------

def build_bs_sweep() -> ArtifactSet:
    if not _SIZE_OK:
        return _blocked("pt60m4-bs-sweep",
                        f"OPTIM_SIZE resolves to {_PROFILE['model_type']}, not {MODEL_TYPE}")
    if not SWEEP_BATCH_SIZE:
        return _blocked("pt60m4-bs-sweep", "fill in SWEEP_BATCH_SIZE")
    if BEST_WD is None:
        return _blocked("pt60m4-bs-sweep", "set BEST_WD from the stage-2 results")

    bad = [b for b in SWEEP_BATCH_SIZE if N_TOKENS % b]
    if bad:
        # A non-dividing batch size silently truncates the budget, which would make
        # that cell train on fewer tokens than the others and ruin the comparison.
        print(f"[pt60m4-bs-sweep] WARNING: {bad} do not divide N_TOKENS={N_TOKENS}; "
              f"those cells would see a truncated budget", flush=True)

    models = []
    for opt in SWEEP_OPTIMIZERS:
        best = _best_lr_for(opt)
        if best is None:
            print(f"[pt60m4-bs-sweep] skipping {opt} — BEST_LR not set yet", flush=True)
            continue
        for gbs in SWEEP_BATCH_SIZE:
            models.append(_model(opt, _scaled_lr(opt, best, gbs), BEST_WD, gbs))
    if not models:
        return _blocked("pt60m4-bs-sweep",
                        "set BEST_LR_ADAMW / BEST_LR_MUON from the stage-1 results")
    return ArtifactSet(models)


# ---------------------------------------------------------------------------
# Stage sets
# ---------------------------------------------------------------------------

pt60m4_lr_sweep = build_lr_sweep()
pt60m4_wd_sweep = build_wd_sweep()
pt60m4_bs_sweep = build_bs_sweep()

# Everything, for a single "what does this experiment contain" view.
pt60m4_all = pt60m4_lr_sweep + pt60m4_wd_sweep + pt60m4_bs_sweep


# ---------------------------------------------------------------------------
# Finetuning (CPT) of the swept pretrained models
#
# One CPTModel per (pretrained model x dataset x FT LR). Finetunes inherit the
# pretraining optimizer unless FT_OPTIMIZERS overrides it, so a muon-pretrained
# cell is muon-finetuned by default and the pretrain/finetune axes stay aligned.
#
# The bases must already be trained: a CPTModel loads the base's final-unsharded
# model.pt from GCS. The pretrain stages above are registered in the launcher, so
# the dependency resolves and an untrained base is trained first rather than
# silently skipped.
# ---------------------------------------------------------------------------

def _adamw(s: ArtifactSet) -> ArtifactSet:
    return ArtifactSet([m for m in s if m.optimizer == "adamw"])


def _muon(s: ArtifactSet) -> ArtifactSet:
    return ArtifactSet([m for m in s if m.optimizer == "muon"])


# Mirrors the MuonExpt3 wiring in pretraining_matrix.py exactly:
#   adamw-pretrained -> CPT with adamw, over CPT_LR_SWEEP["adamw"]
#   muon-pretrained  -> CPT with BOTH muon (adamw component = 0.25 x muon_lr)
#                       and adamw, each over its own sweep
# Nothing about the CPT phase is redefined here; build_cpt_models pulls its
# datasets, token budget, scheduler, batch and context from cpt.py.
def _cpt_adamw(s: ArtifactSet) -> ArtifactSet:
    return build_cpt_models(_adamw(s))


def _cpt_muon(s: ArtifactSet) -> ArtifactSet:
    return build_cpt_models(
        _muon(s), cpt_optimizers=["muon", "adamw"], muon_adamw_multiplier=0.25)


def _cpt_for(bases: ArtifactSet, label: str) -> ArtifactSet:
    if not _SIZE_OK:
        return _blocked(label, f"OPTIM_SIZE resolves to {_PROFILE['model_type']}, not {MODEL_TYPE}")
    if not len(list(bases)):
        return _blocked(label, "no pretrained bases — fill in the matching pretrain sweep first")
    return _cpt_adamw(bases) + _cpt_muon(bases)


def _cpt_muon_for(bases: ArtifactSet, label: str) -> ArtifactSet:
    """Muon-pretrained bases only, same semantics as the main matrix's `cpt-muon`
    stage: each muon base is finetuned with BOTH muon and adamw. AdamW-pretrained
    bases are dropped entirely, so nothing here depends on them."""
    if not _SIZE_OK:
        return _blocked(label, f"OPTIM_SIZE resolves to {_PROFILE['model_type']}, not {MODEL_TYPE}")
    if not len(list(_muon(bases))):
        return _blocked(label, "no muon pretrained bases — fill in the matching pretrain sweep first")
    return _cpt_muon(bases)


# ---------------------------------------------------------------------------
# Full cross-product grid — matches the 210 PTSweep60M-* models on GCS exactly:
# {6 adamw | 8 muon} core LRs x 5 WDs x 3 batch sizes, with every LR (and the
# muon adamw component) scaled by sqrt(batch/1M) via _scaled_lr.
# ---------------------------------------------------------------------------

FULL_GRID_LR_ADAMW: List[float] = [3.5e-3, 5e-3, 7e-3, 1e-2, 1.4e-2, 2e-2]
FULL_GRID_LR_MUON: List[Tuple[float, float]] = [
    (5e-3, 7e-3), (7e-3, 7e-3), (1e-2, 7e-3), (1.4e-2, 7e-3),
    (2e-2, 7e-3), (2.8e-2, 7e-3), (4e-2, 7e-3), (5.6e-2, 7e-3),
]


def build_full_grid() -> ArtifactSet:
    if not _SIZE_OK:
        return _blocked("pt60m4-full-grid",
                        f"OPTIM_SIZE resolves to {_PROFILE['model_type']}, not {MODEL_TYPE}")
    models = []
    for opt, lrs in (("adamw", FULL_GRID_LR_ADAMW), ("muon", FULL_GRID_LR_MUON)):
        for bs in SWEEP_BATCH_SIZE:
            for lr in lrs:
                for wd in SWEEP_WD:
                    models.append(_model(opt, _scaled_lr(opt, lr, bs), wd, bs))
    return ArtifactSet(models)


pt60m4_full_grid = build_full_grid()
pt60m4_cpt_full_grid = _cpt_for(pt60m4_full_grid, "pt60m4-cpt-full-grid")

pt60m4_cpt_lr_sweep = _cpt_for(pt60m4_lr_sweep, "pt60m4-cpt-lr-sweep")
pt60m4_cpt_wd_sweep = _cpt_for(pt60m4_wd_sweep, "pt60m4-cpt-wd-sweep")
pt60m4_cpt_bs_sweep = _cpt_for(pt60m4_bs_sweep, "pt60m4-cpt-bs-sweep")
pt60m4_cpt_wd_sweep_muon = _cpt_muon_for(pt60m4_wd_sweep, "pt60m4-cpt-wd-sweep-muon")
pt60m4_cpt_all = pt60m4_cpt_wd_sweep + pt60m4_cpt_bs_sweep 

# Same eval kwargs the MuonExpt3 cpt-* evals use: the held-out DCLM shard as an
# extra val set, giving the forgetting axis alongside the CPT val sets.
_dclm = dict(extra_val_chunks=dclm_heldout_val_chunks,
             extra_val_max_instances=DCLM_HELDOUT_INSTANCES)
pt60m4_cpt_evals = build_cpt_model_evaluations(pt60m4_cpt_all, **_dclm)
pt60m4_cpt_full_grid_evals = build_cpt_model_evaluations(pt60m4_cpt_full_grid, **_dclm)


# ---------------------------------------------------------------------------
# Pretrain evals — one ModelEvaluation per swept pretrained model.
#
# Same shape as _pretrain_eval in pretraining_matrix.py: the held-out DCLM shard
# is scored alongside the diversity-v2 val sets, so a sweep cell can be ranked on
# the pretraining distribution rather than only on the diversity sets.
# ---------------------------------------------------------------------------

def _pretrain_eval(model) -> ModelEvaluation:
    return ModelEvaluation(model=model, **_dclm)


pt60m4_lr_sweep_evals = ArtifactSet([_pretrain_eval(m) for m in pt60m4_lr_sweep])
pt60m4_wd_sweep_evals = ArtifactSet([_pretrain_eval(m) for m in pt60m4_wd_sweep])
pt60m4_bs_sweep_evals = ArtifactSet([_pretrain_eval(m) for m in pt60m4_bs_sweep])
pt60m4_all_evals = (pt60m4_lr_sweep_evals
                    + pt60m4_wd_sweep_evals
                    + pt60m4_bs_sweep_evals)

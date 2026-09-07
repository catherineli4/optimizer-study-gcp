"""Muon vs AdamW pretraining comparison, parameterized by model size (MODEL_TYPE).

Change MODEL_TYPE (see training.MODEL_ARCHS for the options) and the size-dependent
constants follow automatically: BASE_TOKENS (the Chinchilla-1 token budget) is derived
from the model size, and the tuned LR table is selected per model via PT_LR_BY_MODEL.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import os
import re
import sys
import math
import subprocess
from typing import List, Dict, Any, Optional, Tuple
from experiments import ArtifactSet

from launch_jolmo.data import Chunk
from launch_jolmo.training import (
    JolmoModel,
    ModelEvaluation,
    DivergenceEvaluation,
    C4DivergenceEvaluation,
    CeLossEvaluation,
    LogitPerturbEvaluation,
    LogitPerturbKlEvaluation,
    LogitCosineEvaluation,
    LogitAngleBinEvaluation,
    LogitAngleBinPerturbEvaluation,
    WeightAngleBinPerturbEvaluation,
    SharpnessEvaluation,
    ForgettingSharpnessEvaluation,
    list_training_checkpoints,
    DIVERGENCE_REF_OLMO2_32B,
    DIVERGENCE_REF_OLMO2_13B,
    DIVERGENCE_REF_OLMO2_7B,
    DIVERGENCE_REF_OLMO2_1B,
    CE_LOSS_REF_TAG_1B,
    divergence_max_batch_size,
)
from launch_jolmo.cpt import build_cpt_models, build_cpt_model_evaluations
from launch_jolmo.utils import remote_path
from launch_jolmo.perturb import (
    build_perturbed_models,
    build_perturbed_model_evaluations,
    build_multi_seed_perturbed_models,
    build_multi_seed_perturbed_evaluations,
    LADDER_GAMMAS,
)
from launch_jolmo.interpolate import (
    build_interpolated_models,
    build_interpolated_model_evaluations,
    DEFAULT_ALPHAS as INTERP_ALPHAS,
)


# ---------------------------------------------------------------------------
# GCS paths
# ---------------------------------------------------------------------------

DIVERSITY_V2_GS = "gs://cmu-gpucloud-jspringe/outputs/diversity.pretraining.v2"
# Base prefix of the DCLM shards; the dataset is split into 60 `part-NNN` dirs,
# each holding 5 shards (00000.npy … 00004.npy). The `_part-NNN` suffix is added
# when building all_chunks below.
DCLM_NPY_GS = "gs://cmu-gpucloud-jspringe/shared/datasets/OLMo/dclm/train/preprocessed_dclm_text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_allenai_dolma2-tokenizer"

# ---------------------------------------------------------------------------
# Model & schedule
# ---------------------------------------------------------------------------

TOKENIZER = "allenai/OLMo-2-0425-1B-Instruct"
# Model-size profile (project / model_type / chinchillas) selected by $OPTIM_SIZE
# (default 60M). Must match the project launcher.py initializes — both read
# launch_jolmo.sizes so they can't drift. Examples:
#   OPTIM_SIZE=100M python -m launch_jolmo.launcher <cmd> <stage>
#   OPTIM_SIZE=300M OPTIM_NUM_PROCESSES=8 python -m launch_jolmo.launcher <cmd> <stage>
from launch_jolmo.sizes import active_profile
_SIZE, _PROFILE = active_profile()
PROJECT_NAME = _PROFILE["project"]
MODEL_TYPE = _PROFILE["model_type"]   # 0.06B / 0.1B / 0.3B, …
SCHEDULER = "wsd"
# Drives both the Slurm request (gres=gpu:N) and torchrun's --nproc_per_node.
# Override per-launch with OPTIM_NUM_PROCESSES=8; the global batch is fixed, so
# only the per-rank microbatches below change and the training math is unaffected.
# Default comes from the size profile (300M → 8 GPUs; 60M/100M → 2).
NUM_PROCESSES = int(os.environ.get(
    "OPTIM_NUM_PROCESSES",
    str(_PROFILE.get("num_processes", 2)),
))
CHINCHILLAS: List[float] = list(_PROFILE["chinchillas"])  # token-budget multipliers (may be fractional, e.g. 0.25)

# Pretrain batch & context. CPT uses its OWN (smaller) batch/context — see
# CPT_GLOBAL_BATCH_SIZE / CPT_SEQUENCE_LENGTH in launch_jolmo/cpt.py.
GLOBAL_BATCH_SIZE = 1_048_576   # 1M tokens/step for pretrain
SEQUENCE_LENGTH = 4096          # pretrain context length
RANK_MICROBATCH_SIZE = max(SEQUENCE_LENGTH, GLOBAL_BATCH_SIZE // NUM_PROCESSES // 8)
# Eval logits are (batch, seq_len, vocab_size) — keep small to avoid OOM.
# 4 seqs * 4096 tokens * 100352 vocab * 2 bytes ≈ 3.2 GiB, safely within the ~22 GiB free.
# Floor at one sequence: past 4 ranks the //64 split would drop below seq_len and
# the evaluator cannot form a microbatch smaller than a single sequence.
EVAL_RANK_MICROBATCH_SIZE = max(SEQUENCE_LENGTH, GLOBAL_BATCH_SIZE // NUM_PROCESSES // 64)
VALIDATION_EVAL_INTERVAL = 95


# ---------------------------------------------------------------------------
# Model-size-dependent token budget
#
# The Chinchilla-1 token budget scales with model size (~CHINCHILLA_MULT tokens
# per parameter), so BASE_TOKENS is derived from MODEL_TYPE rather than hard-coded.
# Pin an exact, measured budget for a size in BASE_TOKENS_OVERRIDE; any size not
# listed falls back to CHINCHILLA_MULT × (params parsed from the size tag), rounded
# down to a whole number of global batches.
# ---------------------------------------------------------------------------

CHINCHILLA_MULT = 20    # Chinchilla-optimal tokens per parameter

BASE_TOKENS_OVERRIDE: Dict[str, int] = {
    "0.06B": 1_200_619_520,    # 20 × 60,030,976 measured params
}


def _approx_params(model_type: str) -> int:
    """Approximate parameter count parsed from the size tag, e.g. '0.06B' -> 60_000_000."""
    return int(float(model_type.rstrip("B")) * 1e9)


def _base_tokens_for(model_type: str) -> int:
    """Chinchilla-1 token budget for the given model size."""
    if model_type in BASE_TOKENS_OVERRIDE:
        return BASE_TOKENS_OVERRIDE[model_type]
    raw = CHINCHILLA_MULT * _approx_params(model_type)
    return (raw // GLOBAL_BATCH_SIZE) * GLOBAL_BATCH_SIZE


BASE_TOKENS = _base_tokens_for(MODEL_TYPE)    # tokens at chinchilla-1


def _tokens_for(chinchilla: float) -> Dict[str, Any]:
    """Return the schedule params that depend on chinchilla multiplier."""
    # int() keeps fractional chinchillas (0.25, 0.5) from leaking float
    # n_tokens / warmup_steps into the generated YAML.
    n_tokens = int(BASE_TOKENS * chinchilla)
    total_steps = n_tokens // GLOBAL_BATCH_SIZE
    return {
        "n_tokens": n_tokens,
        "warmup_steps": total_steps // 10,
    }


# ---------------------------------------------------------------------------
# Optimal LR tables — keyed by MODEL_TYPE  (batch: 1M tokens)
#
# PT_LR_BY_MODEL[model_type][scheduler][opt][chinchilla] is the tuned LR:
#   AdamW -> learning_rate
#   Muon  -> (muon_lr, adamw_component_lr)
#
# Selecting MODEL_TYPE picks that size's table (PT_LR below). A model size with no
# entry, or a missing/None cell, falls back to the manual sweep PT_LR_SWEEP — useful
# when the optimal LR is not yet known. Add a table per size as you tune it.
# ---------------------------------------------------------------------------

PT_LR_BY_MODEL: Dict[str, Dict] = {
    "0.03B": {
        # 30M study (OPTIM_SIZE=30M). After
        #     OPTIM_SIZE=30M python -m launch_jolmo.launcher launch pretrain-adamw-wsd
        #     OPTIM_SIZE=30M python -m launch_jolmo.launcher launch eval-pretrain-adamw
        # read the val loss per LR (new_utils/plot_pt_sweep.py, or
        # `python -m new_utils.lr_sweep_dclm plot`) and replace the value in each
        # cell with the LR that won it. Cells fall back individually to
        # PT_LR_SWEEP when None, so a half-filled table is fine — tuned
        # chinchillas pin, the rest keep sweeping.
        #
        # Width prior: this arch is half the width of 0.06B (d_model 128 vs 256),
        # and optimal LR broadly scales inversely with width, so winners were
        # expected at roughly 2x the 0.06B row — the adamw cells below (2e-2 ..
        # 4e-2) are consistent with that.
        "wsd": {
            "adamw": {
                0.25: 1.6e-1,
                0.5: 1.2e-1,
                1: 5.6e-2,
                2: 5.6e-2,
                4: 2.0e-2,
                8: 2.0e-2,
                16: 2.0e-2,
                32: 1.4e-2,
                64: 1.0e-2,
            },
            "muon": {
                # (muon_lr, adamw_component_lr) measured from the completed muon
                # sweep — mean val CE over the four diversity-v2 sets, with the
                # adamw component pinned to the tuned adamw LR above:
                #   c1 : 1.0e-2 -> 4.8592  (1.4e-2 4.8668, 2.8e-2 4.8905, 4e-2 4.8984, 2e-2 4.9197, 5.6e-2 5.0853)
                #   c2 : 1.4e-2 -> 4.7245  (1e-2 4.7247, 2e-2 4.7322, 2.8e-2 4.7473, 4e-2 4.7712, 5.6e-2 4.7930)
                #   c4 : 1.4e-2 -> 4.5875  (2e-2 4.5929, 1e-2 4.5934, 2.8e-2 4.5991, 4e-2 4.6113, 5.6e-2 4.6226)
                #   c8 : 1.0e-2 -> 4.5396  (1.4e-2 4.5400, 2e-2 4.5459, 2.8e-2 4.5518, 4e-2 4.5639, 5.6e-2 4.5764)
                #   c16: 1.4e-2 -> 4.4971  (1e-2 4.4991, 2e-2 4.5019, 2.8e-2 4.5092, 4e-2 4.5355, 5.6e-2 4.5420)
                # Caveats: every winner is at or one step above the BOTTOM of the
                # swept range (1e-2), so the true optimum likely sits below it;
                # and the top-two gaps are <= 0.008 CE, so the picks are soft.
                # Extend the grid downward before treating these as final.
                1: (1.0e-2, 5.6e-2),
                2: (2e-2, 5.6e-2),
                4: (1.4e-2, 2.0e-2),
                8: (1.0e-2, 2.0e-2),
                16: (1.4e-2, 2.0e-2),
                32: (1.0e-2, 1.4e-2),
                64: (1.4e-2, 1.e-2),
            },
        },
    },
    "0.06B": {
        "wsd": {
            "adamw": {
                0.25: 5.6e-2,
                0.5: 5.6e-2,
                1: 2e-2,
                2: 1e-2,
                4: 1e-2,
                8: 1.4e-2,
                16: 1e-2,
                32: 1e-2,
                64: 7e-3,
                128: 1e-2
              
            },
            "muon": {
                # (muon_lr, adamw_component_lr)
                0.25: (1.4e-2,5.6e-2),
                0.5: (1e-2,5.6e-2),
                1: (1.4e-2,2e-2),
                2: (1e-2,1e-2),
                4: (1.4e-2,1e-2),
                8: (1e-2,1.4e-2),
                16: (1.4e-2,1e-2),
                32: (7e-3,1e-2),
                64: (5e-3,7e-3),
                128: (5e-3, 1e-2),

            },
        },
        "cosine": {
            "adamw": {
                1: 4e-2,   # unknown — will sweep PT_LR_SWEEP["adamw"]
                2: 2e-2,
                4: 2e-2,
                8: 1e-2,
            },
            "muon": {
                # (muon_lr, adamw_component_lr)
                1: (1e-2, 4e-2),   # unknown — will sweep PT_LR_SWEEP["muon"]
                2: (1e-2, 2e-2),
                4: (2e-2, 2e-2),
                8: (2e-2, 1e-2),
            },
        },
    },
    "0.1B": {
        # Optimal-LR 0.1B models. Chin 1–4 match the imported jgai sweep
        # (new_utils/import_100m_models.py); chin 8–32 from the local LR sweep.
        # Tied-optimal cells used the first pick.
        "wsd": {
            "adamw": {
                0.25: 4e-2,
                0.5: 2e-2,
                1: 1.4e-2,
                2: 1e-2,
                4: 1e-2,
                8: 7e-3,
                16: 7e-3,
                32: 7e-3,  # tied with 5e-3; first pick kept
            },
            "muon": {
                # (muon_lr, adamw_component_lr)
                0.25: (1.4e-2, 4e-2),
                0.5: (1.4e-2, 2e-2),
                1: (1e-2, 1.4e-2),
                2: (1e-2, 1e-2),
                4: (1e-2, 1e-2),
                8: (1e-2, 7e-3),
                16: (1e-2, 7e-3),
                32: (5e-3, 7e-3),
            },
        },
    },
    # Optimal-LR slots — fill in after LR tuning; a None cell (or missing key)
    # falls back to PT_LR_SWEEP for that chinchilla.
    "0.3B": {
        "wsd": {
            "adamw": {
                0.25: 1e-3,
                0.5: 2.5e-3,
                1: 2.5e-3,
                2: 2.5e-3,
                4: 2.5e-3,
                8: 2.5e-3,
            },
            "muon": {
                # (muon_lr, adamw_component_lr)
                0.25: (2.0e-2, 1e-3),
                0.5: (1.4e-2, 2.5e-3),
                1: (2.0e-2, 2.5e-3),
                2: (1.4e-2, 2.5e-3),
                4: (1.4e-2, 2.5e-3),
                8: (2.0e-2, 2.5e-3),
            },
        },
    },
    "0.6B": {
        "wsd": {
            "adamw": {
                0.25: 2.5e-3,
                0.5: 2.5e-3,
                1: 2.5e-3,
                2: 2.5e-3,
            },
            "muon": {
                0.25: (1.4e-2, 2.5e-3),   # (muon_lr, adamw_component_lr)
                0.5: (1.4e-2, 2.5e-3),
                1: (1.4e-2, 2.5e-3),
                2: (1.4e-2, 2.5e-3),
            },
        },
    },
    # Add a tuned table for each new MODEL_TYPE here; a size left out falls back
    # entirely to PT_LR_SWEEP.
}

# Tuned-LR table for the active model size (empty -> always use PT_LR_SWEEP).
PT_LR = PT_LR_BY_MODEL.get(MODEL_TYPE, {})

# Fallback sweep used when PT_LR[scheduler][ost][chinchilla] is None
PT_LR_SWEEP = {
    "adamw": [2.5e-3, 1e-3, 3.5e-3],
    "muon":  [(1e-2, 2.5e-3), (1.4e-2, 2.5e-3), (2e-2, 2.5e-3)],
}

# ---------------------------------------------------------------------------
# Which optimizers to include in this run — comment out to disable
# ---------------------------------------------------------------------------

OPTIMIZERS: List[str] = [
    "adamw",
    "muon",
]


# ---------------------------------------------------------------------------
# Upload toggle
#
# When False, trained checkpoints are NOT uploaded to GCS (gcloud). The final
# checkpoint is still written + unsharded locally, but nothing is pushed to the
# cloud bucket — useful for throwaway / debug runs. NOTE: downstream artifacts
# (perturbation, CPT, weight-distance, etc.) re-load the uploaded
# final-unsharded/model.pt, so they will NOT find a model from a no-upload run.
# ---------------------------------------------------------------------------

UPLOAD_MODELS: bool = True


# ---------------------------------------------------------------------------
# Training data  (toggle between Option A / B)
# ---------------------------------------------------------------------------

# Option A — Preprocessed DCLM .npy shards (active)
# Layout: 60 part-NNN dirs × 5 shards each (~19.85B tokens/part, raw uint32);
# all 60 = 300 shards ≈ 1.19T unique tokens. Because every train shard is
# downloaded to node-local scratch before training (~74 GiB/part), we include
# only as many WHOLE parts as a run's token budget needs — so a chinchilla-1
# run pulls 1 part (~74 GiB) instead of the full ~4.7 TB. Each model overrides
# train_chunks to its OWN budget (see _make_models); the module-level all_chunks
# default below is sized to the LARGEST chinchilla in the sweep.
DCLM_SHARDS_PER_PART = 5
DCLM_MAX_PARTS = 60
DCLM_TOKENS_PER_PART = 19_852_222_793   # measured: part-000 (5 shards, uint32)


def _dclm_chunks_for_tokens(n_tokens: int) -> tuple:
    """Return just enough DCLM shards (whole parts) to cover ``n_tokens`` unique
    tokens, capped at the full 60-part corpus. Avoids downloading the entire
    ~4.7 TB dataset for small runs (the loader cycles if a run still exceeds the
    included parts)."""
    parts = min(DCLM_MAX_PARTS, max(1, math.ceil(n_tokens / DCLM_TOKENS_PER_PART)))
    return tuple(
        Chunk(uri=f"{DCLM_NPY_GS}_part-{p:03d}/{i:05d}.npy")
        for p in range(parts)
        for i in range(DCLM_SHARDS_PER_PART)
    )


# Default = enough parts for the largest chinchilla in the sweep (per-model
# train_chunks below trim each run down to exactly its own budget).
all_chunks = _dclm_chunks_for_tokens(BASE_TOKENS * max(CHINCHILLAS))

# Option B — Original diversity-v2 .bin shards
# (Comment out Option A and uncomment below to switch back.)
# CHUNK_COUNTS = {"DCLM": 128}
# chunks_by_dataset = {
#     name: tuple(
#         Chunk(uri=f"{DIVERSITY_V2_GS}/PretrainingDataset/Tokenized/Vanilla/{name}/{name}-{i:04d}.bin")
#         for i in range(count)
#     )
#     for name, count in CHUNK_COUNTS.items()
# }
# all_chunks = tuple(c for chunks in chunks_by_dataset.values() for c in chunks)


# ---------------------------------------------------------------------------
# Validation data
# ---------------------------------------------------------------------------

diversity_val_chunks = tuple(
    (n, Chunk(uri=f"{DIVERSITY_V2_GS}/ValidationDataset/Tokenized/{n}.bin"))
    for n in ("C4_val", "Reddit_val", "Wiki_val", "Books_val")
)

# ---------------------------------------------------------------------------
# Held-out DCLM validation
#
# The DCLM corpus is 60 parts (part-000 … part-059). A run only downloads as
# many WHOLE parts as its token budget needs (_dclm_chunks_for_tokens), so this
# sweep — capped at chinchilla=32 ≈ 38.4B tokens ≈ 2 parts — trains only on
# parts 000–001. The LAST shard of the LAST part (part-059/00004.npy) is
# therefore never seen in training: a genuine held-out DCLM eval set. The shard
# is raw uint32 (no .npy header), read identically by validate.py's memmap.
#
# A full shard is ~2.5B tokens (~2.5M sequences); DCLM_HELDOUT_INSTANCES caps
# each eval to a fixed number of sequences (≈ 8.4M tokens) for a fast, stable
# held-out loss (see ModelEvaluation.max_eval_instances).
# ---------------------------------------------------------------------------

DCLM_HELDOUT_PART = DCLM_MAX_PARTS - 1          # 59 — last part, never trained on
DCLM_HELDOUT_SHARD = DCLM_SHARDS_PER_PART - 1   # 4  — last shard of that part
DCLM_HELDOUT_INSTANCES = 1024                   # sequences to score per eval

dclm_heldout_val_chunks = (
    (
        "DCLM_heldout",
        Chunk(uri=f"{DCLM_NPY_GS}_part-{DCLM_HELDOUT_PART:03d}/{DCLM_HELDOUT_SHARD:05d}.npy"),
    ),
)

# ---------------------------------------------------------------------------
# Shared model parameters
# ---------------------------------------------------------------------------

SHARED_MODEL_PARAMS = {
    "model_type": MODEL_TYPE,
    "tokenizer": TOKENIZER,
    "sequence_length": SEQUENCE_LENGTH,
    # Optimizer
    "weight_decay": 0.1,
    "betas": (0.9, 0.98),
    "max_grad_norm": 1.0,
    # Schedule (chinchilla-dependent keys — n_tokens, warmup_steps — added per model)
    "scheduler": SCHEDULER,
    "global_batch_size": GLOBAL_BATCH_SIZE,
    "rank_microbatch_size": RANK_MICROBATCH_SIZE,
    "eval_rank_microbatch_size": EVAL_RANK_MICROBATCH_SIZE,
    "validation_eval_interval": VALIDATION_EVAL_INTERVAL,
    # Parallelism & compilation
    "compile_model": True,
    "parallelism": "ddp",
    "num_processes": NUM_PROCESSES,
    "dp_param_dtype": "bfloat16",
    "dp_reduce_dtype": "float32",
    # Data
    "train_chunks": all_chunks,
    "validation_chunks": diversity_val_chunks,
    # Checkpointing & export
    "save_interval": 1000,
    "ephemeral_save_interval": 500,
    "unshard_checkpoint": True,
    "convert_to_hf": False,
    "upload": UPLOAD_MODELS,
    # Experiment
    "experiment_name": PROJECT_NAME,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lr_tag(lr: float) -> str:
    return f"{lr:.1e}".replace("e-0", "e-")


_jolmo_runs_cache: Optional[set] = None


def _existing_jolmo_runs() -> set:
    """Run names of every JolmoModel dir on GCS (one cached listing)."""
    global _jolmo_runs_cache
    if _jolmo_runs_cache is None:
        base = remote_path("JolmoModel").rstrip("/") + "/"
        out = subprocess.run(["gsutil", "ls", base], capture_output=True, text=True)
        _jolmo_runs_cache = {
            line.strip().rstrip("/").rsplit("/", 1)[-1]
            for line in out.stdout.splitlines() if line.strip()
        }
        print(f"[cpt-all] discovered {len(_jolmo_runs_cache)} pretrained run(s) on GCS")
    return _jolmo_runs_cache


def _pt_run_name(chinchilla, opt: str, tag: str, scheduler: str) -> str:
    """Canonical run name for a pretrained (chinchilla, opt, LR-tag) cell.

    The same trained model can exist under two naming schemas:
    ``MuonExpt3-…-<sched>`` (this matrix) and the LR×BS sweep's
    reference-batch cell ``PTSweep<size>-…-wd0.1-bs1M-wsd`` — an identical
    training configuration. Resolve to whichever name already exists on GCS
    (MuonExpt3 preferred); a cell trained nowhere keeps the MuonExpt3 name.
    """
    legacy = f"MuonExpt3-{MODEL_TYPE}-chinchilla-{chinchilla}-{opt}-{tag}-{scheduler}"
    if scheduler != "wsd":
        return legacy
    runs = _existing_jolmo_runs()
    if legacy in runs:
        return legacy
    twin = (f"PTSweep{_SIZE}-{MODEL_TYPE}-chinchilla-{chinchilla:g}"
            f"-{opt}-{tag}-wd0.1-bs1M-wsd")
    if twin in runs:
        return twin
    return legacy


def _make_models(optimizers: List[str], scheduler: str = SCHEDULER) -> ArtifactSet:
    """Build one JolmoModel per (chinchilla × optimizer × LR) combo for the given scheduler.

    If PT_LR[scheduler][opt][chinchilla] is set, uses that single optimal LR.
    If it is None, falls back to sweeping PT_LR_SWEEP[opt].
    """
    models = []
    sched_table = PT_LR.get(scheduler, {})
    for chinchilla in CHINCHILLAS:
        schedule = _tokens_for(chinchilla)
        # Download only as many DCLM parts as THIS run's token budget needs.
        train_chunks = _dclm_chunks_for_tokens(schedule["n_tokens"])
        for opt in optimizers:
            best = sched_table.get(opt, {}).get(chinchilla)

            if opt == "adamw":
                lrs = [best] if best is not None else PT_LR_SWEEP.get("adamw", [])
                for lr in lrs:
                    tag = f"lr{_lr_tag(lr)}"
                    models.append(JolmoModel(
                        model_name=_pt_run_name(chinchilla, "adamw", tag, scheduler),
                        **{**SHARED_MODEL_PARAMS, **schedule, "scheduler": scheduler, "train_chunks": train_chunks},
                        optimizer="adamw",
                        learning_rate=lr,
                    ))

            elif opt == "muon":
                pairs = [best] if best is not None else PT_LR_SWEEP.get("muon", [])
                for muon_lr, adamw_lr in pairs:
                    tag = f"muonlr{_lr_tag(muon_lr)}-adamwlr{_lr_tag(adamw_lr)}"
                    models.append(JolmoModel(
                        model_name=_pt_run_name(chinchilla, "muon", tag, scheduler),
                        **{**SHARED_MODEL_PARAMS, **schedule, "scheduler": scheduler, "train_chunks": train_chunks},
                        optimizer="muon",
                        muon_lr=muon_lr,
                        learning_rate=adamw_lr,
                    ))

    return ArtifactSet(models)


def _make_all_lr_models(optimizers: List[str], scheduler: str = SCHEDULER) -> ArtifactSet:
    """Like _make_models but always uses ALL sweep LRs for every chinchilla."""
    models = []
    for chinchilla in CHINCHILLAS:
        schedule = _tokens_for(chinchilla)
        # Download only as many DCLM parts as THIS run's token budget needs.
        train_chunks = _dclm_chunks_for_tokens(schedule["n_tokens"])
        for opt in optimizers:
            if opt == "adamw":
                for lr in PT_LR_SWEEP.get("adamw", []):
                    tag = f"lr{_lr_tag(lr)}"
                    models.append(JolmoModel(
                        model_name=_pt_run_name(chinchilla, "adamw", tag, scheduler),
                        **{**SHARED_MODEL_PARAMS, **schedule, "scheduler": scheduler, "train_chunks": train_chunks},
                        optimizer="adamw",
                        learning_rate=lr,
                    ))
            elif opt == "muon":
                for muon_lr, adamw_lr in PT_LR_SWEEP.get("muon", []):
                    tag = f"muonlr{_lr_tag(muon_lr)}-adamwlr{_lr_tag(adamw_lr)}"
                    models.append(JolmoModel(
                        model_name=_pt_run_name(chinchilla, "muon", tag, scheduler),
                        **{**SHARED_MODEL_PARAMS, **schedule, "scheduler": scheduler, "train_chunks": train_chunks},
                        optimizer="muon",
                        muon_lr=muon_lr,
                        learning_rate=adamw_lr,
                    ))
    return ArtifactSet(models)


# ---------------------------------------------------------------------------
# Model sets  (per optimizer × per scheduler)
# ---------------------------------------------------------------------------

pretrain_adamw_wsd     = _make_models(["adamw"], "wsd")    if "adamw" in OPTIMIZERS else ArtifactSet([])
pretrain_adamw_cosine  = _make_models(["adamw"], "cosine") if "adamw" in OPTIMIZERS else ArtifactSet([])
pretrain_muon_wsd      = _make_models(["muon"],  "wsd")    if "muon"  in OPTIMIZERS else ArtifactSet([])
pretrain_muon_cosine   = _make_models(["muon"],  "cosine") if "muon"  in OPTIMIZERS else ArtifactSet([])

# All LR sweep models (includes every LR, not just the optimal one)
pretrain_all_wsd = _make_all_lr_models(OPTIMIZERS, "wsd")

# Convenience aliases
pretrain_adamw_models = pretrain_adamw_wsd
pretrain_muon_models  = pretrain_muon_wsd  + pretrain_muon_cosine


# ---------------------------------------------------------------------------
# CPT models  (via launch_jolmo/cpt.py)
# — always applied to all defined pretrained models
# ---------------------------------------------------------------------------

def tuned_bases_for(chinchillas, optimizers=("adamw", "muon")) -> ArtifactSet:
    """The tuned-LR wsd pretrain for each (chinchilla x optimizer) in the table.

    Independent of the profile's CHINCHILLAS list, so a CPT pass can span a
    wider range than the size is currently focused on. Names go through
    _pt_run_name, so each resolves to whichever schema exists on GCS; a cell
    with no tuned entry is skipped rather than invented.
    """
    sched = PT_LR.get("wsd", {})
    models = []
    for chinchilla in chinchillas:
        schedule = _tokens_for(chinchilla)
        common = {**SHARED_MODEL_PARAMS, **schedule, "scheduler": "wsd",
                  "train_chunks": _dclm_chunks_for_tokens(schedule["n_tokens"])}
        for opt in optimizers:
            best = sched.get(opt, {}).get(chinchilla)
            if best is None:
                continue
            if opt == "adamw":
                tag, kw = f"lr{_lr_tag(best)}", dict(
                    optimizer="adamw", learning_rate=best)
            else:
                muon_lr, adamw_lr = best
                tag = f"muonlr{_lr_tag(muon_lr)}-adamwlr{_lr_tag(adamw_lr)}"
                kw = dict(optimizer="muon", muon_lr=muon_lr,
                          learning_rate=adamw_lr)
            models.append(JolmoModel(
                model_name=_pt_run_name(chinchilla, opt, tag, "wsd"),
                **common, **kw))
    return ArtifactSet(models)


# Tuned-LR CPT over a WIDE chinchilla range (0.25 - 32), not just the profile's
# current focus. adamw bases finetune with adamw; muon bases with muon + adamw.
CPT_WIDE_CHINCHILLAS = (0.25, 0.5, 1, 2, 4, 8, 16, 32)
cpt_wide_adamw_bases = tuned_bases_for(CPT_WIDE_CHINCHILLAS, ("adamw",))
cpt_wide_muon_bases = tuned_bases_for(CPT_WIDE_CHINCHILLAS, ("muon",))
cpt_wide_bases = cpt_wide_adamw_bases + cpt_wide_muon_bases
cpt_wide_models = (
    build_cpt_models(cpt_wide_adamw_bases)
    + build_cpt_models(cpt_wide_muon_bases, cpt_optimizers=["muon", "adamw"],
                       muon_adamw_multiplier=0.25))

cpt_adamw_models = build_cpt_models(pretrain_adamw_wsd)
# cpt-muon: CPT the muon-pretrained models with BOTH optimizers by default —
# muon (adamw-component LR = 0.25 × muon_lr) AND adamw — each over its LR sweep.
cpt_muon_models = build_cpt_models(
    pretrain_muon_wsd, cpt_optimizers=["muon", "adamw"], muon_adamw_multiplier=0.25)
# Muon-pretrained models finetuned with AdamW CPT optimizer (subset of cpt-muon
# above; kept as a standalone stage for launching just the AdamW-FT cells).
cpt_muon_pretrain_adamw_ft = build_cpt_models(pretrain_muon_wsd, cpt_optimizers=["adamw"])
# AdamW-pretrained models finetuned with Muon CPT optimizer
cpt_adamw_pretrain_muon_ft = build_cpt_models(pretrain_adamw_wsd, cpt_optimizers=["muon"], muon_adamw_multiplier=0.25)
# cpt: adamw bases -> adamw CPT, plus muon bases -> BOTH muon and adamw CPT.
cpt_models = cpt_adamw_models + cpt_muon_models


# ---------------------------------------------------------------------------
# CPT over the FULL LR sweep  (cpt-all)
# — CPT the single pretrained checkpoint per (chinchilla × optimizer) whose PT
#   LR is the next step above the tuned optimal, with all four pretrain→CPT
#   optimizer pairings: muon→muon, muon→adamw, adamw→muon, adamw→adamw. The muon_lr while the
#   adamw-component stays at the cell's optimal), so we DISCOVER the actual
#   base models from GCS (a single JolmoModel listing) rather than
#   regenerating them from the LR tables — that way we get exactly the runs
#   that exist, with no phantom cells.
#
# The discovered JolmoModels are fresh objects, so they're registered as the
# `cpt-all-bases` stage: the executor validates CPT dependencies by object
# identity, and a dependency must live in some registered stage. The bases are
# not in the selected `cpt-all` stage, so they are NOT retrained — only used to
# satisfy dependency resolution (their checkpoints already exist on GCS).
#
# Gated on argv so the GCS listing only runs when a cpt-all stage is requested,
# keeping every other launcher command import-time GCS-free.
# ---------------------------------------------------------------------------

_WANT_CPT_ALL = any("cpt-all" in a for a in sys.argv)
# _existing_jolmo_runs (the cached GCS listing) is defined above _make_models,
# which needs it for cross-schema run-name resolution.


def _optimal_pt_lr(chinchilla: int, opt: str) -> Optional[float]:
    """Tuned pretrain LR for (chinchilla, opt); None if unknown.

    For muon the comparable swept channel is muon_lr (adamw component stays at
    the cell's tuned value during the pretrain LR sweep).
    """
    cell = PT_LR.get("wsd", {}).get(opt, {}).get(chinchilla)
    if cell is None:
        return None
    return cell if opt == "adamw" else cell[0]


def _discover_bases(opt: str) -> ArtifactSet:
    """JolmoModels that EXIST on GCS for this optimizer, for chinchillas in
    CHINCHILLAS, reconstructed (with matching run names) from the listing.

    Per chinchilla, keeps only the pretrained checkpoint whose PT LR is the
    smallest value still strictly above the tuned optimal (when known).
    """
    runs = _existing_jolmo_runs()
    models = []
    n_skipped = 0
    for chinchilla in CHINCHILLAS:
        optimal = _optimal_pt_lr(chinchilla, opt)
        if optimal is None:
            continue
        schedule = _tokens_for(chinchilla)
        train_chunks = _dclm_chunks_for_tokens(schedule["n_tokens"])
        common = {**SHARED_MODEL_PARAMS, **schedule, "scheduler": "wsd",
                  "train_chunks": train_chunks}
        prefix = f"MuonExpt3-{MODEL_TYPE}-chinchilla-{chinchilla}-{opt}-"
        candidates = []
        for run in sorted(runs):
            if not (run.startswith(prefix) and run.endswith("-wsd")):
                continue
            if opt == "adamw":
                m = re.search(r"-adamw-lr([0-9.eE+\-]+)-wsd$", run)
                if not m:
                    continue
                pt_lr = float(m.group(1))
                if pt_lr <= optimal:
                    n_skipped += 1
                    continue
                candidates.append((pt_lr, run, dict(optimizer="adamw", learning_rate=pt_lr)))
            else:
                m = re.search(r"-muon-muonlr([0-9.eE+\-]+)-adamwlr([0-9.eE+\-]+)-wsd$", run)
                if not m:
                    continue
                muon_lr = float(m.group(1))
                if muon_lr <= optimal:
                    n_skipped += 1
                    continue
                candidates.append((muon_lr, run, dict(
                    optimizer="muon", muon_lr=muon_lr, learning_rate=float(m.group(2)))))
        if not candidates:
            continue
        _, run, kw = min(candidates, key=lambda c: c[0])
        models.append(JolmoModel(model_name=run, **common, **kw))
        n_skipped += len(candidates) - 1
    if n_skipped:
        print(f"[cpt-all] skipped {n_skipped} {opt} base(s) (not the next-above-optimal PT LR)")
    print(f"[cpt-all] selected {len(models)} {opt} base(s) (next PT LR above tuned optimal per chinchilla)")
    return ArtifactSet(models)


# CPT every discovered base with all four pretrain→CPT optimizer pairings:
#   muon→muon, muon→adamw, adamw→muon, adamw→adamw.
_CPT_ALL_PAIRINGS = dict(cpt_optimizers=["muon", "adamw"], muon_adamw_multiplier=0.25)

if _WANT_CPT_ALL:
    cpt_all_adamw_bases = _discover_bases("adamw") if "adamw" in OPTIMIZERS else ArtifactSet([])
    cpt_all_muon_bases = _discover_bases("muon") if "muon" in OPTIMIZERS else ArtifactSet([])
    cpt_all_adamw_models = build_cpt_models(cpt_all_adamw_bases, **_CPT_ALL_PAIRINGS)
    cpt_all_muon_models = build_cpt_models(cpt_all_muon_bases, **_CPT_ALL_PAIRINGS)
else:
    # Not requested — leave empty (no GCS listing for unrelated commands).
    cpt_all_adamw_bases = ArtifactSet([])
    cpt_all_muon_bases = ArtifactSet([])
    cpt_all_adamw_models = ArtifactSet([])
    cpt_all_muon_models = ArtifactSet([])

cpt_all_bases = cpt_all_adamw_bases + cpt_all_muon_bases
cpt_all_models = cpt_all_adamw_models + cpt_all_muon_models


def _discover_every_base(opt: str) -> ArtifactSet:
    """EVERY JolmoModel that exists on GCS for this optimizer and CHINCHILLAS —
    all pretrain LRs, not just the tuned/next-above-optimal one."""
    runs = _existing_jolmo_runs()
    models = []
    for chinchilla in CHINCHILLAS:
        schedule = _tokens_for(chinchilla)
        train_chunks = _dclm_chunks_for_tokens(schedule["n_tokens"])
        common = {**SHARED_MODEL_PARAMS, **schedule, "scheduler": "wsd",
                  "train_chunks": train_chunks}
        prefix = f"MuonExpt3-{MODEL_TYPE}-chinchilla-{chinchilla}-{opt}-"
        for run in sorted(runs):
            if not (run.startswith(prefix) and run.endswith("-wsd")):
                continue
            if opt == "adamw":
                m = re.search(r"-adamw-lr([0-9.eE+\-]+)-wsd$", run)
                if not m:
                    continue
                kw = dict(optimizer="adamw", learning_rate=float(m.group(1)))
            else:
                m = re.search(r"-muon-muonlr([0-9.eE+\-]+)-adamwlr([0-9.eE+\-]+)-wsd$", run)
                if not m:
                    continue
                kw = dict(optimizer="muon", muon_lr=float(m.group(1)),
                          learning_rate=float(m.group(2)))
            models.append(JolmoModel(model_name=run, **common, **kw))
    print(f"[cpt-all-lrs] discovered {len(models)} existing {opt} base(s) across chinchillas {CHINCHILLAS}")
    return ArtifactSet(models)


# CPT every existing base at every PT LR: adamw bases -> adamw CPT; muon bases
# -> BOTH muon and adamw CPT (same convention as the main `cpt` stage).
if _WANT_CPT_ALL:
    _all_lrs_adamw_bases = _discover_every_base("adamw") if "adamw" in OPTIMIZERS else ArtifactSet([])
    _all_lrs_muon_bases = _discover_every_base("muon") if "muon" in OPTIMIZERS else ArtifactSet([])
    cpt_all_lrs_models = (
        build_cpt_models(_all_lrs_adamw_bases)
        + build_cpt_models(_all_lrs_muon_bases, cpt_optimizers=["muon", "adamw"],
                           muon_adamw_multiplier=0.25)
    )
else:
    cpt_all_lrs_models = ArtifactSet([])


# ---------------------------------------------------------------------------
# Muon alpha sweep  (muon-sweep)
# — CPT the muon-pretrained models while varying alpha = the muon→adamw LR ratio
#   (adamw_component_lr = alpha · muon_lr, via build_cpt_models' muon_adamw_
#   multiplier). One CPT set per alpha; each alpha yields distinct run names
#   (different adamw_lr), so they don't collide and all land in one sweep.
# ---------------------------------------------------------------------------

MUON_ALPHA_SWEEP: List[float] = [0.25, 0.15, 0.35, 0.45]

muon_sweep_models = ArtifactSet([])
for _alpha in MUON_ALPHA_SWEEP:
    muon_sweep_models = muon_sweep_models + build_cpt_models(
        pretrain_muon_wsd, muon_adamw_multiplier=_alpha)


# ---------------------------------------------------------------------------
# Perturbed models  (via launch_jolmo/perturb.py)
# Gaussian weight perturbation (std = γ · ‖W‖_F / √numel) of every float
# weight in the DCLM-pretrained models.
#
# - Single-direction: one noise draw (seed=64) → flat PerturbedModel/{name}/
# - Multi-direction: 10 seeds under PerturbedModel/{name}/seed_XXX/
# ---------------------------------------------------------------------------

perturbed_adamw_models = build_perturbed_models(pretrain_adamw_wsd)
perturbed_muon_models  = build_perturbed_models(pretrain_muon_wsd)

# Gaussian perturbation of EVERY tuned-LR base this size has a table entry for,
# over a gamma ladder one decade above LADDER_GAMMAS' top mantissas. gamma is
# the relative noise scale: std = gamma * ||W||_F / sqrt(numel) per tensor.
PERTURB_WIDE_GAMMAS = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.13]
_perturb_wide_chins = sorted(
    set(PT_LR.get("wsd", {}).get("adamw", {})) | set(PT_LR.get("wsd", {}).get("muon", {})))
perturb_wide_bases = tuned_bases_for(_perturb_wide_chins)
perturb_wide_models = build_perturbed_models(perturb_wide_bases,
                                             gammas=PERTURB_WIDE_GAMMAS)

multiseed_perturbed_adamw_models = build_multi_seed_perturbed_models(pretrain_adamw_wsd, gammas=LADDER_GAMMAS)
multiseed_perturbed_muon_models  = build_multi_seed_perturbed_models(pretrain_muon_wsd, gammas=LADDER_GAMMAS)


# ---------------------------------------------------------------------------
# Pretrained <-> finetuned weight interpolation
# — for each SPECIFIED finetuned (CPT) model, build the convex combination
#   W = alpha * W_pretrained + (1 - alpha) * W_finetuned over INTERP_ALPHAS
#   (= 0.2/0.4/0.6/0.8). `interpolated_models` is the "specified finetuned
#   models" set — the active CPT set (cpt_models); narrow it here (or via the
#   launcher's --head/--tail) to interpolate only particular finetuned runs.
# ---------------------------------------------------------------------------

interpolation_finetuned_models = cpt_adamw_models
interpolated_models = build_interpolated_models(
    interpolation_finetuned_models, alphas=INTERP_ALPHAS)


# ---------------------------------------------------------------------------
# Evaluations
#
# Every pretrain eval scores the diversity-v2 val sets (C4/Reddit/Wiki/Books)
# AND the held-out DCLM shard (part-059/00004.npy), folded into the same eval
# JSON via extra_val_chunks. The held-out DCLM dataset is downloaded shard-only
# and capped to DCLM_HELDOUT_INSTANCES sequences; the diversity sets stay
# uncapped, so their numbers are unchanged from before.
# ---------------------------------------------------------------------------

def _pretrain_eval(model) -> ModelEvaluation:
    """A ModelEvaluation that also scores the held-out DCLM shard."""
    return ModelEvaluation(
        model=model,
        extra_val_chunks=dclm_heldout_val_chunks,
        extra_val_max_instances=DCLM_HELDOUT_INSTANCES,
    )


pretrain_adamw_wsd_evals   = ArtifactSet([_pretrain_eval(m) for m in pretrain_adamw_wsd])
pretrain_adamw_cosine_evals = ArtifactSet([_pretrain_eval(m) for m in pretrain_adamw_cosine])
pretrain_muon_wsd_evals    = ArtifactSet([_pretrain_eval(m) for m in pretrain_muon_wsd])
pretrain_muon_cosine_evals = ArtifactSet([_pretrain_eval(m) for m in pretrain_muon_cosine])

# Evals for all LR sweep models (all LRs, not just optimal)
pretrain_all_wsd_evals     = ArtifactSet([_pretrain_eval(m) for m in pretrain_all_wsd])

# Backward-compat aliases
pretrain_adamw_evals = pretrain_adamw_wsd_evals
pretrain_muon_evals  = pretrain_muon_wsd_evals

# ---------------------------------------------------------------------------
# Every EXISTING adamw pretrain at one chinchilla, reference batch (bs1M) and
# wd0.1, under BOTH naming schemas — discovered from the GCS listing rather
# than rebuilt from a table, so nothing is missed and no name is invented.
# Deliberately no cross-schema collapse: if a config was trained under both
# names, both runs are scored. Gated on argv (the cpt-all pattern).
# ---------------------------------------------------------------------------

_WANT_BS1M_EVALS = any("bs1m" in a.lower() for a in sys.argv)
_WANT_BS_ANY = any("bs-any" in a.lower() for a in sys.argv)


def _existing_adamw_bs1m_models(chinchilla) -> ArtifactSet:
    if not _WANT_BS1M_EVALS:
        return ArtifactSet([])
    tag = re.escape(f"{chinchilla:g}")
    legacy_re = re.compile(
        rf"^MuonExpt3-{re.escape(MODEL_TYPE)}-chinchilla-{tag}"
        rf"-adamw-lr([0-9.e+\-]+)-wsd$")
    sweep_re = re.compile(
        rf"^PTSweep{_SIZE}-{re.escape(MODEL_TYPE)}-chinchilla-{tag}"
        rf"-adamw-lr([0-9.e+\-]+)-wd0\.1-bs1M-wsd$")
    schedule = _tokens_for(chinchilla)
    common = {**SHARED_MODEL_PARAMS, **schedule, "scheduler": "wsd",
              "train_chunks": _dclm_chunks_for_tokens(schedule["n_tokens"])}
    models = []
    for run in sorted(_existing_jolmo_runs()):
        m = legacy_re.match(run) or sweep_re.match(run)
        if not m:
            continue
        models.append(JolmoModel(model_name=run, **common,
                                 optimizer="adamw", learning_rate=float(m.group(1))))
    if models:
        print(f"[bs1m-evals] chinchilla-{chinchilla:g}: {len(models)} existing adamw "
              f"bs1M/wd0.1 pretrain(s) discovered")
    return ArtifactSet(models)


# Every chinchilla either schema has ever trained at 60M.
ADAMW_BS1M_CHINCHILLAS = (0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128)

c8_adamw_bs1m_models = _existing_adamw_bs1m_models(8)
c8_adamw_bs1m_evals = ArtifactSet([_pretrain_eval(m) for m in c8_adamw_bs1m_models])

adamw_bs1m_models = ArtifactSet([
    m for _c in ADAMW_BS1M_CHINCHILLAS for m in _existing_adamw_bs1m_models(_c)])
adamw_bs1m_evals = ArtifactSet([_pretrain_eval(m) for m in adamw_bs1m_models])


# --- Both optimizers, every chinchilla, whatever this $OPTIM_SIZE has ---------
# Chinchillas come from the run names, not a hard-coded list, so the same stage
# works unchanged at 30M / 100M / 300M / 600M.

_BS1M_PATTERNS = (
    # (regex, optimizer) — group 1 chinchilla, 2 main LR, 3 adamw component
    (rf"^MuonExpt3-{re.escape(MODEL_TYPE)}-chinchilla-([0-9.]+)"
     rf"-adamw-lr([0-9.e+\-]+)-wsd$", "adamw"),
    (rf"^PTSweep{_SIZE}-{re.escape(MODEL_TYPE)}-chinchilla-([0-9.]+)"
     rf"-adamw-lr([0-9.e+\-]+)-wd0\.1-bs1M-wsd$", "adamw"),
    (rf"^MuonExpt3-{re.escape(MODEL_TYPE)}-chinchilla-([0-9.]+)"
     rf"-muon-muonlr([0-9.e+\-]+)-adamwlr([0-9.e+\-]+)-wsd$", "muon"),
    (rf"^PTSweep{_SIZE}-{re.escape(MODEL_TYPE)}-chinchilla-([0-9.]+)"
     rf"-muon-muonlr([0-9.e+\-]+)-adamwlr([0-9.e+\-]+)-wd0\.1-bs1M-wsd$", "muon"),
)


def _existing_bs1m_models() -> ArtifactSet:
    """Every existing wd0.1 / bs1M pretrain at this size, both optimizers."""
    if not _WANT_BS1M_EVALS:
        return ArtifactSet([])
    compiled = [(re.compile(p), opt) for p, opt in _BS1M_PATTERNS]
    by_chin = {}
    for run in sorted(_existing_jolmo_runs()):
        for rx, opt in compiled:
            m = rx.match(run)
            if not m:
                continue
            chin = float(m.group(1))
            kw = ({"optimizer": "adamw", "learning_rate": float(m.group(2))}
                  if opt == "adamw" else
                  {"optimizer": "muon", "muon_lr": float(m.group(2)),
                   "learning_rate": float(m.group(3))})
            by_chin.setdefault(chin, []).append((run, kw))
            break
    models = []
    for chin in sorted(by_chin):
        schedule = _tokens_for(chin)
        common = {**SHARED_MODEL_PARAMS, **schedule, "scheduler": "wsd",
                  "train_chunks": _dclm_chunks_for_tokens(schedule["n_tokens"])}
        for run, kw in by_chin[chin]:
            models.append(JolmoModel(model_name=run, **common, **kw))
        n_ad = sum(1 for _, k in by_chin[chin] if k["optimizer"] == "adamw")
        print(f"[bs1m-evals] {_SIZE} chinchilla-{chin:g}: {n_ad} adamw + "
              f"{len(by_chin[chin]) - n_ad} muon")
    print(f"[bs1m-evals] {_SIZE}: {len(models)} bs1M/wd0.1 pretrain(s) total")
    return ArtifactSet(models)


bs1m_models = _existing_bs1m_models()
bs1m_evals = ArtifactSet([_pretrain_eval(m) for m in bs1m_models])


# --- Any batch size, both optimizers, every chinchilla -----------------------
# The full-grid lrbs stages cannot be used to evaluate the bs2M/bs4M cells:
# SWEEP_LRS and the muon component pin have both changed since those ran, so
# the grid they now enumerate mostly does NOT exist on GCS and selecting them
# would train new models. Discovering from the listing instead evaluates
# exactly what is there.

_BS_ANY_PATTERNS = (
    (rf"^PTSweep{_SIZE}-{re.escape(MODEL_TYPE)}-chinchilla-([0-9.]+)"
     rf"-adamw-lr([0-9.e+\-]+)-wd0\.1-bs(\d+[Mk])-wsd$", "adamw"),
    (rf"^PTSweep{_SIZE}-{re.escape(MODEL_TYPE)}-chinchilla-([0-9.]+)"
     rf"-muon-muonlr([0-9.e+\-]+)-adamwlr([0-9.e+\-]+)-wd0\.1-bs(\d+[Mk])-wsd$",
     "muon"),
)


def _existing_bs_any_models() -> ArtifactSet:
    """Every existing wd0.1 PTSweep pretrain at this size, ANY batch size."""
    if not _WANT_BS_ANY:
        return ArtifactSet([])
    compiled = [(re.compile(p), opt) for p, opt in _BS_ANY_PATTERNS]
    by_chin = {}
    for run in sorted(_existing_jolmo_runs()):
        for rx, opt in compiled:
            m = rx.match(run)
            if not m:
                continue
            chin = float(m.group(1))
            kw = ({"optimizer": "adamw", "learning_rate": float(m.group(2))}
                  if opt == "adamw" else
                  {"optimizer": "muon", "muon_lr": float(m.group(2)),
                   "learning_rate": float(m.group(3))})
            by_chin.setdefault(chin, []).append((run, kw))
            break
    models = []
    for chin in sorted(by_chin):
        schedule = _tokens_for(chin)
        common = {**SHARED_MODEL_PARAMS, **schedule, "scheduler": "wsd",
                  "train_chunks": _dclm_chunks_for_tokens(schedule["n_tokens"])}
        for run, kw in by_chin[chin]:
            models.append(JolmoModel(model_name=run, **common, **kw))
    print(f"[bs-any] {_SIZE}: {len(models)} wd0.1 pretrain(s) at any batch size")
    return ArtifactSet(models)


bs_any_models = _existing_bs_any_models()
bs_any_evals = ArtifactSet([_pretrain_eval(m) for m in bs_any_models])

# CPT evals also score the held-out DCLM shard (label "DCLM_heldout"): the
# forgetting / pretrain-loss axis (x) of the Pareto tradeoff plots, vs the CPT
# dataset val loss (y).
_dclm = dict(extra_val_chunks=dclm_heldout_val_chunks,
             extra_val_max_instances=DCLM_HELDOUT_INSTANCES)
cpt_evals                  = build_cpt_model_evaluations(cpt_models, **_dclm)
# Evals for the wide (0.25 - 32) tuned-LR CPT matrix.
cpt_wide_evals             = build_cpt_model_evaluations(cpt_wide_models, **_dclm)
cpt_all_lrs_evals          = build_cpt_model_evaluations(cpt_all_lrs_models, **_dclm)
cpt_muon_pretrain_adamw_ft_evals = build_cpt_model_evaluations(cpt_muon_pretrain_adamw_ft, **_dclm)
cpt_adamw_pretrain_muon_ft_evals = build_cpt_model_evaluations(cpt_adamw_pretrain_muon_ft, **_dclm)

# Full-LR-sweep CPT evals (empty unless a cpt-all stage was requested; see above).
cpt_all_evals       = build_cpt_model_evaluations(cpt_all_models, **_dclm)
cpt_all_adamw_evals = build_cpt_model_evaluations(cpt_all_adamw_models, **_dclm)
cpt_all_muon_evals  = build_cpt_model_evaluations(cpt_all_muon_models, **_dclm)

# Muon alpha-sweep evals (one per CPT model across all alphas).
muon_sweep_evals    = build_cpt_model_evaluations(muon_sweep_models, **_dclm)

# Perturbed-model loss evals. Like the pretrain evals, fold in the held-out DCLM
# shard (label "DCLM_heldout") so each perturbed model is scored on the DCLM split.
# DCLM-heldout evals for the wide perturbation sweep.
perturb_wide_evals = build_perturbed_model_evaluations(
    perturb_wide_models,
    extra_val_chunks=dclm_heldout_val_chunks,
    extra_val_max_instances=DCLM_HELDOUT_INSTANCES)

# ---------------------------------------------------------------------------
# EWC finetuning — a third method beside plain CPT and DCLM replay.
#
# Bases are the chinchilla-4 models of the active size, so run with
# OPTIM_SIZE=60M. The Fisher is estimated on dclm_replay_chunks: the SAME unseen
# DCLM shard replay draws from. It is beyond every run's training data and is
# distinct from the held-out eval shard, so estimating F cannot contaminate the
# DCLM_heldout axis these runs are scored on.
# ---------------------------------------------------------------------------

EWC_CHINCHILLA = 4
EWC_DATASETS = ["gsm8k"]

from launch_jolmo.cpt import CPT_LR_SWEEP  # noqa: E402
from launch_jolmo.ewc import EWC_LAMBDAS, build_ewc_models  # noqa: E402
from launch_jolmo.ft_sweep import REPLAY_DCLM_CHUNK as _EWC_FISHER_CHUNK  # noqa: E402

dclm_replay_chunks: Tuple[Chunk, ...] = (_EWC_FISHER_CHUNK,)

ewc_bases = tuned_bases_for([EWC_CHINCHILLA])
ewc_models = build_ewc_models(
    ewc_bases,
    fisher_chunks=dclm_replay_chunks,
    lr_sweep=CPT_LR_SWEEP,
    lambdas=EWC_LAMBDAS,
    datasets=EWC_DATASETS,
)
ewc_evals = ArtifactSet([_pretrain_eval(m) for m in ewc_models])


perturbed_adamw_evals      = build_perturbed_model_evaluations(
    perturbed_adamw_models,
    extra_val_chunks=dclm_heldout_val_chunks,
    extra_val_max_instances=DCLM_HELDOUT_INSTANCES,
)
perturbed_muon_evals       = build_perturbed_model_evaluations(
    perturbed_muon_models,
    extra_val_chunks=dclm_heldout_val_chunks,
    extra_val_max_instances=DCLM_HELDOUT_INSTANCES,
)

# Multi-seed: mean DCLM_heldout loss over the 10 saved noise directions.
multiseed_perturbed_adamw_evals = build_multi_seed_perturbed_evaluations(
    multiseed_perturbed_adamw_models,
    extra_val_chunks=dclm_heldout_val_chunks,
    extra_val_max_instances=DCLM_HELDOUT_INSTANCES,
)
multiseed_perturbed_muon_evals = build_multi_seed_perturbed_evaluations(
    multiseed_perturbed_muon_models,
    extra_val_chunks=dclm_heldout_val_chunks,
    extra_val_max_instances=DCLM_HELDOUT_INSTANCES,
)

# Interpolated-model evals. Like the CPT evals, fold in the held-out DCLM shard
# (label "DCLM_heldout") so each interpolated point has both the forgetting loss
# (x) and the finetuning-dataset val loss (y) for the Pareto curve.
interpolated_evals         = build_interpolated_model_evaluations(
    interpolated_models, **_dclm)


# ---------------------------------------------------------------------------
# Divergence evaluations  (per-token KL/JSD vs. reference OLMo 2 on DCLM heldout)
#
# One DivergenceEvaluation per pretrained base × reference model. We DISCOVER the
# trained wsd checkpoints from the GCS JolmoModel listing (matching run names)
# instead of rebuilding from the CHINCHILLAS list — so the sweep covers the full
# chinchilla 1..128 × {adamw, muon} set that actually exists, independent of
# whatever single CHINCHILLAS value this config is currently focused on.
# Gated on argv so the GCS listing only runs when a divergence stage is asked
# for, keeping every other launcher command import-time GCS-free.
#
# References: ``DIVERGENCE_REF_OLMO2_32B`` (A100-80GB, batch=1),
# ``DIVERGENCE_REF_OLMO2_13B`` / ``DIVERGENCE_REF_OLMO2_7B`` (A100-40GB, batch=8).
# ---------------------------------------------------------------------------

_WANT_DIVERGENCE = any("divergence" in a for a in sys.argv)
_WANT_CE_LOSS = any("ce-loss" in a for a in sys.argv)
_WANT_C4_DIVERGENCE = any("c4-divergence" in a for a in sys.argv)
_WANT_DIVERGENCE_CPT = any("divergence-cpt" in a for a in sys.argv)
_WANT_DIVERGENCE_PERTURB = any("divergence-perturb" in a for a in sys.argv)
_WANT_LOGIT_PERTURB_KL = any("logit-perturb-kl" in a for a in sys.argv)
_WANT_LOGIT_PERTURB = any(
    ("logit-perturb" in a and "logit-perturb-kl" not in a) for a in sys.argv
)
_WANT_LOGIT_COSINE = any("logit-cosine" in a for a in sys.argv)
_WANT_LOGIT_ANGLE_BINS = any("logit-angle-bins" in a for a in sys.argv)
_WANT_LOGIT_ANGLE_PERTURB = any("logit-angle-perturb" in a for a in sys.argv)
_WANT_WEIGHT_ANGLE_PERTURB = any("weight-angle-perturb" in a for a in sys.argv)

# The canonical pretraining sweep: one tuned-LR model per chinchilla × optimizer
# (the wsd PT_LR table), independent of whatever single CHINCHILLAS value this
# config is currently focused on. These are "the models in pretraining_matrix".
DIVERGENCE_CHINCHILLAS: List[int] = [1, 2, 4, 8, 16, 32, 64, 128]


def _tuned_divergence_models(opt: str) -> ArtifactSet:
    """The tuned-LR wsd JolmoModel for each chinchilla in DIVERGENCE_CHINCHILLAS,
    reconstructed with run names matching the trained checkpoints. When a
    divergence stage is requested, intersect with the GCS listing and warn about
    any tuned model whose checkpoint is not actually present."""
    table = PT_LR.get("wsd", {}).get(opt, {})
    existing = _existing_jolmo_runs() if (
        _WANT_DIVERGENCE or _WANT_CE_LOSS or _WANT_C4_DIVERGENCE
        or _WANT_LOGIT_PERTURB or _WANT_LOGIT_PERTURB_KL         or _WANT_LOGIT_COSINE
        or _WANT_LOGIT_ANGLE_BINS or _WANT_LOGIT_ANGLE_PERTURB
        or _WANT_WEIGHT_ANGLE_PERTURB
    ) else None
    models = []
    for chinchilla in DIVERGENCE_CHINCHILLAS:
        best = table.get(chinchilla)
        if best is None:
            continue
        schedule = _tokens_for(chinchilla)
        common = {**SHARED_MODEL_PARAMS, **schedule, "scheduler": "wsd",
                  "train_chunks": _dclm_chunks_for_tokens(schedule["n_tokens"])}
        if opt == "adamw":
            run = f"MuonExpt3-{MODEL_TYPE}-chinchilla-{chinchilla}-adamw-lr{_lr_tag(best)}-wsd"
            model = JolmoModel(model_name=run, **common, optimizer="adamw", learning_rate=best)
        else:
            muon_lr, adamw_lr = best
            run = (f"MuonExpt3-{MODEL_TYPE}-chinchilla-{chinchilla}-muon-"
                   f"muonlr{_lr_tag(muon_lr)}-adamwlr{_lr_tag(adamw_lr)}-wsd")
            model = JolmoModel(model_name=run, **common, optimizer="muon",
                               muon_lr=muon_lr, learning_rate=adamw_lr)
        if existing is not None and run not in existing:
            print(f"[divergence] WARNING: tuned {opt} chinchilla-{chinchilla} "
                  f"checkpoint not on GCS, skipping: {run}")
            continue
        models.append(model)
    return ArtifactSet(models)


if (
    _WANT_DIVERGENCE or _WANT_CE_LOSS or _WANT_C4_DIVERGENCE
    or _WANT_LOGIT_PERTURB or _WANT_LOGIT_PERTURB_KL     or _WANT_LOGIT_COSINE
    or _WANT_LOGIT_ANGLE_BINS or _WANT_LOGIT_ANGLE_PERTURB
    or _WANT_WEIGHT_ANGLE_PERTURB
):
    divergence_adamw_bases = _tuned_divergence_models("adamw") if "adamw" in OPTIMIZERS else ArtifactSet([])
    divergence_muon_bases  = _tuned_divergence_models("muon")  if "muon"  in OPTIMIZERS else ArtifactSet([])
else:
    divergence_adamw_bases = ArtifactSet([])
    divergence_muon_bases  = ArtifactSet([])

# The discovered bases are fresh JolmoModel objects, so — like cpt-all-bases —
# they must be registered as their own stage for the executor's identity-based
# dependency check to resolve DivergenceEvaluation.model. Their checkpoints
# already exist on GCS (JolmoModel.exists → True), so they are NOT retrained.
divergence_bases = divergence_adamw_bases + divergence_muon_bases


def _divergence_evals(
    bases: ArtifactSet,
    reference_model: str,
    *,
    max_eval_tokens: Optional[int] = None,
    max_eval_instances: Optional[int] = None,
) -> ArtifactSet:
    batch_size = divergence_max_batch_size(reference_model)
    kwargs = {}
    if max_eval_tokens is not None:
        kwargs["max_eval_tokens"] = max_eval_tokens
    if max_eval_instances is not None:
        kwargs["max_eval_instances"] = max_eval_instances
    return ArtifactSet(
        [
            DivergenceEvaluation(
                model=m,
                reference_model=reference_model,
                max_batch_size=batch_size,
                **kwargs,
            )
            for m in bases
        ]
    )


# vs. OLMo 2 32B (existing stages — names unchanged for backward compatibility)
divergence_adamw_evals = _divergence_evals(divergence_adamw_bases, DIVERGENCE_REF_OLMO2_32B)
divergence_muon_evals = _divergence_evals(divergence_muon_bases, DIVERGENCE_REF_OLMO2_32B)
divergence_all_evals = divergence_adamw_evals + divergence_muon_evals

# vs. OLMo 2 13B
divergence_13b_adamw_evals = _divergence_evals(divergence_adamw_bases, DIVERGENCE_REF_OLMO2_13B)
divergence_13b_muon_evals = _divergence_evals(divergence_muon_bases, DIVERGENCE_REF_OLMO2_13B)
divergence_13b_all_evals = divergence_13b_adamw_evals + divergence_13b_muon_evals

# vs. OLMo 2 7B (base pretrained, 1124 family)
divergence_7b_adamw_evals = _divergence_evals(divergence_adamw_bases, DIVERGENCE_REF_OLMO2_7B)
divergence_7b_muon_evals = _divergence_evals(divergence_muon_bases, DIVERGENCE_REF_OLMO2_7B)
divergence_7b_all_evals = divergence_7b_adamw_evals + divergence_7b_muon_evals

# ---------------------------------------------------------------------------
# CE-loss evaluations (per-token NLL on C4_val; legacy 1B-reference pipeline)
# ---------------------------------------------------------------------------

def _ce_loss_evals(bases: ArtifactSet) -> ArtifactSet:
    return ArtifactSet([CeLossEvaluation(model=m) for m in bases])


if _WANT_CE_LOSS:
    ce_loss_teacher_evals = ArtifactSet([
        CeLossEvaluation(
            hf_model=DIVERGENCE_REF_OLMO2_1B,
            output_name=CE_LOSS_REF_TAG_1B,
        ),
    ])
    ce_loss_adamw_evals = _ce_loss_evals(divergence_adamw_bases)
    ce_loss_muon_evals = _ce_loss_evals(divergence_muon_bases)
    ce_loss_all_evals = ce_loss_adamw_evals + ce_loss_muon_evals
else:
    ce_loss_teacher_evals = ArtifactSet([])
    ce_loss_adamw_evals = ArtifactSet([])
    ce_loss_muon_evals = ArtifactSet([])
    ce_loss_all_evals = ArtifactSet([])

# ---------------------------------------------------------------------------
# Logit-perturb evaluations (relative ℓ₂ noise on per-token logits → CE vs σ)
# LLM analogue of AM multi_min_output_perturbation. Uses tuned-LR divergence bases.
# ---------------------------------------------------------------------------

def _logit_perturb_evals(bases: ArtifactSet) -> ArtifactSet:
    return ArtifactSet([LogitPerturbEvaluation(model=m) for m in bases])


if _WANT_LOGIT_PERTURB:
    logit_perturb_adamw_evals = _logit_perturb_evals(divergence_adamw_bases)
    logit_perturb_muon_evals = _logit_perturb_evals(divergence_muon_bases)
    logit_perturb_all_evals = logit_perturb_adamw_evals + logit_perturb_muon_evals
else:
    logit_perturb_adamw_evals = ArtifactSet([])
    logit_perturb_muon_evals = ArtifactSet([])
    logit_perturb_all_evals = ArtifactSet([])

# ---------------------------------------------------------------------------
# Logit-perturb KL evaluations (same noise; KL(Q‖P) vs OLMo-2 1B)
# ---------------------------------------------------------------------------

def _logit_perturb_kl_evals(bases: ArtifactSet) -> ArtifactSet:
    return ArtifactSet([LogitPerturbKlEvaluation(model=m) for m in bases])


if _WANT_LOGIT_PERTURB_KL:
    logit_perturb_kl_adamw_evals = _logit_perturb_kl_evals(divergence_adamw_bases)
    logit_perturb_kl_muon_evals = _logit_perturb_kl_evals(divergence_muon_bases)
    logit_perturb_kl_all_evals = (
        logit_perturb_kl_adamw_evals + logit_perturb_kl_muon_evals
    )
else:
    logit_perturb_kl_adamw_evals = ArtifactSet([])
    logit_perturb_kl_muon_evals = ArtifactSet([])
    logit_perturb_kl_all_evals = ArtifactSet([])

# ---------------------------------------------------------------------------
# Logit cosine: per-token cos(ℓ_adamw, ℓ_muon) for each matched chinchilla pair
# ---------------------------------------------------------------------------

def _parse_chinchilla_from_run(run_name: str) -> Optional[float]:
    m = re.search(r"-chinchilla-([0-9.]+)-", run_name)
    if not m:
        return None
    v = float(m.group(1))
    return int(v) if v.is_integer() else v


def _logit_cosine_evals(adamw_bases: ArtifactSet, muon_bases: ArtifactSet) -> ArtifactSet:
    by_chin_a = {}
    by_chin_m = {}
    for m in adamw_bases:
        chin = _parse_chinchilla_from_run(m.run_name)
        if chin is not None:
            by_chin_a[chin] = m
    for m in muon_bases:
        chin = _parse_chinchilla_from_run(m.run_name)
        if chin is not None:
            by_chin_m[chin] = m
    out = []
    for chin in sorted(set(by_chin_a) & set(by_chin_m)):
        out.append(
            LogitCosineEvaluation(
                adamw_model=by_chin_a[chin],
                muon_model=by_chin_m[chin],
                chinchilla=chin,
            )
        )
    missing_a = sorted(set(by_chin_m) - set(by_chin_a))
    missing_m = sorted(set(by_chin_a) - set(by_chin_m))
    if missing_a:
        print(f"[logit-cosine] WARNING: no adamw pair for chinchillas {missing_a}")
    if missing_m:
        print(f"[logit-cosine] WARNING: no muon pair for chinchillas {missing_m}")
    return ArtifactSet(out)


if _WANT_LOGIT_COSINE:
    logit_cosine_evals = _logit_cosine_evals(
        divergence_adamw_bases, divergence_muon_bases
    )
else:
    logit_cosine_evals = ArtifactSet([])

# ---------------------------------------------------------------------------
# Logit angle-bin metrics: margin / NLL / KL / freq sliced by θ(adamw, muon)
# ---------------------------------------------------------------------------

def _logit_angle_bin_evals(adamw_bases: ArtifactSet, muon_bases: ArtifactSet) -> ArtifactSet:
    by_chin_a = {}
    by_chin_m = {}
    for m in adamw_bases:
        chin = _parse_chinchilla_from_run(m.run_name)
        if chin is not None:
            by_chin_a[chin] = m
    for m in muon_bases:
        chin = _parse_chinchilla_from_run(m.run_name)
        if chin is not None:
            by_chin_m[chin] = m
    out = []
    for chin in sorted(set(by_chin_a) & set(by_chin_m)):
        out.append(
            LogitAngleBinEvaluation(
                adamw_model=by_chin_a[chin],
                muon_model=by_chin_m[chin],
                chinchilla=chin,
            )
        )
    return ArtifactSet(out)


if _WANT_LOGIT_ANGLE_BINS:
    logit_angle_bin_evals = _logit_angle_bin_evals(
        divergence_adamw_bases, divergence_muon_bases
    )
else:
    logit_angle_bin_evals = ArtifactSet([])

# ---------------------------------------------------------------------------
# Angle-bin stratified logit perturbation (ΔNLL distributions per θ-bin)
# ---------------------------------------------------------------------------

def _logit_angle_perturb_evals(adamw_bases: ArtifactSet, muon_bases: ArtifactSet) -> ArtifactSet:
    by_chin_a, by_chin_m = {}, {}
    for m in adamw_bases:
        chin = _parse_chinchilla_from_run(m.run_name)
        if chin is not None:
            by_chin_a[chin] = m
    for m in muon_bases:
        chin = _parse_chinchilla_from_run(m.run_name)
        if chin is not None:
            by_chin_m[chin] = m
    return ArtifactSet([
        LogitAngleBinPerturbEvaluation(
            adamw_model=by_chin_a[chin],
            muon_model=by_chin_m[chin],
            chinchilla=chin,
        )
        for chin in sorted(set(by_chin_a) & set(by_chin_m))
    ])


if _WANT_LOGIT_ANGLE_PERTURB:
    logit_angle_perturb_evals = _logit_angle_perturb_evals(
        divergence_adamw_bases, divergence_muon_bases
    )
else:
    logit_angle_perturb_evals = ArtifactSet([])

# ---------------------------------------------------------------------------
# Weight Gaussian perturb → ΔNLL per adamw↔muon logit-angle data group
# ---------------------------------------------------------------------------

def _weight_angle_perturb_evals(adamw_bases: ArtifactSet, muon_bases: ArtifactSet) -> ArtifactSet:
    by_chin_a, by_chin_m = {}, {}
    for m in adamw_bases:
        chin = _parse_chinchilla_from_run(m.run_name)
        if chin is not None:
            by_chin_a[chin] = m
    for m in muon_bases:
        chin = _parse_chinchilla_from_run(m.run_name)
        if chin is not None:
            by_chin_m[chin] = m
    return ArtifactSet([
        WeightAngleBinPerturbEvaluation(
            adamw_model=by_chin_a[chin],
            muon_model=by_chin_m[chin],
            chinchilla=chin,
        )
        for chin in sorted(set(by_chin_a) & set(by_chin_m))
    ])


if _WANT_WEIGHT_ANGLE_PERTURB:
    weight_angle_perturb_evals = _weight_angle_perturb_evals(
        divergence_adamw_bases, divergence_muon_bases
    )
else:
    weight_angle_perturb_evals = ArtifactSet([])

# ---------------------------------------------------------------------------
# C4 divergence evaluations (per-token KL/JSD on C4_val vs 1B; CPT + pretrain)
#
# CPT stages use AdamW finetune on gsm8k @ 20M tokens, lr=1e-3 — one FT checkpoint
# per tuned pretrain base (chinchilla 1..128). Gated on argv.
# ---------------------------------------------------------------------------

C4_DIVERGENCE_CPT_DATASET = "gsm8k"
C4_DIVERGENCE_CPT_LR = 1e-3


def _c4_divergence_evals(models: ArtifactSet) -> ArtifactSet:
    return ArtifactSet([C4DivergenceEvaluation(model=m) for m in models])


def _tuned_pretrain_run_names() -> set[str]:
    return {m.run_name for m in divergence_adamw_bases + divergence_muon_bases}


def _filter_cpt_for_c4_divergence(models: ArtifactSet) -> ArtifactSet:
    """AdamW CPT on gsm8k @ lr=1e-3 for tuned pretrain bases with checkpoints."""
    tuned = _tuned_pretrain_run_names()
    out = []
    for m in models:
        if m.pretrained_model.run_name not in tuned:
            continue
        if m.cpt_dataset != C4_DIVERGENCE_CPT_DATASET:
            continue
        if m.optimizer != "adamw":
            continue
        if abs(m.learning_rate - C4_DIVERGENCE_CPT_LR) > 1e-12:
            continue
        if not m.exists:
            print(f"[c4-divergence] WARNING: CPT checkpoint not on GCS, skipping: {m.run_name}")
            continue
        out.append(m)
    return ArtifactSet(out)


if _WANT_C4_DIVERGENCE:
    c4_divergence_cpt_bases = _filter_cpt_for_c4_divergence(
        cpt_muon_pretrain_adamw_ft + cpt_adamw_models
    )
    c4_divergence_cpt_muon_ft_evals = _c4_divergence_evals(
        _filter_cpt_for_c4_divergence(cpt_muon_pretrain_adamw_ft)
    )
    c4_divergence_cpt_adamw_evals = _c4_divergence_evals(
        _filter_cpt_for_c4_divergence(cpt_adamw_models)
    )
    c4_divergence_cpt_all_evals = (
        c4_divergence_cpt_muon_ft_evals + c4_divergence_cpt_adamw_evals
    )
    c4_divergence_pretrain_evals = _c4_divergence_evals(divergence_bases)
    # C4 divergence for Gaussian-perturbed pretrain checkpoints (one draw per model × gamma).
    # Depends on perturb-adamw / perturb-muon stages having written checkpoints to GCS.
    c4_divergence_perturbed_adamw_evals = _c4_divergence_evals(perturbed_adamw_models)
    c4_divergence_perturbed_muon_evals  = _c4_divergence_evals(perturbed_muon_models)
    c4_divergence_perturbed_all_evals   = (
        c4_divergence_perturbed_adamw_evals + c4_divergence_perturbed_muon_evals
    )
else:
    c4_divergence_cpt_bases = ArtifactSet([])
    c4_divergence_cpt_muon_ft_evals = ArtifactSet([])
    c4_divergence_cpt_adamw_evals = ArtifactSet([])
    c4_divergence_cpt_all_evals = ArtifactSet([])
    c4_divergence_pretrain_evals = ArtifactSet([])
    c4_divergence_perturbed_adamw_evals = ArtifactSet([])
    c4_divergence_perturbed_muon_evals  = ArtifactSet([])
    c4_divergence_perturbed_all_evals   = ArtifactSet([])


# ---------------------------------------------------------------------------
# DCLM-heldout DivergenceEvaluation for CPT + perturbed (chin-64 gsm8k focus)
#
# Same metric as ``divergence-*`` (per-token KL vs OLMo-2 on DCLM_heldout), but
# for CPT / perturbed checkpoints. Default reference = 7B (batch=8) to match the
# existing chin-64 pretrain dumps. Gated on argv.
# ---------------------------------------------------------------------------

DCLM_DIV_CPT_CHINCHILLA = 64
DCLM_DIV_CPT_DATASET = "gsm8k"
DCLM_DIV_REF = DIVERGENCE_REF_OLMO2_7B
# Cap DCLM heldout per-token KL at ~1M next-token positions (~244 seqs @ L=4096).
DCLM_DIV_MAX_TOKENS = 1_000_000


def _chin_from_run(run_name: str) -> Optional[int]:
    m = re.search(r"-chinchilla-([0-9.]+)-", run_name)
    if not m:
        return None
    v = float(m.group(1))
    return int(v) if v.is_integer() else None


def _filter_cpt_for_dclm_divergence(models: ArtifactSet) -> ArtifactSet:
    """gsm8k CPT on optimal-PT chin-64 bases, every CPT LR × CPT optimizer."""
    out = []
    for m in models:
        if m.cpt_dataset != DCLM_DIV_CPT_DATASET:
            continue
        if _chin_from_run(m.run_name) != DCLM_DIV_CPT_CHINCHILLA:
            continue
        if not m.exists:
            print(f"[divergence-cpt] WARNING: CPT checkpoint not on GCS, "
                  f"skipping: {m.run_name}")
            continue
        out.append(m)
    return ArtifactSet(out)


def _filter_pert_for_dclm_divergence(models: ArtifactSet) -> ArtifactSet:
    """Gaussian-perturbed optimal-PT models at chin-64."""
    out = []
    for m in models:
        if _chin_from_run(m.run_name) != DCLM_DIV_CPT_CHINCHILLA:
            continue
        if not m.exists:
            print(f"[divergence-perturb] WARNING: perturbed checkpoint not on "
                  f"GCS, skipping: {m.run_name}")
            continue
        out.append(m)
    return ArtifactSet(out)


if _WANT_DIVERGENCE_CPT:
    # All optimizer pairings on optimal PT bases (current CPT_LR_SWEEP).
    _cpt_pool = (
        cpt_adamw_models
        + cpt_adamw_pretrain_muon_ft
        + cpt_muon_models
    )
    divergence_cpt_bases = _filter_cpt_for_dclm_divergence(_cpt_pool)
    divergence_cpt_adamw_evals = _divergence_evals(
        ArtifactSet([m for m in divergence_cpt_bases
                     if m.pretrained_model.optimizer == "adamw"]),
        DCLM_DIV_REF,
        max_eval_tokens=DCLM_DIV_MAX_TOKENS,
    )
    divergence_cpt_muon_evals = _divergence_evals(
        ArtifactSet([m for m in divergence_cpt_bases
                     if m.pretrained_model.optimizer == "muon"]),
        DCLM_DIV_REF,
        max_eval_tokens=DCLM_DIV_MAX_TOKENS,
    )
    divergence_cpt_all_evals = divergence_cpt_adamw_evals + divergence_cpt_muon_evals
else:
    divergence_cpt_bases = ArtifactSet([])
    divergence_cpt_adamw_evals = ArtifactSet([])
    divergence_cpt_muon_evals = ArtifactSet([])
    divergence_cpt_all_evals = ArtifactSet([])


if _WANT_DIVERGENCE_PERTURB:
    divergence_perturb_bases = (
        _filter_pert_for_dclm_divergence(perturbed_adamw_models)
        + _filter_pert_for_dclm_divergence(perturbed_muon_models)
    )
    divergence_perturb_adamw_evals = _divergence_evals(
        _filter_pert_for_dclm_divergence(perturbed_adamw_models), DCLM_DIV_REF,
        max_eval_tokens=DCLM_DIV_MAX_TOKENS,
    )
    divergence_perturb_muon_evals = _divergence_evals(
        _filter_pert_for_dclm_divergence(perturbed_muon_models), DCLM_DIV_REF,
        max_eval_tokens=DCLM_DIV_MAX_TOKENS,
    )
    divergence_perturb_all_evals = (
        divergence_perturb_adamw_evals + divergence_perturb_muon_evals
    )
else:
    divergence_perturb_bases = ArtifactSet([])
    divergence_perturb_adamw_evals = ArtifactSet([])
    divergence_perturb_muon_evals = ArtifactSet([])
    divergence_perturb_all_evals = ArtifactSet([])


# ---------------------------------------------------------------------------
# Sharpness (Hessian) evaluations — Lanczos max-eig + Hutch++ trace
# Ported from catastrophic-forgetting (evaluate_sharpness + SharpnessEvaluation).
# ---------------------------------------------------------------------------

def _sharpness_evals(models: ArtifactSet) -> ArtifactSet:
    return ArtifactSet([
        SharpnessEvaluation(model=m, eval_dataset="pretrain") for m in models
    ])


sharpness_adamw_evals = _sharpness_evals(pretrain_adamw_wsd)
sharpness_muon_evals = _sharpness_evals(pretrain_muon_wsd)
sharpness_all_evals = sharpness_adamw_evals + sharpness_muon_evals


# --- Hessian top eigenvalue only (Lanczos; no Hutch++ / no full spectrum) ----
# One JSON per (model, checkpoint): raw λ_max(H) + optimizer-transformed
# λ_max(T∘H). Discovers step*/final on GCS (skips step0).
def _maxeig_evals(models: ArtifactSet) -> ArtifactSet:
    arts = []
    for m in models:
        ckpts = list_training_checkpoints(m.relpath, skip_step0=True)
        if not ckpts:
            # Fall back to final so dry-runs without GCS still define an artifact.
            ckpts = ["final"]
            print(
                f"[maxeig] no step*/final under {m.relpath}; "
                f"scheduling checkpoint=final only"
            )
        else:
            print(f"[maxeig] {m.run_name}: {len(ckpts)} checkpoint(s) → {ckpts}")
        for ckpt in ckpts:
            arts.append(
                SharpnessEvaluation(
                    model=m,
                    eval_dataset="pretrain",
                    metrics="max_eigenvalue",
                    checkpoint=ckpt,
                )
            )
    return ArtifactSet(arts)


maxeig_adamw_evals = _maxeig_evals(pretrain_adamw_wsd)
maxeig_muon_evals = _maxeig_evals(pretrain_muon_wsd)
maxeig_all_evals = maxeig_adamw_evals + maxeig_muon_evals


# --- Hessian spectral density via stochastic Lanczos quadrature -------------
# m × n_v HVPs per model (~1k at the defaults). Raw Ritz nodes/weights land in
# the artifact; new_utils.hessian_spectrum derives the statistics offline.

SLQ_LANCZOS_STEPS = 100
SLQ_NUM_PROBES = 10


def _spectrum_evals(
    models: ArtifactSet,
    lanczos_steps: int = SLQ_LANCZOS_STEPS,
    num_probes: int = SLQ_NUM_PROBES,
) -> ArtifactSet:
    return ArtifactSet([
        SharpnessEvaluation(
            model=m,
            eval_dataset="pretrain",
            metrics="spectral_density",
            lanczos_steps=lanczos_steps,
            num_probes=num_probes,
        )
        for m in models
    ])


spectrum_adamw_evals = _spectrum_evals(pretrain_adamw_wsd)
spectrum_muon_evals = _spectrum_evals(pretrain_muon_wsd)
spectrum_all_evals = spectrum_adamw_evals + spectrum_muon_evals

forgetting_sharpness_evals = ArtifactSet([
    ForgettingSharpnessEvaluation(
        pretrained_model=c.pretrained_model,
        cpt_model=c,
    )
    for c in cpt_models
])


# ---------------------------------------------------------------------------
# Associative-facts pretrain: <bos> v[126] r u[126] <eos> + pad → 256,
# CE only on u (label_mask)
#
# Data written by scripts/generate_associative_facts.py →
#   gs://cmu-gpucloud-catheri4/datasets/associative_facts_v3/
# Layout: train/*.npy + train_label_mask/*.npy (flat, cache-safe).
#
# Protocol (AdamW first):
#   1. AdamW LR sweep (PT_LR_SWEEP["adamw"]), cosine schedule, 10 epochs
#      over 6M facts (~1.536B stream tokens / epoch).
#   2. Each example is <bos> v[126] r u[126] <eos> + 1 pad (seq_len=256).
#   3. CE only on u[126] (label_mask); validation uses the held-out val shard.
#   4. Muon follows once AdamW LR is chosen (stage left empty for now).
# Gated on argv containing "associative-facts".
# ---------------------------------------------------------------------------

_WANT_ASSOCIATIVE_FACTS = any("associative-facts" in a for a in sys.argv)

ASSOCIATIVE_FACTS_GS = os.environ.get(
    "ASSOCIATIVE_FACTS_GS",
    "gs://cmu-gpucloud-catheri4/datasets/associative_facts_v3",
)
# Corpus cap: 6M facts (6 × 1M-fact shards on GCS), 10 epochs.
ASSOCIATIVE_FACTS_NUM_FACTS = int(os.environ.get("ASSOCIATIVE_FACTS_NUM_FACTS", "6000000"))
# Facts per shard in generate_associative_facts.py (--shard-facts 1e6).
ASSOCIATIVE_FACTS_SHARD_FACTS = int(os.environ.get("ASSOCIATIVE_FACTS_SHARD_FACTS", "1000000"))
ASSOCIATIVE_FACTS_ENTITY_LEN = 126
ASSOCIATIVE_FACTS_FACT_LEN = 256  # bos + v + r + u + eos + pad
_default_shards = math.ceil(ASSOCIATIVE_FACTS_NUM_FACTS / ASSOCIATIVE_FACTS_SHARD_FACTS)
ASSOCIATIVE_FACTS_TRAIN_SHARDS = int(
    os.environ.get("ASSOCIATIVE_FACTS_TRAIN_SHARDS", str(_default_shards))
)
ASSOCIATIVE_FACTS_EPOCHS = int(os.environ.get("ASSOCIATIVE_FACTS_EPOCHS", "10"))
ASSOCIATIVE_FACTS_SCHEDULER = os.environ.get("ASSOCIATIVE_FACTS_SCHEDULER", "cosine")


def _associative_facts_sequence_length() -> int:
    """Validate the fixed 256-token example length against microbatch sizes."""
    seq_len = int(
        os.environ.get("ASSOCIATIVE_FACTS_SEQUENCE_LENGTH", str(ASSOCIATIVE_FACTS_FACT_LEN))
    )
    if seq_len != ASSOCIATIVE_FACTS_FACT_LEN:
        raise ValueError(
            f"ASSOCIATIVE_FACTS_SEQUENCE_LENGTH must equal the on-disk fact length "
            f"{ASSOCIATIVE_FACTS_FACT_LEN}, got {seq_len}"
        )
    if RANK_MICROBATCH_SIZE % seq_len != 0:
        raise ValueError(
            f"ASSOCIATIVE_FACTS_SEQUENCE_LENGTH={seq_len} must divide "
            f"rank_microbatch_size={RANK_MICROBATCH_SIZE}"
        )
    if EVAL_RANK_MICROBATCH_SIZE % seq_len != 0:
        raise ValueError(
            f"ASSOCIATIVE_FACTS_SEQUENCE_LENGTH={seq_len} must divide "
            f"eval_rank_microbatch_size={EVAL_RANK_MICROBATCH_SIZE}"
        )
    return seq_len


# Each on-disk fact is exactly one unpadded training sequence.
ASSOCIATIVE_FACTS_SEQUENCE_LENGTH = _associative_facts_sequence_length()
# One epoch = one pass over the fixed-length fact corpus.
ASSOCIATIVE_FACTS_TOKENS_PER_EPOCH = int(
    os.environ.get(
        "ASSOCIATIVE_FACTS_TOKENS_PER_EPOCH",
        str(ASSOCIATIVE_FACTS_NUM_FACTS * ASSOCIATIVE_FACTS_SEQUENCE_LENGTH),
    )
)


def _associative_facts_n_tokens() -> int:
    """N epochs over the capped fact corpus, snapped to a multiple of global batch."""
    raw = ASSOCIATIVE_FACTS_TOKENS_PER_EPOCH * ASSOCIATIVE_FACTS_EPOCHS
    return (raw // GLOBAL_BATCH_SIZE) * GLOBAL_BATCH_SIZE


def _associative_facts_train_chunks(n_shards: int):
    return tuple(
        Chunk(uri=f"{ASSOCIATIVE_FACTS_GS}/train/{i:05d}.npy")
        for i in range(n_shards)
    )


def _associative_facts_label_mask_chunks(n_shards: int):
    return tuple(
        Chunk(uri=f"{ASSOCIATIVE_FACTS_GS}/train_label_mask/{i:05d}.npy")
        for i in range(n_shards)
    )


def _associative_facts_validation_chunks(
) -> Tuple[Tuple[Tuple[str, Chunk], ...], Tuple[Tuple[str, Chunk], ...]]:
    """Use the small held-out validation shard, never the full training corpus."""
    label = "associative_facts"
    return (
        ((label, Chunk(uri=f"{ASSOCIATIVE_FACTS_GS}/val/00000.npy")),),
        ((label, Chunk(uri=f"{ASSOCIATIVE_FACTS_GS}/val_label_mask/00000.npy")),),
    )


def _make_associative_facts_adamw_sweep() -> ArtifactSet:
    """AdamW LR sweep on associative-facts, cosine, N epochs over all facts."""
    n_tokens = _associative_facts_n_tokens()
    total_steps = n_tokens // GLOBAL_BATCH_SIZE
    warmup_steps = max(1, total_steps // 10)  # 10% warmup, same as _tokens_for
    train_chunks = _associative_facts_train_chunks(ASSOCIATIVE_FACTS_TRAIN_SHARDS)
    mask_chunks = _associative_facts_label_mask_chunks(ASSOCIATIVE_FACTS_TRAIN_SHARDS)
    val_chunks, val_mask_chunks = _associative_facts_validation_chunks()
    sched = ASSOCIATIVE_FACTS_SCHEDULER
    common = {
        **SHARED_MODEL_PARAMS,
        "n_tokens": n_tokens,
        "warmup_steps": warmup_steps,
        "scheduler": sched,  # overrides SHARED_MODEL_PARAMS["scheduler"]
        "sequence_length": ASSOCIATIVE_FACTS_SEQUENCE_LENGTH,
        "padded_dataset": False,
        "train_chunks": train_chunks,
        "label_mask_chunks": mask_chunks,
        "validation_chunks": val_chunks,
        "validation_label_mask_chunks": val_mask_chunks,
        "experiment_name": f"{PROJECT_NAME}-associative-facts",
    }
    models = []
    lr_filter = os.environ.get("ASSOCIATIVE_FACTS_ADAMW_LRS")
    adamw_lrs = list(PT_LR_SWEEP.get("adamw", []))
    if lr_filter:
        wanted = {float(x) for x in lr_filter.split(",") if x.strip()}
        adamw_lrs = [lr for lr in adamw_lrs if lr in wanted]
    for lr in adamw_lrs:
        run = (
            f"AssocFacts-{MODEL_TYPE}-adamw-lr{_lr_tag(lr)}-{sched}-"
            f"ep{ASSOCIATIVE_FACTS_EPOCHS}-facts{ASSOCIATIVE_FACTS_NUM_FACTS // 1_000_000}M-"
            f"tok{n_tokens // 1_000_000}M"
        )
        models.append(
            JolmoModel(
                model_name=run,
                **common,
                optimizer="adamw",
                learning_rate=lr,
            )
        )
    return ArtifactSet(models)


if _WANT_ASSOCIATIVE_FACTS:
    # Phase 1: AdamW LR tune only. Muon stages stay empty until best AdamW LR
    # is chosen (then fill via a follow-up).
    pretrain_associative_facts_adamw = (
        _make_associative_facts_adamw_sweep() if "adamw" in OPTIMIZERS else ArtifactSet([])
    )
    pretrain_associative_facts_muon = ArtifactSet([])
    pretrain_associative_facts_all = pretrain_associative_facts_adamw
else:
    pretrain_associative_facts_adamw = ArtifactSet([])
    pretrain_associative_facts_muon = ArtifactSet([])
    pretrain_associative_facts_all = ArtifactSet([])

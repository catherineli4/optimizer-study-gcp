"""`LrBatchSweep` — LR × batch-size cross product for one (size, optimizer, chinchilla) cell.

Builds the full cross product of  main LR × global-batch multiplier  as
JolmoModel artifacts, plus matching ModelEvaluations. See docs/lr_bs_sweep.md
for the full spec.

Semantics:
  - adamw sweeps ``learning_rate``; muon sweeps ``muon_lr`` with the adamw
    component PINNED to the tuned cell ``PT_LR_BY_MODEL[size][sched]["adamw"][chin]``
    (a muon sweep with no tuned adamw cell builds zero models — it never invents
    a component).
  - LRs are defined at the reference batch (GLOBAL_BATCH_SIZE); a cell at
    multiplier ``a`` trains at ``sqrt(a) × lr``. For muon both legs scale by
    default (``scale_adamw_component=False`` holds the pinned component fixed).
  - The token budget is FIXED across the batch axis; only the step count moves.

Naming must keep matching what is already on GCS: ``DEFAULT_NAME_PREFIX`` is
``PTSweep{$OPTIM_SIZE}`` (``PTSweep60M`` at 60M), byte-identical to the names
``pt_sweep_60m_chin4.py`` already wrote — existence is keyed on
``JolmoModel/<model_name>/final-unsharded/model.pt``, so a different prefix
silently retrains every finished cell.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from experiments import ArtifactSet

from launch_jolmo.sizes import active_profile
from launch_jolmo.training import JolmoModel, ModelEvaluation, MODEL_ARCHS
from launch_jolmo.pretraining_matrix import (
    TOKENIZER,
    GLOBAL_BATCH_SIZE,
    SEQUENCE_LENGTH,
    PT_LR_BY_MODEL,
    DCLM_HELDOUT_INSTANCES,
    _base_tokens_for,
    _dclm_chunks_for_tokens,
    _lr_tag,
    diversity_val_chunks,
    dclm_heldout_val_chunks,
)

_SIZE, _PROFILE = active_profile()

# Must keep matching what pt_sweep_60m_chin4.py already wrote to GCS (see module
# docstring). Override per instance with ``name_prefix=`` only when a separate
# tree is wanted.
DEFAULT_NAME_PREFIX = f"PTSweep{_SIZE}"

SWEEP_LRS: Tuple[float, ...] = (7e-3, 1e-2, 1.4e-2, 2e-2, 2.8e-2, 4e-2, 5.6e-2, 8e-2)
SWEEP_BS_MULTIPLIERS: Tuple[int, ...] = (1, 2, 4)   # × GLOBAL_BATCH_SIZE


def _wd_tag(wd: float) -> str:
    """0.1 -> 'wd0.1', 0.0 -> 'wd0' (same format as pt_sweep_60m_chin4)."""
    return "wd" + f"{wd:g}"


def _bs_tag(gbs: int) -> str:
    """1048576 -> 'bs1M', 524288 -> 'bs512k' (same format as pt_sweep_60m_chin4)."""
    if gbs % (1024 * 1024) == 0:
        return f"bs{gbs // (1024 * 1024)}M"
    if gbs % 1024 == 0:
        return f"bs{gbs // 1024}k"
    return f"bs{gbs}"


def _chin_tag(chinchilla: float) -> str:
    """4 and 4.0 -> '4'; 0.25 -> '0.25'."""
    return f"{chinchilla:g}"


# models() results keyed by the sweep instance (frozen dataclass => hashable,
# equal instances hit the same entry). LOAD-BEARING, not an optimization: the
# executor resolves an evaluation's dependency on its base model by object
# identity, and evals() calls models() — without this cache the model and eval
# stages would hold distinct-but-equal JolmoModel instances and selecting both
# stages fails the "all dependencies must be explicitly included" check.
_MODELS_CACHE: Dict["LrBatchSweep", ArtifactSet] = {}


@dataclass(frozen=True)
class LrBatchSweep:
    """Full LR × batch-multiplier grid for one (model size, optimizer, chinchilla)."""

    model_type: str                     # key into training.MODEL_ARCHS
    optimizer: str                      # "adamw" | "muon"
    chinchilla: float                   # may be fractional (0.25, 0.5)
    lrs: Tuple[float, ...] = SWEEP_LRS
    bs_multipliers: Tuple[int, ...] = SWEEP_BS_MULTIPLIERS
    weight_decay: float = 0.1
    scheduler: str = "wsd"
    num_processes: Optional[int] = None          # default: the profile's value
    scale_adamw_component: bool = True
    validation_eval_interval: int = 95
    name_prefix: Optional[str] = None            # default: DEFAULT_NAME_PREFIX

    def __post_init__(self):
        if self.optimizer not in ("adamw", "muon"):
            raise ValueError(f"optimizer must be 'adamw' or 'muon', got {self.optimizer!r}")
        if self.model_type not in MODEL_ARCHS:
            raise ValueError(f"unknown model_type {self.model_type!r}")

    # ------------------------------------------------------------------ props

    @property
    def label(self) -> str:
        return f"lrbs-{self.model_type}-c{_chin_tag(self.chinchilla)}-{self.optimizer}"

    @property
    def size_ok(self) -> bool:
        """The bucket is chosen by $OPTIM_SIZE; refuse to build models for a
        different size so nothing lands in another study's prefix."""
        return _PROFILE["model_type"] == self.model_type

    @property
    def n_tokens(self) -> int:
        return int(_base_tokens_for(self.model_type) * self.chinchilla)

    @property
    def gpus(self) -> int:
        if self.num_processes is not None:
            return self.num_processes
        return int(os.environ.get(
            "OPTIM_NUM_PROCESSES", str(_PROFILE.get("num_processes", 2))))

    @property
    def tuned_adamw_lr(self) -> Optional[float]:
        """The tuned adamw LR the muon sweep pins its component to (None if absent)."""
        cell = (
            PT_LR_BY_MODEL.get(self.model_type, {})
            .get(self.scheduler, {})
            .get("adamw", {})
            .get(self.chinchilla)
        )
        return cell if isinstance(cell, (int, float)) else None

    # ---------------------------------------------------------------- helpers

    @property
    def _prefix(self) -> str:
        return self.name_prefix if self.name_prefix is not None else DEFAULT_NAME_PREFIX

    def _schedule_for(self, gbs: int) -> Dict[str, Any]:
        """Token budget FIXED across the batch axis; only the step count moves."""
        total_steps = self.n_tokens // gbs
        return {"n_tokens": self.n_tokens, "warmup_steps": total_steps // 10}

    def _shared_params(self, gbs: int) -> Dict[str, Any]:
        """Mirrors pt_sweep_60m_chin4._shared_params."""
        gpus = self.gpus
        return {
            "model_type": self.model_type,
            "tokenizer": TOKENIZER,
            "sequence_length": SEQUENCE_LENGTH,
            # Optimizer
            "weight_decay": self.weight_decay,
            "betas": (0.9, 0.98),
            "max_grad_norm": 1.0,
            # Schedule
            "scheduler": self.scheduler,
            "global_batch_size": gbs,
            # Derived from the batch size but CAPPED at 64K tokens: the LM-head
            # logits scale with microbatch tokens (262K tokens -> ~49 GiB in
            # bf16, OOM on 80GB) — the cap trades gradient-accumulation steps
            # for memory and keeps the global batch (and the math) identical.
            "rank_microbatch_size": max(SEQUENCE_LENGTH, min(gbs // gpus // 8, 65_536)),
            "eval_rank_microbatch_size": max(SEQUENCE_LENGTH, min(gbs // gpus // 64, 65_536)),
            "validation_eval_interval": self.validation_eval_interval,
            # Parallelism & compilation
            "compile_model": True,
            "parallelism": "ddp",
            "num_processes": gpus,
            "dp_param_dtype": "bfloat16",
            "dp_reduce_dtype": "float32",
            # Data
            "train_chunks": _dclm_chunks_for_tokens(self.n_tokens),
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

    def _cell_name(self, lr_tag: str, gbs: int) -> str:
        return (
            f"{self._prefix}-{self.model_type}-chinchilla-{_chin_tag(self.chinchilla)}"
            f"-{self.optimizer}-{lr_tag}-{_wd_tag(self.weight_decay)}"
            f"-{_bs_tag(gbs)}-{self.scheduler}"
        )

    # ----------------------------------------------------------------- builds

    def models(self) -> ArtifactSet:
        cached = _MODELS_CACHE.get(self)
        if cached is not None:
            return cached

        if not self.size_ok:
            print(f"[{self.label}] 0 models — OPTIM_SIZE resolves to "
                  f"{_PROFILE['model_type']}, not {self.model_type}")
            result = ArtifactSet([])
            _MODELS_CACHE[self] = result
            return result

        pinned = self.tuned_adamw_lr
        if self.optimizer == "muon" and pinned is None:
            print(f"[{self.label}] 0 models — no tuned adamw LR at "
                  f"PT_LR_BY_MODEL[{self.model_type!r}][{self.scheduler!r}]"
                  f"['adamw'][{self.chinchilla!r}] to pin the muon component to")
            result = ArtifactSet([])
            _MODELS_CACHE[self] = result
            return result

        models = []
        seen_names: Dict[str, Tuple[float, int]] = {}
        for mult in self.bs_multipliers:
            gbs = GLOBAL_BATCH_SIZE * mult
            if self.n_tokens % gbs != 0:
                lost = self.n_tokens % gbs
                print(f"[{self.label}] WARNING: multiplier {mult} does not divide "
                      f"n_tokens={self.n_tokens:,}; that cell trains on a budget "
                      f"truncated by {lost:,} tokens")
            scale = math.sqrt(mult)
            for lr in self.lrs:
                cell_lr = lr * scale
                if self.optimizer == "adamw":
                    lr_tag = f"lr{_lr_tag(cell_lr)}"
                    extra = {"optimizer": "adamw", "learning_rate": cell_lr}
                else:
                    comp = pinned * scale if self.scale_adamw_component else pinned
                    lr_tag = f"muonlr{_lr_tag(cell_lr)}-adamwlr{_lr_tag(comp)}"
                    extra = {"optimizer": "muon", "muon_lr": cell_lr,
                             "learning_rate": comp}
                name = self._cell_name(lr_tag, gbs)
                if name in seen_names:
                    other = seen_names[name]
                    raise ValueError(
                        f"[{self.label}] run-name collision: {name!r} produced by both "
                        f"(lr={other[0]}, mult={other[1]}) and (lr={lr}, mult={mult}) — "
                        f"LRs round together under _lr_tag; thin the lrs list")
                seen_names[name] = (lr, mult)
                models.append(JolmoModel(
                    model_name=name,
                    **self._shared_params(gbs),
                    **self._schedule_for(gbs),
                    **extra,
                ))

        result = ArtifactSet(models)
        _MODELS_CACHE[self] = result
        return result

    def evals(self) -> ArtifactSet:
        """One ModelEvaluation per cell (diversity val sets + held-out DCLM)."""
        return ArtifactSet([
            ModelEvaluation(
                model=m,
                extra_val_chunks=dclm_heldout_val_chunks,
                extra_val_max_instances=DCLM_HELDOUT_INSTANCES,
            )
            for m in self.models()
        ])


# ---------------------------------------------------------------------------
# Declared sweeps — the launcher turns each entry into two stages:
#   <label> (models) and <label>-evals.
# ---------------------------------------------------------------------------

LRBS_60M_CHINCHILLAS: Tuple[float, ...] = (0.25, 0.5, 1, 2, 4, 8)

SWEEPS: Tuple[LrBatchSweep, ...] = tuple(
    LrBatchSweep(model_type="0.06B", optimizer=opt, chinchilla=c)
    for c in LRBS_60M_CHINCHILLAS
    for opt in ("adamw", "muon")
)

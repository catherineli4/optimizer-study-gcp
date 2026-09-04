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
import sys
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
    _existing_jolmo_runs,
    _lr_tag,
    diversity_val_chunks,
    dclm_heldout_val_chunks,
)

# A reference-batch cell (wd0.1, bs x1, wsd) is the SAME training configuration
# as the older MuonExpt3-named pretrain — one model, two naming schemas. When a
# cell's MuonExpt3 twin already exists on GCS, name the cell after the twin so
# the existence check skips it instead of retraining a duplicate under the
# PTSweep name. Gated on argv (the cpt-all pattern) so only lrbs/ft commands
# pay for the one cached GCS listing at import.
_ALIAS_EXISTING = any(("lrbs" in a) or a.startswith("ft-") for a in sys.argv)

_SIZE, _PROFILE = active_profile()

# Must keep matching what pt_sweep_60m_chin4.py already wrote to GCS (see module
# docstring). Override per instance with ``name_prefix=`` only when a separate
# tree is wanted.
DEFAULT_NAME_PREFIX = f"PTSweep{_SIZE}"

SWEEP_LRS: Tuple[float, ...] = (7e-3, 1e-2, 1.4e-2, 2e-2, 2.8e-2, 4e-2, 5.6e-2, 8e-2, 1.2e-1, 1.6e-1,)
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
    # Explicit muon adamw-component pin for chinchillas that have no
    # PT_LR_BY_MODEL entry (e.g. the historical 0.25 sweep). Overrides the table.
    pinned_adamw_lr: Optional[float] = None
    # Reuse an existing MuonExpt3-named twin instead of training a PTSweep cell.
    # Set False when the twin was trained under a DIFFERENT recipe (e.g. the
    # imported 100M jgai runs), so this sweep trains its own comparable cell.
    alias_existing: bool = True
    # Distinguishes subset stages (e.g. "-hi5") from the full grid's stage name.
    # Cell RUN NAMES are unaffected — overlapping cells still dedupe on GCS.
    label_suffix: str = ""

    def __post_init__(self):
        if self.optimizer not in ("adamw", "muon"):
            raise ValueError(f"optimizer must be 'adamw' or 'muon', got {self.optimizer!r}")
        if self.model_type not in MODEL_ARCHS:
            raise ValueError(f"unknown model_type {self.model_type!r}")

    # ------------------------------------------------------------------ props

    @property
    def label(self) -> str:
        return (f"lrbs-{self.model_type}-c{_chin_tag(self.chinchilla)}"
                f"-{self.optimizer}{self.label_suffix}")

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
        if self.pinned_adamw_lr is not None:
            return self.pinned_adamw_lr
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
                if (self.alias_existing
                        and _ALIAS_EXISTING and mult == 1
                        and self.weight_decay == 0.1
                        and self.scheduler == "wsd"
                        and self._prefix == DEFAULT_NAME_PREFIX):
                    twin = (f"MuonExpt3-{self.model_type}"
                            f"-chinchilla-{_chin_tag(self.chinchilla)}"
                            f"-{self.optimizer}-{lr_tag}-wsd")
                    if twin in _existing_jolmo_runs():
                        name = twin
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

# The chinchilla-0.25 sweep already on GCS was trained with this grid (one step
# above SWEEP_LRS) and its muon component pinned to adamw 5.6e-2 — both must
# stay byte-identical to keep matching the existing 48 PTSweep60M-...-0.25-*
# artifact names.
LRBS_60M_C025_LRS: Tuple[float, ...] = (7e-3, 1e-2, 1.4e-2, 2e-2, 2.8e-2, 4e-2, 5.6e-2, 8e-2)
LRBS_60M_C025_ADAMW_PIN: float = 5.6e-2

# c0.25 first: downstream consumers (ft_sweep) preserve this order.
SWEEPS: Tuple[LrBatchSweep, ...] = tuple(
    LrBatchSweep(model_type="0.06B", optimizer=opt, chinchilla=0.25,
                 lrs=LRBS_60M_C025_LRS, pinned_adamw_lr=LRBS_60M_C025_ADAMW_PIN)
    for opt in ("adamw", "muon")
) + tuple(
    LrBatchSweep(model_type="0.06B", optimizer=opt, chinchilla=c)
    for c in LRBS_60M_CHINCHILLAS
    for opt in ("adamw", "muon")
)


# ---------------------------------------------------------------------------
# Above-optimal subset: bs multiplier 1 ONLY, and only the K next-highest LRs
# above the tuned optimal for that (size, optimizer, chinchilla). If fewer than
# K of SWEEP_LRS sit above the optimal, takes all that do. Stage label gets a
# "-hi<K>" suffix; cell run names are the standard ones, so cells that the full
# grid already trained resolve to the SAME artifacts and are skipped.
# ---------------------------------------------------------------------------

def _tuned_main_lr(model_type: str, scheduler: str, optimizer: str, chinchilla: float):
    """The tuned MAIN lr for the cell: adamw -> learning_rate, muon -> muon_lr."""
    cell = (
        PT_LR_BY_MODEL.get(model_type, {})
        .get(scheduler, {})
        .get(optimizer, {})
        .get(chinchilla)
    )
    if optimizer == "adamw":
        return cell if isinstance(cell, (int, float)) else None
    return cell[0] if isinstance(cell, tuple) else None


def above_optimal_sweep(
    model_type: str,
    optimizer: str,
    chinchilla: float,
    k: int = 5,
    lr_pool: Tuple[float, ...] = SWEEP_LRS,
    **kwargs,
) -> Optional[LrBatchSweep]:
    """LrBatchSweep over the K next LRs above the tuned optimal, at bs x1 only.

    Returns None (with a message) when the cell has no tuned optimal or no pool
    LR sits above it — it never invents a starting point.
    """
    label = f"lrbs-{model_type}-c{_chin_tag(chinchilla)}-{optimizer}-hi{k}"
    opt_lr = _tuned_main_lr(model_type, kwargs.get("scheduler", "wsd"), optimizer, chinchilla)
    if opt_lr is None:
        print(f"[{label}] 0 models — no tuned {optimizer} LR at chinchilla {chinchilla:g}")
        return None
    higher = tuple(sorted(lr for lr in lr_pool if lr > opt_lr)[:k])
    if not higher:
        print(f"[{label}] 0 models — no pool LR above the tuned optimal {opt_lr:g}")
        return None
    return LrBatchSweep(
        model_type=model_type, optimizer=optimizer, chinchilla=chinchilla,
        lrs=higher, bs_multipliers=(1,), label_suffix=f"-hi{k}", **kwargs,
    )


# Declared: one above-optimal subset per 60M (chinchilla x optimizer) with a
# tuned cell. Stages: lrbs-0.06B-c<chin>-<opt>-hi5 (+ -evals via the launcher).
# 16 is included beyond LRBS_60M_CHINCHILLAS: the tuned table has a c16 cell,
# so an above-optimal subset is well defined there even though the full LR x BS
# grid was never run at that budget.
HI5_60M_CHINCHILLAS: Tuple[float, ...] = tuple(
    dict.fromkeys(tuple(LRBS_60M_CHINCHILLAS) + (16,)))

HI5_SWEEPS: Tuple[LrBatchSweep, ...] = tuple(
    s for s in (
        above_optimal_sweep("0.06B", opt, c)
        for c in HI5_60M_CHINCHILLAS
        for opt in ("adamw", "muon")
    )
    if s is not None
)

# 100M above-optimal subsets — chinchillas 2 and 4 only. Stages:
# lrbs-0.1B-c<chin>-<opt>-hi5 (+ -evals), umbrella lrbs-hi5-100m in the
# launcher. Under a non-100M $OPTIM_SIZE these build 0 models (size_ok guard).
HI5_100M_CHINCHILLAS: Tuple[float, ...] = (2, 4)
HI5_100M_SWEEPS: Tuple[LrBatchSweep, ...] = tuple(
    s for s in (
        above_optimal_sweep("0.1B", opt, c)
        for c in HI5_100M_CHINCHILLAS
        for opt in ("adamw", "muon")
    )
    if s is not None
)

# Muon LR sweep at the reference batch (bs x1) for chinchillas 1-8. The adamw
# component is pinned, per cell, to PT_LR_BY_MODEL[...]["adamw"][chin] — i.e.
# the RETUNED adamw optima (c1 2e-2, c2 1e-2, c4 1e-2, c8 1.4e-2) — so the only
# axis swept is muon_lr. Grid matches the existing c2 bs1M cells so those are
# skipped rather than retrained.
MUON_C18_LRS: Tuple[float, ...] = (7e-3, 1e-2, 1.4e-2, 2e-2)
MUON_C18_SWEEPS: Tuple[LrBatchSweep, ...] = tuple(
    LrBatchSweep(model_type="0.06B", optimizer="muon", chinchilla=c,
                 lrs=MUON_C18_LRS, bs_multipliers=(1,), label_suffix="-bs1m")
    for c in (1, 2, 4, 8)
)

# 30M muon LR sweep at chinchillas 1-2, reference batch, with the adamw
# component pinned to PT_LR_BY_MODEL's tuned adamw (c1 2.8e-2, c2 4e-2).
MUON_30M_C12_LRS: Tuple[float, ...] = (1e-2, 1.4e-2, 2e-2)
MUON_30M_C12_SWEEPS: Tuple[LrBatchSweep, ...] = tuple(
    LrBatchSweep(model_type="0.03B", optimizer="muon", chinchilla=c,
                 lrs=MUON_30M_C12_LRS, bs_multipliers=(1,),
                 label_suffix="-bs1m")
    for c in (1, 2)
)

# c16/c32 muon sweep at the reference batch with the adamw component pinned to
# the (new) tuned 1e-2 — matches the 10 PTSweep60M-…-adamwlr1.0e-2-…bs1M cells
# already on GCS. Declared here (chinchillas outside LRBS_60M_CHINCHILLAS) so
# their evals have stages. label_suffix keeps the stage names distinct.
LRBS_C1632_MUON_SWEEPS: Tuple[LrBatchSweep, ...] = tuple(
    LrBatchSweep(model_type="0.06B", optimizer="muon", chinchilla=c,
                 lrs=(5e-3, 7e-3, 1e-2, 1.4e-2, 2e-2), bs_multipliers=(1,),
                 pinned_adamw_lr=1e-2, label_suffix="-a1e-2")
    for c in (16, 32)
)

# 600M above-optimal subsets — chinchillas 0.5 and 1. Stages:
# lrbs-0.6B-c<chin>-<opt>-hi5 (+ -evals), umbrella lrbs-hi5-600m in the
# launcher. Under a non-600M $OPTIM_SIZE these build 0 models (size_ok guard).
HI5_600M_CHINCHILLAS: Tuple[float, ...] = (0.5, 1)
HI5_600M_SWEEPS: Tuple[LrBatchSweep, ...] = tuple(
    s for s in (
        above_optimal_sweep("0.6B", opt, c)
        for c in HI5_600M_CHINCHILLAS
        for opt in ("adamw", "muon")
    )
    if s is not None
)

# 300M above-optimal subsets — chinchillas 2 and 4. Stages:
# lrbs-0.3B-c<chin>-<opt>-hi5 (+ -evals), umbrella lrbs-hi5-300m in the
# launcher. num_processes pinned to 8: one run gets all 8 GPUs (and an
# OPTIM_NUM_PROCESSES override cannot shrink it).
HI5_300M_CHINCHILLAS: Tuple[float, ...] = (2, 4)
HI5_300M_SWEEPS: Tuple[LrBatchSweep, ...] = tuple(
    s for s in (
        above_optimal_sweep("0.3B", opt, c, num_processes=8)
        for c in HI5_300M_CHINCHILLAS
        for opt in ("adamw", "muon")
    )
    if s is not None
)

"""`FtSweep` — full finetuning cross product for ONE pretrained base model.

Analog of `lr_bs_sweep.LrBatchSweep` for the CPT/finetuning phase: takes a single
pretrained `JolmoModel` and builds every combination of

    dataset  ×  FT optimizer  ×  FT LR  ×  DCLM replay fraction

as `CPTModel` artifacts, plus matching `ModelEvaluation`s.

Defaults (r=0 only): adamw-pretrained base -> 6 datasets × 8 adamw LRs = 48;
        muon-pretrained base -> 6 × (8 muon + 8 adamw LRs) = **96 finetunes**.

Semantics:
  - FT LRs come from the live `cpt.CPT_LR_SWEEP` ("adamw" as-is; "muon" via
    `build_muon_lr_sweep(0.25)` so the adamw component = 0.25 × muon_lr — the
    same convention as the `cpt` / `cpt-muon` stages).
  - Replay fraction r > 0 mixes r of the token budget from DCLM pretraining data
    via a two-source `SourceMixtureDatasetConfig` (see `CPTModel.replay_dclm`);
    r = 0 is plain finetuning and — deliberately — leaves the run name unchanged
    so every existing CPT artifact on GCS keeps matching.
  - The replay source is one fixed DCLM training shard (part-000/00000.npy,
    ~4.3B tokens — plenty for ≤ 0.3 × 20M replay tokens), downloaded once into
    the shared dataset cache and reused by every task. It is a TRAINING shard;
    the held-out eval shard (part-059/00004.npy) is never trained on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Tuple

from experiments import ArtifactSet

from launch_jolmo.data import Chunk
from launch_jolmo.training import CPTModel, JolmoModel, ModelEvaluation
from launch_jolmo.cpt import (
    CPT_LR_SWEEP,
    CPT_TOKENS,
    CPT_SCHEDULER,
    CPT_SEQUENCE_LENGTH,
    CPT_GLOBAL_BATCH_SIZE,
    CPT_RANK_MICROBATCH_SIZE,
    build_muon_lr_sweep,
)
from launch_jolmo.pretraining_matrix import (
    dclm_heldout_val_chunks,
    DCLM_HELDOUT_INSTANCES,
)

FT_DATASETS: Tuple[str, ...] = (
    "starcoder", "musicpile", "alpaca", "gsm8k", "stackmathqa",
)
# Replay axis removed from the default sweep: plain finetuning only (r=0, no
# name suffix). The replay machinery (CPTModel.replay_dclm + REPLAY_DCLM_CHUNK)
# stays available — pass replay_fractions=(0.0, 0.1, ...) per FtSweep to re-arm.
FT_REPLAY_FRACTIONS: Tuple[float, ...] = (0.0,)

# Fixed DCLM replay source: one full shard (~16 GiB, ~4.3B tokens), cached once
# per node. Deliberately UNSEEN data: pretraining consumes parts from the front
# (part-000 upward; the largest budget, 60M chinchilla-128, reaches ~13 parts),
# so part-058 is beyond every run's training data — same distribution, never
# trained on. It is also distinct from the held-out EVAL shard (part-059/00004),
# so replay training never contaminates the DCLM_heldout metric.
REPLAY_DCLM_CHUNK = Chunk(
    uri=(
        "gs://cmu-gpucloud-jspringe/shared/datasets/OLMo/dclm/train/"
        "preprocessed_dclm_text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_"
        "train_allenai_dolma2-tokenizer_part-058/00000.npy"
    ),
)

# --- Existence-check bypass --------------------------------------------------
# The ft-lrbs sweeps enumerate ~100K artifacts (hundreds of bases x 384
# finetunes + as many evals); the per-artifact GCS pre-flight is pure overhead
# on a fresh sweep. FtSweep artifacts therefore skip the existence stage
# ENTIRELY by default: should_skip() returns False with zero network calls, so
# every enumerated cell is scheduled unconditionally.
#
# CAUTION: this also means an interrupted ft- sweep does NOT resume — a relaunch
# re-trains finished cells. Export OPTIM_FT_EXISTS_CHECK=1 to restore the normal
# skip-what-exists behavior (do that for any rerun/resume of a partial sweep).
import os as _os

_FT_EXISTS_CHECK = _os.environ.get("OPTIM_FT_EXISTS_CHECK", "0") == "1"


class _NoExistsCheck:
    def should_skip(self) -> bool:  # overrides Artifact.should_skip
        if _FT_EXISTS_CHECK:
            return super().should_skip()
        return False


class FtCPTModel(_NoExistsCheck, CPTModel):
    """CPTModel whose existence pre-flight is bypassed (see above)."""


class FtModelEvaluation(_NoExistsCheck, ModelEvaluation):
    """ModelEvaluation whose existence pre-flight is bypassed (see above)."""


# models() memoised per parameter key — load-bearing, same as LrBatchSweep:
# evals() depend on their base CPTModels by object identity.
_MODELS_CACHE: Dict[tuple, ArtifactSet] = {}


@dataclass(frozen=True, eq=False)
class FtSweep:
    """All (dataset × optimizer × LR × replay) finetunes of one pretrained base."""

    base: JolmoModel
    datasets: Tuple[str, ...] = FT_DATASETS
    optimizers: Tuple[str, ...] = ("adamw", "muon")
    replay_fractions: Tuple[float, ...] = FT_REPLAY_FRACTIONS
    muon_adamw_multiplier: float = 0.25
    train_tokens: int = CPT_TOKENS
    label_suffix: str = ""

    def __post_init__(self):
        for opt in self.optimizers:
            if opt not in ("adamw", "muon"):
                raise ValueError(f"optimizers must be 'adamw'/'muon', got {opt!r}")
        for r in self.replay_fractions:
            if not 0 <= r < 1:
                raise ValueError(f"replay fraction must be in [0, 1), got {r}")

    @property
    def label(self) -> str:
        return f"ft-{self.base.run_name}{self.label_suffix}"

    def _key(self) -> tuple:
        return (
            self.base.run_name, self.datasets, self.optimizers,
            self.replay_fractions, self.muon_adamw_multiplier, self.train_tokens,
        )

    def _lr_entries(self, opt: str) -> list:
        if opt == "adamw":
            return list(CPT_LR_SWEEP["adamw"])
        return list(build_muon_lr_sweep(self.muon_adamw_multiplier)["muon"])

    def models(self) -> ArtifactSet:
        key = self._key()
        cached = _MODELS_CACHE.get(key)
        if cached is not None:
            return cached

        # Optimizer pairing follows the main `cpt` convention: adamw-pretrained
        # bases finetune with adamw ONLY; muon-pretrained bases finetune with
        # both muon and adamw (whatever subset of self.optimizers applies).
        opts = self.optimizers
        if getattr(self.base, "optimizer", "adamw") == "adamw":
            opts = tuple(o for o in opts if o == "adamw")

        models = []
        seen = set()
        for ds in self.datasets:
            for opt in opts:
                for lr_entry in self._lr_entries(opt):
                    if opt == "muon":
                        muon_lr, adamw_lr = lr_entry
                    else:
                        muon_lr, adamw_lr = None, lr_entry
                    for r in self.replay_fractions:
                        m = FtCPTModel(
                            pretrained_model=self.base,
                            cpt_dataset=ds,
                            train_tokens=self.train_tokens,
                            optimizer=opt,
                            learning_rate=adamw_lr,
                            muon_lr=muon_lr,
                            scheduler=CPT_SCHEDULER,
                            sequence_length=CPT_SEQUENCE_LENGTH,
                            global_batch_size=CPT_GLOBAL_BATCH_SIZE,
                            rank_microbatch_size=CPT_RANK_MICROBATCH_SIZE,
                            replay_dclm=r,
                            replay_chunks=(REPLAY_DCLM_CHUNK,) if r > 0 else (),
                        )
                        if m.run_name in seen:
                            raise ValueError(
                                f"[{self.label}] run-name collision: {m.run_name!r}"
                            )
                        seen.add(m.run_name)
                        models.append(m)

        result = ArtifactSet(models)
        _MODELS_CACHE[key] = result
        return result

    def evals(self) -> ArtifactSet:
        """One ModelEvaluation per finetune (CPT val sets + held-out DCLM)."""
        return ArtifactSet([
            FtModelEvaluation(
                model=m,
                extra_val_chunks=dclm_heldout_val_chunks,
                extra_val_max_instances=DCLM_HELDOUT_INSTANCES,
            )
            for m in self.models()
        ])


# ---------------------------------------------------------------------------
# Declared sweeps — the launcher turns each entry into two stages:
#   <label> (finetunes) and <label>-evals.
# Append FtSweep(base=<some JolmoModel>) entries here.
# ---------------------------------------------------------------------------

import sys
from dataclasses import replace as _dc_replace

from launch_jolmo.lr_bs_sweep import (SWEEPS as _LRBS_SWEEPS,
                                     DEFAULT_NAME_PREFIX as _PT_PREFIX,
                                     LrBatchSweep, GLOBAL_BATCH_SIZE)
from launch_jolmo.pretraining_matrix import _existing_jolmo_runs

# --- PTSweep ↔ MuonExpt3 aliasing -------------------------------------------
# A PTSweep cell at the reference batch and default weight decay
#     PTSweep60M-0.06B-chinchilla-<c>-<opt>-<lr…>-wd0.1-bs1M-wsd
# is the SAME training configuration as the older
#     MuonExpt3-0.06B-chinchilla-<c>-<opt>-<lr…>-wsd
# — one underlying model under two names. Where the MuonExpt3-named artifact
# exists on GCS, finetune under THAT name, so its existing CPT artifacts keep
# matching and the same model is never finetuned twice under both names.

_PT_ALIAS_SUFFIX = "-wd0.1-bs1M-wsd"


def _canonical_base(m: JolmoModel) -> JolmoModel:
    name = m.model_name
    if not (name.startswith(f"{_PT_PREFIX}-") and name.endswith(_PT_ALIAS_SUFFIX)):
        return m
    mid = name[len(_PT_PREFIX) + 1 : -len(_PT_ALIAS_SUFFIX)]
    twin = f"MuonExpt3-{mid}-wsd"
    if twin in _existing_jolmo_runs():
        return _dc_replace(m, model_name=twin)
    return m


# One FtSweep per 60M pretrained base from the LR × batch-size sweeps, bases
# canonicalized as above and deduplicated. Under a non-60M $OPTIM_SIZE the
# lr_bs sweeps build 0 models, so this stays empty and no ft- stages are
# registered. Gated on argv (same pattern as cpt-all in pretraining_matrix) so
# only ft- commands pay for the GCS listing.
_WANT_FT = any(a.startswith("ft-") for a in sys.argv)

_ft_sweeps = []
if _WANT_FT:
    _seen = set()
    # HI5_SWEEPS is included alongside the full grid: its above-optimal subsets
    # reach chinchillas the grid never ran (e.g. c16), so those bases would
    # otherwise have no ft- stage at all. Dedupe below keeps the overlap free.
    from launch_jolmo.lr_bs_sweep import HI5_SWEEPS as _HI5_SWEEPS
    for _lrbs_sweep in tuple(_LRBS_SWEEPS) + tuple(_HI5_SWEEPS):
        for _m in _lrbs_sweep.models():
            _b = _canonical_base(_m)
            if _b.run_name in _seen:
                continue
            _seen.add(_b.run_name)
            _ft_sweeps.append(FtSweep(base=_b))

SWEEPS: Tuple[FtSweep, ...] = tuple(_ft_sweeps)


# ---------------------------------------------------------------------------
# Crossover sweep: one (dataset, size) -> chinchilla cell each, finetuned WITH
# DCLM replay. Unlike the sweeps above, each base is finetuned on only the one
# dataset its crossover budget was measured for.
# ---------------------------------------------------------------------------

CROSSOVER = {
    #                 30M      60M     100M     300M     600M
    "alpaca":      {"30M": 32, "60M": 8,  "100M": 8, "300M": 4, "600M": 1},
    "gsm8k":       {"30M": 4,  "60M": 2,  "100M": 2, "300M": 2, "600M": 0.5},
    "musicpile":   {"30M": 32, "60M": 16, "100M": 4, "300M": 2, "600M": 1},
    "stackmathqa": {"30M": 32, "60M": 8,  "100M": 4, "300M": 2, "600M": 0.5},
    "starcoder":   {"30M": 8,  "60M": 16, "100M": 4, "300M": 4, "600M": 1},
}
# 60M is deliberately excluded (already covered by the ft-lrbs sweeps).
CROSSOVER_SKIP_SIZES = {"60M"}
# The replay axis this study re-arms; r = 0 is the no-replay control and keeps
# the plain CPT run name, so those cells reuse existing artifacts.
CROSSOVER_REPLAY_FRACTIONS: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3)

_crossover_sweeps = []
if _WANT_FT:
    from launch_jolmo.sizes import active_profile
    from launch_jolmo.pretraining_matrix import tuned_bases_for

    _size, _ = active_profile()
    if _size not in CROSSOVER_SKIP_SIZES:
        for _ds, _per_size in CROSSOVER.items():
            _chin = _per_size.get(_size)
            if _chin is None:
                continue
            for _b in tuned_bases_for([_chin]):
                _crossover_sweeps.append(FtSweep(
                    base=_b,
                    datasets=(_ds,),
                    replay_fractions=CROSSOVER_REPLAY_FRACTIONS,
                    label_suffix=f"-xover-{_ds}",
                ))

CROSSOVER_SWEEPS: Tuple[FtSweep, ...] = tuple(_crossover_sweeps)

# ---------------------------------------------------------------------------
# Best-LR cell per (chinchilla x batch size x optimizer) at 60M, measured from
# the DCLM-heldout evals (see new_utils/plot_pt_lr_dclm_bs.py and its
# -besttable.csv). Unlike the sweeps above these span bs2M / bs4M, so the batch
# axis is represented. Names are frozen here rather than recomputed: the winner
# would otherwise shift silently whenever a new LR eval lands.
# ---------------------------------------------------------------------------

BS_BEST_BASES_60M = (
    "PTSweep60M-0.06B-chinchilla-0.25-adamw-lr5.6e-2-wd0.1-bs1M-wsd",
    "PTSweep60M-0.06B-chinchilla-0.25-adamw-lr1.1e-1-wd0.1-bs2M-wsd",
    "PTSweep60M-0.06B-chinchilla-0.25-adamw-lr1.6e-1-wd0.1-bs4M-wsd",
    "PTSweep60M-0.06B-chinchilla-0.25-muon-muonlr1.4e-2-adamwlr5.6e-2-wd0.1-bs1M-wsd",
    "PTSweep60M-0.06B-chinchilla-0.25-muon-muonlr9.9e-3-adamwlr7.9e-2-wd0.1-bs2M-wsd",
    "PTSweep60M-0.06B-chinchilla-0.25-muon-muonlr1.4e-2-adamwlr1.1e-1-wd0.1-bs4M-wsd",
    "PTSweep60M-0.06B-chinchilla-0.5-adamw-lr5.6e-2-wd0.1-bs1M-wsd",
    "PTSweep60M-0.06B-chinchilla-0.5-adamw-lr7.9e-2-wd0.1-bs2M-wsd",
    "PTSweep60M-0.06B-chinchilla-0.5-adamw-lr1.6e-1-wd0.1-bs4M-wsd",
    "PTSweep60M-0.06B-chinchilla-0.5-muon-muonlr1.0e-2-adamwlr5.6e-2-wd0.1-bs1M-wsd",
    "PTSweep60M-0.06B-chinchilla-0.5-muon-muonlr1.4e-2-adamwlr7.9e-2-wd0.1-bs2M-wsd",
    "PTSweep60M-0.06B-chinchilla-0.5-muon-muonlr1.4e-2-adamwlr1.1e-1-wd0.1-bs4M-wsd",
    "PTSweep60M-0.06B-chinchilla-1-adamw-lr2.0e-2-wd0.1-bs1M-wsd",
    "PTSweep60M-0.06B-chinchilla-1-adamw-lr4.0e-2-wd0.1-bs2M-wsd",
    "PTSweep60M-0.06B-chinchilla-1-adamw-lr5.6e-2-wd0.1-bs4M-wsd",
    "PTSweep60M-0.06B-chinchilla-1-muon-muonlr1.4e-2-adamwlr1.4e-2-wd0.1-bs1M-wsd",
    "PTSweep60M-0.06B-chinchilla-1-muon-muonlr1.4e-2-adamwlr2.0e-2-wd0.1-bs2M-wsd",
    "PTSweep60M-0.06B-chinchilla-1-muon-muonlr1.4e-2-adamwlr2.8e-2-wd0.1-bs4M-wsd",
    "PTSweep60M-0.06B-chinchilla-2-adamw-lr2.0e-2-wd0.1-bs1M-wsd",
    "PTSweep60M-0.06B-chinchilla-2-adamw-lr2.0e-2-wd0.1-bs2M-wsd",
    "PTSweep60M-0.06B-chinchilla-2-adamw-lr2.8e-2-wd0.1-bs4M-wsd",
    "PTSweep60M-0.06B-chinchilla-2-muon-muonlr1.4e-2-adamwlr2.8e-2-wd0.1-bs1M-wsd",
    "PTSweep60M-0.06B-chinchilla-2-muon-muonlr1.4e-2-adamwlr1.4e-2-wd0.1-bs2M-wsd",
    "PTSweep60M-0.06B-chinchilla-2-muon-muonlr1.4e-2-adamwlr2.0e-2-wd0.1-bs4M-wsd",
    "PTSweep60M-0.06B-chinchilla-4-adamw-lr1.0e-2-wd0.1-bs1M-wsd",
    "PTSweep60M-0.06B-chinchilla-4-adamw-lr1.4e-2-wd0.1-bs2M-wsd",
    "PTSweep60M-0.06B-chinchilla-4-adamw-lr1.4e-2-wd0.1-bs4M-wsd",
    "PTSweep60M-0.06B-chinchilla-4-muon-muonlr1.4e-2-adamwlr2.8e-2-wd0.1-bs1M-wsd",
    "PTSweep60M-0.06B-chinchilla-4-muon-muonlr9.9e-3-adamwlr9.9e-3-wd0.1-bs2M-wsd",
    "PTSweep60M-0.06B-chinchilla-4-muon-muonlr1.4e-2-adamwlr1.4e-2-wd0.1-bs4M-wsd",
    "PTSweep60M-0.06B-chinchilla-8-adamw-lr1.4e-2-wd0.1-bs1M-wsd",
    "PTSweep60M-0.06B-chinchilla-8-adamw-lr1.4e-2-wd0.1-bs2M-wsd",
    "PTSweep60M-0.06B-chinchilla-8-adamw-lr2.0e-2-wd0.1-bs4M-wsd",
    "PTSweep60M-0.06B-chinchilla-8-muon-muonlr1.0e-2-adamwlr1.4e-2-wd0.1-bs1M-wsd",
    "PTSweep60M-0.06B-chinchilla-8-muon-muonlr7.1e-3-adamwlr1.4e-2-wd0.1-bs2M-wsd",
    "PTSweep60M-0.06B-chinchilla-8-muon-muonlr2.0e-2-adamwlr2.0e-2-wd0.1-bs4M-wsd",
    "PTSweep60M-0.06B-chinchilla-16-adamw-lr2.8e-2-wd0.1-bs1M-wsd",
    "PTSweep60M-0.06B-chinchilla-16-muon-muonlr1.4e-2-adamwlr1.0e-2-wd0.1-bs1M-wsd",
    "PTSweep60M-0.06B-chinchilla-32-muon-muonlr7.0e-3-adamwlr1.0e-2-wd0.1-bs1M-wsd",
    "PTSweep60M-0.06B-chinchilla-64-muon-muonlr5.0e-3-adamwlr7.0e-3-wd0.1-bs1M-wsd",
)

def _as_ptsweep_name(name: str) -> str:
    """Map a MuonExpt3-aliased bs1M name back to its PTSweep equivalent.

    LrBatchSweep renames a reference-batch cell to its MuonExpt3 twin when that
    twin exists on GCS, so matching the frozen PTSweep names directly misses
    every aliased bs1M cell. Both spellings denote the same run.
    """
    m = re.match(r"^MuonExpt3-(?P<mt>[0-9.]+B)-chinchilla-(?P<c>[0-9.]+)"
                 r"-(?P<rest>.+)-wsd$", name)
    if not m:
        return name
    return (f"PTSweep{_PT_PREFIX.replace('PTSweep', '')}-{m.group('mt')}"
            f"-chinchilla-{m.group('c')}-{m.group('rest')}-wd0.1-bs1M-wsd")


_BS_BEST_RE = re.compile(
    r"^PTSweep(?P<size>\w+?)-(?P<mt>[0-9.]+B)-chinchilla-(?P<chin>[0-9.]+)"
    r"-(?:adamw-lr(?P<alr>[0-9.e\-]+)"
    r"|muon-muonlr(?P<mlr>[0-9.e\-]+)-adamwlr(?P<comp>[0-9.e\-]+))"
    r"-wd0\.1-bs(?P<bs>\d+)M-wsd$")


def _base_from_frozen_name(name: str):
    """Rebuild the JolmoModel for a frozen best-LR cell straight from its name.

    Matching against the live lr_bs grid is not enough: a muon cell's adamw
    component is pinned to whatever PT_LR_BY_MODEL said at the time, so
    retuning the table stops the grid from generating cells that were actually
    trained. The name carries every parameter needed, so parse it instead.
    """
    m = _BS_BEST_RE.match(name)
    if not m:
        return None
    chin = float(m.group("chin"))
    mult = int(m.group("bs"))          # bs1M/2M/4M -> multiplier of the 1M ref
    opt = "adamw" if m.group("alr") else "muon"
    sweep = LrBatchSweep(model_type=m.group("mt"), optimizer=opt,
                         chinchilla=chin, bs_multipliers=(mult,))
    if not sweep.size_ok:
        return None
    gbs = GLOBAL_BATCH_SIZE * mult
    kw = ({"optimizer": "adamw", "learning_rate": float(m.group("alr"))}
          if opt == "adamw" else
          {"optimizer": "muon", "muon_lr": float(m.group("mlr")),
           "learning_rate": float(m.group("comp"))})
    return JolmoModel(model_name=name, **sweep._shared_params(gbs),
                      **sweep._schedule_for(gbs), **kw)


_bs_best_sweeps = []
if _WANT_FT:
    for _name in BS_BEST_BASES_60M:
        _b = _base_from_frozen_name(_name)
        if _b is not None:
            _bs_best_sweeps.append(FtSweep(base=_b, label_suffix="-bsbest"))

BS_BEST_SWEEPS: Tuple[FtSweep, ...] = tuple(_bs_best_sweeps)

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
    "tulu", "starcoder", "musicpile", "alpaca", "gsm8k", "stackmathqa",
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

from launch_jolmo.lr_bs_sweep import SWEEPS as _LRBS_SWEEPS, DEFAULT_NAME_PREFIX as _PT_PREFIX
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
    for _lrbs_sweep in _LRBS_SWEEPS:
        for _m in _lrbs_sweep.models():
            _b = _canonical_base(_m)
            if _b.run_name in _seen:
                continue
            _seen.add(_b.run_name)
            _ft_sweeps.append(FtSweep(base=_b))

SWEEPS: Tuple[FtSweep, ...] = tuple(_ft_sweeps)

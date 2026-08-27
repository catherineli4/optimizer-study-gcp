"""`FtSweep` — full finetuning cross product for ONE pretrained base model.

Analog of `lr_bs_sweep.LrBatchSweep` for the CPT/finetuning phase: takes a single
pretrained `JolmoModel` and builds every combination of

    dataset  ×  FT optimizer  ×  FT LR  ×  DCLM replay fraction

as `CPTModel` artifacts, plus matching `ModelEvaluation`s.

Defaults: 6 datasets × (8 adamw LRs + 8 muon LRs) × 4 replay fractions
        = **384 finetunes** per base.

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
FT_REPLAY_FRACTIONS: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3)

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

        models = []
        seen = set()
        for ds in self.datasets:
            for opt in self.optimizers:
                for lr_entry in self._lr_entries(opt):
                    if opt == "muon":
                        muon_lr, adamw_lr = lr_entry
                    else:
                        muon_lr, adamw_lr = None, lr_entry
                    for r in self.replay_fractions:
                        m = CPTModel(
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
            ModelEvaluation(
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

SWEEPS: Tuple[FtSweep, ...] = ()

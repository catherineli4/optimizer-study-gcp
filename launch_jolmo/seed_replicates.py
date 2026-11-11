"""Seed replicates and WSD decay-floor variants of the tuned 60M bases.

Spec: docs/seed_replicates_60m.md.

Two experiments over the SAME six base cells -- the tuned (best-LR) 60M
pretrain at chinchilla 1 / 2 / 4 for both optimizers, exactly as
``pretraining_matrix.tuned_bases_for`` builds them (same SHARED_MODEL_PARAMS,
LR table, token budget, DCLM parts; the run name resolves to whichever schema
exists on GCS, PTSweep preferred):

  1. ``seed-rep-60m``  -- the same configuration under two more training seeds
     (1 and 2; the reference run is seed 0), so every cell has three seeds.
     Run name: ``<reference run>-seed<s>``.
  2. ``decaymin-60m``  -- the reference configuration (seed 0) but with the WSD
     decay phase ending at 0.1 x max LR instead of 0 (per param group, so the
     Muon leg and its AdamW component each decay to 0.1 x their own max).
     Run name: ``<reference run>-decaymin0.1``.

Each experiment also gets: the canonical pretrain eval (diversity-v2 +
DCLM_heldout, like ``eval-pretrain-*``), gsm8k finetunes with the ``cpt``
recipe (adamw base -> adamw FT over CPT_LR_SWEEP; muon base -> muon (adamw
component = 0.25 x muon_lr) AND adamw FT), and the ``eval-cpt`` evals.

Both suffixes end in something other than ``-wsd``, and every GCS-listing
discovery regex in the repo anchors on ``-wsd$``, so these runs are never
mistaken for grid cells by cpt-all / bs1m / divergence stages.

    OPTIM_SIZE=60M python -m launch_jolmo.seed_replicates            # list
    OPTIM_SIZE=60M python -m launch_jolmo.seed_replicates --verify   # YAML vs reference config.json
    OPTIM_SIZE=60M python -m launch_jolmo.launcher drylaunch seed-rep-60m decaymin-60m
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import replace as _dc_replace
from typing import Any, Dict, List, Tuple

from experiments import ArtifactSet, Project

from launch_jolmo.sizes import active_profile

_SIZE, _PROFILE = active_profile()
if Project.config is None:
    Project.init(_PROFILE["project"])

from launch_jolmo.training import JolmoModel, ModelEvaluation, CPTModel
from launch_jolmo.pretraining_matrix import (
    tuned_bases_for,
    _pretrain_eval,
    dclm_heldout_val_chunks,
    DCLM_HELDOUT_INSTANCES,
)
from launch_jolmo.cpt import build_cpt_models, build_cpt_model_evaluations

MODEL_TYPE = "0.06B"
CHINCHILLAS: Tuple[int, ...] = (1, 2, 4)
OPTIMIZERS: Tuple[str, ...] = ("adamw", "muon")
NEW_SEEDS: Tuple[int, ...] = (1, 2)          # the reference run is seed 0
DECAY_MIN_LR_FRACTION: float = 0.1           # WSD floor = 0.1 x max LR (per group)
FT_DATASETS: List[str] = ["gsm8k"]
_SIZE_OK = _PROFILE["model_type"] == MODEL_TYPE


def _seed_name(base: JolmoModel, seed: int) -> str:
    return f"{base.model_name}-seed{seed}"


def _decay_name(base: JolmoModel) -> str:
    return f"{base.model_name}-decaymin{DECAY_MIN_LR_FRACTION:g}"


def _bases() -> List[JolmoModel]:
    """The six reference cells (seed 0, already on GCS)."""
    if not _SIZE_OK:
        print(f"[seed-rep] 0 models -- OPTIM_SIZE resolves to {_PROFILE['model_type']}, "
              f"not {MODEL_TYPE}; rerun with OPTIM_SIZE=60M", flush=True)
        return []
    bases = list(tuned_bases_for(CHINCHILLAS, OPTIMIZERS))
    missing = {(c, o) for c in CHINCHILLAS for o in OPTIMIZERS} - {
        (c, o) for c in CHINCHILLAS for o in OPTIMIZERS
        if any(f"-chinchilla-{c}-{o}-" in b.model_name for b in bases)}
    if missing:
        print(f"[seed-rep] WARNING: no tuned LR for {sorted(missing)}; those cells are skipped",
              flush=True)
    return bases


def seed_replicates(bases: List[JolmoModel]) -> List[JolmoModel]:
    """Same config as the base, under NEW_SEEDS. ``seed`` reaches model.init_seed,
    data_loader.seed and the top-level init_seed of the generated YAML."""
    return [_dc_replace(b, model_name=_seed_name(b, s), seed=s)
            for b in bases for s in NEW_SEEDS]


def decay_variants(bases: List[JolmoModel]) -> List[JolmoModel]:
    """Same config as the base (seed 0), WSD decaying to 0.1 x max LR."""
    return [_dc_replace(b, model_name=_decay_name(b),
                        wsd_decay_min_lr_fraction=DECAY_MIN_LR_FRACTION)
            for b in bases]


def _finetunes(models: List[JolmoModel]) -> ArtifactSet:
    """gsm8k finetunes with the `cpt` stage's exact pairing and LR grids."""
    adamw = ArtifactSet([m for m in models if m.optimizer == "adamw"])
    muon = ArtifactSet([m for m in models if m.optimizer == "muon"])
    return (build_cpt_models(adamw, datasets=FT_DATASETS)
            + build_cpt_models(muon, datasets=FT_DATASETS,
                               cpt_optimizers=["muon", "adamw"],
                               muon_adamw_multiplier=0.25))


_DCLM = dict(extra_val_chunks=dclm_heldout_val_chunks,
             extra_val_max_instances=DCLM_HELDOUT_INSTANCES)


def _stages(prefix: str, models: List[JolmoModel]) -> Dict[str, ArtifactSet]:
    """<prefix>-models / -adamw-models / -muon-models / -evals / -gsm8k / -gsm8k-evals."""
    fts = _finetunes(models)
    return {
        f"{prefix}-models": ArtifactSet(models),
        f"{prefix}-adamw-models": ArtifactSet([m for m in models if m.optimizer == "adamw"]),
        f"{prefix}-muon-models": ArtifactSet([m for m in models if m.optimizer == "muon"]),
        f"{prefix}-evals": ArtifactSet([_pretrain_eval(m) for m in models]),
        f"{prefix}-gsm8k": fts,
        f"{prefix}-gsm8k-evals": build_cpt_model_evaluations(fts, **_DCLM),
    }


# Everything each group launch needs, bases included, so the executor's
# identity-based dependency check resolves (same pattern as the `ewc` group).
GROUP_MEMBERS = ("models", "evals", "gsm8k", "gsm8k-evals")

BASES: List[JolmoModel] = _bases()
SEED_REP_MODELS: List[JolmoModel] = seed_replicates(BASES)
DECAYMIN_MODELS: List[JolmoModel] = decay_variants(BASES)

seed_rep_stages: Dict[str, ArtifactSet] = _stages("seed-rep-60m", SEED_REP_MODELS)
decaymin_stages: Dict[str, ArtifactSet] = _stages("decaymin-60m", DECAYMIN_MODELS)


# --- verification -------------------------------------------------------------
# Keys ALLOWED to differ between a variant's generated YAML and its reference
# run's resolved config.json (the rest must be identical).
_ALLOWED_SEED = ("model.init_seed", "data_loader.seed", "init_seed",
                 "wandb.", "dataset.paths", "dataset.work_dir",
                 "validation_datasets.", "trainer.save_folder")
_ALLOWED_DECAY = ("train_module.scheduler.decay_min_lr_fraction",
                  "wandb.", "dataset.paths", "dataset.work_dir",
                  "validation_datasets.", "trainer.save_folder")
# Differ only because of the GPU count the reference happened to train with
# (fixed global batch -> same optimisation); reported, never fatal.
_GPU_ONLY = ("train_module.rank_microbatch_size", "wandb.config.num_processes")


def _flat(d: Any, p: str = "") -> Dict[str, Any]:
    if isinstance(d, dict):
        out: Dict[str, Any] = {}
        for k, v in d.items():
            out.update(_flat(v, f"{p}{k}."))
        return out
    return {p[:-1]: d}


def _norm(v: Any, key: str = "") -> Any:
    if isinstance(v, tuple):
        v = list(v)
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x, k) for k, x in sorted(v.items())}
    if key.endswith("_CLASS_") and isinstance(v, str):
        return v.rsplit(".", 1)[-1]        # olmo_core.optim.WSD == ...scheduler.WSD
    return v


def _reference_config(run: str, tmp: str) -> Dict[str, Any]:
    src = f"{Project.config.remote_path}/JolmoModel/{run}/final-unsharded/config.json"
    dst = os.path.join(tmp, run + ".json")
    subprocess.run(["gsutil", "-q", "cp", src, dst], check=True)
    with open(dst) as f:
        return json.load(f)


def _generated(m: JolmoModel) -> Dict[str, Any]:
    val = {label: [f"/dummy/{label}.bin"] for label, _ in m.validation_chunks}
    return _flat(m._build_yaml_config(
        save_folder=f"gs://dummy/{m.model_name}",
        train_paths=[f"/dummy/train-{i}.npy" for i in range(len(m.train_chunks))],
        val_datasets=val,
        work_dir=f"/dummy/work/{m.model_name}",
    ))


def verify() -> bool:
    """Diff each variant's generated YAML against its reference config.json."""
    if not BASES:
        return False
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        refs = {b.model_name: _flat(_reference_config(b.model_name, tmp)) for b in BASES}
        checks = ([(m, m.model_name.rsplit("-seed", 1)[0], _ALLOWED_SEED, {"seed": m.seed})
                   for m in SEED_REP_MODELS]
                  + [(m, m.model_name.rsplit("-decaymin", 1)[0], _ALLOWED_DECAY, {})
                     for m in DECAYMIN_MODELS])
        for m, ref_name, allowed, want in checks:
            ref, gen = refs[ref_name], _generated(m)
            unexpected, expected, gpu_only, missing = [], [], [], []
            for k, v in gen.items():
                if k not in ref:
                    missing.append(k)
                elif _norm(v, k) != _norm(ref[k], k):
                    if k.startswith(allowed):
                        expected.append((k, ref[k], v))
                    elif k in _GPU_ONLY:
                        gpu_only.append((k, ref[k], v))
                    else:
                        unexpected.append((k, ref[k], v))
            print(f"\n== {m.model_name}\n   vs {ref_name}")
            print(f"   {len(gen)} generated keys; "
                  f"{len(gen) - len(missing) - len(expected) - len(unexpected) - len(gpu_only)} identical")
            for k, a, b in expected:
                print(f"   expected diff   {k}: ref={str(a)[:50]}  new={str(b)[:50]}")
            for k, a, b in gpu_only:
                print(f"   gpu-count diff  {k}: ref={a}  new={b}")
            for k in missing:
                print(f"   NOTE generated key absent from reference: {k}")
            for k, a, b in unexpected:
                print(f"   UNEXPECTED diff {k}: ref={a!r}  new={b!r}")
                ok = False
            if "seed" in want:
                for k in ("model.init_seed", "data_loader.seed", "init_seed"):
                    if gen.get(k) != want["seed"]:
                        print(f"   UNEXPECTED: {k} = {gen.get(k)!r}, want {want['seed']}")
                        ok = False
            else:
                k = "train_module.scheduler.decay_min_lr_fraction"
                if gen.get(k) != DECAY_MIN_LR_FRACTION:
                    print(f"   UNEXPECTED: {k} = {gen.get(k)!r}, want {DECAY_MIN_LR_FRACTION}")
                    ok = False
    print("\nVERIFY:", "OK -- only the documented keys differ" if ok else "FAILED")
    return ok


def _summary() -> None:
    print(f"[seed-rep] {len(BASES)} reference base(s):")
    for b in BASES:
        print(f"   {b.model_name}")
    for name, stages in (("seed-rep-60m", seed_rep_stages), ("decaymin-60m", decaymin_stages)):
        print(f"[{name}] " + ", ".join(f"{k.split(name + '-', 1)[1]}={len(v)}"
                                       for k, v in stages.items()))
    for m in SEED_REP_MODELS + DECAYMIN_MODELS:
        print(f"   {m.model_name}: lr={m.learning_rate} muon_lr={m.muon_lr if m.optimizer == 'muon' else '-'} "
              f"n_tokens={m.n_tokens} warmup={m.warmup_steps} gpus={m.num_processes} "
              f"seed={m.seed} wsd_floor={m.wsd_decay_min_lr_fraction}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verify", action="store_true",
                    help="diff each variant's generated YAML against its reference config.json")
    args = ap.parse_args()
    _summary()
    if args.verify:
        sys.exit(0 if verify() else 1)

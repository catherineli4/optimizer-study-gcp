#!/usr/bin/env python3
"""Evaluate saved multi-seed perturbed checkpoints and average DCLM_heldout loss.

For each seed under ``{parent}/seed_XXX/final-unsharded/``, evaluates the
checkpoint with the provided eval config, then writes a JSON with per-seed
losses and their mean.

All seeds share one process: the transformer is built once, the eval batches
are read from disk once, and each seed only swaps in its state dict. (The
previous version shelled out to validate.py per seed — 10 torch imports, 10
model constructions, and 10 re-reads of the eval data per task.)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Sequence

import yaml


def _losses_from_eval(data: dict) -> Dict[str, float]:
    """Extract scalar losses from a validate.py-shaped eval dict."""
    losses: Dict[str, float] = {}
    overall = (data.get("overall") or {}).get("loss")
    if overall is not None:
        losses["overall"] = float(overall)
    for label, entry in (data.get("by_label") or {}).items():
        loss = (entry or {}).get("loss")
        if loss is not None:
            losses[label] = float(loss)
    return losses


def seed_subdir(seed: int) -> str:
    return f"seed_{seed:03d}"


def average_seed_losses(
    sample_losses: List[Dict[str, float]],
    *,
    seeds: Sequence[int],
    gamma: float,
) -> dict:
    """Mean loss per label across seeds; also keep per-seed breakdown."""
    if not sample_losses:
        raise ValueError("no sample losses to average")
    if len(sample_losses) != len(seeds):
        raise ValueError(
            f"len(sample_losses)={len(sample_losses)} != len(seeds)={len(seeds)}"
        )

    labels = sorted({label for losses in sample_losses for label in losses})
    out: Dict[str, Any] = {}

    if "overall" in labels:
        vals = [losses["overall"] for losses in sample_losses if "overall" in losses]
        out["overall"] = {"loss": sum(vals) / len(vals)}

    by_label: Dict[str, Any] = {}
    for label in labels:
        if label == "overall":
            continue
        vals = [losses[label] for losses in sample_losses if label in losses]
        if vals:
            by_label[label] = {"loss": sum(vals) / len(vals)}
    if by_label:
        out["by_label"] = by_label

    per_seed = {
        str(seed): dict(losses)
        for seed, losses in zip(seeds, sample_losses)
    }
    out["perturbation"] = {
        "gamma": gamma,
        "seeds": list(seeds),
        "num_seeds": len(seeds),
        "loss_averaging": "mean_over_saved_seeds",
        "per_seed": per_seed,
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Average validate.py loss over saved multi-seed perturbations",
    )
    parser.add_argument(
        "--parent-dir",
        required=True,
        help="Local PerturbedModel/{base}_perturbed_{γ}/ directory containing seed_*/",
    )
    parser.add_argument(
        "--seeds",
        required=True,
        help="Comma-separated seeds matching seed_XXX subdirs (e.g. 0,1,...,9)",
    )
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument(
        "--eval-config",
        required=True,
        help="validate.py YAML config (model.path is overwritten per seed)",
    )
    parser.add_argument(
        "--validate-script",
        required=True,
        help="Path to JOLMo/src/scripts/validate.py (imported, not spawned)",
    )
    parser.add_argument("--output", required=True, help="Path for averaged eval JSON")
    args = parser.parse_args()

    seeds = [int(p.strip()) for p in args.seeds.split(",") if p.strip()]
    if not seeds:
        raise SystemExit("--seeds must be a non-empty comma-separated list")

    # Reuse validate.py's helpers (batching, loss, model discovery) in-process.
    sys.path.insert(0, os.path.dirname(os.path.abspath(args.validate_script)))
    import validate as V  # noqa: E402
    import numpy as np  # noqa: E402
    import torch  # noqa: E402
    from olmo_core.nn.transformer import TransformerConfig  # noqa: E402

    with open(args.eval_config, "r", encoding="utf-8") as f:
        eval_cfg = yaml.safe_load(f)

    chunk_size = int(eval_cfg["chunk_size"])
    batch_size = int(eval_cfg.get("batch_size", 8))
    device = V.detect_device(eval_cfg.get("device"))

    # Read the eval data once and reuse the batches for every seed.
    cached_batches = []
    for ds in eval_cfg["validation_datasets"]:
        batches = list(
            V.iter_batches_memmap(
                ds["paths"], chunk_size, batch_size, ds.get("max_instances")
            )
        )
        cached_batches.append((ds["name"], batches))

    model = None
    sample_losses: List[Dict[str, float]] = []

    with torch.no_grad():
        for i, seed in enumerate(seeds):
            ckpt_dir = os.path.join(args.parent_dir, seed_subdir(seed), "final-unsharded")
            state_path = V.find_model_state_path(ckpt_dir)

            if model is None:
                # All seeds share the base architecture: build the model once.
                cfg_path = V.find_config_json_near(os.path.dirname(state_path))
                with open(cfg_path, "r", encoding="utf-8") as f:
                    exp_cfg = json.load(f)
                if "model" not in exp_cfg:
                    raise RuntimeError(
                        f"Invalid config at '{cfg_path}': missing 'model' section."
                    )
                model_cfg = TransformerConfig.from_dict(exp_cfg["model"])
                model = model_cfg.build(init_device="cpu").to(device=device)
                model.eval()

            state = torch.load(state_path, map_location="cpu")
            model.load_state_dict(state, strict=True)
            del state

            print(f"  seed {i + 1}/{len(seeds)} (seed={seed}) → {ckpt_dir}")
            totals: Dict[str, Dict[str, Any]] = {}
            overall_sum = 0.0
            overall_tok = 0
            for name, batches in cached_batches:
                lsum = 0.0
                ntok = 0
                for np_batch in batches:
                    ids = torch.from_numpy(np_batch.astype(np.int64)).to(device)
                    with torch.autocast(
                        device.type,
                        dtype=torch.bfloat16,
                        enabled=device.type == "cuda",
                    ):
                        logits = model(input_ids=ids)
                    s, n = V.per_instance_loss_from_logits(logits.float(), ids)
                    lsum += float(s.sum().item())
                    ntok += int(n.sum().item())
                totals[name] = {
                    "loss": (lsum / ntok) if ntok > 0 else None,
                    "num_tokens": ntok,
                }
                overall_sum += lsum
                overall_tok += ntok

            seed_result = {
                "overall": {
                    "loss": (overall_sum / overall_tok) if overall_tok > 0 else None,
                },
                "by_label": totals,
            }
            sample_losses.append(_losses_from_eval(seed_result))

    averaged = average_seed_losses(
        sample_losses, seeds=seeds, gamma=args.gamma,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(averaged, f, indent=2)
    print(f"Wrote averaged multi-seed eval to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Elastic Weight Consolidation finetuning, as a third method beside plain CPT
and DCLM replay.

Instead of mixing pretraining tokens back into the finetune stream, penalise the
WEIGHTS for drifting from the pretrained solution, weighted by how much each one
mattered to the pretraining task::

    L = L_finetune(theta) + (lambda / 2) * sum_i F_i * (theta_i - theta*_i)^2

``theta*`` are the pretrained weights and ``F`` the diagonal Fisher of the
PRETRAINING task. ``lambda = 0`` recovers ordinary finetuning. Semantics follow
WattsIshaan/sharpness-aware-pretraining (``olmo/ewc_trainer.py``), so a given
lambda means the same thing here as there.

Deliberately self-contained: its own training loop, NOT the olmo_core Trainer and
not a Trainer callback. It borrows only the model *definition* — the
TransformerConfig spec in the job's config.yaml — so the architecture matches the
checkpoint being loaded. Nothing in the CPT path changes.

Invoked by launch_jolmo/ewc.py; runnable by hand for debugging.
"""

import argparse
import json
import math
import os
import shutil
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml


# ---------------------------------------------------------------------------
# Data: random contiguous windows over memmapped .npy token shards
# ---------------------------------------------------------------------------

class WindowSampler:
    """Random contiguous token windows drawn across shards by length.

    Sampling is WITH REPLACEMENT and has no epoch structure, so a small finetune
    set repeats implicitly. That differs from the CPT path, which controls
    repetition explicitly via ``max_repetition_ratio`` on the source mixture.
    """

    def __init__(self, paths: Sequence[str], seq_len: int, seed: int,
                 dtype: str = "uint32"):
        if not paths:
            raise ValueError("WindowSampler needs at least one token path")
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)
        self.arrays = []
        lengths = []
        for p in paths:
            # These files carry a .npy extension but hold RAW token arrays with
            # no npy header -- np.load reads the leading tokens as a pickle
            # header and dies with "This file contains pickled (object) data".
            # olmo_core reads them as a flat memmap of `dtype` (uint32 in every
            # dataset spec here), so do the same.
            a = np.memmap(p, dtype=np.dtype(dtype), mode="r")
            if a.shape[0] < seq_len + 1:
                # A short shard would yield a truncated row and break np.stack.
                # Skip it loudly rather than corrupting the batch silently.
                print(f"[ewc] SKIPPING short shard ({a.shape[0]} < {seq_len + 1}): {p}",
                      flush=True)
                continue
            self.arrays.append(a)
            lengths.append(a.shape[0])
        if not self.arrays:
            raise ValueError(
                f"no shard is long enough for sequence_length={seq_len}: {list(paths)}")
        total = float(sum(lengths))
        self.probs = np.array([n / total for n in lengths], dtype=np.float64)
        self.lengths = lengths

    def batch(self, n_seqs: int) -> torch.Tensor:
        """(n_seqs, seq_len + 1) int64 — one extra token for the shifted label."""
        want = self.seq_len + 1
        idx = self.rng.choice(len(self.arrays), size=n_seqs, p=self.probs)
        rows = []
        for i in idx:
            a = self.arrays[i]
            start = int(self.rng.integers(0, self.lengths[i] - want + 1))
            rows.append(np.asarray(a[start:start + want], dtype=np.int64))
        return torch.from_numpy(np.stack(rows))


# ---------------------------------------------------------------------------
# Model / optimizer
# ---------------------------------------------------------------------------

def build_model(config_path: str, device: torch.device):
    from olmo_core.nn.transformer import TransformerConfig

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    if "model" not in cfg:
        raise KeyError(f"{config_path} has no 'model' section")
    model_cfg = TransformerConfig.from_dict(cfg["model"])
    model = model_cfg.build(init_device="cpu")
    return model.to(device), cfg


def load_base_weights(model: torch.nn.Module, checkpoint: str) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # Unsharded checkpoints are sometimes wrapped in {"model": ...}.
    for key in ("model", "state_dict"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"checkpoint is missing parameters: {sorted(missing)[:8]}")
    if unexpected:
        print(f"[ewc] ignoring {len(unexpected)} unexpected key(s), "
              f"e.g. {sorted(unexpected)[:4]}", flush=True)


def build_optimizer(model, args):
    """Mirror launch_jolmo.training._build_optimizer_spec so the finetune uses
    the same param grouping (and embedding weight-decay override) as pretraining.
    """
    from olmo_core.optim import AdamWConfig, MuonConfig, OptimGroupOverride

    override = OptimGroupOverride(params=["embeddings.weight"],
                                  opts={"weight_decay": 0.0})
    if args.optimizer == "muon":
        cfg = MuonConfig(
            lr=args.muon_lr,
            weight_decay=args.muon_weight_decay,
            adamw_lr=args.learning_rate,
            adamw_betas=(args.beta1, args.beta2),
            adamw_weight_decay=args.weight_decay,
            group_overrides=[override],
        )
    else:
        cfg = AdamWConfig(
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            group_overrides=[override],
        )
    return cfg.build(model)


def cross_entropy(model, batch: torch.Tensor) -> torch.Tensor:
    """Next-token CE, logits upcast to fp32 for a stable loss scale."""
    inputs, labels = batch[:, :-1], batch[:, 1:]
    # FlashAttention takes fp16/bf16 only; a plain fp32 forward dies here.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(inputs)
    logits = out.logits if hasattr(out, "logits") else out
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))


# ---------------------------------------------------------------------------
# Fisher
# ---------------------------------------------------------------------------

def estimate_fisher(model, sampler: WindowSampler, n_batches: int,
                    micro_bs: int, device) -> Dict[str, torch.Tensor]:
    """Diagonal empirical Fisher on PRETRAINING data::

        F_i = (1/B) * sum_b (dL_b/dtheta_i)^2

    Batch gradients, not per-example — the standard cheap approximation, and what
    the reference implementation does.

    NOT rescaled to unit mean. Because F stays unnormalised its entries are
    small, which is why lambda is swept in the 1e3-1e5 range rather than near 1.
    Normalising would silently change what a given lambda means relative to the
    reference, so do not "fix" this without restating the whole sweep.
    """
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()
              if p.requires_grad}
    model.train()
    for b in range(n_batches):
        model.zero_grad(set_to_none=True)
        loss = cross_entropy(model, sampler.batch(micro_bs).to(device))
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
        if (b + 1) % 20 == 0:
            print(f"[ewc] fisher batch {b + 1}/{n_batches} loss={loss.item():.4f}",
                  flush=True)
    model.zero_grad(set_to_none=True)
    for n in fisher:
        fisher[n] /= float(n_batches)
    return fisher


def ewc_penalty(model, theta_star: Dict[str, torch.Tensor],
                fisher: Dict[str, torch.Tensor]) -> torch.Tensor:
    total = None
    for n, p in model.named_parameters():
        if not p.requires_grad or n not in fisher:
            continue
        term = (fisher[n] * (p - theta_star[n]) ** 2).sum()
        total = term if total is None else total + term
    return total


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="config.yaml with the model spec")
    ap.add_argument("--base-checkpoint", required=True, help="pretrained model.pt")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--train-paths", nargs="+", required=True)
    ap.add_argument("--fisher-paths", nargs="*", default=[],
                    help="PRETRAINING shards for the Fisher; required when lambda > 0")
    ap.add_argument("--base-config-json", default=None,
                    help="base final-unsharded/config.json, copied beside model.pt")

    ap.add_argument("--ewc-lambda", type=float, default=1e3)
    ap.add_argument("--fisher-batches", type=int, default=100)
    ap.add_argument("--train-tokens", type=int, default=20_000_000)
    ap.add_argument("--sequence-length", type=int, default=1024)
    ap.add_argument("--global-batch-size", type=int, default=65_536,
                    help="TOKENS per optimizer step")
    ap.add_argument("--micro-batch-size", type=int, default=8,
                    help="SEQUENCES per forward; memory only, not a hyperparameter")

    ap.add_argument("--optimizer", choices=("adamw", "muon"), default="adamw")
    ap.add_argument("--learning-rate", type=float, default=1e-4,
                    help="AdamW LR; for muon, the AdamW-COMPONENT LR")
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--muon-weight-decay", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=64)
    ap.add_argument("--token-dtype", default="uint32",
                    help="on-disk token width; matches the dataset specs")
    args = ap.parse_args()

    if args.ewc_lambda > 0 and not args.fisher_paths:
        raise ValueError(
            "ewc_lambda > 0 requires --fisher-paths (the pretraining shards the "
            "diagonal Fisher is estimated on)")

    # Never fall back to CPU. FlashAttention has no CPU kernel, so a CPU run
    # dies ~minutes later in flash_attn with an error that hides the real
    # cause -- which is what a transient "CUDA driver initialization failed"
    # looked like in the 60M sweep. Fail immediately and say why instead.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available to this task (driver init failed or no GPU "
            f"visible; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}). "
            "EWC needs a GPU: FlashAttention has no CPU kernel. Rerun the stage; "
            "the exists-check will retry only this task.")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    seqs_per_step = args.global_batch_size // args.sequence_length
    if seqs_per_step % args.micro_batch_size != 0:
        raise ValueError(
            f"micro_batch_size={args.micro_batch_size} must divide "
            f"{seqs_per_step} sequences per step")
    accum = seqs_per_step // args.micro_batch_size
    steps = args.train_tokens // args.global_batch_size

    print(f"[ewc] lambda={args.ewc_lambda:g} optimizer={args.optimizer} "
          f"lr={args.learning_rate:g}"
          + (f" muon_lr={args.muon_lr:g}" if args.optimizer == "muon" else ""),
          flush=True)
    print(f"[ewc] {steps} steps x {seqs_per_step} seqs "
          f"({accum} x {args.micro_batch_size}) x {args.sequence_length} tok "
          f"= {steps * args.global_batch_size:,} tokens", flush=True)

    model, _ = build_model(args.config, device)
    load_base_weights(model, args.base_checkpoint)

    # theta*, fisher and the live weights are all resident at once: budget ~3x
    # params on device.
    theta_star = {n: p.detach().clone()
                  for n, p in model.named_parameters() if p.requires_grad}

    fisher: Optional[Dict[str, torch.Tensor]] = None
    if args.ewc_lambda > 0:
        # Its own RNG and its own iterator, so the Fisher pass does not advance
        # the training stream.
        fisher_sampler = WindowSampler(args.fisher_paths, args.sequence_length,
                                       args.seed, args.token_dtype)
        fisher = estimate_fisher(model, fisher_sampler, args.fisher_batches,
                                 args.micro_batch_size, device)
        flat = torch.cat([f.reshape(-1) for f in fisher.values()])
        print(f"[ewc] fisher: mean={flat.mean():.3e} max={flat.max():.3e} "
              f"(unnormalised, by design)", flush=True)
    else:
        print("[ewc] lambda=0 -> skipping Fisher; this is plain finetuning",
              flush=True)

    train_sampler = WindowSampler(args.train_paths, args.sequence_length,
                                  args.seed + 1, args.token_dtype)
    optim = build_optimizer(model, args)
    # Apply the schedule as a MULTIPLIER on each group's base LR, so Muon's
    # matrix LR and its AdamW-component LR keep their relative scale instead of
    # being collapsed onto a single value.
    base_lrs = [g["lr"] for g in optim.param_groups]

    model.train()
    for step in range(steps):
        if step < args.warmup_steps:
            mult = (step + 1) / max(1, args.warmup_steps)
        else:
            prog = (step - args.warmup_steps) / max(1, steps - args.warmup_steps)
            mult = 0.5 * (1.0 + math.cos(math.pi * prog))
        for g, base in zip(optim.param_groups, base_lrs):
            g["lr"] = base * mult

        optim.zero_grad(set_to_none=True)
        ce_total = 0.0
        pen_total = 0.0
        for _ in range(accum):
            batch = train_sampler.batch(args.micro_batch_size).to(device)
            ce = cross_entropy(model, batch)
            loss = ce / accum
            if fisher is not None:
                # Recomputed every micro-batch and scaled by 1/accum, so the
                # gradient contribution per OPTIMIZER STEP is exactly
                # lambda * F * (theta - theta*), not accum times that.
                pen = ewc_penalty(model, theta_star, fisher)
                loss = loss + (0.5 * args.ewc_lambda / accum) * pen
                pen_total += float(pen.detach())
            loss.backward()
            ce_total += float(ce.detach())

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optim.step()

        if step % 20 == 0 or step == steps - 1:
            print(f"[ewc] step {step + 1}/{steps} ce={ce_total / accum:.4f} "
                  f"penalty={pen_total / max(1, accum):.4e} lr_mult={mult:.4f}",
                  flush=True)

    out = os.path.join(args.output_dir, "final-unsharded")
    os.makedirs(out, exist_ok=True)
    # Full state_dict (buffers included); theta_star and the penalty cover
    # named_parameters() only.
    torch.save(model.state_dict(), os.path.join(out, "model.pt"))
    if args.base_config_json and os.path.exists(args.base_config_json):
        # Downstream loaders look for config.json beside model.pt ("Could not
        # locate 'config.json' near unsharded model"). The architecture is
        # unchanged, so the base's config is correct here.
        shutil.copyfile(args.base_config_json, os.path.join(out, "config.json"))
    else:
        print("[ewc] WARNING: no base config.json copied; evals will not load "
              "this checkpoint", flush=True)
    print(f"[ewc] wrote {out}/model.pt", flush=True)


if __name__ == "__main__":
    main()

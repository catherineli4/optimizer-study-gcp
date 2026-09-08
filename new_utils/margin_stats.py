#!/usr/bin/env python3
"""Per-token margin statistics h, kappa, lambda on the held-out DCLM shard.

Implements the observables of "An Implicit Closed Form for the Critical Weight
Noise". At one validation position with gold token t and logits z_j = u_j^T x(W),
the margins and their two summary statistics are

    m_j := z_t - z_j   (j != t),   h := mean_j m_j,   kappa := sd_j m_j,
    lambda := min_j m_j.

The theory's payoff is that the critical weight noise factorises as

    sigma* = f(h / kappa) * ||x|| / ||M||_F ,   f strictly increasing,

with f fixed by (V, delta) alone. So h/kappa is the whole model-dependent part
of the brittleness ordering: a larger h/kappa means a more robust model, and two
optimizers can be ranked by this number without ever perturbing anything.
sigma* := 0 where lambda <= 0, so positions with lambda <= 0 are reported but
excluded from the aggregate.

Computing this naively would materialise a (V-1)-vector of margins per position
-- ~100K floats for every one of ~2M positions. It is never needed. Because z_t
does not depend on j:

    h      = z_t - mean_{j!=t} z_j
    kappa  = sd_{j!=t} z_j                  (the shift by z_t cancels: Lemma 5.3)
    lambda = z_t - max_{j!=t} z_j

so everything follows from four O(V) reductions over the logit row.

    python -m new_utils.margin_stats --config cfg.yaml --checkpoint model.pt \\
        --tokens /path/to/part-059/00004.npy --out margins.npz
"""

import argparse
import json
import os
from typing import Optional

import numpy as np
import torch
import yaml


def load_tokens(path: str, dtype: str = "uint32") -> np.memmap:
    """Raw token array. These files carry a .npy extension but have no npy
    header, so np.load misreads the leading tokens as a pickle header; olmo_core
    memmaps them flat, and so do we."""
    return np.memmap(path, dtype=np.dtype(dtype), mode="r")


def build_model(config_path: str, checkpoint: str, device: torch.device):
    from olmo_core.nn.transformer import TransformerConfig

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    if "model" not in cfg:
        raise KeyError(f"{config_path} has no 'model' section")
    model = TransformerConfig.from_dict(cfg["model"]).build(init_device="cpu")

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for key in ("model", "state_dict"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"checkpoint is missing parameters: {sorted(missing)[:8]}")
    if unexpected:
        print(f"[margins] ignoring {len(unexpected)} unexpected key(s)", flush=True)
    return model.to(device).eval(), cfg


@torch.no_grad()
def margin_stats(logits: torch.Tensor, gold: torch.Tensor):
    """h, kappa, lambda for each position of one batch.

    logits: (B, L, V) — any dtype, upcast internally.
    gold:   (B, L)    — the gold token id t at each position.

    Returns three (B, L) float64 tensors. Everything is a reduction over the
    vocabulary axis; the margin vector is never formed.
    """
    z = logits.double()
    V = z.shape[-1]
    n = V - 1                                    # competitors, j != t

    z_t = z.gather(-1, gold.unsqueeze(-1)).squeeze(-1)          # (B, L)

    # mean over j != t: drop the gold entry from the full-vocab sum.
    mean_excl = (z.sum(-1) - z_t) / n
    h = z_t - mean_excl

    # kappa = sd_{j!=t} z_j. Two-pass about mean_excl rather than
    # E[z^2] - E[z]^2: with |z| ~ 20 and V ~ 1e5 that difference cancels ~2
    # significant digits, and kappa enters as a ratio.
    dev = z - mean_excl.unsqueeze(-1)
    ss_excl = dev.pow(2).sum(-1) - (z_t - mean_excl).pow(2)
    kappa = (ss_excl / n).clamp_min(0).sqrt()

    # lambda = z_t - max_{j != t} z_j. Mask the gold entry rather than taking a
    # top-2, so ties at the gold logit are handled correctly.
    z_masked = z.scatter(-1, gold.unsqueeze(-1), float("-inf"))
    lam = z_t - z_masked.max(-1).values

    # argmax-correctness is NOT the same as lambda > 0: a tie at the gold logit
    # gives lambda == 0 while argmax may still return t. Ties are common because
    # FlashAttention forces a bf16 forward (~3 significant digits), so both are
    # reported and the gap is the tie count.
    correct = logits.argmax(-1) == gold

    return h, kappa, lam, correct


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="config.yaml with the model spec")
    ap.add_argument("--checkpoint", required=True, help="final-unsharded/model.pt")
    ap.add_argument("--tokens", required=True,
                    help="held-out DCLM shard (part-059/00004.npy)")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--sequence-length", type=int, default=4096)
    ap.add_argument("--instances", type=int, default=1024,
                    help="sequences to score; matches DCLM_HELDOUT_INSTANCES")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="sequences per forward; memory only")
    ap.add_argument("--token-dtype", default="uint32")
    ap.add_argument("--offset", type=int, default=0,
                    help="token offset into the shard, for disjoint splits")
    ap.add_argument("--save-per-token", action="store_true",
                    help="store every position's h/kappa/lambda, not just the summary")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = build_model(args.config, args.checkpoint, device)

    toks = load_tokens(args.tokens, args.token_dtype)
    L = args.sequence_length
    need = args.instances * L
    if args.offset + need > toks.shape[0]:
        raise ValueError(
            f"shard has {toks.shape[0]:,} tokens; need offset+{need:,}")
    print(f"[margins] {args.instances} seq x {L} tok from {args.tokens}", flush=True)

    hs, ks, ls, cs = [], [], [], []
    for start in range(0, args.instances, args.batch_size):
        b = min(args.batch_size, args.instances - start)
        off = args.offset + start * L
        chunk = np.asarray(toks[off:off + b * L], dtype=np.int64).reshape(b, L)
        ids = torch.from_numpy(chunk).to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(ids[:, :-1])
        logits = out.logits if hasattr(out, "logits") else out
        h, k, lam, corr = margin_stats(logits, ids[:, 1:])

        hs.append(h.float().cpu().numpy())
        ks.append(k.float().cpu().numpy())
        ls.append(lam.float().cpu().numpy())
        cs.append(corr.cpu().numpy())
        if (start // args.batch_size) % 20 == 0:
            print(f"[margins] {start + b}/{args.instances} sequences", flush=True)

    h = np.concatenate(hs).ravel()
    kappa = np.concatenate(ks).ravel()
    lam = np.concatenate(ls).ravel()
    correct = np.concatenate(cs).ravel()

    # sigma* = 0 where lambda <= 0 (the model is already wrong there), and the
    # closed form is stated only for lambda > 0, so the aggregate uses those.
    ok = lam > 0
    ratio = np.full_like(h, np.nan)
    good = ok & (kappa > 0)
    ratio[good] = h[good] / kappa[good]
    r = ratio[good]

    summary = {
        "n_positions": int(h.size),
        "n_lambda_positive": int(ok.sum()),
        "frac_lambda_positive": float(ok.mean()),
        # Cross-check: differs from frac_lambda_positive only by tied positions.
        "top1_accuracy": float(correct.mean()),
        "n_ties_at_gold": int((correct & (lam == 0)).sum()),
        "h_over_kappa_mean": float(r.mean()) if r.size else None,
        "h_over_kappa_median": float(np.median(r)) if r.size else None,
        "h_over_kappa_p10": float(np.percentile(r, 10)) if r.size else None,
        "h_over_kappa_p90": float(np.percentile(r, 90)) if r.size else None,
        "h_mean": float(h[good].mean()) if r.size else None,
        "kappa_mean": float(kappa[good].mean()) if r.size else None,
        "sequence_length": L,
        "instances": args.instances,
        "checkpoint": args.checkpoint,
        "tokens": args.tokens,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    arrays = {"summary": np.frombuffer(json.dumps(summary).encode(), dtype=np.uint8)}
    if args.save_per_token:
        arrays.update(h=h, kappa=kappa, lam=lam, h_over_kappa=ratio)
    np.savez_compressed(args.out, **arrays)
    with open(os.path.splitext(args.out)[0] + ".json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2), flush=True)
    print(f"[margins] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

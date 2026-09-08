#!/usr/bin/env python3
"""Run margin_stats over the tuned (best-LR) base of every chinchilla x optimizer
cell at one size.

Same base set the perturbation, replay and EWC sweeps use -- tuned_bases_for over
every chinchilla with a tuned LR -- so h/kappa lines up row-for-row with those
results. Per (size, chinchilla, optimizer) it writes

    {out_dir}/{size}/{run_name}.json      summary
    {out_dir}/{size}/{run_name}.npz       per-token arrays, with --save-per-token

and skips a cell whose .json already exists, so the sweep is resumable.

Checkpoints are staged one at a time and deleted after scoring: the full ladder
is ~74 models and the 600M ones are ~2.4 GB each.

    OPTIM_SIZE=60M python -m new_utils.margin_stats_sweep --instances 256
"""

import argparse
import json
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="/mnt/localssd/margins")
    ap.add_argument("--stage-dir", default="/mnt/localssd/tmp/margin-stage")
    ap.add_argument("--instances", type=int, default=256,
                    help="sequences per model; 1024 matches DCLM_HELDOUT_INSTANCES")
    ap.add_argument("--sequence-length", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--chinchillas", default="",
                    help="comma list; default every tuned chinchilla at this size")
    ap.add_argument("--save-per-token", action="store_true")
    args = ap.parse_args()

    # Importing the launcher initialises Project.config (remote paths, bucket).
    sys.argv = ["launcher", "launch", "margins"]
    import launch_jolmo.launcher  # noqa: F401
    import yaml
    from launch_jolmo.training import _build_model_spec, remote_path
    from launch_jolmo.pretraining_matrix import PT_LR, tuned_bases_for
    from launch_jolmo.sizes import active_profile

    size, _ = active_profile()
    wsd = PT_LR.get("wsd", {})
    chins = sorted(set(wsd.get("adamw", {})) | set(wsd.get("muon", {})))
    if args.chinchillas.strip():
        keep = {float(x) for x in args.chinchillas.replace(",", " ").split()}
        missing = keep - set(chins)
        if missing:
            raise ValueError(f"{size} has no tuned LR for "
                             f"{[f'{c:g}' for c in sorted(missing)]}; "
                             f"available {[f'{c:g}' for c in chins]}")
        chins = [c for c in chins if c in keep]

    bases = list(tuned_bases_for(chins))
    print(f"[sweep] {size}: {len(bases)} tuned base(s) over "
          f"chinchillas {[f'{c:g}' for c in chins]}", flush=True)

    out_dir = os.path.join(args.out_dir, size)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.stage_dir, exist_ok=True)

    # The held-out DCLM shard these runs are scored on -- the same one the
    # DCLM_heldout eval label uses, never trained on by any run.
    from launch_jolmo.pretraining_matrix import dclm_heldout_val_chunks
    shard_uri = dclm_heldout_val_chunks[0][1].uri
    # Reuse the copy the launcher's own evals already cached, if present --
    # this shard is ~16 GiB and re-fetching it per sweep is pure waste.
    import glob
    cached = glob.glob("/mnt/localssd/cache/datasets/train/*part-059/00004.npy")
    if cached:
        shard_local = cached[0]
        print(f"[sweep] using cached held-out shard {shard_local}", flush=True)
    else:
        shard_local = os.path.join(args.stage_dir, "dclm_heldout.npy")
        if not os.path.exists(shard_local):
            print("[sweep] staging held-out shard (~16 GiB, once)", flush=True)
            subprocess.check_call(["gsutil", "-m", "cp", shard_uri, shard_local])

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "margin_stats.py")
    done = failed = skipped = 0
    for i, base in enumerate(bases, 1):
        name = base.run_name
        out_json = os.path.join(out_dir, f"{name}.json")
        if os.path.exists(out_json):
            skipped += 1
            continue

        ckpt = os.path.join(args.stage_dir, "model.pt")
        cfg_path = os.path.join(args.stage_dir, "config.yaml")
        src = remote_path(base.relpath, "final-unsharded", "model.pt")
        print(f"\n[sweep] ({i}/{len(bases)}) {name}", flush=True)

        try:
            subprocess.check_call(["gsutil", "-q", "-m", "cp", src, ckpt])
        except subprocess.CalledProcessError:
            print(f"[sweep] MISSING checkpoint, skipping: {src}", flush=True)
            failed += 1
            continue

        # Rebuild the architecture spec from the model type, exactly as the
        # training config does, so it matches the checkpoint being loaded.
        with open(cfg_path, "w") as fh:
            yaml.safe_dump(
                {"model": _build_model_spec(base.model_type, base._vocab_size())}, fh)

        cmd = [sys.executable, script,
               "--config", cfg_path, "--checkpoint", ckpt,
               "--tokens", shard_local, "--out",
               os.path.join(out_dir, f"{name}.npz"),
               "--sequence-length", str(args.sequence_length),
               "--instances", str(args.instances),
               "--batch-size", str(args.batch_size)]
        if args.save_per_token:
            cmd.append("--save-per-token")
        rc = subprocess.call(cmd)
        os.path.exists(ckpt) and os.remove(ckpt)   # ~2.4 GB at 600M
        if rc == 0:
            done += 1
            with open(out_json) as fh:
                s = json.load(fh)
            print(f"[sweep] {name}  h/kappa={s['h_over_kappa_mean']:.4f} "
                  f"acc={s['top1_accuracy']:.4f}", flush=True)
        else:
            failed += 1
            print(f"[sweep] FAILED rc={rc}: {name}", flush=True)

    print(f"\n[sweep] {size}: {done} scored, {skipped} already present, "
          f"{failed} failed", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script to perturb model weights by adding Gaussian noise and save to GCS.

Takes a model from a GCS bucket, perturbs all parameters by epsilon ~ N(0, sigma^2),
and saves the perturbed model back to GCS.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def download_from_gcs(gcs_path: str, local_path: str) -> None:
    """Download a file from GCS to local path."""
    subprocess.check_call(["gsutil", "-m", "cp", gcs_path, local_path])


def upload_to_gcs(local_path: str, gcs_path: str) -> None:
    """Upload a local file to GCS."""
    subprocess.check_call(["gsutil", "-m", "cp", local_path, gcs_path])


def rsync_from_gcs(gcs_source: str, local_dest: str) -> None:
    """Rsync a directory from GCS to local path."""
    subprocess.check_call(["gsutil", "-m", "rsync", "-r", gcs_source, local_dest])


def rsync_to_gcs(local_source: str, gcs_dest: str) -> None:
    """Rsync a directory from local to GCS."""
    subprocess.check_call(["gsutil", "-m", "rsync", "-r", local_source, gcs_dest])


def copy_gcs_directory(gcs_source: str, gcs_dest: str, exclude: Optional[str] = None) -> None:
    """Copy a directory from one GCS location to another.

    ``exclude`` is a Python regex on the path relative to the source (gsutil
    rsync -x). The perturbation uses it to leave final-unsharded/model.pt OUT of
    the copy: the old behaviour copied the base's model.pt and then overwrote
    it, so an upload that failed in between left the UNPERTURBED weights at the
    perturbed path -- the exists-check then passed forever and the eval
    reported zero degradation (30M c8 adamw, gamma 0.05: md5 identical to base).
    """
    cmd = ["gsutil", "-m", "rsync", "-r"]
    if exclude:
        cmd += ["-x", exclude]
    subprocess.check_call(cmd + [gcs_source + "/", gcs_dest + "/"])

def perturb_state_dict(
    state_dict: dict,
    gamma: float,
    seed: Optional[int] = None,
    param_names: Optional[set] = None,
) -> dict:
    """
    Perturb parameters in a state_dict by adding Gaussian noise scaled by the
    weight's Frobenius norm: std = gamma * ||W||_F / sqrt(numel) (per tensor).

    Args:
        state_dict: Current model state_dict
        gamma: Relative Frobenius noise scale (‖noise‖_F / ‖W‖_F ≈ gamma)
        seed: Optional random seed
        param_names: If set, only these parameter names are perturbed; all
            other tensors are copied unchanged. Raises if any name is missing.

    Returns:
        Perturbed state dictionary
    """
    if param_names is not None:
        missing = sorted(set(param_names) - set(state_dict.keys()))
        if missing:
            raise KeyError(f"param_names not in state_dict: {missing}")

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    def _randn_like(tensor: torch.Tensor) -> torch.Tensor:
        if generator is None:
            return torch.randn_like(tensor)
        return torch.randn(
            tensor.shape,
            dtype=tensor.dtype,
            device=tensor.device,
            generator=generator,
        )

    perturbed_state = {}

    for name, W in state_dict.items():
        # Only perturb float parameters (and, optionally, a name allow-list).
        if W.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            perturbed_state[name] = W
            continue
        if param_names is not None and name not in param_names:
            perturbed_state[name] = W
            continue

        # Frobenius norm of weight
        s = torch.norm(W)     # ||W||_F

        # std = gamma * ||W||_F / sqrt(numel) = gamma * RMS(W). Dividing by
        # sqrt(m*d) makes the per-entry std track the typical entry magnitude,
        # so the total relative perturbation ||noise||_F / ||W||_F ~ gamma is
        # dimension-independent (comparable across tensors of any shape).
        std = gamma * s / (W.numel() ** 0.5)

        noise = _randn_like(W) * std
        perturbed_state[name] = W + noise

    return perturbed_state



def load_model_from_gcs(gcs_path: str, device: str = "cpu") -> dict:
    """
    Load a PyTorch model state_dict from GCS.
    
    Args:
        gcs_path: GCS path to the model file (e.g., gs://bucket/path/model.pt)
        device: Device to load the model on
    
    Returns:
        Model state dictionary
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        print(f"📥 Downloading model from {gcs_path}...")
        download_from_gcs(gcs_path, tmp_path)
        
        print(f"📂 Loading model state dict...")
        state_dict = torch.load(tmp_path, map_location=device)
        
        # Handle different formats: state_dict might be wrapped in a dict
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        
        return state_dict
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def save_model_to_gcs(state_dict: dict, gcs_path: str) -> None:
    """
    Save a PyTorch model state_dict to GCS.
    
    Args:
        state_dict: Model state dictionary
        gcs_path: GCS path to save the model (e.g., gs://bucket/path/model_perturbed.pt)
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        print(f"💾 Saving perturbed model locally...")
        torch.save(state_dict, tmp_path)
        
        print(f"☁️ Uploading perturbed model to {gcs_path}...")
        upload_to_gcs(tmp_path, gcs_path)
        print(f"✅ Successfully uploaded perturbed model to {gcs_path}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _param_tag(param_name: str) -> str:
    """Filesystem-safe tag for a state-dict key (dots → dashes)."""
    return param_name.replace(".", "-")


def get_perturbed_model_name(
    model_name: str,
    sigma: float,
    param_name: Optional[str] = None,
) -> str:
    """
    Generate perturbed model name with sigma (and optional single-param) suffix.

    Args:
        model_name: Original model name
        sigma: Noise scale γ (same as CLI --sigma)
        param_name: If set, only this matrix was perturbed; tag it in the name.

    Returns:
        e.g. "model_perturbed_2_00e-2" or
        "model_perturbed_2_00e-2_param_blocks-0-attention-w_q-weight"
    """
    sigma_str = f"{sigma:.2e}".replace("e-0", "e-").replace("e+0", "e+").replace(".", "_")
    name = f"{model_name}_perturbed_{sigma_str}"
    if param_name:
        name = f"{name}_param_{_param_tag(param_name)}"
    return name


def seed_subdir(seed: int) -> str:
    """Child directory name for one noise direction under a (base, γ) parent."""
    return f"seed_{seed:03d}"


def _parse_seeds_csv(raw: str) -> list:
    """Parse comma-separated integer seeds, e.g. ``0,1,2``."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("--seeds must be a non-empty comma-separated list of ints")
    return [int(p) for p in parts]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Perturb model weights with Gaussian noise and save to GCS"
    )
    parser.add_argument(
        "--gcs_dir",
        type=str,
        required=True,
        help="GCS directory path (e.g., gs://bucket/path/to/models)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name (directory name containing final-unsharded/model.pt)"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        required=True,
        help="Standard deviation of Gaussian noise (N(0, sigma^2))",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=64,
        help="Random seed for single-direction mode (ignored if --seeds is set)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds for multi-direction mode. Writes each under "
             "{perturbed_name}/seed_XXX/final-unsharded/. Example: 0,1,2,...,9",
    )
    parser.add_argument(
        "--output_gcs_dir",
        type=str,
        required=True,
        help="GCS directory path for storing perturbed model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load model on (default: cpu)",
    )
    parser.add_argument(
        "--param-name",
        type=str,
        default=None,
        help="If set, only this state-dict key is perturbed (others copied). "
             "Used for per-matrix forgetting sweeps.",
    )

    args = parser.parse_args()

    try:
        # Input: gcs_dir/model_name/final-unsharded/model.pt
        original_checkpoint_dir = f"{args.gcs_dir.rstrip('/')}/{args.model_name}"
        original_model_path = f"{original_checkpoint_dir}/final-unsharded/model.pt"

        perturbed_model_name = get_perturbed_model_name(
            args.model_name, args.sigma, param_name=args.param_name,
        )
        parent_dir = f"{args.output_gcs_dir.rstrip('/')}/{perturbed_model_name}"

        multi_seed = args.seeds is not None
        seeds = _parse_seeds_csv(args.seeds) if multi_seed else [args.seed]

        print(f"📋 Original checkpoint directory: {original_checkpoint_dir}")
        print(f"📋 Original model path: {original_model_path}")
        print(f"📋 Perturbed parent directory: {parent_dir}")
        print(f"📋 Seeds: {seeds}" + (" (multi-direction)" if multi_seed else " (single)"))
        if args.param_name:
            print(f"📋 Single-param perturbation: {args.param_name}")

        # Load base weights once; each seed gets an independent noise draw.
        print(f"📥 Loading model from {original_model_path}...")
        state_dict = load_model_from_gcs(original_model_path, device=args.device)
        allow = {args.param_name} if args.param_name else None

        for seed in seeds:
            if multi_seed:
                checkpoint_dir = f"{parent_dir}/{seed_subdir(seed)}"
            else:
                checkpoint_dir = parent_dir
            model_path = f"{checkpoint_dir}/final-unsharded/model.pt"

            print(f"📦 [{seed}] Copying checkpoint (minus model.pt) → {checkpoint_dir}...")
            copy_gcs_directory(original_checkpoint_dir, checkpoint_dir,
                               exclude=r"(^|/)final-unsharded/model\.pt$")

            print(f"🔧 [{seed}] Perturbing with sigma={args.sigma}, seed={seed}...")
            perturbed_state_dict = perturb_state_dict(
                state_dict, args.sigma, seed=seed, param_names=allow,
            )
            # Never publish weights that equal the base under a perturbed name.
            if args.sigma > 0 and all(
                    torch.equal(perturbed_state_dict[k], state_dict[k])
                    for k in state_dict):
                raise RuntimeError(
                    f"perturbation at sigma={args.sigma} left every tensor "
                    "unchanged; refusing to upload")

            print(f"💾 [{seed}] Uploading model.pt...")
            save_model_to_gcs(perturbed_state_dict, model_path)

        print("🎉 Done! Perturbed checkpoint(s) created successfully.")
        return 0

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())


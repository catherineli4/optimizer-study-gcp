"""Model-size profiles for the optimizer study.

A single ``OPTIM_SIZE`` environment variable selects which model-size study the
launcher operates on. Each profile bundles everything that differs between sizes:

  * ``project``      — the ``experiments`` project to ``Project.init`` (its
                       ``project.json`` sets ``remote_path`` = the GCS prefix, so
                       this is what routes artifacts to the right bucket dir).
  * ``model_type``   — key into ``training.MODEL_ARCHS`` (also the size tag baked
                       into every run name, e.g. ``MuonExpt3-0.1B-chinchilla-…``).
  * ``chinchillas``  — pretrain token-budget multipliers that exist for this size.
  * ``num_processes`` (optional) — default GPU/torchrun world size when
                       ``$OPTIM_NUM_PROCESSES`` is unset.

Default is the original 60M study, so existing commands are unchanged. Examples::

    OPTIM_SIZE=100M python -m launch_jolmo.launcher drylaunch cpt
    OPTIM_SIZE=300M OPTIM_NUM_PROCESSES=8 python -m launch_jolmo.launcher drylaunch pretrain-all-wsd

Both ``launcher.py`` (for ``Project.init``) and ``pretraining_matrix.py`` (for
``MODEL_TYPE`` / ``CHINCHILLAS`` / ``experiment_name``) read the same profile, so
they can never drift out of sync.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple


PROFILES: Dict[str, Dict] = {
    "30M": {
        # 0.03B study. Tuned PT LRs live in PT_LR_BY_MODEL["0.03B"] (adamw +
        # muon, chinchillas 1-16, measured from the completed sweeps).
        "project": "Optim-30M-tuning",
        "model_type": "0.03B",
        "chinchillas": [4, 8, 16, 32, 64],
        "num_processes": 2,
    },
    "60M": {
        "project": "Optim-60M-tuning",
        "model_type": "0.06B",
        "chinchillas": [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128],   # trimmed from [1..128] for the chin-1-8 CPT pass
        "num_processes": 8,
    },
    "100M": {
        # Optimal-LR 0.1B models. Chin 8/16/32 are the seq4k-bs1m jgai sweep
        # (new_utils/import_100m_models.py); chin 1–4 are earlier imports.
        "project": "Optim-100M-tuning",
        "model_type": "0.1B",
        "chinchillas": [4, 8],
        "num_processes": 2,
    },
    "300M": {
        # 0.3B study. No tuned PT LRs yet — pretraining_matrix falls back to
        # PT_LR_SWEEP until a table is filled in under PT_LR_BY_MODEL["0.3B"].
        "project": "Optim-300M-tuning",
        "model_type": "0.3B",
        "chinchillas": [1,2,4],
        "num_processes": 8,
    },
    "600M": {
        # 0.6B study. No tuned PT LRs yet — pretraining_matrix falls back to
        # PT_LR_SWEEP until the PT_LR_BY_MODEL["0.6B"] slots are filled in.
        "project": "Optim-600M-tuning",
        "model_type": "0.6B",
        "chinchillas": [0.5, 1],
        "num_processes": 8,
    },
}

DEFAULT_SIZE = "60M"
ENV_VAR = "OPTIM_SIZE"


def active_profile() -> Tuple[str, Dict]:
    """Return ``(size_key, profile_dict)`` selected by ``$OPTIM_SIZE`` (default 60M)."""
    size = os.environ.get(ENV_VAR, DEFAULT_SIZE)
    if size not in PROFILES:
        raise ValueError(
            f"{ENV_VAR}={size!r} is not a known size. Options: {sorted(PROFILES)}"
        )
    return size, PROFILES[size]

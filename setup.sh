#!/usr/bin/env bash
# Idempotent environment bootstrap for a fresh GCP GPU server. See SETUP.md.
# Usage: WANDB_API_KEY=<key> bash setup.sh
set -euo pipefail

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: run as  WANDB_API_KEY=<your-key> bash setup.sh" >&2
    exit 1
fi
if ! mountpoint -q /mnt/localssd; then
    echo "ERROR: /mnt/localssd not mounted — do SETUP.md step 1 first" >&2
    exit 1
fi

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-$HOME/experiments}"
[ -d "$REPO_DIR/JOLMo/src" ] || { echo "ERROR: JOLMo not cloned into $REPO_DIR/JOLMo" >&2; exit 1; }
[ -d "$EXPERIMENTS_DIR" ] || { echo "ERROR: experiments repo not at $EXPERIMENTS_DIR" >&2; exit 1; }

# --- Working dirs on the SSDs (root disk is tiny) --------------------------
mkdir -p /mnt/localssd/{tmp,hf,cache/{datasets,cached_path,training}} /mnt/localssd/data
# cached_path MUST resolve to the SSD no matter what env a launcher runs with.
mkdir -p "$HOME/.cache"
if [ ! -L "$HOME/.cache/cached_path" ]; then
    rm -rf "$HOME/.cache/cached_path"
    ln -s /mnt/localssd/cache/cached_path "$HOME/.cache/cached_path"
fi

# --- Miniconda on the SSD ---------------------------------------------------
if [ ! -x /mnt/localssd/miniconda3/bin/conda ]; then
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /mnt/localssd/tmp/miniconda.sh
    bash /mnt/localssd/tmp/miniconda.sh -b -p /mnt/localssd/miniconda3
fi
[ -e "$HOME/miniconda3" ] || ln -s /mnt/localssd/miniconda3 "$HOME/miniconda3"
source /mnt/localssd/miniconda3/etc/profile.d/conda.sh

# --- Env: python 3.11, torch 2.8 cu128, flash-attn, editable installs -------
if ! conda env list | grep -q "^optim-study "; then
    conda create -y -n optim-study python=3.11
fi
conda activate optim-study
TMPDIR=/mnt/localssd/tmp pip install --cache-dir /mnt/localssd/tmp/pip-cache \
    torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
TMPDIR=/mnt/localssd/tmp pip install --cache-dir /mnt/localssd/tmp/pip-cache \
    flash-attn --no-build-isolation
TMPDIR=/mnt/localssd/tmp pip install --cache-dir /mnt/localssd/tmp/pip-cache \
    -e "$REPO_DIR/JOLMo" -e "$EXPERIMENTS_DIR" \
    wandb google-cloud-storage cached-path transformers omegaconf py-spy

# --- Per-size project configs ----------------------------------------------
for entry in "Optim-30M-tuning" "Optim-60M-tuning" "Optim-100M-tuning" "Optim-300M-tuning" "Optim-600M-tuning"; do
    d="$HOME/.experiments/projects/$entry"
    mkdir -p "$d"
    cat > "$d/project.json" <<EOF
{
  "config": {
    "name": "$entry",
    "cluster": "babel",
    "code_path": "$REPO_DIR",
    "jolmo_path": "$REPO_DIR/JOLMo",
    "local_data_path": "/mnt/localssd/data",
    "local_cache_path": "/mnt/localssd/cache",
    "remote_path": "gs://cmu-gpucloud-catheri4/$entry",
    "wandb_api_key": "$WANDB_API_KEY"
  }
}
EOF
done

# --- Shell env ---------------------------------------------------------------
add_line() { grep -qxF "$1" "$HOME/.bashrc" || echo "$1" >> "$HOME/.bashrc"; }
add_line 'export CACHED_PATH_CACHE_ROOT=/mnt/localssd/cache/cached_path'
add_line 'export TMPDIR=/mnt/localssd/tmp'
add_line 'export HF_HOME=/mnt/localssd/hf'
add_line 'source /mnt/localssd/miniconda3/etc/profile.d/conda.sh'

echo
echo "Done. Next:  gcloud auth login && gcloud auth application-default login"
echo "Then verify: source ~/.bashrc && conda activate optim-study && cd $REPO_DIR && OPTIM_SIZE=60M python -m launch_jolmo.launcher drylaunch cpt"

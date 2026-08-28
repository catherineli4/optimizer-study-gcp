# Setup — new GCP GPU server

Gets a fresh GCP GPU VM (tested: 8×H100, Debian, two local NVMe SSDs) from zero
to running sweeps. Everything of value lives on GCS
(`gs://cmu-gpucloud-catheri4/Optim-*-tuning`), so servers are disposable.

## 0. Prerequisites

- NVIDIA driver + CUDA working (`nvidia-smi`). On a GCP Deep Learning VM image
  this is preinstalled; otherwise install the driver first.
- GitHub access to: `catherineli4/optimizer-study-gcp`, `catherineli4/JOLMo`,
  `jakespringer/experiments`.
- A W&B API key and `gcloud` authenticated with access to the GCS buckets.

## 1. Disks (IMPORTANT — the root disk is ~10G and WILL fill otherwise)

Two local NVMe SSDs, formatted and mounted as:

```bash
sudo mkfs.ext4 -F /dev/nvme1n1 && sudo mkdir -p /mnt/localssd  && sudo mount /dev/nvme1n1 /mnt/localssd
sudo mkfs.ext4 -F /dev/nvme2n1 && sudo mkdir -p /mnt/localssd/data && sudo mount /dev/nvme2n1 /mnt/localssd/data
sudo chown -R $USER:$USER /mnt/localssd
```

Local SSDs on GCP are **ephemeral** — recreate this layout after every VM stop.
`setup.sh` (step 3) creates the working dirs and symlinks under them.

## 2. Clone the code

```bash
cd ~
git clone https://github.com/catherineli4/optimizer-study-gcp.git
git clone https://github.com/catherineli4/JOLMo.git optimizer-study-gcp/JOLMo
git clone https://github.com/jakespringer/experiments.git
```

Layout must be exactly: `~/optimizer-study-gcp/JOLMo` and `~/experiments`
(the repo dir name itself doesn't matter — `setup.sh` resolves its own path —
but JOLMo must live inside it and `experiments` beside it).

## 3. Environment

```bash
cd ~/optimizer-study-gcp
WANDB_API_KEY=<your-key> bash setup.sh
```

`setup.sh` is idempotent and does:
- Miniconda into `/mnt/localssd/miniconda3` (root disk is too small),
  symlinked at `~/miniconda3`
- conda env `optim-study` (python 3.11), torch 2.8.0+cu128, flash-attn,
  `pip install -e ./JOLMo` and `pip install -e ~/experiments`
- Working dirs on the SSDs + the `~/.cache/cached_path` symlink
  (checkpoint-cache MUST stay off the root disk)
- `~/.experiments/projects/<Project>/project.json` for every model size
  (60M/100M/300M/600M), with your W&B key
- `~/.bashrc` exports: `CACHED_PATH_CACHE_ROOT`, `TMPDIR`, `HF_HOME`

Then authenticate GCS (both are needed — gsutil AND the python client):

```bash
gcloud auth login
gcloud auth application-default login
```

## 4. Verify

```bash
source ~/.bashrc && conda activate optim-study
cd ~/optimizer-study-gcp
OPTIM_SIZE=60M python -m launch_jolmo.launcher drylaunch cpt   # enumerates + GCS checks, no training
```

If the drylaunch prints artifact listings and "already exist" skips, everything
(imports, GCS auth, project configs) works.

## 5. Running

```bash
OPTIM_SIZE=<30M|60M|100M|300M|600M> python -m launch_jolmo.launcher runlocal <stage...>
```

- `runlocal` schedules across all local GPUs; `drylaunch` previews.
- Stage names: see `launch_jolmo/launcher.py` (`executor.stage(...)` lines).
  Highlights: `cpt` (finetune matrix), `eval-cpt`, `cpt-all-lrs`,
  `pt60m4-cpt-full-grid`, `lrbs-*` (LR×batch sweeps, `docs/lr_bs_sweep.md`),
  `ft-*` (finetune sweeps w/ DCLM replay, `launch_jolmo/ft_sweep.py`).
- Detached run that survives your laptop/SSH closing:
  `setsid nohup python -m launch_jolmo.launcher runlocal <stage> --no-dashboard > /mnt/localssd/run.log 2>&1 < /dev/null & disown`
- Everything is idempotent: artifacts existing on GCS are skipped, so rerunning
  a stage after failures only picks up what's missing.

## 6. Operational notes (learned the hard way)

- **Disk pressure**: task scripts clean their own staging after upload, but the
  `cached_path` checkpoint cache has NO eviction. For big sweeps run a pruner:
  `while true; do find /mnt/localssd/cache/cached_path -maxdepth 1 -type f -mmin +30 -delete; sleep 180; done &`
- A run that sits idle with one ancient `gsutil` child is wedged — kill and
  relaunch (idempotent).
- One launcher at a time: concurrent launchers double-book GPUs and multiply
  disk staging.
- 60M models: `OPTIM_NUM_PROCESSES=2` (4 concurrent 2-GPU runs) has ~4× the
  aggregate throughput of one 8-GPU run.

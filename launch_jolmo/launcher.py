# ---------------------------------------------------------------------------
# Project and imports
# ---------------------------------------------------------------------------

from experiments import Project, SlurmExecutor

# Model-size profile selected by $OPTIM_SIZE (default 60M). The project's
# project.json sets remote_path = the GCS prefix, so this routes all artifacts to
# the right bucket dir. pretraining_matrix reads the same profile for MODEL_TYPE /
# CHINCHILLAS, so they stay in sync. Examples:
#   OPTIM_SIZE=100M python -m launch_jolmo.launcher <cmd> <stage>
#   OPTIM_SIZE=300M python -m launch_jolmo.launcher <cmd> <stage>
from launch_jolmo.sizes import active_profile

_SIZE, _PROFILE = active_profile()
Project.init(_PROFILE["project"])

from launch_jolmo.pretraining_matrix import (
    pretrain_adamw_wsd,
    pretrain_adamw_cosine,
    pretrain_muon_wsd,
    pretrain_muon_cosine,
    pretrain_all_wsd,  # all LR sweep models (needed as dependency for eval-pretrain-all)
)
from launch_jolmo.pt_sweep_60m_chin4 import (
    pt60m4_lr_sweep,
    pt60m4_wd_sweep,
    pt60m4_bs_sweep,
    pt60m4_all,
    pt60m4_cpt_lr_sweep,
    pt60m4_cpt_wd_sweep,
    pt60m4_cpt_wd_sweep_muon,
    pt60m4_cpt_bs_sweep,
    pt60m4_cpt_all,
    pt60m4_cpt_evals,
    pt60m4_full_grid,
    pt60m4_cpt_full_grid,
    pt60m4_cpt_full_grid_evals,
    pt60m4_lr_sweep_evals,
    pt60m4_wd_sweep_evals,
    pt60m4_bs_sweep_evals,
    pt60m4_all_evals,
)
from launch_jolmo.pretraining_matrix import (
    cpt_adamw_models,
    cpt_muon_models,
    cpt_muon_pretrain_adamw_ft,
    cpt_adamw_pretrain_muon_ft,
    cpt_models,
    cpt_all_models,
    cpt_all_lrs_models,
    cpt_all_lrs_evals,        # CPT over every TRAINED pretrained model (full LR sweep)
    cpt_all_adamw_models,
    cpt_all_muon_models,
    cpt_all_bases,         # the discovered base JolmoModels (for dependency resolution)
    muon_sweep_models,     # muon CPT across alpha = muon→adamw LR ratio
    muon_sweep_evals,
    perturbed_adamw_models,
    perturbed_muon_models,
    multiseed_perturbed_adamw_models,
    multiseed_perturbed_muon_models,
    interpolated_models,
    pretrain_adamw_evals,
    pretrain_muon_evals,
    pretrain_all_wsd_evals,
    cpt_evals,
    cpt_muon_pretrain_adamw_ft_evals,
    cpt_adamw_pretrain_muon_ft_evals,
    cpt_all_evals,
    cpt_all_adamw_evals,
    cpt_all_muon_evals,
    perturbed_adamw_evals,
    perturbed_muon_evals,
    multiseed_perturbed_adamw_evals,
    multiseed_perturbed_muon_evals,
    interpolated_evals,
    divergence_bases,      # discovered base JolmoModels (for dependency resolution)
    divergence_adamw_evals,
    divergence_muon_evals,
    divergence_all_evals,
    divergence_13b_adamw_evals,
    divergence_13b_muon_evals,
    divergence_13b_all_evals,
    divergence_7b_adamw_evals,
    divergence_7b_muon_evals,
    divergence_7b_all_evals,
    ce_loss_teacher_evals,
    ce_loss_adamw_evals,
    ce_loss_muon_evals,
    ce_loss_all_evals,
    logit_perturb_adamw_evals,
    logit_perturb_muon_evals,
    logit_perturb_all_evals,
    logit_perturb_kl_adamw_evals,
    logit_perturb_kl_muon_evals,
    logit_perturb_kl_all_evals,
    logit_cosine_evals,
    logit_angle_bin_evals,
    logit_angle_perturb_evals,
    weight_angle_perturb_evals,
    c4_divergence_cpt_bases,
    c4_divergence_cpt_muon_ft_evals,
    c4_divergence_cpt_adamw_evals,
    c4_divergence_cpt_all_evals,
    c4_divergence_pretrain_evals,
    c4_divergence_perturbed_adamw_evals,
    c4_divergence_perturbed_muon_evals,
    c4_divergence_perturbed_all_evals,
    divergence_cpt_bases,
    divergence_cpt_adamw_evals,
    divergence_cpt_muon_evals,
    divergence_cpt_all_evals,
    divergence_perturb_bases,
    divergence_perturb_adamw_evals,
    divergence_perturb_muon_evals,
    divergence_perturb_all_evals,
    sharpness_adamw_evals,
    sharpness_muon_evals,
    sharpness_all_evals,
    maxeig_adamw_evals,
    maxeig_muon_evals,
    maxeig_all_evals,
    spectrum_adamw_evals,
    spectrum_muon_evals,
    spectrum_all_evals,
    forgetting_sharpness_evals,
    pretrain_associative_facts_adamw,
    pretrain_associative_facts_muon,
    pretrain_associative_facts_all,
)


# ---------------------------------------------------------------------------
# Cluster setup
# ---------------------------------------------------------------------------

if Project.config.cluster == "orchard":
    setup_command = "; ".join(
        [
            "source /home/jspringe/.bashrc",
            "source /home/jspringe/.secrets",
            "source /home/jspringe/env/train/bin/activate",
        ]
    )
elif Project.config.cluster == "babel":
    setup_command = "; ".join(
        [
            "source ~/miniconda3/etc/profile.d/conda.sh",
            "conda activate optim-study",
        ]
    )
else:
    raise ValueError(f"Unknown cluster: {Project.config.cluster}")


# ---------------------------------------------------------------------------
# Stage registration
# ---------------------------------------------------------------------------

executor = SlurmExecutor(setup_command=setup_command)

# --- Pretraining (DCLM) ---
executor.stage("pretrain-adamw-wsd",    pretrain_adamw_wsd)
executor.stage("pretrain-adamw-cosine", pretrain_adamw_cosine)
executor.stage("pretrain-muon-wsd",     pretrain_muon_wsd)
executor.stage("pretrain-muon-cosine",  pretrain_muon_cosine)
executor.stage("pretrain-all-wsd",      pretrain_all_wsd)

# --- 60M 4-chinchilla PT Sweep (PTSweep60M-*; run these with OPTIM_SIZE=60M) ---
# Staged: LR first, then WD at the winning LR, then batch size at both winners.
executor.stage("pt60m4-lr-sweep",       pt60m4_lr_sweep)
executor.stage("pt60m4-wd-sweep",       pt60m4_wd_sweep)
executor.stage("pt60m4-bs-sweep",       pt60m4_bs_sweep)
executor.stage("pt60m4-all",            pt60m4_all)
executor.stage("pt60m4-cpt-lr-sweep",    pt60m4_cpt_lr_sweep)
executor.stage("pt60m4-cpt-wd-sweep",    pt60m4_cpt_wd_sweep)
executor.stage("pt60m4-cpt-wd-sweep-muon", pt60m4_cpt_wd_sweep_muon)  # muon-pretrained bases only
executor.stage("pt60m4-cpt-bs-sweep",    pt60m4_cpt_bs_sweep)
executor.stage("pt60m4-cpt-all",         pt60m4_cpt_all)
executor.stage("pt60m4-cpt-evals",       pt60m4_cpt_evals)
executor.stage("pt60m4-full-grid",       pt60m4_full_grid)
executor.stage("pt60m4-cpt-full-grid",   pt60m4_cpt_full_grid)
executor.stage("pt60m4-cpt-full-grid-evals", pt60m4_cpt_full_grid_evals)
# Pretrain evals for the swept models (held-out DCLM + diversity-v2 val sets).
executor.stage("pt60m4-lr-sweep-evals",  pt60m4_lr_sweep_evals)
executor.stage("pt60m4-wd-sweep-evals",  pt60m4_wd_sweep_evals)
executor.stage("pt60m4-bs-sweep-evals",  pt60m4_bs_sweep_evals)
executor.stage("pt60m4-all-evals",       pt60m4_all_evals)

# --- Associative-facts pretrain (<bos> v[126] r u[126] <eos>+pad, CE only on u) ---
executor.stage("pretrain-associative-facts-adamw", pretrain_associative_facts_adamw)
executor.stage("pretrain-associative-facts-muon",  pretrain_associative_facts_muon)
executor.stage("pretrain-associative-facts",       pretrain_associative_facts_all)

# --- CPT (finetune the DCLM-pretrained models on datasets) ---
executor.stage("cpt",                  cpt_models)
executor.stage("cpt-adamw",            cpt_adamw_models)
executor.stage("cpt-muon",             cpt_muon_models)
executor.stage("cpt-muon-adamw-ft",    cpt_muon_pretrain_adamw_ft)
executor.stage("cpt-adamw-muon-ft",    cpt_adamw_pretrain_muon_ft)
# CPT over the full LR sweep (all trained pretrained models, discovered from GCS).
executor.stage("cpt-all",              cpt_all_models)
executor.stage("cpt-all-adamw",        cpt_all_adamw_models)
executor.stage("cpt-all-muon",         cpt_all_muon_models)
# The discovered base models the CPT runs depend on (registered so the executor's
# identity-based dependency check resolves; not retrained — they exist on GCS).
executor.stage("cpt-all-bases",        cpt_all_bases)
executor.stage("cpt-all-lrs",          cpt_all_lrs_models)
executor.stage("eval-cpt-all-lrs",     cpt_all_lrs_evals)

# LR x batch-size cross-product sweeps (launch_jolmo/lr_bs_sweep.py)
from launch_jolmo.lr_bs_sweep import SWEEPS as LRBS_SWEEPS
for _sweep in LRBS_SWEEPS:
    executor.stage(_sweep.label, _sweep.models())
    executor.stage(f"{_sweep.label}-evals", _sweep.evals())

# Finetuning cross-product sweeps (launch_jolmo/ft_sweep.py)
from launch_jolmo.ft_sweep import SWEEPS as FT_SWEEPS
for _sweep in FT_SWEEPS:
    executor.stage(_sweep.label, _sweep.models())
    executor.stage(f"{_sweep.label}-evals", _sweep.evals())
# Umbrella groups: every ft- sweep (one per 60M lr_bs pretrained base) at once.
executor.stage_group("ft-lrbs-60m",       tuple(_s.label for _s in FT_SWEEPS))
executor.stage_group("ft-lrbs-60m-evals", tuple(f"{_s.label}-evals" for _s in FT_SWEEPS))
# The FT base models (registered, like cpt-all-bases, so the executor's
# identity-based dependency check resolves; MuonExpt3-aliased bases exist on
# GCS and are never retrained). Deliberately NOT in any umbrella group.
executor.stage("ft-lrbs-bases", [_s.base for _s in FT_SWEEPS])
# Just the chinchilla-0.25 bases (the historical 48-model sweep).
executor.stage_group(
    "ft-lrbs-60m-c0.25",
    tuple(_s.label for _s in FT_SWEEPS if "-chinchilla-0.25-" in _s.base.run_name))
executor.stage_group(
    "ft-lrbs-60m-c0.25-evals",
    tuple(f"{_s.label}-evals" for _s in FT_SWEEPS if "-chinchilla-0.25-" in _s.base.run_name))
# Muon alpha sweep: CPT muon models across alpha = muon→adamw LR ratio.
executor.stage("muon-sweep",           muon_sweep_models)

# --- Gaussian weight perturbation of the DCLM-pretrained models ---
executor.stage("perturb-adamw",        perturbed_adamw_models)
executor.stage("perturb-muon",         perturbed_muon_models)
# Multi-direction: 10 seeds under PerturbedModel/{base}_perturbed_{γ}/seed_XXX/
executor.stage("perturb-multiseed-adamw", multiseed_perturbed_adamw_models)
executor.stage("perturb-multiseed-muon",  multiseed_perturbed_muon_models)

# --- Pretrained <-> finetuned weight interpolation (alpha*pt + (1-alpha)*ft) ---
executor.stage("interpolate",          interpolated_models)

# --- Evaluation stages (ModelEvaluation: validation loss) ---
executor.stage("eval-pretrain-adamw", pretrain_adamw_evals)
executor.stage("eval-pretrain-muon",  pretrain_muon_evals)
executor.stage("eval-pretrain-all",   pretrain_all_wsd_evals)
executor.stage("eval-cpt",            cpt_evals)
executor.stage("eval-cpt-muon-adamw-ft", cpt_muon_pretrain_adamw_ft_evals)
executor.stage("eval-cpt-adamw-muon-ft", cpt_adamw_pretrain_muon_ft_evals)
executor.stage("eval-cpt-all",        cpt_all_evals)
executor.stage("eval-cpt-all-adamw",  cpt_all_adamw_evals)
executor.stage("eval-cpt-all-muon",   cpt_all_muon_evals)
executor.stage("eval-muon-sweep",     muon_sweep_evals)
executor.stage("eval-perturb-bases",  perturbed_adamw_models + perturbed_muon_models)  # dependency only
executor.stage("eval-perturb-adamw",  perturbed_adamw_evals)
executor.stage("eval-perturb-muon",   perturbed_muon_evals)
executor.stage("eval-perturb-multiseed-bases", multiseed_perturbed_adamw_models + multiseed_perturbed_muon_models)
executor.stage("eval-perturb-multiseed-adamw", multiseed_perturbed_adamw_evals)
executor.stage("eval-perturb-multiseed-muon",  multiseed_perturbed_muon_evals)
executor.stage("eval-interpolate",    interpolated_evals)

# --- Per-token divergence vs. reference OLMo 2 (KL/JSD on DCLM heldout) stages ---
executor.stage("divergence-bases",       divergence_bases)   # dependency resolution only (not retrained)
executor.stage("divergence-adamw",       divergence_adamw_evals)      # vs. OLMo 2 32B
executor.stage("divergence-muon",        divergence_muon_evals)       # vs. OLMo 2 32B
executor.stage("divergence-all",         divergence_all_evals)        # vs. OLMo 2 32B
executor.stage("divergence-13b-adamw",   divergence_13b_adamw_evals)  # vs. OLMo 2 13B
executor.stage("divergence-13b-muon",    divergence_13b_muon_evals)   # vs. OLMo 2 13B
executor.stage("divergence-13b-all",     divergence_13b_all_evals)    # vs. OLMo 2 13B
executor.stage("divergence-7b-adamw",    divergence_7b_adamw_evals)   # vs. OLMo 2 7B (pretrain)
executor.stage("divergence-7b-muon",     divergence_7b_muon_evals)    # vs. OLMo 2 7B (pretrain)
executor.stage("divergence-7b-all",      divergence_7b_all_evals)     # vs. OLMo 2 7B (pretrain)

# --- Per-token CE loss on C4_val (legacy 1B-reference pipeline) ---
executor.stage("ce-loss-bases",   divergence_bases)   # dependency resolution only
executor.stage("ce-loss-teacher", ce_loss_teacher_evals)
executor.stage("ce-loss-adamw",  ce_loss_adamw_evals)
executor.stage("ce-loss-muon",   ce_loss_muon_evals)
executor.stage("ce-loss-all",    ce_loss_all_evals)

# --- Logit perturbation CE vs σ (relative ℓ₂ noise on per-token logits) ---
executor.stage("logit-perturb-bases", divergence_bases)  # dependency resolution only
executor.stage("logit-perturb-adamw", logit_perturb_adamw_evals)
executor.stage("logit-perturb-muon",  logit_perturb_muon_evals)
executor.stage("logit-perturb-all",   logit_perturb_all_evals)

# --- Logit perturbation KL(Q‖P) vs σ (Q = OLMo-2 1B) ---
executor.stage("logit-perturb-kl-bases", divergence_bases)
executor.stage("logit-perturb-kl-adamw", logit_perturb_kl_adamw_evals)
executor.stage("logit-perturb-kl-muon",  logit_perturb_kl_muon_evals)
executor.stage("logit-perturb-kl-all",   logit_perturb_kl_all_evals)

# --- Per-token logit cosine similarity: adamw vs muon (paired by chinchilla) ---
executor.stage("logit-cosine-bases", divergence_bases)
executor.stage("logit-cosine",       logit_cosine_evals)

# --- Angle-binned metrics (margin / NLL / KL / freq) + examples ---
executor.stage("logit-angle-bins-bases", divergence_bases)
executor.stage("logit-angle-bins",       logit_angle_bin_evals)

# --- Angle-bin stratified logit perturbation (ΔNLL distributions) ---
executor.stage("logit-angle-perturb-bases", divergence_bases)
executor.stage("logit-angle-perturb",       logit_angle_perturb_evals)

# --- Angle-bin stratified weight perturbation (Gaussian γ · ‖W‖_F/√n) ---
executor.stage("weight-angle-perturb-bases", divergence_bases)
executor.stage("weight-angle-perturb",       weight_angle_perturb_evals)

# --- Per-token C4_val KL/JSD vs 1B (legacy pipeline; pretrain + CPT + perturbed) ---
executor.stage("c4-divergence-cpt-bases",      c4_divergence_cpt_bases)       # dependency only
executor.stage("c4-divergence-cpt-muon-adamw-ft", c4_divergence_cpt_muon_ft_evals)
executor.stage("c4-divergence-cpt-adamw",      c4_divergence_cpt_adamw_evals)
executor.stage("c4-divergence-cpt-all",        c4_divergence_cpt_all_evals)
executor.stage("c4-divergence-pretrain",       c4_divergence_pretrain_evals)
executor.stage("c4-divergence-perturb-bases",  perturbed_adamw_models + perturbed_muon_models)  # dependency only
executor.stage("c4-divergence-perturb-adamw",  c4_divergence_perturbed_adamw_evals)
executor.stage("c4-divergence-perturb-muon",   c4_divergence_perturbed_muon_evals)
executor.stage("c4-divergence-perturb-all",    c4_divergence_perturbed_all_evals)

# --- Per-token KL on DCLM heldout: CPT chin-64 gsm8k (all CPT LRs × opts) ---
executor.stage("divergence-cpt-bases",   divergence_cpt_bases)  # dependency only
executor.stage("divergence-cpt-adamw",   divergence_cpt_adamw_evals)
executor.stage("divergence-cpt-muon",    divergence_cpt_muon_evals)
executor.stage("divergence-cpt-all",     divergence_cpt_all_evals)

# --- Per-token KL on DCLM heldout: Gaussian-perturbed chin-64 ---
executor.stage("divergence-perturb-bases",  divergence_perturb_bases)  # dependency only
executor.stage("divergence-perturb-adamw",  divergence_perturb_adamw_evals)
executor.stage("divergence-perturb-muon",   divergence_perturb_muon_evals)
executor.stage("divergence-perturb-all",    divergence_perturb_all_evals)

# --- Hessian sharpness (Lanczos max-eig + Hutch++ trace; CF port) ---
executor.stage("sharpness-adamw",   sharpness_adamw_evals)
executor.stage("sharpness-muon",    sharpness_muon_evals)
executor.stage("sharpness-all",     sharpness_all_evals)
executor.stage("forgetting-sharpness", forgetting_sharpness_evals)

# --- Hessian top eigenvalue only (Lanczos; no Hutch++ / no full spectrum) ---
executor.stage("maxeig-adamw",      maxeig_adamw_evals)
executor.stage("maxeig-muon",       maxeig_muon_evals)
executor.stage("maxeig-all",        maxeig_all_evals)

# --- Hessian spectral density (stochastic Lanczos quadrature) ---
executor.stage("spectrum-adamw",    spectrum_adamw_evals)
executor.stage("spectrum-muon",     spectrum_muon_evals)
executor.stage("spectrum-all",      spectrum_all_evals)

# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    executor.auto_cli()

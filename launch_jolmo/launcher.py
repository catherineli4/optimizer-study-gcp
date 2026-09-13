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
    ewc_models, ewc_evals, ewc_bases,
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
    c8_adamw_bs1m_models,
    c8_adamw_bs1m_evals,
    adamw_bs1m_models,
    adamw_bs1m_evals,
    bs1m_models,
    bs1m_evals,
    bs_any_models,
    bs_any_evals,
    cpt_wide_bases,
    cpt_wide_models,
    cpt_wide_evals,
    perturb_wide_bases,
    perturb_wide_models,
    perturb_wide_evals,
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
# Every existing c8 adamw pretrain at bs1M/wd0.1 (both naming schemas), and its
# DCLM-heldout eval. Bases registered so dependency resolution finds them.
executor.stage("c8-adamw-bs1m",         c8_adamw_bs1m_models)
executor.stage("c8-adamw-bs1m-evals",   c8_adamw_bs1m_evals)
# Same, across every chinchilla (0.25 - 128).
executor.stage("adamw-bs1m",            adamw_bs1m_models)
executor.stage("adamw-bs1m-evals",      adamw_bs1m_evals)
# Both optimizers, every chinchilla, whatever $OPTIM_SIZE selects.
executor.stage("bs1m",                  bs1m_models)
executor.stage("bs1m-evals",            bs1m_evals)
# Any batch size (bs1M/2M/4M...), discovered from the GCS listing.
executor.stage("bs-any",                bs_any_models)
executor.stage("bs-any-evals",          bs_any_evals)

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
# Tuned-LR CPT across chinchillas 0.25-32 (wider than the profile's list).
# Bases registered so dependency resolution finds them; they are never retrained.
executor.stage("cpt-wide-bases",       cpt_wide_bases)
executor.stage("cpt-wide",             cpt_wide_models)
executor.stage("cpt-wide-evals",       cpt_wide_evals)
executor.stage("cpt-adamw",            cpt_adamw_models)
executor.stage("cpt-muon",             cpt_muon_models)
# CPT of the tuned bases at integer chinchillas 2/4/8 only (skips the
# fractional rows whose bases live under the PTSweep naming).
_CPT_C248_TAGS = ("-chinchilla-2-", "-chinchilla-4-", "-chinchilla-8-")
cpt_c248_models = [m for m in cpt_models if any(t in m.run_name for t in _CPT_C248_TAGS)]
executor.stage("cpt-c2-8",             cpt_c248_models)
executor.stage("cpt-c2-8-evals", [
    _e for _e in cpt_evals if any(t in _e.model.run_name for t in _CPT_C248_TAGS)])
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
from launch_jolmo.lr_bs_sweep import HI5_SWEEPS as LRBS_HI5_SWEEPS
for _sweep in LRBS_SWEEPS:
    executor.stage(_sweep.label, _sweep.models())
    executor.stage(f"{_sweep.label}-evals", _sweep.evals())
# Above-optimal bs1M subsets (lrbs-...-hi5): 5 next LRs above each tuned optimal.
for _sweep in LRBS_HI5_SWEEPS:
    executor.stage(_sweep.label, _sweep.models())
    executor.stage(f"{_sweep.label}-evals", _sweep.evals())
executor.stage_group("lrbs-hi5-60m", tuple(_s.label for _s in LRBS_HI5_SWEEPS))
executor.stage_group("lrbs-hi5-60m-evals",
                     tuple(f"{_s.label}-evals" for _s in LRBS_HI5_SWEEPS))
# Just chinchillas 1-16 (excludes the fractional 0.25/0.5 budgets).
_hi5_c1_16 = tuple(_s for _s in LRBS_HI5_SWEEPS if _s.chinchilla in (1, 2, 4, 8, 16))
executor.stage_group("lrbs-hi5-60m-c1-16", tuple(_s.label for _s in _hi5_c1_16))
executor.stage_group("lrbs-hi5-60m-c1-16-evals",
                     tuple(f"{_s.label}-evals" for _s in _hi5_c1_16))
# Muon LR sweep, chinchillas 1-8 at bs1M, adamw component pinned to the retuned
# adamw optima. Stages: lrbs-0.06B-c<chin>-muon-bs1m (+ -evals).
from launch_jolmo.lr_bs_sweep import MUON_C18_SWEEPS as _MUON_C18
for _sweep in _MUON_C18:
    executor.stage(_sweep.label, _sweep.models())
    executor.stage(f"{_sweep.label}-evals", _sweep.evals())
executor.stage_group("lrbs-muon-c1-8", tuple(_s.label for _s in _MUON_C18))
executor.stage_group("lrbs-muon-c1-8-evals",
                     tuple(f"{_s.label}-evals" for _s in _MUON_C18))
# 30M muon LR sweep, chinchillas 1-2 at bs1M, adamw component pinned to the
# tuned adamw (run with OPTIM_SIZE=30M).
from launch_jolmo.lr_bs_sweep import MUON_30M_C12_SWEEPS as _MUON_30M
for _sweep in _MUON_30M:
    executor.stage(_sweep.label, _sweep.models())
    executor.stage(f"{_sweep.label}-evals", _sweep.evals())
executor.stage_group("lrbs-muon-30m-c1-2", tuple(_s.label for _s in _MUON_30M))
executor.stage_group("lrbs-muon-30m-c1-2-evals",
                     tuple(f"{_s.label}-evals" for _s in _MUON_30M))
# c16/c32 muon (adamw component 1e-2) reference-batch cells — models exist on
# GCS; stages exist mainly so their evals can run.
from launch_jolmo.lr_bs_sweep import LRBS_C1632_MUON_SWEEPS as _C1632
for _sweep in _C1632:
    executor.stage(_sweep.label, _sweep.models())
    executor.stage(f"{_sweep.label}-evals", _sweep.evals())
executor.stage_group("lrbs-c1632-evals", tuple(f"{_s.label}-evals" for _s in _C1632))
# 100M above-optimal subsets, chinchillas 2 and 4 only (run with OPTIM_SIZE=100M).
from launch_jolmo.lr_bs_sweep import HI5_100M_SWEEPS as LRBS_HI5_100M_SWEEPS
for _sweep in LRBS_HI5_100M_SWEEPS:
    executor.stage(_sweep.label, _sweep.models())
    executor.stage(f"{_sweep.label}-evals", _sweep.evals())
executor.stage_group("lrbs-hi5-100m", tuple(_s.label for _s in LRBS_HI5_100M_SWEEPS))
executor.stage_group("lrbs-hi5-100m-evals",
                     tuple(f"{_s.label}-evals" for _s in LRBS_HI5_100M_SWEEPS))
# 600M above-optimal subsets, chinchillas 0.5 and 1 (run with OPTIM_SIZE=600M).
from launch_jolmo.lr_bs_sweep import HI5_600M_SWEEPS as LRBS_HI5_600M_SWEEPS
for _sweep in LRBS_HI5_600M_SWEEPS:
    executor.stage(_sweep.label, _sweep.models())
    executor.stage(f"{_sweep.label}-evals", _sweep.evals())
executor.stage_group("lrbs-hi5-600m", tuple(_s.label for _s in LRBS_HI5_600M_SWEEPS))
executor.stage_group("lrbs-hi5-600m-evals",
                     tuple(f"{_s.label}-evals" for _s in LRBS_HI5_600M_SWEEPS))
# 300M above-optimal subsets, chinchillas 1/2/4, one 8-GPU run at a time
# (run with OPTIM_SIZE=300M).
from launch_jolmo.lr_bs_sweep import HI5_300M_SWEEPS as LRBS_HI5_300M_SWEEPS
for _sweep in LRBS_HI5_300M_SWEEPS:
    executor.stage(_sweep.label, _sweep.models())
    executor.stage(f"{_sweep.label}-evals", _sweep.evals())
executor.stage_group("lrbs-hi5-300m", tuple(_s.label for _s in LRBS_HI5_300M_SWEEPS))
executor.stage_group("lrbs-hi5-300m-evals",
                     tuple(f"{_s.label}-evals" for _s in LRBS_HI5_300M_SWEEPS))

# Finetuning cross-product sweeps (launch_jolmo/ft_sweep.py)
from launch_jolmo.ft_sweep import SWEEPS as FT_SWEEPS
from launch_jolmo.ft_sweep import CROSSOVER_SWEEPS as FT_XOVER_SWEEPS
from launch_jolmo.ft_sweep import BS_BEST_SWEEPS as FT_BSBEST_SWEEPS
from launch_jolmo.ft_sweep import ALLCHIN_SWEEPS as FT_ALLCHIN_SWEEPS
from launch_jolmo.ft_sweep import REPLAY_SWEEPS as FT_REPLAY_SWEEPS
from launch_jolmo.ft_sweep import BS_BEST_XSIZE_SWEEPS
from launch_jolmo.ft_sweep import TUNED_SWEEPS as FT_TUNED_SWEEPS
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
# Per-chinchilla umbrella groups: ft-lrbs-60m-c<chin> (+ -evals) for every
# chinchilla in the lr_bs sweep (0.25, 0.5, 1, 2, 4, 8). The trailing dash in
# the match keeps c1 from also matching c16-style names.
from launch_jolmo.lr_bs_sweep import LRBS_60M_CHINCHILLAS as _FT_CHINS
for _c in _FT_CHINS:
    _tag = f"{_c:g}"
    _labels = tuple(
        _s.label for _s in FT_SWEEPS if f"-chinchilla-{_tag}-" in _s.base.run_name)
    executor.stage_group(f"ft-lrbs-60m-c{_tag}", _labels)
    executor.stage_group(
        f"ft-lrbs-60m-c{_tag}-evals", tuple(f"{_l}-evals" for _l in _labels))
# FT over just the c1/c2 above-optimal (hi5) pretrained bases. Base names go
# through the same cross-schema aliasing as the FT sweeps, so they match.
_hi5_c12_base_names = {
    _m.run_name
    for _hs in LRBS_HI5_SWEEPS if _hs.chinchilla in (1, 2)
    for _m in _hs.models()
}
_hi5_c12_labels = tuple(
    _s.label for _s in FT_SWEEPS if _s.base.run_name in _hi5_c12_base_names)
executor.stage_group("ft-lrbs-hi5-c12", _hi5_c12_labels)
executor.stage_group("ft-lrbs-hi5-c12-evals",
                     tuple(f"{_l}-evals" for _l in _hi5_c12_labels))
# FT over the hi5 bases at chinchillas 0.5 - 16.
_hi5_c05_16_names = {
    _m.run_name
    for _hs in LRBS_HI5_SWEEPS if _hs.chinchilla in (0.5, 1, 2, 4, 8, 16)
    for _m in _hs.models()
}
_hi5_c05_16_labels = tuple(
    _s.label for _s in FT_SWEEPS if _s.base.run_name in _hi5_c05_16_names)
executor.stage_group("ft-lrbs-hi5-c05-16", _hi5_c05_16_labels)
executor.stage_group("ft-lrbs-hi5-c05-16-evals",
                     tuple(f"{_l}-evals" for _l in _hi5_c05_16_labels))
# Crossover cells finetuned WITH DCLM replay (one dataset per base).
for _sweep in FT_XOVER_SWEEPS:
    executor.stage(_sweep.label, _sweep.models())
    executor.stage(f"{_sweep.label}-evals", _sweep.evals())
# The bases must be IN the selection: the executor resolves dependencies by
# object identity, so a base that is only registered elsewhere fails the
# "all dependencies must be explicitly included" check. They already exist on
# GCS, so they are skipped rather than retrained.
executor.stage("ft-crossover-bases",
               [_s.base for _s in FT_XOVER_SWEEPS])
executor.stage_group("ft-crossover-replay",
                     ("ft-crossover-bases",)
                     + tuple(_s.label for _s in FT_XOVER_SWEEPS))
executor.stage_group("ft-crossover-replay-evals",
                     tuple(f"{_s.label}-evals" for _s in FT_XOVER_SWEEPS))
# FT of the best-LR cell per (chinchilla x batch size x optimizer) at 60M.
for _sweep in FT_BSBEST_SWEEPS:
    executor.stage(_sweep.label, _sweep.models())
    executor.stage(f"{_sweep.label}-evals", _sweep.evals())
executor.stage("ft-bs-best-bases", [_s.base for _s in FT_BSBEST_SWEEPS])
executor.stage_group("ft-bs-best-60m",
                     ("ft-bs-best-bases",)
                     + tuple(_s.label for _s in FT_BSBEST_SWEEPS))
executor.stage_group("ft-bs-best-60m-evals",
                     tuple(f"{_s.label}-evals" for _s in FT_BSBEST_SWEEPS))
# Just chinchillas 1 and 4: the batch axis at two representative budgets.
import re as _re
_bsb_c14 = tuple(
    _s for _s in FT_BSBEST_SWEEPS
    if float(_re.search(r"chinchilla-([0-9.]+)-", _s.base.run_name).group(1))
    in (1.0, 4.0))
executor.stage("ft-bs-best-c14-bases", [_s.base for _s in _bsb_c14])
executor.stage_group("ft-bs-best-60m-c1-c4",
                     ("ft-bs-best-c14-bases",)
                     + tuple(_s.label for _s in _bsb_c14))
executor.stage_group("ft-bs-best-60m-c1-c4-evals",
                     tuple(f"{_s.label}-evals" for _s in _bsb_c14))
# FT every pretrain LR at one (size, chinchilla) cell — see FT_ALLCHIN_SIZES.
for _sweep in FT_ALLCHIN_SWEEPS:
    executor.stage(_sweep.label, _sweep.models())
    executor.stage(f"{_sweep.label}-evals", _sweep.evals())
if FT_ALLCHIN_SWEEPS:
    executor.stage("ft-allchin-bases", [_s.base for _s in FT_ALLCHIN_SWEEPS])
    executor.stage_group("ft-allchin",
                         ("ft-allchin-bases",)
                         + tuple(_s.label for _s in FT_ALLCHIN_SWEEPS))
    executor.stage_group("ft-allchin-evals",
                         tuple(f"{_s.label}-evals" for _s in FT_ALLCHIN_SWEEPS))
# DCLM-replay FT over the tuned bases, named per size so the two can run on
# separate machines against the same bucket without colliding.
for _sweep in FT_REPLAY_SWEEPS:
    executor.stage(_sweep.label, _sweep.models())
    executor.stage(f"{_sweep.label}-evals", _sweep.evals())
if FT_REPLAY_SWEEPS:
    _rep = f"ft-replay-{_SIZE.lower()}"
    executor.stage(f"{_rep}-bases", [_s.base for _s in FT_REPLAY_SWEEPS])
    executor.stage_group(_rep, (f"{_rep}-bases",)
                         + tuple(_s.label for _s in FT_REPLAY_SWEEPS))
    executor.stage_group(f"{_rep}-evals",
                         tuple(f"{_s.label}-evals" for _s in FT_REPLAY_SWEEPS))
# Same, over the c1/c2/c4/c8 hi5 bases.
_hi5_c1248_base_names = {
    _m.run_name
    for _hs in LRBS_HI5_SWEEPS if _hs.chinchilla in (1, 2, 4, 8)
    for _m in _hs.models()
}
_hi5_c1248_labels = tuple(
    _s.label for _s in FT_SWEEPS if _s.base.run_name in _hi5_c1248_base_names)
executor.stage_group("ft-lrbs-hi5-c1248", _hi5_c1248_labels)
executor.stage_group("ft-lrbs-hi5-c1248-evals",
                     tuple(f"{_l}-evals" for _l in _hi5_c1248_labels))
# Muon alpha sweep: CPT muon models across alpha = muon→adamw LR ratio.
executor.stage("muon-sweep",           muon_sweep_models)

# --- Gaussian weight perturbation of the DCLM-pretrained models ---
# Gaussian perturbation of every tuned-LR base at this size, gamma ladder
# 0.02-0.13, plus DCLM-heldout evals.
executor.stage("perturb-wide-bases",   perturb_wide_bases)
executor.stage("perturb-wide",         perturb_wide_models)
executor.stage("perturb-wide-evals",   perturb_wide_evals)
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

# --- Cross-size batch-size FT (30M c2, 60M c1/2/4, 100M c2 best-LR cells) ---
for _sweep in BS_BEST_XSIZE_SWEEPS:
    executor.stage(_sweep.label, _sweep.models())
    executor.stage(f"{_sweep.label}-evals", _sweep.evals())
if BS_BEST_XSIZE_SWEEPS:
    # Bases go INSIDE the group: the executor resolves dependencies by object
    # identity, so a base outside the selected set raises "not in artifact set".
    executor.stage("ft-bs-best-x-bases",
                   [_s.base for _s in BS_BEST_XSIZE_SWEEPS])
    executor.stage_group("ft-bs-best-x", ("ft-bs-best-x-bases",)
                         + tuple(_s.label for _s in BS_BEST_XSIZE_SWEEPS))
    executor.stage_group("ft-bs-best-x-evals",
                         tuple(f"{_s.label}-evals" for _s in BS_BEST_XSIZE_SWEEPS))

# --- EWC finetuning (best-LR base per chinchilla; OPTIM_EWC_CHINCHILLAS narrows) ---
# The bases go INSIDE the group: the executor resolves dependencies by object
# identity, so an EWCModel whose pretrained_model is not in the selected set
# fails with "... which is not in the artifact set".
executor.stage("ewc-bases",  ewc_bases)
executor.stage("ewc-runs",   ewc_models)
executor.stage_group("ewc",  ("ewc-bases", "ewc-runs"))
executor.stage("ewc-evals",  ewc_evals)

# --- Plain FT on the tuned base of every chinchilla in the size profile ---
for _sweep in FT_TUNED_SWEEPS:
    executor.stage(_sweep.label, _sweep.models())
    executor.stage(f"{_sweep.label}-evals", _sweep.evals())
if FT_TUNED_SWEEPS:
    # Bases INSIDE the group: dependencies resolve by object identity.
    executor.stage("ft-tuned-bases", [_s.base for _s in FT_TUNED_SWEEPS])
    executor.stage_group("ft-tuned", ("ft-tuned-bases",)
                         + tuple(_s.label for _s in FT_TUNED_SWEEPS))
    executor.stage_group("ft-tuned-evals",
                         tuple(f"{_s.label}-evals" for _s in FT_TUNED_SWEEPS))

if __name__ == "__main__":
    executor.auto_cli()

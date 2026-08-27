"""CPT model artifacts for the optimizer-study pretraining experiments."""

from typing import Dict, List, Optional, Tuple, Union
from experiments import ArtifactSet

from launch_jolmo.training import JolmoModel, CPTModel, ModelEvaluation


# ---------------------------------------------------------------------------
# CPT configuration
# ---------------------------------------------------------------------------

CPT_DATASETS: List[str] = [
    "tulu", 
    "starcoder",
    "musicpile",
    "alpaca",
    "gsm8k",
    # "siqa",
    # "open-platypus",
    "stackmathqa",
    # "helpsteer",
]
CPT_TOKENS: int = 20_000_000   # 100 M tokens per CPT run
CPT_SCHEDULER: str = "cosine"

# CPT batch & context (decoupled from the pretrain config in pretraining_matrix.py,
# which uses a 1M-token batch / 4096 context). CPT runs a smaller 64K-token batch
# at 1024 context.
CPT_GLOBAL_BATCH_SIZE: int = 65_536        # 64K tokens/step
CPT_SEQUENCE_LENGTH: int = 1024            # CPT context length
CPT_RANK_MICROBATCH_SIZE: int = CPT_GLOBAL_BATCH_SIZE // 8

# LR sweep applied per optimizer type.
# For adamw: list of learning_rate values.
# For muon:  list of (muon_lr, adamw_component_lr) pairs.
# Set to a single-element list to use a fixed LR instead of sweeping.
CPT_LR_SWEEP: Dict[str, List] = {
    "adamw": [1e-3, 2e-3, 5e-3, 1e-2, 5e-4,8e-4, 2e-4, 1e-4],
    # "muon" : [(5e-3, 1e-4), (6e-3, 1e-4), (7e-3, 1e-4)],
    "muon":  [(6e-4, 1e-4), (8e-4, 1e-4), (1e-3, 1e-4), (2e-3, 1e-4), (4e-3, 1e-4), (8e-3, 1e-4), (1e-2, 1e-4), (2e-2, 1e-4),],
}


def build_muon_lr_sweep(multiplier: float, muon_lrs: Optional[List[float]] = None) -> Dict[str, List]:
    """Build a CPT LR sweep where the adamw component LR = multiplier * muon_lr.

    Args:
        multiplier: Scale factor applied to each muon_lr to get the adamw component LR.
        muon_lrs: List of muon learning rates to sweep. Defaults to the muon_lr
                  values from CPT_LR_SWEEP["muon"].
    """
    if muon_lrs is None:
        muon_lrs = [entry[0] if isinstance(entry, tuple) else entry
                    for entry in CPT_LR_SWEEP["muon"]]
    return {
        "adamw": CPT_LR_SWEEP["adamw"],
        "muon": [(mlr, multiplier * mlr) for mlr in muon_lrs],
    }


def build_cpt_models(
    pretrained_models: ArtifactSet,
    datasets: List[str] = CPT_DATASETS,
    lr_sweep: Optional[Dict[str, List]] = None,
    cpt_optimizers: Optional[List[str]] = None,
    muon_adamw_multiplier: Optional[float] = None,
) -> ArtifactSet:
    """One CPTModel per (pretrained model × dataset × cpt_optimizer × learning rate) combination.

    By default the CPT optimizer is inherited from the pretrained model.
    Pass cpt_optimizers (e.g. ["adamw"]) to override and sweep a different
    optimizer for the CPT phase regardless of how the base model was trained.
    Learning rates are swept over CPT_LR_SWEEP[cpt_optimizer] (or the provided
    lr_sweep override), matching the pattern in catastrophic-forgetting/launch/cpt.py.

    If muon_adamw_multiplier is set, the muon LR sweep is built via
    build_muon_lr_sweep(multiplier) so that adamw_lr = multiplier * muon_lr.
    This takes precedence over lr_sweep for the "muon" key.
    """
    if muon_adamw_multiplier is not None:
        sweep = build_muon_lr_sweep(muon_adamw_multiplier)
        if lr_sweep is not None:
            sweep.update({k: v for k, v in lr_sweep.items() if k != "muon"})
    else:
        sweep = lr_sweep if lr_sweep is not None else CPT_LR_SWEEP

    def _make_cpt(base: JolmoModel) -> ArtifactSet:
        opts = cpt_optimizers if cpt_optimizers is not None else [base.optimizer]

        models = []
        for opt in opts:
            lrs = sweep.get(opt, [1e-4])
            for ds in datasets:
                for lr_entry in lrs:
                    if opt == "muon" and isinstance(lr_entry, tuple):
                        muon_lr, adamw_lr = lr_entry
                    else:
                        muon_lr = base.muon_lr if opt == "muon" else None
                        adamw_lr = lr_entry if isinstance(lr_entry, float) else lr_entry[1]

                    models.append(CPTModel(
                        pretrained_model=base,
                        cpt_dataset=ds,
                        train_tokens=CPT_TOKENS,
                        optimizer=opt,
                        learning_rate=adamw_lr,
                        muon_lr=muon_lr,
                        scheduler=CPT_SCHEDULER,
                        sequence_length=CPT_SEQUENCE_LENGTH,
                        global_batch_size=CPT_GLOBAL_BATCH_SIZE,
                        rank_microbatch_size=CPT_RANK_MICROBATCH_SIZE,
                    ))

        return ArtifactSet(models)

    return pretrained_models.map_flatten(_make_cpt)


def build_cpt_model_evaluations(
    cpt_models: ArtifactSet,
    extra_val_chunks=(),
    extra_val_max_instances=None,
) -> ArtifactSet:
    """One ModelEvaluation per CPT model.

    `extra_val_chunks` / `extra_val_max_instances` mirror the pretrain/perturb
    evals: pass the held-out DCLM shard so each CPT model is also scored on the
    DCLM split (label ``DCLM_heldout``) — the forgetting / pretrain-loss axis used
    by the Pareto tradeoff plots — alongside the diversity-v2 and CPT val sets.
    Callers supply these (defined in ``pretraining_matrix``) to avoid a circular
    import.
    """
    return cpt_models.map(
        lambda model: ModelEvaluation(
            model=model,
            extra_val_chunks=tuple(extra_val_chunks),
            extra_val_max_instances=extra_val_max_instances,
        )
    )

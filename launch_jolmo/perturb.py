"""Perturbed model artifacts for the optimizer-study pretraining experiments."""

from typing import Optional, Sequence, Tuple

from experiments import ArtifactSet

from launch_jolmo.training import (
    DEFAULT_PERTURB_SEEDS,
    ModelEvaluation,
    MultiSeedPerturbedEvaluation,
    MultiSeedPerturbedModel,
    PerturbedModel,
)


DEFAULT_GAMMAS = [2e-2]

# The gamma ladder actually swept for the multi-seed perturbation study
# (multi-seed dirs exist on GCS for each of these).
LADDER_GAMMAS = [0.002, 0.005, 0.007, 0.01, 0.02, 0.04, 0.08, 0.16]


def build_perturbed_models(base_models: ArtifactSet, gammas=None) -> ArtifactSet:
    """One PerturbedModel per (base model × gamma) combination (all weights).

    gamma controls the per-parameter noise scale:
      std = gamma * ||W||_F / sqrt(numel)
    """
    gs = list(DEFAULT_GAMMAS if gammas is None else gammas)
    return base_models.map_flatten(
        lambda model: ArtifactSet.from_product(
            cls=PerturbedModel,
            params=dict(source_model=model, gamma=gs),
        )
    )


def build_multi_seed_perturbed_models(
    base_models: ArtifactSet,
    gammas=None,
    seeds: Optional[Sequence[int]] = None,
) -> ArtifactSet:
    """One MultiSeedPerturbedModel per (base × gamma): N directions under one dir.

    Writes::

        PerturbedModel/{base}_perturbed_{γ}/seed_000/…/model.pt
        …
        PerturbedModel/{base}_perturbed_{γ}/seed_009/…/model.pt
    """
    gs = list(DEFAULT_GAMMAS if gammas is None else gammas)
    seed_tuple: Tuple[int, ...] = tuple(
        DEFAULT_PERTURB_SEEDS if seeds is None else seeds
    )
    # Explicit list: seeds is a tuple that from_product would unpack as an axis.
    return ArtifactSet([
        MultiSeedPerturbedModel(source_model=model, gamma=g, seeds=seed_tuple)
        for model in base_models
        for g in gs
    ])


def build_perturbed_model_evaluations(
    perturbed_models: ArtifactSet,
    extra_val_chunks=(),
    extra_val_max_instances=None,
) -> ArtifactSet:
    """One ModelEvaluation per saved perturbed checkpoint."""
    extra = tuple(extra_val_chunks)
    return perturbed_models.map(
        lambda model: ModelEvaluation(
            model=model,
            extra_val_chunks=extra,
            extra_val_max_instances=extra_val_max_instances,
        )
    )


def build_multi_seed_perturbed_evaluations(
    multiseed_models: ArtifactSet,
    extra_val_chunks=(),
    extra_val_max_instances=None,
) -> ArtifactSet:
    """One MultiSeedPerturbedEvaluation per (base × gamma) multi-seed group.

    Scores DCLM_heldout on every seed, writes mean + per-seed losses.
    Built with an explicit list so ``extra_val_chunks`` is not unpacked by
    ``from_product``.
    """
    extra = tuple(extra_val_chunks)
    return ArtifactSet([
        MultiSeedPerturbedEvaluation(
            model=model,
            extra_val_chunks=extra,
            extra_val_max_instances=extra_val_max_instances,
        )
        for model in multiseed_models
    ])

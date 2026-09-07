"""EWC finetuning as an artifact — a third method beside plain CPT and replay.

Structure mirrors ``PerturbedModel``: the artifact stages its inputs, writes a
config, and shells out to a standalone script (``new_utils/ewc_finetune.py``).
Nothing in the CPT path or the olmo_core trainer changes.

The penalty::

    L = L_finetune(theta) + (lambda / 2) * sum_i F_i * (theta_i - theta*_i)^2

with ``F`` the diagonal Fisher of the PRETRAINING task and ``theta*`` the
pretrained weights. ``ewc_lambda = 0`` recovers ordinary finetuning.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from experiments import Artifact, ArtifactSet, Project, Task

from launch_jolmo.training import (
    CPT_DATASETS,
    Chunk,
    JolmoModel,
    _PRETRAIN_VAL_CHUNKS,
    _download_chunk_dirs,
    _download_chunk_files,
    _resolve_chunk_path,
    _upload_to_gs_with_retry,
    blob_exists,
    local_cache_path,
    local_path,
    remote_path,
)

# The requested four. The reference sweeps [1e3, 4e3, 1e4, 4e4, 1e5]; lambda is
# only meaningful against an UNNORMALISED Fisher, so these values are not
# transferable to an implementation that rescales F.
EWC_LAMBDAS: Tuple[float, ...] = (1e3, 4e3, 1e4, 4e4)


def _lambda_tag(value: float) -> str:
    """1e3 -> 'lam1e3', 4e4 -> 'lam4e4'."""
    return "lam" + f"{value:.0e}".replace("e+0", "e").replace("e+", "e")


@dataclass(frozen=True)
class EWCModel(Artifact):
    """One EWC finetune of a pretrained model on one CPT dataset."""

    pretrained_model: JolmoModel = None  # type: ignore[assignment]
    cpt_dataset: str = "gsm8k"
    ewc_lambda: float = 1e3

    # Pretraining shards the diagonal Fisher is estimated on. REQUIRED — the
    # Fisher must come from the pretraining task, not the finetune set, or the
    # penalty anchors to the wrong objective entirely.
    fisher_chunks: Tuple[Chunk, ...] = ()
    fisher_batches: int = 100

    # Matches CPT_TOKENS so EWC and plain-CPT runs are directly comparable.
    train_tokens: int = 20_000_000
    sequence_length: int = 1024
    global_batch_size: int = 65_536          # tokens -> 64 sequences per step
    micro_batch_size: int = 8                # sequences per forward; memory only

    optimizer: str = "adamw"                 # "adamw" | "muon"
    learning_rate: float = 1e-4              # for muon, the AdamW-COMPONENT LR
    muon_lr: float = 0.02
    muon_weight_decay: float = 0.1
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 20
    max_grad_norm: float = 1.0
    seed: int = 64

    num_processes: int = 1                   # single GPU; no DDP
    upload: bool = True

    def __post_init__(self):
        if self.ewc_lambda > 0 and not self.fisher_chunks:
            raise ValueError(
                f"{self.run_name}: ewc_lambda={self.ewc_lambda:g} requires "
                f"fisher_chunks (the pretraining shards to estimate F on)")

    @property
    def run_name(self) -> str:
        lr_str = f"{self.learning_rate:.1e}".replace("e-0", "e-")
        tk_str = f"{self.train_tokens // 1_000_000}M"
        name = (f"{self.pretrained_model.run_name}-EWC-{self.cpt_dataset}-"
                f"{tk_str}-{_lambda_tag(self.ewc_lambda)}-{self.optimizer}-lr{lr_str}")
        if self.optimizer == "muon":
            name += f"-muonlr{self.muon_lr:.1e}".replace("e-0", "e-")
        return name

    @property
    def relpath(self) -> str:
        # Its own top-level dir, so EWC runs never mix with CPTModel/.
        return f"EWCModel/{self.cpt_dataset}/{self.run_name}"

    @property
    def olmo_model_relpath(self) -> str:
        return os.path.join(self.relpath, "final-unsharded")

    @property
    def validation_chunks(self) -> Tuple[Tuple[str, "Chunk"], ...]:
        """CPT-dataset val + pretrain val, identical to CPTModel.

        The dataset val loss is the LEARNING axis of the learning/forgetting
        plots. Without it the eval emits only the DCLM_heldout side and EWC
        cannot be compared against plain CPT.
        """
        cpt_val: Dict[str, "Chunk"] = CPT_DATASETS[self.cpt_dataset]["val"]
        return tuple(cpt_val.items()) + _PRETRAIN_VAL_CHUNKS

    @property
    def exists(self) -> bool:
        return blob_exists(remote_path(self.olmo_model_relpath, "model.pt"))

    def get_requirements(self):
        return {"cpus": 4, "gres": "gpu:1", "mem": "64G",
                "partition": "preempt", "exclude": "babel-m9-16"}

    def construct(self, builder: Task):
        builder.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        cpt_info = CPT_DATASETS[self.cpt_dataset]
        train_chunks: Tuple[Chunk, ...] = cpt_info["train"]

        data_dir = local_path()
        cache_dir = local_cache_path()
        output_dir = os.path.join(data_dir, self.relpath)
        dataset_cache_dir = os.path.join(cache_dir, "datasets")

        builder.ensure_directory(output_dir)
        builder.ensure_directory(dataset_cache_dir)

        _download_chunk_dirs(builder, list(train_chunks), dataset_cache_dir)
        if self.fisher_chunks:
            # Single large DCLM shards: fetch file-by-file into the shared cache
            # so every task on the node reuses one copy.
            _download_chunk_files(builder, list(self.fisher_chunks), dataset_cache_dir)

        train_paths = [_resolve_chunk_path(c, dataset_cache_dir) for c in train_chunks]
        fisher_paths = [_resolve_chunk_path(c, dataset_cache_dir)
                        for c in self.fisher_chunks]

        # Reuse JolmoModel's YAML builder purely for the model spec: the script
        # needs the TransformerConfig so the architecture matches the checkpoint.
        pseudo = JolmoModel(
            model_name=self.run_name,
            model_type=self.pretrained_model.model_type,
            tokenizer=self.pretrained_model.tokenizer,
            sequence_length=self.sequence_length,
            global_batch_size=self.global_batch_size,
            rank_microbatch_size=self.global_batch_size,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=self.betas,
            warmup_steps=self.warmup_steps,
            muon_lr=self.muon_lr,
            muon_weight_decay=self.muon_weight_decay,
            scheduler="cosine",
            max_grad_norm=self.max_grad_norm,
            n_tokens=self.train_tokens,
            num_processes=self.num_processes,
            save_overwrite=True,
            base_model=self.pretrained_model,
            train_chunks=(train_chunks[0],),
        )
        save_folder = remote_path(self.relpath)
        val_datasets = {
            label: [_resolve_chunk_path(c, dataset_cache_dir)]
            for label, c in self.validation_chunks
        }
        yaml_config = pseudo._build_yaml_config(
            save_folder, train_paths, val_datasets,
            os.path.join(cache_dir, "training", self.run_name))
        config_path = os.path.join(output_dir, "config.yaml")
        builder.create_yaml_file(config_path, yaml_config)

        # Stage the pretrained checkpoint (weights + its config.json, which is
        # copied beside the output so downstream loaders can find it).
        base_remote = remote_path(self.pretrained_model.relpath, "final-unsharded")
        base_local = os.path.join(output_dir, "base")
        builder.ensure_directory(base_local)
        builder.run_command(
            f'gsutil -m cp "{base_remote}/model.pt" "{base_local}/model.pt"')
        builder.run_command(
            f'gsutil -m cp "{base_remote}/config.json" "{base_local}/config.json" '
            f'|| echo "[ewc] no base config.json"')

        script = os.path.join(Project.config.code_path, "new_utils", "ewc_finetune.py")
        cmd = (
            f'python3 {script} '
            f'--config "{config_path}" '
            f'--base-checkpoint "{base_local}/model.pt" '
            f'--base-config-json "{base_local}/config.json" '
            f'--output-dir "{output_dir}" '
            f'--train-paths {" ".join(f_quote(p) for p in train_paths)} '
            f'--ewc-lambda {self.ewc_lambda} '
            f'--fisher-batches {self.fisher_batches} '
            f'--train-tokens {self.train_tokens} '
            f'--sequence-length {self.sequence_length} '
            f'--global-batch-size {self.global_batch_size} '
            f'--micro-batch-size {self.micro_batch_size} '
            f'--optimizer {self.optimizer} '
            f'--learning-rate {self.learning_rate} '
            f'--muon-lr {self.muon_lr} '
            f'--muon-weight-decay {self.muon_weight_decay} '
            f'--weight-decay {self.weight_decay} '
            f'--beta1 {self.betas[0]} --beta2 {self.betas[1]} '
            f'--warmup-steps {self.warmup_steps} '
            f'--max-grad-norm {self.max_grad_norm} '
            f'--seed {self.seed}'
        )
        if fisher_paths:
            cmd += f' --fisher-paths {" ".join(f_quote(p) for p in fisher_paths)}'
        builder.run_command(cmd)

        if self.upload:
            unsharded_dir = os.path.join(output_dir, "final-unsharded")
            _upload_to_gs_with_retry(
                builder, unsharded_dir, remote_path(self.olmo_model_relpath),
                directory=True, contents=True,
            )
            builder.run_command(f'rm -rf -- "{output_dir}"')


def f_quote(path: str) -> str:
    return f'"{path}"'


def build_ewc_models(
    base_models: ArtifactSet,
    fisher_chunks: Tuple[Chunk, ...],
    lr_sweep: Dict[str, List],
    lambdas: Tuple[float, ...] = EWC_LAMBDAS,
    datasets: Optional[List[str]] = None,
) -> ArtifactSet:
    """One EWCModel per (base x arm x LR x lambda x dataset).

    Arms mirror the plain-CPT wiring: an adamw-pretrained base gets an adamw
    finetune; a muon-pretrained base gets BOTH muon and adamw. The LR grid is
    the same ``CPT_LR_SWEEP`` plain CPT uses, so every EWC run has an LR-matched
    counterpart in the CPT sweep.
    """
    datasets = list(datasets or ["gsm8k"])
    models: List[EWCModel] = []
    for base in base_models:
        arms = ["adamw"] if base.optimizer == "adamw" else ["muon", "adamw"]
        for arm in arms:
            for entry in lr_sweep[arm]:
                if arm == "muon":
                    mlr = entry[0] if isinstance(entry, (tuple, list)) else entry
                    kw = {"optimizer": "muon", "muon_lr": mlr,
                          # AdamW component tied to the matrix LR, as in the
                          # muon CPT arm.
                          "learning_rate": 0.25 * mlr}
                else:
                    kw = {"optimizer": "adamw", "learning_rate": float(entry)}
                for lam in lambdas:
                    for ds in datasets:
                        models.append(EWCModel(
                            pretrained_model=base,
                            cpt_dataset=ds,
                            ewc_lambda=lam,
                            fisher_chunks=fisher_chunks,
                            **kw,
                        ))
    return ArtifactSet(models)

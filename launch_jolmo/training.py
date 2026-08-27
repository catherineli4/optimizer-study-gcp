import functools as ft
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal, Tuple, List

from experiments import Project, Artifact, Task
from olmo_core.data import TokenizerConfig

from launch_jolmo.data import Chunk
from launch_jolmo.utils import (
    check_exists_remote,
    blob_exists,
    local_path,
    local_cache_path,
    remote_path,
)


# ---------------------------------------------------------------------------
# Model architecture registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ModelConfig:
    d_model: int
    n_layers: int
    n_heads: int
    hidden_size: int


MODEL_ARCHS: Dict[str, _ModelConfig] = {
    # Approximate parameter scales for small-model sweeps.
    "0.03B":  _ModelConfig(d_model=128 * 1, n_layers=3 * 1, n_heads=2 * 1, hidden_size=512 * 1),
    "0.06B": _ModelConfig(d_model=128 * 2, n_layers=3 * 2, n_heads=2 * 2, hidden_size=512 * 2),
    "0.1B": _ModelConfig(d_model=128 * 3, n_layers=3 * 3, n_heads=2 * 3, hidden_size=512 * 3),
    "0.3B": _ModelConfig(d_model=128 * 6, n_layers=3 * 6, n_heads=2 * 6, hidden_size=512 * 6),
    "0.6B": _ModelConfig(d_model=128 * 8, n_layers=3 * 8, n_heads=2 * 8, hidden_size=512 * 8),
    "1B":   _ModelConfig(d_model=128 * 10, n_layers=3 * 10, n_heads=2 * 10, hidden_size=512 * 10),
    "2.5B": _ModelConfig(d_model=128 * 14, n_layers=3 * 14, n_heads=2 * 14, hidden_size=512 * 14),
}


# ---------------------------------------------------------------------------
# Module-level helpers for YAML config generation
# ---------------------------------------------------------------------------

@ft.lru_cache(maxsize=None)
def _tokenizer_config(tokenizer_id: str) -> TokenizerConfig:
    """Resolve a tokenizer identifier into a fully populated TokenizerConfig.

    Cached per identifier: ``from_hf`` hits the HF Hub, and compiling a large
    sweep calls this once per artifact — uncached, thousands of unauthenticated
    Hub requests get rate-limited into indefinite backoff.
    """
    try:
        cfg = TokenizerConfig(identifier=tokenizer_id)
    except TypeError:
        cfg = TokenizerConfig.from_hf(tokenizer_id)
    else:
        if cfg.vocab_size is None or cfg.eos_token_id is None or cfg.pad_token_id is None:
            cfg = TokenizerConfig.from_hf(tokenizer_id)
    return cfg


def _build_tokenizer_yaml(tokenizer_id: str) -> Dict[str, Any]:
    cfg = _tokenizer_config(tokenizer_id)
    d: Dict[str, Any] = {
        "_CLASS_": "olmo_core.data.TokenizerConfig",
        "identifier": tokenizer_id,
        "vocab_size": cfg.vocab_size,
        "eos_token_id": cfg.eos_token_id,
        "pad_token_id": cfg.pad_token_id,
    }
    if cfg.bos_token_id is not None:
        d["bos_token_id"] = cfg.bos_token_id
    return d


def _build_model_spec(model_type: str, vocab_size: int) -> Dict[str, Any]:
    arch = MODEL_ARCHS[model_type]
    return {
        "_CLASS_": "olmo_core.nn.transformer.config.TransformerConfig",
        "d_model": arch.d_model,
        "vocab_size": vocab_size,
        "n_layers": arch.n_layers,
        "block": {
            "_CLASS_": "olmo_core.nn.transformer.config.TransformerBlockConfig",
            "name": "reordered_norm",
            "attention": {
                "_CLASS_": "olmo_core.nn.attention.AttentionConfig",
                "n_heads": arch.n_heads,
                "n_kv_heads": None,
                "bias": False,
                "qk_norm": {
                    "_CLASS_": "olmo_core.nn.layer_norm.LayerNormConfig",
                    "name": "rms",
                    "eps": 1.0e-06,
                    "bias": False,
                },
                "rope": {
                    "_CLASS_": "olmo_core.nn.rope.RoPEConfig",
                    "name": "default",
                    "theta": 500_000,
                },
                "backend": "flash_2",
                "dtype": "float32",
            },
            "feed_forward": {
                "_CLASS_": "olmo_core.nn.feed_forward.FeedForwardConfig",
                "hidden_size": arch.hidden_size,
                "bias": False,
                "dtype": "float32",
            },
            "layer_norm": {
                "_CLASS_": "olmo_core.nn.layer_norm.LayerNormConfig",
                "name": "rms",
                "eps": 1.0e-06,
                "bias": False,
            },
        },
        "lm_head": {
            "_CLASS_": "olmo_core.nn.lm_head.LMHeadConfig",
            "name": "default",
            "layer_norm": {
                "_CLASS_": "olmo_core.nn.layer_norm.LayerNormConfig",
                "name": "rms",
                "eps": 1.0e-06,
                "bias": False,
            },
            "bias": False,
            "dtype": "float32",
        },
        "dtype": "float32",
    }


def _optimizer_class_name(optimizer: str) -> str:
    name = optimizer.lower()
    if name == "adamw":
        return "olmo_core.optim.AdamWConfig"
    if name == "adam":
        return "olmo_core.optim.AdamConfig"
    if name == "muon":
        return "olmo_core.optim.MuonConfig"
    raise ValueError(f"Unsupported optimizer '{optimizer}'. Supported: ['adamw', 'adam', 'muon']")


def _build_optimizer_spec(
    optimizer: str,
    lr: float,
    betas: Tuple[float, float],
    weight_decay: float,
    *,
    muon_lr: float = 0.02,
    muon_weight_decay: float = 0.1,
) -> Dict[str, Any]:
    cls = _optimizer_class_name(optimizer)

    embedding_override = {
        "_CLASS_": "olmo_core.optim.OptimGroupOverride",
        "params": ["embeddings.weight"],
        "opts": {"weight_decay": 0.0},
    }

    if cls == "olmo_core.optim.MuonConfig":
        return {
            "_CLASS_": cls,
            "lr": muon_lr,
            "weight_decay": muon_weight_decay,
            "adamw_lr": lr,
            "adamw_betas": list(betas),
            "adamw_weight_decay": weight_decay,
            "group_overrides": [embedding_override],
        }

    spec: Dict[str, Any] = {
        "_CLASS_": cls,
        "lr": lr,
        "betas": list(betas),
        "weight_decay": weight_decay,
        "group_overrides": [embedding_override],
    }
    if cls in {"olmo_core.optim.AdamWConfig", "olmo_core.optim.AdamConfig"}:
        spec["fused"] = True
    return spec


def _build_scheduler_spec(scheduler: str, warmup_steps: int) -> Dict[str, Any]:
    name = scheduler.lower()
    if name == "cosine":
        return {"_CLASS_": "olmo_core.optim.CosWithWarmup", "warmup_steps": warmup_steps}
    if name == "constant":
        return {"_CLASS_": "olmo_core.optim.ConstantScheduler"}
    if name in {"inv_sqrt", "inverse_sqrt", "invsqrt"}:
        return {
            "_CLASS_": "olmo_core.optim.InvSqrtWithWarmup",
            "warmup": warmup_steps,
            "alpha_f": 0.1,
        }
    if name == "wsd":
        return {
            "_CLASS_": "olmo_core.optim.WSD",
            "warmup": warmup_steps,
            "decay_min_lr": 0.0,
        }
    raise ValueError(
        "Unsupported scheduler "
        f"'{scheduler}'. Supported: ['cosine', 'constant', 'inv_sqrt', 'wsd']"
    )


def _gs_parent(uri: str) -> str:
    """Return the parent directory of a URI (everything before the last '/')."""
    return uri.rsplit("/", 1)[0]


def _resolve_chunk_path(chunk: Chunk, dataset_cache_dir: str) -> str:
    """Return the local path where a Chunk's data will be cached."""
    if chunk.is_global:
        # Use the last two path components of the GCS parent directory so that
        # datasets sharing a leaf name (e.g. ".../bigcode_starcoderdata/train/"
        # and ".../allenai_tulu-3-sft-mixture/train/") resolve to distinct local
        # paths and don't silently overwrite each other.
        gs_parent = _gs_parent(chunk.uri)
        components = gs_parent.replace("gs://", "").split("/")
        parent_name = os.path.join(*components[-2:]) if len(components) >= 2 else components[-1]
        return os.path.join(dataset_cache_dir, parent_name, os.path.basename(chunk.uri))
    return os.path.join(dataset_cache_dir, chunk.uri)


def _upload_to_gs_with_retry(
    builder: Task,
    local_path: str,
    gs_path: str,
    directory: bool = False,
    contents: bool = True,
    max_attempts: int = 5,
    base_delay: int = 30,
    guard_path: Optional[str] = None,
) -> None:
    """Upload to GCS with exponential-backoff retry to handle 429/503/stall errors.

    When many job-array tasks finish simultaneously they all upload at once,
    which frequently triggers GCS's per-object mutation rate limit (HTTP 429).
    The built-in gsutil retry only covers transient network errors; 429s require
    explicit backoff at the shell level.

    gsutil rsync can also stall indefinitely when it resumes a stale resumable-upload
    tracker from a prior interrupted attempt (the GCS session is dead but gsutil
    waits forever).  We wrap each attempt in `timeout` so a hung gsutil is killed
    and the next retry starts fresh.

    ``guard_path``: when set, the whole upload is wrapped in a test for that path,
    so a worker that legitimately produced no output (e.g. it skipped a missing
    input checkpoint) doesn't abort the combined ``print | bash`` script under
    ``set -e`` — the upload simply no-ops.
    """
    if directory:
        source = local_path.rstrip("/") + "/"
        dest = gs_path.rstrip("/") + "/"
        gsutil_cmd = f'gsutil -m rsync -r "{source}" "{dest}"'
    else:
        gsutil_cmd = f'gsutil cp "{local_path}" "{gs_path}"'

    # Wrap each gsutil invocation in a 5-minute timeout.  If gsutil stalls
    # (e.g. resuming a dead upload session) it will be killed and the next
    # retry attempt fires after the configured delay.
    timeout_cmd = f'timeout 300 {gsutil_cmd}'

    # Use ||‑chained retries instead of a while loop with shell variables.
    # The SLURM executor echoes every command (with $ unexpanded) before running
    # it; any $var reference in that echo triggers set -u "unbound variable" if
    # the variable hasn't been set yet.  A chain of || subshells contains no
    # unset variable references and is safe under set -euo pipefail.
    delays = [base_delay * (2 ** i) for i in range(max_attempts - 1)]
    attempts = [timeout_cmd] + [f'(sleep {d}; {timeout_cmd})' for d in delays]
    retry_cmd = ' || '.join(attempts)
    if guard_path is not None:
        retry_cmd = (
            f'if [ -e "{guard_path}" ]; then {retry_cmd}; '
            f'else echo "SKIP upload: {guard_path} absent (worker produced no output)"; fi'
        )
    builder.run_command(retry_cmd)


def _download_chunk_dirs(
    builder: Task,
    chunks: List[Chunk],
    dataset_cache_dir: str,
) -> None:
    """Download GCS parent directories for all chunks, skipping dirs already present locally.

    Downloads at directory granularity (directory=True) rather than per-file so that
    the executor's built-in failure cleanup fires: on any gsutil error the whole local
    directory is removed (rm -rf), preventing partial downloads from being mistaken for
    complete ones on subsequent runs.

    Path collision between datasets that share a leaf directory name (e.g.
    ".../bigcode_starcoderdata/train/" and ".../allenai_tulu-3-sft-mixture/train/")
    is avoided by _resolve_chunk_path, which uses the last two GCS path components
    as the local subdirectory (e.g. "bigcode_starcoderdata/train/").

    Uses sequential gsutil (no ``-m``) plus retries: parallel multi-file downloads
    from several array tasks race on shared ``~/.gsutil/tracker-files`` and fail
    with missing ``*.etag`` trackers.

    Skip-existing only fires when every chunk file expected under that local dir
    is present — a non-empty but incomplete dir (e.g. missing ``00001.npy``) is
    re-downloaded.
    """
    # gs_dir -> (local_dir, expected local files)
    pending: Dict[str, Tuple[str, List[str]]] = {}
    for chunk in chunks:
        if chunk.uri.startswith("/"):
            local = _resolve_chunk_path(chunk, dataset_cache_dir)
            builder.ensure_directory(os.path.dirname(local))
            builder.run_command(f'[ -e "{local}" ] || ln -s "{chunk.uri}" "{local}"')
            continue

        gs_file = chunk.uri if chunk.uri.startswith("gs://") else remote_path(chunk.uri)
        gs_dir = _gs_parent(gs_file)
        local_file = _resolve_chunk_path(chunk, dataset_cache_dir)
        local_dir = os.path.dirname(local_file)
        if gs_dir not in pending:
            pending[gs_dir] = (local_dir, [])
        pending[gs_dir][1].append(local_file)

    for gs_dir, (local_dir, required_files) in pending.items():
        _download_gs_dir_with_retry(
            builder, gs_dir, local_dir, required_files=required_files,
        )


def _download_gs_dir_with_retry(
    builder: Task,
    gs_dir: str,
    local_dir: str,
    max_attempts: int = 5,
    base_delay: int = 15,
    required_files: Optional[List[str]] = None,
) -> None:
    """Flock + sequential gsutil cp with retries; refuse incomplete skip-existing.

    Retry logic lives in ``gs_cp_dir_retry.sh`` so each task arm is one short
    line.  Inlining 5× ``gsutil`` attempts (plus the executor's debug ``echo``)
    bloated CPT array scripts past Babel's ~4 MB batch-script limit
    (``Pathname of a file, directory or other parameter too long``).

    If ``required_files`` is set, skip/success require every listed path to exist
    (not merely a non-empty directory).
    """
    helper = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gs_cp_dir_retry.sh")
    dest = local_dir.rstrip("/")
    src = gs_dir.rstrip("/")
    req = " ".join(f'"{p}"' for p in (required_files or []))
    builder.run_command(
        f'bash "{helper}" "{src}" "{dest}" {max_attempts} {base_delay}'
        + (f" {req}" if req else "")
    )

def _group_labeled_chunk_paths(
    labeled_chunks: Tuple[Tuple[str, Chunk], ...],
    dataset_cache_dir: str,
) -> Dict[str, List[str]]:
    """Group (label, chunk) pairs into label -> [paths], preserving order."""
    out: Dict[str, List[str]] = {}
    for label, chunk in labeled_chunks:
        out.setdefault(label, []).append(_resolve_chunk_path(chunk, dataset_cache_dir))
    return out


def _download_chunk_files(
    builder: Task,
    chunks: List[Chunk],
    dataset_cache_dir: str,
) -> None:
    """Download individual chunk FILES rather than their whole parent directory.

    Use this when only a few specific shards are needed (e.g. a token-capped
    run) so we don't pull every shard sitting next to them in the same GCS
    directory.  Local paths match :func:`_resolve_chunk_path` so the dataset
    config resolves them identically to the directory-download path.
    """
    for chunk in chunks:
        if chunk.uri.startswith("/"):
            # Local absolute path: symlink into cache dir (create its parent
            # first — _resolve_chunk_path nests under the last two path
            # components, which gsutil would otherwise create on download).
            local = _resolve_chunk_path(chunk, dataset_cache_dir)
            builder.ensure_directory(os.path.dirname(local))
            builder.run_command(f'[ -e "{local}" ] || ln -s "{chunk.uri}" "{local}"')
            continue

        gs_file = chunk.uri if chunk.uri.startswith("gs://") else remote_path(chunk.uri)
        local_file = _resolve_chunk_path(chunk, dataset_cache_dir)
        builder.ensure_directory(os.path.dirname(local_file))
        builder.download_from_gs(gs_file, local_file, directory=False, skip_existing=True)


# ---------------------------------------------------------------------------
# JolmoModel — unified pre-training / finetuning artifact
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JolmoModel(Artifact):
    """A model trained (or continued-trained) with JOLMo.

    If ``base_model`` is None, this is a from-scratch pre-training run.
    If ``base_model`` is set, model weights are loaded from that model's
    checkpoint before training begins (continued training / finetuning).
    """

    # --- Identity ---
    model_name: str = ""
    model_type: str = ""

    # --- Base model (None = train from scratch) ---
    base_model: Optional["JolmoModel"] = None
    base_model_checkpoint: str = "final"
    reload_optimizer: bool = False
    reload_trainer_state: bool = False

    # --- Data ---
    train_chunks: Tuple[Chunk, ...] = ()
    # Optional per-token CE mask (bool memmap, same length as each train chunk).
    # When set, must be 1:1 with train_chunks; False → label_ignore_index (-100).
    label_mask_chunks: Tuple[Chunk, ...] = ()
    validation_chunks: Tuple[Tuple[str, Chunk], ...] = ()
    # Optional per-token CE mask for validation (label must match validation_chunks).
    validation_label_mask_chunks: Tuple[Tuple[str, Chunk], ...] = ()
    validation_eval_interval: int = 1000
    # If set, cap the *unique* training set to this many tokens via a
    # SourceMixtureDatasetConfig (instead of consuming all of train_chunks).
    # Combine with n_tokens = epochs * dataset_requested_tokens to train for a
    # fixed number of genuine epochs over the capped data (data-repetition regime).
    dataset_requested_tokens: Optional[int] = None

    # If set, weight each training example's loss by w_i ∝ (i+1)^-loss_weight_alpha
    # (i = global dataset index), normalized to mean weight 1 — the power-law-loss
    # branch (see launch_jolmo/weighted_train_module.py). Requires that N (the
    # unique-instance count) be fixed, via either dataset_requested_tokens (FSL
    # token-cap path) or dataset_num_examples (padded one-doc-per-instance path).
    loss_weight_alpha: Optional[float] = None

    # If True, use a NumpyPaddedFSLDataset: one document per instance, each padded
    # to sequence_length with the padding masked out of the loss. This makes
    # batch["index"] == document index, so the 150-group loss weighting and the
    # per-group QA eval bin by person cleanly and shuffle-invariantly. The unique
    # set is capped by the data file itself (e.g. a first-N-docs shard), NOT by
    # dataset_requested_tokens (which the padded dataset does not support).
    padded_dataset: bool = False
    # For the padded path: the number of documents (== unique training instances
    # == people), N. Fixes N for power-law-loss normalization and group binning.
    dataset_num_examples: Optional[int] = None
    # If set (with loss_weight_alpha), bin the N examples into this many equal
    # index groups and weight every example in group g by (g+1)^-alpha instead of
    # (i+1)^-alpha — the per-group power-law-loss branch (see weighted_train_module).
    loss_weight_num_groups: Optional[int] = None

    # --- Per-group QA evaluation (Heavy-Tail Knowledge Task) ---
    # If set, attach QAGroupEvalCallback: at the end of EACH EPOCH it scores this
    # held-out QA eval .npz (built by scripts/make_qa_group_eval.py) and logs
    # CE/FTA/accuracy/ECE to W&B, overall and split by frequency group, plus
    # per-entity arrays to qa_eval_per_class_dir. Path must be readable on the
    # compute node (e.g. /data/user_data/<user>/bios/qa_group_eval.npz).
    # qa_eval_interval is optional extra in-loop evals every N steps (None = off).
    qa_eval_npz: Optional[str] = None
    qa_eval_interval: Optional[int] = None
    qa_eval_micro_batch_size: int = 64
    qa_eval_n_bins: int = 15
    qa_eval_on_finish: bool = True
    qa_eval_per_class_dir: Optional[str] = None

    # --- Tokenizer ---
    tokenizer: str = "allenai/OLMo-2-0425-1B-Instruct"

    # --- Training hyperparameters ---
    sequence_length: int = 2048
    global_batch_size: int = 524_288
    rank_microbatch_size: int = 524_288 // 8
    eval_rank_microbatch_size: Optional[int] = None

    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 2000
    optimizer: Literal["adamw", "adam", "muon"] = "adamw"
    muon_lr: float = 0.02
    muon_weight_decay: float = 0.1
    scheduler: Literal["cosine", "constant", "inv_sqrt", "wsd"] = "cosine"
    compile_model: bool = True
    max_grad_norm: float = 1.0
    parallelism: Literal["ddp", "fsdp"] = "fsdp"

    n_tokens: Optional[int] = None

    # --- Infrastructure ---
    num_processes: int = 8
    partition: str = "preempt"
    dp_param_dtype: str = "bfloat16"
    dp_reduce_dtype: str = "float32"

    save_overwrite: bool = True
    metrics_collect_interval: int = 10
    cancel_check_interval: int = 5
    save_interval: Optional[int] = 10_000
    ephemeral_save_interval: Optional[int] = 1_000

    experiment_name: Optional[str] = None
    init_seed: int = 12536
    weight_distance_interval: Optional[int] = 100

    # --- Post-processing flags ---
    unshard_checkpoint: bool = True
    convert_to_hf: bool = True
    upload: bool = True

    # Artifacts that must complete before training (for dependency tracking)
    data_dependencies: Tuple = ()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self):
        if self.model_type not in MODEL_ARCHS:
            raise ValueError(
                f"Unsupported model_type '{self.model_type}'. "
                f"Supported: {sorted(MODEL_ARCHS)}"
            )
        if not self.train_chunks:
            raise ValueError("train_chunks must be a non-empty tuple of Chunk")
        if self.reload_optimizer and self.base_model is None:
            raise ValueError("reload_optimizer requires base_model to be set")
        if self.reload_trainer_state and self.base_model is None:
            raise ValueError("reload_trainer_state requires base_model to be set")
        if (self.loss_weight_alpha is not None
                and self.dataset_requested_tokens is None
                and self.dataset_num_examples is None):
            raise ValueError(
                "loss_weight_alpha requires N (the unique instance count) to be fixed via "
                "either dataset_requested_tokens (FSL token-cap) or dataset_num_examples "
                "(padded one-doc-per-instance) so each example's rank i is well defined."
            )
        if self.padded_dataset and self.dataset_requested_tokens is not None:
            raise ValueError(
                "padded_dataset is incompatible with dataset_requested_tokens "
                "(SourceMixtureDatasetConfig); cap the unique set via the data file "
                "(a first-N-docs shard) and set dataset_num_examples instead."
            )

    # ------------------------------------------------------------------
    # Naming & paths
    # ------------------------------------------------------------------

    @property
    def run_name(self) -> str:
        return self.model_name

    @property
    def relpath(self) -> str:
        return f"JolmoModel/{self.run_name}"

    @property
    def olmo_model_relpath(self) -> str:
        return os.path.join(self.relpath, "final-unsharded")

    @property
    def hf_model_relpath(self) -> str:
        return os.path.join(self.relpath, "final-hf")

    @property
    def exists(self) -> bool:
        # Fast single-object check on the uploaded checkpoint that downstream
        # artifacts (CPT / perturb / eval) actually read. (HF dir isn't produced
        # for pretrain — convert_to_hf=False — so check final-unsharded/model.pt.)
        return blob_exists(remote_path(self.olmo_model_relpath, "model.pt"))

    # ------------------------------------------------------------------
    # Slurm requirements
    # ------------------------------------------------------------------

    def get_requirements(self):
        return {
            "gres": f"gpu:{self.num_processes}",
            "gpus": self.num_processes,
            "cpus": max(4, self.num_processes + 2),
            "mem": "256G",
            "partition": "preempt",
        }

    # ------------------------------------------------------------------
    # YAML config generation
    # ------------------------------------------------------------------

    def _vocab_size(self) -> int:
        return _tokenizer_config(self.tokenizer).padded_vocab_size()

    def _build_dataset_spec(
        self,
        train_paths: List[str],
        work_dir: str,
        label_mask_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Build the dataset config dict the trainer consumes.

        Either references the raw paths directly, or — when
        dataset_requested_tokens is set — caps the unique training set to a
        fixed token budget via a single-source SourceMixtureDatasetConfig.
        Capping the unique data is what makes "N epochs" well defined: the data
        loader wraps the capped dataset, so n_tokens = N * cap yields exactly N
        genuine passes.

        This is the single source of truth for "what data did this model train
        on": TrainExampleLossEvaluation rebuilds the dataset from this exact
        spec (same requested_tokens, global_batch_size, seed defaults, and
        source token counts) so it iterates the identical training instances.
        """
        # Padded one-doc-per-instance dataset: each EOS-delimited document becomes
        # a single instance padded to sequence_length, with the padding masked out
        # of the loss (label_mask). batch["index"] == document index, so per-group
        # loss weighting / eval bin by person. Incompatible with SourceMixture, so
        # the unique set is capped by the data file (a first-N-docs shard).
        if self.padded_dataset:
            return {
                "_CLASS_": "olmo_core.data.NumpyPaddedFSLDatasetConfig",
                "tokenizer": _build_tokenizer_yaml(self.tokenizer),
                "sequence_length": self.sequence_length,
                "work_dir": work_dir,
                "paths": train_paths,
                **(
                    {"label_mask_paths": label_mask_paths}
                    if label_mask_paths is not None
                    else {}
                ),
            }

        dataset_spec: Dict[str, Any] = {
            "_CLASS_": "olmo_core.data.NumpyFSLDatasetConfig",
            "tokenizer": _build_tokenizer_yaml(self.tokenizer),
            "sequence_length": self.sequence_length,
            "work_dir": work_dir,
        }
        if self.dataset_requested_tokens is not None:
            if label_mask_paths is not None:
                raise ValueError(
                    "label_mask_chunks is incompatible with dataset_requested_tokens "
                    "(SourceMixture does not support label_mask_paths)"
                )
            dataset_spec["source_mixture_config"] = {
                "_CLASS_": "olmo_core.data.source_mixture.SourceMixtureDatasetConfig",
                "requested_tokens": self.dataset_requested_tokens,
                "global_batch_size": self.global_batch_size,
                "processes": max(1, self.num_processes),
                "render_tables": False,
                "source_list": {
                    "_CLASS_": "olmo_core.data.source_mixture.SourceMixtureList",
                    "sources": [
                        {
                            "_CLASS_": "olmo_core.data.source_mixture.SourceMixtureConfig",
                            "source_name": "train",
                            "target_ratio": 1.0,
                            "paths": train_paths,
                        }
                    ],
                },
            }
        else:
            dataset_spec["paths"] = train_paths
            if label_mask_paths is not None:
                dataset_spec["label_mask_paths"] = label_mask_paths
        return dataset_spec

    def _build_yaml_config(
        self,
        save_folder: str,
        train_paths: List[str],
        val_datasets: Dict[str, List[str]],
        work_dir: str,
        label_mask_paths: Optional[List[str]] = None,
        validation_label_mask_datasets: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        vocab_size = self._vocab_size()

        if self.label_mask_chunks and len(self.label_mask_chunks) != len(self.train_chunks):
            raise ValueError(
                f"label_mask_chunks ({len(self.label_mask_chunks)}) must match "
                f"train_chunks ({len(self.train_chunks)}) 1:1"
            )

        dataset_spec = self._build_dataset_spec(
            train_paths,
            work_dir,
            label_mask_paths=label_mask_paths,
        )

        cfg: Dict[str, Any] = {
            "model": _build_model_spec(self.model_type, vocab_size),
            "dataset": dataset_spec,
            "data_loader": {
                "_CLASS_": "olmo_core.data.NumpyDataLoaderConfig",
                "global_batch_size": self.global_batch_size,
                "seed": 0,
                "num_workers": 4,
            },
            "train_module_type": "normal",
            "train_module": {
                "_CLASS_": "olmo_core.train.train_module.transformer.config.TransformerTrainModuleConfig",
                "rank_microbatch_size": self.rank_microbatch_size,
                # eval_rank_microbatch_size is deliberately NOT emitted: this
                # vendored TransformerTrainModuleConfig has no such field (it
                # errors on unknown keys) and nothing in the in-training eval
                # path reads it. The dataclass field still sizes standalone
                # ModelEvaluation batches.
                "max_sequence_length": self.sequence_length,
                "optim": _build_optimizer_spec(
                    self.optimizer, self.learning_rate, self.betas, self.weight_decay,
                    muon_lr=self.muon_lr, muon_weight_decay=self.muon_weight_decay,
                ),
                "scheduler": _build_scheduler_spec(self.scheduler, self.warmup_steps),
                "compile_model": self.compile_model,
                "dp_config": {
                    "_CLASS_": "olmo_core.train.train_module.transformer.config.TransformerDataParallelConfig",
                    "name": self.parallelism,
                    "param_dtype": self.dp_param_dtype,
                    "reduce_dtype": self.dp_reduce_dtype,
                },
                "autocast_precision": self.dp_param_dtype,
                "max_grad_norm": self.max_grad_norm,
                # Power-law per-example loss weighting: swap in the weighted train
                # module (still train_module_type="normal" — launch_from_yaml calls
                # cfg.train_module.build(model), which resolves this _CLASS_).
                **({
                    "_CLASS_": "launch_jolmo.weighted_train_module.PowerLawWeightedTransformerTrainModuleConfig",
                    "loss_weight_alpha": self.loss_weight_alpha,
                    # N = unique-instance count: document count for the padded path,
                    # else capped-tokens // seq_len for the FSL token-cap path.
                    "loss_weight_num_examples": (
                        self.dataset_num_examples if self.padded_dataset
                        else self.dataset_requested_tokens // self.sequence_length
                    ),
                    **({"loss_weight_num_groups": self.loss_weight_num_groups}
                       if self.loss_weight_num_groups is not None else {}),
                } if self.loss_weight_alpha is not None else {}),
            },
            "trainer": {
                "_CLASS_": "olmo_core.train.config.TrainerConfig",
                "save_folder": save_folder,
                "save_overwrite": self.save_overwrite,
                "metrics_collect_interval": self.metrics_collect_interval,
                "cancel_check_interval": self.cancel_check_interval,
                "callbacks": {
                    "config_saver": {
                        "_CLASS_": "olmo_core.train.callbacks.config_saver.ConfigSaverCallback",
                    },
                },
            },
            "init_seed": self.init_seed,
        }

        # Weight distance tracking
        if self.weight_distance_interval is not None:
            cfg["trainer"]["callbacks"]["weight_distance"] = {
                "_CLASS_": "olmo_core.train.callbacks.weight_distance.WeightDistanceCallback",
                "log_interval": self.weight_distance_interval,
            }

        # Per-group QA evaluation (Heavy-Tail Knowledge Task). Logs CE/FTA/acc/ECE
        # to W&B overall and per frequency group; dumps per-entity arrays to disk.
        if self.qa_eval_npz is not None:
            per_class_dir = self.qa_eval_per_class_dir
            if per_class_dir is not None:
                per_class_dir = os.path.join(per_class_dir, self.run_name)
            cfg["trainer"]["callbacks"]["qa_group_eval"] = {
                "_CLASS_": "launch_jolmo.qa_eval_callback.QAGroupEvalCallbackConfig",
                "eval_npz": self.qa_eval_npz,
                "eval_interval": self.qa_eval_interval,
                "micro_batch_size": self.qa_eval_micro_batch_size,
                "n_bins": self.qa_eval_n_bins,
                "eval_on_finish": self.qa_eval_on_finish,
                "per_class_dir": per_class_dir,
            }

        # Checkpointer (omit entirely when save_interval is None → no checkpoints)
        if self.save_interval is not None:
            checkpointer = {
                "_CLASS_": "olmo_core.train.callbacks.checkpointer.CheckpointerCallback",
                "save_interval": self.save_interval,
                "save_async": True,
            }
            if self.ephemeral_save_interval is not None:
                checkpointer["ephemeral_save_interval"] = self.ephemeral_save_interval
            cfg["trainer"]["callbacks"]["checkpointer"] = checkpointer

        # W&B
        wandb_project = Project.config.name
        if self.experiment_name:
            wandb_project = f"{wandb_project}.{self.experiment_name}"
        cfg["wandb"] = {
            "enabled": True,
            "project": wandb_project,
            "entity": "catheri4-carnegie-mellon-university",
            "name": self.run_name,
            "resume": "allow",
            "id": f"{self.run_name}-1",
            "config": {
                "model_type": self.model_type,
                "optimizer": self.optimizer,
                "scheduler": self.scheduler,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "betas": list(self.betas),
                "warmup_steps": self.warmup_steps,
                "max_grad_norm": self.max_grad_norm,
                "sequence_length": self.sequence_length,
                "global_batch_size": self.global_batch_size,
                "n_tokens": self.n_tokens,
                "dataset_requested_tokens": self.dataset_requested_tokens,
                "parallelism": self.parallelism,
                "dp_param_dtype": self.dp_param_dtype,
                "dp_reduce_dtype": self.dp_reduce_dtype,
                "compile_model": self.compile_model,
                "num_processes": self.num_processes,
                "tokenizer": self.tokenizer,
                "init_seed": self.init_seed,
                "muon_lr": self.muon_lr,
                "muon_weight_decay": self.muon_weight_decay,
                "base_model": self.base_model.model_name if self.base_model else None,
            },
        }

        # Token budget
        if self.n_tokens is not None:
            cfg["n_tokens"] = self.n_tokens

        # Validation datasets
        if val_datasets:
            cfg["validation_datasets"] = val_datasets
            cfg["validation_eval_interval"] = self.validation_eval_interval
            if validation_label_mask_datasets:
                cfg["validation_label_mask_datasets"] = validation_label_mask_datasets

        # Base model loading. The trainer resolves DCP via
        # Checkpointer.latest_checkpoint(load_path), which expects the run/save
        # DIRECTORY (it appends "final/" itself, or falls back to the latest
        # step<N>/). So load_path must be the base model's run dir — NOT
        # "<run>/final" (passing the leaf checkpoint dir makes latest_checkpoint
        # scan inside it for step<N>/ subdirs and fail with "No checkpoints
        # found"). For a non-default base_model_checkpoint (a specific step),
        # join it so latest_checkpoint resolves that step dir's own final/.
        # If the run has no DCP but does have final-unsharded/model.pt
        # (imported bases), launch_from_yaml falls back to loading that flat
        # state_dict as model weights only.
        if self.base_model is not None:
            base_dir = remote_path(self.base_model.relpath)
            cfg["load_path"] = (
                base_dir if self.base_model_checkpoint == "final"
                else os.path.join(base_dir, self.base_model_checkpoint)
            )
            if self.reload_optimizer:
                cfg["load_optim_state"] = True
            if self.reload_trainer_state:
                cfg["load_trainer_state"] = True
                cfg["load_data_loader_state"] = False
        else:
            cfg["load_path"] = None

        return cfg

    # ------------------------------------------------------------------
    # construct()
    # ------------------------------------------------------------------

    def construct(self, builder: Task):
        builder.set_env("WANDB_API_KEY", Project.config.wandb_api_key)
        # Prevent OOM from memory fragmentation during torch.compile() dry-run.
        # Reserved-but-unallocated memory can exceed free memory; expandable
        # segments let the allocator grow into new non-contiguous regions instead.
        builder.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        data_dir = local_path()
        cache_dir = local_cache_path()

        save_folder = remote_path(self.relpath)
        output_dir = os.path.join(data_dir, self.relpath)
        dataset_cache_dir = os.path.join(cache_dir, "datasets")
        work_dir = os.path.join(cache_dir, "training", self.run_name)

        builder.ensure_directory(output_dir)
        builder.ensure_directory(dataset_cache_dir)

        # Download data. When the unique training set is token-capped
        # (dataset_requested_tokens), fetch only the specific train shards
        # file-by-file instead of their whole GCS directory — a capped run reads
        # far less than one shard, so pulling every sibling shard wastes
        # bandwidth and disk. Validation chunks always download by directory.
        val_chunks_only = [c for _, c in self.validation_chunks]
        val_mask_chunks_only = [c for _, c in self.validation_label_mask_chunks]
        if self.dataset_requested_tokens is not None or self.padded_dataset:
            # Token-capped or padded run: pull only the specific train shard(s)
            # file-by-file (the padded path trains on a single first-N-docs shard).
            _download_chunk_files(builder, list(self.train_chunks), dataset_cache_dir)
            _download_chunk_dirs(builder, val_chunks_only, dataset_cache_dir)
            if val_mask_chunks_only:
                _download_chunk_dirs(builder, val_mask_chunks_only, dataset_cache_dir)
        else:
            all_val = val_chunks_only + val_mask_chunks_only
            _download_chunk_dirs(
                builder, list(self.train_chunks) + all_val, dataset_cache_dir
            )

        # Resolve training data paths
        train_paths: List[str] = [
            _resolve_chunk_path(c, dataset_cache_dir) for c in self.train_chunks
        ]
        label_mask_paths: Optional[List[str]] = None
        if self.label_mask_chunks:
            # Masks may live next to ids under a sibling directory — download
            # their parent dirs (or files when token-capped / padded).
            if self.dataset_requested_tokens is not None or self.padded_dataset:
                _download_chunk_files(
                    builder, list(self.label_mask_chunks), dataset_cache_dir
                )
            else:
                _download_chunk_dirs(
                    builder, list(self.label_mask_chunks), dataset_cache_dir
                )
            label_mask_paths = [
                _resolve_chunk_path(c, dataset_cache_dir) for c in self.label_mask_chunks
            ]

        # Resolve validation data paths (multiple shards may share one label).
        val_datasets = _group_labeled_chunk_paths(self.validation_chunks, dataset_cache_dir)
        validation_label_mask_datasets: Optional[Dict[str, List[str]]] = None
        if self.validation_label_mask_chunks:
            validation_label_mask_datasets = _group_labeled_chunk_paths(
                self.validation_label_mask_chunks, dataset_cache_dir
            )

        # Generate and write YAML config
        yaml_config = self._build_yaml_config(
            save_folder, train_paths, val_datasets, work_dir,
            label_mask_paths=label_mask_paths,
            validation_label_mask_datasets=validation_label_mask_datasets,
        )
        config_path = os.path.join(output_dir, "config.yaml")
        builder.create_yaml_file(config_path, yaml_config)

        # Launch training (rdzv-endpoint=localhost:0 picks a free port, avoids shell quoting issues)
        launch_script = os.path.join(
            Project.config.jolmo_path, "src", "scripts", "launch_from_yaml.py",
        )
        # When the QA-group eval callback is enabled, its _CLASS_ lives in
        # launch_jolmo.qa_eval_callback — make the project root importable on the
        # compute node so launch_from_yaml can resolve it.
        env_prefix = ""
        if self.qa_eval_npz is not None:
            env_prefix = f"PYTHONPATH={Project.config.code_path}:${{PYTHONPATH:-}} "
        builder.run_command(
            f"{env_prefix}torchrun --rdzv-endpoint=localhost:0 --rdzv-backend=c10d "
            f"--nproc_per_node={self.num_processes} {launch_script} {config_path}"
        )

        # Post-training: unshard, convert to HF, upload
        self._postprocess(builder, save_folder, output_dir)

    def _postprocess(self, builder: Task, save_folder: str, output_dir: str,
                     olmo_upload_relpath: Optional[str] = None):
        """Unshard the final checkpoint, convert to HF format, and upload.

        olmo_upload_relpath: GCS relpath for the unsharded upload.  Defaults to
        self.olmo_model_relpath.  Pass CPTModel.olmo_model_relpath when calling
        from CPTModel.construct so the upload goes to CPTModel/ not JolmoModel/.
        """
        jolmo_path = Project.config.jolmo_path
        upload_relpath = olmo_upload_relpath if olmo_upload_relpath is not None else self.olmo_model_relpath

        final_checkpoint_dir = os.path.join(save_folder, "final")
        unsharded_dir = os.path.join(output_dir, "final-unsharded")
        hf_dir = os.path.join(output_dir, "final-hf")

        # Unshard. Resolve final/ — or the latest step*/ when final/ was never
        # written (the last step coincided with a scheduled save, so the
        # checkpointer skipped writing final/). unshard_resolve.py picks the
        # right checkpoint and copies its config.json into unsharded_dir.
        if self.unshard_checkpoint:
            unshard_script = os.path.join(jolmo_path, "src", "scripts", "unshard.py")
            resolve_script = os.path.join(
                Project.config.code_path, "scripts", "unshard_resolve.py",
            )
            builder.run_command(
                f'python3 {resolve_script} --save-folder "{save_folder}" '
                f'--output-dir "{unsharded_dir}" --work-dir "{output_dir}/cache" '
                f'--unshard-script "{unshard_script}"'
            )

        # Convert to HuggingFace format
        if self.convert_to_hf:
            convert_script = os.path.join(
                jolmo_path, "src", "examples", "huggingface", "convert_checkpoint_to_hf.py",
            )
            builder.run_command(
                f'python3 {convert_script} -i "{final_checkpoint_dir}" -o "{hf_dir}" '
                f"--skip-validation --device cpu"
            )

        if self.upload:
            # Upload unsharded OLMo checkpoint
            if self.unshard_checkpoint:
                _upload_to_gs_with_retry(
                    builder,
                    unsharded_dir,
                    remote_path(upload_relpath),
                    directory=True,
                    contents=True,
                )
                # Upload config.json alongside the unsharded weights. It was
                # placed into unsharded_dir by unshard_resolve.py (sourced from
                # whichever checkpoint was unsharded), so read it locally.
                _upload_to_gs_with_retry(
                    builder,
                    os.path.join(unsharded_dir, "config.json"),
                    remote_path(upload_relpath, "config.json"),
                    directory=False,
                )
            # Upload HuggingFace checkpoint
            if self.convert_to_hf:
                _upload_to_gs_with_retry(
                    builder,
                    hf_dir,
                    remote_path(self.hf_model_relpath),
                    directory=True,
                    contents=True,
                )
            # Free node-local staging once everything is uploaded — a large sweep
            # otherwise fills the scratch disk with finished artifacts' staging.
            # Only reached when the uploads above succeeded (script is set -e).
            builder.run_command(f'rm -rf -- "{output_dir}"')


# ---------------------------------------------------------------------------
# CPT dataset registry
# ---------------------------------------------------------------------------

_CPT_GS = "gs://cmu-gpucloud-jspringe/shared/datasets/OLMo"

# Mirrors LIST_OF_CPT_FILES["dclm"] exactly.
# train_tokens (where present) are raw token counts = artifacts.py "M" values × 1_000_000.
# label_mask paths are omitted — CPTModel uses NumpyFSLDatasetConfig (next-token prediction).
CPT_DATASETS: Dict[str, Dict[str, Any]] = {
    "tulu": {
        "train": (
            Chunk(uri=f"{_CPT_GS}/allenai_tulu-3-sft-mixture/train/input_ids.npy"),
        ),
        "val": {
            "tulu-validation": Chunk(uri=f"{_CPT_GS}/allenai_tulu-3-sft-mixture/val/input_ids-tulu.npy"),
        },
    },
    "starcoder": {
        "train": (
            Chunk(uri=f"{_CPT_GS}/bigcode_starcoderdata/train/input_ids.npy"),
        ),
        "val": {
            "starcoder-validation": Chunk(uri=f"{_CPT_GS}/bigcode_starcoderdata/val/input_ids-starcoder.npy"),
        },
    },
    "musicpile": {
        "train": (
            Chunk(uri=f"{_CPT_GS}/musicpile/train/input_ids.npy"),
        ),
        "val": {
            "musicpile-validation": Chunk(uri=f"{_CPT_GS}/musicpile/val/input_ids-musicpile.npy"),
        },
    },
    "alpaca": {
        "train": (
            Chunk(uri=f"{_CPT_GS}/tatsu-lab_alpaca/train/input_ids.npy"),
        ),
        "val": {
            "alpaca-validation": Chunk(uri=f"{_CPT_GS}/tatsu-lab_alpaca/val/input_ids-alpaca.npy"),
        },
        "train_tokens": 3_600_000,   # dataset is small; cap from artifacts.py
    },
    "gsm8k": {
        "train": (
            Chunk(uri=f"{_CPT_GS}/openai_gsm8k/train/input_ids.npy"),
        ),
        "val": {
            "gsm8k-validation": Chunk(uri=f"{_CPT_GS}/openai_gsm8k/val/input_ids-gsm8k.npy"),
        },
        "train_tokens": 1_200_000,   # dataset is small; cap from artifacts.py
    },
    "siqa": {
        "train": (
            Chunk(uri=f"{_CPT_GS}/allenai_social_i_qa/train/input_ids.npy"),
        ),
        "val": {
            "siqa-validation": Chunk(uri=f"{_CPT_GS}/allenai_social_i_qa/val/input_ids-siqa.npy"),
        },
        "train_tokens": 1_180_000,   # dataset is small; cap from artifacts.py
    },
    "open-platypus": {
        "train": (
            Chunk(uri=f"{_CPT_GS}/garage-bAInd_Open-Platypus/train/input_ids.npy"),
        ),
        "val": {
            "open-platypus-validation": Chunk(uri=f"{_CPT_GS}/garage-bAInd_Open-Platypus/val/input_ids-open-platypus.npy"),
        },
        "train_tokens": 7_000_000,
    },
    "stackmathqa": {
        "train": (
            Chunk(uri=f"{_CPT_GS}/math-ai_StackMathQA/train/input_ids.npy"),
        ),
        "val": {
            "stackmathqa-validation": Chunk(uri=f"{_CPT_GS}/math-ai_StackMathQA/val/input_ids-stackmathqa.npy"),
        },
    },
    "helpsteer": {
        "train": (
            Chunk(uri=f"{_CPT_GS}/nvidia_HelpSteer/train/input_ids.npy"),
        ),
        "val": {
            "helpsteer-validation": Chunk(uri=f"{_CPT_GS}/nvidia_HelpSteer/val/input_ids-helpsteer.npy"),
        },
    },
}

# Pretrain val chunks appended to every CPT run's validation set,
# matching how LIST_OF_CPT_FILES merges pretrain val in catastrophic-forgetting.
_DIVERSITY_GS = "gs://cmu-gpucloud-jspringe/outputs/diversity.pretraining.v2"
_PRETRAIN_VAL_CHUNKS = tuple(
    (n, Chunk(uri=f"{_DIVERSITY_GS}/ValidationDataset/Tokenized/{n}.bin"))
    for n in ("C4_val", "Reddit_val", "Wiki_val", "Books_val")
)


# ---------------------------------------------------------------------------
# CPTModel
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CPTModel(Artifact):
    """Continual pre-training on a CPT dataset, starting from a JolmoModel checkpoint."""

    pretrained_model: JolmoModel = None  # type: ignore[assignment]
    cpt_dataset: str = "tulu"
    train_tokens: int = 100_000_000        # CPT budget in tokens

    # DCLM replay: fraction of the CPT token budget drawn from pretraining data
    # (via a two-source SourceMixtureDatasetConfig). 0.0 = plain finetuning, and
    # the run name is unchanged so existing artifacts keep matching. > 0 needs
    # replay_chunks (the DCLM file(s) to sample the replay tokens from).
    replay_dclm: float = 0.0
    replay_chunks: Tuple["Chunk", ...] = ()

    # CPT batch & context — decoupled from the pretrained model so CPT can run a
    # smaller batch/context than pretrain (defaults: 64K-token batch, 1024 ctx).
    sequence_length: int = 1024
    global_batch_size: int = 65_536
    rank_microbatch_size: int = 65_536 // 8

    # Override optimizer/LR independently from the pretrained model
    optimizer: Literal["adamw", "adam", "muon"] = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 20
    muon_lr: float = 0.02
    muon_weight_decay: float = 0.1
    scheduler: Literal["cosine", "constant", "inv_sqrt", "wsd"] = "cosine"
    max_grad_norm: float = 1.0

    # Infrastructure
    num_processes: int = 1
    save_interval: Optional[int] = None
    ephemeral_save_interval: Optional[int] = None
    unshard_checkpoint: bool = True
    convert_to_hf: bool = False
    upload: bool = True
    experiment_name: Optional[str] = "cpt"
    weight_distance_interval: Optional[int] = 5

    @property
    def run_name(self) -> str:
        lr_str = f"{self.learning_rate:.1e}".replace("e-0", "e-")
        tk_str = f"{self.train_tokens // 1_000_000}M"
        base = f"{self.pretrained_model.run_name}-CPT-{self.cpt_dataset}-{tk_str}-{self.optimizer}-lr{lr_str}-exp"
        if self.optimizer == "muon":
            muon_lr_str = f"{self.muon_lr:.1e}".replace("e-0", "e-")
            base = f"{base}-muonlr{muon_lr_str}"
        if self.replay_dclm > 0:
            base = f"{base}-replay{self.replay_dclm:g}"
        return base

    @property
    def relpath(self) -> str:
        return f"CPTModel/{self.cpt_dataset}/{self.run_name}"

    @property
    def olmo_model_relpath(self) -> str:
        return os.path.join(self.relpath, "final-unsharded")

    @property
    def hf_model_relpath(self) -> str:
        return os.path.join(self.relpath, "final-hf")

    @property
    def validation_chunks(self) -> Tuple[Tuple[str, "Chunk"], ...]:
        """CPT-specific val + pretrain val, mirroring LIST_OF_CPT_FILES val merge."""
        cpt_val: Dict[str, "Chunk"] = CPT_DATASETS[self.cpt_dataset]["val"]
        return tuple(cpt_val.items()) + _PRETRAIN_VAL_CHUNKS

    @property
    def exists(self) -> bool:
        # Skip re-training (tier-0) once the finetune checkpoint is on GCS, so
        # downstream eval stages don't re-run the CPT/finetune as a dependency.
        # Fast single-object lookup (no directory listing).
        return blob_exists(remote_path(self.olmo_model_relpath, "model.pt"))

    def get_requirements(self):
        return {
            "gres": f"gpu:{self.num_processes}",
            "gpus": self.num_processes,
            "cpus": self.num_processes + 2,
            "mem": "128G",
            "partition": "preempt",
            "exclude": "babel-m9-16",
        }

    def construct(self, builder: Task):
        builder.set_env("WANDB_API_KEY", Project.config.wandb_api_key)
        builder.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        cpt_info = CPT_DATASETS[self.cpt_dataset]
        train_chunks: Tuple[Chunk, ...] = cpt_info["train"]
        cpt_val: Dict[str, Chunk] = cpt_info["val"]

        # Honour per-dataset token cap (small datasets like alpaca/gsm8k/siqa)
        # matching the "train_tokens" field in LIST_OF_CPT_FILES["dclm"].
        effective_tokens = cpt_info.get("train_tokens", self.train_tokens)

        # Validation = CPT-specific val + pretrain val (mirrors catastrophic-forgetting
        # merge of LIST_OF_PRETRAIN_FILES val into LIST_OF_CPT_FILES val)
        validation_chunks: Tuple[Tuple[str, Chunk], ...] = (
            tuple(cpt_val.items()) + _PRETRAIN_VAL_CHUNKS
        )

        data_dir = local_path()
        cache_dir = local_cache_path()
        save_folder = remote_path(self.relpath)
        output_dir = os.path.join(data_dir, self.relpath)
        dataset_cache_dir = os.path.join(cache_dir, "datasets")
        work_dir = os.path.join(cache_dir, "training", self.run_name)

        builder.ensure_directory(output_dir)
        builder.ensure_directory(dataset_cache_dir)

        all_chunks = list(train_chunks) + [c for _, c in validation_chunks]
        _download_chunk_dirs(builder, all_chunks, dataset_cache_dir)
        if self.replay_dclm > 0:
            if not self.replay_chunks:
                raise ValueError(
                    f"{self.run_name}: replay_dclm={self.replay_dclm} requires replay_chunks"
                )
            # Replay sources are single large files (a DCLM shard); download
            # file-by-file into the shared cache so all tasks reuse one copy.
            _download_chunk_files(builder, list(self.replay_chunks), dataset_cache_dir)

        train_paths = [_resolve_chunk_path(c, dataset_cache_dir) for c in train_chunks]
        val_datasets = {
            label: [_resolve_chunk_path(c, dataset_cache_dir)]
            for label, c in validation_chunks
        }

        # Reuse JolmoModel's YAML builder; temporarily substitute fields
        pseudo = JolmoModel(
            model_name=self.run_name,
            model_type=self.pretrained_model.model_type,
            tokenizer=self.pretrained_model.tokenizer,
            sequence_length=self.sequence_length,
            global_batch_size=self.global_batch_size,
            rank_microbatch_size=self.rank_microbatch_size,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=self.betas,
            warmup_steps=self.warmup_steps,
            muon_lr=self.muon_lr,
            muon_weight_decay=self.muon_weight_decay,
            scheduler=self.scheduler,
            max_grad_norm=self.max_grad_norm,
            compile_model=self.pretrained_model.compile_model,
            parallelism=self.pretrained_model.parallelism,
            dp_param_dtype=self.pretrained_model.dp_param_dtype,
            dp_reduce_dtype=self.pretrained_model.dp_reduce_dtype,
            n_tokens=effective_tokens,
            save_interval=self.save_interval,
            ephemeral_save_interval=self.ephemeral_save_interval,
            save_overwrite=True,
            num_processes=self.num_processes,
            experiment_name=self.experiment_name,
            unshard_checkpoint=self.unshard_checkpoint,
            convert_to_hf=self.convert_to_hf,
            upload=self.upload,
            weight_distance_interval=self.weight_distance_interval,
            base_model=self.pretrained_model,
            train_chunks=(train_chunks[0],),  # placeholder, overridden below
        )
        yaml_config = pseudo._build_yaml_config(save_folder, train_paths, val_datasets, work_dir)

        if self.replay_dclm > 0:
            # Replace the plain-paths dataset with a two-source mixture:
            # (1 - r) of the budget from the CPT dataset, r from DCLM replay.
            # Same spec shape as the proven single-source epoch-cap path in
            # _build_dataset_spec. The CPT source may be smaller than its share
            # of the budget (gsm8k etc.), so allow generous repetition.
            r = self.replay_dclm
            replay_paths = [_resolve_chunk_path(c, dataset_cache_dir) for c in self.replay_chunks]
            ds_spec = yaml_config["dataset"]
            ds_spec.pop("paths", None)
            ds_spec["source_mixture_config"] = {
                "_CLASS_": "olmo_core.data.source_mixture.SourceMixtureDatasetConfig",
                "requested_tokens": effective_tokens,
                "global_batch_size": self.global_batch_size,
                "processes": max(1, self.num_processes),
                "render_tables": False,
                "source_list": {
                    "_CLASS_": "olmo_core.data.source_mixture.SourceMixtureList",
                    "sources": [
                        {
                            "_CLASS_": "olmo_core.data.source_mixture.SourceMixtureConfig",
                            "source_name": self.cpt_dataset,
                            "target_ratio": 1.0 - r,
                            "max_repetition_ratio": 100.0,
                            "paths": train_paths,
                        },
                        {
                            "_CLASS_": "olmo_core.data.source_mixture.SourceMixtureConfig",
                            "source_name": "dclm-replay",
                            "target_ratio": r,
                            "paths": replay_paths,
                        },
                    ],
                },
            }

        config_path = os.path.join(output_dir, "config.yaml")
        builder.create_yaml_file(config_path, yaml_config)

        launch_script = os.path.join(
            Project.config.jolmo_path, "src", "scripts", "launch_from_yaml.py",
        )
        builder.run_command(
            f"torchrun --rdzv-endpoint=localhost:0 --rdzv-backend=c10d "
            f"--nproc_per_node={self.num_processes} {launch_script} {config_path}"
        )
        pseudo._postprocess(builder, save_folder, output_dir,
                            olmo_upload_relpath=self.olmo_model_relpath)


# ---------------------------------------------------------------------------
# PerturbedModel
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PerturbedModel(Artifact):
    """Perturb a JolmoModel's weights by adding scaled Gaussian noise.

    Noise scale per tensor: ``std = gamma * ||W||_F / sqrt(numel)`` (matches
    ``perturb_weights.py``), so ``||noise||_F / ||W||_F ≈ gamma``.

    When ``param_name`` is set, only that state-dict key is noised (per-matrix
    forgetting sweeps); otherwise every float parameter is perturbed. Naming
    mirrors ``perturb_weights.get_perturbed_model_name`` so
    ``gcs_dir/run_name/final-unsharded/model.pt`` is where the script writes.
    """

    source_model: "JolmoModel" = None  # type: ignore[assignment]
    gamma: float = 0.01
    seed: int = 64
    # None = perturb all float params; else a single matrix weight name.
    param_name: Optional[str] = None

    @property
    def run_name(self) -> str:
        # Must match perturb_weights.get_perturbed_model_name() exactly
        return _perturbed_run_name(
            self.source_model.run_name, self.gamma, self.param_name,
        )

    @property
    def relpath(self) -> str:
        return f"PerturbedModel/{self.run_name}"

    @property
    def olmo_model_relpath(self) -> str:
        return os.path.join(self.relpath, "final-unsharded")

    @property
    def exists(self) -> bool:
        return blob_exists(remote_path(self.olmo_model_relpath, "model.pt"))


    def get_requirements(self):
        return {"cpus": 4, "gres": "gpu:1", "mem": "64G",  "partition": "preempt", "exclude": "babel-m9-16"}

    def construct(self, builder: Task):
        perturb_script = os.path.join(
            Project.config.code_path, "new_utils", "perturb_weights.py",
        )
        # perturb_weights.py expects:
        #   input:  {gcs_dir}/{model_name}/final-unsharded/model.pt
        #   output: {output_gcs_dir}/{model_name}_perturbed_{sigma_str}[_param_…]/final-unsharded/model.pt
        src_parent = remote_path(self.source_model.relpath).rsplit("/", 1)[0]  # strip run_name
        cmd = (
            f"python3 {perturb_script} "
            f"--gcs_dir {src_parent} "
            f"--model_name {self.source_model.run_name} "
            f"--sigma {self.gamma} "
            f"--seed {self.seed} "
            f"--output_gcs_dir {remote_path('PerturbedModel')}"
        )
        if self.param_name is not None:
            cmd += f" --param-name {self.param_name}"
        builder.run_command(cmd)


# Default seeds for multi-direction (independent Gaussian noise draws).
DEFAULT_PERTURB_SEEDS: Tuple[int, ...] = tuple(range(10))


@dataclass(frozen=True)
class MultiSeedPerturbedModel(Artifact):
    """One job: 10 (or N) full-weight perturbations of a base model at fixed γ.

    Layout under GCS::

        PerturbedModel/{base}_perturbed_{γ}/
          seed_000/final-unsharded/model.pt
          …
          seed_009/final-unsharded/model.pt

    Each seed is an independent Gaussian direction
    (``std = γ · ‖W‖_F / √numel`` per tensor).
    """

    source_model: "JolmoModel" = None  # type: ignore[assignment]
    gamma: float = 0.02
    seeds: Tuple[int, ...] = DEFAULT_PERTURB_SEEDS

    @property
    def run_name(self) -> str:
        return _perturbed_run_name(self.source_model.run_name, self.gamma)

    @property
    def relpath(self) -> str:
        return f"PerturbedModel/{self.run_name}"

    def seed_olmo_relpath(self, seed: int) -> str:
        return os.path.join(self.relpath, f"seed_{seed:03d}", "final-unsharded")

    @property
    def exists(self) -> bool:
        if not self.seeds:
            return False
        return all(
            blob_exists(remote_path(self.seed_olmo_relpath(s), "model.pt"))
            for s in self.seeds
        )

    def get_requirements(self):
        return {
            "cpus": 4,
            "gres": "gpu:1",
            "mem": "64G",
            "partition": "preempt",
            "exclude": "babel-m9-16",
            "time": "12:00:00",
        }

    def construct(self, builder: Task):
        perturb_script = os.path.join(
            Project.config.code_path, "new_utils", "perturb_weights.py",
        )
        src_parent = remote_path(self.source_model.relpath).rsplit("/", 1)[0]
        seeds_csv = ",".join(str(s) for s in self.seeds)
        builder.run_command(
            f"python3 {perturb_script} "
            f"--gcs_dir {src_parent} "
            f"--model_name {self.source_model.run_name} "
            f"--sigma {self.gamma} "
            f"--seeds {seeds_csv} "
            f"--output_gcs_dir {remote_path('PerturbedModel')}"
        )


# ---------------------------------------------------------------------------
# InterpolatedModel
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InterpolatedModel(Artifact):
    """Convex combination of a pretrained and a finetuned checkpoint.

        W_interp = alpha * W_pretrained + (1 - alpha) * W_finetuned   (per tensor)

    At ``alpha = 1`` this is the pretrained model, at ``alpha = 0`` the
    finetuned model. The interpolated weights are written as a new unsharded
    OLMo checkpoint by ``new_utils/interpolate_weights.py`` so that
    ``InterpolatedModel/<run_name>/final-unsharded/model.pt`` is exactly where
    the eval pipeline reads. The finetuned model supplies the architecture /
    config (its checkpoint dir is copied first), so the two checkpoints must
    share the same architecture (same base JolmoModel).
    """

    pretrained_model: JolmoModel = None  # type: ignore[assignment]
    finetuned_model: Any = None          # CPTModel | JolmoModel
    alpha: float = 0.5                   # weight on the pretrained model

    @property
    def run_name(self) -> str:
        a_str = f"{self.alpha:.2f}".replace(".", "_")
        return f"{self.finetuned_model.run_name}-interp-a{a_str}"

    @property
    def relpath(self) -> str:
        return f"InterpolatedModel/{self.run_name}"

    @property
    def olmo_model_relpath(self) -> str:
        return os.path.join(self.relpath, "final-unsharded")

    @property
    def exists(self) -> bool:
        return blob_exists(remote_path(self.olmo_model_relpath, "model.pt"))

    def get_requirements(self):
        return {"cpus": 4, "gres": "gpu:1", "mem": "64G",
                "partition": "general", "exclude": "babel-m9-16"}

    def construct(self, builder: Task):
        interp_script = os.path.join(
            Project.config.code_path, "new_utils", "interpolate_weights.py",
        )
        pt_parent = remote_path(self.pretrained_model.relpath).rsplit("/", 1)[0]
        ft_parent = remote_path(self.finetuned_model.relpath).rsplit("/", 1)[0]
        builder.run_command(
            f"python3 {interp_script} "
            f"--pt_gcs_dir {pt_parent} "
            f"--pt_model_name {self.pretrained_model.run_name} "
            f"--ft_gcs_dir {ft_parent} "
            f"--ft_model_name {self.finetuned_model.run_name} "
            f"--alpha {self.alpha} "
            f"--output_gcs_dir {remote_path('InterpolatedModel')} "
            f"--output_model_name {self.run_name}"
        )


# ---------------------------------------------------------------------------
# Helpers for resolving validation chunks across all model types
# ---------------------------------------------------------------------------

def _get_val_chunks_for(model: Any) -> Tuple[Tuple[str, Chunk], ...]:
    """Return (label, Chunk) pairs for whichever artifact type is passed."""
    if isinstance(model, (JolmoModel, CPTModel)):
        return model.validation_chunks
    elif isinstance(model, (PerturbedModel, MultiSeedPerturbedModel)):
        return _get_val_chunks_for(model.source_model)
    elif isinstance(model, InterpolatedModel):
        # Evaluate on the finetuned model's val sets (CPT val + pretrain val).
        return _get_val_chunks_for(model.finetuned_model)
    return ()


# ---------------------------------------------------------------------------
# Helper: resolve the root JolmoModel from any artifact type
# ---------------------------------------------------------------------------

def _get_base_jolmo(model: Any) -> JolmoModel:
    """Drill through PerturbedModel / CPTModel to the root JolmoModel so we can
    read training hyper-params (sequence_length, etc.)."""
    if isinstance(model, JolmoModel):
        return model
    elif isinstance(model, CPTModel):
        return _get_base_jolmo(model.pretrained_model)
    elif isinstance(model, (PerturbedModel, MultiSeedPerturbedModel)):
        return _get_base_jolmo(model.source_model)
    elif isinstance(model, InterpolatedModel):
        return _get_base_jolmo(model.finetuned_model)
    raise TypeError(f"Cannot resolve base JolmoModel from {type(model)}")


# ---------------------------------------------------------------------------
# ModelEvaluation — mirrors catastrophic-forgetting ModelEvaluation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelEvaluation(Artifact):
    """Standalone perplexity evaluation using JOLMo/src/scripts/validate.py.

    Uses the same script and the same evaluation parameters (sequence_length,
    eval_rank_microbatch_size) as the training run so results are directly
    comparable to the validation metrics logged during training.

    Mirrors the structure of catastrophic-forgetting/launch/artifacts.py ModelEvaluation.
    """

    model: Any = None   # JolmoModel | CPTModel | PerturbedModel | InterpolatedModel
    device: str = "cuda"
    model_format: str = "olmo"   # "olmo" or "hf"
    # Optional override of the validation datasets. When non-empty, these
    # (label, Chunk) pairs are evaluated instead of the model's own
    # validation_chunks — used to score a model on alternative eval sets
    # (e.g. the typo-corrupted C4 variants). `eval_tag` namespaces the output
    # JSON so these results don't collide with the default eval.
    val_chunks_override: Tuple[Tuple[str, Chunk], ...] = ()
    eval_tag: str = ""
    # Cap the number of sequence_length-token sequences scored per dataset for
    # the model's own (or override) val sets. None = score the whole file.
    val_max_instances: Optional[int] = 1024
    # Extra validation datasets scored IN ADDITION to the model's own (or the
    # override) val chunks, appended to the same eval JSON. Used to fold a
    # held-out DCLM shard into the existing pretrain eval pipeline without a
    # separate stage. These chunks are downloaded file-by-file (not by whole
    # parent dir) since a DCLM "part-NNN/" dir holds 5 shards (~74 GiB) but only
    # one shard is the eval set.
    extra_val_chunks: Tuple[Tuple[str, Chunk], ...] = ()
    # Cap the number of sequence_length-token sequences scored for the
    # extra_val_chunks datasets only (the model's own val sets stay uncapped, so
    # their numbers don't change). A held-out DCLM shard is ~2.5B tokens; a few
    # thousand sequences give a stable, fast held-out loss. None = score all.
    extra_val_max_instances: Optional[int] = None
    # Artifacts that must complete before this eval (dependency ordering only).
    # Used to make the typo evals wait on the TypoEvalData build artifact.
    data_dependencies: Tuple = ()

    @property
    def run_name(self) -> str:
        return self.model.run_name

    @property
    def eval_basename(self) -> str:
        """run_name plus the optional eval tag (used for output file names)."""
        return f"{self.run_name}-{self.eval_tag}" if self.eval_tag else self.run_name

    @property
    def relpath(self) -> str:
        return f"ModelEvaluation/{self.eval_basename}-eval.json"

    @property
    def exists(self) -> bool:
        # Single-object metadata lookup (~60 ms) instead of listing the whole
        # ModelEvaluation dir — the dir holds a huge number of eval JSONs, so a
        # listing scales with total file count while this scales only with the
        # number of evals being checked.
        return blob_exists(remote_path(self.relpath))

    def get_requirements(self):
        return {
            "gpus": 1,
            "cpus": 4,
            "mem": "64G",
            "partition": "preempt",
            "exclude": "babel-m9-16",
        }

    def construct(self, builder: Task):
        builder.set_env("WANDB_API_KEY", Project.config.wandb_api_key)

        # Inherit sequence_length and eval batch size from the source training run
        # so evaluation uses exactly the same parameters as training validation.
        base = _get_base_jolmo(self.model)
        chunk_size = base.sequence_length
        if base.eval_rank_microbatch_size is not None:
            batch_size = max(1, base.eval_rank_microbatch_size // chunk_size)
        else:
            batch_size = max(1, base.rank_microbatch_size // chunk_size)

        data_dir = local_path()
        cache_dir = local_cache_path()

        output_json = os.path.join(data_dir, self.relpath)
        output_dir = os.path.dirname(output_json)
        dataset_cache_dir = os.path.join(cache_dir, "datasets")
        builder.ensure_directory(output_dir)
        builder.ensure_directory(dataset_cache_dir)

        # Download model checkpoint (unsharded OLMo or HF directory)
        checkpoint_gs = remote_path(self.model.olmo_model_relpath)
        checkpoint_local = os.path.join(data_dir, self.model.olmo_model_relpath)
        builder.download_from_gs(checkpoint_gs, checkpoint_local, directory=True)
        # Explicitly download config.json even if the directory already existed
        # (prior runs may have downloaded the directory before config.json was uploaded).
        builder.download_from_gs(
            remote_path(self.model.olmo_model_relpath, "config.json"),
            os.path.join(checkpoint_local, "config.json"),
            directory=False,
            skip_existing=False,
            no_fail=True,
        )

        # Resolve validation data (override takes precedence over the model's own).
        # Whole parent directories are downloaded for these (small .bin val sets).
        val_chunks = self.val_chunks_override or _get_val_chunks_for(self.model)
        _download_chunk_dirs(builder, [c for _, c in val_chunks], dataset_cache_dir)

        validation_datasets = [
            {
                "name": label,
                "paths": [_resolve_chunk_path(chunk, dataset_cache_dir)],
                **(
                    {"max_instances": self.val_max_instances}
                    if self.val_max_instances is not None
                    else {}
                ),
            }
            for label, chunk in val_chunks
        ]

        # Extra datasets (e.g. held-out DCLM) appended to the same eval JSON.
        # Downloaded file-by-file so a single DCLM shard doesn't pull its whole
        # ~74 GiB part dir, and capped per-dataset via max_instances so the work
        # stays bounded while the model's own val sets above remain uncapped.
        extra_val_chunks = _coerce_val_chunk_pairs(self.extra_val_chunks)
        if extra_val_chunks:
            _download_chunk_files(builder, [c for _, c in extra_val_chunks], dataset_cache_dir)
            for label, chunk in extra_val_chunks:
                ds_entry = {"name": label, "paths": [_resolve_chunk_path(chunk, dataset_cache_dir)]}
                if self.extra_val_max_instances is not None:
                    ds_entry["max_instances"] = self.extra_val_max_instances
                validation_datasets.append(ds_entry)

        # Build validate.py YAML config — same format used by the training framework.
        # chunk_size = sequence_length and batch_size = eval_rank_microbatch_size / seq_len
        # ensure results match the validation metrics logged to W&B during training.
        eval_cfg = {
            "model": {"type": self.model_format, "path": checkpoint_local},
            "chunk_size": chunk_size,
            "batch_size": batch_size,
            "device": self.device,
            "validation_datasets": validation_datasets,
        }
        config_path = os.path.join(output_dir, f"{self.eval_basename}-eval.yaml")
        builder.create_yaml_file(config_path, eval_cfg)

        # validate.py uses the same memmap evaluation loop as the training framework
        validate_script = os.path.join(
            Project.config.jolmo_path, "src", "scripts", "validate.py",
        )
        builder.run_command(
            f"python3 {validate_script} {config_path} --output {output_json}"
        )

        # Upload result JSON
        _upload_to_gs_with_retry(builder, output_json, remote_path(self.relpath), directory=False)
        # Drop the downloaded model staging once the JSON is uploaded — a large
        # eval sweep otherwise fills the scratch disk (~hundreds of MB x
        # thousands of evals). Only reached on success (script is set -e).
        builder.run_command(f'rm -rf -- "{checkpoint_local}"')


def _perturbed_run_name(
    source_run_name: str,
    gamma: float,
    param_name: Optional[str] = None,
) -> str:
    """Run name for a perturbed checkpoint/eval at ``gamma`` (optional single param)."""
    sigma_str = f"{gamma:.2e}".replace("e-0", "e-").replace("e+0", "e+").replace(".", "_")
    name = f"{source_run_name}_perturbed_{sigma_str}"
    if param_name:
        name = f"{name}_param_{param_name.replace('.', '-')}"
    return name


def _coerce_val_chunk_pairs(
    chunks: Tuple[Any, ...],
) -> Tuple[Tuple[str, Chunk], ...]:
    """Normalize val-chunk sequences to ``(label, Chunk)`` pairs.

    ``ArtifactSet._cartesian_dicts`` treats any tuple param as a list of product
    axes, so a singleton like ``(("DCLM_heldout", chunk),)`` passed through
    ``from_product`` becomes the flat pair ``("DCLM_heldout", chunk)``. Accept
    both shapes.
    """
    if not chunks:
        return ()
    if isinstance(chunks[0], str) and len(chunks) == 2 and isinstance(chunks[1], Chunk):
        return (chunks,)  # type: ignore[return-value]
    return chunks  # type: ignore[return-value]


def _build_validation_datasets_for_model(
    builder: Task,
    model: Any,
    *,
    val_chunks_override: Tuple[Tuple[str, Chunk], ...],
    extra_val_chunks: Tuple[Tuple[str, Chunk], ...],
    extra_val_max_instances: Optional[int],
    dataset_cache_dir: str,
) -> list:
    """Download val data and return the validate.py ``validation_datasets`` list."""
    val_chunks = _coerce_val_chunk_pairs(val_chunks_override or _get_val_chunks_for(model))
    extra_val_chunks = _coerce_val_chunk_pairs(extra_val_chunks)
    _download_chunk_dirs(builder, [c for _, c in val_chunks], dataset_cache_dir)

    validation_datasets = [
        {"name": label, "paths": [_resolve_chunk_path(chunk, dataset_cache_dir)]}
        for label, chunk in val_chunks
    ]

    if extra_val_chunks:
        _download_chunk_files(builder, [c for _, c in extra_val_chunks], dataset_cache_dir)
        for label, chunk in extra_val_chunks:
            ds_entry = {"name": label, "paths": [_resolve_chunk_path(chunk, dataset_cache_dir)]}
            if extra_val_max_instances is not None:
                ds_entry["max_instances"] = extra_val_max_instances
            validation_datasets.append(ds_entry)

    return validation_datasets


@dataclass(frozen=True)
class PerturbedLossEvaluation(Artifact):
    """Average validation loss over many random weight perturbations at fixed gamma.

    For each of ``num_samples`` independent Gaussian noise draws, perturbs the
    pretrained checkpoint in memory, runs validate.py, and writes the mean loss
    to the same eval JSON path used by single-draw perturbed models.
    """

    source_model: JolmoModel = None  # type: ignore[assignment]
    gamma: float = 0.01
    num_samples: int = 100
    seed: int = 64
    device: str = "cuda"
    model_format: str = "olmo"
    extra_val_chunks: Tuple[Tuple[str, Chunk], ...] = ()
    extra_val_max_instances: Optional[int] = None

    @property
    def run_name(self) -> str:
        return _perturbed_run_name(self.source_model.run_name, self.gamma)

    @property
    def relpath(self) -> str:
        return f"ModelEvaluation/{self.run_name}-eval.json"

    @property
    def exists(self) -> bool:
        return False
        return blob_exists(remote_path(self.relpath))

    def get_requirements(self):
        return {
            "gpus": 1,
            "cpus": 4,
            "mem": "64G",
            "partition": "preempt",
            "exclude": "babel-m9-16",
        }

    def construct(self, builder: Task):
        builder.set_env("WANDB_API_KEY", Project.config.wandb_api_key)

        base = self.source_model
        chunk_size = base.sequence_length
        if base.eval_rank_microbatch_size is not None:
            batch_size = max(1, base.eval_rank_microbatch_size // chunk_size)
        else:
            batch_size = max(1, base.rank_microbatch_size // chunk_size)

        data_dir = local_path()
        cache_dir = local_cache_path()

        output_json = os.path.join(data_dir, self.relpath)
        output_dir = os.path.dirname(output_json)
        dataset_cache_dir = os.path.join(cache_dir, "datasets")
        builder.ensure_directory(output_dir)
        builder.ensure_directory(dataset_cache_dir)

        checkpoint_gs = remote_path(base.olmo_model_relpath)
        checkpoint_local = os.path.join(data_dir, base.olmo_model_relpath)
        builder.download_from_gs(checkpoint_gs, checkpoint_local, directory=True)
        builder.download_from_gs(
            remote_path(base.olmo_model_relpath, "config.json"),
            os.path.join(checkpoint_local, "config.json"),
            directory=False,
            skip_existing=False,
            no_fail=True,
        )

        validation_datasets = _build_validation_datasets_for_model(
            builder,
            base,
            val_chunks_override=(),
            extra_val_chunks=self.extra_val_chunks,
            extra_val_max_instances=self.extra_val_max_instances,
            dataset_cache_dir=dataset_cache_dir,
        )

        eval_cfg = {
            "model": {"type": self.model_format, "path": checkpoint_local},
            "chunk_size": chunk_size,
            "batch_size": batch_size,
            "device": self.device,
            "validation_datasets": validation_datasets,
        }
        config_path = os.path.join(output_dir, f"{self.run_name}-eval.yaml")
        builder.create_yaml_file(config_path, eval_cfg)

        avg_script = os.path.join(
            Project.config.code_path, "new_utils", "evaluate_perturbed_average.py",
        )
        validate_script = os.path.join(
            Project.config.jolmo_path, "src", "scripts", "validate.py",
        )
        builder.run_command(
            f"python3 {avg_script} "
            f"--base-checkpoint-dir {checkpoint_local} "
            f"--gamma {self.gamma} "
            f"--num-samples {self.num_samples} "
            f"--seed {self.seed} "
            f"--eval-config {config_path} "
            f"--validate-script {validate_script} "
            f"--output {output_json}"
        )

        _upload_to_gs_with_retry(builder, output_json, remote_path(self.relpath), directory=False)


@dataclass(frozen=True)
class MultiSeedPerturbedEvaluation(Artifact):
    """Average DCLM_heldout loss over saved multi-seed perturbation directions.

    Depends on a completed :class:`MultiSeedPerturbedModel`. Downloads
    ``PerturbedModel/{base}_perturbed_{γ}/seed_*/``, runs validate.py on each
    seed (DCLM_heldout only), and writes a JSON with the mean loss plus a
    per-seed breakdown.
    """

    model: MultiSeedPerturbedModel = None  # type: ignore[assignment]
    device: str = "cuda"
    model_format: str = "olmo"
    # Required: held-out DCLM (and only that — diversity val is skipped).
    extra_val_chunks: Tuple[Tuple[str, Chunk], ...] = ()
    extra_val_max_instances: Optional[int] = None

    @property
    def run_name(self) -> str:
        return f"{self.model.run_name}_multiseed"

    @property
    def relpath(self) -> str:
        return f"ModelEvaluation/{self.run_name}-eval.json"

    @property
    def exists(self) -> bool:
        return blob_exists(remote_path(self.relpath))

    def get_requirements(self):
        return {
            "gpus": 1,
            "cpus": 4,
            "mem": "64G",
            "partition": "general",
            "exclude": "babel-m9-16",
            "time": "12:00:00",
        }

    def construct(self, builder: Task):
        builder.set_env("WANDB_API_KEY", Project.config.wandb_api_key)

        msp = self.model
        base = msp.source_model
        chunk_size = base.sequence_length
        if base.eval_rank_microbatch_size is not None:
            batch_size = max(1, base.eval_rank_microbatch_size // chunk_size)
        else:
            batch_size = max(1, base.rank_microbatch_size // chunk_size)

        data_dir = local_path()
        cache_dir = local_cache_path()
        output_json = os.path.join(data_dir, self.relpath)
        output_dir = os.path.dirname(output_json)
        dataset_cache_dir = os.path.join(cache_dir, "datasets")
        builder.ensure_directory(output_dir)
        builder.ensure_directory(dataset_cache_dir)

        # Download only what the eval reads: each seed's unsharded model.pt and
        # its config.json. The full seed tree also carries optim.pt (~2x the
        # model) and a sharded final/ checkpoint (~4x) — ~85% wasted bytes.
        parent_gs = remote_path(msp.relpath)
        parent_local = os.path.join(data_dir, msp.relpath)
        for s in msp.seeds:
            seed_rel = os.path.join(f"seed_{s:03d}", "final-unsharded")
            builder.ensure_directory(os.path.join(parent_local, seed_rel))
            for fname in ("model.pt", "config.json"):
                builder.download_from_gs(
                    os.path.join(parent_gs, seed_rel, fname),
                    os.path.join(parent_local, seed_rel, fname),
                    directory=False,
                )

        extra = _coerce_val_chunk_pairs(self.extra_val_chunks)
        if not extra:
            raise ValueError(
                "MultiSeedPerturbedEvaluation requires extra_val_chunks "
                "(expected DCLM_heldout)."
            )
        _download_chunk_files(builder, [c for _, c in extra], dataset_cache_dir)
        validation_datasets = []
        for label, chunk in extra:
            entry = {"name": label, "paths": [_resolve_chunk_path(chunk, dataset_cache_dir)]}
            if self.extra_val_max_instances is not None:
                entry["max_instances"] = self.extra_val_max_instances
            validation_datasets.append(entry)

        # Placeholder path; overwritten per seed inside the eval script.
        first_seed = msp.seeds[0]
        placeholder = os.path.join(
            parent_local, f"seed_{first_seed:03d}", "final-unsharded",
        )
        eval_cfg = {
            "model": {"type": self.model_format, "path": placeholder},
            "chunk_size": chunk_size,
            "batch_size": batch_size,
            "device": self.device,
            "validation_datasets": validation_datasets,
        }
        config_path = os.path.join(output_dir, f"{self.run_name}-eval.yaml")
        builder.create_yaml_file(config_path, eval_cfg)

        script = os.path.join(
            Project.config.code_path, "new_utils", "evaluate_multiseed_perturbed.py",
        )
        validate_script = os.path.join(
            Project.config.jolmo_path, "src", "scripts", "validate.py",
        )
        seeds_csv = ",".join(str(s) for s in msp.seeds)
        builder.run_command(
            f"python3 {script} "
            f"--parent-dir {parent_local} "
            f"--seeds {seeds_csv} "
            f"--gamma {msp.gamma} "
            f"--eval-config {config_path} "
            f"--validate-script {validate_script} "
            f"--output {output_json}"
        )

        _upload_to_gs_with_retry(builder, output_json, remote_path(self.relpath), directory=False)
        # The downloaded seed tree is 10+ unsharded checkpoints (~15-20 GB per
        # task); drop it once the averaged JSON is uploaded (script is set -e,
        # so this only runs on success).
        builder.run_command(f'rm -rf -- "{parent_local}"')


# ---------------------------------------------------------------------------
# DivergenceEvaluation — per-token KL/JSD vs. a reference model (OLMo 2 HF)
# ---------------------------------------------------------------------------

# Canonical HF reference checkpoints (shared vocab dim 100352).
DIVERGENCE_REF_OLMO2_32B = "allenai/OLMo-2-0325-32B"
DIVERGENCE_REF_OLMO2_13B = "allenai/OLMo-2-1124-13B"
DIVERGENCE_REF_OLMO2_7B = "allenai/OLMo-2-1124-7B"   # base pretrained (1124 family)
# Public HF base 1B (``allenai/OLMo-2-1124-1B`` is not on the Hub). Legacy C4
# divergence .npz files are tagged ``OLMo-2-1124-1B`` in the plotting pipeline.
DIVERGENCE_REF_OLMO2_1B = "allenai/OLMo-2-0425-1B"
CE_LOSS_REF_TAG_1B = "OLMo-2-1124-1B"


def divergence_max_batch_size(reference_model: str) -> int:
    """Forward-pass batch size for ``divergence_eval.py`` given the reference."""
    lower = reference_model.lower()
    if "32b" in lower:
        return 1
    if "13b" in lower or "7b" in lower or "1b" in lower:
        return 8
    return 1


@dataclass(frozen=True)
class DivergenceEvaluation(Artifact):
    """Per-token distributional divergence of a model vs. a reference model.

    Sibling of :class:`ModelEvaluation`: same checkpoint download + val-chunk
    plumbing, but loads a SECOND (reference / ground-truth) model —
    ``allenai/OLMo-2-0325-32B`` by default (also ``OLMo-2-1124-13B`` / ``7B``),
    always HF — and records, at every next-token prediction position, the
    divergence between the student's and the reference's predicted distributions.
    Wraps ``scripts/divergence_eval.py``.

    PRIMARY metric: forward KL ``KL(reference || student)``. Reverse KL and JSD
    are also recorded. Output is a compressed .npz with flat ``kl_forward`` /
    ``kl_reverse`` / ``jsd`` arrays plus one ``kl_forward`` array per dataset
    label — plot via the colm-moss-latex pipeline
    (``colm-moss-latex/scripts/download_divergence.py`` →
    ``process_divergence.py`` → ``plot_divergence.py``).

    Defaults to the held-out DCLM shard (``DCLM_heldout``), capped at
    ``max_eval_instances`` sequences (8192 by default — same as pretrain evals).
    If ``max_eval_tokens`` is set, it overrides the instance cap so that
    ``≈ max_eval_tokens`` next-token positions are scored
    (``instances = max_eval_tokens // (chunk_size - 1)``).
    The student and reference must share the OLMo 2 tokenizer / vocab dim
    (100352) — the worker hard-stops on a logit-dim mismatch.
    """

    model: Any = None    # JolmoModel | CPTModel | PerturbedModel
    device: str = "cuda"
    model_format: str = "olmo"   # student backend: "olmo" or "hf"
    reference_model: str = DIVERGENCE_REF_OLMO2_32B
    max_eval_instances: int = 8192
    max_eval_tokens: Optional[int] = None
    # [B, L, V] fp32 softmax tensors are large; keep batch=1 for large refs on CPU.
    max_batch_size: int = 1

    @property
    def run_name(self) -> str:
        return self.model.run_name

    @property
    def reference_tag(self) -> str:
        """Filesystem-safe tag for the ground-truth model, e.g.
        ``allenai/OLMo-2-0325-32B`` -> ``OLMo-2-0325-32B``."""
        leaf = self.reference_model.rstrip("/").split("/")[-1]
        return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in leaf)

    @property
    def relpath(self) -> str:
        return (
            f"DivergenceEvaluation/{self.run_name}-vs-{self.reference_tag}"
            f"-divergence.npz"
        )

    @property
    def exists(self) -> bool:
        return blob_exists(remote_path(self.relpath))

    def get_requirements(self):
        # 32B bf16 ~64 GB; 13B bf16 ~26 GB (+ student + activations).
        if "13b" in self.reference_model.lower():
            return {
                "gres": "gpu:A100_80GB:1",
                "cpus": 8,
                "mem": "128G",
                "partition": "general",
                "exclude": "babel-m9-16",
            }
        return {
            "gres": "gpu:1",
            "cpus": 8,
            "mem": "128G",
            "partition": "general",
            "exclude": "babel-m9-16",
        }

    def construct(self, builder: Task):
        builder.set_env("WANDB_API_KEY", Project.config.wandb_api_key)

        base = _get_base_jolmo(self.model)
        chunk_size = base.sequence_length
        if self.max_batch_size > 1:
            # Small references (1B/13B) fit batched on a 40GB GPU; use the
            # configured cap directly (training microbatch may be smaller).
            batch_size = self.max_batch_size
        elif base.eval_rank_microbatch_size is not None:
            batch_size = max(1, base.eval_rank_microbatch_size // chunk_size)
        else:
            batch_size = max(1, base.rank_microbatch_size // chunk_size)
        batch_size = max(1, min(batch_size, self.max_batch_size))

        data_dir = local_path()
        cache_dir = local_cache_path()

        # DivergenceEvaluation only: HF reference is downloaded from HuggingFace.
        # Hard-code the HF cache under scratch (home quota is too small).
        _setup_hf_hub_env(builder)

        output_npz = os.path.join(data_dir, self.relpath)
        output_dir = os.path.dirname(output_npz)
        dataset_cache_dir = os.path.join(cache_dir, "datasets")
        builder.ensure_directory(output_dir)
        builder.ensure_directory(dataset_cache_dir)

        # Download student checkpoint (unsharded OLMo or HF directory).
        checkpoint_gs = remote_path(self.model.olmo_model_relpath)
        checkpoint_local = os.path.join(data_dir, self.model.olmo_model_relpath)
        builder.download_from_gs(checkpoint_gs, checkpoint_local, directory=True)
        builder.download_from_gs(
            remote_path(self.model.olmo_model_relpath, "config.json"),
            os.path.join(checkpoint_local, "config.json"),
            directory=False,
            skip_existing=False,
            no_fail=True,
        )

        # Score held-out DCLM only (never seen in pretraining), same shard/cap
        # as ModelEvaluation extra_val_chunks — not the diversity-v2 C4_val set.
        from launch_jolmo.pretraining_matrix import (
            DCLM_HELDOUT_INSTANCES,
            dclm_heldout_val_chunks,
        )

        val_chunks = dclm_heldout_val_chunks
        if self.max_eval_tokens is not None:
            # Next-token positions per sequence ≈ chunk_size - 1 (matches
            # divergence_eval.py tokens_so_far accounting).
            per_seq = max(1, chunk_size - 1)
            max_instances = max(1, int(self.max_eval_tokens) // per_seq)
        else:
            max_instances = self.max_eval_instances or DCLM_HELDOUT_INSTANCES
        _download_chunk_files(builder, [c for _, c in val_chunks], dataset_cache_dir)

        validation_datasets = [
            {
                "name": label,
                "paths": [_resolve_chunk_path(chunk, dataset_cache_dir)],
                "max_instances": max_instances,
            }
            for label, chunk in val_chunks
        ]

        eval_cfg = {
            "student": {"type": self.model_format, "path": checkpoint_local},
            "reference": {"path": self.reference_model},
            "reference_device": "cuda",
            "chunk_size": chunk_size,
            "batch_size": batch_size,
            "device": self.device,
            "validation_datasets": validation_datasets,
        }
        config_path = os.path.join(
            output_dir, f"{self.run_name}-vs-{self.reference_tag}-divergence.yaml"
        )
        builder.create_yaml_file(config_path, eval_cfg)

        divergence_script = os.path.join(
            Project.config.code_path, "scripts", "divergence_eval.py",
        )
        builder.run_command(
            f"python3 {divergence_script} {config_path} --output {output_npz}"
        )

        _upload_to_gs_with_retry(builder, output_npz, remote_path(self.relpath), directory=False)


# ---------------------------------------------------------------------------
# CeLossEvaluation — per-token NLL on C4_val (legacy 1B-reference pipeline)
# ---------------------------------------------------------------------------

def _setup_hf_hub_env(builder: Task) -> None:
    """HF cache on scratch (home quota too small); pass through auth tokens if set."""
    builder.ensure_directory("/scratch/catheri4/huggingface/hub")
    builder.set_env("HF_HOME", "/scratch/catheri4/huggingface")
    builder.set_env("HUGGINGFACE_HUB_CACHE", "/scratch/catheri4/huggingface/hub")
    import os
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(key)
        if val:
            builder.set_env(key, val)


@dataclass(frozen=True)
class CeLossEvaluation(Artifact):
    """Per-token cross-entropy (NLL) on the diversity-v2 C4_val split.

    Sibling of :class:`DivergenceEvaluation`, but scores a single model only
    (student OLMo checkpoint or HF teacher) and writes ``nll`` arrays via
    ``scripts/ce_loss_eval.py``.

    Matches the legacy untagged 1B divergence runs: ``C4_val``, chunk_size 1024,
    no ``max_instances`` cap.
    """

    model: Any = None          # JolmoModel — student checkpoint on GCS
    hf_model: Optional[str] = None   # HF id/path for teacher-only eval
    output_name: Optional[str] = None  # basename override (e.g. legacy ref tag)
    device: str = "cuda"
    model_format: str = "olmo"
    chunk_size: int = 1024
    max_batch_size: int = 8

    @property
    def run_name(self) -> str:
        if self.output_name:
            return self.output_name
        if self.model is not None:
            return self.model.run_name
        if self.hf_model is not None:
            leaf = self.hf_model.rstrip("/").split("/")[-1]
            return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in leaf)
        raise ValueError("CeLossEvaluation needs model, hf_model, or output_name")

    @property
    def relpath(self) -> str:
        return f"CeLossEvaluation/{self.run_name}-ce-loss.npz"

    @property
    def exists(self) -> bool:
        return blob_exists(remote_path(self.relpath))

    def get_requirements(self):
        return {
            "gres": "gpu:1",
            "cpus": 8,
            "mem": "64G",
            "partition": "general",
            "exclude": "babel-m9-16",
        }

    def construct(self, builder: Task):
        builder.set_env("WANDB_API_KEY", Project.config.wandb_api_key)
        if self.hf_model is not None:
            _setup_hf_hub_env(builder)

        if self.model is None and self.hf_model is None:
            raise ValueError("CeLossEvaluation: set model (student) or hf_model (teacher)")

        batch_size = max(1, self.max_batch_size)
        data_dir = local_path()
        cache_dir = local_cache_path()
        output_npz = os.path.join(data_dir, self.relpath)
        output_dir = os.path.dirname(output_npz)
        dataset_cache_dir = os.path.join(cache_dir, "datasets")
        builder.ensure_directory(output_dir)
        builder.ensure_directory(dataset_cache_dir)

        if self.hf_model is not None:
            model_cfg = {"type": "hf", "path": self.hf_model}
        else:
            checkpoint_gs = remote_path(self.model.olmo_model_relpath)
            checkpoint_local = os.path.join(data_dir, self.model.olmo_model_relpath)
            builder.download_from_gs(checkpoint_gs, checkpoint_local, directory=True)
            builder.download_from_gs(
                remote_path(self.model.olmo_model_relpath, "config.json"),
                os.path.join(checkpoint_local, "config.json"),
                directory=False,
                skip_existing=False,
                no_fail=True,
            )
            model_cfg = {"type": self.model_format, "path": checkpoint_local}

        from launch_jolmo.pretraining_matrix import diversity_val_chunks

        c4_chunks = tuple((n, c) for n, c in diversity_val_chunks if n == "C4_val")
        if not c4_chunks:
            raise RuntimeError("C4_val chunk not found in diversity_val_chunks")
        _download_chunk_dirs(builder, [c for _, c in c4_chunks], dataset_cache_dir)

        validation_datasets = [
            {"name": label, "paths": [_resolve_chunk_path(chunk, dataset_cache_dir)]}
            for label, chunk in c4_chunks
        ]

        eval_cfg = {
            "model": model_cfg,
            "chunk_size": self.chunk_size,
            "batch_size": batch_size,
            "device": self.device,
            "validation_datasets": validation_datasets,
        }
        config_path = os.path.join(output_dir, f"{self.run_name}-ce-loss.yaml")
        builder.create_yaml_file(config_path, eval_cfg)

        ce_script = os.path.join(Project.config.code_path, "scripts", "ce_loss_eval.py")
        builder.run_command(
            f"python3 {ce_script} {config_path} --output {output_npz}"
        )
        _upload_to_gs_with_retry(builder, output_npz, remote_path(self.relpath), directory=False)


# ---------------------------------------------------------------------------
# C4DivergenceEvaluation — per-token KL/JSD on C4_val vs 1B (legacy pipeline)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class C4DivergenceEvaluation(Artifact):
    """Per-token KL/JSD on C4_val vs the public 1B reference.

    Sibling of :class:`DivergenceEvaluation` configured for the legacy 1B C4
    pipeline: ``C4_val``, chunk_size 1024, no instance cap, output
    ``DivergenceEvaluation/<run>-divergence.npz`` (untagged, like pretrain).

    Supports :class:`JolmoModel`, :class:`CPTModel`, and :class:`PerturbedModel`.
    """

    model: Any = None
    device: str = "cuda"
    model_format: str = "olmo"
    reference_model: str = DIVERGENCE_REF_OLMO2_1B
    chunk_size: int = 1024
    max_batch_size: int = 8

    @property
    def run_name(self) -> str:
        return self.model.run_name

    @property
    def relpath(self) -> str:
        return f"DivergenceEvaluation/{self.run_name}-divergence.npz"

    @property
    def exists(self) -> bool:
        return blob_exists(remote_path(self.relpath))

    def get_requirements(self):
        return {
            "gres": "gpu:1",
            "cpus": 8,
            "mem": "64G",
            "partition": "general",
            "exclude": "babel-m9-16",
        }

    def construct(self, builder: Task):
        builder.set_env("WANDB_API_KEY", Project.config.wandb_api_key)
        _setup_hf_hub_env(builder)

        batch_size = max(1, self.max_batch_size)
        data_dir = local_path()
        cache_dir = local_cache_path()
        output_npz = os.path.join(data_dir, self.relpath)
        output_dir = os.path.dirname(output_npz)
        dataset_cache_dir = os.path.join(cache_dir, "datasets")
        builder.ensure_directory(output_dir)
        builder.ensure_directory(dataset_cache_dir)

        checkpoint_gs = remote_path(self.model.olmo_model_relpath)
        checkpoint_local = os.path.join(data_dir, self.model.olmo_model_relpath)
        builder.download_from_gs(checkpoint_gs, checkpoint_local, directory=True)
        builder.download_from_gs(
            remote_path(self.model.olmo_model_relpath, "config.json"),
            os.path.join(checkpoint_local, "config.json"),
            directory=False,
            skip_existing=False,
            no_fail=True,
        )

        from launch_jolmo.pretraining_matrix import diversity_val_chunks

        c4_chunks = tuple((n, c) for n, c in diversity_val_chunks if n == "C4_val")
        if not c4_chunks:
            raise RuntimeError("C4_val chunk not found in diversity_val_chunks")
        _download_chunk_dirs(builder, [c for _, c in c4_chunks], dataset_cache_dir)

        validation_datasets = [
            {"name": label, "paths": [_resolve_chunk_path(chunk, dataset_cache_dir)]}
            for label, chunk in c4_chunks
        ]

        eval_cfg = {
            "student": {"type": self.model_format, "path": checkpoint_local},
            "reference": {
                "path": self.reference_model,
                "use_cache_if_complete": True,
            },
            "reference_device": "cuda",
            "chunk_size": self.chunk_size,
            "batch_size": batch_size,
            "device": self.device,
            "validation_datasets": validation_datasets,
        }
        config_path = os.path.join(output_dir, f"{self.run_name}-c4-divergence.yaml")
        builder.create_yaml_file(config_path, eval_cfg)

        divergence_script = os.path.join(
            Project.config.code_path, "scripts", "divergence_eval.py",
        )
        builder.run_command(
            f"python3 {divergence_script} {config_path} --output {output_npz}"
        )
        _upload_to_gs_with_retry(builder, output_npz, remote_path(self.relpath), directory=False)


# ---------------------------------------------------------------------------
# SharpnessEvaluation — Hessian max-eig / trace (Lanczos + Hutch++)
# Ported from catastrophic-forgetting/launch/artifacts.py
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SharpnessEvaluation(Artifact):
    """Compute sharpness metrics (max eigenvalue, trace) for a JOLMo checkpoint.

    Runs ``scripts/evaluate_sharpness.py`` on a memmap validation shard
    (default: diversity-v2 ``C4_val``). Metrics match CF:
    ``max_eigenvalue`` (Lanczos) and ``sum_eigenvalue`` / Hutch++ trace.

    For ``metrics`` containing ``max_eigenvalue``, also loads unsharded
    ``optim.pt`` and reports the optimizer-transformed top eigenvalue
    ``λ_max(T∘H)`` (AdamW ``1/√v̂``, Muon Newton–Schulz).

    ``checkpoint`` selects which training save to evaluate:
    ``"final"`` (default; prefers ``final-unsharded/``) or ``"stepN"``.
    Intermediate DCP steps are unsharded on the fly (model + optim).
    """

    model: Any = None  # JolmoModel | CPTModel | PerturbedModel | InterpolatedModel
    eval_dataset: str = "pretrain"  # "pretrain" → C4_val; else a CPT dataset name
    device: str = "cuda"
    chunk_size: int = 1024
    batch_size: int = 4
    metrics: str = "max_eigenvalue,sum_eigenvalue"
    num_samples: int = 20
    max_chunks: int = 50
    dtype: str = "uint32"
    # Stochastic Lanczos quadrature (metrics="spectral_density") only.
    lanczos_steps: int = 100
    num_probes: int = 10
    probe_seed: int = 1234
    # Training checkpoint under JolmoModel/<run>/ — "final" or "stepN".
    checkpoint: str = "final"

    @property
    def _metrics_tag(self) -> str:
        return self.metrics.replace(",", "_")

    @property
    def _is_spectral_density(self) -> bool:
        return "spectral_density" in self.metrics

    @property
    def _wants_optim(self) -> bool:
        return "max_eigenvalue" in self.metrics

    @property
    def run_name(self) -> str:
        return self.model.run_name

    @property
    def relpath(self) -> str:
        # m/n_v only enter the name for SLQ runs, so existing artifact paths for
        # max_eigenvalue/sum_eigenvalue are unchanged when checkpoint="final".
        suffix = (
            f"-m{self.lanczos_steps}-nv{self.num_probes}"
            if self._is_spectral_density
            else ""
        )
        ckpt_tag = "" if self.checkpoint == "final" else f"-{self.checkpoint}"
        return (
            f"SharpnessEvaluation/{self.run_name}-{self.eval_dataset}"
            f"{ckpt_tag}-sharpness-{self._metrics_tag}{suffix}.json"
        )

    @property
    def exists(self) -> bool:
        return blob_exists(remote_path(self.relpath))

    def get_requirements(self):
        return {
            "gres": "gpu:1",
            "cpus": 8,
            "mem": "256G",
            "partition": "preempt",
            "exclude": "babel-m9-16",
            "time": "2-00:00:00",
        }

    def _prepare_checkpoint(self, builder: Task, data_dir: str) -> str:
        """Download / unshard the requested checkpoint; return local model dir.

        The directory contains ``model.pt``, ``config.json``, and (when
        available) ``optim.pt``.
        """
        run_rel = self.model.relpath
        run_gs = remote_path(run_rel)
        jolmo_path = Project.config.jolmo_path
        unshard_script = os.path.join(jolmo_path, "src", "scripts", "unshard.py")

        if self.checkpoint == "final":
            # Prefer the already-uploaded final-unsharded bundle.
            unsharded_rel = self.model.olmo_model_relpath
            if blob_exists(remote_path(unsharded_rel, "model.pt")):
                checkpoint_local = os.path.join(data_dir, unsharded_rel)
                builder.download_from_gs(
                    remote_path(unsharded_rel), checkpoint_local, directory=True
                )
                builder.download_from_gs(
                    remote_path(unsharded_rel, "config.json"),
                    os.path.join(checkpoint_local, "config.json"),
                    directory=False,
                    skip_existing=False,
                    no_fail=True,
                )
                return checkpoint_local
            dcp_gs = f"{run_gs}/final"
        else:
            if not self.checkpoint.startswith("step"):
                raise ValueError(
                    f"checkpoint must be 'final' or 'stepN', got {self.checkpoint!r}"
                )
            dcp_gs = f"{run_gs}/{self.checkpoint}"

        checkpoint_local = os.path.join(
            data_dir, run_rel, f"unsharded-{self.checkpoint}"
        )
        work_dir = os.path.join(data_dir, run_rel, f"unshard-work-{self.checkpoint}")
        builder.ensure_directory(checkpoint_local)
        builder.ensure_directory(work_dir)
        builder.run_command(
            f'python3 "{unshard_script}" "{dcp_gs}" "{checkpoint_local}" '
            f'--overwrite --pre-download --work-dir "{work_dir}"'
        )
        # unshard.py does not copy config.json.
        builder.run_command(
            f'gsutil -q cp "{dcp_gs}/config.json" '
            f'"{os.path.join(checkpoint_local, "config.json")}" || true'
        )
        return checkpoint_local

    def construct(self, builder: Task):
        builder.set_env("WANDB_API_KEY", Project.config.wandb_api_key)

        data_dir = local_path()
        cache_dir = local_cache_path()
        output_json = os.path.join(data_dir, self.relpath)
        output_dir = os.path.dirname(output_json)
        dataset_cache_dir = os.path.join(cache_dir, "datasets")
        builder.ensure_directory(output_dir)
        builder.ensure_directory(dataset_cache_dir)

        checkpoint_local = self._prepare_checkpoint(builder, data_dir)

        # Resolve eval memmap path.
        if self.eval_dataset == "pretrain":
            from launch_jolmo.pretraining_matrix import diversity_val_chunks

            chunks = tuple((n, c) for n, c in diversity_val_chunks if n == "C4_val")
            if not chunks:
                raise RuntimeError("C4_val not found in diversity_val_chunks")
            _download_chunk_dirs(builder, [c for _, c in chunks], dataset_cache_dir)
            data_path = _resolve_chunk_path(chunks[0][1], dataset_cache_dir)
        else:
            if self.eval_dataset not in CPT_DATASETS:
                raise ValueError(
                    f"Unknown eval_dataset={self.eval_dataset!r}; "
                    f"use 'pretrain' or one of {sorted(CPT_DATASETS)}"
                )
            cpt_val = CPT_DATASETS[self.eval_dataset]["val"]
            first_chunk = next(iter(cpt_val.values()))
            _download_chunk_dirs(builder, [first_chunk], dataset_cache_dir)
            data_path = _resolve_chunk_path(first_chunk, dataset_cache_dir)

        eval_script = os.path.join(
            Project.config.code_path, "scripts", "evaluate_sharpness.py"
        )
        env_prefix = f"PYTHONPATH={Project.config.code_path}:${{PYTHONPATH:-}} "
        cmd = (
            f"{env_prefix}python3 {eval_script} "
            f"--model_path {checkpoint_local} "
            f"--device {self.device} "
            f"--data_path {data_path} "
            f"--chunk_size {self.chunk_size} "
            f"--batch_size {self.batch_size} "
            f"--output_path {output_json} "
            f"--metrics {self.metrics} "
            f"--num_samples {self.num_samples} "
            f"--dtype={self.dtype} "
            f"--max_chunks {self.max_chunks} "
            f"--checkpoint_name {self.checkpoint}"
        )
        if self._wants_optim:
            optim_pt = os.path.join(checkpoint_local, "optim.pt")
            cmd += f" --optim_path {optim_pt}"
        if self._is_spectral_density:
            cmd += (
                f" --lanczos_steps {self.lanczos_steps}"
                f" --num_probes {self.num_probes}"
                f" --probe_seed {self.probe_seed}"
            )
        builder.run_command(cmd)
        _upload_to_gs_with_retry(
            builder, output_json, remote_path(self.relpath), directory=False
        )


def list_training_checkpoints(run_relpath: str, *, skip_step0: bool = True) -> List[str]:
    """List ``stepN`` / ``final`` dirs under a JolmoModel GCS run (skip step0).

    Returns names sorted by step number, with ``final`` last when present.
    """
    import re
    import subprocess

    base = remote_path(run_relpath).rstrip("/") + "/"
    proc = subprocess.run(
        ["gsutil", "ls", base], capture_output=True, text=True
    )
    if proc.returncode != 0:
        return []
    steps: List[Tuple[int, str]] = []
    has_final = False
    for line in proc.stdout.splitlines():
        name = line.strip().rstrip("/").rsplit("/", 1)[-1]
        if name == "final":
            has_final = True
            continue
        m = re.fullmatch(r"step(\d+)", name)
        if not m:
            continue
        step = int(m.group(1))
        if skip_step0 and step == 0:
            continue
        steps.append((step, name))
    names = [n for _, n in sorted(steps)]
    if has_final:
        names.append("final")
    return names


@dataclass(frozen=True)
class ForgettingSharpnessEvaluation(Artifact):
    """Forgetting estimate ``0.5 * Δᵀ H Δ`` at θ_PT along Δ = θ_FT − θ_PT.

    Hessian of the pretrain loss is evaluated at the pretrained checkpoint;
    the CPT / FT checkpoint supplies the direction Δ.
    """

    pretrained_model: Any = None  # JolmoModel
    cpt_model: Any = None  # CPTModel
    device: str = "cuda"
    chunk_size: int = 1024
    batch_size: int = 4
    max_chunks: int = 50
    gradient_type: str = "full"
    dtype: str = "uint32"

    @property
    def run_name(self) -> str:
        return self.cpt_model.run_name

    @property
    def relpath(self) -> str:
        return f"SharpnessEvaluation/{self.run_name}-forgetting_sharpness.json"

    @property
    def exists(self) -> bool:
        return blob_exists(remote_path(self.relpath))

    def get_requirements(self):
        return {
            "gres": "gpu:1",
            "cpus": 8,
            "mem": "256G",
            "partition": "general",
            "exclude": "babel-m9-16",
            "time": "2-00:00:00",
        }

    def construct(self, builder: Task):
        builder.set_env("WANDB_API_KEY", Project.config.wandb_api_key)

        data_dir = local_path()
        cache_dir = local_cache_path()
        output_json = os.path.join(data_dir, self.relpath)
        output_dir = os.path.dirname(output_json)
        dataset_cache_dir = os.path.join(cache_dir, "datasets")
        builder.ensure_directory(output_dir)
        builder.ensure_directory(dataset_cache_dir)

        # PT checkpoint (Hessian site).
        pt_gs = remote_path(self.pretrained_model.olmo_model_relpath)
        pt_local = os.path.join(data_dir, self.pretrained_model.olmo_model_relpath)
        builder.download_from_gs(pt_gs, pt_local, directory=True)
        builder.download_from_gs(
            remote_path(self.pretrained_model.olmo_model_relpath, "config.json"),
            os.path.join(pt_local, "config.json"),
            directory=False,
            skip_existing=False,
            no_fail=True,
        )

        # FT / CPT checkpoint (direction).
        ft_gs = remote_path(self.cpt_model.olmo_model_relpath)
        ft_local = os.path.join(data_dir, self.cpt_model.olmo_model_relpath)
        builder.download_from_gs(ft_gs, ft_local, directory=True)
        builder.download_from_gs(
            remote_path(self.cpt_model.olmo_model_relpath, "config.json"),
            os.path.join(ft_local, "config.json"),
            directory=False,
            skip_existing=False,
            no_fail=True,
        )

        from launch_jolmo.pretraining_matrix import diversity_val_chunks

        chunks = tuple((n, c) for n, c in diversity_val_chunks if n == "C4_val")
        if not chunks:
            raise RuntimeError("C4_val not found in diversity_val_chunks")
        _download_chunk_dirs(builder, [c for _, c in chunks], dataset_cache_dir)
        data_path = _resolve_chunk_path(chunks[0][1], dataset_cache_dir)

        eval_script = os.path.join(
            Project.config.code_path, "scripts", "evaluate_sharpness.py"
        )
        env_prefix = f"PYTHONPATH={Project.config.code_path}:${{PYTHONPATH:-}} "
        cmd = (
            f"{env_prefix}python3 {eval_script} "
            f"--model_path {pt_local} "
            f"--device {self.device} "
            f"--data_path {data_path} "
            f"--chunk_size {self.chunk_size} "
            f"--batch_size {self.batch_size} "
            f"--output_path {output_json} "
            f"--metrics directional "
            f"--gradient_type {self.gradient_type} "
            f"--dtype={self.dtype} "
            f"--max_chunks {self.max_chunks} "
            f"--compute_direction_from {ft_local}"
        )
        builder.run_command(cmd)
        _upload_to_gs_with_retry(
            builder, output_json, remote_path(self.relpath), directory=False
        )



# ---------------------------------------------------------------------------
# LogitPerturbEvaluation — perturb per-token logits, measure CE vs σ
# LLM analogue of AM multi_min_output_perturbation (relative ℓ₂ on z_i).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogitPerturbEvaluation(Artifact):
    """Relative ℓ₂ Gaussian noise on per-token logits; CE/NLL vs σ.

    For each next-token logit vector ``ℓ ∈ ℝ^V``::

        ℓ ← ℓ + σ · ‖ℓ‖₂ · ξ,   ‖ξ‖₂ = 1

    One job sweeps a geometric σ schedule (plus σ=0 baseline). Uses C4_val
    with a ``max_instances`` cap so a single forward pass stays cheap.
    """

    model: Any = None
    device: str = "cuda"
    model_format: str = "olmo"
    chunk_size: int = 1024
    max_batch_size: int = 4
    max_instances: int = 512
    num_directions: int = 5
    seed: int = 0
    sigma_min: float = 1e-3
    sigma_max: float = 10.0
    sigma_ratio: float = 1.4142135623730951

    @property
    def run_name(self) -> str:
        return self.model.run_name

    @property
    def relpath(self) -> str:
        return f"LogitPerturbEvaluation/{self.run_name}-logit_perturb.json"

    @property
    def exists(self) -> bool:
        return blob_exists(remote_path(self.relpath))

    def get_requirements(self):
        return {
            "gres": "gpu:1",
            "cpus": 8,
            "mem": "64G",
            "partition": "general",
            "exclude": "babel-m9-16",
            "time": "4:00:00",
        }

    def construct(self, builder: Task):
        builder.set_env("WANDB_API_KEY", Project.config.wandb_api_key)

        batch_size = max(1, self.max_batch_size)
        data_dir = local_path()
        cache_dir = local_cache_path()
        output_json = os.path.join(data_dir, self.relpath)
        output_dir = os.path.dirname(output_json)
        dataset_cache_dir = os.path.join(cache_dir, "datasets")
        builder.ensure_directory(output_dir)
        builder.ensure_directory(dataset_cache_dir)

        checkpoint_gs = remote_path(self.model.olmo_model_relpath)
        checkpoint_local = os.path.join(data_dir, self.model.olmo_model_relpath)
        builder.download_from_gs(checkpoint_gs, checkpoint_local, directory=True)
        builder.download_from_gs(
            remote_path(self.model.olmo_model_relpath, "config.json"),
            os.path.join(checkpoint_local, "config.json"),
            directory=False,
            skip_existing=False,
            no_fail=True,
        )

        from launch_jolmo.pretraining_matrix import diversity_val_chunks

        c4_chunks = tuple((n, c) for n, c in diversity_val_chunks if n == "C4_val")
        if not c4_chunks:
            raise RuntimeError("C4_val chunk not found in diversity_val_chunks")
        _download_chunk_dirs(builder, [c for _, c in c4_chunks], dataset_cache_dir)

        validation_datasets = [
            {
                "name": label,
                "paths": [_resolve_chunk_path(chunk, dataset_cache_dir)],
                "max_instances": self.max_instances,
            }
            for label, chunk in c4_chunks
        ]

        eval_cfg = {
            "model": {"type": self.model_format, "path": checkpoint_local},
            "chunk_size": self.chunk_size,
            "batch_size": batch_size,
            "device": self.device,
            "max_instances": self.max_instances,
            "num_directions": self.num_directions,
            "seed": self.seed,
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "sigma_ratio": self.sigma_ratio,
            "validation_datasets": validation_datasets,
        }
        config_path = os.path.join(output_dir, f"{self.run_name}-logit_perturb.yaml")
        builder.create_yaml_file(config_path, eval_cfg)

        script = os.path.join(
            Project.config.code_path, "scripts", "logit_perturb_ce_eval.py"
        )
        env_prefix = f"PYTHONPATH={Project.config.code_path}:${{PYTHONPATH:-}} "
        builder.run_command(
            f"{env_prefix}python3 {script} {config_path} --output {output_json}"
        )
        _upload_to_gs_with_retry(
            builder, output_json, remote_path(self.relpath), directory=False
        )


# ---------------------------------------------------------------------------
# LogitPerturbKlEvaluation — perturb student logits, KL vs 1B reference
# Same noise as LogitPerturbEvaluation; metric is KL(Q||P) with Q=1B teacher.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogitPerturbKlEvaluation(Artifact):
    """Relative ℓ₂ noise on student logits; KL(Q‖P) vs σ, Q = OLMo-2 1B.

    For each next-token logit vector ``ℓ ∈ ℝ^V``::

        ℓ ← ℓ + σ · ‖ℓ‖₂ · ξ,   ‖ξ‖₂ = 1

    then score ``kl_forward = KL(Q ‖ P)`` with Q the 1B teacher and P the
    (perturbed) student — same primary metric as :class:`C4DivergenceEvaluation`.
    """

    model: Any = None
    device: str = "cuda"
    model_format: str = "olmo"
    reference_model: str = DIVERGENCE_REF_OLMO2_1B
    reference_device: str = "cuda"
    chunk_size: int = 1024
    max_batch_size: int = 2
    max_instances: int = 512
    num_directions: int = 5
    seed: int = 0
    sigma_min: float = 1e-3
    sigma_max: float = 10.0
    sigma_ratio: float = 1.4142135623730951

    @property
    def run_name(self) -> str:
        return self.model.run_name

    @property
    def relpath(self) -> str:
        return f"LogitPerturbKlEvaluation/{self.run_name}-logit_perturb_kl.json"

    @property
    def exists(self) -> bool:
        return blob_exists(remote_path(self.relpath))

    def get_requirements(self):
        return {
            "gres": "gpu:1",
            "cpus": 8,
            "mem": "64G",
            "partition": "general",
            "exclude": "babel-m9-16",
            "time": "8:00:00",
        }

    def construct(self, builder: Task):
        builder.set_env("WANDB_API_KEY", Project.config.wandb_api_key)
        _setup_hf_hub_env(builder)

        batch_size = max(1, self.max_batch_size)
        data_dir = local_path()
        cache_dir = local_cache_path()
        output_json = os.path.join(data_dir, self.relpath)
        output_dir = os.path.dirname(output_json)
        dataset_cache_dir = os.path.join(cache_dir, "datasets")
        builder.ensure_directory(output_dir)
        builder.ensure_directory(dataset_cache_dir)

        checkpoint_gs = remote_path(self.model.olmo_model_relpath)
        checkpoint_local = os.path.join(data_dir, self.model.olmo_model_relpath)
        builder.download_from_gs(checkpoint_gs, checkpoint_local, directory=True)
        builder.download_from_gs(
            remote_path(self.model.olmo_model_relpath, "config.json"),
            os.path.join(checkpoint_local, "config.json"),
            directory=False,
            skip_existing=False,
            no_fail=True,
        )

        from launch_jolmo.pretraining_matrix import diversity_val_chunks

        c4_chunks = tuple((n, c) for n, c in diversity_val_chunks if n == "C4_val")
        if not c4_chunks:
            raise RuntimeError("C4_val chunk not found in diversity_val_chunks")
        _download_chunk_dirs(builder, [c for _, c in c4_chunks], dataset_cache_dir)

        validation_datasets = [
            {
                "name": label,
                "paths": [_resolve_chunk_path(chunk, dataset_cache_dir)],
                "max_instances": self.max_instances,
            }
            for label, chunk in c4_chunks
        ]

        eval_cfg = {
            "model": {"type": self.model_format, "path": checkpoint_local},
            "reference": {
                "path": self.reference_model,
                "use_cache_if_complete": True,
            },
            "reference_device": self.reference_device,
            "chunk_size": self.chunk_size,
            "batch_size": batch_size,
            "device": self.device,
            "max_instances": self.max_instances,
            "num_directions": self.num_directions,
            "seed": self.seed,
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "sigma_ratio": self.sigma_ratio,
            "validation_datasets": validation_datasets,
        }
        config_path = os.path.join(output_dir, f"{self.run_name}-logit_perturb_kl.yaml")
        builder.create_yaml_file(config_path, eval_cfg)

        script = os.path.join(
            Project.config.code_path, "scripts", "logit_perturb_kl_eval.py"
        )
        env_prefix = f"PYTHONPATH={Project.config.code_path}:${{PYTHONPATH:-}} "
        builder.run_command(
            f"{env_prefix}python3 {script} {config_path} --output {output_json}"
        )
        _upload_to_gs_with_retry(
            builder, output_json, remote_path(self.relpath), directory=False
        )


# ---------------------------------------------------------------------------
# LogitCosineEvaluation — per-token logit cos-sim between adamw & muon (same chin)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogitCosineEvaluation(Artifact):
    """Cosine similarity of per-token output logits: adamw vs muon.

    For each next-token position::

        cos = ⟨ℓ_adamw, ℓ_muon⟩ / (‖ℓ_adamw‖₂ ‖ℓ_muon‖₂)

    One job per chinchilla (paired tuned-LR checkpoints).
    """

    adamw_model: Any = None
    muon_model: Any = None
    chinchilla: int = 1
    device: str = "cuda"
    model_format: str = "olmo"
    chunk_size: int = 1024
    max_batch_size: int = 4
    max_instances: int = 512

    @property
    def run_name(self) -> str:
        return f"chinchilla-{self.chinchilla}-adamw-vs-muon"

    @property
    def relpath(self) -> str:
        return f"LogitCosineEvaluation/{self.run_name}-logit_cosine.json"

    @property
    def exists(self) -> bool:
        return blob_exists(remote_path(self.relpath))

    def get_requirements(self):
        return {
            "gres": "gpu:1",
            "cpus": 8,
            "mem": "64G",
            "partition": "general",
            "exclude": "babel-m9-16",
            "time": "4:00:00",
        }

    def _download_checkpoint(self, builder: Task, model, data_dir: str) -> str:
        checkpoint_gs = remote_path(model.olmo_model_relpath)
        checkpoint_local = os.path.join(data_dir, model.olmo_model_relpath)
        builder.download_from_gs(checkpoint_gs, checkpoint_local, directory=True)
        builder.download_from_gs(
            remote_path(model.olmo_model_relpath, "config.json"),
            os.path.join(checkpoint_local, "config.json"),
            directory=False,
            skip_existing=False,
            no_fail=True,
        )
        return checkpoint_local

    def construct(self, builder: Task):
        builder.set_env("WANDB_API_KEY", Project.config.wandb_api_key)

        batch_size = max(1, self.max_batch_size)
        data_dir = local_path()
        cache_dir = local_cache_path()
        output_json = os.path.join(data_dir, self.relpath)
        output_dir = os.path.dirname(output_json)
        dataset_cache_dir = os.path.join(cache_dir, "datasets")
        builder.ensure_directory(output_dir)
        builder.ensure_directory(dataset_cache_dir)

        adamw_local = self._download_checkpoint(builder, self.adamw_model, data_dir)
        muon_local = self._download_checkpoint(builder, self.muon_model, data_dir)

        from launch_jolmo.pretraining_matrix import diversity_val_chunks

        c4_chunks = tuple((n, c) for n, c in diversity_val_chunks if n == "C4_val")
        if not c4_chunks:
            raise RuntimeError("C4_val chunk not found in diversity_val_chunks")
        _download_chunk_dirs(builder, [c for _, c in c4_chunks], dataset_cache_dir)

        validation_datasets = [
            {
                "name": label,
                "paths": [_resolve_chunk_path(chunk, dataset_cache_dir)],
                "max_instances": self.max_instances,
            }
            for label, chunk in c4_chunks
        ]

        eval_cfg = {
            "chinchilla": self.chinchilla,
            "adamw_run": self.adamw_model.run_name,
            "muon_run": self.muon_model.run_name,
            "adamw_model": {"type": self.model_format, "path": adamw_local},
            "muon_model": {"type": self.model_format, "path": muon_local},
            "chunk_size": self.chunk_size,
            "batch_size": batch_size,
            "device": self.device,
            "max_instances": self.max_instances,
            "validation_datasets": validation_datasets,
        }
        config_path = os.path.join(output_dir, f"{self.run_name}-logit_cosine.yaml")
        builder.create_yaml_file(config_path, eval_cfg)

        script = os.path.join(
            Project.config.code_path, "scripts", "logit_cosine_adamw_muon.py"
        )
        env_prefix = f"PYTHONPATH={Project.config.code_path}:${{PYTHONPATH:-}} "
        builder.run_command(
            f"{env_prefix}python3 {script} {config_path} --output {output_json}"
        )
        _upload_to_gs_with_retry(
            builder, output_json, remote_path(self.relpath), directory=False
        )


# ---------------------------------------------------------------------------
# LogitAngleBinEvaluation — metrics + examples binned by adamw↔muon logit angle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogitAngleBinEvaluation(Artifact):
    """Per-token metrics sliced by adamw↔muon logit angle (10° bins).

    For each next-token position computes ``θ = arccos(cos(ℓ_a, ℓ_m))`` and
    records margin / NLL / KL-vs-1B / KL-between / token frequency, plus 20
    decoded examples per angle bin.
    """

    adamw_model: Any = None
    muon_model: Any = None
    chinchilla: int = 1
    device: str = "cuda"
    model_format: str = "olmo"
    reference_model: str = DIVERGENCE_REF_OLMO2_1B
    reference_device: str = "cuda"
    chunk_size: int = 1024
    max_batch_size: int = 2
    max_instances: int = 512
    examples_per_bin: int = 20
    context_tokens: int = 64
    tokenizer: str = "allenai/OLMo-2-0425-1B-Instruct"

    @property
    def run_name(self) -> str:
        return f"chinchilla-{self.chinchilla}-adamw-vs-muon"

    @property
    def relpath(self) -> str:
        # primary artifact pointer (npz); examples/summary share the prefix
        return f"LogitAngleBinEvaluation/{self.run_name}-angle_bin_metrics.npz"

    @property
    def examples_relpath(self) -> str:
        return f"LogitAngleBinEvaluation/{self.run_name}-angle_bin_examples.json"

    @property
    def summary_relpath(self) -> str:
        return f"LogitAngleBinEvaluation/{self.run_name}-angle_bin_summary.json"

    @property
    def exists(self) -> bool:
        return blob_exists(remote_path(self.relpath))

    def get_requirements(self):
        return {
            "gres": "gpu:1",
            "cpus": 8,
            "mem": "64G",
            "partition": "general",
            "exclude": "babel-m9-16",
            "time": "8:00:00",
        }

    def _download_checkpoint(self, builder: Task, model, data_dir: str) -> str:
        checkpoint_gs = remote_path(model.olmo_model_relpath)
        checkpoint_local = os.path.join(data_dir, model.olmo_model_relpath)
        builder.download_from_gs(checkpoint_gs, checkpoint_local, directory=True)
        builder.download_from_gs(
            remote_path(model.olmo_model_relpath, "config.json"),
            os.path.join(checkpoint_local, "config.json"),
            directory=False,
            skip_existing=False,
            no_fail=True,
        )
        return checkpoint_local

    def construct(self, builder: Task):
        builder.set_env("WANDB_API_KEY", Project.config.wandb_api_key)
        _setup_hf_hub_env(builder)

        batch_size = max(1, self.max_batch_size)
        data_dir = local_path()
        cache_dir = local_cache_path()
        output_npz = os.path.join(data_dir, self.relpath)
        output_examples = os.path.join(data_dir, self.examples_relpath)
        output_summary = os.path.join(data_dir, self.summary_relpath)
        output_dir = os.path.dirname(output_npz)
        dataset_cache_dir = os.path.join(cache_dir, "datasets")
        builder.ensure_directory(output_dir)
        builder.ensure_directory(dataset_cache_dir)

        adamw_local = self._download_checkpoint(builder, self.adamw_model, data_dir)
        muon_local = self._download_checkpoint(builder, self.muon_model, data_dir)

        from launch_jolmo.pretraining_matrix import diversity_val_chunks

        c4_chunks = tuple((n, c) for n, c in diversity_val_chunks if n == "C4_val")
        if not c4_chunks:
            raise RuntimeError("C4_val chunk not found in diversity_val_chunks")
        _download_chunk_dirs(builder, [c for _, c in c4_chunks], dataset_cache_dir)

        validation_datasets = [
            {
                "name": label,
                "paths": [_resolve_chunk_path(chunk, dataset_cache_dir)],
                "max_instances": self.max_instances,
            }
            for label, chunk in c4_chunks
        ]

        eval_cfg = {
            "chinchilla": self.chinchilla,
            "adamw_run": self.adamw_model.run_name,
            "muon_run": self.muon_model.run_name,
            "adamw_model": {"type": self.model_format, "path": adamw_local},
            "muon_model": {"type": self.model_format, "path": muon_local},
            "reference": {
                "path": self.reference_model,
                "use_cache_if_complete": True,
            },
            "reference_device": self.reference_device,
            "chunk_size": self.chunk_size,
            "batch_size": batch_size,
            "device": self.device,
            "max_instances": self.max_instances,
            "examples_per_bin": self.examples_per_bin,
            "context_tokens": self.context_tokens,
            "tokenizer": self.tokenizer,
            "validation_datasets": validation_datasets,
        }
        config_path = os.path.join(output_dir, f"{self.run_name}-angle_bin.yaml")
        builder.create_yaml_file(config_path, eval_cfg)

        script = os.path.join(
            Project.config.code_path, "scripts", "logit_angle_bin_metrics.py"
        )
        env_prefix = f"PYTHONPATH={Project.config.code_path}:${{PYTHONPATH:-}} "
        builder.run_command(
            f"{env_prefix}python3 {script} {config_path} "
            f"--output-npz {output_npz} "
            f"--output-examples {output_examples} "
            f"--output-summary {output_summary}"
        )
        _upload_to_gs_with_retry(
            builder, output_npz, remote_path(self.relpath), directory=False
        )
        _upload_to_gs_with_retry(
            builder, output_examples, remote_path(self.examples_relpath), directory=False
        )
        _upload_to_gs_with_retry(
            builder, output_summary, remote_path(self.summary_relpath), directory=False
        )
        ex_txt = output_examples.replace(".json", ".txt")
        _upload_to_gs_with_retry(
            builder,
            ex_txt,
            remote_path(self.examples_relpath.replace(".json", ".txt")),
            directory=False,
        )


# ---------------------------------------------------------------------------
# LogitAngleBinPerturbEvaluation — ΔNLL vs σ, stratified by logit-angle bin
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogitAngleBinPerturbEvaluation(Artifact):
    """Split C4_val tokens by adamw↔muon logit angle, then run Gaussian logit
    perturbation **separately** on each angle-bin data group.

    Same relative-ℓ₂ noise / mean-NLL-vs-σ protocol as
    :class:`LogitPerturbEvaluation`, but the average is restricted to tokens
    whose unperturbed logit angle falls in that 10° bin.
    """

    adamw_model: Any = None
    muon_model: Any = None
    chinchilla: int = 1
    device: str = "cuda"
    model_format: str = "olmo"
    chunk_size: int = 1024
    max_batch_size: int = 2
    max_instances: int = 512
    num_directions: int = 5
    seed: int = 0
    sigma_min: float = 1e-3
    sigma_max: float = 10.0
    sigma_ratio: float = 1.4142135623730951

    @property
    def run_name(self) -> str:
        return f"chinchilla-{self.chinchilla}-adamw-vs-muon"

    @property
    def relpath(self) -> str:
        return f"LogitAngleBinPerturbEvaluation/{self.run_name}-angle_bin_perturb.json"

    @property
    def exists(self) -> bool:
        return blob_exists(remote_path(self.relpath))

    def get_requirements(self):
        return {
            "gres": "gpu:1",
            "cpus": 8,
            "mem": "64G",
            "partition": "general",
            "exclude": "babel-m9-16",
            "time": "8:00:00",
        }

    def _download_checkpoint(self, builder: Task, model, data_dir: str) -> str:
        checkpoint_gs = remote_path(model.olmo_model_relpath)
        checkpoint_local = os.path.join(data_dir, model.olmo_model_relpath)
        builder.download_from_gs(checkpoint_gs, checkpoint_local, directory=True)
        builder.download_from_gs(
            remote_path(model.olmo_model_relpath, "config.json"),
            os.path.join(checkpoint_local, "config.json"),
            directory=False,
            skip_existing=False,
            no_fail=True,
        )
        return checkpoint_local

    def construct(self, builder: Task):
        builder.set_env("WANDB_API_KEY", Project.config.wandb_api_key)

        batch_size = max(1, self.max_batch_size)
        data_dir = local_path()
        cache_dir = local_cache_path()
        output_json = os.path.join(data_dir, self.relpath)
        output_dir = os.path.dirname(output_json)
        dataset_cache_dir = os.path.join(cache_dir, "datasets")
        builder.ensure_directory(output_dir)
        builder.ensure_directory(dataset_cache_dir)

        adamw_local = self._download_checkpoint(builder, self.adamw_model, data_dir)
        muon_local = self._download_checkpoint(builder, self.muon_model, data_dir)

        from launch_jolmo.pretraining_matrix import diversity_val_chunks

        c4_chunks = tuple((n, c) for n, c in diversity_val_chunks if n == "C4_val")
        if not c4_chunks:
            raise RuntimeError("C4_val chunk not found in diversity_val_chunks")
        _download_chunk_dirs(builder, [c for _, c in c4_chunks], dataset_cache_dir)

        validation_datasets = [
            {
                "name": label,
                "paths": [_resolve_chunk_path(chunk, dataset_cache_dir)],
                "max_instances": self.max_instances,
            }
            for label, chunk in c4_chunks
        ]

        eval_cfg = {
            "chinchilla": self.chinchilla,
            "adamw_run": self.adamw_model.run_name,
            "muon_run": self.muon_model.run_name,
            "adamw_model": {"type": self.model_format, "path": adamw_local},
            "muon_model": {"type": self.model_format, "path": muon_local},
            "chunk_size": self.chunk_size,
            "batch_size": batch_size,
            "device": self.device,
            "max_instances": self.max_instances,
            "num_directions": self.num_directions,
            "seed": self.seed,
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "sigma_ratio": self.sigma_ratio,
            "validation_datasets": validation_datasets,
        }
        config_path = os.path.join(output_dir, f"{self.run_name}-angle_bin_perturb.yaml")
        builder.create_yaml_file(config_path, eval_cfg)

        script = os.path.join(
            Project.config.code_path, "scripts", "logit_angle_bin_perturb.py"
        )
        env_prefix = f"PYTHONPATH={Project.config.code_path}:${{PYTHONPATH:-}} "
        builder.run_command(
            f"{env_prefix}python3 {script} {config_path} --output {output_json}"
        )
        _upload_to_gs_with_retry(
            builder, output_json, remote_path(self.relpath), directory=False
        )


# ---------------------------------------------------------------------------
# WeightAngleBinPerturbEvaluation — Gaussian weight noise; ΔNLL per angle-bin
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeightAngleBinPerturbEvaluation(Artifact):
    """Split tokens by adamw↔muon logit angle, then Gaussian-**weight**-perturb
    each model and measure mean NLL on each angle-bin data group separately.

    Weight noise matches ``perturb_weights.perturb_state_dict``
    (``std = γ · ‖W‖_F / √numel``).
    """

    adamw_model: Any = None
    muon_model: Any = None
    chinchilla: int = 1
    device: str = "cuda"
    chunk_size: int = 1024
    max_batch_size: int = 2
    max_instances: int = 512
    num_directions: int = 5
    seed: int = 0
    gamma_min: float = 1e-4
    gamma_max: float = 5e-2
    gamma_ratio: float = 1.4142135623730951

    @property
    def run_name(self) -> str:
        return f"chinchilla-{self.chinchilla}-adamw-vs-muon"

    @property
    def relpath(self) -> str:
        return f"WeightAngleBinPerturbEvaluation/{self.run_name}-weight_angle_bin_perturb.json"

    @property
    def exists(self) -> bool:
        return blob_exists(remote_path(self.relpath))

    def get_requirements(self):
        return {
            "gres": "gpu:1",
            "cpus": 8,
            "mem": "64G",
            "partition": "general",
            "exclude": "babel-m9-16",
            "time": "12:00:00",
        }

    def _download_checkpoint(self, builder: Task, model, data_dir: str) -> str:
        checkpoint_gs = remote_path(model.olmo_model_relpath)
        checkpoint_local = os.path.join(data_dir, model.olmo_model_relpath)
        builder.download_from_gs(checkpoint_gs, checkpoint_local, directory=True)
        builder.download_from_gs(
            remote_path(model.olmo_model_relpath, "config.json"),
            os.path.join(checkpoint_local, "config.json"),
            directory=False,
            skip_existing=False,
            no_fail=True,
        )
        return checkpoint_local

    def construct(self, builder: Task):
        builder.set_env("WANDB_API_KEY", Project.config.wandb_api_key)

        batch_size = max(1, self.max_batch_size)
        data_dir = local_path()
        cache_dir = local_cache_path()
        output_json = os.path.join(data_dir, self.relpath)
        output_dir = os.path.dirname(output_json)
        dataset_cache_dir = os.path.join(cache_dir, "datasets")
        builder.ensure_directory(output_dir)
        builder.ensure_directory(dataset_cache_dir)

        adamw_local = self._download_checkpoint(builder, self.adamw_model, data_dir)
        muon_local = self._download_checkpoint(builder, self.muon_model, data_dir)

        from launch_jolmo.pretraining_matrix import diversity_val_chunks

        c4_chunks = tuple((n, c) for n, c in diversity_val_chunks if n == "C4_val")
        if not c4_chunks:
            raise RuntimeError("C4_val chunk not found in diversity_val_chunks")
        _download_chunk_dirs(builder, [c for _, c in c4_chunks], dataset_cache_dir)

        validation_datasets = [
            {
                "name": label,
                "paths": [_resolve_chunk_path(chunk, dataset_cache_dir)],
                "max_instances": self.max_instances,
            }
            for label, chunk in c4_chunks
        ]

        eval_cfg = {
            "chinchilla": self.chinchilla,
            "adamw_run": self.adamw_model.run_name,
            "muon_run": self.muon_model.run_name,
            "adamw_model": {"type": "olmo", "path": adamw_local},
            "muon_model": {"type": "olmo", "path": muon_local},
            "chunk_size": self.chunk_size,
            "batch_size": batch_size,
            "device": self.device,
            "max_instances": self.max_instances,
            "num_directions": self.num_directions,
            "seed": self.seed,
            "gamma_min": self.gamma_min,
            "gamma_max": self.gamma_max,
            "gamma_ratio": self.gamma_ratio,
            "validation_datasets": validation_datasets,
        }
        config_path = os.path.join(
            output_dir, f"{self.run_name}-weight_angle_bin_perturb.yaml"
        )
        builder.create_yaml_file(config_path, eval_cfg)

        script = os.path.join(
            Project.config.code_path, "scripts", "weight_angle_bin_perturb.py"
        )
        env_prefix = f"PYTHONPATH={Project.config.code_path}:${{PYTHONPATH:-}} "
        builder.run_command(
            f"{env_prefix}python3 {script} {config_path} --output {output_json}"
        )
        _upload_to_gs_with_retry(
            builder, output_json, remote_path(self.relpath), directory=False
        )

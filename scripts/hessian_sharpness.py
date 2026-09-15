"""Hessian-vector products and sharpness metrics for causal LMs.

Ported from ``catastrophic-forgetting/scripts/evaluator.py`` (MaxEigenvalue /
SumEigenvalues / Directional sharpness via Lanczos + Hutch++), adapted for
JOLMo / HF models that expose logits via ``model(input_ids=...)`` rather than
the old ``olmo.model.OLMo`` API.

What is estimated:
  * max eigenvalue of the (empirical) CE Hessian via Lanczos (1 eig)
  * trace via Hutch++ (``scipy.sparse.linalg._expm_multiply.traceest``)
  * directional quadratic form ``v^T H v`` (e.g. forgetting: ``Δ = θ_FT − θ_PT``)
  * spectral density via stochastic Lanczos quadrature (Ritz nodes + weights)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import eigh_tridiagonal
from scipy.sparse.linalg import LinearOperator, eigs, eigsh
from torch.nn.utils import parameters_to_vector
from tqdm import tqdm

try:
    from scipy.sparse.linalg._expm_multiply import traceest
except ImportError:  # pragma: no cover
    traceest = None  # type: ignore[assignment]


def _sdpa_math_ctx():
    """Force MATH SDPA — flash/mem-efficient kernels lack HVP double-backward."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel([SDPBackend.MATH])
    except Exception:
        try:
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_mem_efficient=False, enable_math=True
            )
        except Exception:
            return nullcontext()


def logits_from_model(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Return ``[B, T, V]`` logits for an OLMo-core Transformer or HF CausalLM."""
    out = model(input_ids=input_ids)
    if hasattr(out, "logits"):
        return out.logits
    return out


def rademacher_probe(dim: int, generator: torch.Generator) -> torch.Tensor:
    """Unit-norm Rademacher vector on the host, drawn from ``generator``."""
    bits = torch.randint(0, 2, (dim,), generator=generator, dtype=torch.int8)
    v = bits.to(torch.float32).mul_(2.0).sub_(1.0)
    return v.div_(v.norm())


def lanczos_tridiagonal(
    matvec,
    dim: int,
    steps: int,
    q0: torch.Tensor,
    *,
    device,
    reorth_passes: int = 2,
    beta_tol: float = 1e-10,
    progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Lanczos with full reorthogonalization; returns the tridiagonal ``(α, β)``.

    ``scipy``'s ``eigsh`` hides the tridiagonal matrix, but stochastic Lanczos
    quadrature needs it, so the recurrence is run explicitly here.

    The basis is held on the host (``steps × dim`` float32) and the
    reorthogonalization projections run in host BLAS: at 60M params a single
    basis vector is 240 MB, so a 100-step basis will not fit on the GPU
    alongside the model. Only three vectors live on ``device`` at a time.

    Returns ``(alphas, betas)`` with lengths ``k`` and ``k-1``, where ``k <=
    steps`` (early exit on an invariant subspace).
    """
    basis = torch.empty((steps, dim), dtype=torch.float32, device="cpu")
    alphas: List[float] = []
    betas: List[float] = []

    q = q0.detach().to(dtype=torch.float32, device="cpu")
    q = q / q.norm()
    q_dev = q.to(device)
    q_prev_dev: Optional[torch.Tensor] = None
    beta = 0.0

    iterator = tqdm(range(steps), desc="Lanczos", leave=False) if progress else range(steps)
    for j in iterator:
        basis[j] = q

        w = matvec(q_dev)
        alpha = float(torch.dot(q_dev, w))
        alphas.append(alpha)

        w = w - alpha * q_dev
        if q_prev_dev is not None:
            w = w - beta * q_prev_dev

        w_host = w.to("cpu", dtype=torch.float32)
        del w

        used = basis[: j + 1]
        for _ in range(reorth_passes):
            w_host -= (used @ w_host) @ used

        beta = float(w_host.norm())
        if j + 1 >= steps or beta <= beta_tol:
            break

        betas.append(beta)
        q = w_host / beta
        q_prev_dev = q_dev
        q_dev = q.to(device)

    return (
        np.asarray(alphas, dtype=np.float64),
        np.asarray(betas, dtype=np.float64),
    )


def slq_nodes_weights(
    alphas: Sequence[float], betas: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Ritz values and quadrature weights from a Lanczos tridiagonal matrix.

    The weights are the squared first components of ``T``'s eigenvectors and sum
    to 1; they carry the ``1/p`` normalization of the spectral density, while
    the Ritz values are already in curvature units. Returned sorted descending.
    """
    d = np.asarray(alphas, dtype=np.float64)
    e = np.asarray(betas, dtype=np.float64)
    if d.size == 0:
        raise ValueError("empty Lanczos tridiagonal")
    if d.size == 1:
        return d.copy(), np.ones(1, dtype=np.float64)

    theta, vecs = eigh_tridiagonal(d, e)
    tau = np.square(vecs[0, :]).astype(np.float64)
    order = np.argsort(theta)[::-1]
    return theta[order], tau[order]


class SharpnessEvaluator(ABC):
    def __init__(
        self,
        dataset,
        batch_size: int,
        max_length: Optional[int] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length

    @abstractmethod
    def evaluate(self, model, tokenizer) -> dict:
        pass

    def _default_max_length(self, model) -> int:
        if self.max_length is not None:
            return self.max_length
        cfg = getattr(model, "config", None)
        if cfg is not None:
            for key in ("max_position_embeddings", "max_sequence_length", "max_seq_len"):
                if hasattr(cfg, key) and getattr(cfg, key):
                    return int(getattr(cfg, key))
        return 1024

    def _prepare_batches(self, eval_data, tokenizer, max_length, *, with_labels: bool = False):
        pad_token = (
            tokenizer.pad_token_id
            if getattr(tokenizer, "pad_token_id", None) is not None
            else tokenizer.eos_token_id
        )
        batches = []
        for i in range(0, len(eval_data["input_ids"]), self.batch_size):
            batch_seqs = eval_data["input_ids"][i : i + self.batch_size]
            batch_max_length = min(max_length, max(len(x) for x in batch_seqs))

            input_ids = torch.stack(
                [
                    torch.tensor(
                        [pad_token] * (batch_max_length - len(x)) + x[:batch_max_length]
                    )
                    for x in batch_seqs
                ]
            )
            attention_mask = torch.stack(
                [
                    torch.tensor(
                        [0] * (batch_max_length - len(x)) + [1] * len(x[:batch_max_length])
                    )
                    for x in batch_seqs
                ]
            )
            batch = {"input_ids": input_ids, "attention_mask": attention_mask}
            if with_labels:
                labels = torch.stack(
                    [
                        torch.tensor(
                            [-100] * (batch_max_length - len(x))
                            + eval_data["labels"][i + j][:batch_max_length]
                        )
                        for j, x in enumerate(batch_seqs)
                    ]
                )
                batch["labels"] = labels
            batches.append(batch)
        return batches


def newton_schulz_zeropower(
    grad: torch.Tensor,
    ns_coefficients: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    ns_steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Newton–Schulz orthogonalization (Muon / torch.optim.Muon).

    Same quintic iteration as ``torch.optim._muon._zeropower_via_newtonschulz``.
    Returns float32 on the input device (NS internals run in bfloat16).
    """
    if grad.ndim != 2:
        raise ValueError(f"Newton–Schulz expects a 2D matrix, got shape {tuple(grad.shape)}")
    a, b, c = ns_coefficients
    ortho = grad.to(dtype=torch.bfloat16)
    if grad.size(0) > grad.size(1):
        ortho = ortho.T
    ortho = ortho / ortho.norm().clamp(min=eps)
    for _ in range(ns_steps):
        gram = ortho @ ortho.T
        gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        ortho = torch.addmm(ortho, gram_update, ortho, beta=a)
    if grad.size(0) > grad.size(1):
        ortho = ortho.T
    return ortho.to(dtype=torch.float32)


class OptimizerGradientTransform:
    """Apply the optimizer's gradient transform ``T`` in flat parameter space.

    * AdamW params: ``T(g) = g / (√v̂ + ε)`` with bias-corrected ``exp_avg_sq``.
    * Muon 2D params (``_muon=True``): Newton–Schulz orthogonalization.
    """

    def __init__(
        self,
        named_params: Sequence[Tuple[str, torch.nn.Parameter]],
        optim_flat: Dict[str, torch.Tensor],
        device: torch.device,
    ):
        self._slices: List[dict] = []
        offset = 0
        for name, param in named_params:
            numel = param.numel()
            shape = tuple(param.shape)
            is_muon_raw = optim_flat.get(f"param_groups.{name}._muon", False)
            if torch.is_tensor(is_muon_raw):
                is_muon = bool(is_muon_raw.item())
            else:
                is_muon = bool(is_muon_raw)
            entry: dict = {
                "name": name,
                "offset": offset,
                "numel": numel,
                "shape": shape,
                "is_muon": is_muon,
            }
            if is_muon:
                if len(shape) != 2:
                    raise ValueError(
                        f"Muon transform for {name!r} expects 2D, got shape {shape}"
                    )
                ns_coef = optim_flat.get(f"param_groups.{name}.ns_coefficients")
                if ns_coef is None:
                    ns_coefficients = (3.4445, -4.7750, 2.0315)
                elif torch.is_tensor(ns_coef):
                    ns_coefficients = tuple(float(x) for x in ns_coef.tolist())
                else:
                    ns_coefficients = tuple(float(x) for x in ns_coef)
                ns_steps_raw = optim_flat.get(f"param_groups.{name}.ns_steps", 5)
                ns_steps = (
                    int(ns_steps_raw.item())
                    if torch.is_tensor(ns_steps_raw)
                    else int(ns_steps_raw)
                )
                eps_raw = optim_flat.get(f"param_groups.{name}.eps", 1e-7)
                eps = float(eps_raw.item()) if torch.is_tensor(eps_raw) else float(eps_raw)
                entry.update(
                    ns_coefficients=ns_coefficients,
                    ns_steps=ns_steps,
                    eps=eps,
                )
            else:
                exp_avg_sq = optim_flat.get(f"state.{name}.exp_avg_sq")
                if exp_avg_sq is None:
                    raise KeyError(
                        f"Missing state.{name}.exp_avg_sq in optim.pt "
                        f"(needed for AdamW preconditioner)"
                    )
                step_t = optim_flat.get(f"state.{name}.step")
                step = float(step_t.item()) if step_t is not None else 1.0
                betas = optim_flat.get(f"param_groups.{name}.betas", (0.9, 0.95))
                if torch.is_tensor(betas):
                    beta2 = float(betas.flatten()[1].item())
                else:
                    beta2 = float(betas[1])
                eps = float(optim_flat.get(f"param_groups.{name}.eps", 1e-8))
                # Bias-corrected second moment → per-coordinate AdamW denom.
                bias_correction2 = 1.0 - (beta2 ** max(step, 1.0))
                v_hat = exp_avg_sq.float() / max(bias_correction2, 1e-12)
                denom = v_hat.sqrt().add_(eps).reshape(-1).to(device=device)
                entry["inv_denom"] = (1.0 / denom).contiguous()
            self._slices.append(entry)
            offset += numel
        self.dim = offset

    def __call__(self, flat: torch.Tensor) -> torch.Tensor:
        if flat.numel() != self.dim:
            raise ValueError(
                f"Transform dim mismatch: vector has {flat.numel()} elems, "
                f"expected {self.dim}"
            )
        out = torch.empty_like(flat)
        for entry in self._slices:
            sl = slice(entry["offset"], entry["offset"] + entry["numel"])
            chunk = flat[sl]
            if entry["is_muon"]:
                mat = chunk.view(entry["shape"])
                transformed = newton_schulz_zeropower(
                    mat,
                    ns_coefficients=entry["ns_coefficients"],
                    ns_steps=entry["ns_steps"],
                    eps=entry["eps"],
                )
                out[sl] = transformed.reshape(-1)
            else:
                out[sl] = chunk * entry["inv_denom"]
        return out


class MaxEigenvalueSharpnessEvaluator(SharpnessEvaluator):
    """Lanczos top eigenvalue of ``H``, optionally also of ``T ∘ H``.

    When ``transform`` is set (optimizer gradient map ``T``), also estimates
    ``λ_max(T ∘ H)`` via Arnoldi (``eigs``), since ``T ∘ H`` need not be
    symmetric (and Muon's Newton–Schulz map is nonlinear).
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        max_length: Optional[int] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(dataset, batch_size, max_length)
        self.transform = transform
        # Krylov depth for lambda_max. 40 is far past convergence for a single
        # extremal eigenvalue and bounds the host-side basis at steps x dim
        # float32 (96 GB at 0.6B, against ~1.8 TB of RAM).
        self.lanczos_steps_maxeig = 40
        self.lanczos_seed = 1234

    def evaluate(self, model, tokenizer) -> dict:
        if getattr(self.dataset, "dataset_type", "cpt") not in ("cpt", "sft"):
            raise ValueError("MaxEigenvalueSharpnessEvaluator requires CPT or SFT dataset")

        max_length = self._default_max_length(model)
        eval_data = self.dataset.as_evaluation_data()
        batches = self._prepare_batches(eval_data, tokenizer, max_length)

        def loss_fn(logits, target):
            logits = logits[:, :-1, :].contiguous()
            target = target[:, 1:].contiguous()
            return F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
                ignore_index=-100,
            )

        p = parameters_to_vector(model.parameters()).numel()

        def matrix_vector(v):
            return self._compute_hvp(model, loss_fn, batches, v)

        eigenvalues, _ = self._lanczos_symmetric(matrix_vector, p, 1)
        results = {"max_eigenvalue_sharpness": eigenvalues[0].item()}

        if self.transform is not None:
            def preconditioned_mv(v):
                return self.transform(self._compute_hvp(model, loss_fn, batches, v))

            prec_eig = self._arnoldi_largest_real(preconditioned_mv, p)
            results["max_eigenvalue_preconditioned_sharpness"] = float(prec_eig)

        return results

    def _compute_hvp(self, model, loss_fn, batches, vector):
        p = parameters_to_vector(model.parameters()).numel()
        device = next(model.parameters()).device
        hvp = torch.zeros(p, dtype=torch.float, device=device)
        vector = vector.to(device)

        with _sdpa_math_ctx():
            for batch in tqdm(batches, desc="HVP (max eig)"):
                n = len(batch["input_ids"]) * len(batches)
                input_ids = batch["input_ids"].to(device)
                loss = loss_fn(logits_from_model(model, input_ids), input_ids) / n
                grads = torch.autograd.grad(
                    loss, inputs=tuple(model.parameters()), create_graph=True
                )
                dot = parameters_to_vector(grads).mul(vector).sum()
                grads2 = torch.autograd.grad(
                    dot, tuple(model.parameters()), retain_graph=True
                )
                grads2 = [g.contiguous() for g in grads2]
                hvp += parameters_to_vector(grads2)

        return hvp

    def _lanczos_symmetric(self, matrix_vector, dim: int, neigs: int):
        """Top ``neigs`` eigenvalues of a symmetric operator.

        ARPACK (``eigsh``) is not usable at this scale. Its Fortran core indexes
        workspace with 32-bit integers, and scipy's default ``maxiter = 10 * n``
        overflows once ``n`` passes ~2.1e8: at 0.3B params it raised

            ArpackError -4: The maximum number of Arnoldi update iterations
            allowed must be greater than zero

        and at 0.6B the workspace arithmetic wrapped and segfaulted the process
        outright (exit 139) after the HVPs had already completed.

        So run Lanczos here instead, reusing ``lanczos_tridiagonal`` -- the same
        fully-reorthogonalized, host-backed recurrence the SLQ path uses -- and
        diagonalize the small tridiagonal. The largest Ritz value converges to
        lambda_max from below in a handful of steps, and no vector of length
        ``dim`` ever reaches numpy.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        steps = min(self.lanczos_steps_maxeig, dim)

        g = torch.Generator(device="cpu").manual_seed(self.lanczos_seed)
        q0 = torch.randn(dim, generator=g, dtype=torch.float32)

        alphas, betas = lanczos_tridiagonal(
            matrix_vector, dim, steps, q0, device=device, progress=True
        )
        ritz = eigh_tridiagonal(alphas, betas, eigvals_only=True)
        top = np.sort(np.asarray(ritz, dtype=np.float64))[::-1][:neigs]
        # Ritz vectors would cost steps x dim and no caller uses them.
        return torch.from_numpy(np.ascontiguousarray(top).copy()).float(), None

    def _arnoldi_largest_real(self, matrix_vector, dim: int) -> float:
        """Largest-real-part eigenvalue of a (possibly non-symmetric) operator.

        The ``eigs`` counterpart of the ARPACK problem described in
        ``_lanczos_symmetric``, so the Arnoldi recurrence is run explicitly with
        the same host-backed basis layout: only three length-``dim`` vectors are
        on the device at a time, the basis lives in host memory, and the
        eigenvalue problem is solved on the small upper-Hessenberg matrix.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        steps = min(self.lanczos_steps_maxeig, dim)

        g = torch.Generator(device="cpu").manual_seed(self.lanczos_seed)
        q = torch.randn(dim, generator=g, dtype=torch.float32)
        q /= q.norm()

        basis = torch.empty((steps, dim), dtype=torch.float32, device="cpu")
        H = np.zeros((steps + 1, steps), dtype=np.float64)

        k = steps
        for j in tqdm(range(steps), desc="Arnoldi (precond max eig)", leave=False):
            basis[j] = q
            w = matrix_vector(q.to(device))
            w_host = w.detach().to("cpu", dtype=torch.float32)
            del w

            # Modified Gram-Schmidt against the whole basis, twice: one pass
            # loses orthogonality badly at this conditioning.
            used = basis[: j + 1]
            for _ in range(2):
                proj = used @ w_host
                w_host -= proj @ used
                H[: j + 1, j] += proj.numpy().astype(np.float64)

            h_next = float(w_host.norm())
            H[j + 1, j] = h_next
            if h_next <= 1e-10:          # invariant subspace: stop early
                k = j + 1
                break
            q = w_host / h_next

        vals = np.linalg.eigvals(H[:k, :k])
        return float(np.real(vals).max())


class SumEigenvaluesSharpnessEvaluator(SharpnessEvaluator):
    def __init__(
        self,
        dataset,
        batch_size: int,
        max_length: Optional[int] = None,
        num_samples: int = 20,
    ):
        super().__init__(dataset, batch_size, max_length)
        self.num_samples = num_samples

    def evaluate(self, model, tokenizer) -> dict:
        if getattr(self.dataset, "dataset_type", "cpt") not in ("cpt", "sft"):
            raise ValueError("SumEigenvaluesSharpnessEvaluator requires CPT or SFT dataset")

        max_length = self._default_max_length(model)
        eval_data = self.dataset.as_evaluation_data()
        batches = self._prepare_batches(eval_data, tokenizer, max_length)

        def loss_fn(logits, target):
            logits = logits[:, :-1, :].contiguous()
            target = target[:, 1:].contiguous()
            return F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )

        p = parameters_to_vector(model.parameters()).numel()

        def matrix_vector(v):
            return self._compute_hvp(model, loss_fn, batches, v)

        trace = self._hutchplusplus(matrix_vector, p, self.num_samples)
        return {"sum_eigenvalue_sharpness": float(trace)}

    def _compute_hvp(self, model, loss_fn, batches, vector):
        p = parameters_to_vector(model.parameters()).numel()
        device = next(model.parameters()).device
        hvp = torch.zeros(p, dtype=torch.float, device=device)
        if len(vector.shape) > 1:
            vector = vector[:, 0]
        vector = vector.to(device)

        total_tokens = 0
        with _sdpa_math_ctx():
            for batch in tqdm(batches, desc="HVP (trace)"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                total_tokens += attention_mask.sum().item()
                loss = loss_fn(logits_from_model(model, input_ids), input_ids)
                grads = torch.autograd.grad(
                    loss, inputs=tuple(model.parameters()), create_graph=True
                )
                dot = parameters_to_vector(grads).mul(vector).sum()
                grads2 = torch.autograd.grad(
                    dot, tuple(model.parameters()), retain_graph=True
                )
                grads2 = [g.contiguous() for g in grads2]
                hvp += parameters_to_vector(grads2)

        return hvp / max(total_tokens, 1)

    def _hutchplusplus(self, matrix_vector, dim, m3):
        if traceest is None:
            raise ImportError(
                "scipy.sparse.linalg._expm_multiply.traceest is required for Hutch++ "
                f"(scipy version={getattr(__import__('scipy'), '__version__', '?')})"
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def mv(vec: np.ndarray):
            gpu_vec = torch.tensor(vec, dtype=torch.float, device=device)
            return matrix_vector(gpu_vec).detach().cpu().numpy()

        operator = LinearOperator((dim, dim), matvec=mv, dtype=np.float64)
        return traceest(operator, m3)


class DirectionalSharpnessEvaluator(SharpnessEvaluator):
    def __init__(
        self,
        dataset,
        batch_size: int,
        max_length: Optional[int] = None,
        direction: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__(dataset, batch_size, max_length)
        self.direction = direction

    def evaluate(self, model, tokenizer) -> dict:
        if getattr(self.dataset, "dataset_type", "cpt") not in ("cpt", "sft"):
            raise ValueError("DirectionalSharpnessEvaluator requires CPT or SFT dataset")
        if self.direction is None:
            raise ValueError("DirectionalSharpnessEvaluator requires a direction dict")

        max_length = self._default_max_length(model)
        eval_data = self.dataset.as_evaluation_data()
        batches = self._prepare_batches(eval_data, tokenizer, max_length, with_labels=True)

        def loss_fn(model_, batch):
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            logits = logits_from_model(model_, input_ids)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            return F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )

        direction_p = parameters_to_vector(self.direction.values()).detach()
        device = next(model.parameters()).device

        def matrix_vector(v):
            return self._compute_hvp(model, loss_fn, batches, v)

        hv = matrix_vector(direction_p.float().to(device))
        directional_sharpness = torch.dot(direction_p.float().to(device), hv).item()
        return {"directional_sharpness": directional_sharpness}

    def _compute_hvp(self, model, loss_fn, batches, vector):
        p = parameters_to_vector(model.parameters()).numel()
        device = next(model.parameters()).device
        hvp = torch.zeros(p, dtype=torch.float, device=device)
        vector = vector.to(device)

        total_tokens = 0
        with _sdpa_math_ctx():
            for batch in tqdm(batches, desc="HVP (directional)"):
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = loss_fn(model, batch)
                grads = torch.autograd.grad(
                    loss, inputs=tuple(model.parameters()), create_graph=True
                )
                dot = parameters_to_vector(grads).mul(vector).sum()
                grads2 = torch.autograd.grad(
                    dot, tuple(model.parameters()), retain_graph=True
                )
                grads2 = [g.contiguous() for g in grads2]
                hvp += parameters_to_vector(grads2)
                total_tokens += batch["labels"].ne(-100).float().sum().item()

        return hvp / max(total_tokens, 1)


class SpectralDensityEvaluator(SharpnessEvaluator):
    """Hessian spectral density via stochastic Lanczos quadrature.

    Emits the raw Ritz nodes and quadrature weights for every probe rather than
    a binned histogram: the run costs GPU-hours, the payload is a few tens of
    KB, and every derived statistic (counting function, quantiles, moments,
    effective rank) can then be recomputed offline by
    ``new_utils.hessian_spectrum`` without re-running.

    The loss is normalized mean-per-token, matching
    ``SumEigenvaluesSharpnessEvaluator``, so ``slq_trace`` is directly
    comparable to the Hutch++ ``sum_eigenvalue_sharpness``.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        max_length: Optional[int] = None,
        lanczos_steps: int = 100,
        num_probes: int = 10,
        probe_seed: int = 1234,
    ):
        super().__init__(dataset, batch_size, max_length)
        self.lanczos_steps = lanczos_steps
        self.num_probes = num_probes
        self.probe_seed = probe_seed

    def evaluate(self, model, tokenizer) -> dict:
        if getattr(self.dataset, "dataset_type", "cpt") not in ("cpt", "sft"):
            raise ValueError("SpectralDensityEvaluator requires CPT or SFT dataset")

        max_length = self._default_max_length(model)
        eval_data = self.dataset.as_evaluation_data()
        batches = self._prepare_batches(eval_data, tokenizer, max_length)

        def loss_fn(logits, target):
            logits = logits[:, :-1, :].contiguous()
            target = target[:, 1:].contiguous()
            return F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )

        params = tuple(model.parameters())
        dim = sum(p.numel() for p in params)
        device = next(model.parameters()).device
        total_tokens = int(
            sum(b["attention_mask"].sum().item() for b in batches)
        )

        def matvec(v):
            return self._compute_hvp(model, loss_fn, batches, v, total_tokens)

        generator = torch.Generator().manual_seed(self.probe_seed)
        probes = []
        for index in range(self.num_probes):
            q0 = rademacher_probe(dim, generator)
            alphas, betas = lanczos_tridiagonal(
                matvec, dim, self.lanczos_steps, q0, device=device, progress=True
            )
            theta, tau = slq_nodes_weights(alphas, betas)
            probes.append(
                {"ritz_values": theta.tolist(), "weights": tau.tolist()}
            )
            print(
                f"probe {index + 1}/{self.num_probes}: {theta.size} nodes, "
                f"max Ritz {theta[0]:.6g}"
            )

        nodes = np.concatenate([np.asarray(p["ritz_values"]) for p in probes])
        weights = np.concatenate([np.asarray(p["weights"]) for p in probes])
        weights = weights / float(self.num_probes)

        return {
            "spectral_density": {
                "num_params": int(dim),
                "lanczos_steps": int(self.lanczos_steps),
                "num_probes": int(self.num_probes),
                "probe_seed": int(self.probe_seed),
                "num_tokens": total_tokens,
                "loss_normalization": "mean_per_token",
                "probes": probes,
            },
            "slq_trace": float(dim * (weights * nodes).sum()),
            "slq_max_ritz": float(nodes.max()),
        }

    def _compute_hvp(self, model, loss_fn, batches, vector, total_tokens):
        params = tuple(model.parameters())
        device = next(model.parameters()).device
        hvp = torch.zeros(
            sum(p.numel() for p in params), dtype=torch.float, device=device
        )
        vector = vector.to(device)

        with _sdpa_math_ctx():
            for batch in tqdm(batches, desc="HVP (SLQ)", leave=False):
                input_ids = batch["input_ids"].to(device)
                loss = loss_fn(logits_from_model(model, input_ids), input_ids)
                grads = torch.autograd.grad(
                    loss, inputs=params, create_graph=True
                )
                dot = parameters_to_vector(grads).mul(vector).sum()
                grads2 = torch.autograd.grad(dot, params, retain_graph=True)
                grads2 = [g.contiguous() for g in grads2]
                hvp += parameters_to_vector(grads2)

        return hvp / max(total_tokens, 1)


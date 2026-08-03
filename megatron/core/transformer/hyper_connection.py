# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import nvtx_decorator

if TYPE_CHECKING:
    from megatron.core.tensor_parallel.random import CheckpointWithoutOutputManager


try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _sinkhorn_fused_kernel(
        M_ptr,
        OUT_ptr,
        n_mats,
        num_iterations,
        eps,
        BLOCK: tl.constexpr,
        N: tl.constexpr,
    ):
        """Run all Sinkhorn iterations for BLOCK [N, N] matrices in one launch.

        Each program keeps its tiles in registers across iterations, so the chain
        costs one round-trip to global memory instead of one per normalization.
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_mats
        i = tl.arange(0, N)[None, :, None]
        j = tl.arange(0, N)[None, None, :]
        idx = offs[:, None, None] * (N * N) + i * N + j
        # other=1.0 keeps masked lanes' sums non-zero so the tail block stays finite.
        m = tl.load(M_ptr + idx, mask=mask[:, None, None], other=1.0)
        for _ in range(num_iterations):
            m = m / tl.maximum(tl.sum(m, axis=2)[:, :, None], eps)  # T_r
            m = m / tl.maximum(tl.sum(m, axis=1)[:, None, :], eps)  # T_c
        tl.store(OUT_ptr + idx, m, mask=mask[:, None, None])

    @triton.jit
    def _sinkhorn_bwd_kernel(
        M_INIT_ptr,
        GOUT_ptr,
        GIN_ptr,
        RSUM_ptr,
        CSUM_ptr,
        n_mats,
        eps,
        BLOCK: tl.constexpr,
        N: tl.constexpr,
        T: tl.constexpr,
    ):
        """Analytic gradient of the Sinkhorn iterations, in one launch.

        Reverse-mode of the two per-iteration normalizations:

            fwd  r = rowsum(M).clamp(eps); A = M/r ; c = colsum(A).clamp(eps); B = A/c
            rev  g_A = (g   - sum_i(g  *B)) / c
                 g_M = (g_A - sum_j(g_A*A)) / r

        A forward replay stores each iteration's r_t/c_t in RSUM/CSUM ([n_mats, T, N]) —
        Triton cannot hold them in a Python list — and the reverse sweep reads them
        back, reconstructing tiles as A = B*c, M = A*r. Sums are stored RAW so the
        reverse pass can recover the clamp mask: a clamped sum makes the forward a
        division by a constant, whose gradient is g/eps with no subtraction term.
        T is constexpr, so both loops unroll.
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_mats
        i = tl.arange(0, N)[None, :, None]
        j = tl.arange(0, N)[None, None, :]
        idx = offs[:, None, None] * (N * N) + i * N + j
        m2 = mask[:, None, None]
        # scratch layout [n_mats, T, N]: offs*T*N + t*N + k
        kk = tl.arange(0, N)[None, None, :]

        m_init = tl.load(M_INIT_ptr + idx, mask=m2, other=1.0)
        g = tl.load(GOUT_ptr + idx, mask=m2, other=0.0)

        m = m_init
        for t in tl.static_range(T):
            sc = offs[:, None, None] * (T * N) + t * N + kk
            r_sum = tl.sum(m, axis=2)[:, :, None]
            tl.store(RSUM_ptr + sc, tl.trans(r_sum, 0, 2, 1), mask=m2)
            a = m / tl.maximum(r_sum, eps)
            c_sum = tl.sum(a, axis=1)[:, None, :]
            tl.store(CSUM_ptr + sc, c_sum, mask=m2)
            m = a / tl.maximum(c_sum, eps)

        # `m` holds B_{T-1}; each step rebuilds A_t then M_t.
        for k in tl.static_range(T):
            t = T - 1 - k
            sc = offs[:, None, None] * (T * N) + t * N + kk
            r_sum = tl.trans(tl.load(RSUM_ptr + sc, mask=m2, other=1.0), 0, 2, 1)
            c_sum = tl.load(CSUM_ptr + sc, mask=m2, other=1.0)
            r = tl.maximum(r_sum, eps)
            c = tl.maximum(c_sum, eps)

            b = m
            a = b * c
            g_col = (g - tl.sum(g * b, axis=1)[:, None, :]) / c
            g = tl.where(c_sum > eps, g_col, g / eps)
            g_row = (g - tl.sum(g * a, axis=2)[:, :, None]) / r
            g = tl.where(r_sum > eps, g_row, g / eps)
            m = a * r

        # dL/dH = dL/dM_init * M_init, since M_init = exp(H).
        tl.store(GIN_ptr + idx, g * m_init, mask=m2)

    @triton.jit
    def _sinkhorn_log_fwd_kernel(
        H_ptr,
        OUT_ptr,
        n_mats,
        log_eps,
        BLOCK: tl.constexpr,
        N: tl.constexpr,
        T: tl.constexpr,
    ):
        """Log-space Sinkhorn: row/column log-softmax via LSE, exp only at the end.

        The direct form (exp first, then divide by sums) underflows to 0 once the
        logits span more than ~fp32's exponent range, which sends the eps-clamped
        divide to NaN in the gradient. Working in log-space is unconditionally
        stable; measured NaN-free where the direct form fails at logit scale >= 80.
        Accumulates in fp32 and casts back, so bf16/fp16 inputs are accepted.
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_mats
        i = tl.arange(0, N)[None, :, None]
        j = tl.arange(0, N)[None, None, :]
        idx = offs[:, None, None] * (N * N) + i * N + j
        m2 = mask[:, None, None]

        lw = tl.load(H_ptr + idx, mask=m2, other=0.0).to(tl.float32)
        # Row-max shift cancels in the first row LSE but makes the eps clamp compare
        # the same quantity the eager path clamps.
        lw = lw - tl.max(lw, axis=2)[:, :, None]

        for _ in tl.static_range(T):
            rmax = tl.max(lw, axis=2)[:, :, None]
            rlse = rmax + tl.log(tl.sum(tl.exp(lw - rmax), axis=2)[:, :, None])
            lw = lw - tl.maximum(rlse, log_eps)
            cmax = tl.max(lw, axis=1)[:, None, :]
            clse = cmax + tl.log(tl.sum(tl.exp(lw - cmax), axis=1)[:, None, :])
            lw = lw - tl.maximum(clse, log_eps)

        tl.store(OUT_ptr + idx, tl.exp(lw).to(OUT_ptr.dtype.element_ty), mask=m2)

    @triton.jit
    def _sinkhorn_log_bwd_kernel(
        H_ptr,
        GOUT_ptr,
        GIN_ptr,
        RLSE_ptr,
        CLSE_ptr,
        n_mats,
        log_eps,
        BLOCK: tl.constexpr,
        N: tl.constexpr,
        T: tl.constexpr,
    ):
        """Analytic gradient of the log-space forward, in one launch.

        Reverse-mode of a log-softmax is g - softmax * sum(g), so each phase needs
        exp(lw) at that point. A replay stores the raw LSEs ([n_mats, T, N]) and the
        reverse sweep walks lw back additively (y = lw + clse, lw = y + rlse), which
        keeps the scratch at 2*T*N floats instead of the 2*T*N*N a full log-state
        history would need. A clamped LSE means the phase was a shift by a constant,
        whose gradient passes through unchanged.
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_mats
        i = tl.arange(0, N)[None, :, None]
        j = tl.arange(0, N)[None, None, :]
        idx = offs[:, None, None] * (N * N) + i * N + j
        m2 = mask[:, None, None]
        kk = tl.arange(0, N)[None, None, :]

        lw = tl.load(H_ptr + idx, mask=m2, other=0.0).to(tl.float32)
        lw = lw - tl.max(lw, axis=2)[:, :, None]

        for t in tl.static_range(T):
            sc = offs[:, None, None] * (T * N) + t * N + kk
            rmax = tl.max(lw, axis=2)[:, :, None]
            rlse = rmax + tl.log(tl.sum(tl.exp(lw - rmax), axis=2)[:, :, None])
            tl.store(RLSE_ptr + sc, tl.trans(rlse, 0, 2, 1), mask=m2)
            lw = lw - tl.maximum(rlse, log_eps)
            cmax = tl.max(lw, axis=1)[:, None, :]
            clse = cmax + tl.log(tl.sum(tl.exp(lw - cmax), axis=1)[:, None, :])
            tl.store(CLSE_ptr + sc, clse, mask=m2)
            lw = lw - tl.maximum(clse, log_eps)

        # M = exp(lw_T), so dL/dlw_T = dL/dM * M.
        g = tl.load(GOUT_ptr + idx, mask=m2, other=0.0).to(tl.float32) * tl.exp(lw)

        for k in tl.static_range(T):
            t = T - 1 - k
            sc = offs[:, None, None] * (T * N) + t * N + kk
            rlse = tl.trans(tl.load(RLSE_ptr + sc, mask=m2, other=0.0), 0, 2, 1)
            clse = tl.load(CLSE_ptr + sc, mask=m2, other=0.0)

            g_col = g - tl.exp(lw) * tl.sum(g, axis=1)[:, None, :]
            g = tl.where(clse > log_eps, g_col, g)
            y = lw + tl.maximum(clse, log_eps)

            g_row = g - tl.exp(y) * tl.sum(g, axis=2)[:, :, None]
            g = tl.where(rlse > log_eps, g_row, g)
            lw = y + tl.maximum(rlse, log_eps)

        tl.store(GIN_ptr + idx, g.to(GIN_ptr.dtype.element_ty), mask=m2)


# Whether the fused Triton kernels can serve a given tensor. The log-space kernels
# accumulate in fp32, so fp64 keeps the direct-form kernels to retain double
# precision; anything else (CPU, no Triton) falls back to the eager loop.
_LOG_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


class SinkhornKnopp(torch.autograd.Function):
    """
    Differentiable Sinkhorn-Knopp algorithm for doubly stochastic projection.

    Projects a positive matrix onto the Birkhoff polytope (doubly stochastic matrices)
    via iterative row and column normalization.

    Reference: Eq. (9) in mHC paper - M^{(t)} = T_c(T_r(M^{(t-1)}))
    """

    eps = 1e-6

    # Tiles per Triton program; 64 measured fastest at n=4.
    triton_block = 64

    @staticmethod
    def _use_log_triton(t: Tensor) -> bool:
        """Whether the log-space Triton kernels can serve this tensor."""
        return HAVE_TRITON and t.is_cuda and t.dtype in _LOG_DTYPES

    @staticmethod
    def _use_triton(t: Tensor) -> bool:
        """Whether either family of Triton kernels can serve this tensor."""
        return HAVE_TRITON and t.is_cuda and t.dtype in _LOG_DTYPES + (torch.float64,)

    @staticmethod
    def _sinkhorn_log_triton(H: Tensor, num_iterations: int) -> Tensor:
        """Log-space forward: H_res_logits -> doubly stochastic M, one launch."""
        n = H.shape[-1]
        flat = H.reshape(-1, n, n).contiguous()
        out = torch.empty_like(flat)
        block = SinkhornKnopp.triton_block
        _sinkhorn_log_fwd_kernel[(triton.cdiv(flat.shape[0], block),)](
            flat,
            out,
            flat.shape[0],
            math.log(SinkhornKnopp.eps),
            BLOCK=block,
            N=n,
            T=num_iterations,
        )
        return out.reshape(H.shape)

    @staticmethod
    def _sinkhorn_log_backward_triton(
        grad_output: Tensor, H: Tensor, num_iterations: int
    ) -> Tensor:
        """Log-space analytic dL/dH_res_logits, one launch."""
        n = H.shape[-1]
        h_flat = H.reshape(-1, n, n).contiguous()
        g_flat = grad_output.reshape(-1, n, n).contiguous()
        n_mats = h_flat.shape[0]
        grad_in = torch.empty_like(h_flat)
        r_lse = torch.empty((n_mats, num_iterations, n), device=h_flat.device, dtype=torch.float32)
        c_lse = torch.empty((n_mats, num_iterations, n), device=h_flat.device, dtype=torch.float32)
        block = SinkhornKnopp.triton_block
        _sinkhorn_log_bwd_kernel[(triton.cdiv(n_mats, block),)](
            h_flat,
            g_flat,
            grad_in,
            r_lse,
            c_lse,
            n_mats,
            math.log(SinkhornKnopp.eps),
            BLOCK=block,
            N=n,
            T=num_iterations,
        )
        return grad_in.reshape(H.shape)

    @staticmethod
    def _sinkhorn_normalize(M: Tensor, num_iterations: int) -> Tensor:
        """
        Apply Sinkhorn-Knopp normalization iterations.

        Iteratively applies row and column normalization to project M
        onto the Birkhoff polytope (doubly stochastic matrices).

        Args:
            M: [s, b, n, n] - positive matrix to normalize
            num_iterations: Number of Sinkhorn iterations

        Returns:
            M: [s, b, n, n] - doubly stochastic matrix
        """
        for _ in range(num_iterations):
            # T_r: Row normalization
            M = M / M.sum(dim=-1, keepdim=True).clamp(min=SinkhornKnopp.eps)
            # T_c: Column normalization
            M = M / M.sum(dim=-2, keepdim=True).clamp(min=SinkhornKnopp.eps)
        return M

    @staticmethod
    def _sinkhorn_normalize_triton(M: Tensor, num_iterations: int) -> Tensor:
        """Single-launch _sinkhorn_normalize, WITHOUT an autograd graph.

        Triton kernels are opaque to autograd, so this is valid only where no tape is
        needed: this Function's forward, or any caller under torch.no_grad(). Callers
        must check torch.is_grad_enabled() or gradients are silently dropped.
        """
        if not (HAVE_TRITON and M.is_cuda and M.dtype == torch.float32):
            return SinkhornKnopp._sinkhorn_normalize(M, num_iterations)
        n = M.shape[-1]
        flat = M.reshape(-1, n, n).contiguous()
        out = torch.empty_like(flat)
        block = SinkhornKnopp.triton_block
        _sinkhorn_fused_kernel[(triton.cdiv(flat.shape[0], block),)](
            flat,
            out,
            flat.shape[0],
            num_iterations,
            SinkhornKnopp.eps,
            BLOCK=block,
            N=n,
        )
        return out.reshape(M.shape)

    @staticmethod
    def _sinkhorn_normalize_fwd(M: Tensor, num_iterations: int) -> Tensor:
        """Direct-form forward for an already-exponentiated M, no autograd graph."""
        return SinkhornKnopp._sinkhorn_normalize_triton(M, num_iterations)

    @staticmethod
    def _sinkhorn_backward_triton(
        grad_output: Tensor, M_init: Tensor, num_iterations: int
    ) -> Optional[Tensor]:
        """Analytic dL/dH_res_logits in one launch (see _sinkhorn_bwd_kernel).

        The `* M_init` chain rule is already applied. Returns None when Triton is
        unavailable for this dtype/device, so the caller can fall back.
        """
        if not (HAVE_TRITON and M_init.is_cuda and M_init.dtype in (torch.float32, torch.float64)):
            return None
        n = M_init.shape[-1]
        m_flat = M_init.reshape(-1, n, n).contiguous()
        g_flat = grad_output.reshape(-1, n, n).contiguous()
        n_mats = m_flat.shape[0]
        grad_in = torch.empty_like(m_flat)
        # Per-iteration raw row/column sums; 2 * T * N floats per matrix.
        r_sums = torch.empty((n_mats, num_iterations, n), device=m_flat.device, dtype=m_flat.dtype)
        c_sums = torch.empty((n_mats, num_iterations, n), device=m_flat.device, dtype=m_flat.dtype)
        block = SinkhornKnopp.triton_block
        _sinkhorn_bwd_kernel[(triton.cdiv(n_mats, block),)](
            m_flat,
            g_flat,
            grad_in,
            r_sums,
            c_sums,
            n_mats,
            SinkhornKnopp.eps,
            BLOCK=block,
            N=n,
            T=num_iterations,
        )
        return grad_in.reshape(M_init.shape)

    @staticmethod
    def forward(ctx, H_res_logits: Tensor, num_iterations: int) -> Tensor:
        """
        Project to doubly stochastic matrix via iterative row/col normalization.

        Args:
            H_res_logits: [s, b, n, n] - raw logits for residual mixing matrix
            num_iterations: Number of Sinkhorn iterations (paper uses 20)

        Returns:
            H_res: [s, b, n, n] - doubly stochastic matrix
        """
        ctx.num_iterations = num_iterations

        # Log-space path: never materializes exp(H), so it survives logit ranges
        # where the direct form underflows to 0 and the eps-clamped divide NaNs.
        if SinkhornKnopp._use_log_triton(H_res_logits):
            ctx.log_space = True
            ctx.save_for_backward(H_res_logits)
            return SinkhornKnopp._sinkhorn_log_triton(H_res_logits, num_iterations)

        ctx.log_space = False
        # Gradients are computed explicitly in backward via recomputation.
        # Numerical-stability shift: subtract the per-row max before exp to prevent
        # overflow. This row-wise constant is invariant under Sinkhorn-Knopp
        # normalization: the first iteration's row normalization (T_r) divides each
        # row by its sum, which cancels any per-row scalar — i.e. exp(H_ij - c_i) and
        # exp(H_ij) produce identical row-normalized matrices, hence the same Sinkhorn
        # fixed point and the same gradient. The shift therefore changes only the
        # numeric stability of the exp, not the algorithm's output.
        M_init = torch.exp(H_res_logits - H_res_logits.max(dim=-1, keepdim=True).values)

        # autograd.Function.forward always runs with grad disabled, so no tape is
        # needed here — the fast no-grad backend is safe.
        M = SinkhornKnopp._sinkhorn_normalize_fwd(M_init, num_iterations)

        ctx.save_for_backward(M_init)
        return M

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        """
        Backward through Sinkhorn-Knopp iterations.

        A single analytic Triton kernel where Triton can serve the tensor; otherwise
        the forward pass is recomputed with gradient tracking.
        """
        (saved,) = ctx.saved_tensors
        num_iterations = ctx.num_iterations

        if ctx.log_space:
            # `saved` is H_res_logits; the kernel returns dL/dH directly.
            return (
                SinkhornKnopp._sinkhorn_log_backward_triton(
                    grad_output, saved, num_iterations
                ),
                None,
            )

        M_init = saved
        grad_input = SinkhornKnopp._sinkhorn_backward_triton(
            grad_output, M_init, num_iterations
        )
        if grad_input is not None:
            return grad_input, None
        # Triton unavailable for this dtype/device — recompute under autograd.

        with torch.enable_grad():
            # Leaf for recomputation
            M_input = M_init.detach().requires_grad_(True)

            M_current = SinkhornKnopp._sinkhorn_normalize(M_input, num_iterations)

            # Compute dL/dM_input (i.e., dL/dM_init) via autograd
            (grad_M_init,) = torch.autograd.grad(
                outputs=M_current,
                inputs=M_input,
                grad_outputs=grad_output,
                create_graph=False,
                retain_graph=False,
            )
        # Apply chain rule: dL/dH = dL/dM_init * dM_init/dH = dL/dM_init * M_init
        # Since M_init = exp(H_res_logits), we have d(exp(x))/dx = exp(x) = M_init
        grad_input = grad_M_init * M_init

        return grad_input, None


# TODO: keep hyper connection in fp32 computation
class HyperConnectionModule(MegatronModule):
    """
    Unified mHC (Manifold-Constrained Hyper-Connections) module.

    Implements the complete mHC propagation:
        x_{l+1} = H_res @ x_l + H_post^T @ F(H_pre @ x_l)

    This module handles:
    1. Computing learnable mappings: H_pre, H_post, H_res (with Sinkhorn-Knopp projection)
    2. Aggregation: n-stream → 1-stream (H_pre @ x)
    3. Expansion: 1-stream → n-stream (H_post^T @ output)
    4. Residual merge: H_res @ x + expanded_output
    5. Block-level expand/contract for TransformerBlock boundaries

    Args:
        config: TransformerConfig with hyper-connection fields
        layer_number: Current layer index for initialization
    """

    def __init__(self, config: TransformerConfig, layer_number: int):
        super().__init__(config)
        self.config = config
        self.layer_number = layer_number
        self.n = config.num_residual_streams
        self.hidden_size = config.hidden_size
        self.sinkhorn_iterations = config.mhc_sinkhorn_iterations

        # Projection weights for dynamic mappings
        # Input: [s, b, n*C] -> Output: n^2 + 2n values per token
        # - H_pre: n values
        # - H_post: n values
        # - H_res: n^2 values (before Sinkhorn projection)
        self.mapping_proj = nn.Linear(
            self.n * self.hidden_size, self.n * self.n + 2 * self.n, bias=False
        )

        init_alpha = config.mhc_init_gating_factor
        # Learnable scaling factors (Eq. 5 in paper)
        self.alpha_pre = nn.Parameter(torch.full((1,), init_alpha))
        self.alpha_post = nn.Parameter(torch.full((1,), init_alpha))
        self.alpha_res = nn.Parameter(torch.full((1,), init_alpha))

        # Static bias terms
        self.bias = nn.Parameter(torch.zeros(self.n * self.n + 2 * self.n))
        self.norm_eps = 1e-6

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for stable training."""
        nn.init.xavier_uniform_(self.mapping_proj.weight)

        # Set sequence_parallel attribute on parameters for gradient synchronization
        # across TP ranks when sequence_parallel is enabled.
        # This is required because HyperConnectionModule uses non-TP-aware layers
        # (nn.Linear, nn.RMSNorm) whose gradients need to be all-reduced.
        if self.config.sequence_parallel:
            setattr(self.mapping_proj.weight, 'sequence_parallel', True)
            setattr(self.alpha_pre, 'sequence_parallel', True)
            setattr(self.alpha_post, 'sequence_parallel', True)
            setattr(self.alpha_res, 'sequence_parallel', True)
            setattr(self.bias, 'sequence_parallel', True)

    @torch.compile
    def _projection_and_get_norm(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Project input hidden states to mapping space and apply RMS normalization.

        Args:
            x: [s, b, n*C] - n-stream hidden states
        """
        nC = x.shape[-1]
        r = x.norm(dim=-1, keepdim=True) / math.sqrt(nC)  # shape: [s, b, 1]
        r = 1.0 / (r + self.norm_eps)  # shape: [s, b, 1]
        # Upcast the projection weight to the input dtype so the mapping stays in fp32
        # even when params are stored in bf16/fp16 (compute_mappings feeds fp32 x here).
        proj = torch.nn.functional.linear(
            x, self.mapping_proj.weight.to(x.dtype)
        )  # [s, b, n^2 + 2n]
        return proj, r

    @torch.compile
    def _compute_h(self, proj: Tensor, r: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute h from projected hidden states and scaling factors.

        Args:
            proj: [s, b, n^2 + 2n] - projected hidden states
            r: [s, b, 1] - scaling factors

        Returns:
            h_pre: [s, b, n] - aggregation weights
            h_post: [s, b, n] - expansion weights
            h_res: [s, b, n^2] - residual mixing logits
        """
        alpha_ = torch.cat(
            [
                self.alpha_pre.expand(self.n),
                self.alpha_post.expand(self.n),
                self.alpha_res.expand(self.n * self.n),
            ],
            dim=-1,
        )
        h = r * proj * alpha_ + self.bias
        # H_pre = σ(α_pre * (θ_pre @ x̃) + b_pre)
        h_pre = h[..., : self.n].sigmoid()  # [s, b, n]

        # H_post = 2σ(α_post * (θ_post @ x̃) + b_post)
        h_post = h[..., self.n : 2 * self.n].sigmoid() * 2  # [s, b, n]
        h_res = h[..., 2 * self.n :]
        return h_pre, h_post, h_res

    @nvtx_decorator(message="HyperConnection::compute_mappings")
    def compute_mappings(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute mHC mappings from input hidden states.

        Reference: Eq. (5) and (8) in mHC paper

        Args:
            x: [s, b, n*C] - n-stream hidden states

        Returns:
            h_pre: [s, b, n] - aggregation weights (sigmoid activated)
            h_post: [s, b, n] - expansion weights (2*sigmoid activated)
            h_res: [s, b, n, n] - residual mixing matrix (doubly stochastic)

        The mappings (RMS-norm scaling, sigmoid gating, Sinkhorn doubly-stochastic
        projection) are numerically sensitive, so they are always computed AND returned
        in fp32 — regardless of the hidden/param dtype or any active autocast. The
        downstream mixing ops (aggregate / apply_h_res / _apply_h_post) do their
        arithmetic in fp32 and cast the result back to the hidden dtype, so returning
        fp32 mappings never breaks the bf16/fp16 hidden-state contract.
        """
        s, b, _ = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            with torch.cuda.nvtx.range("HyperConnection::projection_and_get_norm"):
                proj, r = self._projection_and_get_norm(x)
            with torch.cuda.nvtx.range("HyperConnection::compute_h"):
                h_pre, h_post, h_res = self._compute_h(proj, r)
            h_res_logits = h_res.view(s, b, self.n, self.n)
            # SinkhornKnopp.apply serves every path, including full-RC's recompute
            # pass: its Triton backward computes the analytic gradient from the saved
            # input, so nesting it inside an outer recompute costs one extra kernel
            # rather than a second 20-iteration chain.
            h_res = SinkhornKnopp.apply(
                h_res_logits, self.sinkhorn_iterations
            )  # [s, b, n, n]

        return h_pre, h_post, h_res

    @torch.compile
    def _apply_h_post(self, x: Tensor, h_post: Tensor) -> Tensor:
        """
        Core implementation of H_post application to a single tensor.

        Computes: H_post^T @ x

        Args:
            x: Input tensor, can be either:
               - [s, b, C] - standard hidden states
               - [C] - bias tensor (will be broadcast)
            h_post: [s, b, n] - expansion weights

        Returns:
            output: [s, b, n*C] - expanded tensor
        """
        n = self.n
        s, b, _ = h_post.shape

        if x.dim() == 1:
            # x is bias with shape [C], need to broadcast to [s, b, 1, C]
            C = x.shape[0]
            x_expanded = x.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(s, b, 1, C)
        else:
            # x is [s, b, C]
            C = x.shape[-1]
            x_expanded = x.unsqueeze(2)  # [s, b, 1, C]

        # h_post^T @ x : [s, b, n, 1] * [s, b, 1, C] -> [s, b, n, C]
        # h_post is fp32 (see compute_mappings); do the expand in fp32, cast back to x's dtype.
        result = h_post.float().unsqueeze(-1) * x_expanded.float()
        return result.view(s, b, n * C).to(x.dtype)

    @nvtx_decorator(message="HyperConnection::apply_h_post")
    def apply_h_post(
        self,
        x_with_bias: Tuple[Tensor, Optional[Tensor]],
        h_post: Tensor,
        manager: Optional['CheckpointWithoutOutputManager'] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Apply H_post to x and optionally bias, with optional checkpointing.

        This is the unified entry point that handles both normal execution
        and checkpoint-based execution for memory efficiency.

        Args:
            x_with_bias: Tuple of (x, bias) where:
                - x: [s, b, C] - hidden states
                - bias: [C] or None - optional bias tensor
            h_post: [s, b, n] - expansion weights
            manager: Optional CheckpointWithoutOutputManager for checkpoint management.
                When provided, wraps _apply_h_post with CheckpointWithoutOutput.

        Returns:
            Tuple of (x_out, bias_out) where:
                - x_out: [s, b, n*C] - expanded hidden states
                - bias_out: [s, b, n*C] or None - expanded bias if input bias was not None
        """
        x, bias = x_with_bias

        if manager is not None:
            from megatron.core.tensor_parallel.random import CheckpointWithoutOutput

            # Checkpoint _apply_h_post to discard the output
            x_out = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(
                self._apply_h_post, x, h_post
            )

            # Checkpoint _apply_h_post for bias if not None
            if bias is not None:
                bias_out = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(
                    self._apply_h_post, bias, h_post
                )
            else:
                bias_out = None
        else:
            # Normal execution without checkpoint
            x_out = self._apply_h_post(x, h_post)
            bias_out = self._apply_h_post(bias, h_post) if bias is not None else None

        return x_out, bias_out

    @torch.compile
    def aggregate(self, x: Tensor, h_pre: Tensor) -> Tensor:
        """
        Aggregate n-stream to 1-stream using H_pre weights.

        Computes: sum_i(h_pre_i * x_stream_i)

        Args:
            x: [s, b, n*C] - n-stream hidden states
            h_pre: [s, b, n] - aggregation weights

        Returns:
            aggregated: [s, b, C] - single stream hidden states
        """
        s, b, _ = x.shape
        C = self.hidden_size

        # Reshape to [s, b, n, C]
        x_streams = x.view(s, b, self.n, C)

        # Weighted sum: [s, b, n, C] * [s, b, n, 1] -> sum over n -> [s, b, C]
        # h_pre is fp32 (see compute_mappings); reduce in fp32, cast back to x's dtype.
        aggregated = (x_streams.float() * h_pre.float().unsqueeze(-1)).sum(dim=2)

        return aggregated.to(x.dtype)

    @torch.compile
    def apply_h_res(self, h_res: Tensor, residual: Tensor) -> Tensor:
        """
        Apply H_res to residual using H_res weights.

        Computes: H_res @ residual

        Args:
            h_res: [s, b, n, n] - residual mixing matrix
            residual: [s, b, n*C] - n-stream hidden states
        """
        s, b, _ = residual.shape
        n = self.n
        C = self.hidden_size

        # Reshape for bmm: [s, b, n, n] -> [s*b, n, n]
        h_res_batched = h_res.view(s * b, n, n)
        # [s, b, n*C] -> [s, b, n, C] -> [s*b, n, C]
        residual_batched = residual.view(s, b, n, C).view(s * b, n, C)

        # Batch matrix multiply: [s*b, n, n] @ [s*b, n, C] -> [s*b, n, C]
        # h_res is fp32 (see compute_mappings); do the mix in fp32, cast back to residual's dtype.
        mixed = torch.bmm(h_res_batched.float(), residual_batched.float())

        return mixed.view(s, b, n * C).to(residual.dtype)

    def forward(
        self,
        hidden_states: Tensor,
        mhc_recompute_manager: Optional['CheckpointWithoutOutputManager'] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Full mHC forward pass.

        Args:
            hidden_states: [s, b, n*C] - n-stream hidden states
            mhc_recompute_manager: Optional CheckpointWithoutOutputManager for checkpoint mgmt.
                When provided, uses _forward_with_checkpoint for memory-efficient execution.

        Returns:
            aggregated: [s, b, C] - aggregated input for layer computation
            h_res: [s, b, n, n] - residual mixing matrix (for fused kernel)
            h_post: [s, b, n] - expansion weights
        """
        if mhc_recompute_manager is not None:
            return self._forward_with_checkpoint(hidden_states, mhc_recompute_manager)
        else:
            return self._forward_normal(hidden_states)

    def _forward_normal(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Normal forward pass without checkpointing.

        Args:
            hidden_states: [s, b, n*C] - n-stream hidden states

        Returns:
            aggregated: [s, b, C] - aggregated input for layer computation
            h_res: [s, b, n, n] - residual mixing matrix (for fused kernel)
            h_post: [s, b, n] - expansion weights
        """
        # Compute mappings
        h_pre, h_post, h_res = self.compute_mappings(hidden_states)

        # Aggregate for layer input
        with torch.cuda.nvtx.range("HyperConnection::aggregate"):
            aggregated = self.aggregate(hidden_states, h_pre)

        return aggregated, h_res, h_post

    def _forward_with_checkpoint(
        self, hidden_states: Tensor, manager: 'CheckpointWithoutOutputManager'
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass with checkpointing for memory efficiency.

        compute_mappings is called directly (not checkpointed) since its outputs
        (h_pre, h_post, h_res) are needed downstream. Only aggregate is wrapped with
        CheckpointWithoutOutput and auto-registered to the manager.
        apply_h_res is deferred to fused_h_res_h_post_bda for kernel fusion.

        Args:
            hidden_states: [s, b, n*C] - n-stream hidden states
            manager: CheckpointWithoutOutputManager for unified recomputation

        Returns:
            aggregated: [s, b, C] - aggregated input for layer computation
            h_res: [s, b, n, n] - residual mixing matrix (for fused kernel)
            h_post: [s, b, n] - expansion weights
        """
        from megatron.core.tensor_parallel.random import CheckpointWithoutOutput

        h_pre, h_post, h_res = self.compute_mappings(hidden_states)

        # Checkpoint aggregate - auto-registers to manager
        aggregated = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(
            self.aggregate, hidden_states, h_pre
        )

        return aggregated, h_res, h_post

    # ==================== Block-level utilities ====================

    @staticmethod
    def input_expand(x: Tensor, n: int) -> Tensor:
        """
        Expand 1-stream to n-stream at TransformerBlock entry.

        Simple replication strategy: each stream initialized as a copy of input.

        Args:
            x: [s, b, C] - single stream hidden states
            n: Number of residual streams

        Returns:
            expanded: [s, b, n*C] - n-stream hidden states
        """
        s, b, C = x.shape
        # Replicate input to n streams
        expanded = x.unsqueeze(2).expand(s, b, n, C).contiguous()
        return expanded.view(s, b, n * C)

    @staticmethod
    def output_contract(x: Tensor, n: int) -> Tensor:
        """
        Contract n-stream to 1-stream at TransformerBlock exit.

        Simple averaging strategy: average all streams.

        Args:
            x: [s, b, n*C] - n-stream hidden states
            n: Number of residual streams

        Returns:
            contracted: [s, b, C] - single stream hidden states
        """
        s, b, nC = x.shape
        C = nC // n
        # Average all streams
        x_streams = x.view(s, b, n, C)
        contracted = x_streams.mean(dim=2)
        return contracted

    # ==================== Fused kernel placeholder ====================

    @nvtx_decorator(message="HyperConnection::fused_h_res_h_post_bda")
    def fused_h_res_h_post_bda(
        self,
        h_res: Tensor,
        original_residual: Tensor,
        h_post: Tensor,
        layer_output_with_bias: Tuple[Tensor, Optional[Tensor]],
        dropout_prob: float,
        training: bool,
        fused: bool,
        manager: Optional['CheckpointWithoutOutputManager'] = None,
    ) -> Tensor:
        """
        Fused kernel combining apply_h_res, apply_h_post and bias-dropout-add.

        This is a placeholder for future kernel fusion optimization.
        Currently implements the operations sequentially using native PyTorch.

        The computation flow is:
            1. mixed = H_res @ original_residual (apply_h_res)
            2. expanded = H_post^T @ layer_output (apply_h_post)
            3. output = dropout(expanded + bias) + mixed (bias-dropout-add)

        Args:
            h_res: [s, b, n, n] - residual mixing matrix
            original_residual: [s, b, n*C] - n-stream hidden states (before H_res applied)
            h_post: [s, b, n] - expansion weights
            layer_output_with_bias: Tuple of (x, bias) where:
                - x: [s, b, C] - layer output (attention or MLP output)
                - bias: [C] or None - optional bias tensor
            dropout_prob: Dropout probability
            training: Whether in training mode
            fused: Whether to use fused BDA implementation
            manager: Optional CheckpointWithoutOutputManager for checkpoint management.
                When provided, each operation is wrapped with CheckpointWithoutOutput.

        Returns:
            output: [s, b, n*C] - final output after all operations
        """
        if manager is not None:
            return self._fused_h_res_h_post_bda_with_checkpoint(
                h_res,
                original_residual,
                h_post,
                layer_output_with_bias,
                dropout_prob,
                training,
                fused,
                manager,
            )
        else:
            return self._fused_h_res_h_post_bda_native(
                h_res,
                original_residual,
                h_post,
                layer_output_with_bias,
                dropout_prob,
                training,
                fused,
            )

    def _fused_h_res_h_post_bda_native(
        self,
        h_res: Tensor,
        original_residual: Tensor,
        h_post: Tensor,
        layer_output_with_bias: Tuple[Tensor, Optional[Tensor]],
        dropout_prob: float,
        training: bool,
        fused: bool,
    ) -> Tensor:
        """
        Native implementation of fused h_res, h_post and bda operations.

        Args:
            h_res: [s, b, n, n] - residual mixing matrix
            original_residual: [s, b, n*C] - n-stream hidden states
            h_post: [s, b, n] - expansion weights
            layer_output_with_bias: Tuple of (x, bias)
            dropout_prob: Dropout probability
            training: Whether in training mode
            fused: Whether to use fused BDA implementation

        Returns:
            output: [s, b, n*C] - final output
        """
        from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add

        # Step 1: Apply H_res to original residual
        with torch.cuda.nvtx.range("HyperConnection::apply_h_res"):
            mixed = self.apply_h_res(h_res, original_residual)

        # Step 2: Apply H_post to layer output
        x, bias = layer_output_with_bias
        with torch.cuda.nvtx.range("HyperConnection::apply_h_post"):
            x_expanded = self._apply_h_post(x, h_post)
            bias_expanded = self._apply_h_post(bias, h_post) if bias is not None else None

        # Step 3: Bias-dropout-add
        bda_func = get_bias_dropout_add(training, fused)
        with torch.cuda.nvtx.range("HyperConnection::bda"):
            output = bda_func((x_expanded, bias_expanded), mixed, dropout_prob)

        return output

    @nvtx_decorator(message="HyperConnection::fused_h_res_h_post_bda_with_checkpoint")
    def _fused_h_res_h_post_bda_with_checkpoint(
        self,
        h_res: Tensor,
        original_residual: Tensor,
        h_post: Tensor,
        layer_output_with_bias: Tuple[Tensor, Optional[Tensor]],
        dropout_prob: float,
        training: bool,
        fused: bool,
        manager: 'CheckpointWithoutOutputManager',
    ) -> Tensor:
        """
        Checkpointed implementation of fused h_res, h_post and bda operations.

        Uses a single checkpoint wrapper around all operations for memory efficiency.

        Args:
            h_res: [s, b, n, n] - residual mixing matrix
            original_residual: [s, b, n*C] - n-stream hidden states
            h_post: [s, b, n] - expansion weights
            layer_output_with_bias: Tuple of (x, bias)
            dropout_prob: Dropout probability
            training: Whether in training mode
            fused: Whether to use fused BDA implementation
            manager: CheckpointWithoutOutputManager for checkpoint management

        Returns:
            output: [s, b, n*C] - final output
        """
        from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
        from megatron.core.tensor_parallel.random import CheckpointWithoutOutput

        # Get BDA function (captured via closure)
        bda_func = get_bias_dropout_add(training, fused)

        # Unpack layer_output_with_bias to avoid tuple tensors in checkpoint args
        x, bias = layer_output_with_bias
        has_bias = bias is not None

        # Native wrapper that combines all operations without internal checkpointing.
        # Non-tensor args (dropout_prob, has_bias) are captured via closure.
        def _native_wrapper(h_res, original_residual, h_post, x, *optional_bias):
            # Step 1: Apply H_res to original residual
            with torch.cuda.nvtx.range("HyperConnection::apply_h_res"):
                mixed = self.apply_h_res(h_res, original_residual)

            # Step 2: Apply H_post to x and bias
            with torch.cuda.nvtx.range("HyperConnection::apply_h_post"):
                x_expanded = self._apply_h_post(x, h_post)
                if has_bias:
                    bias_expanded = self._apply_h_post(optional_bias[0], h_post)
                else:
                    bias_expanded = None

            # Step 3: Bias-dropout-add
            with torch.cuda.nvtx.range("HyperConnection::bda"):
                output = bda_func((x_expanded, bias_expanded), mixed, dropout_prob)

            return output

        # Use a single checkpoint wrapper for all operations
        ckpt = CheckpointWithoutOutput(ckpt_manager=manager)
        if has_bias:
            output = ckpt.checkpoint(_native_wrapper, h_res, original_residual, h_post, x, bias)
        else:
            output = ckpt.checkpoint(_native_wrapper, h_res, original_residual, h_post, x)

        return output


# ==================== Checkpoint utilities for mHC ====================


class HyperConnectionCheckpoint:
    """
    Checkpoint utility for mHC intermediate activations.

    Implements the paper's "recomputing strategy" to reduce memory footprint
    by discarding intermediate n-stream activations and recomputing on-the-fly.
    """

    @staticmethod
    def compute_optimal_block_size(num_layers: int, num_streams: int) -> int:
        """
        Compute optimal recomputation block size.

        From paper Eq. (20): L_r^* ≈ sqrt(nL/(n+2))

        Args:
            num_layers: Total number of transformer layers
            num_streams: Number of residual streams (n)

        Returns:
            block_size: Optimal block size for checkpointing
        """
        block_size = int(math.sqrt(num_streams * num_layers / (num_streams + 2)))
        return max(1, block_size)

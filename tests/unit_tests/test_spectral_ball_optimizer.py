# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os

import pytest
import torch
from packaging.version import Version

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.emerging_optimizers import (
    HAVE_EMERGING_OPTIMIZERS,
    _EMERGING_OPTIMIZERS,
)
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

if HAVE_EMERGING_OPTIMIZERS:
    from megatron.core.optimizer.spectral_ball import MuonBall, SpectralBall
    from megatron.core.optimizer.spectral_ball.spectral_ball_utils import (
        compute_target_radius,
        msign,
        power_iteration,
    )
else:
    MuonBall = SpectralBall = None


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(80, 48)
        self.fc2 = torch.nn.Linear(48, 32)

    def forward(self, x):
        return self.fc2(torch.nn.functional.relu(self.fc1(x)))


pytestmark = [
    pytest.mark.skipif(
        Version(os.getenv('NVIDIA_PYTORCH_VERSION', "24.01")) <= Version("25.05"),
        reason="Skip emerging optimizer tests for LTS test",
    ),
    pytest.mark.skipif(
        not HAVE_EMERGING_OPTIMIZERS, reason="emerging_optimizers package is not installed"
    ),
]


def _make_opt(cls, param, **kwargs):
    defaults = dict(
        lr=0.01,
        momentum=0.9,
        nesterov=True,
        weight_decay=0.0,
        radius_mode="identity",
        msign_steps=5,
        power_iteration_steps=10,
    )
    defaults.update(kwargs)
    return cls(params=[param], **defaults)


@pytest.mark.parametrize("cls_name", ["spectral_ball", "muon_ball"])
def test_spectral_ball_registered(cls_name):
    """Both optimizers must be discoverable through the emerging-optimizer registry."""
    assert cls_name in _EMERGING_OPTIMIZERS
    entry = _EMERGING_OPTIMIZERS[cls_name]
    assert entry.config_to_kwargs is not None
    # Managed 2D params must get wd_mult=0.0: the sphere constraint bounds the weights.
    overrides = entry.default_param_overrides
    assert any(o.get('wd_mult') == 0.0 for o in overrides.values())
    assert any(o.get('optimizer') == 'adam' for o in overrides.values())


@pytest.mark.parametrize("cls", [SpectralBall, MuonBall])
def test_spectral_ball_smoke(cls):
    """One step must change the weight and keep it finite."""
    model = torch.nn.Linear(64, 32, bias=False, dtype=torch.float32, device='cuda')
    optimizer = _make_opt(cls, model.weight)

    assert len(optimizer.param_groups) > 0
    # HEAD's OrthogonalizedOptimizer names these 'momentum' / 'nesterov'.
    assert optimizer.param_groups[0]['momentum'] == 0.9
    assert optimizer.nesterov is True

    model(torch.randn(8, 64, device='cuda')).sum().backward()
    original = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(model.weight.data, original)
    assert torch.isfinite(model.weight.data).all()

    state_dict = optimizer.state_dict()
    assert 'state' in state_dict and 'param_groups' in state_dict
    optimizer.load_state_dict(state_dict)


@pytest.mark.parametrize("cls", [SpectralBall, MuonBall])
def test_spectral_ball_multiple_steps(cls):
    """Repeated steps must stay finite and hold the spectral radius near R."""
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.randn(64, 32, device='cuda') * 0.1)
    optimizer = _make_opt(cls, param, radius_scaler=1.0)

    for _ in range(5):
        param.grad = torch.randn_like(param) * 0.01
        optimizer.step()
        assert torch.isfinite(param).all()

    sigma = torch.linalg.matrix_norm(param.detach(), ord=2).item()
    # radius_mode="identity" with radius_scaler=1.0 targets R=1.
    assert 0.5 < sigma < 2.0, f"spectral norm drifted off the sphere: {sigma}"


@pytest.mark.parametrize("radius_mode", ["spectral_mup", "identity"])
def test_compute_target_radius(radius_mode):
    """spectral_mup scales with sqrt(fan_out/fan_in); identity ignores shape."""
    r = compute_target_radius(shape=(64, 16), radius_mode=radius_mode, radius_scaler=1.0)
    assert r > 0
    if radius_mode == "identity":
        assert r == pytest.approx(1.0)
    else:
        assert r == pytest.approx((64 / 16) ** 0.5)


def test_msign_is_approximately_orthogonal():
    """msign(G) must approximate the orthogonal polar factor of G.

    Guards the branch's own warning that Newton-Schulz is rounding sensitive: if the
    iteration is run in a precision too low to converge, the singular values of the
    result stop being close to 1 and this fails.
    """
    torch.manual_seed(0)
    G = torch.randn(64, 64, device='cuda', dtype=torch.float32)
    Q = msign(G, steps=8).float()
    svals = torch.linalg.svdvals(Q)
    assert torch.isfinite(Q).all()
    assert svals.max().item() < 1.3
    assert svals.min().item() > 0.7


def test_power_iteration_matches_svd():
    """power_iteration must recover the leading singular value."""
    torch.manual_seed(0)
    W = torch.randn(48, 32, device='cuda', dtype=torch.float32)
    sigma, u, v = power_iteration(W, steps=50)
    expected = torch.linalg.matrix_norm(W, ord=2).item()
    assert sigma.item() == pytest.approx(expected, rel=0.05)
    assert u.shape[0] == 48 and v.shape[0] == 32


def test_spectral_ball_rejects_bad_config():
    """Invalid enum values must fail loudly at construction."""
    param = torch.nn.Parameter(torch.randn(16, 8, device='cuda'))
    with pytest.raises(ValueError):
        _make_opt(SpectralBall, param, radius_mode="not_a_mode")
    with pytest.raises(ValueError):
        _make_opt(SpectralBall, param, retract_mode="not_a_mode")
    with pytest.raises(ValueError):
        _make_opt(SpectralBall, param, msign_steps=0)


class TestSpectralBallGetMegatronOptimizer:
    """End-to-end construction through ``get_megatron_optimizer``."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    def _ddp(self, model):
        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        return DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )

    @pytest.mark.parametrize("optimizer_name", ["spectral_ball", "muon_ball"])
    def test_get_megatron_optimizer_smoke(self, optimizer_name):
        """2D weights must route to the spectral-ball optimizer, the rest to Adam."""
        model = self._ddp(Net().bfloat16().cuda().requires_grad_(True))

        optimizer_config = OptimizerConfig(
            optimizer=optimizer_name,
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_distributed_optimizer=False,
            spectral_ball_radius_mode="identity",
            spectral_ball_msign_steps=5,
            spectral_ball_power_iteration_steps=10,
            spectral_ball_solver_max_iterations=10,
        )
        optimizer = get_megatron_optimizer(
            config=optimizer_config, model_chunks=[model], use_gloo_process_groups=True
        )
        assert hasattr(optimizer, 'chained_optimizers')
        inner = [type(o.optimizer).__name__ for o in optimizer.chained_optimizers]
        expected = 'SpectralBall' if optimizer_name == 'spectral_ball' else 'MuonBall'
        assert expected in inner, f"{expected} not among {inner}"
        assert 'Adam' in ' '.join(inner) or 'AdamW' in ' '.join(inner), inner

        model(torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')).sum().backward()
        original = {n: p.data.clone() for n, p in model.named_parameters()}
        optimizer.step()

        updated = sum(
            1 for n, p in model.named_parameters() if not torch.equal(p.data, original[n])
        )
        assert updated > 0
        for _, p in model.named_parameters():
            assert torch.isfinite(p.data).all()

    def test_weight_decay_disabled_on_managed_params(self):
        """The sphere constraint bounds ‖W‖₂, so managed 2D params must get wd_mult=0.

        ``weight_decay`` itself is only multiplied by ``wd_mult`` when
        ``OptimizerParamScheduler.step`` runs, so assert on ``wd_mult``.
        """
        model = self._ddp(Net().bfloat16().cuda().requires_grad_(True))
        optimizer_config = OptimizerConfig(
            optimizer='spectral_ball',
            lr=0.01,
            weight_decay=0.1,
            bf16=True,
            use_distributed_optimizer=False,
            spectral_ball_radius_mode="identity",
        )
        optimizer = get_megatron_optimizer(
            config=optimizer_config, model_chunks=[model], use_gloo_process_groups=True
        )
        checked = 0
        for chained in optimizer.chained_optimizers:
            if type(chained.optimizer).__name__ != 'SpectralBall':
                continue
            for group in chained.optimizer.param_groups:
                assert group['wd_mult'] == 0.0, group['wd_mult']
                checked += 1
        assert checked > 0, "no SpectralBall param groups found"


@pytest.mark.parametrize("dtype_name,dtype", [("fp32", torch.float32), ("bf16", torch.bfloat16)])
def test_msign_dtype_switch(dtype_name, dtype):
    """Both dtypes must run and stay finite, and the switch must take effect.

    Note: on a well-conditioned random matrix the orthogonality residuals of the two
    dtypes are comparable, so this does NOT assert fp32 is numerically better. It only
    pins the plumbing. fp32 remains the default because the upstream branch documents
    bf16 as unsafe for this iteration, not because this test demonstrates it.
    """
    from megatron.core.optimizer.spectral_ball import spectral_ball_utils as sbu

    torch.manual_seed(0)
    G = torch.randn(128, 128, device='cuda', dtype=torch.float32)
    original = sbu.MSIGN_DTYPE
    try:
        sbu.set_msign_dtype(dtype)
        assert sbu.MSIGN_DTYPE is dtype
        Q = sbu.msign(G, steps=8).float()
    finally:
        sbu.set_msign_dtype(original)

    assert torch.isfinite(Q).all()
    svals = torch.linalg.svdvals(Q)
    assert svals.max().item() < 1.3


def test_msign_dtype_rejects_other_dtypes():
    from megatron.core.optimizer.spectral_ball import spectral_ball_utils as sbu

    with pytest.raises(ValueError):
        sbu.set_msign_dtype(torch.float16)

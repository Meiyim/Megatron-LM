# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Spectral-sphere-constrained optimizers (SSO).

Vendored from the SSO branch (Unakar/Megatron-LM ``SSO_main``). Upstream kept these
inside the ``emerging_optimizers`` package; here they live in-tree because the
installed ``emerging_optimizers`` release does not ship them.
"""

from .muon_ball import MuonBall
from .spectral_ball import SpectralBall
from .spectral_ball_utils import (
    apply_retract,
    compute_spectral_ball_update,
    compute_target_radius,
    get_spectral_ball_scale_factor,
    msign,
    power_iteration,
)


__all__ = [
    "MuonBall",
    "SpectralBall",
    "apply_retract",
    "compute_spectral_ball_update",
    "compute_target_radius",
    "get_spectral_ball_scale_factor",
    "msign",
    "power_iteration",
]

"""
Differentiable RL task interface (torch-side).

The differentiable analog of :mod:`cosmic_warp.rl.tasks`: a :class:`DiffTask`
supplies the action mapping, observation and reward as **pure torch functions
of the bridged state tensor** (see :data:`cosmic_warp.rl.diff_bridge.X_DIM`),
so torch autodiffs them as part of a policy loss — no adjoint-safe warp
kernels needed, and numerics can be made smooth where a kernel task's math is
not.

The package defines only the protocol; tasks are application code.
``examples/shac.py`` has a reference implementation (``AttitudeDiffTask``,
used by example 17).

Everything is float64 on the dynamics graph; casting to the policy's float32
happens in the trainer, not here.
"""
from typing import Protocol

import torch


class DiffTask(Protocol):
    """Task interface consumed by :class:`~cosmic_warp.rl.diff_env.DiffVecEnv`.

    All three operations are differentiable torch functions; ``bind`` is
    called once and caches whatever constants they need (inertia, torque
    limits) as device tensors. A cached value must stay constant: domain
    randomization redraws vehicle parameters at every reset, so a task that
    caches a randomized parameter must reject that randomization.
    """

    #: Per-environment action dimension (set during :meth:`bind`).
    act_dim: int
    #: Per-environment observation dimension (set during :meth:`bind`).
    obs_dim: int

    def bind(self, sim) -> None:
        """Cache simulator constants and compute ``act_dim`` / ``obs_dim``."""
        ...

    def commands(self, action: torch.Tensor) -> torch.Tensor:
        """Map actions ``[N, act_dim]`` (f64, in ``[-1, 1]``) to flat actuator
        commands ``[E, U_DIM]`` (f64, physical units).

        Actions must already be in range (e.g. tanh-squashed by the policy):
        clamping here would zero the gradient in saturation.
        """
        ...

    def observation(self, x: torch.Tensor) -> torch.Tensor:
        """Pack the observation ``[N, obs_dim]`` (f64) from the state ``[E, X_DIM]``."""
        ...

    def reward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Per-environment reward ``[N]`` (f64) from the post-step state and
        the action that produced it."""
        ...

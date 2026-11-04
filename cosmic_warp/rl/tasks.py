"""
RL task interface.

A *task* supplies the three environment-specific pieces of an RL problem:
the action mapping (normalized policy output -> actuator commands), the
observation packing, and the reward. Termination and truncation are handled
generically by :class:`cosmic_warp.rl.env.VecEnv` from the simulator's own
termination checker and episode-step counters.

The package defines only the :class:`Task` protocol; tasks themselves are
application code. ``examples/attitude_task.py`` is a reference implementation
(reaction-wheel attitude control) with all packing and reward math in fused
Warp kernels on device.
"""
from typing import Protocol, runtime_checkable

import warp as wp


@runtime_checkable
class Task(Protocol):
    """Minimal task interface consumed by :class:`~cosmic_warp.rl.env.VecEnv`.

    A task exposes its per-environment action/observation widths and three
    device-side operations. ``VecEnv`` calls ``bind`` once at construction,
    then ``ingest_actions`` / ``write_reward`` / ``write_obs`` every step.
    """

    #: Per-environment action dimension (set during :meth:`bind`).
    act_dim: int
    #: Per-environment observation dimension (set during :meth:`bind`).
    obs_dim: int

    def bind(self, env) -> None:
        """Cache the simulator handles and compute ``act_dim`` / ``obs_dim``."""
        ...

    def ingest_actions(self, action: wp.array) -> None:
        """Map normalized actions ``[N, act_dim]`` (f32) into actuator commands."""
        ...

    def write_obs(self, obs: wp.array, reset_env: wp.array, masked: int) -> None:
        """Pack the observation into ``obs`` ``[N, obs_dim]`` (f32).

        When ``masked == 1`` only rows where ``reset_env`` is True are written
        (the post-reset rewrite for auto-reset environments); when ``masked ==
        0`` every row is written.
        """
        ...

    def write_reward(self, reward: wp.array, action: wp.array) -> None:
        """Write the per-environment reward into ``reward`` ``[N]`` (f32).

        ``action`` is the (already ingested) normalized action of the current
        step, available for control-effort penalties.
        """
        ...

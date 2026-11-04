"""
Vectorized RL environment.

:class:`VecEnv` wraps a :class:`~cosmic_warp.core.pipeline.SpacecraftSimulator`
and adds a fused, fully on-device RL step:

    action -> ingest -> physics -> reward -> done -> observation -> auto-reset

Everything stays on the GPU. The whole step is captured once into a CUDA graph
and replayed, so per-step launch overhead is eliminated and there is no
host synchronization inside :meth:`step`. Boundary tensors (action, obs,
reward) are float32; the physics stays float64 internally.

The step returns ``(obs, reward, terminated, truncated)`` as Warp arrays.
``terminated`` and ``truncated`` are kept separate (never collapsed into a
single ``done``) so trainers can bootstrap truncated episodes correctly;
:attr:`terminal_obs` holds the pre-reset observation for that.

Trainer interop is deliberately out of scope here: the arrays are exposed
natively and via zero-copy ``*_torch()`` views (dlpack, same GPU memory),
and trainer-specific adapters live separately so this core stays neutral.
"""
from typing import Optional, Tuple

import warp as wp

from cosmic_warp.core import validation
from cosmic_warp.core.pipeline import SpacecraftSimulator
from cosmic_warp.core.termination import TerminationStatus
from cosmic_warp.rl.tasks import Task

# The step kernel compares episode_steps (int32) with the limit
_MAX_EPISODE_STEPS = 2**31 - 1


def _buffer(name: str) -> property:
    """A read-only boundary buffer: the captured graph and the torch views
    hold the array itself, so write into it, never rebind it."""
    return property(lambda self: getattr(self, "_" + name))


class VecEnv:
    """Vectorized, GPU-resident RL environment over a spacecraft simulator.

    Args:
        sim: A constructed simulator (e.g. from ``create_simulator_from_yaml``).
            ``sim.num_envs`` environments are stepped in parallel.
        task: The task defining action mapping, observation and reward.
        max_episode_steps: Episode length after which an environment is
            truncated (time limit) and auto-reset, in [1, 2**31 - 1].
    """

    # Read-only buffers (documented by the string after each)
    actions = _buffer("actions")
    """Normalized actions [N, act_dim] (f32), read by step()."""
    obs = _buffer("obs")
    """Observations [N, obs_dim] (f32)."""
    terminal_obs = _buffer("terminal_obs")
    """Pre-reset observations [N, obs_dim] (f32) of the last step."""
    reward = _buffer("reward")
    """Rewards [N] (f32) of the last step."""
    terminated = _buffer("terminated")
    """Terminated flags [N] (bool) of the last step."""
    truncated = _buffer("truncated")
    """Truncated (time-limit) flags [N] (bool) of the last step."""

    def __init__(self, sim: SpacecraftSimulator, task: Task, max_episode_steps: int):
        self.sim = sim
        self.task = task
        self._graph = None
        self.max_episode_steps = max_episode_steps

        self.num_envs = sim.num_envs
        self.num_bodies = sim.num_bodies
        self.num_entities = sim.num_entities
        self.device = sim.device

        task.bind(self)
        self.obs_dim = task.obs_dim
        self.act_dim = task.act_dim

        d = self.device
        N, E = self.num_envs, self.num_entities

        # Persistent boundary buffers (allocated once, written in place).
        self._actions = wp.zeros((N, self.act_dim), dtype=wp.float32, device=d)
        self._obs = wp.zeros((N, self.obs_dim), dtype=wp.float32, device=d)
        self._terminal_obs = wp.zeros((N, self.obs_dim), dtype=wp.float32, device=d)
        self._reward = wp.zeros(N, dtype=wp.float32, device=d)
        self._terminated = wp.zeros(N, dtype=wp.bool, device=d)
        self._truncated = wp.zeros(N, dtype=wp.bool, device=d)

        # Internal masks: per-env done (drives obs rewrite / terminal copy) and
        # per-entity done (drives the masked resets).
        self.reset_env = wp.zeros(N, dtype=wp.bool, device=d)
        self.done_mask = wp.zeros(E, dtype=wp.bool, device=d)

    @property
    def max_episode_steps(self) -> int:
        """Episode length after which an environment is truncated and
        auto-reset. Setting it re-captures the step graph on the next step."""
        return self._max_episode_steps

    @max_episode_steps.setter
    def max_episode_steps(self, value: int) -> None:
        self._max_episode_steps = validation.positive_int(
            "max_episode_steps", value, maximum=_MAX_EPISODE_STEPS)
        self.invalidate_graph()

    @property
    def _use_graph(self) -> bool:
        return "cuda" in str(self.device)

    # ------------------------------------------------------------------ step

    def reset(self) -> wp.array:
        """Reset every environment and return the initial observation ``[N, O]``."""
        self.sim.reset()
        self.terminated.zero_()
        self.truncated.zero_()
        self.reset_env.zero_()
        self.task.write_obs(self.obs, self.reset_env, 0)
        return self.obs

    def step(
        self, action: Optional[wp.array] = None
    ) -> Tuple[wp.array, wp.array, wp.array, wp.array]:
        """Advance every environment by one control step.

        Args:
            action: Normalized actions ``[N, act_dim]`` (f32). If given, copied
                into the persistent action buffer; if ``None``, whatever is
                already in :attr:`actions` (or its torch view) is used.

        Raises:
            ValueError: if ``action`` is not a Warp array of the action
                buffer's shape, dtype and device (checked on metadata only,
                no device sync).

        Returns:
            ``(obs, reward, terminated, truncated)`` as Warp arrays. All are
            reused buffers — copy into trainer storage each step, do not retain
            a view across steps.
        """
        # A set_config(inertia=...) since the last step becomes the nominal the
        # auto-reset's inertia_scale applies to. Host-side version check,
        # outside the graph; uploads only when the inertia changed.
        self.sim.randomizer.sync_nominal()

        if action is not None:
            # wp.copy of a smaller or differently typed array would silently
            # fill the buffer partially
            if (not isinstance(action, wp.array) or action.shape != self.actions.shape
                    or action.dtype != wp.float32 or action.device != self.actions.device):
                raise ValueError(
                    f"action must be a float32 Warp array of shape {self.actions.shape} on "
                    f"{self.actions.device}, got {validation.describe(action)}"
                )
            wp.copy(self.actions, action)

        if self._use_graph:
            if self._graph is None:
                with wp.ScopedCapture(device=self.device) as capture:
                    self._step_body()
                self._graph = capture.graph
            wp.capture_launch(self._graph)
        else:
            self._step_body()

        return self.obs, self.reward, self.terminated, self.truncated

    def invalidate_graph(self) -> None:
        """Drop the captured CUDA graph so the next :meth:`step` re-captures it.

        The capture bakes in every scalar passed by value to a kernel launch
        (task parameters such as reward weights) and the
        ``randomizer.enabled`` branch. Call this after changing any of them
        mid-run; otherwise the change silently has no effect on CUDA. Setting
        :attr:`max_episode_steps` calls it for you.
        """
        self._graph = None

    def _step_body(self) -> None:
        """The captured per-step kernel sequence (see module docstring)."""
        # 1. Normalized action -> actuator commands.
        self.task.ingest_actions(self.actions)

        # 2. Physics: integrate, increment episode_steps, flag terminations.
        self.sim.integrator.step(self.sim.ctx, self.sim.dt)

        # 3. Reward from the post-step, pre-reset state.
        self.task.write_reward(self.reward, self.actions)

        # 4. Terminated / truncated / done masks.
        wp.launch(
            kernel=_done_kernel,
            dim=self.num_entities,
            inputs=[
                self.sim.termination_checker.status,
                self.sim.episode_steps,
                self.num_bodies,
                self.max_episode_steps,
                self.terminated,
                self.truncated,
                self.reset_env,
                self.done_mask,
            ],
            device=self.device,
        )

        # 5. Observation of the post-step, pre-reset state (all envs).
        self.task.write_obs(self.obs, self.reset_env, 0)

        # 6. Snapshot terminal observation (only read where done downstream).
        wp.copy(self.terminal_obs, self.obs)

        # 7. Auto-reset done environments (state, actuators, status, counters).
        self._reset_done()

        # 8. Rewrite observation for the reset environments only.
        self.task.write_obs(self.obs, self.reset_env, 1)

    def _reset_done(self) -> None:
        """Reset every entity flagged in ``done_mask`` (terminated or truncated).

        Mirrors the simulator's ``reset_terminated`` but keys off the combined
        done mask so time-limit truncations are reset too. ``reset_masked``
        zeroes ``episode_steps`` for the reset entities.
        """
        sim = self.sim
        sim.reset_manager.reset_masked(sim.states, self.done_mask)
        if sim.randomizer.enabled:
            sim.randomizer.randomize_masked(sim.states, self.done_mask)
        sim.actuators.reset_masked(self.done_mask)
        sim.termination_checker.reset_status_masked(self.done_mask)

    # -------------------------------------------------------- torch interop

    def observations_torch(self):
        """Zero-copy torch view of :attr:`obs` (requires torch)."""
        return self._torch(self.obs)

    def actions_torch(self):
        """Zero-copy torch view of :attr:`actions`; write policy output here."""
        return self._torch(self.actions)

    def rewards_torch(self):
        """Zero-copy torch view of :attr:`reward`."""
        return self._torch(self.reward)

    def terminated_torch(self):
        """Zero-copy torch view of :attr:`terminated`."""
        return self._torch(self.terminated)

    def truncated_torch(self):
        """Zero-copy torch view of :attr:`truncated`."""
        return self._torch(self.truncated)

    def terminal_observations_torch(self):
        """Zero-copy torch view of :attr:`terminal_obs`."""
        return self._torch(self.terminal_obs)

    @staticmethod
    def _torch(arr: wp.array):
        try:
            return wp.to_torch(arr)
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
            raise ImportError(
                "PyTorch is required for the zero-copy torch views "
                "(pip install 'cosmic-warp[torch]')."
            ) from exc


@wp.kernel
def _done_kernel(
    status: wp.array(dtype=TerminationStatus),
    episode_steps: wp.array(dtype=wp.int32),
    num_bodies: int,
    max_steps: int,
    terminated: wp.array(dtype=wp.bool),   # [N]
    truncated: wp.array(dtype=wp.bool),    # [N]
    reset_env: wp.array(dtype=wp.bool),    # [N]
    done_mask: wp.array(dtype=wp.bool),    # [E]
):
    """Reduce per-entity termination/step counters to per-env done signals.

    Termination is env-level in the simulator (all bodies of an env share one
    status), so the first body of each env carries the per-env flags. When a
    step both terminates and hits the time limit, termination wins and
    ``truncated`` stays False so value bootstrapping treats it as a true
    terminal.
    """
    e = wp.tid()
    env = e // num_bodies

    term = status[e].terminated
    trunc = episode_steps[e] >= max_steps
    done_mask[e] = term or trunc

    if e - env * num_bodies == 0:
        terminated[env] = term
        truncated[env] = trunc and (not term)
        reset_env[env] = term or trunc

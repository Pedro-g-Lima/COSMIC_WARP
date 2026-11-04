"""
Differentiable vectorized environment: short-horizon windows over the bridge.

:class:`DiffVecEnv` chains :class:`~cosmic_warp.rl.diff_bridge.DiffSimStepper`
steps into H-step **windows** — the rollout unit of short-horizon
first-order trainers such as SHAC (example 17). Within a window the whole
chain (observation -> policy -> command -> dynamics -> reward) stays on the
torch autograd graph; between windows the carried state is detached
(backprop-through-time is truncated at window boundaries by design).

Episodes are truncation-only: termination checks are off on the
differentiable path (a terminated entity's gradients silently vanish), so
``max_episode_steps`` must be a multiple of ``horizon`` and every environment
resets together — a plain full ``sim.reset()`` between windows, outside
autograd. Value targets (e.g. TD(lambda)) therefore always bootstrap: a time
limit is not an MDP terminal.
"""
from dataclasses import dataclass, field
from typing import Callable, List

import torch

from cosmic_warp.core import validation
from cosmic_warp.core.pipeline import SpacecraftSimulator
from cosmic_warp.rl.diff_bridge import DiffSimStepper
from cosmic_warp.rl.diff_tasks import DiffTask


@dataclass
class Window:
    """One H-step differentiable rollout.

    ``obs[t]`` is the observation the policy acted on at step t (``obs[H]`` is
    the terminal observation for value bootstrapping); ``rewards[t]`` is the
    reward of that transition. All tensors are float64 and **on the autograd
    graph** — detach before storing for the critic.
    """

    obs: List[torch.Tensor] = field(default_factory=list)      # H+1 x [N, obs_dim]
    rewards: List[torch.Tensor] = field(default_factory=list)  # H   x [N]


class DiffVecEnv:
    """Windowed differentiable environment over a spacecraft simulator.

    Args:
        sim: Simulator built with ``differentiable=True`` and termination
            checks disabled (validated by the underlying stepper).
        task: The differentiable task (torch obs/reward/action mapping).
        horizon: Window length H (steps per rollout / actor update).
        max_episode_steps: Episode time limit; must be a multiple of
            ``horizon`` so episodes end exactly on a window boundary.
    """

    def __init__(self, sim: SpacecraftSimulator, task: DiffTask,
                 horizon: int, max_episode_steps: int):
        horizon = validation.positive_int("horizon", horizon)
        max_episode_steps = validation.positive_int("max_episode_steps", max_episode_steps)
        if max_episode_steps % horizon != 0:
            raise ValueError(
                f"max_episode_steps ({max_episode_steps}) must be a multiple "
                f"of horizon ({horizon}) so episodes end on window boundaries"
            )
        self.sim = sim
        self.task = task
        self.horizon = horizon
        self.max_episode_steps = max_episode_steps

        self.stepper = DiffSimStepper(sim)
        task.bind(sim)
        self.num_envs = sim.num_envs
        self.obs_dim = task.obs_dim
        self.act_dim = task.act_dim
        self.device = str(sim.device)

        self._x = None
        self._x_final = None
        self._steps_since_reset = 0

    @property
    def state(self) -> torch.Tensor:
        """The carried state ``[num_entities, X_DIM]`` (f64, detached) the next window
        starts from; ``None`` before :meth:`reset`."""
        return self._x

    def reset(self) -> torch.Tensor:
        """Full reset of every environment; returns the initial observation
        ``[N, obs_dim]`` (f64, detached)."""
        self.sim.reset()
        self._x = self.stepper.read_state()
        self._x_final = None   # a pending window no longer continues from here
        self._steps_since_reset = 0
        with torch.no_grad():
            return self.task.observation(self._x)

    def rollout(self, policy_fn: Callable[[torch.Tensor], torch.Tensor]) -> Window:
        """Run one H-step window from the carried state, on the autograd graph.

        Args:
            policy_fn: Maps the observation ``[N, obs_dim]`` (f64) to an
                action ``[N, act_dim]`` (f64, in ``[-1, 1]``); anything
                torch-differentiable (the policy gradient flows through it).

        Call :meth:`end_window` after consuming the window to advance the
        carried state (detached) and handle the episode time limit. Calling
        ``rollout`` again first re-runs the window from the same carried state.
        The window's backward uses the parameters of its forward, so it may
        run before or after ``end_window()``.
        """
        if self._x is None:
            self.reset()
        window = Window()
        x = self._x
        # One parameter snapshot for the window's steps: their backward does
        # not see a later end_window() reset or set_config
        with self.stepper.frozen_params():
            for _ in range(self.horizon):
                obs = self.task.observation(x)
                window.obs.append(obs)
                action = policy_fn(obs)
                x = self.stepper.step(x, self.task.commands(action))
                window.rewards.append(self.task.reward(x, action))
        window.obs.append(self.task.observation(x))
        self._x_final = x
        return window

    def end_window(self) -> bool:
        """Detach and carry the final state; reset at the episode time limit.

        Returns True when the windows just closed an episode (all envs were
        reset). The carried state is also written back to the simulator so
        its own getters reflect reality between windows.

        Raises:
            RuntimeError: No window is pending: no :meth:`rollout` since the
                last ``end_window`` or :meth:`reset`.
        """
        if self._x_final is None:
            raise RuntimeError(
                "end_window() needs a rollout() first (none since the last "
                "end_window() or reset())"
            )
        self._x = self._x_final.detach()
        self._x_final = None
        self.stepper.write_state(self._x)
        self._steps_since_reset += self.horizon
        if self._steps_since_reset >= self.max_episode_steps:
            self.reset()
            return True
        return False

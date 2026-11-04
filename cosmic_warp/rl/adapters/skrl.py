"""
skrl adapter for :class:`cosmic_warp.rl.VecEnv`.

Wraps the neutral environment as a ``skrl`` torch environment wrapper so it can
be trained with any ``skrl`` agent (PPO, SAC, …). The mapping is near
pass-through: ``skrl`` already uses the split ``terminated`` / ``truncated``
convention and torch tensors, so the adapter only reshapes rewards/flags to the
``[N, 1]`` layout ``skrl`` expects and clones the observation (``skrl`` retains
the reference across the next step; our buffer is single-buffered). The
terminal observations go in ``infos["terminal_observations"]``.

``build_ppo_agent`` is a convenience that assembles a Gaussian-policy PPO agent
with sensible defaults; use it or build your own models/agent directly. It
returns a :class:`TerminalObsPPO`, which bootstraps truncated transitions from
the terminal observation: skrl's own PPO would evaluate the value of the
returned observation, which for a truncated environment is already the first
observation of its next episode.

Requires the ``skrl`` optional dependency (``pip install "cosmic-warp[skrl]"``),
which pulls in PyTorch. Tested against skrl 2.1.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from skrl.agents.torch.ppo import PPO
from skrl.envs.wrappers.torch.base import Wrapper
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

from cosmic_warp.rl.env import VecEnv as _CwVecEnv

#: The ``infos`` key holding the terminal observations ``[N, obs_dim]``.
TERMINAL_OBS_KEY = "terminal_observations"


class SkrlVecEnv(Wrapper):
    """Present a :class:`cosmic_warp.rl.VecEnv` through the skrl wrapper interface.

    ``step`` returns ``infos[TERMINAL_OBS_KEY]``, the observation of the state
    each environment reached (pre-reset; meaningful only where the step
    ended the episode), for :class:`TerminalObsPPO`.

    Args:
        env: The COSMIC WARP environment to wrap. ``num_bodies`` must be 1.
    """

    def __init__(self, env: _CwVecEnv):
        if env.num_bodies != 1:
            raise NotImplementedError(
                "the skrl adapter currently supports num_bodies == 1"
            )
        super().__init__(env)
        self._cw = env
        self._observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(env.obs_dim,), dtype=np.float32
        )
        self._action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(env.act_dim,), dtype=np.float32
        )

    @property
    def vec_env(self) -> _CwVecEnv:
        """The wrapped COSMIC WARP :class:`~cosmic_warp.rl.VecEnv` (for
        evaluation loops outside the trainer)."""
        return self._cw

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    @property
    def state_space(self):
        return None

    @property
    def num_agents(self) -> int:
        return 1

    def reset(self):
        self._cw.reset()
        return self._cw.observations_torch().clone(), {}

    def step(self, actions: torch.Tensor):
        self._cw.actions_torch().copy_(actions)
        self._cw.step()
        obs = self._cw.observations_torch().clone()
        reward = self._cw.rewards_torch().clone().view(-1, 1)
        terminated = self._cw.terminated_torch().clone().view(-1, 1)
        truncated = self._cw.truncated_torch().clone().view(-1, 1)
        # Valid only where terminated or truncated (see VecEnv.terminal_obs)
        infos = {TERMINAL_OBS_KEY: self._cw.terminal_observations_torch().clone()}
        return obs, reward, terminated, truncated, infos

    def state(self):
        return None

    def render(self, *args, **kwargs):
        return None

    def close(self):
        pass


class TerminalObsPPO(PPO):
    """skrl PPO that bootstraps truncated transitions from the terminal
    observation.

    skrl's PPO, with ``time_limit_bootstrap``, adds ``gamma * V(next_obs)`` to
    the reward of a truncated transition. Under auto-reset ``next_obs`` is
    the first observation of the next episode, so this agent substitutes
    ``infos[TERMINAL_OBS_KEY]`` (from :class:`SkrlVecEnv`) on the truncated
    rows before skrl evaluates it. Keep ``time_limit_bootstrap`` on: GAE also
    uses it to stop at the truncation. Without the key in ``infos`` it
    behaves as skrl's PPO.
    """

    def record_transition(self, *, next_observations, truncated, infos, **kwargs):
        terminal = infos.get(TERMINAL_OBS_KEY) if isinstance(infos, dict) else None
        if terminal is not None:
            # skrl also keeps this as the rollout's last observation, for the
            # bootstrap at the update; its truncated rows are not read there,
            # since GAE stops at a truncation under time_limit_bootstrap.
            next_observations = torch.where(truncated, terminal, next_observations)
        super().record_transition(next_observations=next_observations, truncated=truncated,
                                  infos=infos, **kwargs)


def _mlp(in_dim: int, out_dim: int, hidden, activation=nn.ELU) -> nn.Sequential:
    layers, prev = [], in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), activation()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def _obs_of(inputs: dict) -> torch.Tensor:
    obs = inputs.get("observations")
    return inputs["states"] if obs is None else obs


class _GaussianPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, hidden):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=False)
        self.net = _mlp(self.num_observations, self.num_actions, hidden)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role=""):
        return self.net(_obs_of(inputs)), {"log_std": self.log_std_parameter}


class _Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, hidden):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = _mlp(self.num_observations, 1, hidden)

    def compute(self, inputs, role=""):
        return self.net(_obs_of(inputs)), {}


def build_ppo_agent(
    env: SkrlVecEnv,
    rollouts: int = 24,
    hidden=(128, 128),
    learning_rate: float = 1.0e-3,
    entropy_loss_scale: float = 0.005,
):
    """Assemble a Gaussian-policy PPO agent for continuous control.

    A convenience default, not a tuned recipe — build your own models/agent for
    anything beyond a quick run (with :class:`TerminalObsPPO` to keep the
    terminal-observation bootstrap). Returns a :class:`TerminalObsPPO` with
    ``time_limit_bootstrap`` on.
    """
    from skrl.agents.torch.ppo import PPO_CFG
    from skrl.memories.torch import RandomMemory

    device = env.device
    obs_space, act_space = env.observation_space, env.action_space

    models = {
        "policy": _GaussianPolicy(obs_space, act_space, device, hidden),
        "value": _Value(obs_space, act_space, device, hidden),
    }
    memory = RandomMemory(memory_size=rollouts, num_envs=env.num_envs, device=device)

    cfg = PPO_CFG()
    cfg.rollouts = rollouts
    cfg.learning_rate = learning_rate
    cfg.entropy_loss_scale = entropy_loss_scale
    # Our truncated flag marks time-limit episodes; TerminalObsPPO bootstraps
    # them from the terminal observation
    cfg.time_limit_bootstrap = True
    # Quiet by default: no tensorboard/checkpoint artifacts. Build your own
    # agent (or reset these on the returned agent's cfg) to enable logging.
    cfg.experiment.write_interval = 0
    cfg.experiment.checkpoint_interval = 0

    return TerminalObsPPO(
        models=models,
        memory=memory,
        observation_space=obs_space,
        action_space=act_space,
        device=device,
        cfg=cfg,
    )

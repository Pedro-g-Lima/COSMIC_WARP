"""
rsl_rl adapter for :class:`cosmic_warp.rl.VecEnv`.

Wraps the neutral environment as an ``rsl_rl.env.VecEnv`` so it can be trained
with ``rsl_rl``'s ``OnPolicyRunner`` (PPO). The translation is thin: it maps
observations into the ``TensorDict`` group format, collapses the split
``terminated``/``truncated`` into ``rsl_rl``'s single ``dones`` plus
``extras["time_outs"]``, and copies actions in through the zero-copy action
view.

Requires the ``rsl-rl`` optional dependency (``pip install "cosmic-warp[rsl-rl]"``),
which pulls in PyTorch. Tested against rsl-rl-lib 5.x.
"""
from __future__ import annotations

import torch
import warp as wp
from tensordict import TensorDict
from rsl_rl.env import VecEnv as _RslRlVecEnv

from cosmic_warp.rl.env import VecEnv as _CwVecEnv


class RslRlVecEnv(_RslRlVecEnv):
    """Present a :class:`cosmic_warp.rl.VecEnv` through the rsl_rl interface.

    Args:
        env: The COSMIC WARP environment to wrap. ``num_bodies`` must be 1
            (per-env == per-entity); multi-body reduction is not yet defined
            for this adapter.
    """

    def __init__(self, env: _CwVecEnv):
        if env.num_bodies != 1:
            raise NotImplementedError(
                "the rsl_rl adapter currently supports num_bodies == 1"
            )
        self._env = env

        # rsl_rl.env.VecEnv interface attributes.
        self.num_envs = env.num_envs
        self.num_actions = env.act_dim
        self.max_episode_length = env.max_episode_steps
        self.device = str(env.device)
        self.cfg = {}
        # Per-env current episode length (int32); rsl_rl reads/writes this only
        # for init_at_random_ep_len. With num_bodies == 1 the simulator's
        # per-entity counter is already per-env.
        self._episode_length_buf = wp.to_torch(env.sim.episode_steps)

        env.reset()

    @property
    def vec_env(self) -> _CwVecEnv:
        """The wrapped COSMIC WARP :class:`~cosmic_warp.rl.VecEnv` (for
        evaluation loops outside the trainer)."""
        return self._env

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self._episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        # rsl_rl assigns a fresh tensor here (init_at_random_ep_len); copy it
        # into the zero-copy view so the write reaches sim.episode_steps
        # instead of rebinding the attribute.
        self._episode_length_buf.copy_(value)

    def _obs(self) -> TensorDict:
        # Clone: rsl_rl holds the observation reference across the next
        # env.step() (act -> step -> process_env_step), and our buffer is
        # single-buffered and overwritten in place. The clone is a small
        # [N, obs_dim] f32 copy that keeps the stored transition correct.
        obs = self._env.observations_torch().clone()
        return TensorDict({"policy": obs}, batch_size=[self.num_envs], device=self.device)

    def get_observations(self) -> TensorDict:
        return self._obs()

    def step(self, actions: torch.Tensor):
        # Write the policy output into the env's action buffer (zero-copy view)
        # and advance one on-device step.
        self._env.actions_torch().copy_(actions)
        self._env.step()

        rewards = self._env.rewards_torch()
        terminated = self._env.terminated_torch()
        truncated = self._env.truncated_torch()
        dones = (terminated | truncated).to(torch.long)
        extras = {"time_outs": truncated}
        return self._obs(), rewards, dones, extras

    def reset(self):
        self._env.reset()
        return self._obs(), {}


def default_ppo_cfg(
    num_steps_per_env: int = 24,
    hidden_dims=(128, 128),
    learning_rate: float = 1.0e-3,
    entropy_coef: float = 0.005,
) -> dict:
    """A minimal, working ``OnPolicyRunner`` training config for continuous
    control. Tweak or replace freely — this is a convenience default, not a
    tuned recipe.
    """
    return {
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": learning_rate,
            "gamma": 0.99,
            "lam": 0.95,
            "entropy_coef": entropy_coef,
            "clip_param": 0.2,
            "value_loss_coef": 1.0,
            "max_grad_norm": 1.0,
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": list(hidden_dims),
            "activation": "elu",
            # Stochastic Gaussian policy (learnable std) for continuous control.
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0},
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": list(hidden_dims),
            "activation": "elu",
        },
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "num_steps_per_env": num_steps_per_env,
        "empirical_normalization": False,
    }

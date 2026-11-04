"""
rl_games adapter for :class:`cosmic_warp.rl.VecEnv`.

Wraps the neutral environment as an ``rl_games`` ``IVecEnv`` and wires it into
rl_games' environment registry so it can be trained with rl_games' PPO
(``a2c_continuous``). Observations are returned in rl_games' ``{"obs": tensor}``
form, ``terminated``/``truncated`` are collapsed into a single ``dones`` with
``truncated`` surfaced as ``infos["time_outs"]`` (enable ``value_bootstrap`` in
the config to use it).

``register_env`` registers a built environment under a name; ``default_ppo_params``
returns a minimal working training config. Requires the ``rl-games`` optional
dependency (``pip install "cosmic-warp[rl-games]"``), which pulls in PyTorch.
Tested against rl-games 1.6.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.ivecenv import IVecEnv

from cosmic_warp.rl.env import VecEnv as _CwVecEnv


class RlGamesVecEnv(IVecEnv):
    """Present a :class:`cosmic_warp.rl.VecEnv` through the rl_games IVecEnv API.

    Args:
        env: The COSMIC WARP environment to wrap. ``num_bodies`` must be 1.
    """

    def __init__(self, env: _CwVecEnv):
        if env.num_bodies != 1:
            raise NotImplementedError(
                "the rl_games adapter currently supports num_bodies == 1"
            )
        self._cw = env
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(env.obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(env.act_dim,), dtype=np.float32
        )

    @property
    def vec_env(self) -> _CwVecEnv:
        """The wrapped COSMIC WARP :class:`~cosmic_warp.rl.VecEnv` (for
        evaluation loops outside the trainer)."""
        return self._cw

    def get_number_of_agents(self) -> int:
        return 1

    def get_env_info(self) -> dict:
        return {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "agents": 1,
        }

    def reset(self):
        self._cw.reset()
        return {"obs": self._cw.observations_torch().clone()}

    def step(self, actions: torch.Tensor):
        self._cw.actions_torch().copy_(actions)
        self._cw.step()
        obs = {"obs": self._cw.observations_torch().clone()}
        rewards = self._cw.rewards_torch().clone()
        dones = (self._cw.terminated_torch() | self._cw.truncated_torch()).clone()
        infos = {"time_outs": self._cw.truncated_torch().clone()}
        return obs, rewards, dones, infos


def register_env(name: str, env: RlGamesVecEnv) -> None:
    """Register a built environment with rl_games under ``name``.

    After this, an rl_games config with ``config.env_name == name`` resolves to
    ``env``, for training and for the player (``Runner.create_player()``,
    which builds its env through ``env_creator`` by default). Registration is
    idempotent per name (re-registering overwrites).
    """
    vec_type = f"cw_vec_{name}"
    vecenv.register(vec_type, lambda config_name, num_actors, **kwargs: env)
    env_configurations.register(name, {"vecenv_type": vec_type, "env_creator": lambda **kwargs: env})


def default_ppo_params(
    env_name: str,
    num_envs: int,
    horizon_length: int = 24,
    num_minibatches: int = 4,
    units=(128, 128),
    learning_rate: float = 1.0e-3,
    max_epochs: int = 100,
    device: str = "cuda:0",
    train_dir: str | None = None,
) -> dict:
    """A minimal, working rl_games PPO config for continuous control.

    Returns the ``{"params": {...}}`` dict expected by ``Runner.load``. A
    convenience default, not a tuned recipe. ``device`` is used for training
    and for the player (``Runner.create_player()``). ``train_dir`` redirects
    rl_games' log/checkpoint output (default: rl_games' ``runs/`` in the
    working dir).
    """
    minibatch_size = (num_envs * horizon_length) // num_minibatches
    params = {
        "params": {
            "seed": 0,
            "algo": {"name": "a2c_continuous"},
            "model": {"name": "continuous_a2c_logstd"},
            "network": {
                "name": "actor_critic",
                "separate": False,
                "space": {
                    "continuous": {
                        "mu_activation": "None",
                        "sigma_activation": "None",
                        "mu_init": {"name": "default"},
                        "sigma_init": {"name": "const_initializer", "val": 0.0},
                        "fixed_sigma": True,
                    }
                },
                "mlp": {
                    "units": list(units),
                    "activation": "elu",
                    "initializer": {"name": "default"},
                },
            },
            "config": {
                "name": env_name,
                "env_name": env_name,
                "device": device,
                "device_name": device,   # the player's device key
                "num_actors": num_envs,
                "ppo": True,
                "reward_shaper": {"scale_value": 1.0},
                "normalize_advantage": True,
                "normalize_input": False,
                "normalize_value": False,
                "value_bootstrap": True,
                "gamma": 0.99,
                "tau": 0.95,
                "learning_rate": learning_rate,
                "lr_schedule": "adaptive",
                "kl_threshold": 0.01,
                "grad_norm": 1.0,
                "entropy_coef": 0.005,
                "truncate_grads": True,
                "e_clip": 0.2,
                "clip_value": True,
                "horizon_length": horizon_length,
                "minibatch_size": minibatch_size,
                "mini_epochs": 5,
                "critic_coef": 2.0,
                "bounds_loss_coef": 0.0001,
                "max_epochs": max_epochs,
            },
        }
    }
    if train_dir is not None:
        params["params"]["config"]["train_dir"] = train_dir
    return params

"""
Trainer adapters for :class:`cosmic_warp.rl.VecEnv`.

Each adapter is a thin, optional translation from the environment's neutral
step contract to one RL trainer's expected interface. The core environment
depends on none of them; importing this package pulls in no trainer.

Import an adapter directly, which imports its trainer at that point:

    from cosmic_warp.rl.adapters.rsl_rl import RslRlVecEnv     # needs [rsl-rl]
    from cosmic_warp.rl.adapters.skrl import SkrlVecEnv        # needs [skrl]
    from cosmic_warp.rl.adapters.rl_games import RlGamesVecEnv # needs [rl-games]

Available adapters:

- ``rsl_rl`` — ETH RSL's ``rsl-rl-lib`` (PPO). Install with
  ``pip install "cosmic-warp[rsl-rl]"``.
- ``skrl`` — ``skrl`` (PPO/SAC/…). Install with ``pip install "cosmic-warp[skrl]"``.
- ``rl_games`` — ``rl-games`` (PPO). Install with ``pip install "cosmic-warp[rl-games]"``.
"""

__all__ = []

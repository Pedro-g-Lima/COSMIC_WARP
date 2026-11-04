"""
COSMIC WARP reinforcement-learning layer.

A thin, GPU-resident, framework-neutral environment over the spacecraft
simulator. The core (:class:`VecEnv` + the :class:`Task` protocol) has no
learning-framework dependency; PyTorch is used only for optional zero-copy
tensor views. Tasks are application code: ``examples/attitude_task.py`` is a
reference implementation.

    from cosmic_warp.core.config import create_simulator_from_yaml
    from cosmic_warp.rl import VecEnv
    from attitude_task import AttitudeTask   # examples/attitude_task.py

    sim = create_simulator_from_yaml("examples/rl_attitude_training.yaml")
    env = VecEnv(sim, AttitudeTask(), max_episode_steps=500)
    obs = env.reset()
    for _ in range(horizon):
        obs, reward, terminated, truncated = env.step(action)
"""
from cosmic_warp.rl.env import VecEnv
from cosmic_warp.rl.tasks import Task

__all__ = ["VecEnv", "Task"]

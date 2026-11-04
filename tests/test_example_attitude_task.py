"""
Tests for the example reference task (examples/attitude_task.py), used by
examples 13-16: the reward formula, and agreement with the test-local
DetumbleTask the VecEnv and adapter tests run on.
"""
import sys
from pathlib import Path

import numpy as np
import warp as wp

from cosmic_warp.core.pipeline import SpacecraftSimulator
from cosmic_warp.rl import VecEnv
from rl_tasks import DetumbleTask

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from attitude_task import AttitudeTask  # noqa: E402


def _make_sim(device, seed=42):
    sim = SpacecraftSimulator(
        num_envs=16, integrator="symplectic", dt=0.1, device=device,
        gravity=True, j2_perturbation=False, gravity_gradient=False,
        check_bounds=False, reset_config={"seed": seed},
    )
    sim.rw_manager.set_config(
        max_torque=0.02, max_momentum=0.5, wheel_inertia=0.005, viscous_damping=1.0e-5
    )
    return sim


def test_reward_non_positive(device):
    """The attitude reward is a penalty: always <= 0."""
    env = VecEnv(_make_sim(device), AttitudeTask(), max_episode_steps=10**9)
    env.reset()
    _, reward, _, _ = env.step(wp.zeros((env.num_envs, env.act_dim), dtype=wp.float32,
                                        device=device))
    assert (reward.numpy() <= 1e-6).all()


def test_reward_matches_formula(device):
    """Reward equals -(w_att*angle^2 + w_rate*|omega|^2 + w_ctrl*|a|^2)."""
    task = AttitudeTask(w_att=2.0, w_rate=0.5, w_ctrl=0.1)
    env = VecEnv(_make_sim(device), task, max_episode_steps=10**9)
    env.reset()
    a = 0.25
    act = wp.full((env.num_envs, env.act_dim), a, dtype=wp.float32, device=device)
    _, reward, _, _ = env.step(act)  # no terminations: obs is the reward state

    obs = env.obs.numpy().astype(np.float64)
    quat = obs[:, 0:4] / np.linalg.norm(obs[:, 0:4], axis=1, keepdims=True)
    angle = 2.0 * np.arccos(np.minimum(np.abs(quat[:, 3]), 1.0))
    rate_sq = (obs[:, 4:7] ** 2).sum(axis=1)
    expected = -(2.0 * angle**2 + 0.5 * rate_sq + 0.1 * env.act_dim * a**2)
    np.testing.assert_allclose(reward.numpy(), expected, rtol=1e-3, atol=1e-4)


def test_matches_test_task_without_attitude_term(device):
    """With w_att = 0, AttitudeTask and the tests' DetumbleTask give the same
    observations and rewards (and actions spin the wheels identically)."""
    example = VecEnv(_make_sim(device, seed=5), AttitudeTask(w_att=0.0, w_rate=1.0, w_ctrl=1e-3),
                     max_episode_steps=10**9)
    local = VecEnv(_make_sim(device, seed=5), DetumbleTask(w_rate=1.0, w_ctrl=1e-3),
                   max_episode_steps=10**9)
    example.reset()
    local.reset()
    rng = np.random.default_rng(0)
    for _ in range(5):
        act = rng.uniform(-1.5, 1.5, (example.num_envs, example.act_dim)).astype(np.float32)
        example.step(wp.array(act, dtype=wp.float32, device=device))
        local.step(wp.array(act, dtype=wp.float32, device=device))
        np.testing.assert_allclose(example.obs.numpy(), local.obs.numpy(), rtol=1e-6, atol=0.0)
        np.testing.assert_allclose(example.reward.numpy(), local.reward.numpy(),
                                   rtol=1e-6, atol=0.0)


def test_nonfinite_action_terminates_its_environment(device):
    """A NaN or Inf action reaches the commands (wp.clamp would make it a
    finite torque), so a diverged policy ends its episode with reason 6."""
    env = VecEnv(_make_sim(device), AttitudeTask(), max_episode_steps=10**9)
    env.reset()
    act = np.zeros((env.num_envs, env.act_dim), dtype=np.float32)
    act[3, 0], act[5, 2] = np.nan, np.inf
    _, _, terminated, _ = env.step(wp.array(act, dtype=wp.float32, device=device))
    np.testing.assert_array_equal(np.flatnonzero(terminated.numpy()), [3, 5])

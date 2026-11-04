"""
Tests for the rsl_rl adapter (cosmic_warp.rl.adapters.rsl_rl).

Skipped unless the optional ``rsl-rl`` dependency (and torch) is installed.
Validate the adapter's interface shape and that PPO actually learns on a
detumble task — the end-to-end proof that the step contract, zero-copy torch
interop, and trainer integration line up.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")

from cosmic_warp.core.pipeline import SpacecraftSimulator
from cosmic_warp.rl import VecEnv
from cosmic_warp.rl.adapters.rsl_rl import RslRlVecEnv, default_ppo_cfg
from rsl_rl.runners import OnPolicyRunner
from rl_tasks import DetumbleTask


def _make_env(device, num_envs=256):
    sim = SpacecraftSimulator(
        num_envs=num_envs, integrator="symplectic", dt=0.1, device=device,
        gravity=True, j2_perturbation=False, gravity_gradient=False,
        check_bounds=True, ang_mom_limit=5.0, reset_config={"seed": 0},
    )
    sim.vehicle.set_config(mass_dry=12.0, inertia=[[0.4, 0, 0], [0, 0.4, 0], [0, 0, 0.3]])
    sim.rw_manager.set_config(
        max_torque=0.02, max_momentum=1.0, wheel_inertia=0.005, viscous_damping=1.0e-5
    )
    sim.reset_manager.set_bounds(
        att_max_angle=0.5236, ang_mom_bounds=((-0.2,) * 3, (0.2,) * 3), mass_bounds=(12.0, 12.0)
    )
    task = DetumbleTask(w_rate=1.0, w_ctrl=1.0e-3)
    return RslRlVecEnv(VecEnv(sim, task, max_episode_steps=200))


def test_adapter_interface(device):
    """get_observations / step return the shapes and keys rsl_rl expects."""
    env = _make_env(device)
    obs = env.get_observations()
    assert set(obs.keys()) == {"policy"}
    assert tuple(obs["policy"].shape) == (env.num_envs, env.vec_env.obs_dim)
    assert env.num_actions == env.vec_env.act_dim

    actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    obs, rewards, dones, extras = env.step(actions)
    assert tuple(rewards.shape) == (env.num_envs,)
    assert tuple(dones.shape) == (env.num_envs,)
    assert "time_outs" in extras and tuple(extras["time_outs"].shape) == (env.num_envs,)


def test_vec_env_accessor(device):
    env = _make_env(device)
    assert isinstance(env.vec_env, VecEnv) and env.vec_env.num_envs == env.num_envs


def test_episode_length_buf_setter_reaches_sim(device):
    """Assigning episode_length_buf (as rsl_rl's init_at_random_ep_len does)
    writes through to the simulator's episode counters instead of rebinding."""
    env = _make_env(device)
    env.episode_length_buf = torch.full_like(env.episode_length_buf, 7)
    assert (env.vec_env.sim.episode_steps.numpy() == 7).all()


def test_multibody_rejected(device):
    """The adapter is single-body for now and says so."""
    sim = SpacecraftSimulator(num_envs=8, num_bodies=2, device=device, check_bounds=False)
    with pytest.raises(NotImplementedError):
        RslRlVecEnv(VecEnv(sim, DetumbleTask(), max_episode_steps=100))


@torch.inference_mode()
def _mean_rate(env, policy, steps=150):
    env.vec_env.reset()
    rates = []
    for _ in range(steps):
        actions = policy(env.get_observations())
        env.vec_env.actions_torch().copy_(actions)
        env.vec_env.step()
        rates.append(np.linalg.norm(env.vec_env.obs.numpy()[:, 4:7], axis=1).mean())
    return float(np.mean(rates[-40:]))


@pytest.mark.slow
def test_ppo_learns_detumble(device):
    """PPO reduces the tumble rate — the whole pipeline learns end-to-end."""
    if "cuda" not in str(device):
        pytest.skip("rsl_rl PPO training is only exercised on cuda")
    # Seeded: unseeded torch init/exploration makes the learning-rate
    # assertion flaky (warp-side reset randomness is seeded separately).
    torch.manual_seed(0)
    env = _make_env(device)
    runner = OnPolicyRunner(env, default_ppo_cfg(num_steps_per_env=24), log_dir=None, device=env.device)

    rate_before = _mean_rate(env, runner.get_inference_policy(env.device))
    runner.learn(num_learning_iterations=150)
    rate_after = _mean_rate(env, runner.get_inference_policy(env.device))

    assert rate_after < 0.6 * rate_before, (
        f"expected the tumble rate to drop substantially, got {rate_before:.3f} -> {rate_after:.3f}"
    )

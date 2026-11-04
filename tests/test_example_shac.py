"""
Tests for example 17's SHAC trainer and differentiable attitude task
(examples/shac.py): the torch task matches the warp-kernel example task, its
reward gradient is finite at the goal and matches finite differences through
DiffVecEnv, TD(lambda) targets match a naive reference, and SHAC runs, is
reproducible, checkpoints and learns to detumble. Skipped without torch.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

torch = pytest.importorskip("torch")

from cosmic_warp.core.pipeline import SpacecraftSimulator
from cosmic_warp.rl.diff_env import DiffVecEnv
from conftest import set_state

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from attitude_task import AttitudeTask  # noqa: E402
from shac import SHAC, AttitudeDiffTask, td_lambda_targets  # noqa: E402

MASS0 = 2.0    # strictly above the default mass_dry (1.0): mass floor inactive
H = 4          # window length of the gradient check


def _make_toy_sim(make_sim, integrator="symplectic"):
    """Well-conditioned O(1) dynamics for gradcheck: no gravity, pure
    rotational dynamics plus drift, all state components of similar scale."""
    sim = make_sim(num_envs=2, integrator=integrator, dt=0.05,
                   differentiable=True, gravity=False)
    set_state(sim, pos=[[1.0, -2.0, 0.5]] * 2, vel=[[0.1, 0.05, -0.02]] * 2,
              mass=[MASS0] * 2, ang_mom=[[0.05, -0.03, 0.02]] * 2)
    sim.actuators.set_rw_state([0.02, -0.01, 0.015, 0.0])
    return sim


def _fd_vs_grad(analytic, fd, rtol=1e-4):
    scale = max(abs(fd), abs(analytic))
    if scale == 0.0:
        return
    rel = abs(fd - analytic) / scale
    assert rel < rtol, f"autograd {analytic:.9e} vs FD {fd:.9e} (rel {rel:.2e})"


def _x_from_sim(sim):
    """Flat [E, X_DIM] state tensor built directly from the sim's arrays."""
    s = sim.states.numpy()
    rw = sim.actuators.states.numpy()["rw_ang_mom"]
    x = np.concatenate(
        [s["pos"], s["mom"], s["mass"][:, None], s["quat"], s["ang_mom"], rw],
        axis=1)
    return torch.tensor(x, dtype=torch.float64, device=str(sim.device))


def test_diff_task_matches_kernel_task(make_sim):
    """AttitudeDiffTask (torch) reproduces AttitudeTask's kernel obs/reward."""
    sim = make_sim(num_envs=4, dt=0.1)
    rng = np.random.default_rng(3)
    quat = rng.normal(size=(sim.num_entities, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    set_state(sim, quat=quat, mass=[2.0] * sim.num_entities,
              ang_mom=rng.uniform(-0.2, 0.2, (sim.num_entities, 3)))
    sim.actuators.set_rw_state([0.1, -0.05, 0.2, 0.0])

    kernel_task = AttitudeTask(w_att=2.0, w_rate=0.5, w_ctrl=0.1)
    kernel_task.bind(SimpleNamespace(sim=sim))
    diff_task = AttitudeDiffTask(w_att=2.0, w_rate=0.5, w_ctrl=0.1)
    diff_task.bind(sim)

    action_np = rng.uniform(-1.0, 1.0, (sim.num_envs, diff_task.act_dim)).astype(np.float32)
    action_wp = wp.array(action_np, dtype=wp.float32, device=sim.device)
    obs_wp = wp.zeros((sim.num_envs, diff_task.obs_dim), dtype=wp.float32, device=sim.device)
    reward_wp = wp.zeros(sim.num_envs, dtype=wp.float32, device=sim.device)
    no_reset = wp.zeros(sim.num_envs, dtype=wp.bool, device=sim.device)
    kernel_task.write_obs(obs_wp, no_reset, 0)
    kernel_task.write_reward(reward_wp, action_wp)

    x = _x_from_sim(sim)
    action = torch.tensor(action_np, dtype=torch.float64, device=str(sim.device))
    obs = diff_task.observation(x)
    reward = diff_task.reward(x, action)

    np.testing.assert_allclose(obs.cpu().numpy(), obs_wp.numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(reward.cpu().numpy(), reward_wp.numpy(), rtol=1e-4, atol=1e-6)


def test_reward_grad_finite_at_goal(make_sim):
    """Reward gradient is finite (and the angle term flat) exactly at the goal
    quaternion — the acos formulation would return NaN/inf here."""
    sim = make_sim(num_envs=2, dt=0.1)
    set_state(sim, mass=[2.0] * 2)  # identity quat, zero rates
    task = AttitudeDiffTask()
    task.bind(sim)

    x = _x_from_sim(sim).requires_grad_(True)
    action = torch.zeros((2, task.act_dim), dtype=torch.float64,
                         device=x.device, requires_grad=True)
    task.reward(x, action).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(action.grad).all()


def test_diff_task_rejects_randomized_inertia(make_sim):
    """AttitudeDiffTask caches the inverse inertia at bind, but inertia
    randomization redraws it at every reset: obs and reward used to compute
    omega from the stale inertia (H12)."""
    sim = make_sim(num_envs=4, dt=0.1, differentiable=True)
    sim.reset_manager.set_bounds(mass_bounds=(2.0, 2.0))
    sim.randomizer.set_bounds(inertia_scale=(0.5, 2.0))
    with pytest.raises(ValueError, match="inertia randomization"):
        DiffVecEnv(sim, AttitudeDiffTask(), horizon=4, max_episode_steps=8)


def test_env_rollout_grad_matches_fd(make_sim):
    """d(loss)/d(policy params) through DiffVecEnv + AttitudeDiffTask matches
    central FD — the task's torch math (safe angle, omega) keeps the bridge's
    gradient."""
    sim = _make_toy_sim(make_sim)
    env = DiffVecEnv(sim, AttitudeDiffTask(), horizon=H, max_episode_steps=4 * H)
    env.reset()
    dev = env.device

    torch.manual_seed(0)
    W0 = 0.1 * torch.randn(env.act_dim, env.obs_dim, dtype=torch.float64, device=dev)

    def loss_fn(W):
        window = env.rollout(lambda obs: torch.tanh(obs @ W.T))
        return -torch.stack(window.rewards).sum()

    W = W0.clone().requires_grad_(True)
    g = torch.autograd.grad(loss_fn(W), W)[0]
    assert float(g.abs().max()) > 0.0

    h = 1e-5
    with torch.no_grad():
        for i, j in [(0, 0), (1, 4), (2, 8), (0, 10)]:
            Wp = W0.clone()
            Wp[i, j] += h
            Wm = W0.clone()
            Wm[i, j] -= h
            fd = float(loss_fn(Wp) - loss_fn(Wm)) / (2.0 * h)
            _fd_vs_grad(float(g[i, j]), fd)


def test_td_lambda_targets_match_naive():
    """Vectorized TD(lambda) equals the closed-form n-step-return mixture."""
    torch.manual_seed(0)
    H, N, gamma, lam = 5, 3, 0.9, 0.8
    rewards = torch.randn(H, N, dtype=torch.float64)
    values = torch.randn(H + 1, N, dtype=torch.float64)

    def n_step(t, n):
        g = sum(gamma ** k * rewards[t + k] for k in range(n))
        return g + gamma ** n * values[t + n]

    expected = torch.empty_like(rewards)
    for t in range(H):
        span = H - t
        acc = sum((1 - lam) * lam ** (n - 1) * n_step(t, n) for n in range(1, span))
        expected[t] = acc + lam ** (span - 1) * n_step(t, span)

    got = td_lambda_targets(rewards, values, gamma, lam)
    torch.testing.assert_close(got, expected)


def test_shac_smoke(make_sim):
    """A few SHAC updates run end-to-end: finite stats, parameters move."""
    sim = make_sim(num_envs=8, dt=0.1, differentiable=True)
    sim.reset_manager.set_bounds(
        att_max_angle=0.5, ang_mom_bounds=((-0.1,) * 3, (0.1,) * 3),
        mass_bounds=(2.0, 2.0))
    env = DiffVecEnv(sim, AttitudeDiffTask(w_att=0.0, w_rate=1.0, w_ctrl=1e-3),
                     horizon=8, max_episode_steps=16)
    shac = SHAC(env, {"critic_epochs": 2, "critic_minibatches": 2})

    p0 = [p.detach().clone() for p in shac.actor.parameters()]
    history = shac.train(num_updates=3)

    assert len(history) == 3
    for stats in history:
        assert all(np.isfinite(v) for v in stats.values()), stats
    assert any(not torch.equal(p0[i], p.detach())
               for i, p in enumerate(shac.actor.parameters()))


def _small_env(make_sim):
    sim = make_sim(num_envs=4, dt=0.1, differentiable=True)
    sim.reset_manager.set_bounds(
        att_max_angle=0.5, ang_mom_bounds=((-0.1,) * 3, (0.1,) * 3),
        mass_bounds=(2.0, 2.0))
    return DiffVecEnv(sim, AttitudeDiffTask(w_att=0.0, w_rate=1.0, w_ctrl=1e-3),
                      horizon=4, max_episode_steps=8)


@pytest.mark.parametrize("minibatches", [0, 17])
def test_shac_rejects_empty_critic_minibatches(make_sim, minibatches):
    """A window holds horizon * num_envs = 16 critic samples: 17 minibatches
    left some empty (NaN critic loss), and 0 divided by zero (H27)."""
    env = _small_env(make_sim)
    with pytest.raises(ValueError, match="critic_minibatches"):
        SHAC(env, {"critic_minibatches": minibatches})
    SHAC(env, {"critic_epochs": 1, "critic_minibatches": 16}).train(num_updates=1)


def test_shac_seed_reproducible(make_sim):
    """The same cfg seed gives identical networks; a different one does not."""
    s1 = SHAC(_small_env(make_sim), {"seed": 7})
    s2 = SHAC(_small_env(make_sim), {"seed": 7})
    s3 = SHAC(_small_env(make_sim), {"seed": 8})

    for p1, p2 in zip(s1.actor.parameters(), s2.actor.parameters()):
        torch.testing.assert_close(p1, p2, rtol=0.0, atol=0.0)
    assert any(not torch.equal(a, b)
               for a, b in zip(s1.actor.parameters(), s3.actor.parameters()))


def test_shac_save_load_roundtrip(make_sim, tmp_path):
    """save() -> load() restores the policy exactly (networks, optimizers,
    observation statistics)."""
    env = _small_env(make_sim)
    shac = SHAC(env, {"critic_epochs": 1, "critic_minibatches": 1, "seed": 0})
    shac.train(num_updates=2)   # move optimizer state and obs statistics
    path = tmp_path / "shac_checkpoint.pt"
    shac.save(path)

    other = SHAC(_small_env(make_sim),
                 {"critic_epochs": 1, "critic_minibatches": 1, "seed": 123})
    obs = torch.randn(env.num_envs, env.obs_dim, dtype=torch.float64,
                      device=env.device)
    assert not torch.allclose(other.policy(obs, deterministic=True),
                              shac.policy(obs, deterministic=True))

    other.load(path)
    torch.testing.assert_close(other.policy(obs, deterministic=True),
                               shac.policy(obs, deterministic=True),
                               rtol=0.0, atol=0.0)
    torch.testing.assert_close(other.obs_rms.mean, shac.obs_rms.mean)
    assert other.obs_rms.count == shac.obs_rms.count
    assert other.update_count == 2   # lr decay resumes, not restarts


@pytest.mark.slow
def test_shac_learns_detumble(device):
    """SHAC reduces the tumble rate end-to-end — and does it in ~5x fewer
    updates than the PPO baselines (first-order sample efficiency)."""
    if "cuda" not in str(device):
        pytest.skip("SHAC training is only exercised on cuda")
    sim = SpacecraftSimulator(
        num_envs=256, integrator="symplectic", dt=0.1, device=device,
        gravity=True, j2_perturbation=False, check_bounds=False,
        differentiable=True, reset_config={"seed": 0},
    )
    sim.vehicle.set_config(mass_dry=12.0, inertia=[[0.4, 0, 0], [0, 0.4, 0], [0, 0, 0.3]])
    sim.rw_manager.set_config(
        max_torque=0.02, max_momentum=1.0, wheel_inertia=0.005, viscous_damping=1.0e-5
    )
    sim.reset_manager.set_bounds(
        att_max_angle=0.5236, ang_mom_bounds=((-0.2,) * 3, (0.2,) * 3), mass_bounds=(12.0, 12.0)
    )
    env = DiffVecEnv(sim, AttitudeDiffTask(w_att=0.0, w_rate=1.0, w_ctrl=1e-3),
                     horizon=32, max_episode_steps=192)
    # Seeded: unseeded torch init/exploration makes learning-rate assertions
    # flaky (as with the PPO detumble baselines).
    shac = SHAC(env, {"seed": 0})

    @torch.no_grad()
    def mean_rate(steps=150):
        env.reset()
        x = env.state
        rates = []
        for _ in range(steps):
            obs = env.task.observation(x)
            x = env.stepper.step(x, env.task.commands(shac.policy(obs, deterministic=True)))
            rates.append(float(env.task.observation(x)[:, 4:7].norm(dim=1).mean()))
        env.reset()
        return float(np.mean(rates[-40:]))

    rate_before = mean_rate()
    shac.train(num_updates=30)
    rate_after = mean_rate()

    assert rate_after < 0.6 * rate_before, (
        f"expected the tumble rate to drop substantially, got {rate_before:.3f} -> {rate_after:.3f}"
    )

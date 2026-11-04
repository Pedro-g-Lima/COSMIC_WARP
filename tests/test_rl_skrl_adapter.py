"""
Tests for the skrl adapter (cosmic_warp.rl.adapters.skrl).

Skipped unless the optional ``skrl`` dependency (and torch) is installed.
Validate the wrapper's interface and that a skrl PPO agent learns to detumble
end-to-end.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("skrl")

from cosmic_warp.core.pipeline import SpacecraftSimulator
from cosmic_warp.rl import VecEnv
from cosmic_warp.rl.adapters.skrl import TERMINAL_OBS_KEY, SkrlVecEnv, TerminalObsPPO, build_ppo_agent
from skrl.trainers.torch import SequentialTrainer
from rl_tasks import DetumbleTask


@pytest.fixture(autouse=True)
def _quiet_skrl_at_shutdown():
    """skrl closes its env at interpreter exit and logs to stdout/stderr, which
    pytest has already closed by then (harmless ValueError traceback). Silence
    skrl's logger after each test so nothing tries to write at shutdown."""
    import logging
    yield
    logging.getLogger("skrl").setLevel(logging.CRITICAL + 1)


def _make_env(device, num_envs=256, max_episode_steps=200):
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
    return SkrlVecEnv(VecEnv(sim, task, max_episode_steps=max_episode_steps))


def test_adapter_interface(device):
    """reset/step return skrl's tensor shapes; spaces have the right dims."""
    env = _make_env(device)
    assert env.observation_space.shape == (env.vec_env.obs_dim,)
    assert env.action_space.shape == (env.vec_env.act_dim,)
    assert env.num_agents == 1

    obs, info = env.reset()
    assert tuple(obs.shape) == (env.num_envs, env.vec_env.obs_dim)

    actions = torch.zeros((env.num_envs, env.vec_env.act_dim), device=env.device)
    obs, reward, terminated, truncated, info = env.step(actions)
    assert tuple(obs.shape) == (env.num_envs, env.vec_env.obs_dim)
    assert tuple(info[TERMINAL_OBS_KEY].shape) == (env.num_envs, env.vec_env.obs_dim)
    for t in (reward, terminated, truncated):
        assert tuple(t.shape) == (env.num_envs, 1)


def test_truncated_rows_bootstrap_from_terminal_obs(device):
    """build_ppo_agent's PPO evaluates the time-limit bootstrap value on the
    terminal observation for truncated rows, and on the returned observation
    for the others. skrl's PPO evaluates it on next_observations, which for a
    truncated environment is already its next episode's first observation."""
    env = _make_env(device, num_envs=8, max_episode_steps=5)
    agent = build_ppo_agent(env, rollouts=24)
    assert isinstance(agent, TerminalObsPPO) and agent.cfg.time_limit_bootstrap
    agent.init()
    agent.enable_training_mode(True)          # as the trainer does

    value_inputs = []
    value_act = agent.value.act

    def spy(inputs, role=""):
        value_inputs.append(inputs["observations"].clone())
        return value_act(inputs, role=role)

    agent.value.act = spy
    obs, _ = env.reset()
    for t in range(5):
        if t == 2:
            env.vec_env.sim.reset(env_ids=[0, 1, 2, 3])   # these truncate later
        with torch.no_grad():
            actions = agent.act(obs, None, timestep=t, timesteps=5)[0]
        next_obs, reward, terminated, truncated, infos = env.step(actions)
        value_inputs.clear()                  # keep only record_transition's calls
        agent.record_transition(
            observations=obs, states=None, actions=actions, rewards=reward,
            next_observations=next_obs, next_states=None, terminated=terminated,
            truncated=truncated, infos=infos, timestep=t, timesteps=5)
        obs = next_obs

    trunc = truncated.view(-1)
    assert trunc.tolist() == [False] * 4 + [True] * 4
    assert not terminated.any()
    terminal = infos[TERMINAL_OBS_KEY]
    assert len(value_inputs) == 1
    torch.testing.assert_close(value_inputs[0][trunc], terminal[trunc], rtol=0.0, atol=0.0)
    torch.testing.assert_close(value_inputs[0][~trunc], next_obs[~trunc], rtol=0.0, atol=0.0)
    assert not torch.allclose(terminal[trunc], next_obs[trunc])   # post-reset obs differ


def test_gae_does_not_read_truncated_last_values():
    """Pins the skrl assumption TerminalObsPPO relies on: the observation it
    substitutes also becomes the rollout's last observation, whose value GAE
    uses only for rows that did not end (time_limit_bootstrap on)."""
    from skrl.agents.torch.ppo.ppo import compute_gae

    T, N = 3, 4
    gen = torch.Generator().manual_seed(0)
    kw = dict(rewards=torch.randn(T, N, 1, generator=gen),
              values=torch.randn(T, N, 1, generator=gen),
              terminated=torch.zeros(T, N, 1, dtype=torch.bool),
              truncated=torch.zeros(T, N, 1, dtype=torch.bool),
              time_limit_bootstrap=True)
    kw["truncated"][-1, :2] = True
    last = torch.randn(N, 1, generator=gen)
    changed = last.clone()
    changed[:2] += 100.0                       # only the truncated rows
    returns, _ = compute_gae(last_values=last, **kw)
    returns_changed, _ = compute_gae(last_values=changed, **kw)
    torch.testing.assert_close(returns, returns_changed, rtol=0.0, atol=0.0)


def test_multibody_rejected(device):
    sim = SpacecraftSimulator(num_envs=8, num_bodies=2, device=device, check_bounds=False)
    with pytest.raises(NotImplementedError):
        SkrlVecEnv(VecEnv(sim, DetumbleTask(), max_episode_steps=100))


@torch.inference_mode()
def _mean_rate(env, agent, steps=150):
    obs, _ = env.reset()
    rates = []
    for _ in range(steps):
        actions = agent.act(obs, None, timestep=0, timesteps=1)[0]
        obs, _, _, _, _ = env.step(actions)
        rates.append(np.linalg.norm(env.vec_env.obs.numpy()[:, 4:7], axis=1).mean())
    return float(np.mean(rates[-40:]))


@pytest.mark.slow
def test_ppo_learns_detumble(device):
    """A skrl PPO agent reduces the tumble rate end-to-end."""
    if "cuda" not in str(device):
        pytest.skip("skrl PPO training is only exercised on cuda")
    # Seeded: unseeded torch init/exploration makes the learning-rate
    # assertion flaky (warp-side reset randomness is seeded separately).
    torch.manual_seed(0)
    env = _make_env(device)
    agent = build_ppo_agent(env, rollouts=24)
    agent.init()

    rate_before = _mean_rate(env, agent)
    trainer = SequentialTrainer(
        env=env, agents=agent,
        cfg={"timesteps": 24 * 100, "headless": True, "disable_progressbar": True},
    )
    trainer.train()
    rate_after = _mean_rate(env, agent)

    assert rate_after < 0.6 * rate_before, (
        f"expected the tumble rate to drop substantially, got {rate_before:.3f} -> {rate_after:.3f}"
    )

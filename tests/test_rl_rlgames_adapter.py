"""
Tests for the rl_games adapter (cosmic_warp.rl.adapters.rl_games).

Skipped unless the optional ``rl-games`` dependency (and torch) is installed.
Validate the IVecEnv interface and that rl_games' PPO learns to detumble
end-to-end. rl_games log/checkpoint output is redirected to a temp dir.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rl_games")

from cosmic_warp.core.pipeline import SpacecraftSimulator
from cosmic_warp.rl import VecEnv
from cosmic_warp.rl.adapters.rl_games import (
    RlGamesVecEnv, register_env, default_ppo_params,
)
from rl_tasks import DetumbleTask


def _make_env(device, num_envs=1024):
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
    return RlGamesVecEnv(VecEnv(sim, task, max_episode_steps=200))


def test_adapter_interface(device):
    """IVecEnv methods return rl_games' expected structures."""
    env = _make_env(device, num_envs=256)
    n = env.vec_env.num_envs

    info = env.get_env_info()
    assert info["observation_space"].shape == (env.vec_env.obs_dim,)
    assert info["action_space"].shape == (env.vec_env.act_dim,)
    assert env.get_number_of_agents() == 1

    obs = env.reset()
    assert "obs" in obs and tuple(obs["obs"].shape) == (n, env.vec_env.obs_dim)

    obs, rewards, dones, infos = env.step(torch.zeros((n, env.vec_env.act_dim), device=env.vec_env.device))
    assert "obs" in obs
    assert tuple(rewards.shape) == (n,)
    assert tuple(dones.shape) == (n,)
    assert "time_outs" in infos and tuple(infos["time_outs"].shape) == (n,)


def test_multibody_rejected(device):
    sim = SpacecraftSimulator(num_envs=8, num_bodies=2, device=device, check_bounds=False)
    with pytest.raises(NotImplementedError):
        RlGamesVecEnv(VecEnv(sim, DetumbleTask(), max_episode_steps=100))


@torch.inference_mode()
def _mean_rate(env, algo, steps=150):
    obs = env.reset()["obs"]
    rates = []
    for _ in range(steps):
        if algo is None:
            actions = torch.zeros((env.vec_env.num_envs, env.vec_env.act_dim), device=env.vec_env.device)
        else:
            res = algo.model({"is_train": False, "prev_actions": None,
                              "obs": algo._preproc_obs(obs), "rnn_states": None})
            actions = res["mus"]  # deterministic mean
        obs = env.step(actions)[0]["obs"]
        rates.append(np.linalg.norm(env.vec_env.obs.numpy()[:, 4:7], axis=1).mean())
    return float(np.mean(rates[-40:]))


@pytest.mark.slow
def test_ppo_learns_detumble(device, tmp_path):
    """rl_games PPO reduces the tumble rate end-to-end."""
    if "cuda" not in str(device):
        pytest.skip("rl_games PPO training is only exercised on cuda")
    from rl_games.torch_runner import Runner
    from rl_games.common.algo_observer import AlgoObserver

    # Seeded: unseeded torch init/exploration makes the learning-rate
    # assertion flaky (warp-side reset randomness is seeded separately).
    torch.manual_seed(0)
    env = _make_env(device, num_envs=1024)
    rate_before = _mean_rate(env, None)

    register_env("cw_test", env)
    captured = {}

    class _Grab(AlgoObserver):
        def after_init(self, algo):
            captured["algo"] = algo

    params = default_ppo_params("cw_test", env.vec_env.num_envs, max_epochs=100, train_dir=str(tmp_path))
    runner = Runner(algo_observer=_Grab())
    runner.load(params)
    runner.run_train({})

    rate_after = _mean_rate(env, captured["algo"])
    assert rate_after < 0.6 * rate_before, (
        f"expected the tumble rate to drop substantially, got {rate_before:.3f} -> {rate_after:.3f}"
    )


def test_player_with_default_params(device):
    """Runner.create_player() works after register_env with the default
    params. register_env used to register an env_creator returning None,
    which the player calls by default (AttributeError on observation_space)."""
    from rl_games.torch_runner import Runner

    env = _make_env(device, num_envs=16)
    register_env("cw_test_player", env)
    runner = Runner()
    runner.load(default_ppo_params("cw_test_player", env.vec_env.num_envs, device=str(device)))
    player = runner.create_player()
    assert player.env is env
    assert player.device == torch.device(str(device))

    player.has_batch_dimension = True        # [N, obs_dim] batches
    obs = env.reset()["obs"]
    actions = player.get_action(obs, is_deterministic=True)
    assert tuple(actions.shape) == (env.vec_env.num_envs, env.vec_env.act_dim)
    assert env.step(actions)[0]["obs"].shape == obs.shape

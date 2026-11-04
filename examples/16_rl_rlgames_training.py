"""PPO with rl_games

Trains an attitude-control policy with rl_games' `a2c_continuous` PPO,
through the package's rl_games adapter. The simulation, observation, reward
and policy all live on the GPU.

- wraps the RL environment, with the `AttitudeTask` from
  `examples/attitude_task.py` (as in [example 13](13_rl_attitude_env.md)),
  in `RlGamesVecEnv` and registers it with `register_env`
- builds the configuration with `default_ppo_params` and runs rl_games'
  `Runner` for 100 epochs by default
- reports the mean tumble rate of zero actions before training and of the
  deterministic policy after it, averaged over the last 50 of 200 steps

It needs the optional dependency:

```bash
pip install "cosmic-warp[rl-games]"      # rl-games + torch
python examples/16_rl_rlgames_training.py
```

Options: `--num-envs`, `--epochs`, `--w-att`, `--device`. By default it
trains the pure-detumble task (`--w-att 0`); a nonzero `--w-att` adds the
harder attitude-regulation term (full repointing). See the
[reinforcement learning guide](../guides/rl.md) for the adapters.

**What to look for:** the mean tumble rate drops from about 0.54 rad/s with
no control to about 0.04 rad/s after training: the same on-device pipeline
as the other adapters, through rl_games' registry-driven `IVecEnv`
interface.
"""
import argparse

import numpy as np
import torch
import warp as wp

from cosmic_warp.core.pipeline import SpacecraftSimulator
from cosmic_warp.rl import VecEnv
from cosmic_warp.rl.adapters.rl_games import (
    RlGamesVecEnv, register_env, default_ppo_params,
)
from rl_games.torch_runner import Runner
from rl_games.common.algo_observer import AlgoObserver

from attitude_task import AttitudeTask


def build_env(num_envs: int, device: str, w_att: float) -> RlGamesVecEnv:
    sim = SpacecraftSimulator(
        num_envs=num_envs, integrator="symplectic", dt=0.1, device=device,
        gravity=True, j2_perturbation=True, gravity_gradient=True,
        check_bounds=True, ang_mom_limit=5.0, reset_config={"seed": 0},
    )
    sim.vehicle.set_config(mass_dry=12.0, inertia=[[0.4, 0, 0], [0, 0.4, 0], [0, 0, 0.3]])
    sim.rw_manager.set_config(
        max_torque=0.02, max_momentum=1.0, wheel_inertia=0.005, viscous_damping=1.0e-5
    )
    sim.reset_manager.set_bounds(
        att_max_angle=0.5236, ang_mom_bounds=((-0.2,) * 3, (0.2,) * 3), mass_bounds=(12.0, 12.0)
    )
    task = AttitudeTask(w_att=w_att, w_rate=1.0, w_ctrl=1.0e-3)
    return RlGamesVecEnv(VecEnv(sim, task, max_episode_steps=200))


@torch.inference_mode()
def mean_tumble_rate(env: RlGamesVecEnv, algo, steps: int = 200) -> float:
    obs = env.reset()["obs"]
    rates = []
    for _ in range(steps):
        if algo is None:
            actions = torch.zeros((env.vec_env.num_envs, env.vec_env.act_dim), device=env.vec_env.device)
        else:
            res = algo.model({"is_train": False, "prev_actions": None,
                              "obs": algo._preproc_obs(obs), "rnn_states": None})
            actions = res["mus"]
        obs = env.step(actions)[0]["obs"]
        rates.append(np.linalg.norm(env.vec_env.obs.numpy()[:, 4:7], axis=1).mean())
    return float(np.mean(rates[-50:]))


def main():
    p = argparse.ArgumentParser(description="PPO attitude control via rl_games")
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--w-att", type=float, default=0.0, help="attitude-regulation weight (0 = pure detumble)")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    wp.init()
    env = build_env(args.num_envs, args.device, args.w_att)
    print(f"before training : mean tumble rate = {mean_tumble_rate(env, None):.4f} rad/s")

    register_env("cw_attitude", env)
    captured = {}

    class _Grab(AlgoObserver):
        def after_init(self, algo):
            captured["algo"] = algo

    params = default_ppo_params("cw_attitude", args.num_envs, max_epochs=args.epochs, device=args.device)
    runner = Runner(algo_observer=_Grab())
    runner.load(params)
    runner.run_train({})

    after = mean_tumble_rate(env, captured["algo"])
    print(f"after training  : mean tumble rate = {after:.4f} rad/s")


if __name__ == "__main__":
    main()

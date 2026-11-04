"""PPO with skrl

Trains a Gaussian-policy PPO agent with `skrl`, through the package's skrl
adapter. The simulation, observation, reward and policy all live on the GPU.

- wraps the RL environment, with the `AttitudeTask` from
  `examples/attitude_task.py` (as in [example 13](13_rl_attitude_env.md)),
  in `SkrlVecEnv`
- builds a PPO agent with `build_ppo_agent` (24-step rollouts) and runs
  skrl's `SequentialTrainer` for 100 rollouts by default
- reports the mean tumble rate of the deterministic policy (the Gaussian
  mean) before and after training, averaged over the last 50 of 200 steps

It needs the optional dependency:

```bash
pip install "cosmic-warp[skrl]"          # skrl + torch
python examples/15_rl_skrl_training.py
```

Options: `--num-envs`, `--iterations`, `--w-att`, `--device`. By default it
trains the pure-detumble task (`--w-att 0`); a nonzero `--w-att` adds the
harder attitude-regulation term (full repointing). See the
[reinforcement learning guide](../guides/rl.md) for the adapters.

**What to look for:** the mean tumble rate drops by about 95 % after
training: the same on-device pipeline as the other adapters, through skrl's
wrapper interface.
"""
import argparse

import numpy as np
import torch
import warp as wp

from cosmic_warp.core.pipeline import SpacecraftSimulator
from cosmic_warp.rl import VecEnv
from cosmic_warp.rl.adapters.skrl import SkrlVecEnv, build_ppo_agent
from skrl.trainers.torch import SequentialTrainer

from attitude_task import AttitudeTask


def build_env(num_envs: int, device: str, w_att: float) -> SkrlVecEnv:
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
    return SkrlVecEnv(VecEnv(sim, task, max_episode_steps=200))


@torch.inference_mode()
def mean_tumble_rate(env: SkrlVecEnv, agent, steps: int = 200) -> float:
    obs, _ = env.reset()
    rates = []
    for _ in range(steps):
        # Deterministic policy (the Gaussian mean), as examples 14 and 17
        actions = agent.act(obs, None, timestep=0, timesteps=1)[1]["mean_actions"]
        obs, _, _, _, _ = env.step(actions)
        rates.append(np.linalg.norm(env.vec_env.obs.numpy()[:, 4:7], axis=1).mean())
    return float(np.mean(rates[-50:]))


def main():
    p = argparse.ArgumentParser(description="PPO attitude control via skrl")
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--w-att", type=float, default=0.0, help="attitude-regulation weight (0 = pure detumble)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    wp.init()
    rollouts = 24
    env = build_env(args.num_envs, args.device, args.w_att)
    agent = build_ppo_agent(env, rollouts=rollouts)
    agent.init()

    before = mean_tumble_rate(env, agent)
    print(f"before training : mean tumble rate = {before:.4f} rad/s")

    trainer = SequentialTrainer(
        env=env, agents=agent,
        cfg={"timesteps": rollouts * args.iterations, "headless": True},
    )
    trainer.train()

    after = mean_tumble_rate(env, agent)
    print(f"after training  : mean tumble rate = {after:.4f} rad/s   ({100*(1-after/before):.0f}% lower)")


if __name__ == "__main__":
    main()

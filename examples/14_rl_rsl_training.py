"""PPO with rsl_rl

Trains an attitude-control policy with PPO from `rsl-rl-lib`, through the
package's rsl_rl adapter. The simulation, observation, reward and policy all
live on the GPU; the only host traffic is logging.

- wraps the RL environment, with the `AttitudeTask` from
  `examples/attitude_task.py` (as in [example 13](13_rl_attitude_env.md)),
  in `RslRlVecEnv`
- runs rsl_rl's `OnPolicyRunner` with `default_ppo_cfg` (24 steps per
  environment per iteration, 100 iterations by default)
- reports the mean tumble rate of the deterministic policy before and after
  training, averaged over the last 50 of 200 steps

It needs the optional dependency:

```bash
pip install "cosmic-warp[rsl-rl]"        # rsl-rl-lib + torch
python examples/14_rl_rsl_training.py
```

Options: `--num-envs`, `--iterations`, `--w-att`, `--device`. Default task: detumble
(`--w-att 0`, a few seconds on a desktop GPU); a nonzero `--w-att` adds
repointing. See the [reinforcement learning guide](../guides/rl.md)
for the adapters.

**What to look for:** the mean tumble rate drops by about 70-90 % from before
to after training (the trainer is not seeded, so runs differ), the end-to-end
check that the step contract, the zero-copy torch interop and the trainer
integration line up.
"""
import argparse

import numpy as np
import torch
import warp as wp

from cosmic_warp.core.pipeline import SpacecraftSimulator
from cosmic_warp.rl import VecEnv
from cosmic_warp.rl.adapters.rsl_rl import RslRlVecEnv, default_ppo_cfg
from rsl_rl.runners import OnPolicyRunner

from attitude_task import AttitudeTask


def build_env(num_envs: int, device: str, w_att: float) -> RslRlVecEnv:
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
    return RslRlVecEnv(VecEnv(sim, task, max_episode_steps=200))


@torch.inference_mode()
def mean_tumble_rate(env: RslRlVecEnv, policy, steps: int = 200) -> float:
    """Deterministic-policy rollout; mean body-rate magnitude over the tail."""
    env.vec_env.reset()
    rates = []
    for _ in range(steps):
        actions = policy(env.get_observations())
        env.vec_env.actions_torch().copy_(actions)
        env.vec_env.step()
        rates.append(np.linalg.norm(env.vec_env.obs.numpy()[:, 4:7], axis=1).mean())
    return float(np.mean(rates[-50:]))


def main():
    p = argparse.ArgumentParser(description="PPO attitude control via rsl_rl")
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--w-att", type=float, default=0.0, help="attitude-regulation weight (0 = pure detumble)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    wp.init()
    env = build_env(args.num_envs, args.device, args.w_att)
    runner = OnPolicyRunner(env, default_ppo_cfg(num_steps_per_env=24), log_dir=None, device=args.device)

    before = mean_tumble_rate(env, runner.get_inference_policy(args.device))
    print(f"before training : mean tumble rate = {before:.4f} rad/s")

    runner.learn(num_learning_iterations=args.iterations)

    after = mean_tumble_rate(env, runner.get_inference_policy(args.device))
    print(f"after training  : mean tumble rate = {after:.4f} rad/s   ({100*(1-after/before):.0f}% lower)")


if __name__ == "__main__":
    main()

"""SHAC training (differentiable RL)

Trains an attitude-control policy with SHAC (Short-Horizon Actor-Critic),
using first-order gradients backpropagated analytically through the
simulator, with no external RL framework. Unlike the PPO examples (14-16),
which treat the simulator as a black box, SHAC differentiates through the
dynamics: the actor loss is backpropagated across each short window (Warp
adjoints inside every simulator step, torch autograd around the policy in
the loop), while a TD(λ) critic bootstraps beyond the window.

- builds a `differentiable=True` simulator with termination checks off
  (episodes end only by truncation, after 6 windows) and wraps it in the
  package's `DiffVecEnv` with the torch-side `AttitudeDiffTask`
- runs the `SHAC` trainer: analytic actor gradient through 32-step windows
  (`--horizon`), TD(λ) critic, 50 updates by default
- the trainer and the task are a minimal reference implementation in
  [`examples/shac.py`](shac.md), next to the script, not part of the package
- reports the mean tumble rate of the deterministic policy before and after
  training, averaged over the last 50 of 200 steps

It needs only the optional torch dependency:

```bash
pip install "cosmic-warp[torch]"
python examples/17_rl_shac_training.py
```

Options: `--num-envs`, `--updates`, `--horizon`, `--w-att`, `--device`. By
default it trains the pure-detumble task (`--w-att 0`); a nonzero `--w-att`
adds the harder attitude-regulation term (full repointing). See the
[differentiable RL guide](../guides/differentiable-rl.md) for the
environment and the gradient path.

**What to look for:** the mean tumble rate drops by about 90 % in the
default 50 updates, about 7 s for the whole script on a desktop GPU (the
trainer is not seeded, so runs differ). The logged reward jumps down every
6 updates: that is where the episodes are truncated and reset to a new
random tumble.
"""
import argparse

import numpy as np
import torch
import warp as wp

from cosmic_warp.core.pipeline import SpacecraftSimulator
from cosmic_warp.rl.diff_env import DiffVecEnv

from shac import SHAC, AttitudeDiffTask


def build_env(num_envs: int, device: str, w_att: float, horizon: int) -> DiffVecEnv:
    # The differentiable path needs differentiable=True and termination checks
    # off (truncation-only episodes) — see the differentiable-RL guide.
    sim = SpacecraftSimulator(
        num_envs=num_envs, integrator="symplectic", dt=0.1, device=device,
        gravity=True, j2_perturbation=True, gravity_gradient=True,
        check_bounds=False, differentiable=True, reset_config={"seed": 0},
    )
    sim.vehicle.set_config(mass_dry=12.0, inertia=[[0.4, 0, 0], [0, 0.4, 0], [0, 0, 0.3]])
    sim.rw_manager.set_config(
        max_torque=0.02, max_momentum=1.0, wheel_inertia=0.005, viscous_damping=1.0e-5
    )
    sim.reset_manager.set_bounds(
        att_max_angle=0.5236, ang_mom_bounds=((-0.2,) * 3, (0.2,) * 3), mass_bounds=(12.0, 12.0)
    )
    task = AttitudeDiffTask(w_att=w_att, w_rate=1.0, w_ctrl=1.0e-3)
    return DiffVecEnv(sim, task, horizon=horizon, max_episode_steps=6 * horizon)


@torch.no_grad()
def mean_tumble_rate(env: DiffVecEnv, shac: SHAC, steps: int = 200) -> float:
    """Deterministic-policy rollout; mean body-rate magnitude over the tail."""
    env.reset()
    x = env.state
    rates = []
    for _ in range(steps):
        obs = env.task.observation(x)
        action = shac.policy(obs, deterministic=True)
        x = env.stepper.step(x, env.task.commands(action))
        omega = env.task.observation(x)[:, 4:7]
        rates.append(float(omega.norm(dim=1).mean()))
    env.reset()
    return float(np.mean(rates[-50:]))


def main():
    p = argparse.ArgumentParser(description="SHAC attitude control (differentiable sim)")
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--updates", type=int, default=50)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--w-att", type=float, default=0.0, help="attitude-regulation weight (0 = pure detumble)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    wp.init()
    env = build_env(args.num_envs, args.device, args.w_att, args.horizon)
    shac = SHAC(env)

    before = mean_tumble_rate(env, shac)
    print(f"before training : mean tumble rate = {before:.4f} rad/s")

    shac.train(args.updates, log_fn=lambda i, s: print(
        f"  update {i:3d} | reward {s['mean_reward']:+.4f} | "
        f"critic {s['critic_loss']:.4f} | policy std {s['actor_std']:.3f}"))

    after = mean_tumble_rate(env, shac)
    print(f"after training  : mean tumble rate = {after:.4f} rad/s   ({100*(1-after/before):.0f}% lower)")


if __name__ == "__main__":
    main()

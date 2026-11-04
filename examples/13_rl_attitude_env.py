"""RL attitude-control environment

Drives the vectorized RL environment, `cosmic_warp.rl.VecEnv`, with a policy
that is itself a Warp kernel, so the whole loop (policy, environment step,
auto-reset) stays on the GPU with no host synchronization, as in an RL
trainer's inner loop.

- wraps a simulator in `VecEnv` with the reference `AttitudeTask` from
  `examples/attitude_task.py`, which examples 14-16 share (the package ships
  the `Task` protocol, not tasks)
- 4096 spacecraft with three reaction wheels, a random initial attitude
  error of up to 30 deg and a random tumble
- the policy is a fixed proportional detumbler, `action = gain * ω`
  (clamped to [-1, 1]): the wheel reaction torque opposes the tumble
- compares an uncontrolled rollout with the controlled one from identical
  initial conditions (same seed)

This is not learning: it shows that the environment works, that the reward
responds to control, and that the step is fast. Replace the kernel with a
neural-network policy (through the zero-copy torch views) to train, as in
[example 14](14_rl_rsl_training.md). See the
[reinforcement learning guide](../guides/rl.md) for the environment API.

```bash
python examples/13_rl_attitude_env.py
python examples/13_rl_attitude_env.py --num-envs 65536 --gain 4
```

Options: `--num-envs`, `--steps`, `--gain`, `--device`.

**What to look for:** over the default 1000 steps (100 s) the controlled
mean body rate falls from about 0.54 rad/s to about 0.002 rad/s, while the
uncontrolled one stays at 0.54 rad/s. The printed throughput is tens of
millions of env-steps/s on a desktop GPU (about 40 M with the default 4096
environments on an RTX 4070 Ti SUPER, more with more environments); the
first run is slower because it includes compiling the policy kernel.
"""
import argparse
import time

import warp as wp

from cosmic_warp.core.pipeline import SpacecraftSimulator
from cosmic_warp.rl import VecEnv

from attitude_task import AttitudeTask


@wp.kernel
def detumble_policy(
    obs: wp.array(dtype=wp.float32, ndim=2),
    actions: wp.array(dtype=wp.float32, ndim=2),
    num_bodies: int,
    gain: wp.float32,
):
    """Proportional detumble: spin each wheel up along the body rate so the
    reaction torque bleeds off the tumble (commanding +axis wheel torque
    removes +axis body angular momentum).

    Reads angular velocity straight from the observation buffer and writes the
    action buffer in place — both are the env's persistent GPU tensors.
    """
    env = wp.tid()
    for b in range(num_bodies):
        obs_base = b * 11   # [quat(4), angvel(3), rw_ang_mom(4)]
        act_base = b * 3
        for k in range(3):
            a = gain * obs[env, obs_base + 4 + k]
            actions[env, act_base + k] = wp.clamp(a, wp.float32(-1.0), wp.float32(1.0))


def build_env(num_envs: int, device: str, seed: int) -> VecEnv:
    # No termination + a long episode limit: the comparison rollouts run
    # uninterrupted so both policies see identical initial conditions.
    sim = SpacecraftSimulator(
        num_envs=num_envs,
        integrator="symplectic",
        dt=0.1,
        device=device,
        gravity=True,
        j2_perturbation=True,
        gravity_gradient=True,
        check_bounds=False,
        reset_config={"seed": seed},
    )
    sim.vehicle.set_config(mass_dry=12.0, inertia=[[0.4, 0, 0], [0, 0.4, 0], [0, 0, 0.3]])
    sim.rw_manager.set_config(
        max_torque=0.02, max_momentum=1.0, wheel_inertia=0.005, viscous_damping=1.0e-5
    )
    # Randomized initial attitude error and modest tumble, fixed orbit.
    sim.reset_manager.set_bounds(
        att_max_angle=0.5236, ang_mom_bounds=((-0.2,) * 3, (0.2,) * 3), mass_bounds=(12.0, 12.0)
    )
    return VecEnv(sim, AttitudeTask(), max_episode_steps=10**9)


def mean_rate(env: VecEnv) -> float:
    """Mean body angular-rate magnitude across envs [rad/s] (obs[:, 4:7])."""
    import numpy as np

    omega = env.obs.numpy()[:, 4:7]
    return float(np.linalg.norm(omega, axis=1).mean())


def rollout(env: VecEnv, steps: int, gain: float) -> None:
    for _ in range(steps):
        if gain > 0.0:
            wp.launch(detumble_policy, dim=env.num_envs,
                      inputs=[env.obs, env.actions, env.num_bodies, wp.float32(gain)],
                      device=env.device)
        env.step()  # consumes env.actions in place
    wp.synchronize()


def main():
    p = argparse.ArgumentParser(description="RL attitude-control environment demo")
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--gain", type=float, default=8.0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    # Two envs, same seed -> identical initial tumble, fair comparison.
    env_zero = build_env(args.num_envs, args.device, seed=0)
    env_ctrl = build_env(args.num_envs, args.device, seed=0)
    print(f"{env_zero.num_envs} envs | obs_dim={env_zero.obs_dim} act_dim={env_zero.act_dim}")

    env_zero.reset()
    env_ctrl.reset()
    print(f"initial mean body rate    : {mean_rate(env_zero):.4f} rad/s")

    rollout(env_zero, args.steps, gain=0.0)

    t = time.perf_counter()
    rollout(env_ctrl, args.steps, gain=args.gain)
    dt = time.perf_counter() - t
    rate = args.steps * env_ctrl.num_envs / dt

    print(f"final mean rate (no control): {mean_rate(env_zero):.4f} rad/s")
    print(f"final mean rate (detumble)  : {mean_rate(env_ctrl):.4f} rad/s   (lower is better)")
    print(f"throughput                  : {rate/1e6:.1f} M env-steps/s")


if __name__ == "__main__":
    main()

# RL environments

[`VecEnv`](../reference/api/cosmic_warp/rl/env.md) wraps a
`SpacecraftSimulator` for reinforcement learning: persistent action,
observation and reward buffers on the device, termination, time-limit
truncation and automatic reset, all in one step with no host
synchronization. A task supplies the action mapping, observation and
reward.

The package ships the environment and the
[`Task`](../reference/api/cosmic_warp/rl/tasks.md) protocol, not tasks or
training algorithms. The reference `AttitudeTask` (reaction-wheel attitude
control) is in `examples/attitude_task.py`, shared by examples 13–16: copy
and adapt it. `VecEnv` imports no learning library; optional
[adapters](#trainer-adapters) connect rsl_rl, skrl and rl_games. The exact
step sequence is the [RL step contract](../theory/execution.md#the-rl-step-contract).

## Build an environment

Configure the simulator first (vehicle, actuators, bounds, randomization),
then wrap it; the task's `bind` runs in the `VecEnv` constructor and may
read that configuration.

```python
import warp as wp

from cosmic_warp.core.pipeline import SpacecraftSimulator
from cosmic_warp.rl import VecEnv

from attitude_task import AttitudeTask   # examples/attitude_task.py

sim = SpacecraftSimulator(
    num_envs=64, integrator="symplectic", dt=0.1,
    check_bounds=True, ang_mom_limit=5.0,      # spinning up past 5 kg*m^2/s terminates
    reset_config={"seed": 0},
)
sim.vehicle.set_config(mass_dry=12.0, inertia=[[0.4, 0, 0], [0, 0.4, 0], [0, 0, 0.3]])
sim.rw_manager.set_config(max_torque=0.02, max_momentum=1.0, wheel_inertia=0.005)
sim.reset_manager.set_bounds(
    att_max_angle=0.5236, ang_mom_bounds=((-0.2,) * 3, (0.2,) * 3), mass_bounds=(12.0, 12.0),
)

task = AttitudeTask(w_att=0.0, w_rate=1.0, w_ctrl=1.0e-3)
env = VecEnv(sim, task, max_episode_steps=200)

obs = env.reset()                                   # [64, 11] float32
action = wp.zeros((env.num_envs, env.act_dim), dtype=wp.float32, device=env.device)
for _ in range(10):
    obs, reward, terminated, truncated = env.step(action)
```

`examples/rl_attitude_training.yaml` declares the same setup for
[`create_simulator_from_yaml`](configuration.md).

## Step

`env.step(action)` copies `action` — float32, `[num_envs, act_dim]`, on the
environment's device, else `ValueError` — into `env.actions` and advances
every environment one step; `env.step()` uses `env.actions` as it is, for a
policy that writes it in place. It returns float32 `obs` and `reward` and
bool `terminated` and `truncated`, one row per environment.
`env.reset()` resets every environment and returns the first observation.

- **The returned arrays are the environment's buffers**, overwritten by the
  next step: copy what a rollout needs (the adapters do).
- **Ended environments are already reset**: where `terminated` or
  `truncated` is set, `obs` is the new episode's first observation and
  `env.terminal_obs` the state the step reached.
- **`terminated`**: the simulator's [termination](running.md#termination)
  ended the episode. **`truncated`**: it reached `max_episode_steps`. Never
  both.
- **A NaN or Inf command** ends its environment with reason 6, state
  unchanged. A task that clamps actions must pass a non-finite action on to
  the commands (`wp.clamp` makes it finite), as `AttitudeTask` does.

For value targets, bootstrap from `obs` on a continuing transition, from
`terminal_obs` on a truncated one, and not at all on a terminated one. The
skrl adapter's agent does exactly this; the rsl_rl and rl_games adapters
bootstrap a truncation from the value of the pre-step observation, the usual
on-policy approximation.

## Write a task

A task sets `act_dim` and `obs_dim` in `bind` and implements three
device-side methods:

| Method | Called | Writes |
| --- | --- | --- |
| `bind(env)` | once, by the `VecEnv` constructor | `act_dim`, `obs_dim`, cached handles |
| `ingest_actions(action)` | first in every step | actuator commands from the f32 actions `[N, act_dim]` |
| `write_reward(reward, action)` | after physics, before reset | f32 `reward` `[N]` |
| `write_obs(obs, reset_env, masked)` | after physics, and again after reset | f32 `obs` `[N, obs_dim]`; with `masked == 1`, only rows where `reset_env` is set |

Termination and truncation are not task concerns. Every method runs inside
the step and its CUDA graph, so it may only launch kernels: no host reads,
no Python branching on device data.

`AttitudeTask`: action `[N, 3·B]` in `[-1, 1]`, wheel torques for wheels
0–2 scaled by `max_torque`; observation per body
`[quat(4), angvel(3), rw_ang_mom(4)]` in raw units (normalization is the
trainer's); reward `-(w_att·angle² + w_rate·|ω|² + w_ctrl·|a|²)` summed over
bodies, `a` the unclamped action. `w_att=0, w_rate=1` is detumble; adding
`w_att` makes it repointing. Widths follow `num_bodies`.

## Torch views

Every buffer has a zero-copy torch view (`wp.to_torch`, same memory; needs
the `torch` extra): `observations_torch()`, `actions_torch()`,
`rewards_torch()`, `terminated_torch()`, `truncated_torch()`,
`terminal_observations_torch()`.

```python
import torch

policy = torch.nn.Linear(env.obs_dim, env.act_dim, device=env.observations_torch().device)

with torch.no_grad():
    env.actions_torch().copy_(torch.tanh(policy(env.observations_torch())))
env.step()                                          # reads env.actions in place
```

A policy can also be a Warp kernel writing `env.actions`, keeping the loop
in Warp without torch: [example 13](../examples/13_rl_attitude_env.md) does
this and prints the throughput on your GPU.

## CUDA graphs

On CUDA the first `step()` captures the step and later steps replay it.
After changing a task setting passed by value (a reward weight, say) or
enabling randomization, call `env.invalidate_graph()`; array contents,
including `set_config` values, stay live
([what the graph fixes](../theory/execution.md#invariants)). Keep host reads
(`.numpy()`, `.item()`, Python branching on device data) out of the loop:
each stalls the GPU.

## Trainer adapters

Each adapter wraps a `VecEnv` in its trainer's interface
([interfaces](../theory/execution.md#trainer-adapters)) and has its own
extra ([versions](../getting-started.md#install)); `vec_env` returns the
wrapped `VecEnv`. **The adapters support one body per environment** and
raise `NotImplementedError` otherwise; `VecEnv` itself is multi-body.

=== "rsl_rl"

    ```bash
    pip install "cosmic-warp[rsl-rl]"
    ```

    ```python
    from cosmic_warp.rl.adapters.rsl_rl import RslRlVecEnv, default_ppo_cfg
    from rsl_rl.runners import OnPolicyRunner

    rsl_env = RslRlVecEnv(VecEnv(sim, task, max_episode_steps=200))
    runner = OnPolicyRunner(rsl_env, default_ppo_cfg(num_steps_per_env=24),
                            log_dir=None, device=rsl_env.device)
    runner.learn(num_learning_iterations=2)   # 100+ to learn
    ```

    Evaluate deterministically (the action mean):

    ```python
    policy = runner.get_inference_policy(device=rsl_env.device)
    obs, _ = rsl_env.reset()
    with torch.inference_mode():
        for _ in range(10):
            obs, _, _, _ = rsl_env.step(policy(obs))
    ```

=== "skrl"

    ```bash
    pip install "cosmic-warp[skrl]"
    ```

    ```python
    from cosmic_warp.rl.adapters.skrl import SkrlVecEnv, build_ppo_agent
    from skrl.trainers.torch import SequentialTrainer

    skrl_env = SkrlVecEnv(VecEnv(sim, task, max_episode_steps=200))
    agent = build_ppo_agent(skrl_env, rollouts=24)
    agent.init()
    SequentialTrainer(env=skrl_env, agents=agent,
                      cfg={"timesteps": 2 * 24, "headless": True}).train()   # 24 * 100+ to learn
    ```

    The terminal observations are in `infos["terminal_observations"]`.
    `build_ppo_agent` builds a
    Gaussian-policy `TerminalObsPPO` agent: skrl's PPO, but bootstrapping a
    truncated transition from the terminal observation rather than the next
    episode's first one. Build your own agent on `TerminalObsPPO` to keep
    that.

    Evaluate deterministically (the Gaussian mean):

    ```python
    obs, _ = skrl_env.reset()
    with torch.inference_mode():
        for _ in range(10):
            actions = agent.act(obs, None, timestep=0, timesteps=1)[1]["mean_actions"]
            obs, _, _, _, _ = skrl_env.step(actions)
    ```

=== "rl_games"

    ```bash
    pip install "cosmic-warp[rl-games]"
    ```

    ```python
    import tempfile

    from cosmic_warp.rl.adapters.rl_games import (
        RlGamesVecEnv, register_env, default_ppo_params,
    )
    from rl_games.torch_runner import Runner

    rlg_env = RlGamesVecEnv(VecEnv(sim, task, max_episode_steps=200))
    register_env("cw_attitude", rlg_env)
    train_dir = tempfile.mkdtemp()
    params = default_ppo_params("cw_attitude", sim.num_envs, max_epochs=2,   # 100+ to learn
                                device=str(sim.device), train_dir=train_dir)
    runner = Runner()
    runner.load(params)
    runner.run_train({})
    ```

    `default_ppo_params` defaults to `device="cuda:0"` and writes
    checkpoints under `train_dir` (rl_games' `runs/` if omitted).

    Evaluate with rl_games' player, deterministically, from the last
    checkpoint:

    ```python
    from pathlib import Path

    player = runner.create_player()
    player.restore(str(next(Path(train_dir).rglob("last_*.pth"))))
    player.has_batch_dimension = True                  # obs is [N, obs_dim]
    obs = rlg_env.reset()["obs"]
    with torch.inference_mode():
        for _ in range(10):
            obs = rlg_env.step(player.get_action(obs, is_deterministic=True))[0]["obs"]
    ```

The `default_*` factories are starting points, not tuned recipes. The
examples train the detumble task with each trainer
([rsl_rl](../examples/14_rl_rsl_training.md),
[skrl](../examples/15_rl_skrl_training.md),
[rl_games](../examples/16_rl_rlgames_training.md)); the slow tests require
each to cut the mean tumble rate by at least 40 %
([validation](../validation/learning.md#trainer-adapters)).

Single-env `gym.Env` and Stable-Baselines3 are not supported: both force a
host copy every step.

## First-order training

The adapters are zero-order: the trainer never sees inside the simulator. A
policy can instead be trained with gradients through the
[differentiable](differentiable.md) dynamics:
[Differentiable RL](differentiable-rl.md).

API: [`cosmic_warp.rl`](../reference/api/cosmic_warp/rl/index.md) ·
[adapters](../reference/api/cosmic_warp/rl/adapters/index.md)

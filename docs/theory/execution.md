# GPU execution and the RL step

Every step runs on the device: one fused kernel integrates every entity and
checks its termination, and on CUDA `simulate()` and `VecEnv.step()` replay
the step as a CUDA graph, with no host round trip.

## The step on the device

![One simulate() step: the step kernel, the multi-body reduction, the optional trajectory store and auto-reset, and the trajectory index. The step kernel, per entity: episode counter, terminated check, thruster wrench, integration stages (forces, torques, wheel momenta), store, termination check](../assets/step-light.svg#only-light)
![One simulate() step: the step kernel, the multi-body reduction, the optional trajectory store and auto-reset, and the trajectory index. The step kernel, per entity: episode counter, terminated check, thruster wrench, integration stages (forces, torques, wheel momenta), store, termination check](../assets/step-dark.svg#only-dark)

/// caption
One `simulate()` step, and what each thread of the step kernel does for its
entity, in the same order in all three integrators. Dashed kernels run only
when their option is set; orange marks termination and reset.
`tests/test_diagram.py` fails if this order stops matching the code.
///

`sim.step()` is `Integrator.step()`: one kernel over all `num_entities`
entities. Each thread:

1. increments its entity's `episode_steps` counter;
2. returns at once if the entity is terminated (its state stays frozen);
3. evaluates the thruster wrench, forces, torques and wheel momenta, and
   performs the integrator update ([integrators](integrators.md));
4. writes the new state and wheel momenta;
5. checks the new state against the [termination](#termination) conditions,
   flagging only its own entity.

With several bodies per environment, a second kernel (one thread per
environment) reduces the flags: one launch per step, two for multi-body.

`simulate(num_steps, ...)` runs the step, the optional trajectory stores
(`save_states`, `save_rw_states`), the optional reset of terminated
environments (`auto_reset_terminated`) and a one-thread kernel advancing the
trajectory index. On CUDA this sequence is captured once per call and
replayed `num_steps` times; on CPU it runs eagerly. `simulate_monte_carlo()`
does the same with the bare step; `sim.step()` always launches eagerly.
Replay matches eager stepping, and a capture with every per-step feature on
succeeds, so the step makes no host synchronization
([validation](../validation/execution.md#cuda-graphs)).

| Baked into a graph | Read live from device arrays |
| --- | --- |
| `dt`, `num_bodies`, the kernels in the sequence (store, auto-reset, randomization on or off), the addresses of the arrays | state, wheel momenta, commands, vehicle and actuator parameters, physics flags, termination limits, reset and randomization bounds, RNG states |

`simulate()` captures afresh on every call, so configuration changes between
calls take effect; within a call commands are constant. `VecEnv` keeps its
graph across steps ([RL step](#the-rl-step-contract)). Throughput:
[Benchmarks](../benchmarks.md).

## Termination

Termination is per **environment**. The kernel checks each body's new state
in this order and records the first condition met:

<!-- generated: termination-reasons -->

The limits are constructor arguments, with the defaults in the
[`termination` table](../guides/configuration.md#termination).
`mom_limit` and `ang_mom_limit` bound the linear momentum and the body-only
angular momentum, not velocities; `mass_limit = m_dry` ends the episode at
burnout.

- **Non-finite states** end the environment with reason 6, whatever
  `check_bounds` says: a NaN passes every comparison and would spread
  silently. So does a NaN or Inf actuator command, which the clamps would
  turn into a finite actuation; that entity skips the step, state unchanged.
- **Every body completes the terminating step.** A thread flags only its own
  entity. After all bodies finish, the reduction flags every body of the
  environment and gives them the **lowest** reason code met, independent of
  body order. Saved trajectories and RL terminal observations contain the
  full terminating step.
- **Frozen until reset.** A terminated entity skips integration; its
  `episode_steps` still counts attempted steps. `get_terminated()` returns
  the `TerminationStatus` structs (`terminated`, `reason`) on the device;
  `termination_checker.get_termination_reasons()` reads them to the host as
  `{env: code}`.
- **Counters.** Every reset path zeroes `episode_steps` (per entity, on the
  device) for what it resets; `sim.step_count` is a host counter zeroed only
  by a full reset.
- **Partial resets of a terminated environment raise**: siblings left
  terminated would re-terminate it on the next step.

```python
import numpy as np
import warp as wp
from cosmic_warp.core.pipeline import SpacecraftSimulator

sim = SpacecraftSimulator(num_envs=3, num_bodies=2, dt=1.0, ang_mom_limit=1.0)
sim.reset()

ang_mom = np.zeros((sim.num_entities, 3))
ang_mom[1 * sim.num_bodies + 1] = [2.0, 0.0, 0.0]     # env 1, body 1: over the limit
sim.reset_to_conditions(
    angular_momenta=wp.array(ang_mom, dtype=wp.vec3d, device=sim.device))
sim.simulate(num_steps=5)

terminated = sim.get_terminated().numpy()["terminated"].reshape(3, 2)
assert terminated.tolist() == [[False, False], [True, True], [False, False]]
assert sim.termination_checker.get_termination_reasons() == {1: 3}

try:
    sim.reset(env_ids=[1], body_ids=[1])               # half of a terminated env
    raise AssertionError("expected ValueError")
except ValueError:
    pass
sim.reset(env_ids=[1])                                  # the whole env
assert not sim.get_terminated().numpy()["terminated"].any()
```

Differentiable rollouts need the limit checks off; the non-finite check
still applies ([differentiable dynamics](differentiable.md#smoothness-caveats)).

## Resets and random streams

| Path | What it does | Host work |
| --- | --- | --- |
| `reset(env_ids=…, body_ids=…, entity_ids=…)` | samples the state ([sampling](sampling.md)), redraws randomized parameters, clears the status, zeroes commands and wheel momenta | checks the bounds (mass, perigee) and partial resets |
| `reset_to_conditions(...)` | writes the given state (defaults: bound centres, identity attitude); no randomization; otherwise as `reset()` | the same checks |
| `reset_terminated()`, `simulate(auto_reset_terminated=True)` | the `reset()` kernels, masked by the termination status | none on the device: graph-safe (a host check of the configured inertia, skipped while capturing) |

All three zero `episode_steps`. After an automatic reset inside
`simulate()`, the reset entities keep zero commands until the call returns.
Draw counts live in device memory and advance with every reset, so resets
inside a replayed graph give fresh, reproducible samples
([random streams](sampling.md#random-streams)).

## The RL step contract

[`VecEnv`](../reference/api/cosmic_warp/rl/env.md) runs its whole step on the
device. Usage is in the [RL guide](../guides/rl.md); this is the contract
the adapters and custom trainers rely on.

### Invariants

- **No host synchronization inside `step()`**: no `.numpy()`, no Python
  branch on device data.
- **Physics stays float64**; only actions, observations and rewards are
  float32.
- **Persistent buffers**, allocated once and written in place.
- **One captured graph, replayed once per step.** It bakes in scalars passed
  by value (`max_episode_steps`, task parameters) and whether randomization
  is on: call `env.invalidate_graph()` after changing them (setting
  `env.max_episode_steps` does; on CPU changes apply at once). Arrays are
  read live. `step()` also makes a newly configured inertia the nominal that
  inertia randomization scales (a host-side version check before replay).
- **Framework-neutral core**: torch only for the zero-copy views
  (`env.observations_torch()`, …, same memory as `env.obs`, …); trainer
  specifics in the adapters.
- **Single-buffered**: every buffer is overwritten each step, so a trainer
  copies what it keeps.

### Indexing and shapes

`N = num_envs`, `B = num_bodies`, `E = N·B` entities. RL buffers are **per
environment**, the body dimension flattened in entity order.

| Buffer | Shape | dtype | Notes |
| --- | --- | --- | --- |
| `actions` | `[N, act_dim]` | f32 | normalized, `[-1, 1]` by convention |
| `obs` | `[N, obs_dim]` | f32 | raw physical units |
| `terminal_obs` | `[N, obs_dim]` | f32 | valid only where done |
| `reward` | `[N]` | f32 | task units |
| `terminated` | `[N]` | bool | the simulator terminated the environment |
| `truncated` | `[N]` | bool | time limit; `True` only when not terminated |
| `reset_env` | `[N]` | bool | internal: per-environment done |
| `done_mask` | `[E]` | bool | internal: per-entity done, drives the resets |

### The step sequence

`VecEnv.step(action)` copies `action` (if given) into `actions`, then runs
`VecEnv._step_body`, the captured graph:

1. **Ingest**: `task.ingest_actions` writes f64 actuator commands.
2. **Physics**: `sim.integrator.step`, as [above](#the-step-on-the-device)
   (`sim.step_count` does not advance).
3. **Reward**: `task.write_reward`, from the post-step, pre-reset state.
4. **Done**: termination status and step counters become per-environment
   `terminated` and `truncated` (`episode_steps >= max_episode_steps`) and
   the masks; the first body of each environment carries its flags. If a
   step both terminates and hits the limit, termination wins.
5. **Observation**: `task.write_obs` for every environment, pre-reset.
6. **Terminal snapshot**: `terminal_obs` ← `obs`.
7. **Auto-reset** of every entity in `done_mask`, on the device: sampling,
   randomization, zeroed commands, wheel momenta, status and
   `episode_steps`. It keys off the done mask, so **truncations reset too**
   (unlike `reset_terminated()`).
8. **Observation of the reset environments**: `task.write_obs`, masked.

`VecEnv.reset()` runs a full simulator reset and one observation write,
outside the graph. The returned `obs` is what the policy acts on next;
`terminal_obs` is the last observation of an episode that just ended, for
exact bootstrapping of truncations
([how each adapter uses it](../guides/rl.md#step)).

### The Task protocol

Termination and truncation are not task concerns; a task supplies the rest:

```python
from typing import Protocol

import warp as wp


class Task(Protocol):
    act_dim: int
    obs_dim: int

    def bind(self, env) -> None: ...
    def ingest_actions(self, action: wp.array) -> None: ...
    def write_obs(self, obs: wp.array, reset_env: wp.array, masked: int) -> None: ...
    def write_reward(self, reward: wp.array, action: wp.array) -> None: ...
```

`write_obs` with `masked == 1` writes only the rows where `reset_env` is
set. Tasks are application code; the reference
[`AttitudeTask`](../examples/attitude_task.md) is described in the
[RL guide](../guides/rl.md#write-a-task).

### Trainer adapters

The core returns `terminated`, `truncated` and `terminal_obs` separately,
so no adapter loses information. The adapters are thin and import their
trainer only when imported ([usage](../guides/rl.md#trainer-adapters)):

| Adapter | Tested with | Interface |
| --- | --- | --- |
| `adapters/rsl_rl.py` | rsl-rl-lib 5.5 | `TensorDict` `{"policy": …}`, `dones = terminated ∨ truncated`, `extras["time_outs"]`; observation cloned (rsl_rl keeps the reference across steps) |
| `adapters/skrl.py` | skrl 2.1 | skrl `Wrapper`, `terminated`, `truncated` and reward as `[N, 1]`; observation cloned; `build_ppo_agent` |
| `adapters/rl_games.py` | rl-games 1.6 | registered `IVecEnv`, `{"obs": t}`, combined `dones`, `infos["time_outs"]` with `value_bootstrap`; `register_env`, `default_ppo_params` |

A single-environment `gym.Env` is excluded on purpose: it forces a host
synchronization per environment.

### Differentiable path

[`DiffSimStepper`](../reference/api/cosmic_warp/rl/diff_bridge.md) exposes
one step as a `torch.autograd.Function`: flat float64 tensors in and out
(`X_DIM = 18` state, `U_DIM = 12` commands), the step's kernels recorded on a
per-call `wp.Tape` whose adjoint runs in the node's backward. Chained, it
gives the gradient through `obs_t → a_t → x_{t+1}`. The
[rollout constraints](../guides/differentiable.md#requirements-and-limits)
apply. A taped step runs on a copy of the configurations, which its backward
reuses, so resetting or reconfiguring the simulator before `backward()` does
not change the gradient.
[`DiffVecEnv`](../reference/api/cosmic_warp/rl/diff_env.md) chains it into
H-step windows with truncation-only episodes, the unit of example 17's SHAC
trainer ([differentiable RL](../guides/differentiable-rl.md)).

**Implementation:**
[`core/integrators/base.py`](../reference/api/cosmic_warp/core/integrators/base.md) ·
[`core/termination.py`](../reference/api/cosmic_warp/core/termination.md) ·
[`core/pipeline.py`](../reference/api/cosmic_warp/core/pipeline.md) ·
[`core/reset.py`](../reference/api/cosmic_warp/core/reset.md) ·
[`rl/env.py`](../reference/api/cosmic_warp/rl/env.md) ·
[`rl/diff_env.py`](../reference/api/cosmic_warp/rl/diff_env.md)

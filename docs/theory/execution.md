# GPU execution and the RL step

Every step runs on the device: one fused kernel integrates every entity and
checks its termination, and on CUDA `simulate()` and `VecEnv.step()` replay
the step as a CUDA graph, with no host round trip inside it.

## The step on the device

![One simulate() step: the step kernel, the multi-body reduction, the optional trajectory store and auto-reset, and the trajectory index. The step kernel, per entity: episode counter, terminated check, thruster wrench, integration stages (forces, torques, wheel momenta), store, termination check](../assets/step-light.svg#only-light)
![One simulate() step: the step kernel, the multi-body reduction, the optional trajectory store and auto-reset, and the trajectory index. The step kernel, per entity: episode counter, terminated check, thruster wrench, integration stages (forces, torques, wheel momenta), store, termination check](../assets/step-dark.svg#only-dark)

/// caption
One `simulate()` step, and the step kernel opened up: what each thread does
for its entity, in the same order in all three integrators. Dashed kernels
run only when their option is set; orange marks termination and reset.
`tests/test_diagram.py` fails if this order stops matching `simulate()`,
`Integrator.step()` or any of the three integrator kernels.
///

`sim.step()` is `Integrator.step()`, which launches one kernel over all
`num_entities` entities. Each thread:

1. increments its entity's `episode_steps` counter;
2. returns at once if the entity is terminated (its state stays frozen);
3. evaluates the thruster wrench, the forces and torques and the wheel
   momenta, and performs the integrator update ([integrators](integrators.md));
4. writes the new state and wheel momenta;
5. checks the new state against the termination conditions
   ([below](#termination)) and flags only its own entity.

With more than one body per environment a second kernel, one thread per
environment, then reduces the flags over the environment's bodies. A step is
therefore one launch, or two for multi-body simulators.

`simulate(num_steps, ...)` runs a fixed per-step sequence: the step, the
optional trajectory stores (`save_states`, `save_rw_states`), the optional
on-device reset of terminated environments (`auto_reset_terminated`), and a
one-thread kernel that advances the trajectory index. On CUDA the sequence
is captured into a CUDA graph once per `simulate()` call and replayed
`num_steps` times; on CPU the same kernels run eagerly. `simulate_monte_carlo()`
does the same with the bare step. A direct `sim.step()` call always launches
eagerly. Graph replay and eager stepping match (states to 1e-12 relative,
termination status and counters bitwise), and a capture with every per-step
feature switched on succeeds, which proves the step makes no host
synchronization ([Validation: execution](../validation/execution.md#cuda-graphs)).

| Baked into a graph | Read live from device arrays |
| --- | --- |
| `dt`, `num_bodies`, the kernels in the sequence (store, auto-reset, randomization on or off), the addresses of the arrays | state, wheel momenta, commands, vehicle and actuator parameters, physics flags, termination limits, reset and randomization bounds, RNG states |

`simulate()` captures afresh on every call, so a configuration change
between calls always takes effect; within one call nothing returns to
Python, so commands are constant over the call. `VecEnv` keeps its graph
across steps, which is where the left column matters
([RL step](#the-rl-step-contract)).

The kernels are float64 throughout; on consumer GPUs FP64 runs at 1/64 of
the FP32 rate and bounds the step. Throughput per integrator is on the
[Benchmarks](../benchmarks.md) page.

## Termination

Termination is per **environment**. The kernel checks each body's new
state, in this order, and records the first condition it meets:

<!-- generated: termination-reasons -->

The limits are constructor arguments (`pos_limit=1e8`, `mom_limit=inf`,
`ang_mom_limit=inf`, `mass_limit=0`, `planet_radius=RE_EARTH` by default;
`check_bounds` is on and `check_planet_impact` off by default) and the
[`termination` YAML section](../guides/configuration.md#termination).
`mom_limit` and `ang_mom_limit` apply to the linear momentum and the
body-only angular momentum, not to velocities; `mass_limit = m_dry` ends the
episode when the propellant runs out.

**Non-finite states.** The NaN/Inf check is always on, whatever
`check_bounds` says: every comparison with a NaN is false, so a NaN would
pass every limit above and spread silently. It runs in the step kernel, with
no host synchronization, and ends the environment with reason 6. The same
reason ends an environment with a NaN or Inf actuator command, which the
command clamps would otherwise turn into a finite actuation: that entity
skips the step, its state unchanged.

**Every body completes the terminating step.** A thread flags only its own
entity, so no body can see a sibling's flag in the middle of the step. Once
all bodies have finished, the reduction kernel flags every body of an
environment in which any body tripped and gives them all the **lowest**
reason code met by any of them, so the result does not depend on body
order. Saved trajectories and RL terminal observations therefore contain
the full terminating step.

**Frozen until reset.** A terminated entity skips integration on every later
step: its state and wheel momenta stay as they were, while its
`episode_steps` counter still counts attempted steps. `get_terminated()`
returns the per-entity `TerminationStatus` structs (`terminated`, `reason`)
on the device; `termination_checker.get_termination_reasons()` reads them to
the host as `{env: code}`, one entry per terminated environment.

**Counters.** `episode_steps` (per entity, on the device) is zeroed by every
reset path for the entities it resets, so per-environment episode lengths
stay right across partial and automatic resets. `sim.step_count` is a host
counter of steps taken, zeroed only by a full reset.

**Partial resets of terminated environments raise.** A reset that selects
only some bodies of a terminated environment would leave siblings that
re-terminate it on the next step, so `reset()` and `reset_to_conditions()`
reject it; reset the whole environment.

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

Differentiable rollouts require `check_bounds=False` and
`check_planet_impact=False`, since a frozen entity has no gradient path; the
non-finite check still applies ([differentiable dynamics](differentiable.md#smoothness-caveats)).

## Resets and random streams

| Path | What it does | Host work |
| --- | --- | --- |
| `reset(env_ids, body_ids, entity_ids)` | samples the state ([sampling](sampling.md)), redraws randomized parameters, clears the status, zeroes commands and wheel momenta | checks the bounds (reset mass not below dry mass, orbit perigee clear of the impact radius) and partial resets |
| `reset_to_conditions(...)` | writes the given state (defaults: bound centres, identity attitude), clears the status, zeroes commands and wheel momenta; no randomization | the same checks |
| `reset_terminated()`, `simulate(auto_reset_terminated=True)` | the `reset()` kernels, masked by the termination status | none on the device (a host-side check of the configured inertia, skipped while capturing): graph-safe |

All three zero `episode_steps` for the entities they reset. After an
automatic reset inside `simulate()`, the reset entities keep zero commands
until the call returns and new commands are set.

Sampling draws from counter-based per-entity streams whose draw counts live
in device memory and advance with every reset, so resets inside a replayed
graph give fresh, reproducible samples
([random streams](sampling.md#random-streams)).

## The RL step contract

[`VecEnv`](../reference/api/cosmic_warp/rl/env.md) wraps a simulator in a
vectorized environment whose whole step runs on the device. The usage is in
the [RL guide](../guides/rl.md); this section is the contract the trainer
adapters and custom trainers rely on.

### Invariants

- **No host synchronization inside `step()`.** No `.numpy()`, no Python
  branch on device data: a host round trip per step is what the design
  avoids.
- **Physics stays float64.** Only the boundary buffers (actions,
  observations, rewards) are float32.
- **Persistent buffers.** Every buffer is allocated once and written in
  place, which graph replay requires.
- **One captured graph, replayed once per step.** On CUDA the first
  `step()` captures the step and later calls replay it. The capture bakes in
  every scalar passed by value — `max_episode_steps`, task parameters such
  as reward weights — and whether domain randomization is enabled; after
  changing a task parameter or enabling randomization call
  `env.invalidate_graph()` (setting `env.max_episode_steps` calls it; on CPU
  the change applies at once). Arrays, including vehicle parameters,
  reset bounds and termination limits, are read live. `step()` also makes a
  new configured inertia the nominal that inertia randomization scales
  (a host-side version check before the replay).
- **Framework-neutral core.** `VecEnv` imports no learning framework; torch
  is needed only for the zero-copy views, and trainer specifics live in the
  adapters.

### Indexing and shapes

`N = num_envs`, `B = num_bodies`, `E = N·B` entities. RL buffers are **per
environment**; the body dimension is flattened in entity order.

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
this sequence (`VecEnv._step_body`, the captured graph):

1. **Ingest** — `task.ingest_actions`: f32 actions to f64 actuator commands,
   written into the simulator's command array.
2. **Physics** — `sim.integrator.step`: integrate, increment
   `episode_steps`, flag and reduce terminations, as in
   [the step on the device](#the-step-on-the-device). (`sim.step_count` is
   not advanced.)
3. **Reward** — `task.write_reward`, from the post-step, pre-reset state.
4. **Done** — one kernel turns the termination status and the step counters
   into per-environment `terminated` and `truncated`
   (`episode_steps >= max_episode_steps`) and the masks. Termination is per
   environment, so the first body of each environment carries its flags. If
   a step both terminates and reaches the time limit, termination wins and
   `truncated` stays `False`, so value bootstrapping treats it as terminal.
5. **Observation** — `task.write_obs` for every environment, post-step,
   pre-reset.
6. **Terminal snapshot** — `terminal_obs` ← `obs`.
7. **Auto-reset** — every entity in `done_mask` is reset on the device:
   state sampling, randomization if enabled, zeroed commands and wheel
   momenta, cleared status, zeroed `episode_steps`. Unlike
   `reset_terminated()`, it keys off the done mask, so **time-limit
   truncations reset too**.
8. **Observation of the reset environments** — `task.write_obs` again,
   writing only the rows of the reset environments.

`VecEnv.reset()` runs a full simulator reset and one observation write,
outside the graph.

**Terminal observations.** The returned `obs` is what the policy acts on
next (post-reset for done environments); `terminal_obs` holds the last
observation of the episode that just ended. Exact bootstrapping of a
truncated transition needs it. The skrl adapter returns it in `infos`, and
the `TerminalObsPPO` agent of `build_ppo_agent` bootstraps truncated
transitions from its value (with `time_limit_bootstrap`). The rsl_rl and
rl_games adapters do not read it: they bootstrap from the value already
computed for the pre-step observation (rsl_rl internally, rl_games with
`value_bootstrap`), a standard on-policy approximation. A custom trainer
can read `terminal_obs` for the exact terminal value.

### The Task protocol

`VecEnv` is generic; a task supplies the three environment-specific pieces.
Termination and truncation are not task concerns: they come from the
simulator's termination check and `max_episode_steps`.

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

`write_obs` with `masked == 1` must write only the rows where `reset_env` is
set. The package defines the protocol
([`rl/tasks.py`](../reference/api/cosmic_warp/rl/tasks.md)); tasks are
application code. The reference task, `AttitudeTask` in
[`examples/attitude_task.py`](../examples/attitude_task.md) (examples 13–16),
is reaction-wheel attitude control: action = torques for wheels 0–2 of each
body in `[-1, 1]`, clamped and scaled by `max_torque`; observation = per body
`[quat(4) with q_w ≥ 0, angular velocity(3), wheel momenta(4)]`; reward =
`-(w_att·angle² + w_rate·|ω|² + w_ctrl·|a|²)` summed over bodies, with `a`
the unclamped action. Observation normalization is left to the trainer.

### Zero-copy interop

The buffers are exposed natively (`env.obs`, `env.reward`, …) and as
zero-copy torch views (`env.observations_torch()`, `env.actions_torch()`, …)
built on `wp.to_torch` (same device memory). They are single-buffered and
overwritten every step: a trainer must copy each step's tensors into its own
storage, never keep a view expecting history.

### Trainer adapters

The core returns the superset signal — `terminated` and `truncated`
separately, plus `terminal_obs` — so no adapter loses information. The
adapters in `cosmic_warp.rl.adapters` are thin, import their trainer only
when imported, support one body per environment, and are each covered by a
test that skips unless the extra is installed and requires PPO to cut the
mean tumble rate of a detumble task by at least 40 %
([Validation: learning](../validation/learning.md#trainer-adapters)):

- **rsl_rl** (`adapters/rsl_rl.py`, tested with rsl-rl-lib 5.5) —
  observations as a `TensorDict` `{"policy": …}`,
  `dones = terminated ∨ truncated`, `extras["time_outs"] = truncated`. The
  observation is cloned, because rsl_rl keeps the reference across the next
  step.
- **skrl** (`adapters/skrl.py`, tested with skrl 2.1) — a skrl `Wrapper`;
  skrl already uses split `terminated`/`truncated`, reshaped to `[N, 1]`, and
  the observation is cloned for the same reason. `build_ppo_agent` assembles
  a PPO agent.
- **rl_games** (`adapters/rl_games.py`, tested with rl-games 1.6) — an
  `IVecEnv` registered with rl_games; observations as `{"obs": t}`, combined
  `dones`, `infos["time_outs"]` (with `value_bootstrap`). `register_env` and
  `default_ppo_params` do the wiring.

A single-environment `gym.Env` is excluded on purpose: it forces a host
synchronization per environment and defeats the graph.

### Differentiable path

[`DiffSimStepper`](../reference/api/cosmic_warp/rl/diff_bridge.md) exposes
one integrator step as a `torch.autograd.Function`: flat float64 tensors in
and out (`X_DIM = 18` state, `U_DIM = 12` command layouts), with the step's
kernels recorded on a per-call `wp.Tape` whose adjoint runs in the node's
backward. Chained under torch autograd it gives the gradient through
`obs_t → a_t → x_{t+1}`; it is checked against finite differences and
`torch.autograd.gradcheck`. The constraints are those of
[differentiable rollouts](differentiable.md): `differentiable=True`,
termination checks off, float64, no graph capture on taped steps. A taped
step runs on a copy of the vehicle, thruster and wheel configurations, which
its backward reuses, so resetting or reconfiguring the simulator before
`backward()` does not change the gradient.
[`DiffVecEnv`](../reference/api/cosmic_warp/rl/diff_env.md) chains it into
H-step windows with truncation-only episodes (`max_episode_steps` a multiple
of the horizon, every environment reset together), the unit of example 17's
SHAC trainer; see [differentiable RL](../guides/differentiable-rl.md).

### Cost

With the example 13 configuration (symplectic, gravity with J2 and gravity
gradient, three wheels, 4096 environments) the full RL step ran at about
42 M environment-steps/s on an RTX 4070 Ti SUPER, about 20 % more time per
step than `simulate()` on the same simulator (51 M); measured once on
2026-10-29, not part of the benchmark gate. The difference is the
ten launches of the RL step against two: action mapping, reward, done
reduction, two observation writes, the terminal copy and three masked reset
kernels, which run every step and test a device mask rather than branching
on the host.

**Implementation:**
[`core/integrators/base.py`](../reference/api/cosmic_warp/core/integrators/base.md) ·
[`core/termination.py`](../reference/api/cosmic_warp/core/termination.md) ·
[`core/pipeline.py`](../reference/api/cosmic_warp/core/pipeline.md) ·
[`core/reset.py`](../reference/api/cosmic_warp/core/reset.md) ·
[`rl/env.py`](../reference/api/cosmic_warp/rl/env.md) ·
[`rl/diff_env.py`](../reference/api/cosmic_warp/rl/diff_env.md)

# RL, randomization and autodiff

Gradients are checked against central finite differences, as relative
error against the larger magnitude. The forward RL tests use a test-local
`DetumbleTask` (`tests/rl_tasks.py`), against which the example
`AttitudeTask` is checked. Rows needing torch run in the
[RL job](index.md#ci-and-local-gpu-checks).

## Vectorized environment

`tests/test_rl_env.py`, `tests/test_termination.py`: `VecEnv` and the
[step contract](../theory/execution.md#the-rl-step-contract).

| Claim | Referent | Tolerance |
| --- | --- | --- |
| Observation and action sizes scale with bodies per environment | 11 and 3 per body | exact |
| Packed observation after one step | the simulator's getters, quaternion in the q_w ≥ 0 hemisphere | 1e-6 relative, 1e-7 absolute: the float32 cast (measured 2.9e-8) |
| Reward from the returned observation and the ingested action | −(w_rate \|ω\|² + w_ctrl \|a\|²) recomputed in NumPy | 1e-6 relative, 1e-7 absolute (float32) |
| Truncation | fires on exactly the step reaching `max_episode_steps`, not before; counters reset | exact |
| `terminal_obs` at truncation | a never-truncated twin's observation at that step; the live observation is the new episode | bitwise |
| Termination and truncation on the same step | terminated, not truncated (a true terminal for bootstrapping) | exact |
| Terminal observation and reward of a multi-body environment that terminates, all integrators | a twin with checks off, which completes the same step | 1e-7 relative; reset environment's counters and wheels exact |
| A tight angular-momentum limit | some environments terminate and auto-reset within 20 steps | exact |
| A +1 action | spins every wheel positive | exact |
| Torch views of observations, actions and terminal observations | share memory with the Warp buffers | exact |
| An inertia `set_config` between steps, with inertia randomization at scale 1 | the inertia after the next auto-reset (through the graph on CUDA) | bitwise |
| CUDA-graph replay of `step()`, 15 steps (*CUDA only*) | eager stepping | bitwise |
| `max_episode_steps` set after the first step | truncates at the new limit (the graph re-captured) | exact |
| A task weight changed after the first step | the old reward until `invalidate_graph()` on CUDA, the new one at once on the CPU; the new one after `invalidate_graph()` | bitwise |
| Auto-reset with propellant randomization (through the graph on CUDA) | masses redrawn inside the bounds | inside, and changed |
| A NaN action in one environment | that environment terminates, its terminal observation the unchanged state; the others step on | exact |
| Rebinding `actions`, `obs`, `terminal_obs`, `reward`, `terminated` or `truncated` | read-only | raises |
| Actions of the wrong shape or dtype; `max_episode_steps` not a positive integer or above 2³¹ − 1, at construction or set later | rejected, the limit unchanged | raises |

## Example attitude task

`tests/test_example_attitude_task.py`: `examples/attitude_task.py`, the task
of examples 13–16.

| Claim | Referent | Tolerance |
| --- | --- | --- |
| Reward is a penalty | ≤ 0 | 1e-6 |
| Reward from the returned observation | −(w_att θ² + w_rate \|ω\|² + w_ctrl \|a\|²), θ the rotation angle | 1e-3 relative, 1e-4 absolute |
| With w_att = 0, five steps of random actions | `DetumbleTask`'s observations and rewards | 1e-6 relative |
| A NaN or an Inf action | passed on to the commands: its environment terminates (a clamp would make it a finite torque) | exact |

## Trainer adapters

`tests/test_rl_rsl_adapter.py`, `tests/test_rl_skrl_adapter.py`,
`tests/test_rl_rlgames_adapter.py`, on the detumble task (12 kg, initial
angular momentum up to 0.2 N·m·s per axis, attitude within 30°).

| Claim | Referent | Tolerance |
| --- | --- | --- |
| `reset`/`step` return the shapes, keys and spaces each library expects, including the time-out flags | each library's interface | exact |
| Assigning rsl_rl's `episode_length_buf` | writes through to the simulator's episode counters | exact |
| skrl: the value input of the time-limit bootstrap, with half the environments truncating | `terminal_obs` on the truncated rows, the returned observation on the others; skrl's GAE ignores the last value of a truncated row | bitwise |
| rl_games: `Runner.create_player()` after `register_env`, default params | a player on the registered environment and device, whose deterministic actions have the action shape | exact |
| More than one body per environment | not supported | raises `NotImplementedError` |
| PPO learns to detumble: rsl_rl (256 envs, 150 iterations), skrl (256 envs, 2400 timesteps), rl_games (1024 envs, 100 epochs); seeded (*CUDA only*, `slow`) | mean body rate over the last 40 of 150 evaluation steps, before training | below 0.6× |

## Differentiable rollouts

`tests/test_gradients.py`: tape gradients of a terminal loss through 20
steps, all integrators ([theory](../theory/differentiable.md),
[[17]](../references.md#ref-17)).

| Claim | Referent | Tolerance |
| --- | --- | --- |
| Initial in-plane position and momentum, point-mass gravity | central finite differences | 1e-4 relative |
| Two entities on the same orbit | each other's gradient | 1e-12 relative |
| Initial position, momentum, quaternion and angular momentum with J2, drag at 300 km, gravity gradient, magnetic torque, a non-identity attitude and a tumble | central finite differences, each nonzero | 1e-4 relative |
| Thruster throttle and wheel torque at a given step | central finite differences | 1e-4 relative |
| Initial mass through a step where the propellant runs out 40 % in | central finite differences | 1e-4 relative |
| Vehicle parameters: C_D, drag area, inertia (including its precomputed inverse), magnetic dipole | central finite differences, each nonzero | 1e-4 relative |
| Thruster parameters: maximum thrust, Isp, position, direction; wheel inertia and viscous damping | central finite differences, each nonzero | 1e-4 relative |
| Newton steps on C_D from 1.2, 100-step rollout | the true C_D = 2.2 behind the observed final position | 1e-2 relative |
| 40 gradient-descent steps on the initial velocity towards a target 10 km along-track | the loss | below 1e-4 of the initial loss |
| A rollout never advances the simulator's episode counters | specified semantics | exact |
| A reset and an inertia `set_config` between `forward()` and `backward()`, held commands | the gradients without them | bitwise |
| A non-finite entity inside a rollout, all integrators | frozen at its first non-finite step and carried over unchanged (not zeroed); the other entity stays finite; the simulator's termination status untouched | bitwise |
| A rollout needs `differentiable=True`, no termination checks, a positive integer step count, and `reset()` between forward passes | rejected | raises |

## Torch bridge

`tests/test_rl_diff_env.py`: `DiffSimStepper`, one step as a torch autograd
node ([guide](../guides/differentiable-rl.md)).

| Claim | Referent | Tolerance |
| --- | --- | --- |
| `read_state` and `write_state` | the simulator's state and wheel arrays; round trip | 1e-7 relative; round trip to `torch.testing.assert_close` defaults |
| Four chained bridge steps, all integrators | the Warp-native differentiable rollout | 1e-13 relative |
| One step, all integrators | `torch.autograd.gradcheck` | eps 1e-6, atol 1e-5, rtol 1e-3 |
| Gradients through four chained steps, w.r.t. initial state and thruster command, all integrators | central finite differences | 1e-4 relative |
| Closed-loop gradient of a linear feedback policy, all 9 gains | central finite differences, and different from the open-loop gradient | 1e-4 relative; differs beyond 1e-3 |
| Backward twice with `retain_graph=True` | the first backward | `torch.testing.assert_close` defaults |
| Double backward | the step Jacobian is not a torch graph | raises |
| Taped steps | leave the simulator's state and counters untouched | bitwise |
| A non-finite state through a step | the simulator's termination status stays clear; the finite entity stays finite | exact |
| The untaped fast path (`no_grad` or detached inputs) | the taped step | bitwise |
| An inertia `set_config` between four taped RK4 steps and backward, with and without `frozen_params()` | the gradients without it | 1e-12 relative |
| A simulator without `differentiable=True` or with termination checks; float32 or wrongly shaped input | rejected | raises |

## Differentiable environment

`tests/test_rl_diff_env.py`, `tests/test_example_shac.py`: `DiffVecEnv`
windows, on a test-local task and `AttitudeDiffTask`.

| Claim | Referent | Tolerance |
| --- | --- | --- |
| Windows chain with the carried state detached; the time limit resets; simulator counters untouched | specified semantics | exact |
| `end_window()` without a pending `rollout()`, or after a reset; a horizon that is not a positive integer or does not divide the episode | rejected | raises |
| Gradient of a window that ends the episode, inertia randomized 0.2–5×, with `end_window()` before `backward()` | the same with `backward()` first | 1e-12 relative |
| Gradient of a window's return w.r.t. tanh-policy weights, both tasks | central finite differences | 1e-4 relative |
| `AttitudeDiffTask` with randomized inertia (it caches I⁻¹) | rejected | raises |

## SHAC example

`tests/test_example_shac.py`: `examples/shac.py`, the trainer of example 17.

| Claim | Referent | Tolerance |
| --- | --- | --- |
| `AttitudeDiffTask` (torch) | `AttitudeTask` (Warp kernels): observation; reward | 1e-5 relative, 1e-6 absolute; 1e-4 relative, 1e-6 absolute |
| Reward gradient exactly at the goal attitude | finite (an arccos form would give NaN) | finite |
| TD(λ) targets | the closed-form mixture of n-step returns | `torch.testing.assert_close` defaults |
| Three updates | finite statistics; actor parameters move | finite |
| Same seed; different seed | identical networks; different networks | bitwise |
| `save()` then `load()` into a differently seeded trainer | the policy output, observation statistics and update count | bitwise |
| Critic minibatches that leave some empty, or zero | rejected | raises |
| 30 updates on the detumble task, 256 envs, seeded (*CUDA only*, `slow`) | mean body rate over the last 40 of 150 evaluation steps, before training | below 0.6× |

## Domain randomization

`tests/test_randomization.py`, `tests/test_orbit_sampling.py`
([guide](../guides/randomization.md)).

| Claim | Referent | Tolerance |
| --- | --- | --- |
| Batched `set_config_per_entity` | the given per-entity values; the inverse inertia computed in the kernel; other fields keep their defaults | values and defaults bitwise; the inverse within 1e-15 relative (one ulp) |
| Dry mass, propellant and inertia scale | inside their bounds; dry mass actually dispersed | exact |
| Physical couplings | total mass = dry + propellant; inertia = one scale × each entity's own tensor; I I⁻¹ = identity | propellant inside its bounds; inertia 1e-15 relative; I I⁻¹ 1e-15 absolute |
| Inertia scaling after a per-env or later `set_config` inertia | scales that tensor, not a copy taken at `set_bounds` | 1e-7 relative |
| Every full reset redraws; a partial reset redraws only its environments | the previous draw | differs; others bitwise |
| Auto-reset inside `simulate()` redraws (through the graph on CUDA) | the bounds | inside, and changed |
| `reset_terminated()` after an inertia `set_config`, scale 1 | the new inertia and its inverse | inertia bitwise; inverse 1e-15 relative |
| YAML `randomization:` with per-body overrides, inertia from the body config, and validation | the loader | inside bounds; raises |
| Mass randomization after an orbit draw | rescales momentum, keeping the orbit's velocity | 1e-6 relative |

Bounds and couplings only; distributions are checked for reset states
([random streams](execution.md#random-streams-and-determinism)).

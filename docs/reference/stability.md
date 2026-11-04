# API stability

COSMIC WARP is **0.x**: breaking changes are allowed, never silent. This
page says what you can build against; every breaking change to it is listed
in the [changelog](changelog.md).

## Promised

A change here is a breaking change. Each name is promised at the import path
shown, nothing else about module layout. Arguments are promised by name, in
the order listed, with their defaults and meaning; those after a `*` are
keyword-only. `sim` is a `SpacecraftSimulator`, `env` a `VecEnv`, `rollout`
what `sim.differentiable_rollout()` returns, `stepper` a `DiffSimStepper`,
`diff_env` a `DiffVecEnv`. `tests/test_stability.py` checks every name,
argument list, field and key below against the code, and that everything
`cosmic_warp` and `cosmic_warp.rl` export is listed.

| Surface | Import from | What is promised |
| --- | --- | --- |
| `SpacecraftSimulator`, `__version__` | `cosmic_warp` | the simulator class, also importable from `cosmic_warp.core.pipeline`; the package version |
| `SpacecraftSimulator(*, num_envs, integrator, dt, device, num_bodies, gravity, j2_perturbation, gravity_gradient, magnetic_torque, drag, check_bounds, check_planet_impact, pos_limit, mom_limit, ang_mom_limit, mass_limit, planet_radius, reset_config, differentiable)` | `cosmic_warp.core.pipeline` | arguments, defaults and meaning; `ValueError` naming the argument for an invalid value |
| `sim` methods | — | `reset(*, env_ids, body_ids, entity_ids)`, `reset_to_conditions(*, positions, momenta, angular_momenta, quaternions, masses, env_ids, body_ids, entity_ids)`, `reset_terminated()`, `step()`, `simulate(num_steps, *, save_states, save_rw_states, auto_reset_terminated)`, `simulate_monte_carlo(num_steps)`, `differentiable_rollout(num_steps, *, commands)`, `metadata()`, `set_thrust_commands(thrust_cmd)`, `set_thrust_commands_per_entity(thrust_cmds)`, `set_rw_torque_commands(rw_torque_cmds)`, `set_rw_torque_commands_per_entity(rw_torque_cmds)`, `set_rw_angular_momentum_state(rw_ang_mom)`, `set_rw_angular_momentum_state_per_entity(rw_ang_mom)`, `get_positions()`, `get_velocities()`, `get_quaternions()`, `get_angular_velocities()`, `get_masses()`, `get_rw_angular_momenta()`, `get_terminated()`, `get_episode_steps()` — what each does, as the [guides](../guides/index.md) describe it: the reset paths and what they clear, the getters' shapes and reused buffers, commands holding until set again or reset, thrust commands as up to 8 throttles (one per thruster slot, omitted trailing ones zero) |
| `sim` attributes | — | `num_envs`, `num_bodies`, `num_entities`, `dt`, `device`, `differentiable`, `states`, `step_count`, `vehicle`, `thruster_manager`, `rw_manager`, `reset_manager`, `randomizer`, `termination_checker`, `actuators` — read-only (assign into the arrays, never rebind them); `states` is the `State` array, one row per entity, which you may `assign` to or write from a kernel |
| `simulate()` keys | — | `time`, `positions`, `velocities`, `quaternions`, `angular_velocities`, `masses`, `rw_angular_velocities` — dtypes, shapes and timing, as in [Run, record, terminate](../guides/running.md#record-a-trajectory) |
| `simulate_monte_carlo()` keys | — | `final_positions`, `final_velocities`, `final_quaternions`, `final_angular_velocities`, `final_masses`, `termination_status`, `sim_time` — as in [final states only](../guides/running.md#final-states-only) |
| `reset_config` keys | — | `pos_bounds`, `vel_bounds`, `mom_bounds`, `ang_mom_bounds`, `mass_bounds`, `att_nominal`, `att_max_angle`, `seed` — shapes and meaning, as for `sim.reset_manager.set_bounds` |
| `sim.vehicle` | — | `set_config(*, mass_dry, inertia, area_drag, cd, magnetic_dipole, env_ids, body_ids, entity_ids)`, `set_config_per_entity(*, mass_dry, inertia, area_drag, cd, magnetic_dipole)`, `configs` — `set_config` writes only the fields given (`None` = unchanged); the accepted values in [Vehicles](../guides/vehicle.md); `configs` is the `VehicleConfig` array your kernels may read |
| `sim.thruster_manager` | — | `set_config(*, max_thrust, isp, positions, directions, env_ids, body_ids, entity_ids)`, `configs` — `set_config` writes only the fields given (`None` = unchanged; `positions` and `directions` together); the layout rules and the default six-thruster layout in [Actuators](../guides/actuators.md#thrusters); `configs` is the `ThrusterConfig` array |
| `sim.rw_manager` | — | `set_config(*, max_torque, max_momentum, wheel_inertia, viscous_damping, spin_axes, env_ids, body_ids, entity_ids)`, `configs` — `set_config` writes only the fields given (`None` = unchanged); as in [Actuators](../guides/actuators.md#reaction-wheels); `configs` is the `ReactionWheelConfig` array |
| `sim.reset_manager` | — | `set_bounds(*, pos_bounds, vel_bounds, mom_bounds, ang_mom_bounds, mass_bounds, att_nominal, att_max_angle, body_ids)`, `set_orbit_bounds(*, semi_major_axis_bounds, altitude_bounds, eccentricity_bounds, inclination_bounds, raan_bounds, arg_periapsis_bounds, true_anomaly_bounds, mean_anomaly_bounds, body_ids)`, `set_sample_mode(mode, *, body_ids)`, `get_bounds(body_id)`, `reseed(seed)` — only the bounds you pass change; the default bounds; the sampling distributions of [Initial conditions and resets](../guides/reset.md); `get_bounds` returns a copy keyed by the setters' argument names, plus `sample_mode` |
| `sim.randomizer` | — | `set_bounds(*, mass_dry, propellant, inertia_scale, body_ids)`, `get_bounds(body_id)`, `reseed(seed)` — the bounds, couplings and reset paths of [Domain randomization](../guides/randomization.md) |
| `sim.termination_checker` | — | `get_termination_reasons()` — `{env: code}` for every terminated environment, read to the host |
| `sim.actuators` | — | `commands`, `states` — the `ActuatorCommand` and `ActuatorState` arrays, one row per entity, that a task's kernels write commands into and read wheel momenta from, in place |
| `State` | `cosmic_warp.core.state` | `pos`, `mom`, `mass`, `quat`, `ang_mom` — field names, types, frames and units on [Frames, state and units](../theory/conventions.md#state) |
| `VehicleConfig` | `cosmic_warp.core.vehicle` | `mass_dry`, `inertia`, `inertia_inv`, `area_drag`, `cd`, `magnetic_dipole` — field names, types and units |
| `ActuatorCommand` | `cosmic_warp.actuators.base` | `thrust_cmd`, `thrust_cmd_ext`, `rw_torque_cmd` — thrusters 0–3, thrusters 4–7 (throttle in [0, 1]) and wheels 0–3 (N·m) |
| `ActuatorState` | `cosmic_warp.actuators.base` | `rw_ang_mom` — wheel momenta 0–3 about their spin axes (kg·m²/s) |
| `ThrusterConfig` | `cosmic_warp.actuators.thrusters` | `positions`, `directions`, `max_thrust`, `isp` — field names, types and units |
| `ReactionWheelConfig` | `cosmic_warp.actuators.reaction_wheels` | `max_torque`, `max_momentum`, `wheel_inertia`, `viscous_damping`, `spin_axis_0`, `spin_axis_1`, `spin_axis_2`, `spin_axis_3` — field names, types and units; a zero axis is an unused wheel |
| `TerminationStatus` | `cosmic_warp.core.termination` | `terminated`, `reason` — the struct `get_terminated()` returns |
| `REASON_*`, `REASONS` | `cosmic_warp.core.termination` | the reason codes and their names, and the rules on [termination](../theory/execution.md#termination): per environment, the lowest code wins, frozen until reset, the non-finite check always on |
| `MU_EARTH`, `RE_EARTH`, `J2_EARTH` | `cosmic_warp.physics.gravity` | the Earth constants the dynamics use, in SI units |
| `elements_to_state(semi_major_axis, eccentricity, inclination, raan, arg_periapsis, true_anomaly, mu)`, `circular_orbit_state(altitude, inclination, raan, true_anomaly, mu, planet_radius)`, `state_to_elements(position, velocity, mu)` | `cosmic_warp.physics.orbits` | arguments, radians, ECI NumPy `(position, velocity)` in and out, and the keys of the dict `state_to_elements` returns |
| `create_simulator_from_yaml(path, *, device)`, `create_simulator_from_config(config)` | `cosmic_warp.core.config` | arguments; the returned simulator is already reset |
| YAML scenario | — | `simulation`, `physics`, `termination`, `reset`, `body`, `initial_conditions`, `randomization` — every key, default, unit and validation rule in [Configure a scenario](../guides/configuration.md); `tests/test_docs.py` checks that page against the loader |
| `command_sequence(sim, num_steps)` | `cosmic_warp.core.autodiff` | one zeroed `ActuatorCommand` array per step, with gradients |
| `DifferentiableRollout` | `cosmic_warp.core.autodiff` | the type of `sim.differentiable_rollout(...)`, for annotations; build it through the simulator |
| `rollout` | — | `tape`, `forward()`, `reset()`, `states`, `initial_states`, `final_states`, `vehicle_configs`, `thruster_configs`, `rw_configs`, `inertia_grad()`, `num_steps`, `commands` — and the gradient behaviour of [Differentiable rollouts](../guides/differentiable.md): the requirements, what is differentiated and the [smoothness caveats](../theory/differentiable.md#smoothness-caveats) |
| `VecEnv(sim, task, max_episode_steps)`, `Task` | `cosmic_warp.rl` | the environment and the task protocol, also importable from `cosmic_warp.rl.env` and `cosmic_warp.rl.tasks` |
| `env` | — | `step(action)`, `reset()`, `invalidate_graph()`, `observations_torch()`, `actions_torch()`, `rewards_torch()`, `terminated_torch()`, `truncated_torch()`, `terminal_observations_torch()`, `sim`, `task`, `num_envs`, `num_bodies`, `num_entities`, `device`, `obs_dim`, `act_dim`, `max_episode_steps`, `actions`, `obs`, `terminal_obs`, `reward`, `terminated`, `truncated` — the six buffers are read-only (write into them, never rebind them), and setting `max_episode_steps` re-captures the step graph; buffer shapes and dtypes, the step sequence, terminal and auto-reset semantics and what the CUDA graph fixes, in [the RL step contract](../theory/execution.md#the-rl-step-contract) |
| `Task` protocol | — | `act_dim`, `obs_dim`, `bind(env)`, `ingest_actions(action)`, `write_obs(obs, reset_env, masked)`, `write_reward(reward, action)` — when each is called and what it writes, in [Write a task](../guides/rl.md#write-a-task) |
| `DiffSimStepper(sim)`, `X_DIM`, `U_DIM` | `cosmic_warp.rl.diff_bridge` | the flat float64 layouts, in this order: the state `[E, X_DIM]` is `pos`, `mom`, `mass`, `quat` (x, y, z, w), `ang_mom`, `rw_ang_mom`; the command `[E, U_DIM]` is the 8 thruster throttles (slots 0–7), then the 4 wheel torques, in physical units |
| `stepper` | — | `step(x, u)`, `read_state()`, `write_state(x)`, `frozen_params()` — `step` never changes the simulator, and its backward uses the configuration of its forward; `write_state` copies a state into it; first-order gradients only |
| `DiffTask` | `cosmic_warp.rl.diff_tasks` | the differentiable task protocol |
| `DiffTask` protocol | — | `act_dim`, `obs_dim`, `bind(sim)`, `commands(action)`, `observation(x)`, `reward(x, action)` — torch functions of the state, in float64 |
| `DiffVecEnv(sim, task, horizon, max_episode_steps)` | `cosmic_warp.rl.diff_env` | windowed, truncation-only episodes, as in [Differentiable RL](../guides/differentiable-rl.md#windows) |
| `diff_env` | — | `reset()`, `rollout(policy_fn)`, `end_window()`, `state`, `sim`, `task`, `stepper`, `horizon`, `max_episode_steps`, `num_envs`, `obs_dim`, `act_dim`, `device` |
| `Window` | `cosmic_warp.rl.diff_env` | `obs`, `rewards` — what `rollout()` returns: `horizon + 1` observations and `horizon` rewards, on the autograd graph |
| `RslRlVecEnv(env)`, `default_ppo_cfg(num_steps_per_env, hidden_dims, learning_rate, entropy_coef)` | `cosmic_warp.rl.adapters.rsl_rl` | the wrapper, its `vec_env` property and the mapping of `terminated`, `truncated` and observations in [Trainer adapters](../guides/rl.md#trainer-adapters); the factory's argument names |
| `SkrlVecEnv(env)`, `build_ppo_agent(env, rollouts, hidden, learning_rate, entropy_loss_scale)`, `TerminalObsPPO`, `TERMINAL_OBS_KEY` | `cosmic_warp.rl.adapters.skrl` | the same, for skrl; the terminal observations in `infos[TERMINAL_OBS_KEY]`, and `TerminalObsPPO`, skrl's `PPO` bootstrapping truncated transitions from them |
| `RlGamesVecEnv(env)`, `register_env(name, env)`, `default_ppo_params(env_name, num_envs, horizon_length, num_minibatches, units, learning_rate, max_epochs, device, train_dir)` | `cosmic_warp.rl.adapters.rl_games` | the same, for rl_games |
| Conventions | — | everything on [Frames, state and units](../theory/conventions.md): SI units and radians everywhere, ECI and body frames, quaternions stored scalar-last and rotating body to inertial, momenta rather than velocities in the state, environment-major entity order, float64 physics |

- Struct fields are promised by name, type and unit; structs may gain
  fields and their order is not promised, so use names.
- The `default_*` factories and `build_ppo_agent` promise their arguments,
  not their hyperparameter values.
- The adapters follow the trainer versions in the
  [extras table](../getting-started.md#install); a trainer's breaking change
  can force an adapter change, listed in the changelog.
- Results are reproducible within a release on the same machine and
  versions, not bit for bit across releases: a model fix or a change to the
  [random streams](../theory/sampling.md#random-streams) changes them, with
  a changelog entry.

## Internal

No notice, no changelog entry:

- module paths and names not listed above (`allocate_state`,
  `SimulationFlags`, `compute_forces`, `IntegratorFactory`, ...);
- `sim.ctx` and the `SimContext` behind it: no guide reads it, and the
  integrator interface it feeds is expected to change before 1.0
  ([Extending](../guides/extending.md#known-limitations));
- `sim.integrator` and the integrator classes, `sim.flags` (read
  `metadata()`), `sim.episode_steps` (read `get_episode_steps()`) and
  `sim.termination_mask`;
- the manager classes' names and constructors (reach them through `sim`),
  and their methods not listed above: `reset_all`, `reset_masked`,
  `build_mask`, `randomize_all`, `sync_nominal`, `reset_status` and the
  like, their `rng_state` arrays, and the `ActuatorManager` setters, which
  `sim` wraps;
- the configuration structs not listed above (`TerminationConditions`,
  `ResetConfig`, `RandomizationConfig`, as `sim.randomizer.config`) and the
  fields not listed (`ThrusterConfig.active`); the termination limits are
  fixed at construction;
- the kernels and `@wp.func` stages of `cosmic_warp.physics` and
  `cosmic_warp.actuators` (`gravity_accel_j2`, `thruster_wrench`,
  `elements_to_rv`, ...) — callable from your kernels, but their signatures
  follow the physics;
- `cosmic_warp.core.validation`, `cosmic_warp.core.entities` and
  `cosmic_warp.core.flags`;
- `VecEnv`'s `reset_env` and `done_mask` masks; the rollout's `act_states`;
  the buffers inside `DiffSimStepper`;
- everything in `examples/`: `AttitudeTask`, `AttitudeDiffTask`, `SHAC`,
  `default_shac_cfg`, the scripts and the scenario files. They are
  application code to copy and adapt, not library API; what they use from
  the library is promised only where listed above;
- performance: step rates, memory use and launch counts, on
  [Benchmarks](../benchmarks.md) or elsewhere, are measurements, not
  promises;
- test fixtures, `tests/rl_tasks.py` and benchmark baselines.

## What 1.0 means

From 1.0 the promised surface changes only with a major version. 1.0 is not
scheduled: it comes once the external force and torque input and the
custom-integrator interface ([roadmap](../roadmap.md)) have landed,
COSMIC WARP has outside users, and a release passes without a breaking
change.

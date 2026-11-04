# Guides

Task-focused pages, each building on [Getting started](../getting-started.md).
The conventions every page uses (environments, bodies, entities, frames and
units) are on [Frames, state and units](../theory/conventions.md).

- [Configure a scenario (YAML)](configuration.md) — a whole scenario in one file.
- [Vehicles](vehicle.md) — mass, inertia, drag and magnetic properties, per body.
- [Actuators](actuators.md) — thrusters and reaction wheels: configuration and commands.
- [Initial conditions and resets](reset.md) — exact initial states, sampled resets, partial resets.
- [Domain randomization](randomization.md) — per-reset draws of vehicle parameters.
- [Physics models](physics.md) — switching gravity, J2, drag and the torques on and off.
- [Integrators](integrators.md) — choosing Euler, RK4 or the symplectic scheme.
- [Run, record, terminate](running.md) — stepping, trajectories, reading state, termination limits, provenance.
- [Scale up](scaling.md) — batch size, CUDA graphs, keeping data on the device.
- [RL environments](rl.md) — `VecEnv`, the `Task` protocol, torch views and the trainer adapters.
- [Differentiable rollouts](differentiable.md) — gradients through the dynamics.
- [Differentiable RL](differentiable-rl.md) — `DiffVecEnv` windows and first-order training (SHAC).
- [Extending](extending.md) — what extends through the API, kernel conventions, contributing.

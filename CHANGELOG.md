# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [semantic versioning](https://semver.org/).

## [Unreleased]

### Documentation

- Citation: Zenodo DOI [10.5281/zenodo.23950779](https://doi.org/10.5281/zenodo.23950779)
  (all versions; 0.1.0 is 10.5281/zenodo.23950780) in `CITATION.cff`, the
  README badge and the BibTeX entries.
- Shortened the README and the documentation site (prose by about a third):
  tables and lists in place of paragraphs, each caveat stated once next to
  the claim it qualifies, status kept to this changelog and the roadmap.
  Every tested claim, number and link stays.
- Profiling findings (the float64 limit, register spills, the RL step's
  cost) moved to an experiment record, `docs/data/README.md`, outside the
  site.
- Getting started explains installing where the system Python is older than
  3.12, with conda or uv.
- Fixed stale details: keyword-only `commands=` and reset selections in the
  theory pages; validation rows name the `*_per_entity` setters instead of
  "batched" ones.

## [0.1.0] - 2026-11-04

First public release, as a **beta**: the public API may change before 1.0,
and each breaking change will be listed here.

### Features

- GPU-parallel spacecraft dynamics on NVIDIA Warp: thousands to millions of
  independent environments, 6-DOF orbital and attitude dynamics in float64,
  with physics, integration and termination fused into one kernel per step
  and replayed as a CUDA graph by `simulate()`.
- Integrators: Euler, RK4 (default) and symplectic kick–drift–kick, each
  verified to reach its nominal order of convergence.
- Physics: point-mass gravity, J2, atmospheric drag, gravity-gradient and
  magnetic torques, variable mass.
- Actuators: up to 8 thrusters (Isp, propellant use) and 4 reaction wheels
  (momentum saturation, viscous friction) per body.
- Multi-body environments, configured per body from Python or YAML.
- Reset sampling (Cartesian boxes or orbital elements) and domain
  randomization on the GPU, from counter-based (Philox) per-entity random
  streams with 64-bit seeds; on-device termination, including non-finite
  states and actuator commands.
- Differentiable rollouts: reverse-mode gradients w.r.t. initial state,
  control sequences and vehicle parameters.
- Reinforcement learning: on-device vectorized environment with auto-reset
  and zero-copy torch views; adapters for rsl_rl, skrl and rl_games; a
  differentiable environment for first-order methods. A reference attitude
  task and a minimal SHAC trainer are in `examples/`.
- Seventeen examples, a YAML scenario format, a physics validation suite
  against analytic results, and a per-GPU performance regression benchmark.

### Known limitations

- No user force/torque input and no supported custom-integrator interface;
  both are planned for 1.x.
- Earth is the only central body: orbit specs take no `mu`.
- The integrators do not locate events: reaction-wheel saturation inside a
  step limits every integrator to second order around it, and propellant
  burnout is spread over its step (delta-v exact, position error
  O(dt²) once per burnout).
- On consumer GPUs the step is bound by FP64 throughput (1/64 of FP32 rate);
  GPUs with full-rate FP64 have not been measured.
- The RL trainer adapters support one body per environment.

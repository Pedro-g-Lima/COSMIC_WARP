# Roadmap

0.1 is a beta: the public API may change before 1.0, and every breaking
change is listed in the [changelog](reference/changelog.md).

## Planned for 1.x

- [ ] Per-entity, differentiable external force/torque input, so custom force
      models and disturbances need no changes to library code
- [ ] A supported interface for custom integrators
- [ ] Configurable central body (today the dynamics use Earth's `mu`, and
      orbit specs take no other)
- [ ] Multi-body support in the RL trainer adapters

## Further ideas

Done (checked) and candidate enhancements beyond 1.x:

- [x] J2 perturbations
- [x] Magnetic torques
- [x] Gravity gradient torques
- [x] Multiple spacecraft (multi-body environments)
- [x] Domain randomization (uniform, on-GPU, per body)
- [ ] Non-uniform randomization distributions
- [ ] Inertia/CoM variation with propellant mass
- [ ] Event location inside a step (reaction-wheel saturation, propellant
      burnout), restoring the integrators' full order around events
- [ ] Attitude-dependent drag area; higher-order geopotential terms (J3 and
      beyond)
- [ ] Simulation epoch & rotating tilted-dipole / IGRF magnetic field
- [ ] Multiple reference frames (ECI, ECEF, body, etc.) with conversions
- [ ] Configurable planetary systems (e.g. Earth-Moon, Sun-Earth,
      Sun-planet-moon) as the central-body hierarchy
- [ ] Per-body collision radius
- [ ] Third-body perturbations
- [x] Atmospheric drag (piecewise-exponential density, co-rotating atmosphere)
- [x] Differentiable dynamics (reverse-mode gradients w.r.t. initial state
      and control sequences; termination checks must be off)
- [ ] Gradient checkpointing & captured forward/backward CUDA graphs for
      long-horizon differentiable rollouts
- [x] Vehicle-parameter gradients (system identification)
- [ ] Solar radiation pressure
- [ ] Fuel slosh
- [ ] N-body gravitational interactions
- [x] RL support — GPU-resident vectorized environment (`cosmic_warp.rl`):
      fully on-device fused step (action → physics → reward → done →
      observation → auto-reset), zero-copy torch views; trainer adapters for
      rsl_rl, skrl and rl_games (single-body); differentiable-RL path
      (warp↔torch autograd bridge, windowed differentiable environment). A
      reference attitude task and a minimal SHAC trainer live in `examples/`
      (tasks and training algorithms are not part of the package).
- [x] Non-finite state detection on device (terminates the environment)
- [ ] Sensor models
- [ ] Additional solvers
- [ ] Visualization tools
- [ ] Data export utilities
- [ ] Performance profiling tools

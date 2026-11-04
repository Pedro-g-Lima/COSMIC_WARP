# Roadmap

What has shipped is in the [changelog](reference/changelog.md); what may
change before 1.0, in [API stability](reference/stability.md).

## Planned for 1.x

- Per-entity, differentiable external force and torque input, so custom
  force models and disturbances need no library changes
- A supported interface for custom integrators
- A configurable central body (today Earth's `mu` only)
- Multi-body support in the RL trainer adapters

## Ideas

- Physics: solar radiation pressure, third-body perturbations, J3 and
  higher harmonics, attitude-dependent drag area, a simulation epoch with a
  rotating tilted-dipole or IGRF field, N-body gravity, fuel slosh
- Planetary systems (Earth–Moon, Sun–Earth, …) as a central-body hierarchy;
  reference frames (ECI, ECEF, body) with conversions
- Inertia and centre-of-mass variation with propellant; per-body collision
  radius
- Event location inside a step (wheel saturation, burnout), restoring the
  integrators' full order around events; additional solvers
- Gradient checkpointing and captured forward/backward graphs for long
  differentiable rollouts
- Non-uniform randomization distributions; sensor models
- Visualization, data export and profiling tools

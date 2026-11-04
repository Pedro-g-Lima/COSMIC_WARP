# Initial conditions and resets

`sim.reset()` draws every entity's initial state from sampling bounds;
`sim.reset_to_conditions(...)` writes exact values. Both act on all entities
or a selection.

## What a reset does

```python
import math
import numpy as np
import warp as wp
from cosmic_warp.core.pipeline import SpacecraftSimulator

sim = SpacecraftSimulator(num_envs=4)
sim.reset()
```

For each selected entity, `reset()`:

1. samples position, momentum, angular momentum, attitude and mass within
   the bounds;
2. applies [domain randomization](randomization.md), if set;
3. clears the termination status, the actuator commands, the wheel momenta
   and the episode step counter.

It first checks the bounds on the host: a body's `mass_bounds` must not
reach below its `mass_dry`, and with `check_planet_impact` an orbit's lowest
perigee must clear the impact radius. The default bounds put the default
1 kg spacecraft within ±1 km and ±1 m/s of a circular 400 km orbit, with a
small random spin and a uniformly random attitude
([`ResetManager`](../reference/api/cosmic_warp/core/reset.md) lists them).

## Sampling bounds

Set them at construction with `reset_config`, or later with
`sim.reset_manager.set_bounds(...)`, which changes only what you pass:

```python
from cosmic_warp.physics.gravity import MU_EARTH, RE_EARTH

r0 = RE_EARTH + 400.0e3       # 400 km circular orbit, position along +X
v0 = (MU_EARTH / r0) ** 0.5   # circular speed, along +Y

sim = SpacecraftSimulator(
    num_envs=4,
    reset_config={
        "pos_bounds": ((r0 - 100.0, -100.0, -100.0), (r0 + 100.0, 100.0, 100.0)),
        "vel_bounds": ((-1.0, v0 - 1.0, -1.0), (1.0, v0 + 1.0, 1.0)),
        "ang_mom_bounds": ((-0.01, -0.01, -0.01), (0.01, 0.01, 0.01)),
        "mass_bounds": (1.0, 1.2),
        "seed": 7,
    },
)
sim.reset()
```

- Each box is `(min, max)`, sampled uniformly per component, in ECI: with
  gravity on, a box around the origin puts the spacecraft at Earth's centre.
- `vel_bounds` samples a velocity (momentum = sampled mass × velocity);
  `mom_bounds` samples the momentum. Give one.
- `seed`, an integer in [0, 2⁶⁴), seeds the per-entity streams
  ([random streams](../theory/sampling.md#random-streams)). `set_bounds`
  does not take it; `sim.reset_manager.reseed(seed)` restarts the streams.

YAML: the [`reset` section](configuration.md#reset).

## Attitude sampling

Uniform (Haar) within a cone of half-angle `att_max_angle` around
`att_nominal`, a quaternion `(x, y, z, w)` normalized for you:

```python
sim.reset_manager.set_bounds(
    att_nominal=(0.0, 0.0, 0.0, 1.0),
    att_max_angle=math.radians(10.0),  # attitude errors up to 10 degrees
)
```

| `att_max_angle` | Attitudes |
| --- | --- |
| `π` (default, largest accepted) | uniform over all attitudes |
| between 0 and π | uniform within the cone, not clustered near the nominal ([density](../theory/sampling.md#attitude-sampling)) |
| `0` | exactly `att_nominal` |

## Orbital-element sampling

A body can draw classical orbital elements instead of Cartesian boxes, each
uniform within its bounds, converted to an ECI state on the device, so
position and velocity lie on one orbit. Angles in radians:

```python
sim.reset_manager.set_orbit_bounds(
    altitude_bounds=(350.0e3, 450.0e3),         # or semi_major_axis_bounds [m]
    eccentricity_bounds=(0.0, 0.02),
    inclination_bounds=(math.radians(51.6),) * 2,
    raan_bounds=(0.0, 2.0 * math.pi),
    mean_anomaly_bounds=(0.0, 2.0 * math.pi),   # or true_anomaly_bounds
)
sim.reset()
```

The arguments are `semi_major_axis_bounds` or `altitude_bounds`,
`eccentricity_bounds` (within `[0, 1)`), `inclination_bounds`,
`raan_bounds`, `arg_periapsis_bounds`, and `true_anomaly_bounds` or
`mean_anomaly_bounds`.

- Mean anomaly is uniform in time (a spacecraft seen at a random instant);
  true anomaly is uniform in angle.
- Elements left out keep their values; by default a circular 400 km
  equatorial orbit, uniform in true anomaly.
- The Cartesian position and velocity/momentum boxes are ignored; angular
  momentum, mass and attitude are sampled as before.
  `sim.reset_manager.set_sample_mode("cartesian")` switches back.
- The elements are osculating at the reset instant. With J2 on, the orbit
  oscillates about a different mean orbit: a 500 km circular sample can sit
  kilometres off 500 km on average.

YAML: [`reset.orbit`](configuration.md#resetorbit) (keys without `_bounds`).
Densities: [Initial-condition sampling](../theory/sampling.md).

## Per-body bounds

`body_ids` gives each body its own bounds, such as a chaser and a target
that reset into different regions:

```python
sim = SpacecraftSimulator(num_envs=4, num_bodies=2)
sim.vehicle.set_config(mass_dry=80.0, body_ids=[0])
sim.vehicle.set_config(mass_dry=500.0, body_ids=[1])
sim.reset_manager.set_bounds(mass_bounds=(80.0, 120.0), body_ids=[0])
sim.reset_manager.set_bounds(mass_bounds=(500.0, 500.0), body_ids=[1])
sim.reset()
```

`set_orbit_bounds` and `set_sample_mode` take `body_ids` too;
`sim.reset_manager.get_bounds(body_id)` returns one body's bounds, keyed by
the setters' argument names. YAML:
[`reset.per_body`](configuration.md#resetper_body).

## Partial resets

`env_ids`, `body_ids` and `entity_ids` select what to reset:

```python
sim.reset(env_ids=[0, 2])
```

Termination is per environment, so resetting only some bodies of a
terminated environment raises. Inside a loop,
`simulate(auto_reset_terminated=True)` resets terminated environments on
the device ([Run, record, terminate](running.md)).

## Exact initial conditions

`reset_to_conditions` writes given values. Each array has one row per
entity, `num_envs * num_bodies`, even when resetting a subset:

```python
from cosmic_warp.physics.orbits import circular_orbit_state

n = sim.num_entities
pos, vel = circular_orbit_state(altitude=400.0e3, inclination=math.radians(51.6))
masses = np.tile([100.0, 500.0], sim.num_envs)       # chaser, target, per env
along = np.tile([-1.0e3, 0.0], sim.num_envs)         # chaser about 1 km behind
positions = pos + along[:, None] * vel / np.linalg.norm(vel)

sim.reset_to_conditions(
    positions=wp.array(positions, dtype=wp.vec3d, device=sim.device),
    momenta=wp.array(masses[:, None] * vel, dtype=wp.vec3d, device=sim.device),
    masses=wp.array(masses, dtype=wp.float64, device=sim.device),
)
```

- Pass momenta, `mass * velocity`.
- An omitted array takes the centre of the reset bounds (identity for the
  quaternion); without `masses`, that centre must not be below `mass_dry`.
  Bodies with orbital-element sampling need `positions` and `momenta`.
- Types, lengths and devices are checked; values only for masses (at least
  `mass_dry`) and quaternions (finite, nonzero).
- It clears termination, commands, wheel momenta and the step counter like
  `reset()`, but does not randomize and does not change the bounds.

[`cosmic_warp.physics.orbits`](../reference/api/cosmic_warp/physics/orbits.md)
converts on the host: `elements_to_state(semi_major_axis, eccentricity, ...)`
and `circular_orbit_state(altitude, ...)` return an ECI
`(position, velocity)` pair; `state_to_elements(position, velocity)` returns
osculating elements. YAML: [`initial_conditions`](configuration.md#initial_conditions).

## Episode step counters

`sim.get_episode_steps()` counts each entity's steps since its last reset;
every reset path zeroes it. Use it to truncate episodes when environments
reset at different times:

```python
max_episode_steps = 500
truncated = sim.get_episode_steps().numpy() >= max_episode_steps
```

API: [`ResetManager`](../reference/api/cosmic_warp/core/reset.md) ·
[`SpacecraftSimulator.reset_to_conditions`](../reference/api/cosmic_warp/core/pipeline.md)

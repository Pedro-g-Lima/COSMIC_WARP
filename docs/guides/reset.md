# Initial conditions and resets

`sim.reset()` draws every entity's initial state from sampling bounds;
`sim.reset_to_conditions(...)` sets exact values. Both act on all entities or
a selection.

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
   the reset bounds;
2. applies [domain randomization](randomization.md), if enabled;
3. clears the termination status, zeroes the actuator commands and wheel
   momenta, and zeroes the episode step counter.

Before sampling, it checks the bounds on the host and raises `ValueError`
if a body's `mass_bounds` can fall below its `mass_dry`, or — with
`check_planet_impact` — if an orbit's lowest perigee is at or below the
impact radius.

The default bounds place the default 1 kg spacecraft within ±1 km and
±1 m/s of a circular 400 km orbit, with a small random spin and a uniformly
random attitude; [`ResetManager`](../reference/api/cosmic_warp/core/reset.md)
lists them.

## Sampling bounds

Set the bounds at construction with `reset_config`, or later with
`sim.reset_manager.set_bounds(...)`, which changes only the bounds you pass:

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

- Each box is `(min, max)` with three components per corner, sampled
  uniformly per component. Positions and velocities are in ECI: with gravity
  on, a box around the origin puts the spacecraft at Earth's centre.
- `vel_bounds` samples a velocity and multiplies it by the sampled mass, so
  the velocity holds for any mass. `mom_bounds` samples the momentum instead;
  give one or the other.
- `seed` seeds the reset sampler's random streams: an integer in [0, 2⁶⁴).
  `set_bounds` does not take it; `sim.reset_manager.reseed(seed)` restarts
  the streams later. Each entity
  has its own stream, and no two streams overlap
  ([random streams](../theory/sampling.md#random-streams)).

The [`reset` YAML section](configuration.md#reset) takes the same bounds.

## Attitude sampling

Attitudes are uniform (Haar) within a cone of half-angle `att_max_angle`
around `att_nominal`, a quaternion `(x, y, z, w)` normalized for you:

```python
sim.reset_manager.set_bounds(
    att_nominal=(0.0, 0.0, 0.0, 1.0),
    att_max_angle=math.radians(10.0),  # attitude errors up to 10 degrees
)
```

- `att_max_angle = π` (the default, and the largest accepted; the range is
  `[0, π]`): uniform over all attitudes.
- `att_max_angle = 0`: exactly `att_nominal` on every reset.
- In between: uniform over the attitudes within the cone, not clustered near
  the nominal. The density is derived in
  [Initial-condition sampling](../theory/sampling.md).

## Orbital-element sampling

Instead of Cartesian boxes, a body can draw classical orbital elements,
each uniform within its bounds, converted to an ECI state on the device.
Position and velocity then always lie on one orbit. Angles are in radians:

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

- `mean_anomaly_bounds` is uniform in time along the orbit (a spacecraft
  observed at a random instant); `true_anomaly_bounds` is uniform in angle.
  Give at most one; likewise `semi_major_axis_bounds` or `altitude_bounds`.
- Elements you leave out keep their values: by default a circular 400 km
  equatorial orbit, uniform in true anomaly.
- The Cartesian `pos_bounds`, `vel_bounds` and `mom_bounds` are ignored;
  angular momentum, mass and attitude are sampled as before.
  `sim.reset_manager.set_sample_mode("cartesian")` switches back.
- The elements are osculating: the two-body orbit through the sampled state
  at the reset instant. With J2 on, the orbit then oscillates about a mean
  orbit that differs from it, so a 500 km circular sample does not stay at
  500 km on average; the difference can be kilometres.
- Eccentricity must be in `[0, 1)`. The orbits are about Earth
  (`MU_EARTH`), the only central body the dynamics know.

In YAML: [`reset.orbit`](configuration.md#resetorbit), whose keys drop the
`_bounds` suffix. The sampling densities
are derived in [Initial-condition sampling](../theory/sampling.md).

## Per-body bounds

`body_ids` gives each body its own bounds — a chaser and a target that reset
into different regions, say:

```python
sim = SpacecraftSimulator(num_envs=4, num_bodies=2)
sim.vehicle.set_config(mass_dry=80.0, body_ids=[0])
sim.vehicle.set_config(mass_dry=500.0, body_ids=[1])
sim.reset_manager.set_bounds(mass_bounds=(80.0, 120.0), body_ids=[0])
sim.reset_manager.set_bounds(mass_bounds=(500.0, 500.0), body_ids=[1])
sim.reset()
```

`set_orbit_bounds` and `set_sample_mode` take `body_ids` too, and
`sim.reset_manager.get_bounds(body_id)` returns one body's bounds, keyed by
the setters' argument names. In YAML:
[`reset.per_body`](configuration.md#resetper_body).

## Partial resets

`env_ids`, `body_ids` and `entity_ids` select what to reset:

```python
sim.reset(env_ids=[0, 2])
```

Termination is per environment: `reset` raises if the selection covers only
some bodies of a terminated environment. Inside a loop,
`simulate(auto_reset_terminated=True)` resets terminated environments on the
device instead — see [Run, record, terminate](running.md).

## Exact initial conditions

`reset_to_conditions` writes given values instead of sampling. Each array
has one row per entity, `num_envs * num_bodies`, even when you reset a
subset:

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

- The state holds momenta: pass `mass * velocity`.
- An omitted array takes the centre of the reset bounds; the quaternion
  defaults to identity. Without `masses`, the centre of `mass_bounds` must
  not be below `mass_dry`. For a body with orbital-element sampling,
  `positions` and `momenta` are required.
- Array types, lengths and devices are checked; the values are not (they
  are on the device), except that masses must be at least the body's
  `mass_dry` and quaternions finite and nonzero.
- Like `reset()`, it clears termination, actuator commands, wheel momenta
  and the step counter. It does not apply domain randomization.
- It does not change the bounds: the next `reset()` samples again.

`elements_to_state(semi_major_axis, eccentricity, inclination, raan,
arg_periapsis, true_anomaly)` and `circular_orbit_state(altitude, ...)` in
[`cosmic_warp.physics.orbits`](../reference/api/cosmic_warp/physics/orbits.md)
convert elements to an ECI `(position, velocity)` pair of NumPy arrays, and
`state_to_elements(position, velocity)` returns a state's osculating
elements, to analyse results. In YAML,
[`initial_conditions`](configuration.md#initial_conditions) takes the same
orbit specification.

## Episode step counters

`sim.get_episode_steps()` counts, per entity, the steps since its last
reset; every reset path zeroes it for the entities it resets. Use it to
truncate episodes when environments reset at different times:

```python
max_episode_steps = 500
truncated = sim.get_episode_steps().numpy() >= max_episode_steps
```

API: [`ResetManager`](../reference/api/cosmic_warp/core/reset.md) ·
[`SpacecraftSimulator.reset_to_conditions`](../reference/api/cosmic_warp/core/pipeline.md)

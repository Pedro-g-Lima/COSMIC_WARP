# Configure a scenario

A whole scenario — simulator, physics, termination, vehicles, actuators,
reset bounds, initial conditions and randomization — fits in one YAML file
(these ship with the repository; run from its root):

```python
from cosmic_warp.core.config import create_simulator_from_config, create_simulator_from_yaml

sim = create_simulator_from_yaml("examples/minimal_orbit.yaml")
sim = create_simulator_from_yaml("examples/minimal_orbit.yaml", device="cpu")  # overrides simulation.device
sim.simulate(num_steps=100)
```

`create_simulator_from_config(config)` takes the same content as a
dictionary. Both return a reset simulator, ready to step.

The loader is strict: an unknown key raises `ValueError` naming it and the
allowed set, and every value is checked
([validation summary](#validation-summary)). A typo fails loudly instead of
falling back to a default.

## Conventions

- SI units; **all angles in radians**, orbital elements included.
- Positions and momenta in ECI; angular momenta and spin axes in the body
  frame.
- The state holds **momenta** (`mass * velocity`). Where a key accepts a
  velocity, the loader converts it, so a mass is required alongside.
- Entities are environment-major: `entity_id = env_id * num_bodies + body_id`.
- Numbers follow YAML 1.2 (`1e8`, `1.0e+8` are floats); quoted numbers
  (`"40"`) and booleans (`"false"`) are rejected.
- A `per_body` list names each `body_id` at most once.

## Top-level structure

```yaml
simulation:          # dimensions and integration
physics:             # force and torque switches
termination:         # termination limits
reset:               # sampling bounds for every reset (global + per body)
body:                # vehicles and actuators (defaults + per body)
initial_conditions:  # exact initial state, applied once at load
randomization:       # vehicle parameters redrawn at every reset
```

Every section is optional; omitted keys take the defaults below. Ready-made
scenarios are listed under [YAML scenarios](../examples/index.md#yaml-scenarios).

## Random resets vs. initial conditions

| | `reset` | `initial_conditions` |
|---|---|---|
| Describes | a **distribution** (bounds) | exact **values** |
| Applied | at load, on `sim.reset()`, on every auto-reset | **once**, at load |
| Typical use | RL training, Monte Carlo | one reproducible scenario, multi-body geometry |

At load the simulator is reset (drawing from `reset`, applying
`randomization`), then `initial_conditions` **overwrite** the drawn state.
Later resets draw from `reset` again; the initial conditions are not
restored. For the same state on every reset, give each `reset` bound equal
min and max (and `att_max_angle: 0.0`), or call `sim.reset_to_conditions(...)`.

Both sections accept an `orbit` spec with the same key names:
[`reset.orbit`](#resetorbit) takes `[min, max]` ranges, the
[`initial_conditions` one](#orbit-spec) scalars only.

## `simulation`

| Key | Default | Description |
|---|---|---|
| `num_envs` | `1` | Number of parallel environments |
| `num_bodies` | `1` | Bodies (spacecraft) per environment |
| `integrator` | `rk4` | `euler`, `rk4`, or `symplectic` |
| `dt` | `0.01` | Time step [s] |
| `device` | `cuda` | Warp device; the loader's `device=` overrides it |

## `physics`

| Key | Default | Description |
|---|---|---|
| `gravity` | `true` | Point-mass gravity |
| `j2_perturbation` | same as `gravity` | J2 perturbation; `true` without `gravity` is an error |
| `gravity_gradient` | `false` | Gravity-gradient torque |
| `magnetic_torque` | `false` | Magnetic dipole torque (vehicle `magnetic_dipole`) |
| `drag` | `false` | Atmospheric drag (vehicle `area_drag`, `cd`) |

## `termination`

Termination is per environment: when any body violates a limit, the whole
environment terminates (and resets together under auto-reset). Reason codes:
[Termination](../theory/execution.md#termination).

| Key | Default | Description |
|---|---|---|
| `check_bounds` | `true` | Enable the position, momentum, angular-momentum and mass limits |
| `check_planet_impact` | `false` | Enable central-body impact detection |
| `pos_limit` | `1.0e+8` | Max position magnitude [m] |
| `mom_limit` | `.inf` | Max momentum magnitude [kg·m/s]; absolute, so size it for the vehicle |
| `ang_mom_limit` | `.inf` | Max angular momentum magnitude [kg·m²/s]; absolute |
| `mass_limit` | `0.0` | Terminate once mass ≤ this [kg]; `0` disables, `mass_dry` ends the episode at burnout |
| `planet_radius` | `6.3781363e+6` | Impact radius [m] for `check_planet_impact`; Earth's equatorial radius (`RE_EARTH`) |

## `reset`

Uniform sampling bounds for `sim.reset()` and the automatic resets. Attitudes
are uniform (Haar) within a cone of half-angle `att_max_angle` around
`att_nominal` ([attitude sampling](reset.md#attitude-sampling)).

| Key | Default | Description |
|---|---|---|
| `pos_bounds` | ~400 km circular LEO box | `[[xmin,ymin,zmin],[xmax,ymax,zmax]]` position box [m] |
| `vel_bounds` | LEO circular velocity ±1 m/s | Velocity box [m/s]; momentum = sampled mass × velocity |
| `mom_bounds` | — | Momentum box [kg·m/s]; alternative to `vel_bounds` |
| `ang_mom_bounds` | `[[-0.1,...],[0.1,...]]` | Angular momentum box [kg·m²/s] |
| `mass_bounds` | `[1.0, 1.0]` | `[min, max]` total mass [kg]; not below `mass_dry` |
| `att_nominal` | `[0, 0, 0, 1]` | Nominal quaternion `[x, y, z, w]`, normalized on load |
| `att_max_angle` | `pi` | Cone half-angle [rad] in `[0, pi]`: `pi` uniform SO(3), `0` fixed |
| `orbit` | — | Orbital-element bounds (below); replaces the position and velocity/momentum boxes |
| `seed` | `42` | Seed of the reset sampler's RNG streams |
| `per_body` | — | Per-body overrides (below) |

### `reset.orbit`

Each element is drawn uniformly within its `[min, max]` (a scalar is a fixed
value) and converted to an ECI state on the device, so position and momentum
lie on one orbit ([sampling theory](../theory/sampling.md)). Unspecified
elements default to a circular, equatorial 400 km orbit with uniform true
anomaly.

| Key | Description |
|---|---|
| `semi_major_axis` | Semi-major axis [m] |
| `altitude` | Altitude above Earth's equatorial radius [m]; alternative to `semi_major_axis` |
| `eccentricity` | Within `[0, 1)` |
| `inclination` | [rad], within `[0, pi]` |
| `raan` | Right ascension of the ascending node [rad] |
| `arg_periapsis` | Argument of periapsis [rad] |
| `true_anomaly` | [rad], uniform in angle |
| `mean_anomaly` | [rad], uniform in time; alternative to `true_anomaly` |

```yaml
reset:
  orbit:
    altitude: [350.0e3, 450.0e3]
    eccentricity: [0.0, 0.02]
    inclination: 0.9006     # 51.6 deg; scalar = fixed value
    raan: [0.0, 6.2832]
    mean_anomaly: [0.0, 6.2832]
```

### `reset.per_body`

Each entry needs a `body_id` and takes any sampling key, `orbit` included;
omitted keys keep the global values. Python:
`sim.reset_manager.set_bounds(..., body_ids=[0])`.

```yaml
reset:
  # +/- 1 km around a 400 km LEO point (RE + 400 km = 6.7781363e+6 m along +X);
  # velocities keep the default circular-LEO vel_bounds
  pos_bounds: [[6.7771363e+6, -1000.0, -1000.0], [6.7791363e+6, 1000.0, 1000.0]]
  per_body:
    - body_id: 0                      # chaser
      mass_bounds: [80.0, 120.0]
    - body_id: 1                      # target: fixed mass, its own box ~1 km radially out
      pos_bounds: [[6.7790363e+6, -50.0, -50.0], [6.7792363e+6, 50.0, 50.0]]
      mass_bounds: [500.0, 500.0]
```

```python
sim.reset_manager.set_bounds(mass_bounds=(80.0, 120.0), body_ids=[0])
```

## `body`

`defaults` applies to every body; each `per_body` entry overrides it for one
body in every environment. Per-body keys merge over the defaults, which
merge over the built-in values.

```yaml
body:
  defaults:
    vehicle: { ... }
    actuators:
      thrusters: { ... }
      reaction_wheels: { ... }
  per_body:
    - body_id: 0
      vehicle: { ... }
      actuators: { ... }
```

### `vehicle`

| Key | Default | Description |
|---|---|---|
| `mass_dry` | `1.0` | Dry mass [kg] |
| `inertia` | cube (`m/6 · I`) | 3×3 inertia tensor [kg·m²] |
| `area_drag` | `1.0` | Drag reference area [m²] |
| `cd` | `2.2` | Drag coefficient |
| `magnetic_dipole` | `[0,0,0]` | Residual dipole, body frame [A·m²] |

### `actuators.thrusters`

| Key | Default | Description |
|---|---|---|
| `max_thrust` | `10.0` | Max thrust per thruster [N] |
| `isp` | `0.0` | Specific impulse [s]; `0` uses no propellant |
| `positions` | 6-face layout | Up to 8 `[x,y,z]` positions from the CoM [m] |
| `directions` | 6-face layout | Up to 8 `[dx,dy,dz]` force directions, normalized; `[0,0,0]` = unused slot |

### `actuators.reaction_wheels`

| Key | Default | Description |
|---|---|---|
| `max_torque` | `0.1` | Max torque per wheel [N·m] |
| `max_momentum` | `0.0` | Momentum capacity per wheel [kg·m²/s]; `0` = unlimited |
| `wheel_inertia` | `0.01` | Wheel inertia [kg·m²] |
| `viscous_damping` | `0.0` | Viscous friction [N·m·s/rad] |
| `spin_axes` | orthogonal X/Y/Z | Up to 4 `[x,y,z]` axes, body frame |

## `initial_conditions`

Exact initial state, applied **once** at load over the initial draw. Two
mutually exclusive forms.

### Per-body form

One entry per body, **broadcast to every environment**. Every `body_id` from
`0` to `num_bodies - 1` appears exactly once; each field is given for all
bodies or none.

| Key | Description |
|---|---|
| `body_id` | Required |
| `position` | `[x,y,z]` ECI [m] |
| `velocity` | `[vx,vy,vz]` ECI [m/s]; requires `mass` |
| `momentum` | `[px,py,pz]` ECI [kg·m/s]; alternative to `velocity` |
| `angular_momentum` | `[hx,hy,hz]` body frame [kg·m²/s] |
| `quaternion` | `[qx,qy,qz,qw]`, normalized; zero rejected |
| `mass` | Total mass [kg], at least the body's `mass_dry` |
| `orbit` | Orbit spec (below); replaces `position` and `velocity` |

#### `orbit` spec

Classical elements to an ECI state. Requires `mass` in the same entry;
excludes `position`, `velocity` and `momentum`.

| Key | Default | Description |
|---|---|---|
| `semi_major_axis` | — | [m]; give this **or** `altitude` |
| `altitude` | — | Above Earth's equatorial radius [m] |
| `eccentricity` | `0.0` | `0 ≤ e < 1` |
| `inclination` | `0.0` | [rad], in `[0, pi]` |
| `raan` | `0.0` | [rad] |
| `arg_periapsis` | `0.0` | [rad] |
| `true_anomaly` | `0.0` | [rad] |
| `mean_anomaly` | — | [rad], via Kepler's equation; alternative to `true_anomaly` |

A chaser trailing a target on the same circular orbit (51.6° = 0.9006 rad):

```yaml
simulation:
  num_envs: 1024
  num_bodies: 2

initial_conditions:
  per_body:
    - body_id: 0   # target
      orbit: {altitude: 400.0e+3, inclination: 0.9006}
      mass: 500.0
    - body_id: 1   # chaser, 0.2 mrad behind (~1.4 km along-track)
      orbit: {altitude: 400.0e+3, inclination: 0.9006, true_anomaly: -0.0002}
      mass: 120.0
```

### Per-entity form

One row per entity (`num_envs * num_bodies`, environment-major), for
programmatic setups.

| Key | Shape | Description |
|---|---|---|
| `positions` | `[N,3]` | ECI [m] |
| `velocities` | `[N,3]` | ECI [m/s]; requires `masses` |
| `momenta` | `[N,3]` | ECI [kg·m/s]; alternative to `velocities` |
| `angular_momenta` | `[N,3]` | Body frame [kg·m²/s] |
| `quaternions` | `[N,4]` | `[qx,qy,qz,qw]`, normalized; zero rejected |
| `masses` | `[N]` | Total masses [kg], each ≥ its body's `mass_dry` |

### Fields left out

They keep the state drawn at load. A mass given without a momentum or
velocity keeps the drawn velocity (the momentum is rescaled). For bodies
sampled with `reset.orbit`, positions and momenta (or velocities) come
together or not at all: half of a drawn orbit state with a prescribed other
half would lie on a different orbit.

## `randomization`

Vehicle parameters redrawn per entity at every reset, auto-resets included
([Domain randomization](randomization.md); Python:
`sim.randomizer.set_bounds(...)`).

```yaml
randomization:
  seed: 1234                  # optional RNG seed for the parameter draws
  propellant: [10.0, 20.0]    # kg, on top of the (possibly sampled) dry mass
  inertia_scale: [0.9, 1.1]   # scale factor on the nominal inertia tensor
  per_body:                   # overrides, same keys plus body_id
    - body_id: 1
      mass_dry: [75.0, 85.0]
      propellant: [30.0, 40.0]
```

| Key | Type | Meaning |
| --- | --- | --- |
| `seed` | int | Seed of the randomization streams (separate from `reset.seed`) |
| `mass_dry` | `[min, max]` | Dry mass [kg]; **requires `propellant`** for the same bodies |
| `propellant` | `[min, max]` | Propellant [kg]; reset mass = dry + propellant, and `reset.mass_bounds` is then unused |
| `inertia_scale` | `[min, max]` | Multiplier on the body's inertia tensor |
| `per_body` | list | Overrides of the keys above, with `body_id` |

## Validation summary

The loader raises `ValueError` naming the field, before building the
simulator where it can, for:

- an unknown key in any section or entry;
- a wrong type or range: a quoted number or boolean, a non-finite number, a
  non-positive value where a positive one is needed (`dt`, `mass_dry`,
  `wheel_inertia`, ...), a non-integer count, an unknown `integrator`, an
  inertia tensor no rigid body has, a zero quaternion, a direction or axis
  too short to normalize, an inclination or `att_max_angle` outside
  `[0, pi]`;
- `j2_perturbation: true` with `gravity: false`;
- a `body_id` missing, duplicated or outside `[0, num_bodies - 1]`;
- mixed per-body and per-entity `initial_conditions`, or a per-entity list
  of the wrong length;
- `velocity`/`velocities`/`orbit` without a mass, or with
  `momentum`/`momenta`;
- only one of positions and momenta for a body that uses `reset.orbit`;
- an orbit spec with both or neither of `semi_major_axis` and `altitude`,
  both anomalies, or an eccentricity outside `[0, 1)`; for `reset.orbit`
  also an out-of-order pair, or — with `check_planet_impact` — a worst-case
  perigee `sma_min * (1 - ecc_max)` at or below `planet_radius`;
- an initial mass below the body's effective `mass_dry` (the randomization
  maximum when `mass_dry` is randomized);
- out-of-order or non-positive `randomization` bounds, or `mass_dry`
  randomized without `propellant`.

Reset `mass_bounds` (including the 1 kg default) are checked against the dry
mass by `sim.reset()`, so the check covers YAML and Python alike: it raises
on every call while the bounds can sample below a body's dry mass (bodies
with randomized propellant are exempt), or while orbit bounds allow a
perigee at or below the impact radius with `check_planet_impact` on.

## Complete example

`examples/multibody_orbit.yaml`, the scenario of the
[multi-body orbit example](../examples/09_multi_body_orbit.md): 3
environments × 4 bodies with per-body dry masses, reset mass bounds and
circular-orbit initial conditions.

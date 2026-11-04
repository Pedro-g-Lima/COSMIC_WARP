# Configure a scenario

A whole scenario — simulator, physics, termination, vehicles, actuators,
reset bounds, initial conditions and randomization — fits in one YAML file
(the files below ship with the repository; run from its root):

```python
from cosmic_warp.core.config import create_simulator_from_config, create_simulator_from_yaml

sim = create_simulator_from_yaml("examples/minimal_orbit.yaml")
sim = create_simulator_from_yaml("examples/minimal_orbit.yaml", device="cpu")  # overrides simulation.device
sim.simulate(num_steps=100)
```

`create_simulator_from_config(config)` takes the same content as a Python
dictionary. The loader returns a simulator that is already reset, ready to
step.

The loader is strict. An unknown key anywhere raises `ValueError` naming the
key and the allowed set, and every value is checked when it is applied — see
the [validation summary](#validation-summary). A typo fails loudly instead of
falling back to a default.

## Conventions

- All quantities are SI: metres, kilograms, seconds, newtons.
- Positions and (linear) momenta are expressed in the inertial (ECI) frame;
  angular momenta and spin axes in the body frame.
- The state uses **momenta**, not velocities (`momentum = mass * velocity`).
  Wherever the configuration accepts `velocity`/`velocities`, the loader
  performs the conversion for you — which is why a mass is required alongside.
- All angles are in **radians**, including orbital elements — in YAML and in
  the Python API alike.
- Entities are indexed environment-major:
  `entity_id = env_id * num_bodies + body_id`.
- Numbers follow YAML 1.2: `1e8`, `1.0e8` and `1.0e+8` are all floats.
  Quoted numbers (`"40"`) and quoted booleans (`"false"`) are rejected.
- `per_body` lists may name each `body_id` at most once.

## Top-level structure

```yaml
simulation:          # simulator dimensions and integration settings
physics:             # force/torque model switches
termination:         # termination limits
reset:               # random-reset sampling bounds (global + per-body)
body:                # vehicle and actuator configuration (defaults + per-body)
initial_conditions:  # deterministic initial state (per-entity or per-body)
randomization:       # domain randomization of vehicle parameters at reset
```

Every section is optional; omitted keys take the defaults listed below.
Ready-to-run scenarios, from a single orbit to a rendezvous, are listed under
[YAML scenarios](../examples/index.md#yaml-scenarios) in the examples.

## Random resets vs. initial conditions

Two sections describe where entities start, and they answer different
questions:

| | `reset` | `initial_conditions` |
|---|---|---|
| Describes | A **distribution** (sampling bounds) | Exact **values** |
| Applied | Every reset: at load, on `sim.reset()`, and on every auto-reset of a terminated environment | **Once**, at load time |
| Varies between draws | Yes — that is the point | No |
| Typical use | RL training, Monte Carlo | A specific reproducible scenario, relative multi-body geometry, physics validation |

At load time the simulator is first reset (drawing from the `reset` bounds
and applying `randomization`), then any `initial_conditions` **overwrite**
the drawn state. Any later reset re-draws from the `reset` bounds — the
pinned initial conditions are *not* restored. The two compose naturally: pin
the first episode with `initial_conditions` and let `reset` govern what
subsequent episodes look like.

To get the *same* exact state on every reset, make the `reset` distribution
degenerate instead: give each bound identical min and max (and
`att_max_angle: 0.0` for a fixed attitude), or call
`sim.reset_to_conditions(...)` from Python.

One naming caveat: both sections accept an `orbit` spec with the same key
names but different value shapes. Under [`reset.orbit`](#resetorbit) every
key is a `[min, max]` sampling range (scalars mean a fixed value); under
[`initial_conditions`](#initial_conditions) the [`orbit` spec](#orbit-spec)
takes scalars only — one exact orbit.

## `simulation`

| Key | Default | Description |
|---|---|---|
| `num_envs` | `1` | Number of parallel environments |
| `num_bodies` | `1` | Bodies (spacecraft) per environment |
| `integrator` | `rk4` | `euler`, `rk4`, or `symplectic` |
| `dt` | `0.01` | Time step [s] |
| `device` | `cuda` | Warp device; `create_simulator_from_yaml(path, device=...)` overrides it |

## `physics`

Boolean switches for the force/torque models evaluated inside the fused
integration kernel.

| Key | Default | Description |
|---|---|---|
| `gravity` | `true` | Central-body point-mass gravity |
| `j2_perturbation` | same as `gravity` | J2 oblateness perturbation (requires `gravity`; `true` without it is an error) |
| `gravity_gradient` | `false` | Gravity-gradient torque |
| `magnetic_torque` | `false` | Magnetic dipole torque (set `magnetic_dipole` on the vehicle) |
| `drag` | `false` | Atmospheric drag (uses `area_drag` and `cd` on the vehicle) |

## `termination`

Termination is **environment-level**: when any body of an environment violates
a limit, all bodies of that environment are flagged terminated (and reset
together if auto-reset is enabled). The momentum and angular-momentum limits
are off by default (`.inf`); they are absolute values, so size them for the
vehicle. The reason codes are listed in
[Termination](../theory/execution.md#termination).

| Key | Default | Description |
|---|---|---|
| `check_bounds` | `true` | Enable position/momentum/angular-momentum/mass limits |
| `check_planet_impact` | `false` | Enable central-body impact detection |
| `pos_limit` | `1.0e+8` | Max position magnitude [m] |
| `mom_limit` | `.inf` | Max momentum magnitude [kg·m/s]; `.inf` is no limit |
| `ang_mom_limit` | `.inf` | Max angular momentum magnitude [kg·m²/s]; `.inf` is no limit |
| `mass_limit` | `0.0` | Terminate once mass ≤ this [kg]; `0` disables, `mass_dry` ends the episode when propellant runs out |
| `planet_radius` | `6.3781363e+6` | Impact radius [m], used by `check_planet_impact`; the default is Earth's equatorial radius (`RE_EARTH`) |

## `reset`

Bounds for **uniform random sampling** used by `sim.reset()` and by automatic
resets of terminated environments — a distribution, not a fixed state (see
[Random resets vs. initial conditions](#random-resets-vs-initial-conditions)).
Attitude quaternions are sampled uniformly
(Haar) within a cone of half-angle `att_max_angle` around `att_nominal`; the
default (`pi`) is uniform over all of SO(3), and `0` pins the attitude to
exactly `att_nominal` on every reset.

| Key | Default | Description |
|---|---|---|
| `pos_bounds` | ~400 km circular LEO box | `[[xmin,ymin,zmin],[xmax,ymax,zmax]]` position box [m] |
| `vel_bounds` | LEO circular velocity ±1 m/s | Velocity box [m/s]; momentum = sampled mass × velocity |
| `mom_bounds` | — | Momentum box [kg·m/s]; alternative to `vel_bounds` (give at most one) |
| `ang_mom_bounds` | `[[-0.1,...],[0.1,...]]` | Angular momentum box [kg·m²/s] |
| `mass_bounds` | `[1.0, 1.0]` | `[min, max]` total mass [kg] |
| `att_nominal` | `[0, 0, 0, 1]` | Nominal attitude quaternion `[x, y, z, w]` (normalized on load) |
| `att_max_angle` | `pi` | Attitude cone half-angle [rad] around `att_nominal`, in `[0, pi]` (`pi`: uniform SO(3), `0`: fixed attitude) |
| `orbit` | — | Orbital-element sampling bounds (below); replaces `pos_bounds`/`vel_bounds`/`mom_bounds` for the affected bodies |
| `seed` | `42` | Seed of the reset sampler's RNG streams |
| `per_body` | — | List of per-body overrides (below) |

### `reset.orbit`

Switches the affected bodies to orbital-element sampling: each element is
drawn uniformly within its bounds and converted to an ECI state on device,
so position and momentum are dynamically consistent (see
[Initial-condition sampling](../theory/sampling.md)). Every key takes a
`[min, max]` pair (a scalar means a fixed value); angles in radians.
Unspecified elements default to a circular, equatorial 400 km orbit with
uniform true anomaly.

| Key | Description |
|---|---|
| `semi_major_axis` | Semi-major axis bounds [m] |
| `altitude` | Altitude bounds above the Earth equatorial radius [m] (alternative to `semi_major_axis`) |
| `eccentricity` | Eccentricity bounds, within `[0, 1)` |
| `inclination` | Inclination bounds [rad], within `[0, pi]` |
| `raan` | Right ascension of the ascending node bounds [rad] |
| `arg_periapsis` | Argument of periapsis bounds [rad] |
| `true_anomaly` | True anomaly bounds [rad] — uniform in angle |
| `mean_anomaly` | Mean anomaly bounds [rad] — uniform in time (alternative to `true_anomaly`) |

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

Each body can have its own sampling bounds — e.g. a chaser and a target that
must re-randomize into different regions. Every entry needs a `body_id`; any
of the sampling keys (including an `orbit` spec) may be given, and omitted
ones keep the global values.

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

The same override is available programmatically:

```python
sim.reset_manager.set_bounds(mass_bounds=(80.0, 120.0), body_ids=[0])
```

**Validation:** the sampled mass must not fall below the corresponding body's
`mass_dry` — `sim.reset()` (which the loader calls) rejects `mass_bounds`
whose minimum is below the dry mass of that body (such entities would spawn
with no propellant); see the [validation summary](#validation-summary).

## `body`

Vehicle and actuator configuration. `defaults` applies to all bodies;
`per_body` entries override the defaults for one body (across all
environments). Per-body keys merge on top of the defaults, which merge on top
of the built-in base values.

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
| `mass_dry` | `1.0` | Dry mass (without propellant) [kg] |
| `inertia` | cube (`m/6 · I`) | 3×3 inertia tensor [kg·m²] |
| `area_drag` | `1.0` | Drag reference area [m²] |
| `cd` | `2.2` | Drag coefficient |
| `magnetic_dipole` | `[0,0,0]` | Residual dipole moment, body frame [A·m²] |

### `actuators.thrusters`

Up to 8 thrusters per body.

| Key | Default | Description |
|---|---|---|
| `max_thrust` | `10.0` | Max thrust per thruster [N] |
| `isp` | `0.0` | Specific impulse [s] (`0` = no propellant consumption) |
| `positions` | 6-face layout | List of up to 8 `[x,y,z]` positions relative to CoM [m] |
| `directions` | 6-face layout | List of up to 8 `[dx,dy,dz]` thrust directions, normalized on load (`[0,0,0]` = unused slot) |

### `actuators.reaction_wheels`

Up to 4 wheels per body.

| Key | Default | Description |
|---|---|---|
| `max_torque` | `0.1` | Max torque per wheel [N·m] |
| `max_momentum` | `0.0` | Momentum capacity per wheel [kg·m²/s] (`0` = unlimited); the wheel saturates at this value |
| `wheel_inertia` | `0.01` | Wheel moment of inertia [kg·m²] |
| `viscous_damping` | `0.0` | Viscous friction [N·m·s/rad] |
| `spin_axes` | orthogonal X/Y/Z | List of up to 4 `[x,y,z]` spin axes (body frame) |

## `initial_conditions`

Deterministic initial state, applied **once** at load time, overwriting the
initial random draw; later resets re-draw from the `reset` bounds instead
(see [Random resets vs. initial conditions](#random-resets-vs-initial-conditions)).
Two mutually exclusive forms:

### Per-body form (recommended for multi-body scenarios)

One entry per body, **broadcast to every environment**. All `body_id`s from
`0` to `num_bodies - 1` must appear exactly once. Each field must be given
either for all bodies or for none; fields left out fall back to the defaults
listed under [Defaults](#defaults-for-unspecified-fields).

| Key | Description |
|---|---|
| `body_id` | Which body this entry configures (required) |
| `position` | `[x,y,z]` ECI position [m] |
| `velocity` | `[vx,vy,vz]` ECI velocity [m/s] — requires `mass` |
| `momentum` | `[px,py,pz]` ECI momentum [kg·m/s] — alternative to `velocity` |
| `angular_momentum` | `[hx,hy,hz]` body-frame angular momentum [kg·m²/s] |
| `quaternion` | `[qx,qy,qz,qw]` attitude quaternion (normalized on load; zero rejected) |
| `mass` | Total initial mass [kg] (must be ≥ the body's `mass_dry`) |
| `orbit` | Orbital-element spec (below) — replaces `position` + `velocity` |

#### `orbit` spec

Converts classical orbital elements to the Cartesian ECI state. Angles in
**radians**. Requires `mass` in the same entry (for the momentum conversion)
and cannot be combined with `position`, `velocity`, or `momentum`.

| Key | Default | Description |
|---|---|---|
| `semi_major_axis` | — | Semi-major axis [m] (give this **or** `altitude`) |
| `altitude` | — | Altitude above Earth's equatorial radius [m] |
| `eccentricity` | `0.0` | Eccentricity (`0 ≤ e < 1`) |
| `inclination` | `0.0` | Inclination [rad], in `[0, pi]` |
| `raan` | `0.0` | Right ascension of the ascending node [rad] |
| `arg_periapsis` | `0.0` | Argument of periapsis [rad] |
| `true_anomaly` | `0.0` | True anomaly [rad] |
| `mean_anomaly` | — | Mean anomaly [rad], converted with Kepler's equation (alternative to `true_anomaly`) |

Example — a chaser trailing a target on the same circular orbit (51.6° =
0.9006 rad):

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

Explicit lists with one row per entity (`num_envs * num_bodies` rows, in
environment-major order). Use this for fully programmatic setups; for anything
hand-written, prefer the per-body form.

| Key | Shape | Description |
|---|---|---|
| `positions` | `[N,3]` | ECI positions [m] |
| `velocities` | `[N,3]` | ECI velocities [m/s] — requires `masses` |
| `momenta` | `[N,3]` | ECI momenta [kg·m/s] — alternative to `velocities` |
| `angular_momenta` | `[N,3]` | Body-frame angular momenta [kg·m²/s] |
| `quaternions` | `[N,4]` | Attitude quaternions `[qx,qy,qz,qw]` (normalized on load; zero rejected) |
| `masses` | `[N]` | Total masses [kg] (each ≥ its body's `mass_dry`) |

### Defaults for unspecified fields

The loader first draws a random state with `sim.reset()` (randomization
included). Fields not provided in `initial_conditions` keep that drawn
value. When a mass is given without a momentum or velocity, the drawn
velocity is kept and the momentum rescaled to the new mass.
For bodies sampled with `reset.orbit`, positions and momenta (or
velocities) must be given together or not at all: one half of a drawn orbit
state combined with a prescribed other half would lie on a different orbit.

## `randomization`

Vehicle parameters drawn anew, per entity and uniformly, at every reset —
including the automatic resets inside a running loop. How the draws work is
in [Domain randomization](randomization.md); `sim.randomizer.set_bounds(...)`
is the Python equivalent.

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
| `seed` | int | Seed for the randomization RNG streams (separate from `reset.seed`) |
| `mass_dry` | `[min, max]` | Dry mass [kg]; **requires `propellant`** for the same bodies |
| `propellant` | `[min, max]` | Propellant mass [kg]; state mass = dry + propellant at reset |
| `inertia_scale` | `[min, max]` | Scalar multiplier on the body's inertia tensor from the `body` section |
| `per_body` | list | Per-body overrides of the keys above (with `body_id`) |

For a body with `propellant` bounds, the reset mass is
`mass_dry + propellant`: its `reset.mass_bounds` are not used, and not checked
against the dry mass.

## Validation summary

The loader raises `ValueError` naming the field, before building the
simulator where it can, if:

- any section or entry contains an unknown key;
- a value has the wrong type or range: a quoted number or boolean, a
  non-finite number, a negative or zero value where a positive one is needed
  (`dt`, `mass_dry`, `wheel_inertia`, ...), a non-integer count, an unknown
  `integrator`, an inertia tensor no rigid body has, a zero quaternion, a
  thruster direction or wheel axis too short to normalize, or an inclination
  or `att_max_angle` outside `[0, pi]`;
- `physics.j2_perturbation` is `true` while `physics.gravity` is `false`;
- a `body_id` is missing, duplicated where uniqueness is required, or outside
  `[0, num_bodies - 1]`;
- `initial_conditions` mixes the per-body and per-entity forms, or a
  per-entity list has the wrong length;
- `velocity`/`velocities`/`orbit` is given without the corresponding mass, or
  together with `momentum`/`momenta`;
- `initial_conditions` gives only one of positions and momenta/velocities
  while a body uses `reset.orbit` sampling;
- an `initial_conditions` `orbit` spec gives both (or neither of)
  `semi_major_axis` and `altitude`, both `true_anomaly` and `mean_anomaly`,
  or an eccentricity outside `[0, 1)`;
- a `reset.orbit` spec gives both `semi_major_axis` and `altitude`, both
  `true_anomaly` and `mean_anomaly`, an out-of-order `[min, max]` pair, an
  eccentricity outside `[0, 1)`, or — with `check_planet_impact` enabled — a
  worst-case perigee `sma_min * (1 - ecc_max)` at or below
  `termination.planet_radius`;
- an initial mass is below the effective `mass_dry` of the corresponding
  body (using the randomization maximum when `mass_dry` is randomized);
- `randomization` bounds are out of order or non-positive where positivity is
  required, or `mass_dry` is randomized without `propellant`.

Reset `mass_bounds` are checked against the dry mass by `sim.reset()`, which
the loader calls, so the check covers YAML and the Python API alike,
including bounds left at their 1 kg default. `sim.reset()` raises
`ValueError` on every call while the reset mass bounds can sample below a
body's dry mass (bodies with randomized propellant are exempt), or while
orbit-mode bounds allow a perigee at or below the impact radius with
`check_planet_impact` enabled.

## Complete example

`examples/multibody_orbit.yaml`, the default scenario of the
[multi-body orbit example](../examples/09_multi_body_orbit.md): 3
environments × 4 bodies with per-body dry masses, per-body reset mass bounds
and per-body circular-orbit initial conditions.

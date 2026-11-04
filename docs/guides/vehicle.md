# Vehicles

Each spacecraft has a dry mass, an inertia tensor, a drag area and
coefficient, and a residual magnetic dipole. `sim.vehicle.set_config` sets
them for every entity, or for a selection:

```python
import numpy as np
from cosmic_warp.core.pipeline import SpacecraftSimulator

sim = SpacecraftSimulator(num_envs=4)
sim.vehicle.set_config(
    mass_dry=10.0,
    inertia=[[1.0, 0.0, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 0.8]],
    area_drag=0.5,
    cd=2.2,
    magnetic_dipole=[0.0, 0.0, 0.0],
)
sim.reset_manager.set_bounds(mass_bounds=(11.0, 12.0))  # 1 to 2 kg of propellant
sim.reset()
```

| Parameter | Unit | Accepted values |
| --- | --- | --- |
| `mass_dry` | kg | `> 0` |
| `inertia` | kg·m² | 3×3, symmetric, positive definite, principal moments satisfying `I1 + I2 ≥ I3`; not rescaled when `mass_dry` changes |
| `area_drag` | m² | `≥ 0`; used by [drag](physics.md) |
| `cd` | — | `≥ 0`; used by drag |
| `magnetic_dipole` | A·m² | body frame; used by the [magnetic torque](physics.md) |

A value outside these ranges raises `ValueError` naming the field.

**Only the fields you pass are written**; the others keep their current
value for the selected entities, including values set per entity. The
simulator starts with a 1 kg, 1 m cube: `mass_dry` 1, `inertia` `I / 6`,
`area_drag` 1, `cd` 2.2, no magnetic dipole.

## Mass and propellant

The total mass is part of the state, not of the vehicle: each reset samples
it from `mass_bounds` (or sets it, with `reset_to_conditions`). The
propellant is the mass above `mass_dry`. Thrusters use it up and stop when
none is left, with the mass exactly at `mass_dry`; the mass never falls below
`mass_dry`.

- **Reset masses must be at least `mass_dry`.** `sim.reset()` raises
  `ValueError` while a body's `mass_bounds` can sample below its dry mass,
  and `reset_to_conditions` for a mass below it. The default `mass_bounds`
  are those of the default 1 kg vehicle, so after raising `mass_dry`, set
  `mass_bounds` too — as above. With
  [propellant randomization](randomization.md) the reset mass is
  `mass_dry + propellant` instead, and `mass_bounds` are not used. The
  device-side resets (`reset_terminated`, `simulate(auto_reset_terminated=True)`
  and the [RL environment](rl.md)) cannot raise: should the bounds have been
  lowered since, they reset to `mass_dry` rather than below it.
- **Set `mass_dry` before the first reset.** Once entities are reset,
  `set_config` and `set_config_per_entity` raise `ValueError` for a `mass_dry`
  above an entity's current mass.
- **The inertia tensor is fixed.** It does not change as propellant is used.
- **The termination limits do not scale with the vehicle.** The momentum
  and angular-momentum limits are off by default; when you set them, size
  them for the vehicle — see
  [Troubleshooting](../troubleshooting.md#every-environment-terminates-on-the-first-step).

## Per body

`body_ids` configures one body across all environments; `env_ids` and
`entity_ids` (`env_id * num_bodies + body_id`) select further:

```python
sim = SpacecraftSimulator(num_envs=2, num_bodies=3)
for body, mass_dry in enumerate([5.0, 10.0, 15.0]):
    sim.vehicle.set_config(mass_dry=mass_dry, body_ids=[body])
    sim.reset_manager.set_bounds(mass_bounds=(mass_dry, mass_dry + 2.0), body_ids=[body])
sim.reset()
```

In YAML, the [`body` section](configuration.md#body) takes `defaults` and
`per_body` overrides.

## Per entity

`set_config_per_entity` writes one value per entity from arrays, for
dispersions drawn on the host from any distribution. Each argument has length
`num_envs * num_bodies` (inertia: `(N, 3, 3)`); `None` leaves that field
unchanged, as in `set_config`:

```python
sim = SpacecraftSimulator(num_envs=4, num_bodies=3)
rng = np.random.default_rng(0)
n = sim.num_entities
mass_dry = rng.uniform(5.0, 8.0, n)
sim.vehicle.set_config_per_entity(mass_dry=mass_dry)
sim.reset_manager.set_bounds(mass_bounds=(8.0, 9.0))  # above every drawn dry mass
sim.reset()
```

The values are checked like those of `set_config`. They stay fixed across
resets; to draw new values at every reset, use
[domain randomization](randomization.md).

API: [`VehicleManager`](../reference/api/cosmic_warp/core/vehicle.md)

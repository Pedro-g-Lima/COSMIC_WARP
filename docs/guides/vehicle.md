# Vehicles

Each spacecraft has a dry mass, an inertia tensor, a drag area and
coefficient, and a residual magnetic dipole. `sim.vehicle.set_config` sets
them for every entity or a selection:

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

| Parameter | Unit | Accepted values | Default |
| --- | --- | --- | --- |
| `mass_dry` | kg | `> 0` | 1 |
| `inertia` | kg·m² | 3×3, symmetric, positive definite, `I1 + I2 ≥ I3`; not rescaled with `mass_dry` | `I / 6` (1 m cube) |
| `area_drag` | m² | `≥ 0`; used by [drag](physics.md) | 1 |
| `cd` | — | `≥ 0`; used by drag | 2.2 |
| `magnetic_dipole` | A·m² | body frame; used by the [magnetic torque](physics.md) | 0 |

A value outside these ranges raises `ValueError` naming the field. **Only
the fields you pass are written**; the others keep their current values,
including per-entity ones.

## Mass and propellant

The total mass is part of the state: each reset samples it from
`mass_bounds` (or `reset_to_conditions` sets it). Propellant is the mass
above `mass_dry`; thrusters use it up and stop with the mass exactly at
`mass_dry`, never below.

- **Reset masses must be at least `mass_dry`.** `sim.reset()` raises while a
  body's `mass_bounds` can sample below its dry mass, and
  `reset_to_conditions` for a mass below it. The default bounds fit the
  default 1 kg vehicle, so after raising `mass_dry`, raise `mass_bounds`
  too. With [propellant randomization](randomization.md) the reset mass is
  `mass_dry + propellant` and `mass_bounds` is unused. Device-side resets
  (`reset_terminated`, auto-reset, the [RL environment](rl.md)) cannot
  raise: if the bounds were lowered since, they floor the mass at
  `mass_dry`.
- **Set `mass_dry` before the first reset.** Afterwards, `set_config` and
  `set_config_per_entity` reject a `mass_dry` above an entity's current
  mass.
- **The inertia tensor is fixed**: it does not change as propellant is used.

## Per body

`body_ids` configures one body in every environment; `env_ids` and
`entity_ids` (`env_id * num_bodies + body_id`) select further:

```python
sim = SpacecraftSimulator(num_envs=2, num_bodies=3)
for body, mass_dry in enumerate([5.0, 10.0, 15.0]):
    sim.vehicle.set_config(mass_dry=mass_dry, body_ids=[body])
    sim.reset_manager.set_bounds(mass_bounds=(mass_dry, mass_dry + 2.0), body_ids=[body])
sim.reset()
```

In YAML: the [`body` section](configuration.md#body).

## Per entity

`set_config_per_entity` writes one value per entity, for dispersions drawn
on the host from any distribution. Each argument has length
`num_envs * num_bodies` (inertia `(N, 3, 3)`); `None` leaves a field
unchanged:

```python
sim = SpacecraftSimulator(num_envs=4, num_bodies=3)
rng = np.random.default_rng(0)
n = sim.num_entities
mass_dry = rng.uniform(5.0, 8.0, n)
sim.vehicle.set_config_per_entity(mass_dry=mass_dry)
sim.reset_manager.set_bounds(mass_bounds=(8.0, 9.0))  # above every drawn dry mass
sim.reset()
```

The values are checked like `set_config`'s and stay fixed across resets;
for new values at every reset use [domain randomization](randomization.md).

API: [`VehicleManager`](../reference/api/cosmic_warp/core/vehicle.md)

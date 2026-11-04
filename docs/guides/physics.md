# Physics models

Each force and torque model is a constructor switch. The models are
evaluated inside the step kernel.

```python
from cosmic_warp.core.pipeline import SpacecraftSimulator

sim = SpacecraftSimulator(
    num_envs=4,
    gravity=True,
    j2_perturbation=True,
    gravity_gradient=True,
    magnetic_torque=True,
    drag=True,
)
sim.vehicle.set_config(
    mass_dry=1.0,
    inertia=[[0.10, 0.0, 0.0], [0.0, 0.15, 0.0], [0.0, 0.0, 0.20]],
    area_drag=0.01,                  # m², used by drag
    cd=2.2,
    magnetic_dipole=[0.01, 0.0, 0.0],  # A·m², body frame, used by the magnetic torque
)
sim.reset()
sim.simulate(num_steps=100)
```

| Switch | Model | Uses |
| --- | --- | --- |
| `gravity` | point-mass Earth gravity | `MU_EARTH` |
| `j2_perturbation` | J2 oblateness, added to the point-mass term | `J2_EARTH`, `RE_EARTH`; needs `gravity` |
| `gravity_gradient` | gravity-gradient torque | the vehicle's `inertia` |
| `magnetic_torque` | torque of Earth's field on the vehicle's dipole, `m × B` | the vehicle's `magnetic_dipole` |
| `drag` | atmospheric drag force | the vehicle's `area_drag` and `cd` |

`j2_perturbation` follows `gravity` unless you set it; `j2_perturbation=True`
with `gravity=False` raises `ValueError`. The defaults of every switch are in
the [`physics` YAML table](configuration.md#physics). The vehicle parameters
are set as in [Vehicles](vehicle.md) — one call, all fields.

## What each model assumes

- **Gravity and J2.** Earth constants from EGM2008. Earth is the only
  central body: the reset sampler and the YAML orbit specifications convert
  orbits with `MU_EARTH` and take no other `mu`. The gravity-gradient torque uses the same
  `MU_EARTH`, and applies whether or not `gravity` is on.
- **Magnetic field.** A centred dipole aligned with the ECI −Z axis, fixed in
  inertial space: no tilt and no Earth rotation.
- **Drag.** Piecewise-exponential density (Vallado, Table 8-4) on the
  geometric altitude above a spherical Earth, in an atmosphere co-rotating
  about the ECI +Z axis. It does not depend on epoch or solar activity.
  The area is fixed — it does not depend on attitude — and drag makes no
  torque. Below sea level the density stays at its sea-level value.

Not modelled: solar radiation pressure, third bodies, higher-order gravity
harmonics, eclipses. The [Roadmap](../roadmap.md) lists what is planned.

The equations are in [Gravity & J2](../theory/gravity.md) and
[Environment models](../theory/environment.md); the
[Validation](../validation/index.md) page lists what is tested against
analytic results. In YAML: the [`physics` section](configuration.md#physics).

API: [`gravity`](../reference/api/cosmic_warp/physics/gravity.md) ·
[`magnetic`](../reference/api/cosmic_warp/physics/magnetic.md) ·
[`drag`](../reference/api/cosmic_warp/physics/drag.md) ·
[`aggregation`](../reference/api/cosmic_warp/physics/aggregation.md)

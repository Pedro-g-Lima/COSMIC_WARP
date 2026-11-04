# Physics models

Each force and torque model is a constructor switch, evaluated inside the
step kernel:

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

| Switch | Default | Model | Uses |
| --- | --- | --- | --- |
| `gravity` | on | point-mass Earth gravity | `MU_EARTH` |
| `j2_perturbation` | follows `gravity` | J2 oblateness; `True` without `gravity` raises | `J2_EARTH`, `RE_EARTH` |
| `gravity_gradient` | off | gravity-gradient torque, also with `gravity` off | the vehicle's `inertia` |
| `magnetic_torque` | off | `m × B` on the vehicle's dipole | `magnetic_dipole` |
| `drag` | off | atmospheric drag force | `area_drag`, `cd` |

## What each model assumes

| Model | Assumes |
| --- | --- |
| Gravity, J2 | EGM2008 constants. **Earth is the only central body**: resets and YAML orbit specs use `MU_EARTH` and take no other `mu`. |
| Magnetic field | A centred dipole along ECI −Z, fixed in inertial space: no tilt, no Earth rotation, no epoch. |
| Drag | Piecewise-exponential density (Vallado, Table 8-4) on the geometric altitude above a spherical Earth, in an atmosphere co-rotating about ECI +Z; no epoch or solar activity. Fixed area (no attitude dependence), no drag torque; below sea level the density stays at its sea-level value. |

Not modelled: solar radiation pressure, third bodies, higher-order gravity
harmonics, eclipses ([Roadmap](../roadmap.md)).

Equations: [Gravity & J2](../theory/gravity.md),
[Environment models](../theory/environment.md). Evidence:
[Validation](../validation/physics.md). YAML: the
[`physics` section](configuration.md#physics).

API: [`gravity`](../reference/api/cosmic_warp/physics/gravity.md) ·
[`magnetic`](../reference/api/cosmic_warp/physics/magnetic.md) ·
[`drag`](../reference/api/cosmic_warp/physics/drag.md) ·
[`aggregation`](../reference/api/cosmic_warp/physics/aggregation.md)

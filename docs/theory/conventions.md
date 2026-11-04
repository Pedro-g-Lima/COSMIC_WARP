# Frames, state and units

Every kernel, configuration key and result array follows these
conventions.

## Environments, bodies and entities

A simulator holds `num_envs` independent **environments** of `num_bodies`
rigid **bodies** each. Every per-body quantity (state, parameters, commands,
trajectories) is one flat device array of
`num_entities = num_envs * num_bodies` **entities**, environment-major:

```
entity_id = env_id * num_bodies + body_id
```

```python
import numpy as np
from cosmic_warp.core.pipeline import SpacecraftSimulator

sim = SpacecraftSimulator(num_envs=2, num_bodies=3)
sim.reset()

env_id, body_id = 1, 2
entity_id = env_id * sim.num_bodies + body_id
positions = sim.get_positions().numpy()        # [num_entities, 3], ECI [m]
assert positions.shape == (sim.num_entities, 3)
```

Bodies of one environment exert no forces on each other; each entity is
integrated on its own. They share [termination](execution.md#termination)
and, in the RL layer, one row of the per-environment tensors. `env_ids` and
`body_ids` expand to entity ids by the rule above
([`core/entities.py`](../reference/api/cosmic_warp/core/entities.md)).

## Units

SI everywhere, angles in radians: API, YAML, kernels and results.

| Quantity | Unit |
| --- | --- |
| position | m |
| linear momentum / velocity | kg·m/s / m/s |
| angular momentum (body, wheels) | kg·m²/s |
| angular velocity | rad/s |
| mass | kg |
| inertia | kg·m² |
| force / torque | N / N·m |
| magnetic dipole / field | A·m² / T |
| specific impulse | s |
| angles (attitude, orbital elements) | rad |
| thruster command | throttle in [0, 1] (dimensionless) |
| reaction-wheel command | torque, N·m |

## Frames

**Inertial frame (ECI).** Earth-centred, non-rotating. $+z$ is Earth's
rotation axis: J2 is symmetric about it, the atmosphere rotates about it,
the magnetic dipole points along $-z$. There is no simulation epoch, so $x$
and $y$ are not tied to the vernal equinox and the surface has no
longitude. Forces are summed here.

**Body frame.** Fixed in the body, origin at the centre of mass. Torques are
summed here.

| Quantity | Frame |
| --- | --- |
| position $\mathbf{r}$, linear momentum $\mathbf{p}$, velocity | inertial |
| gravity, drag, total force | inertial |
| thruster force (computed in body, rotated to inertial before summing) | body → inertial |
| angular momentum $\mathbf{L}$, angular velocity $\boldsymbol{\omega}$ | body |
| inertia tensor $\mathbf{I}$, thruster geometry, wheel spin axes, dipole $\mathbf{m}_b$ | body |
| all torques (thruster, wheel reaction, gravity gradient, magnetic) | body |
| magnetic field $\mathbf{B}$ | computed inertial, rotated to body |

## State

The dynamic state of an entity is the
[`State`](../reference/api/cosmic_warp/core/state.md) struct,
$(\mathbf{r}, \mathbf{p}, m, q, \mathbf{L})$, plus the wheel momenta
$h_{0..3}$ in [`ActuatorState`](../reference/api/cosmic_warp/actuators/base.md).
It carries **momenta, not velocities**:

| Field | Symbol | Meaning |
| --- | --- | --- |
| `pos` | $\mathbf{r}$ | position, inertial [m] |
| `mom` | $\mathbf{p} = m\mathbf{v}$ | linear momentum, inertial [kg·m/s] |
| `mass` | $m$ | total mass, dry + propellant [kg] |
| `quat` | $q = (q_x, q_y, q_z, q_w)$ | attitude, body → inertial |
| `ang_mom` | $\mathbf{L} = \mathbf{I}\boldsymbol{\omega}$ | angular momentum of the body **alone** (wheel momenta excluded), body frame [kg·m²/s] |
| `rw_ang_mom` | $h_k$ | momentum of wheel $k$ about its spin axis [kg·m²/s] |

Velocities are derived: $\mathbf{v} = \mathbf{p} / \max(m, m_{dry})$,
$\boldsymbol{\omega} = \mathbf{I}^{-1}\mathbf{L}$ (the velocity getters and
trajectories). Inputs named `momenta`, `angular_momenta`, `mom_bounds`,
`ang_mom_bounds`, and the limits `mom_limit` and `ang_mom_limit`, refer to
$\mathbf{p}$ and the body-only $\mathbf{L}$.

Other symbols used across the theory pages:

| Symbol | Meaning |
| --- | --- |
| $\mathbf{h} = \sum_k h_k \hat{\mathbf{a}}_k$ | total wheel momentum, body frame; $\hat{\mathbf{a}}_k$ the spin axes |
| $m_{dry}$ | dry mass; propellant is $m - m_{dry}$ |
| $\mu$ | gravitational parameter $GM$ of the Earth [m³/s²] |
| $R_\oplus$ | Earth equatorial radius [m] |
| $\Delta t$ | integrator step `dt` [s] |

## Attitude

The unit quaternion $q$ rotates body coordinates into inertial ones,
$\mathbf{v}_I = q \otimes \mathbf{v}_b \otimes q^*$ (in Warp,
`wp.quat_rotate(q, v_b)`; the inverse rotation is
`wp.quat_rotate(wp.quat_inverse(q), v_I)`). Storage order is
**$(x, y, z, w)$, scalar last**, Warp's `wp.quatd` layout, and the product is
Hamilton's,
$(\mathbf{a}, a_w) \otimes (\mathbf{b}, b_w) = (a_w\mathbf{b} + b_w\mathbf{a} + \mathbf{a} \times \mathbf{b},\ a_w b_w - \mathbf{a}\cdot\mathbf{b})$.

```python
def rotate(q, v):
    """Body -> inertial: q (x, y, z, w) applied to v."""
    u, w = np.asarray(q[:3]), q[3]
    v = np.asarray(v)
    return v + 2.0 * w * np.cross(u, v) + 2.0 * np.cross(u, np.cross(u, v))

q = sim.get_quaternions().numpy()[entity_id]   # (x, y, z, w)
x_body_in_eci = rotate(q, [1.0, 0.0, 0.0])      # the body x axis, ECI
assert np.isclose(np.linalg.norm(x_body_in_eci), 1.0)
```

The integrators renormalize $q$ after every update. $q$ and $-q$ are the
same attitude and the state is not canonicalized; do it where a unique
representative matters (the example attitude task observes $q_w \ge 0$).

## Precision

**Float64** throughout: state, physics and integrator kernels, adjoints,
and the uniform draws behind resets and randomization (53 random bits).
Only `VecEnv`'s actions, observations and rewards are float32.

## Architecture

`cosmic_warp.rl` (`VecEnv`, `DiffVecEnv`, the adapters) and
`DifferentiableRollout` drive a `SpacecraftSimulator`. Its managers
(vehicles, thrusters, wheels, resets, randomization, termination) write the
configuration into device arrays, which
[`SimContext`](../reference/api/cosmic_warp/core/context.md) bundles for the
integrator's one kernel launch per step
([GPU execution](execution.md#the-step-on-the-device)).

**Implementation:**
[`core/state.py`](../reference/api/cosmic_warp/core/state.md) ·
[`core/entities.py`](../reference/api/cosmic_warp/core/entities.md) ·
[`core/pipeline.py`](../reference/api/cosmic_warp/core/pipeline.md)

# Initial-condition sampling

How every reset path draws an entity's state
$(\mathbf{r}, \mathbf{p}, m, q, \mathbf{L})$: bounds per body, one sample per
entity, on the device.

## Random streams

Draws are counter-based: draw $n$ of entity $i$ is Philox4x32-10
([[23]](../references.md#ref-23)) of the
counter $(n, i, \text{stream})$ under the seed as its 64-bit key
([`core/rng.py`](../reference/api/cosmic_warp/core/rng.md)). The draw count
$n$ lives in device memory and advances with every reset: fresh samples
each reset, reproducible from the seed, no host involvement (graph-safe).
Distinct entities, streams or draws are distinct counters, so streams never
overlap. Each draw is a float64 uniform on $[0, 1)$ with 53 random bits.

| | Stream | Default seed | Draws per reset |
| --- | --- | --- | --- |
| Reset sampler | its own id | `42` | 13 at fixed slots, whatever the mode: six elements or the position and momentum boxes, angular momentum, three for attitude, mass |
| [Domain randomization](../guides/randomization.md) | another id | `1234` | after the state is drawn |

The two never share a draw, even under equal seeds. `reseed(seed)` restarts
a stream at draw 0.

## Cartesian box mode (default)

Each component of position, momentum (or velocity, then
$\mathbf{p} = m\,\mathbf{v}$ with the sampled mass) and angular momentum is
uniform in an axis-aligned box; the mass is uniform in its bounds. Defaults:
a 1 kg body within ±1 km of $(R_\oplus + 400\ \mathrm{km}, 0, 0)$ at the
circular speed along $+y$ ±1 m/s, angular momentum within ±0.1 kg·m²/s per
axis, uniformly random attitude. Position and velocity are independent, so
wide boxes give crashing or escaping trajectories: use the orbital-element
mode for dispersions over orbits.

## Orbital-element mode

`set_orbit_bounds(...)` switches a body to drawing the classical elements,
each uniformly within its bounds ([[2]](../references.md#ref-2)):

| Element | Symbol | Bounds |
| --- | --- | --- |
| Semi-major axis | $a$ | `semi_major_axis_bounds` [m], or `altitude_bounds` with $a = R_\oplus + h$ |
| Eccentricity | $e$ | `eccentricity_bounds`, $0 \le e < 1$ (bound orbits only) |
| Inclination | $i$ | `inclination_bounds` [rad], within $[0, \pi]$ |
| Right ascension of the ascending node | $\Omega$ | `raan_bounds` [rad] |
| Argument of periapsis | $\omega$ | `arg_periapsis_bounds` [rad] |
| True or mean anomaly | $\nu$ / $M$ | `true_anomaly_bounds` or `mean_anomaly_bounds` [rad] |

The position and momentum boxes are ignored; angular momentum, mass and
attitude are drawn as in box mode. The elements are osculating at the reset
instant (with J2 on, the mean elements differ). With the semi-latus rectum
$p = a(1-e^2)$ and $r = p / (1 + e\cos\nu)$, the perifocal-frame state is

$$
\mathbf{r}_{pf} = \begin{bmatrix} r\cos\nu \\ r\sin\nu \\ 0 \end{bmatrix},
\qquad
\mathbf{v}_{pf} = \sqrt{\frac{\mu}{p}}
\begin{bmatrix} -\sin\nu \\ e + \cos\nu \\ 0 \end{bmatrix},
$$

rotated into the inertial frame by
$\mathbf{R} = \mathbf{R}_3(-\Omega)\,\mathbf{R}_1(-i)\,\mathbf{R}_3(-\omega)$
([[2]](../references.md#ref-2), [[3]](../references.md#ref-3)), and
$\mathbf{p} = m\,\mathbf{v}$ with the sampled mass, so every sample lies on
its drawn orbit; $\mu$ is `MU_EARTH`.

```python
import numpy as np
from cosmic_warp.core.pipeline import SpacecraftSimulator
from cosmic_warp.physics.orbits import state_to_elements

sim = SpacecraftSimulator(num_envs=16)
sim.reset_manager.set_orbit_bounds(
    altitude_bounds=(400e3, 600e3), eccentricity_bounds=(0.0, 0.05),
    inclination_bounds=(0.9, 1.0), mean_anomaly_bounds=(0.0, 2.0 * np.pi),
)
sim.reset()

pos, vel = sim.get_positions().numpy(), sim.get_velocities().numpy()
elements = [state_to_elements(r, v) for r, v in zip(pos, vel)]
assert all(0.9 - 1e-6 <= el["inclination"] <= 1.0 + 1e-6 for el in elements)
```

`state_to_elements` (host, NumPy [[19]](../references.md#ref-19)) is the
inverse. It raises for rectilinear ($h = 0$) and unbound ($e \ge 1$) states;
equatorial orbits report $\Omega = 0$, circular ones $\omega = 0$ with $\nu$
from the node.

### True vs. mean anomaly

Uniform *true* anomaly is uniform in angle; on an eccentric orbit a
spacecraft seen at a random time is mostly near apoapsis (Kepler's second
law). Uniform *mean* anomaly $M$ is uniform in time. The kernel solves

$$
M = E - e\sin E
$$

for the eccentric anomaly $E$ by Newton iteration from Danby's starter
$E_0 = M + 0.85\,e\,\operatorname{sign}(\sin M)$
([[22]](../references.md#ref-22)), with a fixed 16 iterations (graph-safe;
residual below $10^{-13}$ rad for $e \le 0.9999$), and

$$
\nu = 2\,\operatorname{atan2}\!\left(\sqrt{1+e}\,\sin\tfrac{E}{2},\;
\sqrt{1-e}\,\cos\tfrac{E}{2}\right).
$$

### Density caveats

Uniform in each element is a *modelling choice*, not uniform in space:

- **Inclination**: uniform $i$ concentrates orbit normals near the poles of
  the sphere; isotropic orbit planes need $\cos i$ uniform on $[-1, 1]$.
- **Anomaly**: uniform $\nu$ oversamples the periapsis region relative to
  time; use the mean anomaly for time-uniform sampling (above).
- **Perigee safety**: with `check_planet_impact` on, bounds whose
  worst-case perigee $a_{\min}(1 - e_{\max})$ does not clear
  `planet_radius` are rejected.

## Attitude sampling

Attitudes are drawn uniformly (Haar measure) within a geodesic ball of
radius $\theta_{\max}$ (`att_max_angle`) around a nominal quaternion $q_0$
(`att_nominal`):

- $\theta_{\max} = \pi$ (the default, and the largest accepted): uniform
  over all of SO(3), by Shoemake's subgroup algorithm
  ([[21]](../references.md#ref-21)).
- $\theta_{\max} < \pi$: a uniform random axis $\hat{\mathbf{n}}$ and a
  rotation angle $\theta$ drawn from the SO(3) Haar density
  $f(\theta) \propto \sin^2(\theta/2)$ on $[0, \theta_{\max}]$, whose CDF
  $F(\theta) \propto \theta - \sin\theta$ is inverted by 32 bisection steps;
  the attitude is $q = q_0 \otimes \delta q(\hat{\mathbf{n}}, \theta)$.
- $\theta_{\max} = 0$: every reset yields exactly $q_0$.

A uniformly drawn angle would instead cluster samples near the nominal
attitude, and would not tend to the uniform distribution as
$\theta_{\max} \to \pi$.

**Implementation:**
[`core/reset.py`](../reference/api/cosmic_warp/core/reset.md)
(`ResetManager`, `_sample_state`) ·
[`physics/orbits.py`](../reference/api/cosmic_warp/physics/orbits.md)
(`elements_to_rv`, `mean_to_true_anomaly`, `state_to_elements`) ·
[`core/rng.py`](../reference/api/cosmic_warp/core/rng.md) · usage in the
[resets guide](../guides/reset.md)

**References:** [[2]](../references.md#ref-2),
[[3]](../references.md#ref-3), [[19]](../references.md#ref-19),
[[21]](../references.md#ref-21), [[22]](../references.md#ref-22),
[[23]](../references.md#ref-23) — see the
full [bibliography](../references.md).

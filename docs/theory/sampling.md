# Initial-condition sampling

How `sim.reset()`, `reset_terminated()` and the RL auto-reset draw the
dynamic state $(\mathbf{r}, \mathbf{p}, m, q, \mathbf{L})$ of an entity. The
bounds are set per body
([`ResetManager`](../reference/api/cosmic_warp/core/reset.md)) and every
entity of that body draws its own sample, entirely on the device.

## Random streams

Draws are counter-based: draw $n$ of entity $i$ is Philox4x32-10
([[23]](../references.md#ref-23)) of the
counter $(n, i, \text{stream})$ under the seed as its 64-bit key
([`core/rng.py`](../reference/api/cosmic_warp/core/rng.md)). Each entity's
draw count $n$ lives in device memory and advances with every reset, so
repeated resets give fresh samples, the same seed reproduces them, and the
draws need no host involvement, which keeps them valid inside CUDA graphs.
Distinct entities, streams or draws are distinct counters, so no two
streams ever overlap. Each draw is a float64 uniform on $[0, 1)$ with 53
random bits, mapped onto the bounds.

Every reset consumes 13 draws of its entity's stream at fixed slots, whatever
the mode: the six orbital elements or the position and momentum boxes, the
angular momentum, three for the attitude, the mass. The seed is
`seed=42` unless configured; `reset_manager.reseed(seed)` restarts the
streams at draw 0.

Domain randomization ([guide](../guides/randomization.md)) draws from its
own stream id (`seed=1234` unless configured), so the two never share a
draw, even under equal seeds; it runs after the state is drawn.

## Cartesian box mode (default)

Each component of position, of momentum (or of velocity, see below) and of
angular momentum is drawn independently and uniformly within an axis-aligned
box; the mass is uniform within its bounds. With `vel_bounds` instead of
`mom_bounds`, the drawn vector is a velocity and the momentum is
$\mathbf{p} = m\,\mathbf{v}$ with the entity's sampled mass, so the bounds
hold whatever the mass. The defaults are a 1 kg body within ±1 km of
$(R_\oplus + 400\ \mathrm{km}, 0, 0)$ moving at the circular speed along $+y$
±1 m/s, with body angular momentum within ±0.1 kg·m²/s per axis and a
uniformly random attitude.

Position and velocity are drawn independently: the states belong to no
particular orbit family, and wide boxes produce crashing or escaping
trajectories. Use the orbital-element mode for dispersions over orbits.

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

The position and momentum boxes are ignored in this mode; angular momentum,
mass and attitude are drawn as in box mode. The elements are osculating
(two-body) elements at the reset instant, converted to an inertial state
inside the reset kernel; with J2 on, the mean elements of the resulting orbit
differ from them. With the semi-latus rectum
$p = a(1-e^2)$ and $r = p / (1 + e\cos\nu)$, the perifocal-frame state is

$$
\mathbf{r}_{pf} = \begin{bmatrix} r\cos\nu \\ r\sin\nu \\ 0 \end{bmatrix},
\qquad
\mathbf{v}_{pf} = \sqrt{\frac{\mu}{p}}
\begin{bmatrix} -\sin\nu \\ e + \cos\nu \\ 0 \end{bmatrix},
$$

rotated into the inertial frame by
$\mathbf{R} = \mathbf{R}_3(-\Omega)\,\mathbf{R}_1(-i)\,\mathbf{R}_3(-\omega)$
([[2]](../references.md#ref-2), [[3]](../references.md#ref-3)). The momentum
is $\mathbf{p} = m\,\mathbf{v}$ with the entity's sampled mass, so every
sample lies exactly on its drawn orbit. $\mu$ is `MU_EARTH`, the value the
dynamics use; orbit specs take no other.

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

`state_to_elements` (host side, NumPy [[19]](../references.md#ref-19)) is
the inverse conversion. It raises
for rectilinear motion (zero angular momentum) and for unbound states
($e \ge 1$), where the elements of an ellipse are undefined; for equatorial
orbits it reports $\Omega = 0$, and for circular ones $\omega = 0$ with
$\nu$ measured from the node.

### True vs. mean anomaly

Uniform *true* anomaly places samples uniformly in angle around the orbit.
On an eccentric orbit that is **not** where a spacecraft observed at a random
time is found: by Kepler's second law it spends most of the period near
apoapsis. Uniform *mean* anomaly $M$ gives the time-uniform ("random epoch")
distribution. The kernel then solves Kepler's equation

$$
M = E - e\sin E
$$

for the eccentric anomaly $E$ by Newton iteration from Danby's starter
$E_0 = M + 0.85\,e\,\operatorname{sign}(\sin M)$
([[22]](../references.md#ref-22)), with a fixed count of 16 iterations (no
data-dependent loop, so graph-safe). The residual in $M$ is at rounding level
(below $10^{-13}$ rad for $e \le 0.9999$), and the true anomaly is

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
- **Perigee safety**: the worst-case perigee over the bounds is
  $a_{\min}(1 - e_{\max})$. With `check_planet_impact` on and a positive
  `planet_radius`, `sim.reset()` and the YAML loader reject bounds whose
  worst-case perigee does not clear that radius.

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

The $\sin^2(\theta/2)$ weighting matters: an angle drawn uniformly would
cluster samples near the nominal attitude relative to the Haar measure, and
the ball distribution would not tend to the uniform one as
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

# Environment models

The magnetic field and the atmosphere: the environmental perturbations
beyond gravity. Both are epoch-free: the Earth does not rotate under the
inertial frame, except that the atmosphere co-rotates with it.

## Magnetic dipole field

Earth's field is a centred dipole ([[9]](../references.md#ref-9),
[[11]](../references.md#ref-11)), computed in the inertial frame:

$$
\mathbf{B}(\mathbf{r}) = B_0 \left(\frac{R_\oplus}{r}\right)^{3}
\left[\, 3(\hat{\mathbf{m}} \cdot \hat{\mathbf{r}})\,\hat{\mathbf{r}}
- \hat{\mathbf{m}} \,\right]
$$

(zero at $r = 0$), with equatorial surface strength
$B_0 = 3.12 \times 10^{-5}\ \mathrm{T}$ ([[9]](../references.md#ref-9)) and
$R_\oplus = 6\,378\,136.3\ \mathrm{m}$, the same radius as the gravity model
(`RE_MAG = RE_EARTH`). The dipole axis $\hat{\mathbf{m}} = -\hat{\mathbf{z}}$
is fixed in the inertial frame: magnetic north is at the geographic south
pole, so at the equator $\mathbf{B}$ points along $+\hat{\mathbf{z}}$
(northward) with magnitude $B_0 (R_\oplus/r)^3$, and over the poles it is
twice as strong. The ~11° tilt of the real dipole and its rotation with the
Earth are not modelled; they need a simulation epoch (see the
[roadmap](../roadmap.md)).

### Magnetic torque

With `magnetic_torque=True` (off by default), a residual dipole
$\mathbf{m}_b$ [A·m²] (body frame, `magnetic_dipole` on the vehicle) feels
([[9]](../references.md#ref-9))

$$
\boldsymbol{\tau}_{mag} = \mathbf{m}_b \times \mathbf{B}_b,
\qquad
\mathbf{B}_b = R(q)^{\top}\,\mathbf{B}(\mathbf{r}),
$$

the field rotated into the body frame.

**Implementation:**
[`physics/magnetic.py`](../reference/api/cosmic_warp/physics/magnetic.md)
(`earth_magnetic_field`, `magnetic_torque`)

## Atmospheric drag

Density follows the piecewise-exponential model of Vallado, Table 8-4
([[2]](../references.md#ref-2)): 28 bands with base altitudes $h_0$ from 0 to
1000 km, each with a nominal density $\rho_0$ and a scale height $H$
(`DENSITY_TABLE`):

$$
\rho(h) = \rho_0 \, e^{-(h - h_0)/H},
\qquad h = r - R_\oplus,
$$

using the band with the largest $h_0 \le h$. Altitude is geometric above a
spherical Earth. Below sea level ($h < 0$) the density is held at the
sea-level value $1.225\ \mathrm{kg/m^3}$ rather than extrapolated: a
trajectory inside the Earth keeps a finite drag force (stop it with the
`planet_radius` termination), and the density has zero gradient with respect
to position there. Above 1000 km the last band ($H = 268$ km) is
extrapolated. At a band edge the density jumps where the table's nominal
$\rho_0$ does not exactly continue the band below (by 0.14 % at 25 km, under
0.01 % elsewhere), and its slope changes with the scale height.

The atmosphere co-rotates with the Earth at
$\omega_\oplus = 7.2921159 \times 10^{-5}\ \mathrm{rad/s}$ about inertial
$+\hat{\mathbf{z}}$ (the sidereal rate, [[2]](../references.md#ref-2); the
IERS nominal value is $7.292115 \times 10^{-5}$,
[[4]](../references.md#ref-4)), so drag acts on the velocity relative to the
air:

$$
\mathbf{v}_{rel} = \frac{\mathbf{p}}{m} - \omega_\oplus\hat{\mathbf{z}} \times \mathbf{r},
\qquad
\mathbf{F}_{drag} = -\tfrac{1}{2}\, \rho\, C_d A\,
\lVert \mathbf{v}_{rel} \rVert\, \mathbf{v}_{rel}.
$$

$C_d$ and the cross-sectional area $A$ are per-vehicle constants (`cd`,
`area_drag`, default 2.2 and 1 m²), independent of attitude. Drag is enabled
by `drag=True` (off by default). For a circular orbit whose velocity is
parallel to $\mathbf{v}_{rel}$ (equatorial, prograde), the secular decay of
the semi-major axis is

$$
\frac{da}{dt} = -\rho\, \frac{C_d A}{m}\, v_{rel}^2 \sqrt{\frac{a^3}{\mu}},
$$

checked against the simulator at 400 km in the test suite.

**Implementation:**
[`physics/drag.py`](../reference/api/cosmic_warp/physics/drag.md)
(`atmosphere_density`, `drag_force`)

**References:** [[2]](../references.md#ref-2), [[4]](../references.md#ref-4),
[[9]](../references.md#ref-9), [[11]](../references.md#ref-11) — see the full
[bibliography](../references.md).

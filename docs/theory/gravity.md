# Gravity & J2

Earth's constants come from EGM2008
([[1]](../references.md#ref-1)) and are module constants of
[`physics/gravity.py`](../reference/api/cosmic_warp/physics/gravity.md):

| Constant | Value |
| --- | --- |
| `MU_EARTH`, $\mu$ | $3.986004415 \times 10^{14}\ \mathrm{m^3/s^2}$ |
| `RE_EARTH`, $R_\oplus$ | $6\,378\,136.3\ \mathrm{m}$ |
| `J2_EARTH`, $J_2$ | $1.08262617385222 \times 10^{-3}$ |

Earth is the only central body ([physics models](../guides/physics.md#what-each-model-assumes)).

## Point-mass gravity

With `gravity=True` (the default) the force is $m\,\mathbf{a}_g$, with $m$
the current total mass and

$$
\mathbf{a}_{g} = -\frac{\mu}{r^3}\,\mathbf{r},
\qquad r = \lVert \mathbf{r} \rVert
$$

(zero at $r = 0$).

## J2 oblateness perturbation

Earth's equatorial bulge is the second zonal harmonic. With
`j2_perturbation=True` (the default whenever gravity is on) the perturbing
acceleration added to $\mathbf{a}_g$ is, in inertial coordinates with $z$
along Earth's rotation axis ([[2]](../references.md#ref-2),
[[3]](../references.md#ref-3)),

$$
\mathbf{a}_{J_2} = \frac{3}{2} J_2 \frac{\mu R_\oplus^2}{r^4}
\begin{bmatrix}
\dfrac{x}{r}\left(5\dfrac{z^2}{r^2} - 1\right) \\[2ex]
\dfrac{y}{r}\left(5\dfrac{z^2}{r^2} - 1\right) \\[2ex]
\dfrac{z}{r}\left(5\dfrac{z^2}{r^2} - 3\right)
\end{bmatrix}.
$$

Its dominant secular effects are the regression of the node and the
rotation of the line of apsides ([[2]](../references.md#ref-2)),

$$
\dot{\Omega} = -\frac{3}{2}\, n J_2 \left(\frac{R_\oplus}{p}\right)^2 \cos i,
\qquad
\dot{\omega} = \frac{3}{4}\, n J_2 \left(\frac{R_\oplus}{p}\right)^2 (5\cos^2 i - 1),
$$

with $n = \sqrt{\mu/a^3}$ and $p = a(1-e^2)$
([checked](../validation/physics.md#j2-oblateness)). The
[J2 example](../examples/08_j2_comparison.md) shows the short-period
effects.

## Gravity-gradient torque

A body with non-spherical inertia in the central field feels, in the body
frame ([[5]](../references.md#ref-5), [[3]](../references.md#ref-3)),

$$
\boldsymbol{\tau}_{gg} = \frac{3\mu}{r^3}\,
\hat{\mathbf{r}}_b \times \left(\mathbf{I}\, \hat{\mathbf{r}}_b\right),
\qquad
\hat{\mathbf{r}}_b = R(q)^{\top}\,\frac{\mathbf{r}}{r},
$$

It drives the minimum-inertia axis towards the local vertical. Enabled by
`gravity_gradient=True`, independently of `gravity`.

**Implementation:**
[`physics/gravity.py`](../reference/api/cosmic_warp/physics/gravity.md)
(`gravity_accel`, `gravity_accel_j2`, `gravity_force`,
`gravity_gradient_torque`)

**References:** [[1]](../references.md#ref-1), [[2]](../references.md#ref-2),
[[3]](../references.md#ref-3), [[5]](../references.md#ref-5) — see the full
[bibliography](../references.md).

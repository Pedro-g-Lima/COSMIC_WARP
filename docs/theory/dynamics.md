# Rigid-body dynamics

Each entity is a 6-DOF rigid body with state
$(\mathbf{r}, \mathbf{p}, m, q, \mathbf{L})$ and up to four reaction-wheel
momenta $h_k$ (frames and symbols in [Frames, state and units](conventions.md)).
The right-hand side below is `compute_state_derivative` in
[`core/integrators/base.py`](../reference/api/cosmic_warp/core/integrators/base.md),
shared by the Euler and RK4 kernels; the symplectic kernel splits the same
equations into kicks and a drift ([integrators](integrators.md)).

## Translational motion

In the inertial frame,

$$
\dot{\mathbf{r}} = \frac{\mathbf{p}}{\tilde m},
\qquad
\dot{\mathbf{p}} = \mathbf{F} + \dot{m}\,\frac{\mathbf{p}}{\tilde m},
\qquad
\tilde m = \max(m, m_{dry}),
$$

with the total force

$$
\mathbf{F} = m\,\mathbf{a}_g(\mathbf{r}) + \mathbf{F}_{drag}(\mathbf{r}, \mathbf{v})
+ R(q)\,\mathbf{F}_{thr,b},
$$

assembled in
[`physics/aggregation.py`](../reference/api/cosmic_warp/physics/aggregation.md):
gravity and J2 ([gravity](gravity.md)), drag
([environment](environment.md#atmospheric-drag)) and the body-frame thruster
force rotated into the inertial frame by $R(q)$
([actuators](actuators.md#thrusters)). Each term is present only when its
flag is on. The $\dot m\,\mathbf{v}$ term (Meshchersky) accounts for the
expelled propellant, which leaves at the vehicle velocity; the thrust it
produces is already in $\mathbf{F}$. Together they give
$m\,\dot{\mathbf{v}} = \mathbf{F}$, so a burn reproduces the rocket equation.
The floor $\tilde m$ only guards the division: the mass never falls below
$m_{dry}$, since every reset floors it there and every step ends at the exact
step-end mass (see [burnout](#variable-mass)).

## Rotational motion

The state stores the angular momentum $\mathbf{L}$ of the **body alone**, in
the body frame, so $\boldsymbol{\omega} = \mathbf{I}^{-1}\mathbf{L}$ with
$\mathbf{I}$ the constant body inertia. The wheels add
$\mathbf{h} = \sum_k h_k\,\hat{\mathbf{a}}_k$ along their unit spin axes
$\hat{\mathbf{a}}_k$. Euler's equation for the body with its wheels
([[5]](../references.md#ref-5), [[6]](../references.md#ref-6)) is

$$
\dot{\mathbf{L}} = \boldsymbol{\tau}_{ext} - \dot{\mathbf{h}}
- \boldsymbol{\omega} \times \left(\mathbf{L} + \mathbf{h}\right),
$$

where $\boldsymbol{\tau}_{ext}$ is the sum of the thruster, gravity-gradient
and magnetic torques in the body frame, $-\dot{\mathbf{h}}$ is the reaction
of the wheel motor torques on the body, and the gyroscopic term accounts for
expressing the total angular momentum $\mathbf{L} + \mathbf{h}$ in the
rotating frame. Each wheel follows

$$
\dot h_k = \operatorname{clamp}(\tau_{cmd,k}, \pm\tau_{max}) - \frac{b}{J_w}\,h_k,
\qquad |h_k| \le h_{max},
$$

solved in closed form within the step ([actuators](actuators.md#reaction-wheels)).
The integrators never form $\dot{\mathbf{h}}$: they subtract the wheel
momentum change $\Delta\mathbf{h}$ over each (sub)step from $\mathbf{L}$,
which exchanges momentum exactly.

## Quaternion kinematics

With $q$ mapping body to inertial coordinates and $\boldsymbol{\omega}$ in
the body frame ([[10]](../references.md#ref-10),
[[7]](../references.md#ref-7)):

$$
\dot{q} = \tfrac{1}{2}\, q \otimes
\begin{bmatrix} \boldsymbol{\omega} \\ 0 \end{bmatrix}.
$$

$q$ is normalized on every reset (`reset_to_conditions` normalizes a given
quaternion), before the product and after every update.

## Variable mass

Thrusters consume propellant at the mass-flow rate
([[16]](../references.md#ref-16))

$$
\dot{m} = -\frac{T_{max} \sum_i u_i}{I_{sp}\, g_0},
\qquad g_0 = 9.80665\ \mathrm{m/s^2},
$$

summed over the configured thrusters with throttles $u_i \in [0, 1]$. With
$I_{sp} = 0$ (the default) the thrusters are ideal and consume nothing.
Whatever $I_{sp}$, thrusters produce no force, torque or flow once
$m \le m_{dry}$: an ideal thruster also needs $m > m_{dry}$ to fire (the
default reset mass equals the default dry mass, 1 kg).

**Burnout within a step.** Thrust, thruster torque and $\dot m$ are evaluated
once per step, from the commands and the propellant at the step start
(`thruster_wrench` in
[`actuators/thrusters.py`](../reference/api/cosmic_warp/actuators/thrusters.md));
in the body frame they do not depend on the rest of the state. On the step
where the propellant would run out after $t_b < \Delta t$, all three are
scaled by the burn fraction $f = t_b / \Delta t$, so the mass reaches
$m_{dry}$ exactly at the end of the step: every integrator stores the
step-end mass $m - \Delta t\,\dot m$ rounded once, and $m_{dry}$ itself on
the burnout step, so `mass_limit = mass_dry` fires on that step. This
stretches the burn in time without changing the impulse: with a fixed thrust
direction, the delta-v and the propellant used are exact. The position
differs from an exact cutoff at $t_b$ by $O(\Delta t^2)$, once per burnout:
to leading order it lags by
$\tfrac{T}{m_{dry}}\,\Delta t^2 f(1-f)/2$, at most
$T\,\Delta t^2 / (8\,m_{dry})$ with $T$ the total thrust (checked against
the analytic rocket trajectory in the test suite); the attitude differs at
the same order through the thruster torque. Every integrator uses this
model.

!!! warning "Constant mass properties"
    The inertia tensor and the centre of mass do not change as propellant is
    consumed. Only the scalar mass varies.

!!! note
    While thrusters fire, the system is not Hamiltonian and energy is
    (correctly) not conserved — see the
    [symplectic integrator](integrators.md#symplectic-kickdriftkick).

**Implementation:**
[`core/integrators/base.py`](../reference/api/cosmic_warp/core/integrators/base.md)
(`compute_state_derivative`, `state_add_scaled`) ·
[`physics/aggregation.py`](../reference/api/cosmic_warp/physics/aggregation.md) ·
[`actuators/thrusters.py`](../reference/api/cosmic_warp/actuators/thrusters.md)
(`thruster_wrench`)

**References:** [[5]](../references.md#ref-5), [[6]](../references.md#ref-6),
[[7]](../references.md#ref-7), [[10]](../references.md#ref-10),
[[16]](../references.md#ref-16) — see the full
[bibliography](../references.md).

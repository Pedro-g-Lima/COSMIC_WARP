# Actuator models

Commands live in one
[`ActuatorCommand`](../reference/api/cosmic_warp/actuators/base.md) per
entity: eight thruster throttles (`thrust_cmd` for thrusters 0–3,
`thrust_cmd_ext` for 4–7) and four wheel torques (`rw_torque_cmd`, N·m).
They are held constant over a step. Every reset of an entity zeroes its
commands and wheel momenta. A NaN or Inf command, which the clamps below
would turn into a finite actuation (a NaN wheel torque into the full negative
torque), instead ends the environment before the step with the non-finite
reason ([termination](execution.md#termination)).

## Reaction wheels

Each body has up to four wheels with unit spin axes $\hat{\mathbf{a}}_k$ in
the body frame (normalized when configured; a zero axis leaves the wheel
unused) and shared parameters: torque limit $\tau_{max}$, momentum capacity
$h_{max}$ (0 = unlimited), wheel inertia $J_w$ and viscous friction $b$
([[8]](../references.md#ref-8), [[7]](../references.md#ref-7)). The state of
wheel $k$ is its angular momentum $h_k$ about its axis; its speed is
reported as $h_k / J_w$. The motor torque is the clamped command
$\tau_k = \operatorname{clamp}(\tau_{cmd,k}, -\tau_{max}, \tau_{max})$, and

$$
\dot h_k = \tau_k - \frac{b}{J_w}\, h_k .
$$

Over an interval $s$ with constant command this has the exact solution

$$
h_k(s) = h_k(0) + \left(h_{ss} - h_k(0)\right)\left(1 - e^{-b s / J_w}\right),
\qquad h_{ss} = \frac{J_w \tau_k}{b},
$$

or $h_k(0) + \tau_k s$ when $b = 0$, stable for any step size. It is
evaluated as the undamped step scaled by a relaxation factor,

$$
h_k(s) = h_k(0) + \left(\tau_k - \frac{b}{J_w} h_k(0)\right) s\,
\frac{1 - e^{-x}}{x}, \qquad x = \frac{b s}{J_w},
$$

with $(1 - e^{-x})/x$ computed without cancellation for small $x$, so a
vanishing $b$ converges to the undamped wheel. When
$h_{max} > 0$ the result is clamped to $[-h_{max}, h_{max}]$: a saturated
wheel absorbs no further momentum in that direction, while torquing away
from saturation still works ([[8]](../references.md#ref-8)). Wheel momenta
above $h_{max}$ are rejected when set, and so is an $h_{max}$ lowered below a
stored wheel momentum (the next step would dump the excess into the body
at once).

The integrators evaluate this solution at the times they need (the step end
for Euler; $\Delta t/2$ and $\Delta t$ for RK4; two half steps for the
symplectic scheme) and subtract the change of the wheel momentum vector

$$
\Delta\mathbf{h} = \sum_k \left(h_k(s) - h_k(0)\right) \hat{\mathbf{a}}_k
$$

from the body angular momentum $\mathbf{L}$, so the total $\mathbf{L} + \mathbf{h}$
changes only through external torques and the gyroscopic term
([dynamics](dynamics.md#rotational-motion)). The reaction torque on the body
is $-\dot{\mathbf{h}}$. Saturation inside a step is not located: the clamp
is exact for the wheel, but the body sees the momentum change spread over the
step, which caps every integrator at second order around it
([integrators](integrators.md#events-inside-a-step)).

See the [reaction-wheel example](../examples/07_reaction_wheels.md) for the
spin-up response and the body counter-rotation.

## Thrusters

Up to eight thrusters per body, each with a mounting position
$\mathbf{d}_i$ relative to the centre of mass and a unit direction
$\hat{\mathbf{u}}_i$ (the direction of the force on the body, not of the
exhaust), both in the body frame. All thrusters of a body share the maximum
thrust $T_{max}$ and specific impulse $I_{sp}$. Directions are normalized when
configured; a zero direction leaves the slot unused. With throttle
$u_i = \operatorname{clamp}(u_{cmd,i}, 0, 1)$,

$$
\mathbf{F}_{thr,b} = \sum_i u_i\, T_{max}\, \hat{\mathbf{u}}_i,
\qquad
\boldsymbol{\tau}_{thr} = \sum_i \mathbf{d}_i \times u_i\, T_{max}\, \hat{\mathbf{u}}_i,
\qquad
\dot m = -\frac{T_{max} \sum_i u_i}{I_{sp}\, g_0}
$$

([[16]](../references.md#ref-16)), with $\dot m = 0$ for $I_{sp} = 0$ (ideal
thrusters) and the sum in $\dot m$ over configured slots only. In the body
frame these depend only on the commands and the configuration, so they are
evaluated once per step, from the propellant at the step start, and held for
all stages; the force is rotated into the inertial frame with the attitude of
each stage, $R(q)\,\mathbf{F}_{thr,b}$. With no propellant
($m \le m_{dry}$) all three are zero; on the step where the propellant runs
out, all three are scaled by the burn fraction
([variable mass](dynamics.md#variable-mass)).

The default layout (that of a new simulator) is six
thrusters at $\pm 1$ m on each body axis, each pushing outward along its own
axis (no torque), with $T_{max} = 10$ N and $I_{sp} = 0$.

**Implementation:**
[`actuators/reaction_wheels.py`](../reference/api/cosmic_warp/actuators/reaction_wheels.md)
(`compute_rw_angular_momentum`) ·
[`actuators/thrusters.py`](../reference/api/cosmic_warp/actuators/thrusters.md)
(`thruster_wrench`) · usage in the [actuators guide](../guides/actuators.md)

**References:** [[7]](../references.md#ref-7), [[8]](../references.md#ref-8),
[[16]](../references.md#ref-16) — see the full [bibliography](../references.md).

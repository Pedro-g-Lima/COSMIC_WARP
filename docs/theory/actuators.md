# Actuator models

Commands live in one
[`ActuatorCommand`](../reference/api/cosmic_warp/actuators/base.md) per
entity: eight thruster throttles (`thrust_cmd` for thrusters 0–3,
`thrust_cmd_ext` for 4–7) and four wheel torques (`rw_torque_cmd`, N·m).
They are constant over a step and zeroed, with the wheel momenta, by every
reset. A NaN or Inf command, which the clamps below would turn into a finite
actuation (a NaN wheel torque into the full negative torque), ends the
environment instead ([termination](execution.md#termination)).

## Reaction wheels

Up to four wheels per body, unit spin axes $\hat{\mathbf{a}}_k$ (body frame),
sharing torque limit $\tau_{max}$, capacity $h_{max}$ (0 = unlimited),
wheel inertia $J_w$ and viscous friction $b$
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
vanishing $b$ converges to the undamped wheel. With $h_{max} > 0$ the result
is clamped to $[-h_{max}, h_{max}]$; torquing away from saturation still
works ([[8]](../references.md#ref-8)). Momenta above $h_{max}$, and an
$h_{max}$ lowered below a stored momentum, are rejected.

The integrators evaluate this solution at the times they need (the step end
for Euler; $\Delta t/2$ and $\Delta t$ for RK4; two half steps for the
symplectic scheme) and subtract the change of the wheel momentum vector

$$
\Delta\mathbf{h} = \sum_k \left(h_k(s) - h_k(0)\right) \hat{\mathbf{a}}_k
$$

from $\mathbf{L}$, so $\mathbf{L} + \mathbf{h}$ changes only through external
torques and the gyroscopic term ([dynamics](dynamics.md#rotational-motion)).
Saturation inside a step is not located
([order](integrators.md#events-inside-a-step)). See the
[reaction-wheel example](../examples/07_reaction_wheels.md).

## Thrusters

Up to eight per body, each with a position $\mathbf{d}_i$ from the centre
of mass and a unit force direction $\hat{\mathbf{u}}_i$ (body frame), sharing
$T_{max}$ and $I_{sp}$. With throttle
$u_i = \operatorname{clamp}(u_{cmd,i}, 0, 1)$,

$$
\mathbf{F}_{thr,b} = \sum_i u_i\, T_{max}\, \hat{\mathbf{u}}_i,
\qquad
\boldsymbol{\tau}_{thr} = \sum_i \mathbf{d}_i \times u_i\, T_{max}\, \hat{\mathbf{u}}_i,
\qquad
\dot m = -\frac{T_{max} \sum_i u_i}{I_{sp}\, g_0}
$$

([[16]](../references.md#ref-16)), with $\dot m = 0$ for $I_{sp} = 0$ and the
sum over configured slots only. These depend only on commands and
configuration, so they are evaluated once per step and held for all stages;
the force is rotated with each stage's attitude, $R(q)\,\mathbf{F}_{thr,b}$.
Without propellant all three are zero; on the burnout step they are scaled
by the burn fraction ([variable mass](dynamics.md#variable-mass)). The
default layout and the rest of the usage are in the
[actuators guide](../guides/actuators.md).

**Implementation:**
[`actuators/reaction_wheels.py`](../reference/api/cosmic_warp/actuators/reaction_wheels.md)
(`compute_rw_angular_momentum`) ·
[`actuators/thrusters.py`](../reference/api/cosmic_warp/actuators/thrusters.md)
(`thruster_wrench`) · usage in the [actuators guide](../guides/actuators.md)

**References:** [[7]](../references.md#ref-7), [[8]](../references.md#ref-8),
[[16]](../references.md#ref-16) — see the full [bibliography](../references.md).

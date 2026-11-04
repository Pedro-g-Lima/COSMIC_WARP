# Numerical integration

The state $y = (\mathbf{r}, \mathbf{p}, m, q, \mathbf{L})$ evolves as
$\dot{y} = f(y)$, the right-hand side of [rigid-body dynamics](dynamics.md).
Three fixed-step schemes are available (`integrator="euler" | "rk4" |
"symplectic"`, default `"rk4"`). Each step of each scheme is **one kernel
launch** per entity ([GPU execution](execution.md)); within it, three things
are common to all of them:

- The thruster force, torque and mass flow are evaluated **once per step**,
  from the commands and the propellant at the step start (scaled by the burn
  fraction on the burnout step); only the rotation of the force into the
  inertial frame is re-evaluated with each stage's attitude
  ([actuators](actuators.md#thrusters)).
- The reaction-wheel momenta follow their closed-form solution at the times
  the scheme needs, and the body angular momentum loses exactly what the
  wheels gain ([actuators](actuators.md#reaction-wheels)).
- The step ends at the exact step-end mass $m - \Delta t\,\dot m$, rounded
  once, never below $m_{dry}$ and exactly $m_{dry}$ on the burnout step; the
  quaternion is renormalized after every update.

## Forward Euler

First order ([[14]](../references.md#ref-14)), one evaluation of $f$:

$$
y_{n+1} = y_n + \Delta t\, f(y_n),
$$

with the wheel exchange applied exactly:
$\mathbf{L}_{n+1} = \mathbf{L}_n + \Delta t\,[\boldsymbol{\tau}_{ext} - \boldsymbol{\omega}_n \times (\mathbf{L}_n + \mathbf{h}_n)] - \Delta\mathbf{h}$.
The local error is $O(\Delta t^2)$ and the orbital energy drifts secularly;
use it for debugging or with very small steps.

## Runge–Kutta 4

Fourth order, four evaluations of $f$ ([[14]](../references.md#ref-14)):

$$
\begin{aligned}
k_1 &= f(y_n) \\
k_2 &= f\!\left(y_n + \tfrac{\Delta t}{2} k_1\right) \\
k_3 &= f\!\left(y_n + \tfrac{\Delta t}{2} k_2\right) \\
k_4 &= f\!\left(y_n + \Delta t\, k_3\right) \\
y_{n+1} &= y_n + \tfrac{\Delta t}{6}\left(k_1 + 2k_2 + 2k_3 + k_4\right)
\end{aligned}
$$

The default, and the most accurate per step; it does not preserve energy
over long horizons.

The wheel momenta $\mathbf{h}(t)$ are known in closed form within the step,
so each stage uses them at its own time ($0$, $\Delta t/2$, $\Delta t$). The
stages advance the body-frame total angular momentum
$\mathbf{H} = \mathbf{L} + \mathbf{h}$, whose rate
$\boldsymbol{\tau}_{ext} - \boldsymbol{\omega} \times \mathbf{H}$ contains no
wheel torque, and each stage state recovers $\mathbf{L}$ by subtracting
$\mathbf{h}$ at the stage time. The wheel/body coupling is therefore
fourth-order too (until a wheel saturates, below).

## Symplectic (kick–drift–kick)

A second-order splitting into momentum updates (*kicks*) and a free flight
(*drift*) ([[12]](../references.md#ref-12), [[13]](../references.md#ref-13),
[[15]](../references.md#ref-15)). For a point mass in gravity and J2 at
constant mass it is the Störmer–Verlet method:

$$
\begin{aligned}
\mathbf{p}_{n+1/2} &= \mathbf{p}_n + \tfrac{\Delta t}{2}\,
\mathbf{F}(\mathbf{r}_n) && \text{(half kick)} \\
\mathbf{r}_{n+1} &= \mathbf{r}_n + \Delta t\,
\frac{\mathbf{p}_{n+1/2}}{m} && \text{(drift)} \\
\mathbf{p}_{n+1} &= \mathbf{p}_{n+1/2} + \tfrac{\Delta t}{2}\,
\mathbf{F}(\mathbf{r}_{n+1}) && \text{(half kick)}
\end{aligned}
$$

Each sub-step is the exact flow of part of the Hamiltonian, so the
composition is symplectic and the energy error stays **bounded**
(oscillatory) instead of growing secularly
([[12]](../references.md#ref-12)) — hence its use for long orbit
propagation, shown in the
[energy-conservation example](../examples/05_energy_conservation.md). The
kernel extends this to mass flow, non-conservative forces, rotation and
wheels as follows.

**Kicks with mass flow.** Each half kick holds the forces fixed and lets the
mass fall linearly, $m_b = \max(m_a - \tfrac{\Delta t}{2}|\dot m|, m_{dry})$
(the second kick ends at the step-end mass common to all schemes, so the mass
is rounded once per step), and solves $m\,\dot{\mathbf{v}} = \mathbf{F}$ exactly. Gravity is
proportional to $m$, a constant acceleration; drag and thrust
$\mathbf{F}_o$ are not, and act through $\int dt/m(t)$:

$$
\mathbf{v}_b = \mathbf{v}_a + \tfrac{\Delta t}{2}\,\mathbf{a}_g
+ \frac{\Delta t}{2\,m_a}\,\phi(x)\,\mathbf{F}_o,
\qquad \phi(x) = -\frac{\ln(1-x)}{x},\quad x = \frac{m_a - m_b}{m_a},
\qquad \mathbf{p}_b = m_b \mathbf{v}_b,
$$

with $\phi \to 1$ (a series) for small $x$. The same half kick adds
$\tfrac{\Delta t}{2}\boldsymbol{\tau}_{ext}$ to $\mathbf{L}$, advances the
wheels by $\Delta t/2$ and subtracts their momentum change. Burns stay
second-order, and on the burnout step the
[burn fraction](dynamics.md#variable-mass) keeps the flow on for the whole
step, so delta-v matches the rocket equation to rounding.

**Drift.** Position moves with the half-kicked velocity,
$\mathbf{r}_{n+1} = \mathbf{r}_n + \Delta t\,\mathbf{p}_{n+1/2}/m_{n+1/2}$.
The body rotates freely with the wheel momenta $\mathbf{h}$ held, so the
total angular momentum $\mathbf{H} = \mathbf{L} + \mathbf{h}$ is constant in
the inertial frame. The drift is the explicit midpoint rule on the rotation
group, with $\exp(\boldsymbol{\omega} s)$ the rotation quaternion by angle
$\lVert\boldsymbol{\omega}\rVert s$ about $\boldsymbol{\omega}$:

$$
\begin{aligned}
\delta q_{1/2} &= \exp\!\left(\mathbf{I}^{-1}\mathbf{L}\ \tfrac{\Delta t}{2}\right),
\qquad
\boldsymbol{\omega}_{1/2} = \mathbf{I}^{-1}\!\left(R(\delta q_{1/2})^{\top}\mathbf{H} - \mathbf{h}\right), \\
\delta q &= \exp\!\left(\boldsymbol{\omega}_{1/2}\,\Delta t\right),
\qquad
q_{n+1} = q_n \otimes \delta q,
\qquad
\mathbf{L} \leftarrow R(\delta q)^{\top}\mathbf{H} - \mathbf{h}.
\end{aligned}
$$

The body-frame $\mathbf{H}$ is rotated back by exactly the attitude
increment, so the inertial angular momentum is conserved to rounding; the
attitude is second-order.

**Second kick.** Forces and torques are evaluated at
$(\mathbf{r}_{n+1}, \mathbf{p}_{n+1/2}, m_{n+1/2}, q_{n+1})$ and applied as in
the first kick, with the wheels advanced from their half-step values.

!!! note "Accuracy limits of the implementation"
    - The scheme is symplectic only for the translational motion at
      **constant mass** under gravity and J2. While thrusters fire,
      propellant consumption makes the system non-Hamiltonian and energy is
      (correctly) not conserved.
    - The rotational drift is not the exact free rigid-body flow: the
      inertial angular momentum is conserved to rounding, the rotational
      kinetic energy only to the scheme's second order (relative drift
      about $10^{-9}$ over a 5 s tumble at $\Delta t = 10^{-3}$ s in the
      test suite).
    - Velocity-dependent forces (drag) are evaluated at the start of each
      kick, the second one with the half-kicked momentum; their (usually
      tiny) contribution is first-order.

## Events inside a step

The integrators do not locate events. When a reaction wheel reaches its
momentum limit inside a step, the clamp is a kink in the right-hand side and
every integrator is limited to second order around it (measured: RK4 drops
from 4 to 2). Propellant burnout is handled by the
[burn fraction](dynamics.md#variable-mass), with an $O(\Delta t^2)$ position
error once per burnout. Choose `dt` with these events in mind when RK4
accuracy matters.

## Order verification

The test suite checks the observed order of convergence of each scheme
against its nominal order: for Kepler orbits against the analytic solution,
and by self-convergence for a torque-free tumble, reaction-wheel momentum
exchange with a transverse body rate, a finite burn in orbit, and wheel
saturation inside a step (where the expected order is capped at 2). The
converged answers of the three integrators are also checked against each
other, since self-convergence alone would not catch a model error shared by
every step size. See [Validation](../validation/index.md); the cost per step
of each scheme is on the [Benchmarks](../benchmarks.md) page.

**Implementation:**
[`euler.py`](../reference/api/cosmic_warp/core/integrators/euler.md) ·
[`rk4.py`](../reference/api/cosmic_warp/core/integrators/rk4.md) ·
[`symplectic.py`](../reference/api/cosmic_warp/core/integrators/symplectic.md)
(`kick_momentum`, `quat_exp`) ·
practical guidance in the [integrators guide](../guides/integrators.md)

**References:** [[12]](../references.md#ref-12), [[13]](../references.md#ref-13),
[[14]](../references.md#ref-14), [[15]](../references.md#ref-15) — see the full
[bibliography](../references.md).

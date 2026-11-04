# Numerical integration

The state $y = (\mathbf{r}, \mathbf{p}, m, q, \mathbf{L})$ evolves as
$\dot{y} = f(y)$, the right-hand side of [rigid-body dynamics](dynamics.md).
Three fixed-step schemes (`integrator="euler" | "rk4" | "symplectic"`,
default `"rk4"`), each one kernel launch per step
([GPU execution](execution.md)). Common to all:

- thrust, its torque and mass flow are evaluated **once per step**; only the
  force's rotation follows each stage's attitude ([actuators](actuators.md#thrusters));
- the wheel momenta follow their closed form at the times the scheme needs,
  and the body loses exactly what the wheels gain
  ([actuators](actuators.md#reaction-wheels));
- the step ends at the step-end mass, rounded once
  ([burnout](dynamics.md#variable-mass)); the quaternion is renormalized
  after every update.

## Forward Euler

First order ([[14]](../references.md#ref-14)), one evaluation of $f$:

$$
y_{n+1} = y_n + \Delta t\, f(y_n),
$$

with the wheel exchange applied exactly:
$\mathbf{L}_{n+1} = \mathbf{L}_n + \Delta t\,[\boldsymbol{\tau}_{ext} - \boldsymbol{\omega}_n \times (\mathbf{L}_n + \mathbf{h}_n)] - \Delta\mathbf{h}$.
The orbital energy drifts secularly.

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

The stages advance the total angular momentum
$\mathbf{H} = \mathbf{L} + \mathbf{h}$, whose rate
$\boldsymbol{\tau}_{ext} - \boldsymbol{\omega} \times \mathbf{H}$ has no
wheel torque, and recover $\mathbf{L}$ by subtracting the closed-form
$\mathbf{h}$ at each stage time ($0$, $\Delta t/2$, $\Delta t$), so the
wheel coupling is fourth-order too until a wheel saturates.

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

Each sub-step is the exact flow of part of the Hamiltonian, so the energy
error stays **bounded** instead of growing secularly
([[12]](../references.md#ref-12);
[example](../examples/05_energy_conservation.md)). The kernel extends this
to mass flow, non-conservative forces, rotation and wheels:

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

with $\phi \to 1$ (a series) for small $x$. The half kick also adds
$\tfrac{\Delta t}{2}\boldsymbol{\tau}_{ext}$ to $\mathbf{L}$ and advances the
wheels by $\Delta t/2$. Burns stay second-order and delta-v matches the
rocket equation to rounding.

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

$\mathbf{H}$ is rotated back by exactly the attitude increment, so the
inertial angular momentum is conserved to rounding.

**Second kick.** Forces and torques are evaluated at
$(\mathbf{r}_{n+1}, \mathbf{p}_{n+1/2}, m_{n+1/2}, q_{n+1})$ and applied as in
the first kick, with the wheels advanced from their half-step values.

!!! note "Accuracy limits"
    - Symplectic only for the translation at **constant mass** under gravity
      and J2: while thrusters fire, energy is (correctly) not conserved.
    - The rotational drift is not the exact free rigid-body flow: rotational
      kinetic energy is kept only to second order
      ([measured](../validation/numerics.md#conserved-quantities)).
    - Drag is evaluated at the start of each kick (the second with the
      half-kicked momentum), so its contribution is first-order.

## Events inside a step

Events are not located. A wheel reaching its momentum limit inside a step
is a kink in the right-hand side, which limits every integrator to second
order around it ([measured](../validation/numerics.md#order-of-convergence));
burnout is handled by the [burn fraction](dynamics.md#variable-mass).

## Order verification

Each scheme's observed order, and the agreement of their converged
answers, are tested: [numerics validation](../validation/numerics.md).
Cost per step: [Benchmarks](../benchmarks.md).

**Implementation:**
[`euler.py`](../reference/api/cosmic_warp/core/integrators/euler.md) ·
[`rk4.py`](../reference/api/cosmic_warp/core/integrators/rk4.md) ·
[`symplectic.py`](../reference/api/cosmic_warp/core/integrators/symplectic.md)
(`kick_momentum`, `quat_exp`) ·
practical guidance in the [integrators guide](../guides/integrators.md)

**References:** [[12]](../references.md#ref-12), [[13]](../references.md#ref-13),
[[14]](../references.md#ref-14), [[15]](../references.md#ref-15) — see the full
[bibliography](../references.md).

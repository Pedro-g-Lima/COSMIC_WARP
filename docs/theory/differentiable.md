# Differentiable dynamics

A simulator built with `differentiable=True` computes reverse-mode gradients
of a scalar loss through the integrator — every force model, actuator and
stage — with respect to the initial state, the per-step commands and the
vehicle and actuator parameters. The gradients are those of the discrete
trajectory the simulator computes, exact to rounding away from the
non-smooth points listed below.

## The discrete adjoint

A rollout is the composition of the one-step map $\Phi$ of the chosen
[integrator](integrators.md):

$$
\mathbf{x}_{t+1} = \Phi(\mathbf{x}_t, \mathbf{u}_t, \boldsymbol{\theta}),
\qquad t = 0, \dots, N-1,
$$

with $\mathbf{x}$ the state and wheel momenta, $\mathbf{u}_t$ the actuator
commands of step $t$ and $\boldsymbol{\theta}$ the constant parameters. For
a loss $L(\mathbf{x}_N)$, reverse-mode differentiation propagates the
adjoint backwards ([[17]](../references.md#ref-17)):

$$
\boldsymbol{\lambda}_N = \frac{\partial L}{\partial \mathbf{x}_N},
\qquad
\boldsymbol{\lambda}_t =
\left(\frac{\partial \Phi}{\partial \mathbf{x}_t}\right)^{\!\top}
\boldsymbol{\lambda}_{t+1},
\qquad
\frac{\partial L}{\partial \mathbf{u}_t} =
\left(\frac{\partial \Phi}{\partial \mathbf{u}_t}\right)^{\!\top}
\boldsymbol{\lambda}_{t+1},
\qquad
\frac{\partial L}{\partial \boldsymbol{\theta}} = \sum_{t}
\left(\frac{\partial \Phi}{\partial \boldsymbol{\theta}}\right)^{\!\top}
\boldsymbol{\lambda}_{t+1}.
$$

One backward sweep gives the gradient with respect to every input, at a cost
proportional to the forward rollout and independent of the number of inputs.
The parameter gradient sums the contributions of every step — the total
sensitivity used for system identification (drag coefficient and area,
inertia, magnetic dipole, thruster thrust, $I_{sp}$ and geometry, wheel
parameters). This is the discrete (differentiate-the-scheme) adjoint, not a
discretization of the continuous adjoint equation: it is the exact
derivative of what the integrator computes, including its truncation error.

## Implementation

Warp generates the adjoint of each kernel and replays the launches recorded
on a `wp.Tape` in reverse ([[18]](../references.md#ref-18)).
[`DifferentiableRollout`](../reference/api/cosmic_warp/core/autodiff.md)
(`sim.differentiable_rollout(num_steps, commands)`) builds on it:

- **Per-step buffers.** Step $t$ reads buffer $t$ and writes buffer $t+1$
  instead of updating the state in place, so the backward pass sees every
  intermediate state. The rollout keeps `num_steps + 1` copies of the state
  and wheel momenta plus their gradients, about
  `2 x (num_steps + 1) x num_entities x 144` bytes.
- **Parameter shadows.** The rollout copies the vehicle, thruster and wheel
  configurations into arrays with gradients, read at every step. The kernels
  read both `inertia` and its inverse `inertia_inv`, which is computed when
  the inertia is set, outside the tape; `rollout.inertia_grad()` combines the
  two paths through the adjoint of the matrix inverse,
  $\bar{\mathbf{I}} \mathrel{-}= \mathbf{I}^{-\top}\,\overline{\mathbf{I}^{-1}}\,\mathbf{I}^{-\top}$.
- **Adjoint-safe kernels.** The integrator kernels are written in
  single-assignment form with per-field state stores, the patterns Warp's
  code generation differentiates correctly.
- **Opt-in compilation.** Adjoint code generation is off by default and is
  enabled for the chosen integrator's module by `differentiable=True`, so
  other simulators pay no extra compile time. Forward stepping runs the same
  kernels either way.
- **No graph capture.** A taped rollout launches its kernels step by step,
  eagerly.

The test suite checks the gradients of every integrator against central
finite differences: initial state, thruster and wheel commands (including
across a burnout step), and drag, thruster, inertia, magnetic-dipole and
wheel parameters. [`DiffSimStepper`](../reference/api/cosmic_warp/rl/diff_bridge.md)
wraps one step as a `torch.autograd.Function` on the same kernels, for
policy gradients through the dynamics
([differentiable RL](../guides/differentiable-rl.md)).

## Smoothness caveats

Reverse-mode gradients are exact for the branch the forward pass took. The
dynamics contain isolated non-smooth points where the derivative is
one-sided or zero:

- **Propellant cutoff** at $m = m_{dry}$: once dry, the thrusters produce
  nothing and the gradient through them is zero. $m_{dry}$ enters only
  through this cutoff, the mass floors and the
  [burn fraction](dynamics.md#variable-mass) of the burnout step, so its
  gradient is zero on every other step.
- **Clamps** on the wheel torque command and momentum, and on the thruster
  throttles $u_i \in [0, 1]$: while a clamp is active, the gradient with
  respect to the command is zero and the one with respect to the limit is
  not; while inactive, the reverse.
- **Atmosphere band edges** and the sea-level clamp of the
  [density](environment.md#atmospheric-drag): below altitude 0 the density
  is constant, so drag has no gradient with respect to position there (it
  still depends on velocity).
- **Termination**: a terminated entity is frozen and has no gradient path,
  so differentiable rollouts require `check_bounds=False` and
  `check_planet_impact=False` (they raise otherwise). The
  [non-finite check](execution.md#termination) cannot be turned off: a NaN
  or Inf in the state or in a command still terminates the entity inside
  the rollout, which then carries its last state over unchanged. The rollout keeps its own
  termination status: the simulator's is not touched.

Away from these points the dynamics are smooth and the gradients are exact.

**Implementation:**
[`core/autodiff.py`](../reference/api/cosmic_warp/core/autodiff.md) ·
[`rl/diff_bridge.py`](../reference/api/cosmic_warp/rl/diff_bridge.md) ·
usage in the [differentiable rollouts guide](../guides/differentiable.md) ·
demos in [example 11](../examples/11_gradient_optimization.md) and
[example 12](../examples/12_system_identification.md)

**References:** [[17]](../references.md#ref-17),
[[18]](../references.md#ref-18) — see the full
[bibliography](../references.md).

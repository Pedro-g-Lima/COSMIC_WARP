# Differentiable dynamics

With `differentiable=True` the simulator computes reverse-mode gradients of
a scalar loss through every force model, actuator and stage, with respect to
the initial state, the per-step commands and the parameters: the gradients
of the discrete trajectory, exact to rounding away from the non-smooth
points below. Usage: [differentiable rollouts](../guides/differentiable.md).

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

One backward sweep gives every input's gradient, at a cost proportional to
the forward rollout. The parameter gradient sums every step's contribution,
the total sensitivity system identification needs. This is the discrete
(differentiate-the-scheme) adjoint: the exact derivative of what the
integrator computes, truncation error included.

## Implementation

Warp generates the adjoint of each kernel and replays the launches recorded
on a `wp.Tape` in reverse ([[18]](../references.md#ref-18)).
[`DifferentiableRollout`](../reference/api/cosmic_warp/core/autodiff.md)
(`sim.differentiable_rollout(num_steps, commands=...)`) builds on it:

- **Per-step buffers.** Step $t$ reads buffer $t$ and writes $t+1$, so the
  backward pass sees every intermediate state
  ([memory](../guides/differentiable.md#requirements-and-limits)).
- **Parameter shadows.** The configurations are copied into arrays with
  gradients. The kernels read both `inertia` and `inertia_inv` (computed
  outside the tape); `rollout.inertia_grad()` combines the two paths through
  the adjoint of the inverse,
  $\bar{\mathbf{I}} \mathrel{-}= \mathbf{I}^{-\top}\,\overline{\mathbf{I}^{-1}}\,\mathbf{I}^{-\top}$.
- **Adjoint-safe kernels**: single-assignment form with per-field state
  stores ([kernel conventions](../guides/extending.md#kernel-conventions)).
- **Opt-in compilation**: adjoint code is generated only for the chosen
  integrator's module, only with `differentiable=True`.

Every integrator's gradients are checked against central finite differences
([validation](../validation/learning.md#differentiable-rollouts)).
[`DiffSimStepper`](../reference/api/cosmic_warp/rl/diff_bridge.md) wraps one
step as a `torch.autograd.Function` on the same kernels
([differentiable RL](../guides/differentiable-rl.md)).

## Smoothness caveats

Gradients are exact for the branch the forward pass took. At these
non-smooth points the derivative is one-sided or zero:

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
- **Termination**: a terminated entity is frozen, with no gradient path, so
  rollouts require `check_bounds=False` and `check_planet_impact=False`. The
  [non-finite check](execution.md#termination) stays on: a NaN or Inf in the
  state or a command freezes the entity at its last state, on the rollout's
  own termination status, not the simulator's.

**Implementation:**
[`core/autodiff.py`](../reference/api/cosmic_warp/core/autodiff.md) ·
[`rl/diff_bridge.py`](../reference/api/cosmic_warp/rl/diff_bridge.md) ·
demos in [example 11](../examples/11_gradient_optimization.md) and
[example 12](../examples/12_system_identification.md)

**References:** [[17]](../references.md#ref-17),
[[18]](../references.md#ref-18) — see the full
[bibliography](../references.md).

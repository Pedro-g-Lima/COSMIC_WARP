# Differentiable rollouts

A differentiable rollout records a multi-step simulation on a Warp tape and
backpropagates a scalar loss to the initial state, the vehicle and actuator
parameters, and an optional per-step command sequence: for trajectory
optimization, system identification and gradient-based control. The
gradients are those of the discrete integrator, exact to rounding except at
the [non-smooth points](../theory/differentiable.md#smoothness-caveats).

## Enable

```python
import numpy as np
import warp as wp

from cosmic_warp.core.pipeline import SpacecraftSimulator

sim = SpacecraftSimulator(
    num_envs=4,
    integrator="symplectic",   # every integrator is differentiable
    dt=1.0,
    differentiable=True,       # compile the integrator's adjoint kernels
    check_bounds=False,        # termination checks must be off
)
sim.reset()                    # or reset_to_conditions(...)
```

`differentiable=True` turns on adjoint code generation for the chosen
integrator's kernel module, which lengthens its one-off compile; stepping
runs the same forward kernels. The setting is per module and lasts for the
process.

## Tape a rollout

`sim.differentiable_rollout(num_steps)` copies the simulator's current state
and parameters. Write the loss as Warp kernels over the rollout's states,
launched inside the same tape:

```python
from cosmic_warp.core.state import State


@wp.kernel
def miss_distance(states: wp.array(dtype=State), target: wp.vec3d,
                  loss: wp.array(dtype=wp.float64)):
    tid = wp.tid()
    d = states[tid].pos - target
    wp.atomic_add(loss, 0, wp.dot(d, d))


num_steps = 50
target = wp.vec3d(6.75e6, 4.0e5, 0.0)    # [m], ECI

rollout = sim.differentiable_rollout(num_steps)
loss = wp.zeros(1, dtype=wp.float64, device=sim.device, requires_grad=True)

with rollout.tape:                        # record the forward pass and the loss
    rollout.forward()
    wp.launch(miss_distance, dim=sim.num_entities,
              inputs=[rollout.final_states, target, loss], device=sim.device)
rollout.tape.backward(loss)               # adjoint sweep

dl_dx0 = rollout.initial_states.grad.numpy()   # structured: "pos", "mom", "quat", ...
```

The state holds momentum; at fixed mass ∂L/∂v₀ = m ∂L/∂p₀:

```python
dl_dv0 = sim.get_masses().numpy()[:, None] * dl_dx0["mom"]   # [num_entities, 3]
```

`rollout.states[t]` is the state after `t` steps (`initial_states` is
`states[0]`, `final_states` is `states[num_steps]`), so a loss can read any
point of the history ([fields](../theory/conventions.md#state)).

## Command gradients

Pass a per-step command sequence to differentiate with respect to the
controls:

```python
from cosmic_warp.core.autodiff import command_sequence

cmds = command_sequence(sim, num_steps)     # one ActuatorCommand array per step
u = cmds[0].numpy()
u["rw_torque_cmd"][:, 0] = 1.0e-3            # wheel 0 at 1 mN*m on step 0
cmds[0].assign(u)

rollout = sim.differentiable_rollout(num_steps, commands=cmds)
loss = wp.zeros(1, dtype=wp.float64, device=sim.device, requires_grad=True)
with rollout.tape:
    rollout.forward()
    wp.launch(miss_distance, dim=sim.num_entities,
              inputs=[rollout.final_states, target, loss], device=sim.device)
rollout.tape.backward(loss)

dl_du0 = cmds[0].grad.numpy()   # fields "thrust_cmd", "thrust_cmd_ext", "rw_torque_cmd"
# (zero here: a wheel torque turns the body but cannot move it)
```

Without `commands`, the current commands are held for every step and not
differentiated. Thruster commands have gradients only while the body has
propellant.

## Parameter gradients

The rollout differentiates copies of the per-entity parameters:
`rollout.vehicle_configs` (`mass_dry`, `inertia`, `inertia_inv`,
`area_drag`, `cd`, `magnetic_dipole`), `rollout.thruster_configs`
(`positions`, `directions`, `max_thrust`, `isp`) and `rollout.rw_configs`
(`max_torque`, `max_momentum`, `wheel_inertia`, `viscous_damping`, spin
axes). After `backward()` each `.grad` holds the derivative summed over the
horizon:

```python
dl_dcd = rollout.vehicle_configs.grad.numpy()["cd"]        # per entity (zero: drag is off)
dl_dinertia = rollout.inertia_grad()                       # [num_entities, 3, 3]
```

`inertia_grad()` combines the paths through `inertia` and its precomputed
inverse into the total derivative. [Example 12](../examples/12_system_identification.md)
recovers an unknown drag coefficient this way.

## Optimization loops

Allocate the rollout once and call `rollout.reset()` each iteration: it
clears the tape, zeroes the gradients and copies the simulator's state and
parameters again. Change them on the simulator (`sim.states.assign(...)`,
`sim.vehicle.set_config(...)`), not on the rollout:

```python
x0 = sim.states.numpy()
lr = 0.25 / (num_steps * sim.dt) ** 2

for _ in range(5):
    sim.states.assign(x0)                 # candidate initial state
    rollout.reset()
    loss = wp.zeros(1, dtype=wp.float64, device=sim.device, requires_grad=True)
    with rollout.tape:
        rollout.forward()
        wp.launch(miss_distance, dim=sim.num_entities,
                  inputs=[rollout.final_states, target, loss], device=sim.device)
    rollout.tape.backward(loss)
    x0["mom"] -= lr * rollout.initial_states.grad.numpy()["mom"]   # gradient step
```

`forward()` raises if called twice without `reset()`.
[Example 11](../examples/11_gradient_optimization.md) retargets a trajectory
this way, from a 50 km miss to below a metre in 60 iterations.

## Requirements and limits

- **Termination checks off**: `check_bounds=False` and
  `check_planet_impact=False`, or `differentiable_rollout()` raises; a
  terminated entity has no gradient path. The non-finite check stays on: a
  NaN or Inf freezes the entity on the rollout's own termination status,
  not the simulator's ([smoothness caveats](../theory/differentiable.md#smoothness-caveats)).
- **The rollout and the simulator are independent.** The rollout never
  changes the simulator, and its backward reads only its own copies, so
  resetting or reconfiguring the simulator between `forward()` and
  `backward()` does not change the gradients.
- **Memory grows with the horizon**: `num_steps + 1` copies of the state and
  wheel momenta plus gradients, about
  `2 × (num_steps + 1) × num_entities × 144` bytes. For long horizons, split
  the trajectory and chain the gradients yourself.
- **No CUDA graph**: the taped forward pass launches step by step, with
  Python launch overhead each step.

API: [`autodiff`](../reference/api/cosmic_warp/core/autodiff.md)

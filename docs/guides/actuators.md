# Actuators

Each spacecraft carries up to 8 thrusters and up to 4 reaction wheels.
Configure them once, then command them between steps. The models are
derived in [Actuator models](../theory/actuators.md).

Like `vehicle.set_config`, each `set_config` call writes only the fields you
pass; the others keep their current value. All of them take `env_ids`,
`body_ids` and `entity_ids`.

## Thrusters

```python
import numpy as np
import warp as wp
from cosmic_warp.core.pipeline import SpacecraftSimulator

sim = SpacecraftSimulator(num_envs=4, dt=0.1, gravity=False)
sim.vehicle.set_config(mass_dry=10.0, inertia=np.diag([1.5, 1.5, 1.5]))
sim.reset_manager.set_bounds(mass_bounds=(12.0, 12.0))  # 2 kg of propellant

arm = 0.5
sim.thruster_manager.set_config(
    max_thrust=1.0,              # per thruster [N]
    isp=220.0,                   # [s]; 0 = ideal, no propellant use
    positions=[(arm, 0.0, 0.0), (-arm, 0.0, 0.0),        # body frame, from the CoM [m]
               (0.0, arm, 0.0), (0.0, -arm, 0.0)],
    directions=[(0.0, 1.0, 0.0), (0.0, -1.0, 0.0),       # direction of the force
                (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
)
sim.reset()
sim.set_thrust_commands([1.0, 1.0, 0.0, 0.0])  # thrusters 0 and 1: a pure +Z torque
sim.simulate(num_steps=50)
```

- **Layout.** `positions` and `directions` have one entry per thruster, at
  most 8, and the same length; otherwise `set_config` raises. Directions are
  normalized, so the magnitude comes from `max_thrust` alone. A direction of
  `(0, 0, 0)` leaves the slot unused: no force, torque or propellant use.
  Give both lists or neither; neither keeps the current layout. A new
  simulator has six thrusters on the faces of a 2 m cube, each pushing along
  its outward normal, with `max_thrust` 10 N and `isp` 0.
- **One `max_thrust` and `isp` per body**, shared by its thrusters.
- **Force and torque.** Thruster *i* produces `cmd_i · max_thrust` along its
  direction, at its position, so a torque `position × force` about the
  centre of mass. The direction is that of the force on the spacecraft, not
  of the exhaust.
- **Propellant.** With `isp > 0` the mass drops at
  `Σ thrust / (isp · g0)`, `g0 = 9.80665 m/s²`. No thruster produces force
  while the mass is at or below `mass_dry` — whatever the `isp`, so the
  spacecraft needs propellant (a reset mass above `mass_dry`). On the step
  where the propellant runs out, force, torque and mass flow are scaled by
  the fraction of the step the propellant lasts, so the burn's delta-v is
  exact ([variable mass](../theory/dynamics.md#variable-mass)).

### Thrust commands

A command is a throttle per thruster, clamped to `[0, 1]`: thrusters only
push. `set_thrust_commands` broadcasts the same commands to every entity:
up to 8 values, one per thruster slot; omitted trailing slots are zero:

```python
sim.set_thrust_commands([0.5, 0.5])   # thrusters 0 and 1 at half; 2-7 off
```

For a different command per entity, pass a `float64` Warp array of shape
`(num_envs * num_bodies, 8)` on the simulator's device, one column per slot
(the columns of unused slots have no effect):

```python
thrust_cmds = wp.zeros((sim.num_entities, 8), dtype=wp.float64, device=sim.device)
sim.set_thrust_commands_per_entity(thrust_cmds)
```

The commands are copied when you call; changing the array later does not
change them. Commands hold until you set new ones or the entity is reset:
every reset zeroes them. A NaN or Inf in any command of an entity, thrust or
wheel torque (a diverged policy, say), ends its environment before the step
with reason 6, its state unchanged, instead of being clamped into a finite
command ([termination](../theory/execution.md#termination)).

## Reaction wheels

```python
sim = SpacecraftSimulator(num_envs=4, dt=0.01, gravity=False)
sim.rw_manager.set_config(
    max_torque=0.01,           # per wheel [N·m]
    max_momentum=0.05,         # capacity per wheel [N·m·s]; 0 = unlimited
    wheel_inertia=1.0e-3,      # [kg·m²]
    viscous_damping=1.0e-6,    # [N·m·s/rad]
    spin_axes=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
)
sim.reset()
sim.set_rw_torque_commands([0.0, 0.0, 0.005])  # [N·m], one per wheel
results = sim.simulate(num_steps=1000, save_rw_states=True)
wheel_speeds = results["rw_angular_velocities"].numpy()  # (steps, entities, wheels) [rad/s]
```

- **Axes.** Up to 4 spin axes in the body frame, normalized; an axis of
  `(0, 0, 0)` leaves the wheel unused. A new simulator has three wheels,
  along X, Y and Z, with `max_torque` 0.1 N·m, `wheel_inertia` 0.01 kg·m²,
  no momentum limit and no friction.
- **Commands** are motor torques on the wheels, clamped to `±max_torque`.
  The body receives the opposite torque, so body and wheels exchange angular
  momentum without creating it.
- **Friction.** Each wheel follows `J ω̇ = τ − b ω`, integrated exactly over
  the step.
- **Saturation.** With `max_momentum > 0`, a wheel's momentum is clamped to
  `±max_momentum`: a saturated wheel absorbs no more momentum in that
  direction, and torquing it back still works. `set_config` rejects a
  `max_momentum` below a wheel's stored momentum, which the next step would
  dump into the body at once.

### Wheel commands and state

`set_rw_torque_commands` broadcasts up to 4 torques, zero-padded;
`set_rw_torque_commands_per_entity` takes a `float64` Warp array of shape
`(num_envs * num_bodies, 4)`. Commands hold until set again or reset.

Every reset zeroes the wheel momenta. To start with spinning wheels, set
their momentum after the reset; each value must be finite and within
`max_momentum`:

```python
sim.set_rw_angular_momentum_state([0.0, 0.0, 0.02])  # [N·m·s], broadcast
```

`set_rw_angular_momentum_state_per_entity` takes a `(num_envs * num_bodies, 4)`
array; `sim.get_rw_angular_momenta()` reads them back, like the
[state getters](running.md#read-the-current-state). With `save_rw_states=True`, `simulate()` records each wheel's speed
(momentum over `wheel_inertia`) after every step, shape
`(num_steps, num_envs * num_bodies, 4)`.

In YAML, actuators are configured per body under
[`body.actuators`](configuration.md#actuatorsthrusters).

API: [`ThrusterManager`](../reference/api/cosmic_warp/actuators/thrusters.md) ·
[`ReactionWheelManager`](../reference/api/cosmic_warp/actuators/reaction_wheels.md) ·
[`ActuatorManager`](../reference/api/cosmic_warp/actuators/base.md)

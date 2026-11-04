# Actuators

Each spacecraft carries up to 8 thrusters and 4 reaction wheels. Configure
them once, then command them between steps. Each `set_config` writes only
the fields you pass and takes `env_ids`, `body_ids` and `entity_ids`. The
models are derived in [Actuator models](../theory/actuators.md).

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

- **Layout.** `positions` and `directions`: one entry each per thruster, at
  most 8, same length; give both or neither (neither keeps the current
  layout). Directions are normalized; `(0, 0, 0)` leaves the slot unused.
  The direction is that of the force on the spacecraft, not of the exhaust.
- **Default.** Six thrusters on the faces of a 2 m cube, each pushing along
  its outward normal; `max_thrust` 10 N, `isp` 0.
- **One `max_thrust` and `isp` per body.** Thruster *i* produces
  `cmd_i · max_thrust` along its direction, and a torque
  `position × force`.
- **Propellant.** With `isp > 0` the mass falls at `Σ thrust / (isp · g0)`.
  No thruster fires while the mass is at or below `mass_dry`, whatever the
  `isp`. On the step the propellant runs out, force, torque and mass flow
  are scaled by the fraction of the step it lasts, so the burn's delta-v is
  exact ([variable mass](../theory/dynamics.md#variable-mass)).

### Thrust commands

A command is a throttle per thruster, clamped to `[0, 1]`.
`set_thrust_commands` takes one command of up to 8 values, one per slot
(trailing slots zero), and broadcasts it to every entity:

```python
sim.set_thrust_commands([0.5, 0.5])   # thrusters 0 and 1 at half; 2-7 off
```

`set_thrust_commands_per_entity` takes a `float64` Warp array of shape
`(num_envs * num_bodies, 8)` on the simulator's device:

```python
thrust_cmds = wp.zeros((sim.num_entities, 8), dtype=wp.float64, device=sim.device)
sim.set_thrust_commands_per_entity(thrust_cmds)
```

Commands are copied on the call and hold until set again; every reset
zeroes them. A NaN or Inf in any thrust or wheel command of an entity ends
its environment before the step with reason 6, state unchanged, rather than
being clamped into a finite command ([termination](../theory/execution.md#termination)).

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

- **Axes.** Up to 4, body frame, normalized; `(0, 0, 0)` leaves a wheel
  unused. Default: wheels along X, Y and Z, `max_torque` 0.1 N·m,
  `wheel_inertia` 0.01 kg·m², no momentum limit, no friction.
- **Commands** are motor torques, clamped to `±max_torque`; the body gets
  the opposite torque, so momentum is exchanged, not created.
- **Friction.** `J ω̇ = τ − b ω`, integrated exactly over the step.
- **Saturation.** With `max_momentum > 0`, a wheel's momentum is clamped to
  `±max_momentum`; torquing it back still works. `set_config` rejects a
  `max_momentum` below a stored wheel momentum.

### Wheel commands and state

`set_rw_torque_commands` broadcasts up to 4 torques (zero-padded);
`set_rw_torque_commands_per_entity` takes a `float64` array of shape
`(num_envs * num_bodies, 4)`. Commands hold until set again or reset.

Every reset zeroes the wheel momenta. To start with spinning wheels, set
them after the reset, finite and within `max_momentum`:

```python
sim.set_rw_angular_momentum_state([0.0, 0.0, 0.02])  # [N·m·s], broadcast
```

`set_rw_angular_momentum_state_per_entity` takes a
`(num_envs * num_bodies, 4)` array; `sim.get_rw_angular_momenta()` reads
them back. `save_rw_states=True` records each wheel's speed (momentum over
`wheel_inertia`) after every step, shape
`(num_steps, num_envs * num_bodies, 4)`.

YAML: [`body.actuators`](configuration.md#actuatorsthrusters).

API: [`ThrusterManager`](../reference/api/cosmic_warp/actuators/thrusters.md) ·
[`ReactionWheelManager`](../reference/api/cosmic_warp/actuators/reaction_wheels.md) ·
[`ActuatorManager`](../reference/api/cosmic_warp/actuators/base.md)

"""
Base types and utilities for actuator models.

All actuator modules share common command and state structures for
unified interface across the simulation pipeline.
"""
import warp as wp

from cosmic_warp.core import validation

# Command slots per body: thrusters (ActuatorCommand.thrust_cmd holds 0-3,
# thrust_cmd_ext 4-7) and reaction wheels
MAX_THRUSTERS = 8
MAX_WHEELS = 4


@wp.struct
class ActuatorCommand:
    """
    Unified command input for all actuators.

    This struct provides a standardized interface for control inputs,
    supporting both thruster and reaction wheel commands simultaneously.
    """
    # Thruster commands (normalized 0-1 for each thruster, up to 8)
    thrust_cmd: wp.vec4d       # Thrusters 0-3 command [0, 1]
    thrust_cmd_ext: wp.vec4d   # Thrusters 4-7 command [0, 1]

    # Reaction wheel commands (up to 4 wheels)
    rw_torque_cmd: wp.vec4d    # RW torques [N*m] for wheels 0-3


@wp.func
def command_is_finite(cmd: ActuatorCommand) -> bool:
    """True when every thrust and wheel torque command is finite (no NaN or
    Inf, which the actuator clamps would turn into a finite actuation)."""
    return (wp.isfinite(cmd.thrust_cmd) and wp.isfinite(cmd.thrust_cmd_ext)
            and wp.isfinite(cmd.rw_torque_cmd))


@wp.struct
class ActuatorState:
    """
    Dynamic state of actuators (momentum storage, etc.).

    Reaction wheels have internal momentum that must be tracked.
    Thrusters have no actuator state here; propellant use is tracked as
    the body's mass (State.mass).
    """
    # Control state (reaction wheel angular momenta - up to 4 wheels)
    rw_ang_mom: wp.vec4d  # reaction wheel angular momenta [kg*m^2/s] for wheels 0-3


class ActuatorManager:
    """Centralized command/state interface for spacecraft actuators."""

    def __init__(self, num_envs: int, device: str = "cuda", num_bodies: int = 1):
        self.num_envs = num_envs
        self.num_bodies = num_bodies
        self.num_entities = num_envs * num_bodies
        self.device = device
        self.commands = wp.zeros(self.num_entities, dtype=ActuatorCommand, device=device)
        self.states = wp.zeros(self.num_entities, dtype=ActuatorState, device=device)

    def set_thrust_commands(self, thrust_cmd: list) -> None:
        """Set thruster commands for all entities (broadcast)."""
        cmd = _padded("thrust_cmd", thrust_cmd, MAX_THRUSTERS)
        wp.launch(
            kernel=set_thrust_commands_kernel,
            dim=self.num_entities,
            inputs=[self.commands, wp.vec4d(*cmd[:4]), wp.vec4d(*cmd[4:])],
            device=self.device
        )

    def set_thrust_commands_per_entity(self, thrust_cmds: wp.array) -> None:
        """Set per-entity thruster commands (float64 array [num_entities, 8])."""
        self._check_per_entity("thrust_cmds", thrust_cmds, MAX_THRUSTERS)
        wp.launch(
            kernel=set_thrust_commands_per_entity_kernel,
            dim=self.num_entities,
            inputs=[self.commands, thrust_cmds],
            device=self.device,
        )

    def set_rw_torque_commands(self, rw_torque_cmds: list) -> None:
        """Set reaction wheel torque commands for all entities (broadcast)."""
        rw_cmd_vec = wp.vec4d(*_padded("rw_torque_cmds", rw_torque_cmds, MAX_WHEELS))

        wp.launch(
            kernel=set_rw_commands_kernel,
            dim=self.num_entities,
            inputs=[self.commands, rw_cmd_vec],
            device=self.device
        )

    def set_rw_torque_commands_per_entity(self, rw_torque_cmds: wp.array) -> None:
        """Set per-entity reaction wheel torque commands (float64 array [num_entities, 4])."""
        self._check_per_entity("rw_torque_cmds", rw_torque_cmds, MAX_WHEELS)
        wp.launch(
            kernel=set_rw_commands_per_entity_kernel,
            dim=self.num_entities,
            inputs=[self.commands, rw_torque_cmds],
            device=self.device,
        )

    def set_rw_state(self, rw_ang_mom: list) -> None:
        """Set reaction wheel angular momentum state for all entities (broadcast)."""
        rw_state_vec = wp.vec4d(*_padded("rw_ang_mom", rw_ang_mom, MAX_WHEELS))

        wp.launch(
            kernel=set_rw_state_kernel,
            dim=self.num_entities,
            inputs=[self.states, rw_state_vec],
            device=self.device,
        )

    def set_rw_state_per_entity(self, rw_ang_mom: wp.array) -> None:
        """Set per-entity reaction wheel angular momentum state (float64 array [num_entities, 4])."""
        self._check_per_entity("rw_ang_mom", rw_ang_mom, MAX_WHEELS)
        wp.launch(
            kernel=set_rw_state_per_entity_kernel,
            dim=self.num_entities,
            inputs=[self.states, rw_ang_mom],
            device=self.device,
        )

    def _check_per_entity(self, name: str, values, width: int) -> None:
        """Per-entity arrays are read in place by the kernels: check shape,
        dtype and device (metadata only, no device sync)."""
        if (not isinstance(values, wp.array) or tuple(values.shape) != (self.num_entities, width)
                or values.dtype != wp.float64 or values.device != self.commands.device):
            raise ValueError(
                f"{name} must be a float64 Warp array of shape ({self.num_entities}, {width}) on "
                f"{self.commands.device}, got {validation.describe(values)}"
            )

    def reset(self) -> None:
        """Reset actuator commands and states for all entities."""
        wp.launch(
            kernel=reset_actuator_commands_kernel,
            dim=self.num_entities,
            inputs=[self.commands],
            device=self.device,
        )
        wp.launch(
            kernel=reset_actuator_states_kernel,
            dim=self.num_entities,
            inputs=[self.states],
            device=self.device,
        )

    def reset_masked(self, mask: wp.array) -> None:
        """Reset actuator commands and states where mask is True."""
        wp.launch(
            kernel=reset_actuators_masked_kernel,
            dim=self.num_entities,
            inputs=[self.commands, self.states, mask],
            device=self.device,
        )


@wp.kernel
def reset_actuator_commands_kernel(
    commands: wp.array(dtype=ActuatorCommand)
):
    """Zero all actuator commands."""
    tid = wp.tid()
    commands[tid].thrust_cmd = wp.vec4d()
    commands[tid].thrust_cmd_ext = wp.vec4d()
    commands[tid].rw_torque_cmd = wp.vec4d()


@wp.kernel
def reset_actuator_states_kernel(
    states: wp.array(dtype=ActuatorState)
):
    """Zero all actuator states."""
    tid = wp.tid()
    states[tid].rw_ang_mom = wp.vec4d()


@wp.kernel
def reset_actuators_masked_kernel(
    commands: wp.array(dtype=ActuatorCommand),
    states: wp.array(dtype=ActuatorState),
    mask: wp.array(dtype=wp.bool),
):
    tid = wp.tid()

    if mask[tid]:
        # Commands
        commands[tid].thrust_cmd = wp.vec4d()
        commands[tid].thrust_cmd_ext = wp.vec4d()
        commands[tid].rw_torque_cmd = wp.vec4d()

        # States
        states[tid].rw_ang_mom = wp.vec4d()


@wp.kernel
def set_thrust_commands_kernel(
    commands: wp.array(dtype=ActuatorCommand),
    thrust_cmd: wp.vec4d,
    thrust_cmd_ext: wp.vec4d
):
    """Set thrust commands for all environments."""
    tid = wp.tid()
    commands[tid].thrust_cmd = thrust_cmd
    commands[tid].thrust_cmd_ext = thrust_cmd_ext


@wp.kernel
def set_rw_commands_kernel(
    commands: wp.array(dtype=ActuatorCommand),
    rw_torque: wp.vec4d
):
    """Set RW torque commands for all environments."""
    tid = wp.tid()
    commands[tid].rw_torque_cmd = rw_torque


@wp.kernel
def set_thrust_commands_per_entity_kernel(
    commands: wp.array(dtype=ActuatorCommand),
    thrust_cmds: wp.array(dtype=wp.float64, ndim=2)
):
    """Set thrust commands per entity: columns 0-3 and 4-7 of thrust_cmds."""
    tid = wp.tid()
    commands[tid].thrust_cmd = wp.vec4d(
        thrust_cmds[tid, 0],
        thrust_cmds[tid, 1],
        thrust_cmds[tid, 2],
        thrust_cmds[tid, 3]
    )
    commands[tid].thrust_cmd_ext = wp.vec4d(
        thrust_cmds[tid, 4],
        thrust_cmds[tid, 5],
        thrust_cmds[tid, 6],
        thrust_cmds[tid, 7]
    )


@wp.kernel
def set_rw_commands_per_entity_kernel(
    commands: wp.array(dtype=ActuatorCommand),
    rw_torques: wp.array(dtype=wp.float64, ndim=2)
):
    """Set RW torque commands per entity."""
    tid = wp.tid()
    commands[tid].rw_torque_cmd = wp.vec4d(
        rw_torques[tid, 0],
        rw_torques[tid, 1],
        rw_torques[tid, 2],
        rw_torques[tid, 3]
    )


@wp.kernel
def set_rw_state_kernel(
    states: wp.array(dtype=ActuatorState),
    rw_ang_mom: wp.vec4d,
):
    """Set RW angular momentum state for all environments."""
    tid = wp.tid()
    states[tid].rw_ang_mom = rw_ang_mom


@wp.kernel
def set_rw_state_per_entity_kernel(
    states: wp.array(dtype=ActuatorState),
    rw_ang_mom_values: wp.array(dtype=wp.float64, ndim=2),
):
    """Set RW angular momentum state per entity."""
    tid = wp.tid()
    states[tid].rw_ang_mom = wp.vec4d(
        rw_ang_mom_values[tid, 0],
        rw_ang_mom_values[tid, 1],
        rw_ang_mom_values[tid, 2],
        rw_ang_mom_values[tid, 3],
    )


def _padded(name: str, values, count: int) -> list:
    """Up to count finite numbers, zero-padded to count; more are rejected, not dropped."""
    values = validation.sequence(name, values)
    if len(values) > count:
        raise ValueError(f"{name}: at most {count} values, got {len(values)}")
    return [validation.scalar(name, v) for v in values] + [0.0] * (count - len(values))

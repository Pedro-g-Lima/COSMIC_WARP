"""
Reaction wheel models for spacecraft attitude control.

Supports up to 4 wheels per body with arbitrary spin axes (e.g. 3 orthogonal
wheels or a 4-wheel pyramid), with:
- Momentum storage and hardware saturation (max_momentum)
- Torque limits
- Wheel momentum tracking (speed = momentum / wheel_inertia)
"""
import math

import numpy as np
import warp as wp
from cosmic_warp.core import validation
from cosmic_warp.core.entities import expand_entity_ids
from cosmic_warp.actuators.base import ActuatorCommand, ActuatorState, MAX_WHEELS


@wp.struct
class ReactionWheelConfig:
    """
    Configuration for reaction wheel assembly.

    Supports up to 4 reaction wheels with individual spin axes.
    """
    # Wheel parameters
    max_torque: wp.float64         # Maximum torque per wheel [N*m]
    max_momentum: wp.float64       # Momentum capacity per wheel [kg*m^2/s] (0 = unlimited)

    # Wheel inertia (for computing wheel speed)
    wheel_inertia: wp.float64      # Wheel moment of inertia [kg*m^2]

    # Viscous friction coefficient
    viscous_damping: wp.float64    # Viscous friction coefficient [N*m*s/rad]

    # Spin axis directions (unit vectors in body frame) for up to 4 wheels
    spin_axis_0: wp.vec3d      # Spin axis wheel 0 [unit vector]
    spin_axis_1: wp.vec3d      # Spin axis wheel 1 [unit vector]
    spin_axis_2: wp.vec3d      # Spin axis wheel 2 [unit vector]
    spin_axis_3: wp.vec3d      # Spin axis wheel 3 [unit vector]


@wp.func
def relaxation_factor(x: wp.float64) -> wp.float64:
    """
    (1 - exp(-x)) / x for x >= 0, which tends to 1 as x -> 0.

    The direct form loses all precision for small x (1 - exp(-x) cancels) and
    Warp has no expm1, so small x uses Kahan's expm1 form
    (1 - u) / -log(u), u = exp(-x) rounded, accurate to a few ulps.
    """
    u = wp.exp(-x)
    factor = wp.float64(1.0)                   # u rounds to 1: x below an ulp
    if x >= wp.float64(1.0):
        factor = (wp.float64(1.0) - u) / x
    elif u < wp.float64(1.0):
        factor = (wp.float64(1.0) - u) / -wp.log(u)
    return factor


@wp.func
def compute_rw_angular_momentum(
    config: ReactionWheelConfig,
    cmd: ActuatorCommand,
    actuator_state: ActuatorState,
    wheel_index: int,
    dt: wp.float64
) -> wp.float64:
    """
    Update reaction wheel angular momentum using analytical solution.

    For the dynamics: J * ω̇ = τ_cmd - b*ω

    The exact analytical solution is:
    ω(t) = ω₀ * exp(-b/J * t) + (τ_cmd/b) * (1 - exp(-b/J * t))

    This is more accurate than Euler integration, especially for larger time steps
    or higher damping coefficients.

    Args:
        config: RW configuration
        cmd: Actuator command (vec4 with torques for all wheels)
        actuator_state: Actuator state (current wheel angular momenta)
        wheel_index: Index of the wheel (0-3)
        dt: Time step [s]

    Returns:
        New wheel angular momentum (scalar) [kg*m^2/s]
    """
    # Get commanded torque for this wheel and clamp to limits
    cmd_torque = cmd.rw_torque_cmd[wheel_index]
    cmd_torque = wp.clamp(cmd_torque, -config.max_torque, config.max_torque)

    # Get current momentum for this wheel
    current_mom = actuator_state.rw_ang_mom[wheel_index]

    if config.wheel_inertia <= wp.float64(0.0):
        return wp.float64(0.0)

    if config.viscous_damping <= wp.float64(0.0):
        # Undamped case (b = 0): pure acceleration
        new_ang_mom = current_mom + cmd_torque * dt
    else:
        # Damped case: analytical solution
        # L(t) = L₀ * exp(-b/J * t) + (J * τ/b) * (1 - exp(-b/J * t)),
        # rewritten as the undamped step scaled by (1 - exp(-x))/x, x = b/J*t,
        # which tends to the undamped result as b -> 0 (no 1 - exp(-x)
        # cancellation, no J*τ/b overflow)
        damping_ratio = config.viscous_damping / config.wheel_inertia
        rate = cmd_torque - damping_ratio * current_mom
        new_ang_mom = current_mom + rate * dt * relaxation_factor(damping_ratio * dt)

    # Hardware momentum/speed saturation: a saturated wheel cannot absorb
    # further momentum, so the commanded torque is effectively cut to zero
    # at the boundary (torquing away from saturation still works)
    if config.max_momentum > wp.float64(0.0):
        new_ang_mom = wp.clamp(new_ang_mom, -config.max_momentum, config.max_momentum)

    return new_ang_mom


# Default spin axes of a new simulator: three orthogonal wheels on X, Y, Z
DEFAULT_SPIN_AXES = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


# ============================================================================
# Reaction Wheel Manager Class
# ============================================================================

@wp.kernel
def _set_rw_config_kernel(
    configs: wp.array(dtype=ReactionWheelConfig),
    entity_ids: wp.array(dtype=wp.int32),
    set_max_torque: wp.int32,
    max_torque: wp.float64,
    set_max_momentum: wp.int32,
    max_momentum: wp.float64,
    set_wheel_inertia: wp.int32,
    wheel_inertia: wp.float64,
    set_viscous_damping: wp.int32,
    viscous_damping: wp.float64,
    set_spin_axes: wp.int32,
    spin_axis_0: wp.vec3d,
    spin_axis_1: wp.vec3d,
    spin_axis_2: wp.vec3d,
    spin_axis_3: wp.vec3d
):
    """Write the flagged fields of the RW config of the given entities."""
    tid = wp.tid()
    idx = entity_ids[tid]
    if set_max_torque != 0:
        configs[idx].max_torque = max_torque
    if set_max_momentum != 0:
        configs[idx].max_momentum = max_momentum
    if set_wheel_inertia != 0:
        configs[idx].wheel_inertia = wheel_inertia
    if set_viscous_damping != 0:
        configs[idx].viscous_damping = viscous_damping
    if set_spin_axes != 0:
        configs[idx].spin_axis_0 = spin_axis_0
        configs[idx].spin_axis_1 = spin_axis_1
        configs[idx].spin_axis_2 = spin_axis_2
        configs[idx].spin_axis_3 = spin_axis_3


class ReactionWheelManager:
    """
    Manages reaction wheel configurations and states for parallel entities (envs * bodies).
    """

    def __init__(self, num_envs: int, device: str = "cuda", num_bodies: int = 1,
                 actuator_states: wp.array = None):
        """
        Initialize reaction wheel manager.

        Args:
            num_envs: Number of parallel environments
            device: Compute device
            num_bodies: Number of bodies per environment
            actuator_states: The simulator's ActuatorState array, so a new
                max_momentum can be checked against the stored wheel momenta
                (None: no check)
        """
        self.num_envs = num_envs
        self.num_bodies = num_bodies
        self.num_entities = num_envs * num_bodies
        self.device = device
        self._actuator_states = actuator_states
        self.configs = wp.zeros(self.num_entities, dtype=ReactionWheelConfig, device=device)
        self.set_config(max_torque=0.1, max_momentum=0.0, wheel_inertia=0.01,
                        viscous_damping=0.0, spin_axes=DEFAULT_SPIN_AXES)

    def set_config(
        self,
        *,
        max_torque: float = None,
        max_momentum: float = None,
        wheel_inertia: float = None,
        viscous_damping: float = None,
        spin_axes: list = None,
        env_ids: list = None,
        body_ids: list = None,
        entity_ids: list = None
    ) -> None:
        """
        Set RW configuration for all entities or a subset.

        Only the fields given are checked and written; None leaves a field
        unchanged for the selected entities. A new simulator starts with
        max_torque 0.1, max_momentum 0 (no limit), wheel_inertia 0.01, no
        damping and three orthogonal wheels (DEFAULT_SPIN_AXES).

        Args:
            max_torque: Maximum torque per wheel [N*m], >= 0
            max_momentum: Momentum capacity per wheel [kg*m^2/s], >= 0; the
                          wheel saturates (absorbs no further momentum) at
                          this value. 0 disables the limit. A positive value
                          below a selected wheel's stored momentum is
                          rejected (the next step would dump the excess into
                          the body at once).
            wheel_inertia: Wheel moment of inertia [kg*m^2], > 0
            viscous_damping: Viscous friction coefficient [N*m*s/rad], >= 0
            spin_axes: List of (x, y, z) tuples for each wheel's spin axis (max 4),
                      normalized here; (0, 0, 0) leaves the wheel unused
            env_ids: List of environment indices to configure (None = all)
            body_ids: List of body indices to configure (None = all bodies)
            entity_ids: Optional flat entity indices (env_id * num_bodies + body_id)
        """
        if max_torque is not None:
            max_torque = validation.scalar("max_torque", max_torque, minimum=0.0)
        if max_momentum is not None:
            max_momentum = validation.scalar("max_momentum", max_momentum, minimum=0.0)
        if wheel_inertia is not None:
            wheel_inertia = validation.scalar("wheel_inertia", wheel_inertia, positive=True)
        if viscous_damping is not None:
            viscous_damping = validation.scalar("viscous_damping", viscous_damping, minimum=0.0)
        axes = [wp.vec3d()] * 4
        if spin_axes is not None:
            axes = self._spin_axes(spin_axes)

        entity_ids = expand_entity_ids(self.num_envs, self.num_bodies,
                                       env_ids, body_ids, entity_ids)
        if entity_ids is None:
            entity_ids = list(range(self.num_entities))
        if len(entity_ids) == 0:
            return
        if max_momentum is not None and max_momentum > 0.0:
            self._check_stored_momentum(max_momentum, entity_ids)

        entity_ids_device = wp.array(entity_ids, dtype=wp.int32, device=self.device)
        wp.launch(
            kernel=_set_rw_config_kernel,
            dim=len(entity_ids),
            inputs=[
                self.configs,
                entity_ids_device,
                int(max_torque is not None),
                float(max_torque or 0.0),
                int(max_momentum is not None),
                float(max_momentum or 0.0),
                int(wheel_inertia is not None),
                float(wheel_inertia or 0.0),
                int(viscous_damping is not None),
                float(viscous_damping or 0.0),
                int(spin_axes is not None),
                axes[0],
                axes[1],
                axes[2],
                axes[3]
            ],
            device=self.device
        )

    def _check_stored_momentum(self, max_momentum: float, entity_ids: list) -> None:
        """Reject a capacity below a wheel's stored momentum: the next step
        would clamp the wheel and dump the excess into the body as one
        impulse, beyond any torque limit. Configuration time only (reads the
        wheel momenta to the host)."""
        if self._actuator_states is None:
            return
        ids = np.asarray(entity_ids)
        h = np.abs(self._actuator_states.numpy()["rw_ang_mom"][ids])
        over = h > max_momentum
        if over.any():
            k, wheel = np.argwhere(over)[0]
            raise ValueError(
                f"max_momentum {max_momentum} is below the stored momentum "
                f"{h[k, wheel]} of wheel {wheel} of entity {int(ids[k])}: lower the "
                f"wheel momenta first (set_rw_angular_momentum_state) or reset"
            )

    @staticmethod
    def _spin_axes(spin_axes: list) -> list:
        """Check up to 4 spin axes; return 4 normalized axes (zero = unused)."""
        spin_axes = validation.sequence("spin_axes", spin_axes)
        if len(spin_axes) > MAX_WHEELS:
            raise ValueError(
                f"at most {MAX_WHEELS} reaction wheels per body, got {len(spin_axes)}")
        normalized_axes = []
        for i in range(4):
            axis = (validation.vec3(f"spin_axes[{i}]", spin_axes[i])
                    if i < len(spin_axes) else (0.0, 0.0, 0.0))
            length = math.sqrt(sum(c * c for c in axis))
            if length == 0.0:
                normalized_axes.append(wp.vec3d(0.0, 0.0, 0.0))
            elif length < 1e-10:
                raise ValueError(
                    f"spin_axes[{i}] {axis} has magnitude {length:g}, too small to "
                    f"normalize; use (0.0, 0.0, 0.0) to leave the wheel unused"
                )
            else:
                normalized_axes.append(wp.vec3d(*[c / length for c in axis]))
        return normalized_axes

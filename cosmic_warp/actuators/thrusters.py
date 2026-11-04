"""
Thruster models for spacecraft propulsion.

Supports configurable thruster layouts with position and direction vectors.
Computes both force and torque contributions from thrust commands.
"""
import math

import warp as wp
from cosmic_warp.core import validation
from cosmic_warp.core.state import State
from cosmic_warp.core.vehicle import VehicleConfig
from cosmic_warp.core.entities import expand_entity_ids
from cosmic_warp.actuators.base import ActuatorCommand, MAX_THRUSTERS

# Standard gravity [m/s^2]
G0_STD = 9.80665

# Default layout of a new simulator: one thruster on each face of a 2 m cube,
# thrusting outward along the face normal (pure translation, no torque)
DEFAULT_POSITIONS = (
    (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
)
DEFAULT_DIRECTIONS = DEFAULT_POSITIONS

# 8x3 matrix holding one row per thruster (positions or directions)
mat83d = wp.types.matrix(shape=(8, 3), dtype=wp.float64)

# Per-thruster active flag (1 = configured thruster, 0 = unused slot). Integer
# so the adjoint pass carries no gradient through it.
vec8i = wp.types.vector(length=8, dtype=wp.int32)


@wp.struct
class ThrusterConfig:
    """
    Configuration for a thruster array (up to 8 thrusters).

    Each thruster has:
    - Position relative to spacecraft CoM (row of `positions`)
    - Direction (unit vector in body frame, row of `directions`)
    - Maximum thrust magnitude
    """
    positions: mat83d          # Thruster positions in body frame [m]
    directions: mat83d         # Thruster directions (unit vectors in body frame)
    active: vec8i              # 1 = configured thruster, 0 = unused slot

    max_thrust: wp.float64     # Maximum thrust per thruster [N]
    isp: wp.float64            # Specific impulse [s] (for mass flow computation)


@wp.func
def _get_thruster_pos(config: ThrusterConfig, idx: int) -> wp.vec3d:
    """Get thruster position by index."""
    return wp.vec3d(config.positions[idx, 0], config.positions[idx, 1], config.positions[idx, 2])


@wp.func
def _get_thruster_dir(config: ThrusterConfig, idx: int) -> wp.vec3d:
    """Get thruster direction by index."""
    return wp.vec3d(config.directions[idx, 0], config.directions[idx, 1], config.directions[idx, 2])


@wp.func
def _get_thrust_cmd(cmd: ActuatorCommand, idx: int) -> wp.float64:
    """Get thrust command by index (0-7)."""
    if idx < 4:
        return cmd.thrust_cmd[idx]
    else:
        return cmd.thrust_cmd_ext[idx - 4]


@wp.struct
class ThrusterWrench:
    """Combined thruster output over one step.

    force/torque are in the body frame; mdot is the total propellant mass
    flow rate (>= 0), zero when isp <= 0 (ideal thruster, no consumption).
    All three are already scaled by the step's burn fraction. mass_end is
    the mass at the step end, the exact solution of the constant-flow mass
    equation that every integrator stores: exactly the dry mass on the
    burnout step.
    """
    force: wp.vec3d      # Total thrust force, body frame [N]
    torque: wp.vec3d     # Total thrust torque about CoM, body frame [N*m]
    mdot: wp.float64     # Total mass flow rate [kg/s]
    mass_end: wp.float64 # Mass at the step end [kg]


@wp.func
def thruster_wrench(
    config: ThrusterConfig,
    cmd: ActuatorCommand,
    state: State,
    vehicle: VehicleConfig,
    dt: wp.float64
) -> ThrusterWrench:
    """
    Compute the total thruster force, torque and mass flow over one step.

    In the body frame these depend only on the commands, the configuration
    and the propellant left at the step start, so integrators evaluate them
    once per step. Struct built once via its constructor (adjoint-safe).

    On the step where the propellant runs out, all three are scaled by the
    burn fraction (the share of the step the propellant lasts at full flow),
    so the mass reaches the dry mass exactly at the step end. mass_end is
    then mass_dry itself, not a computed difference: one rounded an ulp
    above dry would leave a remaining flow that rounds away, so the mass
    would never reach dry and mass_limit = mass_dry never fire. This only
    stretches the mass profile in time: with a fixed thrust direction, the
    delta-v and propellant use are exact, and the position and attitude
    differ from an exact cutoff by O(dt^2), once per burnout.

    Args:
        config: Thruster configuration
        cmd: Actuator command with thrust values
        state: Spacecraft state at the step start
        vehicle: Vehicle configuration
        dt: Step length [s]

    Returns:
        ThrusterWrench with body-frame force [N], torque [N*m], mass flow
        rate [kg/s] and step-end mass [kg]
    """
    force = wp.vec3d()
    torque = wp.vec3d()
    total_thrust = wp.float64(0.0)

    # Out of propellant, thrusters produce nothing
    propellant = state.mass - vehicle.mass_dry
    if propellant <= wp.float64(0.0):
        return ThrusterWrench(force, torque, wp.float64(0.0),
                              wp.max(state.mass, vehicle.mass_dry))

    for i in range(MAX_THRUSTERS):
        thrust_val = wp.clamp(_get_thrust_cmd(cmd, i), wp.float64(0.0), wp.float64(1.0))
        direction = _get_thruster_dir(config, i)
        position = _get_thruster_pos(config, i)

        # F = T_max * cmd * dir
        thrust_force = config.max_thrust * thrust_val * direction
        force += thrust_force
        # τ = r × F
        torque += wp.cross(position, thrust_force)
        # Unused slots make no force, so they burn no propellant either.
        # config.active is integer: no adjoint traffic on the tape.
        if config.active[i] != 0:
            total_thrust += config.max_thrust * thrust_val

    # mdot = T / (Isp * g0); isp <= 0 models an ideal thruster (no consumption)
    mdot = wp.float64(0.0)
    if config.isp > wp.float64(0.0):
        mdot = total_thrust / (config.isp * wp.float64(G0_STD))

    # On the burnout step the remaining propellant is spread over the step
    # and the mass lands on dry exactly; otherwise one rounding of m - dt*mdot
    burn = wp.float64(1.0)
    mass_end = wp.max(state.mass - dt * mdot, vehicle.mass_dry)
    if mdot * dt >= propellant:
        burn = propellant / (mdot * dt)
        mass_end = vehicle.mass_dry

    return ThrusterWrench(burn * force, burn * torque, burn * mdot, mass_end)


# ============================================================================
# Thruster Manager Class
# ============================================================================

@wp.kernel
def _set_thruster_config_kernel(
    configs: wp.array(dtype=ThrusterConfig),
    entity_ids: wp.array(dtype=wp.int32),
    set_max_thrust: wp.int32,
    max_thrust: wp.float64,
    set_isp: wp.int32,
    isp: wp.float64,
    set_layout: wp.int32,
    positions: mat83d,
    directions: mat83d,
    active: vec8i
) -> None:
    """Write the flagged fields of the thruster config of the given entities."""
    tid = wp.tid()
    idx = entity_ids[tid]
    if set_max_thrust != 0:
        configs[idx].max_thrust = max_thrust
    if set_isp != 0:
        configs[idx].isp = isp
    if set_layout != 0:
        configs[idx].positions = positions
        configs[idx].directions = directions
        configs[idx].active = active


class ThrusterManager:
    """
    Manages thruster configurations for parallel entities (envs * bodies).

    A new manager holds the default 6-thruster layout; set_config (or the
    `body.actuators.thrusters` YAML section) changes it.
    """

    def __init__(self, num_envs: int, device: str = "cuda", num_bodies: int = 1):
        """
        Initialize thruster manager.

        Args:
            num_envs: Number of parallel environments
            device: Compute device
            num_bodies: Number of bodies per environment
        """
        self.num_envs = num_envs
        self.num_bodies = num_bodies
        self.num_entities = num_envs * num_bodies
        self.device = device
        self.configs = wp.zeros(self.num_entities, dtype=ThrusterConfig, device=device)
        self.set_config(max_thrust=10.0, isp=0.0,
                        positions=DEFAULT_POSITIONS, directions=DEFAULT_DIRECTIONS)

    def set_config(
        self,
        *,
        max_thrust: float = None,
        isp: float = None,
        positions: list = None,
        directions: list = None,
        env_ids: list = None,
        body_ids: list = None,
        entity_ids: list = None
    ) -> None:
        """
        Set thruster configuration for all entities or a subset.

        Only the fields given are checked and written; None leaves a field
        unchanged for the selected entities. positions and directions make up
        the layout and are given together (or both omitted). A new simulator
        starts with max_thrust 10 N, isp 0 (no propellant use) and the
        6-thruster DEFAULT_POSITIONS / DEFAULT_DIRECTIONS layout.

        Args:
            max_thrust: Maximum thrust per thruster [N], >= 0
            isp: Specific impulse [s], >= 0 (0 = ideal, no propellant use)
            positions: Up to 8 position tuples (x, y, z) in metres
            directions: Direction tuples [(dx,dy,dz), ...], one per position,
                normalized here; (0,0,0) leaves the slot unused
            env_ids: List of environment indices to configure (None = all)
            body_ids: List of body indices to configure (None = all bodies)
            entity_ids: Optional flat entity indices (env_id * num_bodies + body_id)
        """
        if max_thrust is not None:
            max_thrust = validation.scalar("max_thrust", max_thrust, minimum=0.0)
        if isp is not None:
            isp = validation.scalar("isp", isp, minimum=0.0)
        if (positions is None) != (directions is None):
            raise ValueError(
                "positions and directions must be given together (the thruster "
                "layout), or both omitted to keep the current layout"
            )
        pos_mat, dir_mat, active_vec = mat83d(), mat83d(), vec8i()
        if positions is not None:
            pos_mat, dir_mat, active_vec = self._layout(positions, directions)

        entity_ids = expand_entity_ids(self.num_envs, self.num_bodies,
                                       env_ids, body_ids, entity_ids)
        if entity_ids is None:
            entity_ids = list(range(self.num_entities))
        if len(entity_ids) == 0:
            return

        entity_ids_device = wp.array(entity_ids, dtype=wp.int32, device=self.device)
        wp.launch(
            kernel=_set_thruster_config_kernel,
            dim=len(entity_ids),
            inputs=[
                self.configs,
                entity_ids_device,
                int(max_thrust is not None),
                float(max_thrust or 0.0),
                int(isp is not None),
                float(isp or 0.0),
                int(positions is not None),
                pos_mat,
                dir_mat,
                active_vec
            ],
            device=self.device
        )

    @staticmethod
    def _layout(positions: list, directions: list) -> tuple:
        """Check a thruster layout; return its (positions, directions,
        active) packed into 8 slots, directions normalized."""
        positions = validation.sequence("positions", positions)
        directions = validation.sequence("directions", directions)
        if len(positions) != len(directions):
            raise ValueError(
                f"positions and directions must have the same length, got "
                f"{len(positions)} and {len(directions)}"
            )
        if len(positions) > MAX_THRUSTERS:
            raise ValueError(
                f"at most {MAX_THRUSTERS} thrusters per body, got {len(positions)}"
            )
        positions = [validation.vec3(f"positions[{i}]", p) for i, p in enumerate(positions)]
        directions = [validation.vec3(f"directions[{i}]", d) for i, d in enumerate(directions)]

        # Pad to 8 slots (unused) and pack into 8x3 matrices
        while len(positions) < MAX_THRUSTERS:
            positions.append((0.0, 0.0, 0.0))
            directions.append((0.0, 0.0, 0.0))

        # Normalize thrust directions: thrust magnitude belongs to max_thrust
        # alone. A non-unit direction would scale force but not mass flow,
        # which silently rescales Isp. Zero rows stay zero (inert slots).
        for i, d in enumerate(directions):
            mag = math.sqrt(sum(c * c for c in d))
            if mag == 0.0:
                directions[i] = (0.0, 0.0, 0.0)
            elif mag < 1e-10:
                raise ValueError(
                    f"thruster {i} direction {tuple(d)} has magnitude {mag:g}, "
                    f"too small to normalize; use (0.0, 0.0, 0.0) to leave the "
                    f"slot unused"
                )
            else:
                directions[i] = tuple(c / mag for c in d)

        active_vec = vec8i(*[1 if any(c != 0.0 for c in d) else 0
                             for d in directions])
        pos_mat = mat83d(*[float(c) for p in positions for c in p])
        dir_mat = mat83d(*[float(c) for d in directions for c in d])
        return pos_mat, dir_mat, active_vec

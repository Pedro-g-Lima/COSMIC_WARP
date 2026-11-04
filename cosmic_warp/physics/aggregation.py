import warp as wp

from cosmic_warp.physics.gravity import gravity_force, gravity_gradient_torque
from cosmic_warp.physics.magnetic import magnetic_torque
from cosmic_warp.physics.drag import drag_force
from cosmic_warp.core.state import State
from cosmic_warp.core.vehicle import VehicleConfig
from cosmic_warp.core.flags import SimulationFlags

@wp.func
def compute_forces(
    state: State,
    vehicle: VehicleConfig,
    flags: SimulationFlags,
    thrust_force_body: wp.vec3d
) -> wp.vec3d:
    """
    Compute total forces acting on spacecraft.

    Args:
        state: Current spacecraft state
        vehicle: Vehicle configuration (mass, etc.)
        flags: Simulation flags
        thrust_force_body: Total thruster force in the body frame [N]
            (from thruster_wrench, computed once per step)

    Returns:
        Total force vector in inertial frame [N]
    """
    F_tot = compute_nongravity_forces(state, vehicle, flags, thrust_force_body)

    # Gravitational force (already in inertial frame)
    if flags.gravity:
        F_tot += gravity_force(state, flags)

    return F_tot


@wp.func
def compute_nongravity_forces(
    state: State,
    vehicle: VehicleConfig,
    flags: SimulationFlags,
    thrust_force_body: wp.vec3d
) -> wp.vec3d:
    """
    Forces that do not scale with the spacecraft mass (drag and thrust), in
    the inertial frame [N]. Split out so integrators can treat them apart
    from gravity, which is proportional to the (changing) mass.
    """
    F_tot = wp.vec3d()

    # Atmospheric drag (already in inertial frame)
    if flags.drag:
        F_tot += drag_force(state, vehicle)

    # Thrust force (body frame, needs rotation to inertial frame)
    F_tot += wp.quat_rotate(state.quat, thrust_force_body)

    return F_tot

@wp.func
def compute_torques(
    state: State,
    vehicle: VehicleConfig,
    flags: SimulationFlags,
    thrust_torque: wp.vec3d,
) -> wp.vec3d:
    """
    Compute total torques acting on spacecraft.

    Args:
        state: Current spacecraft state
        vehicle: Vehicle configuration (inertia, etc.)
        flags: Simulation flags
        thrust_torque: Total thruster torque in the body frame [N*m]
            (from thruster_wrench, computed once per step)

    Returns:
        Total torque vector in body frame [N*m]
    """
    M_tot = thrust_torque

    # Gravity gradient torque
    if flags.gravity_gradient:
        M_tot += gravity_gradient_torque(state, vehicle)

    # Magnetic torque
    if flags.magnetic_torque:
        M_tot += magnetic_torque(state, vehicle)

    return M_tot

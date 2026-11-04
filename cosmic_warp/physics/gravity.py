"""
Gravitational force models for spacecraft simulation.

All computations are performed in float64 for research-grade accuracy.
"""
import warp as wp

# Gravitational constants (host-side Python floats; kernel defaults are typed below)
# Earth taken from EGM2008 model
MU_EARTH = 3.986004415e14  # Earth's gravitational parameter [m^3/s^2]
RE_EARTH = 6.3781363e6        # Earth's equatorial radius [m]
J2_EARTH = 1.08262617385222e-3   # Earth's J2 coefficient


@wp.func
def gravity_accel(pos: wp.vec3d, mu: wp.float64) -> wp.vec3d:
    """
    Compute gravitational acceleration from a point mass.

    Args:
        pos: Position vector relative to gravitating body [m]
        mu: Gravitational parameter (G*M) [m^3/s^2]

    Returns:
        Gravitational acceleration [m/s^2]
    """
    r = wp.length(pos)
    if r > wp.float64(0.0):
        return -mu * pos / (r * r * r)
    else:
        return wp.vec3d()


from cosmic_warp.core.state import State
from cosmic_warp.core.vehicle import VehicleConfig
from cosmic_warp.core.flags import SimulationFlags

@wp.func
def gravity_force(
    state: State,
    flags: SimulationFlags,
    mu: wp.float64 = wp.float64(MU_EARTH),
    j2: wp.float64 = wp.float64(J2_EARTH),
    re: wp.float64 = wp.float64(RE_EARTH)
) -> wp.vec3d:
    """
    Compute gravitational force on spacecraft.

    Args:
        state: Spacecraft state (contains position and mass)
        flags: Simulation flags (controls J2 perturbation)
        mu: Gravitational parameter [m^3/s^2], defaults to Earth
        j2: J2 coefficient, defaults to Earth
        re: Reference radius for J2 (e.g. Earth's equatorial radius) [m], defaults to Earth
    Returns:
        Gravitational force vector [N]
    """
    accel_g = gravity_accel(state.pos, mu)

    # Conditionally add J2 perturbation
    if flags.j2_perturbation:
        accel_g += gravity_accel_j2(state.pos, mu, j2, re)

    return state.mass * accel_g


@wp.func
def gravity_gradient_torque(
    state: State,
    vehicle: VehicleConfig,
    mu: wp.float64 = wp.float64(MU_EARTH)
) -> wp.vec3d:
    """
    Compute gravity gradient torque on spacecraft.

    Args:
        state: Spacecraft state (contains position)
        vehicle: Vehicle configuration (contains inertia tensor)
        mu: Gravitational parameter [m^3/s^2], defaults to Earth

    Returns:
        Gravity gradient torque vector [N*m]
    """
    r_body = wp.quat_rotate(wp.quat_inverse(state.quat), state.pos)  # Position from inertial to body frame
    r_len = wp.length(r_body)

    if r_len > wp.float64(0.0):
        r_unit = r_body / r_len
        I_r_unit = wp.mul(vehicle.inertia, r_unit)
        cross_term = wp.cross(r_unit, I_r_unit)

        return wp.float64(3.0) * mu / (r_len * r_len * r_len) * cross_term
    else:
        return wp.vec3d()


@wp.func
def gravity_accel_j2(
    pos: wp.vec3d,
    mu: wp.float64 = wp.float64(MU_EARTH),
    j2: wp.float64 = wp.float64(J2_EARTH),
    re: wp.float64 = wp.float64(RE_EARTH)
) -> wp.vec3d:
    '''Compute the J2 perturbation acceleration delta (without central gravity).
    Args:
        pos: Position vector [m]
        mu: Gravitational parameter [m^3/s^2], defaults to Earth
        j2: J2 coefficient, defaults to Earth
        re: Reference radius for J2 (e.g. Earth's equatorial radius) [m], defaults to Earth
    Returns:
        J2 perturbation acceleration vector [m/s^2]
    '''
    r_len = wp.length(pos)
    if r_len > wp.float64(0.0):
        x, y, z = pos[0], pos[1], pos[2]
        inv_r = wp.float64(1.0) / r_len
        inv_r2 = inv_r * inv_r
        inv_r3 = inv_r2 * inv_r
        z_r2 = z * z * inv_r2

        # Standard J2 perturbation delta term; central -mu*r/r^3 is added in gravity_accel().
        coeff = wp.float64(1.5) * j2 * re * re * inv_r2
        common = mu * inv_r3 * coeff

        ax = common * x * (wp.float64(5.0) * z_r2 - wp.float64(1.0))
        ay = common * y * (wp.float64(5.0) * z_r2 - wp.float64(1.0))
        az = common * z * (wp.float64(5.0) * z_r2 - wp.float64(3.0))
        return wp.vec3d(ax, ay, az)
    else:
        return wp.vec3d()

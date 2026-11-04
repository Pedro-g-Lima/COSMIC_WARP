"""
Magnetic environment and torque models for spacecraft simulation.

Models Earth's magnetic field as a centered dipole aligned with the ECI
Z-axis (no tilt) and fixed in the inertial frame (no Earth rotation). The
~11.3° tilt and Earth's rotation are not modeled yet (see the roadmap).

Magnetic torque:
    τ_mag = m × B

where:
    m  – spacecraft residual magnetic dipole moment [A·m²] (body frame)
    B  – Earth's magnetic field at spacecraft position [T] (body frame)
"""
import warp as wp

from cosmic_warp.physics.gravity import RE_EARTH


# ── Earth magnetic dipole constants ──────────────────────────────────────────
# Equatorial surface field strength [T]
B0_EARTH = wp.constant(wp.float64(3.12e-5))

# Earth radius [m], shared with the gravity model
RE_MAG = wp.constant(wp.float64(RE_EARTH))

# Dipole axis (unit vector, ECI): aligned with -Z, fixed in the inertial frame
_MX = wp.constant(wp.float64(0.0))
_MY = wp.constant(wp.float64(0.0))
_MZ = wp.constant(wp.float64(-1.0))   # negative: geographic North ≈ magnetic South


from cosmic_warp.core.state import State
from cosmic_warp.core.vehicle import VehicleConfig


@wp.func
def earth_magnetic_field(pos: wp.vec3d) -> wp.vec3d:
    """
    Compute Earth's magnetic field at a position (aligned, non-rotating dipole).

    The dipole field is:
        B = B0 * (RE/r)^3 * [3(m̂·r̂)r̂ - m̂]

    Args:
        pos: Spacecraft position in ECI frame [m]

    Returns:
        Magnetic field vector in ECI frame [T]
    """
    r = wp.length(pos)
    if r > wp.float64(0.0):
        r_hat = pos / r
        m_hat = wp.vec3d(_MX, _MY, _MZ)

        scale = B0_EARTH * (RE_MAG / r) * (RE_MAG / r) * (RE_MAG / r)
        m_dot_r = wp.dot(m_hat, r_hat)

        return scale * (wp.float64(3.0) * m_dot_r * r_hat - m_hat)
    else:
        return wp.vec3d()


@wp.func
def magnetic_torque(state: State, vehicle: VehicleConfig) -> wp.vec3d:
    """
    Compute magnetic torque on spacecraft.

    The torque is computed in the body frame:
        τ = m_body × B_body

    where B_body is the Earth field rotated from ECI to body frame.

    Args:
        state:   Spacecraft state (position + attitude quaternion)
        vehicle: Vehicle config (contains magnetic_dipole in body frame [A·m²])

    Returns:
        Magnetic torque vector in body frame [N·m]
    """
    # Earth's field at spacecraft position (ECI frame)
    B_eci = earth_magnetic_field(state.pos)

    # Rotate ECI field to body frame: B_body = R^T · B_eci
    B_body = wp.quat_rotate(wp.quat_inverse(state.quat), B_eci)

    # τ = m × B  (both in body frame)
    return wp.cross(vehicle.magnetic_dipole, B_body)

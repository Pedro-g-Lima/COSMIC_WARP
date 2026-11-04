"""
Atmospheric drag model for spacecraft simulation.

Density comes from the piecewise-exponential atmosphere (Vallado, "Fundamentals
of Astrodynamics and Applications", Table 8-4): the altitude range 0-1000+ km
is split into bands, each with a nominal base density and scale height:

    rho(h) = rho0 * exp(-(h - h0) / H)

Altitude is geometric above a spherical Earth (|r| - RE). Below sea level
(h < 0) the density is held at the sea-level value 1.225 kg/m^3 rather than
extrapolated, so it has zero gradient with respect to position there; above
1000 km the last band is extrapolated. The atmosphere is assumed to co-rotate
with the Earth, so drag acts on the relative velocity:

    v_rel = v - omega_earth x r
    F_drag = -1/2 * rho * cd * A * |v_rel| * v_rel
"""
import warp as wp

from cosmic_warp.physics.gravity import RE_EARTH


# ── Constants ────────────────────────────────────────────────────────────────
# Earth sidereal rotation rate [rad/s] (Vallado)
OMEGA_EARTH = 7.2921159e-5

# Altitude reference: the equatorial radius of the gravity model [m]
RE_DRAG = wp.constant(wp.float64(RE_EARTH))

# Rotation axis along inertial +z: the frame has no epoch (see the theory
# pages), so the atmosphere co-rotates about +z
OMEGA_DRAG = wp.constant(wp.float64(OMEGA_EARTH))

# Piecewise-exponential atmosphere (Vallado Table 8-4).
# Host-side reference copy of the table used in atmosphere_density(), one row
# per band: (base altitude h0 [m], nominal density rho0 [kg/m^3], scale height H [m])
DENSITY_TABLE = [
    (0.0e3, 1.225, 7.249e3),
    (25.0e3, 3.899e-2, 6.349e3),
    (30.0e3, 1.774e-2, 6.682e3),
    (40.0e3, 3.972e-3, 7.554e3),
    (50.0e3, 1.057e-3, 8.382e3),
    (60.0e3, 3.206e-4, 7.714e3),
    (70.0e3, 8.770e-5, 6.549e3),
    (80.0e3, 1.905e-5, 5.799e3),
    (90.0e3, 3.396e-6, 5.382e3),
    (100.0e3, 5.297e-7, 5.877e3),
    (110.0e3, 9.661e-8, 7.263e3),
    (120.0e3, 2.438e-8, 9.473e3),
    (130.0e3, 8.484e-9, 12.636e3),
    (140.0e3, 3.845e-9, 16.149e3),
    (150.0e3, 2.070e-9, 22.523e3),
    (180.0e3, 5.464e-10, 29.740e3),
    (200.0e3, 2.789e-10, 37.105e3),
    (250.0e3, 7.248e-11, 45.546e3),
    (300.0e3, 2.418e-11, 53.628e3),
    (350.0e3, 9.518e-12, 53.298e3),
    (400.0e3, 3.725e-12, 58.515e3),
    (450.0e3, 1.585e-12, 60.828e3),
    (500.0e3, 6.967e-13, 63.822e3),
    (600.0e3, 1.454e-13, 71.835e3),
    (700.0e3, 3.614e-14, 88.667e3),
    (800.0e3, 1.170e-14, 124.64e3),
    (900.0e3, 5.245e-15, 181.05e3),
    (1000.0e3, 3.019e-15, 268.00e3),
]


from cosmic_warp.core.state import State
from cosmic_warp.core.vehicle import VehicleConfig


@wp.func
def atmosphere_density(pos: wp.vec3d) -> wp.float64:
    """
    Piecewise-exponential atmospheric density (Vallado Table 8-4).

    Args:
        pos: Spacecraft position in ECI frame [m]

    Returns:
        Atmospheric density [kg/m^3]; the sea-level value for altitude < 0
    """
    h = wp.length(pos) - RE_DRAG
    # Clamp below sea level to the first band; the last band extrapolates
    # above 1000 km (density becomes negligible).
    if h < wp.float64(0.0):
        h = wp.float64(0.0)

    # Band selection: sequential lower-bound tests, last match wins.
    # Values must stay in sync with DENSITY_TABLE above.
    h0 = wp.float64(0.0)
    rho0 = wp.float64(1.225)
    hs = wp.float64(7.249e3)
    if h >= wp.float64(25.0e3):
        h0 = wp.float64(25.0e3)
        rho0 = wp.float64(3.899e-2)
        hs = wp.float64(6.349e3)
    if h >= wp.float64(30.0e3):
        h0 = wp.float64(30.0e3)
        rho0 = wp.float64(1.774e-2)
        hs = wp.float64(6.682e3)
    if h >= wp.float64(40.0e3):
        h0 = wp.float64(40.0e3)
        rho0 = wp.float64(3.972e-3)
        hs = wp.float64(7.554e3)
    if h >= wp.float64(50.0e3):
        h0 = wp.float64(50.0e3)
        rho0 = wp.float64(1.057e-3)
        hs = wp.float64(8.382e3)
    if h >= wp.float64(60.0e3):
        h0 = wp.float64(60.0e3)
        rho0 = wp.float64(3.206e-4)
        hs = wp.float64(7.714e3)
    if h >= wp.float64(70.0e3):
        h0 = wp.float64(70.0e3)
        rho0 = wp.float64(8.770e-5)
        hs = wp.float64(6.549e3)
    if h >= wp.float64(80.0e3):
        h0 = wp.float64(80.0e3)
        rho0 = wp.float64(1.905e-5)
        hs = wp.float64(5.799e3)
    if h >= wp.float64(90.0e3):
        h0 = wp.float64(90.0e3)
        rho0 = wp.float64(3.396e-6)
        hs = wp.float64(5.382e3)
    if h >= wp.float64(100.0e3):
        h0 = wp.float64(100.0e3)
        rho0 = wp.float64(5.297e-7)
        hs = wp.float64(5.877e3)
    if h >= wp.float64(110.0e3):
        h0 = wp.float64(110.0e3)
        rho0 = wp.float64(9.661e-8)
        hs = wp.float64(7.263e3)
    if h >= wp.float64(120.0e3):
        h0 = wp.float64(120.0e3)
        rho0 = wp.float64(2.438e-8)
        hs = wp.float64(9.473e3)
    if h >= wp.float64(130.0e3):
        h0 = wp.float64(130.0e3)
        rho0 = wp.float64(8.484e-9)
        hs = wp.float64(12.636e3)
    if h >= wp.float64(140.0e3):
        h0 = wp.float64(140.0e3)
        rho0 = wp.float64(3.845e-9)
        hs = wp.float64(16.149e3)
    if h >= wp.float64(150.0e3):
        h0 = wp.float64(150.0e3)
        rho0 = wp.float64(2.070e-9)
        hs = wp.float64(22.523e3)
    if h >= wp.float64(180.0e3):
        h0 = wp.float64(180.0e3)
        rho0 = wp.float64(5.464e-10)
        hs = wp.float64(29.740e3)
    if h >= wp.float64(200.0e3):
        h0 = wp.float64(200.0e3)
        rho0 = wp.float64(2.789e-10)
        hs = wp.float64(37.105e3)
    if h >= wp.float64(250.0e3):
        h0 = wp.float64(250.0e3)
        rho0 = wp.float64(7.248e-11)
        hs = wp.float64(45.546e3)
    if h >= wp.float64(300.0e3):
        h0 = wp.float64(300.0e3)
        rho0 = wp.float64(2.418e-11)
        hs = wp.float64(53.628e3)
    if h >= wp.float64(350.0e3):
        h0 = wp.float64(350.0e3)
        rho0 = wp.float64(9.518e-12)
        hs = wp.float64(53.298e3)
    if h >= wp.float64(400.0e3):
        h0 = wp.float64(400.0e3)
        rho0 = wp.float64(3.725e-12)
        hs = wp.float64(58.515e3)
    if h >= wp.float64(450.0e3):
        h0 = wp.float64(450.0e3)
        rho0 = wp.float64(1.585e-12)
        hs = wp.float64(60.828e3)
    if h >= wp.float64(500.0e3):
        h0 = wp.float64(500.0e3)
        rho0 = wp.float64(6.967e-13)
        hs = wp.float64(63.822e3)
    if h >= wp.float64(600.0e3):
        h0 = wp.float64(600.0e3)
        rho0 = wp.float64(1.454e-13)
        hs = wp.float64(71.835e3)
    if h >= wp.float64(700.0e3):
        h0 = wp.float64(700.0e3)
        rho0 = wp.float64(3.614e-14)
        hs = wp.float64(88.667e3)
    if h >= wp.float64(800.0e3):
        h0 = wp.float64(800.0e3)
        rho0 = wp.float64(1.170e-14)
        hs = wp.float64(124.64e3)
    if h >= wp.float64(900.0e3):
        h0 = wp.float64(900.0e3)
        rho0 = wp.float64(5.245e-15)
        hs = wp.float64(181.05e3)
    if h >= wp.float64(1000.0e3):
        h0 = wp.float64(1000.0e3)
        rho0 = wp.float64(3.019e-15)
        hs = wp.float64(268.00e3)

    return rho0 * wp.exp(-(h - h0) / hs)


@wp.func
def drag_force(state: State, vehicle: VehicleConfig) -> wp.vec3d:
    """
    Compute atmospheric drag force on spacecraft.

        F = -1/2 * rho * cd * A * |v_rel| * v_rel

    with v_rel the velocity relative to the co-rotating atmosphere.

    Args:
        state:   Spacecraft state (position, momentum, mass)
        vehicle: Vehicle config (contains area_drag [m^2] and cd)

    Returns:
        Drag force vector in inertial frame [N]
    """
    if state.mass > wp.float64(0.0):
        rho = atmosphere_density(state.pos)

        v = state.mom / state.mass
        # Co-rotating atmosphere: v_atm = omega x r, axis along inertial +z
        # (see OMEGA_DRAG placeholder note)
        v_atm = wp.cross(
            wp.vec3d(wp.float64(0.0), wp.float64(0.0), OMEGA_DRAG), state.pos
        )
        v_rel = v - v_atm

        coeff = wp.float64(-0.5) * rho * vehicle.cd * vehicle.area_drag
        return coeff * wp.length(v_rel) * v_rel
    else:
        return wp.vec3d()

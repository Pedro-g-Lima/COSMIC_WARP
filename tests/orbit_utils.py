"""
Analytic orbital-mechanics oracles used by the physics validation tests.

All functions are host-side numpy (float64) and independent of the simulator,
so a bug in cosmic_warp cannot hide in the reference solutions.
"""
import numpy as np


def solve_kepler(M: float, e: float, tol: float = 1e-14) -> float:
    """Solve Kepler's equation M = E - e*sin(E) for E (Newton's method)."""
    E = M if e < 0.8 else np.pi
    for _ in range(100):
        f = E - e * np.sin(E) - M
        E -= f / (1.0 - e * np.cos(E))
        if abs(f) < tol:
            break
    return E


def elliptical_state(a: float, e: float, mu: float, t: float):
    """
    Analytic two-body state at time t after periapsis passage.

    Orbit lies in the xy-plane with periapsis on +x, prograde (h along +z).

    Returns:
        (r, v): position [m] and velocity [m/s] as numpy arrays (3,)
    """
    n = np.sqrt(mu / a**3)
    M = n * t
    E = solve_kepler(np.mod(M, 2.0 * np.pi), e)

    cosE, sinE = np.cos(E), np.sin(E)
    r_mag = a * (1.0 - e * cosE)

    # Position in perifocal frame
    x = a * (cosE - e)
    y = a * np.sqrt(1.0 - e**2) * sinE

    # Velocity in perifocal frame
    factor = np.sqrt(mu * a) / r_mag
    vx = -factor * sinE
    vy = factor * np.sqrt(1.0 - e**2) * cosE

    return np.array([x, y, 0.0]), np.array([vx, vy, 0.0])


def orbital_elements(r: np.ndarray, v: np.ndarray, mu: float) -> dict:
    """Compute orbital elements and invariants from an inertial state vector."""
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)

    energy = 0.5 * v_mag**2 - mu / r_mag
    a = -mu / (2.0 * energy)

    # Eccentricity (Laplace-Runge-Lenz) vector
    e_vec = np.cross(v, h) / mu - r / r_mag
    e = np.linalg.norm(e_vec)

    inc = np.arccos(np.clip(h[2] / h_mag, -1.0, 1.0))

    # Node vector and RAAN
    node = np.cross([0.0, 0.0, 1.0], h)
    node_mag = np.linalg.norm(node)
    if node_mag > 1e-12:
        raan = np.arctan2(node[1], node[0])
    else:
        raan = 0.0

    return {"a": a, "e": e, "e_vec": e_vec, "inc": inc, "raan": raan,
            "energy": energy, "h_vec": h, "h": h_mag}


def circular_velocity(mu: float, r: float) -> float:
    return np.sqrt(mu / r)


def orbit_period(mu: float, a: float) -> float:
    return 2.0 * np.pi * np.sqrt(a**3 / mu)


def cw_state(rho0: np.ndarray, drho0: np.ndarray, n: float, t: float):
    """
    Closed-form Clohessy-Wiltshire solution.

    Frame: x radial (outward), y along-track, z cross-track; n is the target's
    mean motion. rho0/drho0 are the initial relative position/velocity in the
    rotating LVLH frame.

    Returns:
        (rho, drho) at time t
    """
    x0, y0, z0 = rho0
    vx0, vy0, vz0 = drho0
    s, c = np.sin(n * t), np.cos(n * t)

    x = (4.0 - 3.0 * c) * x0 + (s / n) * vx0 + (2.0 / n) * (1.0 - c) * vy0
    y = 6.0 * (s - n * t) * x0 + y0 - (2.0 / n) * (1.0 - c) * vx0 \
        + ((4.0 * s - 3.0 * n * t) / n) * vy0
    z = z0 * c + (vz0 / n) * s

    vx = 3.0 * n * s * x0 + c * vx0 + 2.0 * s * vy0
    vy = 6.0 * n * (c - 1.0) * x0 - 2.0 * s * vx0 + (4.0 * c - 3.0) * vy0
    vz = -z0 * n * s + vz0 * c

    return np.array([x, y, z]), np.array([vx, vy, vz])


def lvlh_relative_state(r_t, v_t, r_c, v_c):
    """
    Relative state of a chaser w.r.t. a target, in the target's rotating LVLH
    frame (x radial, y along-track, z cross-track).
    """
    r_mag = np.linalg.norm(r_t)
    x_hat = r_t / r_mag
    h = np.cross(r_t, v_t)
    z_hat = h / np.linalg.norm(h)
    y_hat = np.cross(z_hat, x_hat)
    R = np.vstack([x_hat, y_hat, z_hat])

    omega = h / r_mag**2  # rotation rate of the LVLH frame (inertial coords)

    d_r = r_c - r_t
    d_v = v_c - v_t - np.cross(omega, d_r)

    return R @ d_r, R @ d_v


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (x, y, z, w convention, body->inertial)."""
    qv = q[:3]
    w = q[3]
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + w * v)

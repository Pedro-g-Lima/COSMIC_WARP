"""
Orbital-element helpers.

Host-side (pure Python/NumPy) utilities that convert between classical
orbital elements and Cartesian ECI states, plus Warp twins of the forward
conversion used inside reset kernels (kept side by side with the NumPy
implementation so the two cannot silently drift apart).
"""
import math

import numpy as np
import numpy.typing as npt
import warp as wp

from cosmic_warp.core import validation
from cosmic_warp.physics.gravity import MU_EARTH, RE_EARTH


def elements_to_state(
    semi_major_axis: float,
    eccentricity: float = 0.0,
    inclination: float = 0.0,
    raan: float = 0.0,
    arg_periapsis: float = 0.0,
    true_anomaly: float = 0.0,
    mu: float = MU_EARTH,
):
    """
    Convert classical orbital elements to an ECI position/velocity pair.

    Args:
        semi_major_axis: Semi-major axis [m]
        eccentricity: Eccentricity (0 <= e < 1, bound orbits only)
        inclination: Inclination [rad], in [0, pi]
        raan: Right ascension of the ascending node [rad]
        arg_periapsis: Argument of periapsis [rad]
        true_anomaly: True anomaly [rad]
        mu: Gravitational parameter of the central body [m^3/s^2]

    Returns:
        (position, velocity): NumPy arrays of shape (3,), ECI frame [m], [m/s]
    """
    inclination = validation.polar_angle("inclination", inclination)
    if not 0.0 <= eccentricity < 1.0:
        raise ValueError("eccentricity must be in [0, 1) for a bound orbit.")
    if semi_major_axis <= 0.0:
        raise ValueError("semi_major_axis must be positive.")

    # Perifocal-frame state
    p = semi_major_axis * (1.0 - eccentricity * eccentricity)
    cn, sn = math.cos(true_anomaly), math.sin(true_anomaly)
    r = p / (1.0 + eccentricity * cn)
    r_pf = np.array([r * cn, r * sn, 0.0])
    v_pf = math.sqrt(mu / p) * np.array([-sn, eccentricity + cn, 0.0])

    # Perifocal -> ECI rotation: R3(-raan) R1(-inclination) R3(-arg_periapsis)
    cO, sO = math.cos(raan), math.sin(raan)
    ci, si = math.cos(inclination), math.sin(inclination)
    cw, sw = math.cos(arg_periapsis), math.sin(arg_periapsis)
    rot = np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si, cw * si, ci],
    ])
    return rot @ r_pf, rot @ v_pf


def circular_orbit_state(
    altitude: float,
    inclination: float = 0.0,
    raan: float = 0.0,
    true_anomaly: float = 0.0,
    mu: float = MU_EARTH,
    planet_radius: float = RE_EARTH,
):
    """
    Convenience wrapper: circular orbit specified by altitude above the surface.

    Args:
        altitude: Altitude above planet_radius [m]
        inclination: Inclination [rad], in [0, pi]
        raan: Right ascension of the ascending node [rad]
        true_anomaly: Angular position along the orbit [rad]
        mu: Gravitational parameter of the central body [m^3/s^2]
        planet_radius: Central body equatorial radius [m]

    Returns:
        (position, velocity): NumPy arrays of shape (3,), ECI frame [m], [m/s]
    """
    return elements_to_state(
        planet_radius + altitude,
        eccentricity=0.0,
        inclination=inclination,
        raan=raan,
        arg_periapsis=0.0,
        true_anomaly=true_anomaly,
        mu=mu,
    )


def state_to_elements(position: npt.ArrayLike, velocity: npt.ArrayLike, mu: float = MU_EARTH) -> dict:
    """
    Inverse of elements_to_state: ECI position/velocity -> classical elements.

    Conventions for the degenerate cases: equatorial orbits report raan = 0
    with arg_periapsis measured from +X; circular orbits report
    arg_periapsis = 0 with true_anomaly measured from the ascending node
    (argument of latitude). Angles are returned in [0, 2*pi).

    Args:
        position: ECI position, array-like of shape (3,) [m]
        velocity: ECI velocity, array-like of shape (3,) [m/s]
        mu: Gravitational parameter of the central body [m^3/s^2]

    Returns:
        dict with semi_major_axis, eccentricity, inclination, raan,
        arg_periapsis, true_anomaly

    Raises:
        ValueError: for rectilinear motion (zero angular momentum, including
            r = 0) or an unbound orbit (e >= 1), where the elements of an
            ellipse are undefined.
    """
    two_pi = 2.0 * math.pi
    r = np.asarray(position, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    rn = float(np.linalg.norm(r))

    h = np.cross(r, v)
    hn = float(np.linalg.norm(h))
    if not hn > 1e-12 * rn * float(np.linalg.norm(v)):
        raise ValueError("state_to_elements: zero angular momentum (rectilinear "
                         "motion), the orbit plane is undefined")
    h_hat = h / hn
    node = np.cross([0.0, 0.0, 1.0], h)
    nn = float(np.linalg.norm(node))

    e_vec = np.cross(v, h) / mu - r / rn
    ecc = float(np.linalg.norm(e_vec))
    if ecc >= 1.0:
        raise ValueError(f"state_to_elements: unbound orbit (eccentricity {ecc:.6g} "
                         f">= 1), elliptical elements are undefined")
    sma = -mu / (2.0 * (float(v @ v) / 2.0 - mu / rn))
    # atan2 rather than acos(h_z/|h|), which loses half the digits near
    # i = 0 and i = pi (an error of ~5e-9 rad at i = 1e-8)
    inc = math.atan2(math.hypot(h[0], h[1]), h[2])

    if nn > 1e-12 * hn:
        raan = math.atan2(node[1], node[0]) % two_pi
        node_hat = node / nn
    else:
        raan = 0.0
        node_hat = np.array([1.0, 0.0, 0.0])

    if ecc > 1e-11:
        e_hat = e_vec / ecc
        argp = math.atan2(float(e_vec @ np.cross(h_hat, node_hat)),
                          float(e_vec @ node_hat)) % two_pi
        nu = math.atan2(float(r @ np.cross(h_hat, e_hat)),
                        float(r @ e_hat)) % two_pi
    else:
        argp = 0.0
        nu = math.atan2(float(r @ np.cross(h_hat, node_hat)),
                        float(r @ node_hat)) % two_pi

    return {
        "semi_major_axis": sma,
        "eccentricity": ecc,
        "inclination": inc,
        "raan": raan,
        "arg_periapsis": argp,
        "true_anomaly": nu,
    }


@wp.func
def elements_to_rv(
    sma: wp.float64,
    ecc: wp.float64,
    inc: wp.float64,
    raan: wp.float64,
    argp: wp.float64,
    true_anomaly: wp.float64,
    mu: wp.float64
) -> wp.spatial_vectord:
    """
    Warp twin of elements_to_state for use inside kernels.

    Returns position and velocity packed in a spatial vector: unpack with
    wp.spatial_top (position [m]) and wp.spatial_bottom (velocity [m/s]).
    """
    p = sma * (wp.float64(1.0) - ecc * ecc)
    cn = wp.cos(true_anomaly)
    sn = wp.sin(true_anomaly)
    r = p / (wp.float64(1.0) + ecc * cn)
    r_pf = wp.vec3d(r * cn, r * sn, wp.float64(0.0))
    v_scale = wp.sqrt(mu / p)
    v_pf = wp.vec3d(-v_scale * sn, v_scale * (ecc + cn), wp.float64(0.0))

    # Perifocal -> ECI rotation: R3(-raan) R1(-inclination) R3(-arg_periapsis)
    cO = wp.cos(raan)
    sO = wp.sin(raan)
    ci = wp.cos(inc)
    si = wp.sin(inc)
    cw = wp.cos(argp)
    sw = wp.sin(argp)
    rot = wp.mat33d(
        cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si,
        sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si,
        sw * si, cw * si, ci
    )
    r_eci = wp.mul(rot, r_pf)
    v_eci = wp.mul(rot, v_pf)
    return wp.spatial_vectord(r_eci[0], r_eci[1], r_eci[2], v_eci[0], v_eci[1], v_eci[2])


@wp.func
def mean_to_true_anomaly(mean_anomaly: wp.float64, ecc: wp.float64) -> wp.float64:
    """
    Solve Kepler's equation E - e*sin(E) = M and return the true anomaly.

    Newton iteration with Danby's starter E0 = M + 0.85*e*sign(sin M) and a
    fixed iteration count, so it is branch-free in data-dependent trip count
    and safe inside CUDA graph captures. Accurate to machine precision for
    bound orbits (e < 1).
    """
    e_anom = mean_anomaly + wp.float64(0.85) * ecc * wp.sign(wp.sin(mean_anomaly))
    for _ in range(16):
        f = e_anom - ecc * wp.sin(e_anom) - mean_anomaly
        fp = wp.float64(1.0) - ecc * wp.cos(e_anom)
        e_anom = e_anom - f / fp
    half = e_anom * wp.float64(0.5)
    return wp.float64(2.0) * wp.atan2(
        wp.sqrt(wp.float64(1.0) + ecc) * wp.sin(half),
        wp.sqrt(wp.float64(1.0) - ecc) * wp.cos(half)
    )

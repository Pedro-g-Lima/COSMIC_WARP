"""
J2 perturbation validation against analytic secular rates.

The J2 zonal harmonic causes secular drift of the node and argument of
periapsis:

    dRAAN/dt = -(3/2) * J2 * n * (Re/p)^2 * cos(i)
    dargp/dt =  (3/4) * J2 * n * (Re/p)^2 * (5 cos^2(i) - 1)

We propagate inclined LEOs for one day and compare the fitted secular rates
against the formulas. Short-period J2 oscillations average out in the fit.
"""
import numpy as np

from cosmic_warp.physics.gravity import MU_EARTH, RE_EARTH, J2_EARTH
from conftest import set_state
from orbit_utils import circular_velocity


def nodal_regression(make_sim):
    """One day of a circular 500 km orbit at i = 51.6 deg: sample times [s], the
    unwrapped RAAN [rad], and the analytic secular rate [rad/s]. Also drawn in
    the docs (scripts/figures.py)."""
    a = 6.878e6              # ~500 km altitude
    inc = np.radians(51.6)   # ISS-like inclination

    # Circular inclined orbit, ascending node at +x
    r0 = np.array([a, 0.0, 0.0])
    v_c = circular_velocity(MU_EARTH, a)
    v0 = v_c * np.array([0.0, np.cos(inc), np.sin(inc)])

    dt = 10.0
    sim = make_sim(num_envs=1, integrator="symplectic", dt=dt, j2_perturbation=True)
    set_state(sim, pos=[r0], vel=[v0])

    result = sim.simulate(num_steps=8640, save_states=True)  # one day
    pos = result["positions"].numpy()[:, 0, :]
    vel = result["velocities"].numpy()[:, 0, :]

    # RAAN from the node vector at every sample, unwrapped
    h = np.cross(pos, vel)
    raan = np.unwrap(np.arctan2(h[:, 0], -h[:, 1]))
    t = (np.arange(len(raan)) + 1) * dt

    n = np.sqrt(MU_EARTH / a**3)
    analytic_rate = -1.5 * J2_EARTH * n * (RE_EARTH / a) ** 2 * np.cos(inc)
    return t, raan, analytic_rate


def test_nodal_regression_rate(make_sim):
    t, raan, analytic_rate = nodal_regression(make_sim)
    fitted_rate = np.polyfit(t, raan, 1)[0]

    rel_err = abs(fitted_rate - analytic_rate) / abs(analytic_rate)
    assert rel_err < 0.03, (
        f"nodal regression {fitted_rate:.3e} rad/s vs analytic "
        f"{analytic_rate:.3e} rad/s (rel err {rel_err:.1%})"
    )


def apsidal_rotation(make_sim):
    """One day of an a = 7500 km, e = 0.1 orbit at i = 30 deg: sample times [s],
    the unwrapped argument of periapsis [rad], and the analytic secular rate
    [rad/s]. Also drawn in the docs (scripts/figures.py)."""
    a, e, inc = 7.5e6, 0.1, np.radians(30.0)
    p = a * (1.0 - e**2)
    r_p = a * (1.0 - e)
    v_p = np.sqrt(MU_EARTH * (1.0 + e) / r_p)
    # Periapsis at the ascending node (+x), so argp starts at 0
    r0 = np.array([r_p, 0.0, 0.0])
    v0 = v_p * np.array([0.0, np.cos(inc), np.sin(inc)])

    dt = 10.0
    sim = make_sim(num_envs=1, integrator="symplectic", dt=dt, j2_perturbation=True)
    set_state(sim, pos=[r0], vel=[v0])
    result = sim.simulate(num_steps=8640, save_states=True)
    pos = result["positions"].numpy()[:, 0, :]
    vel = result["velocities"].numpy()[:, 0, :]

    h = np.cross(pos, vel)
    h_hat = h / np.linalg.norm(h, axis=1)[:, None]
    node = np.cross([0.0, 0.0, 1.0], h)
    node_hat = node / np.linalg.norm(node, axis=1)[:, None]
    e_vec = np.cross(vel, h) / MU_EARTH - pos / np.linalg.norm(pos, axis=1)[:, None]
    argp = np.unwrap(np.arctan2(np.einsum("ij,ij->i", e_vec, np.cross(h_hat, node_hat)),
                                np.einsum("ij,ij->i", e_vec, node_hat)))
    t = (np.arange(len(argp)) + 1) * dt

    n = np.sqrt(MU_EARTH / a**3)
    analytic_rate = 0.75 * J2_EARTH * n * (RE_EARTH / p) ** 2 * (5.0 * np.cos(inc) ** 2 - 1.0)
    return t, argp, analytic_rate


def test_apsidal_rotation_rate(make_sim):
    """dargp/dt = (3/4) J2 n (Re/p)^2 (5 cos^2 i - 1): the second secular J2
    rate, fitted from the eccentricity vector over one day."""
    t, argp, analytic_rate = apsidal_rotation(make_sim)
    fitted_rate = np.polyfit(t, argp, 1)[0]
    rel_err = abs(fitted_rate - analytic_rate) / abs(analytic_rate)
    assert rel_err < 0.03, (
        f"apsidal rotation {fitted_rate:.3e} rad/s vs analytic "
        f"{analytic_rate:.3e} rad/s (rel err {rel_err:.1%})"
    )


def test_j2_flag_off_conserves_plane(make_sim):
    """With J2 disabled the orbital plane must not regress at all."""
    a = 6.878e6
    inc = np.radians(51.6)
    r0 = np.array([a, 0.0, 0.0])
    v_c = circular_velocity(MU_EARTH, a)
    v0 = v_c * np.array([0.0, np.cos(inc), np.sin(inc)])

    dt = 10.0
    sim = make_sim(num_envs=1, integrator="symplectic", dt=dt, j2_perturbation=False)
    set_state(sim, pos=[r0], vel=[v0])

    result = sim.simulate(num_steps=2000, save_states=True)
    pos = result["positions"].numpy()[:, 0, :]
    vel = result["velocities"].numpy()[:, 0, :]

    h = np.cross(pos, vel)
    h_hat = h / np.linalg.norm(h, axis=1, keepdims=True)
    drift = np.linalg.norm(h_hat - h_hat[0], axis=1).max()
    assert drift < 1e-9, f"orbital plane drifted without J2: {drift:.2e}"

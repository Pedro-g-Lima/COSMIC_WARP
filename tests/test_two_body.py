"""
Two-body (Kepler) validation against analytic solutions.

Checks, per integrator:
  1. Propagated position matches the analytic Kepler solution.
  2. Conserved quantities: energy, angular momentum vector, eccentricity
     (Laplace-Runge-Lenz) vector.
  3. Order of convergence: halving dt reduces the error by ~2^order. This is
     the strongest test - it validates the *method*, not just the tolerance.
"""
import numpy as np
import pytest

from cosmic_warp.physics.gravity import MU_EARTH
from conftest import get_state, set_state
from orbit_utils import elliptical_state, orbital_elements


A = 7.5e6       # semi-major axis [m]
ECC = 0.1       # eccentricity


def propagate(make_sim, integrator, dt, t_final):
    """Propagate the reference elliptical orbit from periapsis for t_final."""
    r0, v0 = elliptical_state(A, ECC, MU_EARTH, 0.0)
    sim = make_sim(num_envs=1, integrator=integrator, dt=dt)
    set_state(sim, pos=[r0], vel=[v0])

    num_steps = int(round(t_final / dt))
    for _ in range(num_steps):
        sim.step()

    s = get_state(sim)
    return s["pos"][0], s["mom"][0] / s["mass"][0], num_steps * dt


@pytest.mark.parametrize("integrator,dt,pos_tol", [
    # Tolerances 5-15x the measured error (symplectic 9.6 m, RK4 7.8e-7 m,
    # Euler 3.7e3 m)
    ("symplectic", 1.0, 50.0),
    ("rk4", 1.0, 1e-5),
    ("euler", 0.05, 5000.0),  # first-order: dominated by O(dt) truncation
])
def test_position_vs_kepler(make_sim, integrator, dt, pos_tol):
    """Position after ~1/2 orbit matches the analytic Kepler propagation."""
    t_final = 3000.0
    r_sim, _, t = propagate(make_sim, integrator, dt, t_final)
    r_ref, _ = elliptical_state(A, ECC, MU_EARTH, t)
    err = np.linalg.norm(r_sim - r_ref)
    assert err < pos_tol, f"{integrator}: position error {err:.3f} m > {pos_tol} m"


# Tolerances ~10-100x the measured maximum over the orbit:
#   symplectic: energy 1.0e-7, e 5.0e-7, e-vector 5.1e-6; RK4: all <= 4e-13;
#   angular momentum <= 8e-15 for both (central force: no torque)
@pytest.mark.parametrize("integrator,energy_tol,e_tol,e_dir_tol", [
    ("symplectic", 1e-6, 1e-5, 1e-4),
    ("rk4", 1e-13, 1e-12, 1e-11),
])
def test_invariants(make_sim, integrator, energy_tol, e_tol, e_dir_tol):
    """Energy, H vector and eccentricity vector are conserved over one
    orbit: the largest deviation along the trajectory, not just at the end."""
    dt = 1.0
    t_final = 2.0 * np.pi * np.sqrt(A**3 / MU_EARTH)  # one period

    r0, v0 = elliptical_state(A, ECC, MU_EARTH, 0.0)
    el0 = orbital_elements(r0, v0, MU_EARTH)

    sim = make_sim(num_envs=1, integrator=integrator, dt=dt)
    set_state(sim, pos=[r0], vel=[v0])
    result = sim.simulate(int(round(t_final / dt)), save_states=True)
    pos = result["positions"].numpy()[::10, 0]
    vel = result["velocities"].numpy()[::10, 0]
    els = [orbital_elements(p, v, MU_EARTH) for p, v in zip(pos, vel)]

    energy_rel = max(abs(el["energy"] - el0["energy"]) for el in els) / abs(el0["energy"])
    assert energy_rel < energy_tol, f"energy drift {energy_rel:.2e}"

    h_rel = max(np.linalg.norm(el["h_vec"] - el0["h_vec"]) for el in els) / el0["h"]
    assert h_rel < 1e-13, f"angular momentum drift {h_rel:.2e}"

    e_err = max(abs(el["e"] - el0["e"]) for el in els)
    assert e_err < e_tol, f"e drift {e_err:.2e}"
    e_dir_err = max(np.linalg.norm(el["e_vec"] - el0["e_vec"]) for el in els) / el0["e"]
    assert e_dir_err < e_dir_tol, f"eccentricity vector drift {e_dir_err:.2e}"


# integrator, step sizes, expected order (also drawn in the docs: scripts/figures.py)
CONVERGENCE_CASES = [
    ("euler", [2.0, 1.0, 0.5], 1),
    ("symplectic", [8.0, 4.0, 2.0], 2),
    ("rk4", [32.0, 16.0, 8.0], 4),
]


def kepler_errors(make_sim, integrator, dts, t_final=2048.0):
    """Position error against the Kepler solution after t_final, per step size."""
    errors = []
    for dt in dts:
        r_sim, _, t = propagate(make_sim, integrator, dt, t_final)
        r_ref, _ = elliptical_state(A, ECC, MU_EARTH, t)
        errors.append(np.linalg.norm(r_sim - r_ref))
    return errors


@pytest.mark.parametrize("integrator,dts,expected_order", CONVERGENCE_CASES)
def test_convergence_order(make_sim, integrator, dts, expected_order):
    """
    Halving dt must reduce the global error by ~2^order.

    This catches implementation bugs (wrong half-step, stale stage variable)
    that plain tolerance tests absorb silently.
    """
    errors = kepler_errors(make_sim, integrator, dts)

    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    for order in orders:
        assert order > expected_order - 0.5, (
            f"{integrator}: observed order {order:.2f}, expected ~{expected_order} "
            f"(errors: {[f'{e:.2e}' for e in errors]})"
        )

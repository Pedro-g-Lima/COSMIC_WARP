"""
Maneuver validation against analytic astrodynamics results.

Checks:
  1. Hohmann transfer: impulsive burns at periapsis/apoapsis reach the target
     circular orbit predicted by the vis-viva equation.
  2. Finite burn: achieved delta-v matches the Tsiolkovsky rocket equation.
"""
import numpy as np
import pytest

from cosmic_warp.physics.gravity import MU_EARTH
from cosmic_warp.actuators.thrusters import G0_STD
from conftest import get_state, set_state, apply_delta_v
from orbit_utils import circular_velocity, orbit_period, orbital_elements


def test_hohmann_transfer(make_sim):
    r1, r2 = 7.0e6, 9.0e6
    a_t = 0.5 * (r1 + r2)

    v_c1 = circular_velocity(MU_EARTH, r1)
    v_c2 = circular_velocity(MU_EARTH, r2)
    v_peri = np.sqrt(MU_EARTH * (2.0 / r1 - 1.0 / a_t))
    v_apo = np.sqrt(MU_EARTH * (2.0 / r2 - 1.0 / a_t))

    dt = 1.0
    sim = make_sim(num_envs=1, integrator="symplectic", dt=dt)
    set_state(sim, pos=[[r1, 0.0, 0.0]], vel=[[0.0, v_c1, 0.0]])

    # Burn 1: prograde impulse at periapsis onto the transfer ellipse
    s = get_state(sim)
    v_hat = s["mom"][0] / np.linalg.norm(s["mom"][0])
    apply_delta_v(sim, (v_peri - v_c1) * v_hat)

    # Coast half the transfer ellipse to apoapsis
    t_transfer = 0.5 * orbit_period(MU_EARTH, a_t)
    for _ in range(int(round(t_transfer / dt))):
        sim.step()

    s = get_state(sim)
    r_apo_sim = np.linalg.norm(s["pos"][0])
    # Tolerances ~10x the measured errors (apoapsis 4.3 m, a 0.57 m,
    # e 4.5e-5, radius 406 m = e * r2)
    assert abs(r_apo_sim - r2) < 50.0, (
        f"transfer apoapsis {r_apo_sim:.1f} m vs target {r2:.1f} m"
    )

    # Burn 2: circularize at apoapsis
    v_hat = s["mom"][0] / np.linalg.norm(s["mom"][0])
    apply_delta_v(sim, (v_c2 - v_apo) * v_hat)

    s = get_state(sim)
    el = orbital_elements(s["pos"][0], s["mom"][0] / s["mass"][0], MU_EARTH)
    assert abs(el["a"] - r2) < 10.0, f"final a = {el['a']:.1f} m vs {r2:.1f} m"
    assert el["e"] < 2e-4, f"final orbit not circular: e = {el['e']:.2e}"

    # Radius must stay near r2 over a quarter of the final orbit
    for _ in range(int(round(0.25 * orbit_period(MU_EARTH, r2) / dt))):
        sim.step()
    s = get_state(sim)
    r_check = np.linalg.norm(s["pos"][0])
    assert abs(r_check - r2) < 3e3, f"final orbit radius {r_check:.1f} m vs {r2:.1f} m"


@pytest.mark.parametrize("integrator,rel_tol", [
    # ~50x the measured error (symplectic 2.0e-12, RK4 1.3e-12, Euler 1.8e-5)
    ("symplectic", 1e-10),
    ("rk4", 1e-10),
    ("euler", 5e-4),  # first-order in the mass/momentum coupling
])
def test_tsiolkovsky_finite_burn(make_sim, integrator, rel_tol):
    """Burn all propellant along +x in empty space; check dv = Isp*g0*ln(m0/mf)."""
    m0, m_dry = 2.0, 1.0
    isp = 200.0
    max_thrust = 10.0

    dt = 0.01
    sim = make_sim(num_envs=1, integrator=integrator, dt=dt, gravity=False)
    sim.vehicle.set_config(mass_dry=m_dry)
    # Single thruster at the CoM pointing +x: pure force, no torque
    sim.thruster_manager.set_config(
        max_thrust=max_thrust, isp=isp,
        positions=[(0.0, 0.0, 0.0)], directions=[(1.0, 0.0, 0.0)],
    )
    set_state(sim, mass=[m0])
    sim.set_thrust_commands([1.0, 0.0, 0.0, 0.0])

    # mdot = T/(Isp*g0) -> burn time = (m0-m_dry)/mdot; run past cutoff
    mdot = max_thrust / (isp * G0_STD)
    burn_steps = int((m0 - m_dry) / mdot / dt)
    for _ in range(burn_steps + 1000):
        sim.step()

    s = get_state(sim)
    assert abs(s["mass"][0] - m_dry) < 1e-9, "propellant not fully depleted"

    dv_sim = np.linalg.norm(s["mom"][0]) / s["mass"][0]
    dv_ref = isp * G0_STD * np.log(m0 / m_dry)
    rel_err = abs(dv_sim - dv_ref) / dv_ref
    assert rel_err < rel_tol, (
        f"delta-v {dv_sim:.3f} m/s vs Tsiolkovsky {dv_ref:.3f} m/s (rel {rel_err:.2e})"
    )

    # No spurious torque or attitude change from a CoM-aligned thruster
    assert np.linalg.norm(s["ang_mom"][0]) < 1e-12
    assert abs(s["quat"][0][3] - 1.0) < 1e-12


def _burn_to_depletion(make_sim, integrator, dt, m0, steps):
    """Fire one 10 N, Isp 200 s thruster along +x from rest in empty space."""
    sim = make_sim(num_envs=1, integrator=integrator, dt=dt, gravity=False)
    sim.vehicle.set_config(mass_dry=1.0)
    sim.thruster_manager.set_config(max_thrust=10.0, isp=200.0,
                                    positions=[(0.0, 0.0, 0.0)], directions=[(1.0, 0.0, 0.0)])
    set_state(sim, mass=[m0])
    sim.set_thrust_commands([1.0, 0.0, 0.0, 0.0])
    for _ in range(steps):
        sim.step()
    return get_state(sim)


@pytest.mark.parametrize("integrator,rel_tol", [
    ("symplectic", 1e-12),   # exact kick: only rounding
    ("rk4", 1e-9),           # measured 6e-11
])
def test_burnout_inside_a_step_gives_tsiolkovsky_dv(make_sim, integrator, rel_tol):
    """Propellant running out 37 % into a step used to leave thrust on for the
    rest of it: +2.2 m/s (symplectic) and +0.7 m/s (RK4) of free delta-v at
    dt = 1.7 s. The burnout step now spreads the remaining propellant over
    the step, which keeps delta-v exact."""
    dt = 1.7
    burn_time = 1.0 / (10.0 / (200.0 * G0_STD))   # 1 kg of propellant
    s = _burn_to_depletion(make_sim, integrator, dt, 2.0, int(burn_time / dt) + 5)
    assert s["mass"][0] == 1.0
    dv_ref = 200.0 * G0_STD * np.log(2.0)
    np.testing.assert_allclose(s["mom"][0][0] / s["mass"][0], dv_ref, rtol=rel_tol)


BURNOUT_THRUST, BURNOUT_M0, BURNOUT_M_DRY = 10.0, 2.0, 1.0
BURNOUT_DTS = [1.7, 0.85, 0.425]


def burnout_position_error(make_sim, dt):
    """(measured, expected) position error after a burn to depletion at step dt
    (RK4): the simulated position minus the exact one, and the closed form
    -(T/m_dry) dt^2 f (1 - f) / 2. Also drawn in the docs (scripts/figures.py)."""
    c, m0, m_dry, thrust = 200.0 * G0_STD, BURNOUT_M0, BURNOUT_M_DRY, BURNOUT_THRUST
    mdot = thrust / c
    burn_time = (m0 - m_dry) / mdot
    steps = int(burn_time / dt) + 5
    s = _burn_to_depletion(make_sim, "rk4", dt, m0, steps)

    x_burnout = c / mdot * (m0 - m_dry - m_dry * np.log(m0 / m_dry))
    x_exact = x_burnout + c * np.log(m0 / m_dry) * (steps * dt - burn_time)
    return s["pos"][0][0] - x_exact, burnout_closed_form(dt)


BURNOUT_TIME = (BURNOUT_M0 - BURNOUT_M_DRY) / (BURNOUT_THRUST / (200.0 * G0_STD))


def burnout_closed_form(dt):
    """-(T/m_dry) dt^2 f (1 - f) / 2, with f the fraction of the burnout step
    the propellant lasts."""
    f = BURNOUT_TIME / dt % 1.0
    return -BURNOUT_THRUST / BURNOUT_M_DRY * dt**2 * f * (1.0 - f) / 2.0


@pytest.mark.parametrize("dt", BURNOUT_DTS)
def test_burnout_position_error_is_second_order(make_sim, dt):
    """Spreading the last propellant over the burnout step shifts the position
    by exactly -(T/m_dry) dt^2 f (1 - f) / 2, with f the fraction of the step
    the propellant lasts (RK4, whose own error is negligible here): second
    order, at most T dt^2 / (8 m_dry)."""
    measured, expected = burnout_position_error(make_sim, dt)
    np.testing.assert_allclose(measured, expected, rtol=5e-3)


def test_burnout_step_impulse_is_the_remaining_propellants(make_sim):
    """Euler applies the step-start force for the whole step: with propellant
    for 30 % of the step, the impulse is the thrust over that 30 %."""
    dt, mdot = 1.0, 10.0 / (200.0 * G0_STD)
    s = _burn_to_depletion(make_sim, "euler", dt, 1.0 + 0.3 * mdot * dt, 1)
    assert s["mass"][0] == 1.0
    np.testing.assert_allclose(s["mom"][0][0], 10.0 * 0.3 * dt, rtol=1e-12)

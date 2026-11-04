"""
Closed-form checks of the torque, thrust and coupling models.

Each test starts from rest (zero body angular momentum, or zero momentum
with gravity off), takes one short step, and compares the impulse with the
analytic force or torque: over a step of dt the impulse is F dt (or tau dt)
up to terms of order dt^2 (the spin or velocity the step itself induces),
bounded explicitly where they are visible.
"""
import numpy as np
import pytest

from cosmic_warp.physics.gravity import MU_EARTH
from conftest import get_state, set_state

DT = 1.0e-3
R_LEO = 7.0e6
INERTIA = np.diag([2.0, 3.0, 4.0])


def _quat_z(theta):
    """Rotation by theta about z, Warp order [x, y, z, w] (body -> inertial)."""
    return [0.0, 0.0, np.sin(theta / 2.0), np.cos(theta / 2.0)]


def _torque_after_one_step(sim, **state):
    set_state(sim, **state)
    sim.step()
    return get_state(sim)["ang_mom"][0] / DT


def test_gravity_gradient_torque(make_sim):
    """tau = 3 mu / r^3 (r_hat x I r_hat); for a rotation theta about z with
    r along inertial x: tau_z = 3 mu / r^3 (Ix - Iy) sin(theta) cos(theta)."""
    theta = 0.3
    sim = make_sim(integrator="rk4", dt=DT, gravity=False, gravity_gradient=True)
    sim.vehicle.set_config(inertia=INERTIA)
    tau = _torque_after_one_step(sim, pos=[[R_LEO, 0.0, 0.0]], quat=[_quat_z(theta)])
    expected = 3.0 * MU_EARTH / R_LEO**3 * (2.0 - 3.0) * np.sin(theta) * np.cos(theta)
    np.testing.assert_allclose(tau, [0.0, 0.0, expected], rtol=1e-6, atol=1e-12 * abs(expected))


@pytest.mark.parametrize("pos, b_dir", [
    ([R_LEO, 0.0, 0.0], [0.0, 0.0, 1.0]),    # equator: B = B0 (Re/r)^3 (+z)
    ([0.0, 0.0, R_LEO], [0.0, 0.0, -2.0]),   # north pole: B = -2 B0 (Re/r)^3 z
])
def test_magnetic_dipole_torque(make_sim, pos, b_dir):
    """Aligned dipole (axis -z): B = B0 (Re/r)^3 [3 (m.r) r - m]; tau = m x B."""
    sim = make_sim(integrator="rk4", dt=DT, gravity=False, magnetic_torque=True)
    dipole = np.array([1.0, 0.5, 0.0])
    sim.vehicle.set_config(magnetic_dipole=dipole.tolist())
    tau = _torque_after_one_step(sim, pos=[pos])
    b = 3.12e-5 * (6.3781363e6 / R_LEO) ** 3 * np.array(b_dir)   # model constants B0, Re
    np.testing.assert_allclose(tau, np.cross(dipole, b), rtol=1e-6, atol=1e-18)


def test_thruster_force_and_torque(make_sim):
    """An off-centre thruster gives F = T u (rotated body -> inertial) and
    tau = d x F; with the body rotated 90 deg about z, body +x thrust points
    along inertial +y."""
    thrust, arm = 2.0, np.array([0.0, 0.5, 0.1])
    sim = make_sim(integrator="rk4", dt=DT, gravity=False)
    sim.vehicle.set_config(mass_dry=1.0, inertia=INERTIA)
    sim.thruster_manager.set_config(max_thrust=thrust, isp=0.0,   # ideal: constant mass
                                    positions=[arm.tolist()], directions=[(1.0, 0.0, 0.0)])
    sim.set_thrust_commands([1.0, 0.0, 0.0, 0.0])
    set_state(sim, pos=[[R_LEO, 0.0, 0.0]], quat=[_quat_z(np.pi / 2.0)], mass=[5.0])
    sim.step()
    s = get_state(sim)
    np.testing.assert_allclose(s["mom"][0] / DT, [0.0, thrust, 0.0], rtol=1e-6, atol=1e-6 * thrust)
    # atol: the spin this step induces couples back at O(dt) (measured 6e-9)
    np.testing.assert_allclose(s["ang_mom"][0] / DT, np.cross(arm, [thrust, 0.0, 0.0]),
                               rtol=1e-6, atol=2e-8)


def test_wheel_gyroscopic_coupling(make_sim):
    """A body spinning about x with a wheel holding momentum h along z feels
    dL/dt = -omega x (L + h) = (0, w h, 0) (L parallel to omega drops out)."""
    w, h = 0.2, 0.05
    sim = make_sim(integrator="rk4", dt=DT, gravity=False)
    sim.vehicle.set_config(inertia=INERTIA)
    sim.rw_manager.set_config(max_torque=0.1, max_momentum=1.0, wheel_inertia=0.01,
                              viscous_damping=0.0)
    sim.actuators.set_rw_state([0.0, 0.0, h, 0.0])   # wheel 2 spins about +z
    L0 = INERTIA @ [w, 0.0, 0.0]
    set_state(sim, pos=[[R_LEO, 0.0, 0.0]], ang_mom=[L0])
    sim.step()
    dL = (get_state(sim)["ang_mom"][0] - L0) / DT
    # atol: the induced spin couples back at O(dt) (measured 3e-5 of w h)
    np.testing.assert_allclose(dL, [0.0, w * h, 0.0], rtol=1e-4, atol=1e-4 * w * h)

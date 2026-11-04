"""
Rotational dynamics validation against analytic rigid-body results.

Checks:
  1. Torque-free tumble of an asymmetric body conserves the inertial angular
     momentum vector and (approximately) the rotational kinetic energy.
  2. Torque-free axisymmetric body: the transverse angular velocity precesses
     in the body frame at the analytic rate omega_p = (Is/It - 1) * omega_s.
  3. Quaternion normalization is preserved.
"""
import numpy as np
import pytest

from conftest import set_state
from orbit_utils import quat_rotate


# ~10-100x the measured maxima (symplectic: H 3.3e-14, KE 1.2e-9;
# RK4: H 5.7e-15, KE 7.6e-15)
@pytest.mark.parametrize("integrator,h_tol,ke_tol", [
    ("symplectic", 1e-12, 1e-7),
    ("rk4", 1e-13, 1e-13),
])
def test_torque_free_tumble_invariants(make_sim, integrator, h_tol, ke_tol):
    inertia = np.diag([0.2, 0.3, 0.4])
    L0 = np.array([0.05, 0.12, 0.08])  # body-frame angular momentum

    dt = 1e-3
    sim = make_sim(num_envs=1, integrator=integrator, dt=dt, gravity=False)
    sim.vehicle.set_config(mass_dry=1.0, inertia=inertia.tolist())
    set_state(sim, ang_mom=[L0])

    ke0 = 0.5 * L0 @ np.linalg.inv(inertia) @ L0
    h0_inertial = L0  # identity initial attitude

    result = sim.simulate(num_steps=5000, save_states=True)
    quats = result["quaternions"].numpy()[:, 0, :]
    ang_vels = result["angular_velocities"].numpy()[:, 0, :]

    L_body = ang_vels @ inertia  # I @ omega per sample (I symmetric)
    h_inertial = np.array([quat_rotate(q, L) for q, L in zip(quats, L_body)])

    h_err = np.linalg.norm(h_inertial - h0_inertial, axis=1).max() / np.linalg.norm(h0_inertial)
    assert h_err < h_tol, f"inertial H drift {h_err:.2e}"

    ke = 0.5 * np.einsum("ij,ij->i", ang_vels, L_body)
    ke_err = np.abs(ke - ke0).max() / ke0
    assert ke_err < ke_tol, f"kinetic energy drift {ke_err:.2e}"

    quat_norm_err = np.abs(np.linalg.norm(quats, axis=1) - 1.0).max()
    assert quat_norm_err < 1e-12, f"quaternion norm error {quat_norm_err:.2e}"


def test_axisymmetric_precession_rate(make_sim):
    """
    For I = diag(It, It, Is) spinning about z with a transverse component,
    the body-frame transverse angular velocity rotates at
        omega_p = (Is/It - 1) * omega_z
    """
    It, Is = 0.2, 0.1
    omega0 = np.array([0.1, 0.0, 2.0])
    inertia = np.diag([It, It, Is])
    L0 = inertia @ omega0

    dt = 1e-3
    t_final = 10.0
    sim = make_sim(num_envs=1, integrator="symplectic", dt=dt, gravity=False)
    sim.vehicle.set_config(mass_dry=1.0, inertia=inertia.tolist())
    set_state(sim, ang_mom=[L0])

    num_steps = int(round(t_final / dt))
    result = sim.simulate(num_steps=num_steps, save_states=True)
    ang_vels = result["angular_velocities"].numpy()[:, 0, :]

    omega_p = (Is / It - 1.0) * omega0[2]

    # Phase of the transverse component at the final sample
    t = num_steps * dt
    phase_sim = np.arctan2(ang_vels[-1, 1], ang_vels[-1, 0])
    phase_ref = np.mod(omega_p * t + np.pi, 2.0 * np.pi) - np.pi

    err = np.abs(np.angle(np.exp(1j * (phase_sim - phase_ref))))
    # Measured 2.5e-6 rad (CPU and CUDA)
    assert err < 5e-5, f"precession phase error {err:.2e} rad after {t} s"

    # Spin-axis component and transverse magnitude are constants of motion;
    # measured drift 2.2e-11 and 1.3e-10 rad/s over the run
    assert np.abs(ang_vels[:, 2] - omega0[2]).max() < 1e-9
    transverse = np.linalg.norm(ang_vels[:, :2], axis=1)
    assert np.abs(transverse - omega0[0]).max() < 5e-9

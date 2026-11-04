"""
Relative motion validation against the Clohessy-Wiltshire equations.

A chaser offset from a target on a circular orbit follows the closed-form CW
solution in the target's rotating LVLH frame (to linearization accuracy).
Both spacecraft are propagated as two bodies of one environment, which also
exercises the multi-body path.
"""
import numpy as np

from cosmic_warp.physics.gravity import MU_EARTH
from conftest import set_state
from orbit_utils import circular_velocity, orbit_period, cw_state, lvlh_relative_state


def test_cw_radial_offset(make_sim):
    r_t = 6.878e6
    n = np.sqrt(MU_EARTH / r_t**3)
    offset = 100.0  # radial offset [m] - small enough for CW linearization

    v_c = circular_velocity(MU_EARTH, r_t)
    r_target = np.array([r_t, 0.0, 0.0])
    v_target = np.array([0.0, v_c, 0.0])
    r_chaser = np.array([r_t + offset, 0.0, 0.0])
    v_chaser = v_target.copy()  # same inertial velocity

    dt = 1.0
    sim = make_sim(num_envs=1, num_bodies=2, integrator="symplectic", dt=dt)
    set_state(sim, pos=[r_target, r_chaser], vel=[v_target, v_chaser])

    # Initial relative state in the rotating LVLH frame
    rho0, drho0 = lvlh_relative_state(r_target, v_target, r_chaser, v_chaser)

    period = orbit_period(MU_EARTH, r_t)
    num_steps = int(round(period / dt))
    result = sim.simulate(num_steps=num_steps, save_states=True)
    pos = result["positions"].numpy()   # [steps, 2, 3]
    vel = result["velocities"].numpy()

    # Compare at a handful of epochs across one orbit
    for k in np.linspace(200, num_steps - 1, 8, dtype=int):
        t = (k + 1) * dt
        rho_sim, _ = lvlh_relative_state(pos[k, 0], vel[k, 0], pos[k, 1], vel[k, 1])
        rho_ref, _ = cw_state(rho0, drho0, n, t)
        err = np.linalg.norm(rho_sim - rho_ref)
        # Tolerance is dominated by the CW linearization itself (~offset^2/r)
        assert err < 0.5, (
            f"t={t:.0f} s: relative position error {err:.3f} m "
            f"(sim {rho_sim}, CW {rho_ref})"
        )

    # Cross-track must stay identically zero for a planar setup
    assert abs(pos[:, 1, 2]).max() < 1e-6

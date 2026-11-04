"""
Reaction wheel validation.

Checks:
  1. Wheel torquing is a pure momentum exchange: the total angular momentum
     (body + wheels) in the inertial frame is conserved to near machine
     precision, while the wheel momentum follows L = tau * t (undamped).
  2. The damped wheel model matches an independent fine-step RK4 reference of
     J*omega_dot = tau - b*omega.
"""
import numpy as np
import pytest
import warp as wp

from conftest import set_state
from orbit_utils import quat_rotate


def total_inertial_momentum(quats, ang_vels, inertia, rw_mom, axis):
    """Inertial H = R(q) @ (I*omega + sum_wheels L_w * axis_w)."""
    L_body = ang_vels @ inertia
    H = np.array([
        quat_rotate(q, L + Lw * axis)
        for q, L, Lw in zip(quats, L_body, rw_mom)
    ])
    return H


@pytest.mark.parametrize("integrator", ["symplectic", "euler", "rk4"])
def test_momentum_exchange_conservation(make_sim, integrator):
    inertia = np.diag([0.2, 0.25, 0.3])
    axis = np.array([0.0, 0.0, 1.0])
    tau = 0.05
    dt = 0.01
    num_steps = 2000  # 20 s

    sim = make_sim(num_envs=1, integrator=integrator, dt=dt, gravity=False)
    sim.vehicle.set_config(mass_dry=1.0, inertia=inertia.tolist())
    sim.rw_manager.set_config(
        max_torque=0.1, wheel_inertia=0.01, viscous_damping=0.0,
        spin_axes=[tuple(axis)],
    )
    set_state(sim)
    sim.set_rw_torque_commands([tau])

    result = sim.simulate(num_steps=num_steps, save_states=True, save_rw_states=True)
    quats = result["quaternions"].numpy()[:, 0, :]
    ang_vels = result["angular_velocities"].numpy()[:, 0, :]
    rw_omega = result["rw_angular_velocities"].numpy()[:, 0, 0]
    rw_mom = rw_omega * 0.01  # wheel momentum = J_w * omega_w

    # Undamped wheel: L_w(t) = tau * t exactly
    t = (np.arange(num_steps) + 1) * dt
    assert np.abs(rw_mom - tau * t).max() < 1e-12, "wheel momentum not tau*t"

    # Total angular momentum starts at zero and must stay zero
    H = total_inertial_momentum(quats, ang_vels, inertia, rw_mom, axis)
    H_err = np.linalg.norm(H, axis=1).max()
    assert H_err < 1e-10, f"{integrator}: total H not conserved: {H_err:.2e}"

    # Body counter-rotates: I*omega_z = -L_w
    body_L_z = ang_vels[:, 2] * inertia[2, 2]
    assert np.abs(body_L_z + rw_mom).max() < 1e-10


SATURATION = {"J_w": 0.01, "tau": 0.05, "L_max": 0.02, "dt": 0.01}


def saturating_wheel(make_sim, num_steps=200):
    """A wheel torqued into saturation at t = L_max/tau = 0.4 s, run 2 s.
    Returns the simulator (for the test to continue) and the saved trajectory.
    Also drawn in the docs (scripts/figures.py)."""
    inertia = np.diag([0.2, 0.25, 0.3])
    axis = np.array([0.0, 0.0, 1.0])
    J_w, tau, L_max, dt = (SATURATION[k] for k in ("J_w", "tau", "L_max", "dt"))

    sim = make_sim(num_envs=1, integrator="symplectic", dt=dt, gravity=False)
    sim.vehicle.set_config(mass_dry=1.0, inertia=inertia.tolist())
    sim.rw_manager.set_config(
        max_torque=0.1, max_momentum=L_max, wheel_inertia=J_w,
        viscous_damping=0.0, spin_axes=[tuple(axis)],
    )
    set_state(sim)
    sim.set_rw_torque_commands([tau])
    return sim, sim.simulate(num_steps=num_steps, save_states=True, save_rw_states=True)


def test_momentum_saturation(make_sim):
    """
    A wheel with max_momentum pins at the limit (absorbing no further
    momentum), total H stays conserved, and torquing back down unwinds it.
    """
    inertia = np.diag([0.2, 0.25, 0.3])
    axis = np.array([0.0, 0.0, 1.0])
    J_w, tau, L_max, dt = (SATURATION[k] for k in ("J_w", "tau", "L_max", "dt"))

    # Saturation is reached at t = L_max/tau = 0.4 s; run well past it
    sim, result = saturating_wheel(make_sim)
    quats = result["quaternions"].numpy()[:, 0, :]
    ang_vels = result["angular_velocities"].numpy()[:, 0, :]
    rw_mom = result["rw_angular_velocities"].numpy()[:, 0, 0] * J_w

    # Wheel momentum never exceeds the capacity and ends pinned at it
    assert rw_mom.max() <= L_max + 1e-12
    assert abs(rw_mom[-1] - L_max) < 1e-12, "wheel did not pin at max_momentum"

    # Total angular momentum stays zero through and beyond saturation
    H = total_inertial_momentum(quats, ang_vels, inertia, rw_mom, axis)
    assert np.linalg.norm(H, axis=1).max() < 1e-10

    # Body spin rate is frozen once the wheel saturates (no more exchange)
    omega_z = ang_vels[:, 2]
    settle = int(0.5 / dt)  # first sample safely past saturation
    assert np.abs(omega_z[settle:] - omega_z[-1]).max() < 1e-12

    # Reversed torque must unwind the wheel from saturation
    sim.set_rw_torque_commands([-tau])
    result = sim.simulate(num_steps=50, save_rw_states=True, save_states=False)
    rw_mom_rev = result["rw_angular_velocities"].numpy()[:, 0, 0] * J_w
    assert rw_mom_rev[-1] < L_max - 1e-6, "wheel stuck at saturation"


DAMPED = {"J_w": 0.005, "b": 0.02, "tau": 0.04, "dt": 0.05}


def damped_wheel(make_sim, num_steps=1000):
    """Wheel momentum [N·m·s] of a damped wheel under constant torque, one
    sample per step (1000 steps: 50 s, many damping time constants J/b).
    Also drawn in the docs (scripts/figures.py)."""
    J_w, b, tau, dt = (DAMPED[k] for k in ("J_w", "b", "tau", "dt"))
    sim = make_sim(num_envs=1, integrator="symplectic", dt=dt, gravity=False)
    sim.rw_manager.set_config(
        max_torque=0.1, wheel_inertia=J_w, viscous_damping=b,
        spin_axes=[(0.0, 0.0, 1.0)],
    )
    set_state(sim)
    sim.set_rw_torque_commands([tau])

    result = sim.simulate(num_steps=num_steps, save_rw_states=True, save_states=False)
    return result["rw_angular_velocities"].numpy()[:, 0, 0] * J_w


def test_damped_wheel_vs_reference(make_sim):
    """Damped wheel momentum matches an independent RK4 integration."""
    J_w, b, tau, dt = (DAMPED[k] for k in ("J_w", "b", "tau", "dt"))
    num_steps = 1000  # 50 s, several damping time constants
    rw_mom = damped_wheel(make_sim, num_steps)

    # Independent reference: fine-step RK4 of L' = tau - (b/J)*L
    def deriv(L):
        return tau - (b / J_w) * L

    L, ref = 0.0, []
    fine = dt / 100.0
    for _ in range(num_steps * 100):
        k1 = deriv(L)
        k2 = deriv(L + 0.5 * fine * k1)
        k3 = deriv(L + 0.5 * fine * k2)
        k4 = deriv(L + fine * k3)
        L += fine / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        ref.append(L)
    ref = np.array(ref[99::100])

    rel_err = np.abs(rw_mom - ref).max() / np.abs(ref).max()
    assert rel_err < 1e-9, f"damped wheel deviates from ODE reference: {rel_err:.2e}"


@pytest.mark.parametrize("kwargs, match", [
    ({"spin_axes": [(1.0, 0.0, 0.0)] * 5}, "at most 4 reaction wheels"),
    ({"spin_axes": [(1e-20, 0.0, 0.0)]}, "too small to normalize"),
    ({"spin_axes": [(0.0, float("inf"), 0.0)]}, "must be finite"),
    ({"wheel_inertia": 0.0}, "wheel_inertia must be > 0"),
    ({"max_torque": -0.1}, "max_torque must be >= 0"),
    ({"max_momentum": -1.0}, "max_momentum must be >= 0"),
    ({"viscous_damping": -0.01}, "viscous_damping must be >= 0"),
])
def test_invalid_config_raises(make_sim, kwargs, match):
    """Degenerate axes and unphysical parameters are rejected, not substituted."""
    sim = make_sim()
    with pytest.raises(ValueError, match=match):
        sim.rw_manager.set_config(**kwargs)


def test_zero_spin_axis_is_unused_wheel(make_sim):
    """An explicit zero axis leaves the wheel inert, like an unused thruster slot."""
    sim = make_sim()
    sim.rw_manager.set_config(spin_axes=[(0.0, 0.0, 2.0), (0.0, 0.0, 0.0)])
    cfg = sim.rw_manager.configs.numpy()
    np.testing.assert_array_equal(cfg["spin_axis_0"][0], [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(cfg["spin_axis_1"][0], [0.0, 0.0, 0.0])


def test_wheel_momentum_above_capacity_rejected(make_sim):
    """A wheel cannot be set above max_momentum: the next step would clamp it
    and dump the excess into the body as a spurious impulse."""
    sim = make_sim()
    sim.rw_manager.set_config(max_momentum=0.5)
    with pytest.raises(ValueError, match="exceeds max_momentum"):
        sim.set_rw_angular_momentum_state([2.0])
    sim.set_rw_angular_momentum_state([0.5, -0.5])   # at capacity is fine


def test_max_momentum_below_stored_momentum_rejected(make_sim):
    """Lowering max_momentum below a wheel's stored momentum used to be
    accepted: the next step clamped the wheel and dumped the excess into the
    body at once (0.7 N·m·s in one 0.1 s step for a 0.01 N·m wheel). It is now
    rejected, for the selected entities only."""
    sim = make_sim(num_envs=2)
    sim.rw_manager.set_config(max_momentum=1.0)
    h = wp.array([[0.8, 0.0, 0.0, 0.0], [0.05, -0.05, 0.0, 0.0]], dtype=wp.float64,
                 device=sim.device)
    sim.set_rw_angular_momentum_state_per_entity(h)
    with pytest.raises(ValueError, match="below the stored momentum 0.8 of wheel 0 of entity 0"):
        sim.rw_manager.set_config(max_momentum=0.1)
    np.testing.assert_array_equal(sim.rw_manager.configs.numpy()["max_momentum"], 1.0)
    sim.rw_manager.set_config(max_momentum=0.1, env_ids=[1])
    sim.rw_manager.set_config(max_momentum=0.0)   # unlimited: always allowed
    sim.rw_manager.set_config(max_momentum=0.8)   # at the stored momentum


@pytest.mark.parametrize("b", [1e-18, 1e-14, 1e-9, 1e-4, 1.0])
def test_damped_wheel_matches_closed_form_for_any_damping(make_sim, b):
    """The damped update computed 1 - exp(-b dt / J) directly, which cancels
    for small damping: b = 1e-12 was off by 2e-5 and b = 1e-18 left the wheel
    at rest (relative error 1), a jump from b = 0. It now uses
    (1 - exp(-x)) / x in a form accurate for every x, so tiny damping matches
    the closed form and converges to the undamped wheel."""
    J_w, tau, dt, num_steps = 0.01, 0.1, 0.01, 100
    sim = make_sim(num_envs=1, dt=dt, gravity=False)
    sim.rw_manager.set_config(max_torque=1.0, wheel_inertia=J_w, viscous_damping=b,
                              spin_axes=[(0.0, 0.0, 1.0)])
    set_state(sim)
    sim.set_rw_torque_commands([tau])
    sim.simulate(num_steps=num_steps)
    h = sim.actuators.states.numpy()["rw_ang_mom"][0, 0]

    t = num_steps * dt
    exact = tau * J_w / b * -np.expm1(-b / J_w * t)
    np.testing.assert_allclose(h, exact, rtol=1e-13)
    if b < 1e-16:
        np.testing.assert_allclose(h, tau * t, rtol=1e-13)   # the b = 0 wheel

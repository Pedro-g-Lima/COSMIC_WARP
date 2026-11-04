"""
Convergence order of every integrator beyond point-mass translation.

Self-convergence: with y(n) the final state after n steps over a fixed span,
e(n) = |y(n) - y(2n)| shrinks by 2^order per halving. This checks the rate,
not the answer: a model error that every step size of one integrator shares
would pass it. test_integrators_agree checks the answers against each other.
"""
import numpy as np
import pytest

from conftest import get_state, set_state


EXPECTED_ORDER = {"euler": 1, "symplectic": 2, "rk4": 4}

# Asymmetric body, non-principal initial spin: a genuine tumble
INERTIA = np.diag([1.0, 2.0, 3.0])
ANG_MOM0 = [[0.3, 0.05, 0.4]]
LEO = [[6.778e6, 0.0, 0.0]]


def final_state(sim, n):
    sim.simulate(n)
    s = get_state(sim)
    q = s["quat"][0] * np.sign(s["quat"][0][3])   # fix the double cover
    return np.concatenate([s["pos"][0], s["mom"][0], [s["mass"][0]], q, s["ang_mom"][0]])


def observed_orders(run, steps):
    y = [run(n) for n in steps]
    diffs = [np.linalg.norm(y[i] - y[i + 1]) for i in range(len(y) - 1)]
    return [np.log2(diffs[i] / diffs[i + 1]) for i in range(len(diffs) - 1)], diffs


def tumble(make_sim, integrator, n, span=4.0):
    sim = make_sim(integrator=integrator, dt=span / n, gravity=False)
    sim.vehicle.set_config(mass_dry=1.0, inertia=INERTIA)
    set_state(sim, pos=LEO, ang_mom=ANG_MOM0)
    return final_state(sim, n)


def wheel_exchange(make_sim, integrator, n, span=4.0):
    # Damped wheels on all three axes with a transverse body rate, so the
    # gyroscopic coupling omega x (L_body + h_wheels) is active and h(t) is
    # time-varying within each step
    sim = make_sim(integrator=integrator, dt=span / n, gravity=False)
    sim.vehicle.set_config(mass_dry=1.0, inertia=INERTIA)
    sim.rw_manager.set_config(max_torque=1.0, wheel_inertia=0.01, viscous_damping=0.002)
    sim.set_rw_torque_commands([0.1, -0.2, 0.15])
    set_state(sim, pos=LEO, ang_mom=ANG_MOM0)
    return final_state(sim, n)


def wheel_saturation(make_sim, integrator, n, span=4.0):
    # Wheel 1 hits max_momentum at t = 1.35 s, inside a step for every n:
    # the clamp is a kink no integrator locates, so order is capped at 2
    sim = make_sim(integrator=integrator, dt=span / n, gravity=False)
    sim.vehicle.set_config(mass_dry=1.0, inertia=INERTIA)
    sim.rw_manager.set_config(max_torque=1.0, max_momentum=0.27, wheel_inertia=0.01)
    sim.set_rw_torque_commands([0.1, -0.2, 0.15])
    set_state(sim, pos=LEO, ang_mom=ANG_MOM0)
    return final_state(sim, n)


def burn(make_sim, integrator, n, span=100.0):
    # Finite burn in orbit: thrust, mass flow and gravity all act together
    sim = make_sim(integrator=integrator, dt=span / n, gravity=True)
    sim.vehicle.set_config(mass_dry=0.5)
    sim.thruster_manager.set_config(max_thrust=15.0, isp=200.0)
    sim.set_thrust_commands([0.0, 0.0, 1.0, 0.0])
    set_state(sim, pos=LEO, vel=[[0.0, 7668.6, 0.0]], mass=[2.0])
    return final_state(sim, n)


SCENARIOS = {
    "tumble": (tumble, {"euler": [64, 128, 256, 512], "symplectic": [64, 128, 256, 512],
                        "rk4": [16, 32, 64, 128]}),
    "wheel_exchange": (wheel_exchange, {"euler": [64, 128, 256, 512],
                                        "symplectic": [64, 128, 256, 512],
                                        "rk4": [16, 32, 64, 128]}),
    "burn": (burn, {"euler": [64, 128, 256, 512], "symplectic": [16, 32, 64, 128],
                    "rk4": [4, 8, 16, 32]}),
    "wheel_saturation": (wheel_saturation, {"euler": [64, 128, 256, 512],
                                            "symplectic": [64, 128, 256, 512],
                                            "rk4": [32, 64, 128, 256]}),
}

# Events inside a step (actuator limits) are not located: order <= 2
ORDER_CAP = {"wheel_saturation": 2}


def expected_order(integrator, scenario):
    return min(EXPECTED_ORDER[integrator], ORDER_CAP.get(scenario, 4))


@pytest.mark.parametrize("scenario", list(SCENARIOS))
@pytest.mark.parametrize("integrator", list(EXPECTED_ORDER))
def test_integrator_order(make_sim, integrator, scenario):
    run_fn, steps = SCENARIOS[scenario]
    orders, diffs = observed_orders(lambda n: run_fn(make_sim, integrator, n), steps[integrator])
    expected = expected_order(integrator, scenario)
    for order in orders:
        assert order > expected - 0.3, (
            f"{integrator}/{scenario}: observed order {order:.2f}, expected {expected} "
            f"(successive differences: {[f'{d:.2e}' for d in diffs]})"
        )


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_integrators_agree(make_sim, scenario):
    """Each integrator converges to the same state: its distance from a fine
    RK4 reference matches its own Richardson error estimate. The symplectic
    kicks and drift share no stepping code with Euler and RK4, so a model
    error in either would show up here."""
    run_fn, steps = SCENARIOS[scenario]
    reference = run_fn(make_sim, "rk4", 4 * steps["rk4"][-1])
    for integrator in EXPECTED_ORDER:
        order = expected_order(integrator, scenario)
        n = steps[integrator][-1]
        coarse, fine = run_fn(make_sim, integrator, n), run_fn(make_sim, integrator, 2 * n)
        estimate = np.linalg.norm(fine - coarse) / (2**order - 1)
        error = np.linalg.norm(fine - reference)
        # A mid-step kink's error constant depends on where in the step it
        # falls, which changes as the step halves: there the estimate only
        # gives the scale (measured 0.75 for RK4)
        lo, hi = (0.3, 2.0) if scenario in ORDER_CAP else (0.8, 1.2)
        assert lo < error / estimate < hi, (
            f"{integrator}/{scenario}: distance from the RK4 reference {error:.3e}, "
            f"own error estimate {estimate:.3e}"
        )

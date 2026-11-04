"""
No host work in the hot loop.

On CUDA, simulate() captures one step into a CUDA graph and replays it.
Any host synchronization or device-to-host read inside the step (a
validation, a .numpy(), a Python-side branch on device data) makes the
capture raise, so running it with every per-step feature switched on
catches such a regression. VecEnv.step() is covered the same way by
test_rl_env.py::test_cuda_graph_matches_eager.
"""
import numpy as np
import pytest

from cosmic_warp.core.pipeline import SpacecraftSimulator


def _everything_on(device, integrator):
    sim = SpacecraftSimulator(
        num_envs=8, num_bodies=2, integrator=integrator, dt=0.5, device=device,
        gravity=True, j2_perturbation=True, gravity_gradient=True, magnetic_torque=True,
        drag=True, check_bounds=True, check_planet_impact=True, planet_radius=6.3e6,
        pos_limit=1.0e8, mom_limit=1.0e9, ang_mom_limit=1.0, mass_limit=2.0,
    )
    sim.vehicle.set_config(mass_dry=2.0)
    sim.reset_manager.set_bounds(mass_bounds=(3.0, 4.0), body_ids=[0])
    sim.reset_manager.set_orbit_bounds(altitude_bounds=(4.0e5, 5.0e5), body_ids=[1])
    sim.randomizer.set_bounds(inertia_scale=(0.9, 1.1), propellant=(1.0, 2.0))
    sim.thruster_manager.set_config(max_thrust=1.0, isp=200.0)
    sim.set_thrust_commands([0.5, 0.0, 0.0, 0.0])
    sim.set_rw_torque_commands([0.001, 0.0, 0.0])
    sim.reset()
    return sim


@pytest.mark.parametrize("integrator", ["euler", "rk4", "symplectic"])
def test_simulate_runs_under_graph_capture(device, integrator):
    if "cuda" not in str(device):
        pytest.skip("CUDA graph capture needs a CUDA device")
    sim = _everything_on(device, integrator)
    sim.simulate(20, save_states=True, save_rw_states=True, auto_reset_terminated=True)
    assert np.isfinite(sim.states.numpy()["pos"]).all()


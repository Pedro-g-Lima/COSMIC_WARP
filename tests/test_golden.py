"""
Golden trajectories: a seeded run must reproduce bit for bit.

Property tests (convergence, invariants, determinism) cannot see a change
that is internally consistent: a Warp upgrade altering codegen or the RNG,
or an accidental physics edit, would pass them all while silently changing
every trajectory. This test pins one seeded scenario per integrator — every
physics model, thrusters and wheels, two bodies, termination with automatic
resets and domain randomization — to a committed fixture, so such a change
fails at the commit that causes it.

Goldens are per device (`goldens/cpu.npz`, `goldens/<gpu-slug>.npz`) and are
never compared across devices. They are bit-exact to the Warp version the
`dev` dependency group pins; CI's dependency-range job deselects them
(`-m "not golden"`). Regenerating is a deliberate act, reviewed like any
other change:

    pytest tests/test_golden.py --device cpu --update-goldens
    pytest tests/test_golden.py --device cuda --update-goldens
"""
import re
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from cosmic_warp.core.pipeline import SpacecraftSimulator
from cosmic_warp.physics.gravity import RE_EARTH

pytestmark = pytest.mark.golden

GOLDENS = Path(__file__).parent / "goldens"
INTEGRATORS = ("euler", "rk4", "symplectic")
NUM_STEPS = 150


def golden_path(device) -> Path:
    name = "cpu" if str(device) == "cpu" else wp.get_device(device).name
    return GOLDENS / (re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") + ".npz")


def run_scenario(device, integrator) -> dict:
    sim = SpacecraftSimulator(
        num_envs=8, num_bodies=2, integrator=integrator, dt=0.5, device=device,
        drag=True, gravity_gradient=True, magnetic_torque=True,
        check_bounds=True, ang_mom_limit=0.4,
        check_planet_impact=True, planet_radius=RE_EARTH,
        reset_config={"seed": 7},
    )
    sim.vehicle.set_config(mass_dry=10.0, inertia=np.diag([0.8, 1.0, 1.2]),
                           area_drag=1.0, cd=2.2, magnetic_dipole=[0.05, 0.0, 0.0])
    sim.thruster_manager.set_config(max_thrust=0.5, isp=220.0)
    sim.rw_manager.set_config(max_torque=0.01, max_momentum=0.05,
                              wheel_inertia=0.01, viscous_damping=1e-4)
    # Initial tumbles up to |L| = 0.52 straddle ang_mom_limit: some
    # environments terminate and are reset (with new draws) during the run,
    # others run to the end (test_scenario_coverage checks both)
    sim.reset_manager.set_bounds(mass_bounds=(10.5, 11.0),
                                 ang_mom_bounds=([-0.3] * 3, [0.3] * 3))
    sim.randomizer.set_bounds(propellant=(0.2, 0.8), inertia_scale=(0.9, 1.1))
    sim.randomizer.reseed(11)
    sim.reset()
    # A reset zeroes its environment's commands: command again halfway, so
    # the environments that do not reset in the second half end thrusting
    masses = []
    for _ in range(2):
        masses.append(sim.get_masses().numpy().copy())
        sim.set_thrust_commands([0.3, 0.0, 0.1, 0.0])
        sim.set_rw_torque_commands([0.005, -0.005, 0.002, 0.0])
        sim.simulate(num_steps=NUM_STEPS // 2, auto_reset_terminated=True)

    s = sim.states.numpy()
    status = sim.ctx.termination_status.numpy()
    configs = sim.vehicle.configs.numpy()
    out = {f"state_{k}": s[k] for k in ("pos", "mom", "quat", "ang_mom", "mass")}
    out.update(
        rw_ang_mom=sim.actuators.states.numpy()["rw_ang_mom"],
        terminated=status["terminated"],
        reason=status["reason"],
        episode_steps=sim.episode_steps.numpy(),
        reset_rng=sim.reset_manager.rng_state.numpy(),
        randomizer_rng=sim.randomizer.rng_state.numpy(),
        mass_dry=configs["mass_dry"],
        mass_halfway=masses[1],
        inertia=configs["inertia"],
    )
    return {f"{integrator}/{k}": np.asarray(v) for k, v in out.items()}


@pytest.fixture(scope="module")
def runs(device):
    return {k: v for i in INTEGRATORS for k, v in run_scenario(device, i).items()}


def test_scenario_coverage(runs):
    """Guards the scenario itself: resets happen during the run, so the
    goldens cover termination, automatic reset and randomization; and some
    environments end the run under command, so they cover the thrusters
    (propellant burned) and the wheels (momentum stored)."""
    for integrator in INTEGRATORS:
        steps = runs[f"{integrator}/episode_steps"]
        assert (steps < NUM_STEPS).any(), steps
        commanded = steps >= NUM_STEPS // 2   # not reset in the second half
        assert commanded.any(), steps
        burned = runs[f"{integrator}/state_mass"] < runs[f"{integrator}/mass_halfway"]
        assert burned[commanded].all(), runs[f"{integrator}/state_mass"]
        wheels = np.abs(runs[f"{integrator}/rw_ang_mom"]).max(axis=1) > 0.0
        assert wheels[commanded].all(), runs[f"{integrator}/rw_ang_mom"]


def test_golden_trajectories(runs, device, request):
    path = golden_path(device)
    if request.config.getoption("--update-goldens"):
        GOLDENS.mkdir(exist_ok=True)
        np.savez_compressed(path, **runs)
        pytest.skip(f"wrote {path.name}: review and commit it")
    if not path.is_file():
        pytest.skip(f"no golden for this device ({path.name}); record one with --update-goldens")
    golden = np.load(path)
    assert set(golden.files) == set(runs), "the scenario's outputs changed: re-record"
    for key in sorted(runs):
        np.testing.assert_array_equal(runs[key], golden[key], err_msg=key)

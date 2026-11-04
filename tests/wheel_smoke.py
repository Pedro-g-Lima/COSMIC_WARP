"""Installed-wheel smoke test: run from outside the checkout, in a fresh
environment holding only the wheel and its runtime dependencies (CI's
`wheel` job). Not collected by pytest.

Propagates a batch on the CPU through the public entry points — the Python
constructor, a YAML-style config, actuators, resets, termination and the RL
environment without torch — on the installed package only.

    python tests/wheel_smoke.py [CHECKOUT]   # CHECKOUT: must not be imported
"""
import math
import sys
from pathlib import Path

import numpy as np

import cosmic_warp
from cosmic_warp import SpacecraftSimulator
from cosmic_warp.core.config import create_simulator_from_config
from cosmic_warp.core.termination import REASONS
from cosmic_warp.physics.gravity import MU_EARTH, RE_EARTH
from cosmic_warp.rl import VecEnv  # noqa: F401  (importable without torch)

checkout = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
installed = Path(cosmic_warp.__file__).resolve()
assert checkout is None or checkout not in installed.parents, f"imported the checkout: {installed}"

# A batch on the default reset orbit, with thrust and wheels
sim = SpacecraftSimulator(num_envs=8, device="cpu", dt=1.0)
sim.vehicle.set_config(mass_dry=10.0)
sim.reset_manager.set_bounds(mass_bounds=(12.0, 12.0))
sim.thruster_manager.set_config(max_thrust=1.0, isp=220.0)
sim.reset()
sim.set_thrust_commands([1.0])
sim.set_rw_torque_commands([0.01, 0.0, 0.0])
sim.simulate(num_steps=60)
mass = sim.get_masses().numpy()
expected = 12.0 - 60.0 / (220.0 * 9.80665)
assert np.allclose(mass, expected, rtol=1e-12), mass
assert not sim.get_terminated().numpy()["terminated"].any()

# A config dict, as the YAML loader builds it: an exact circular orbit
r = RE_EARTH + 400e3
v = math.sqrt(MU_EARTH / r)
sim = create_simulator_from_config({
    "simulation": {"num_envs": 2, "integrator": "rk4", "dt": 1.0, "device": "cpu"},
    "physics": {"j2_perturbation": False},   # point-mass gravity: constant radius
    "initial_conditions": {"positions": [[r, 0.0, 0.0]] * 2,
                           "velocities": [[0.0, v, 0.0]] * 2,
                           "masses": [1.0, 1.0]},
})
sim.simulate(num_steps=100)
radius = np.linalg.norm(sim.get_positions().numpy(), axis=1)
assert np.allclose(radius, r, rtol=1e-9), radius - r
assert set(REASONS) == {1, 2, 3, 4, 5, 6}

print(f"cosmic_warp {cosmic_warp.__version__} from {installed.parent}: OK")

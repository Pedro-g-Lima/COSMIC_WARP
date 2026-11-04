"""
Shared fixtures for the physics validation suite.

Tests run on CUDA by default; `pytest --device cpu` runs them without a GPU.
"""
import numpy as np
import pytest
import warp as wp

from cosmic_warp.core.pipeline import SpacecraftSimulator


def pytest_addoption(parser):
    parser.addoption(
        "--device", action="store", default="cuda",
        help="Warp device for tests (default: cuda)"
    )
    parser.addoption(
        "--update-goldens", action="store_true", default=False,
        help="Rewrite the golden-trajectory fixtures for this device (tests/test_golden.py)"
    )


@pytest.fixture(scope="session")
def device(request):
    wp.init()
    return request.config.getoption("--device")


def simulator_factory(device):
    """Factory for simulators with test-friendly defaults (no termination).
    A plain function, so the docs' figures (scripts/figures.py) build the
    same simulators as the tests."""
    def _make(**kwargs):
        kwargs.setdefault("device", device)
        kwargs.setdefault("check_bounds", False)
        kwargs.setdefault("j2_perturbation", False)
        return SpacecraftSimulator(**kwargs)
    return _make


@pytest.fixture
def make_sim(device):
    """Factory for simulators with test-friendly defaults (no termination)."""
    return simulator_factory(device)


def get_state(sim) -> dict:
    """Current dynamic state as a dict of numpy arrays [num_entities, ...]."""
    s = sim.states.numpy()
    return {
        "pos": s["pos"].copy(),
        "mom": s["mom"].copy(),
        "mass": s["mass"].copy(),
        "quat": s["quat"].copy(),
        "ang_mom": s["ang_mom"].copy(),
    }


def set_state(sim, pos=None, vel=None, mom=None, quat=None, ang_mom=None, mass=None):
    """
    Set the full dynamic state from host arrays (shape [num_entities, ...]).
    `vel` is converted to momentum using `mass` (which defaults to 1 kg).
    """
    n = sim.num_entities
    if mass is None:
        mass = np.ones(n)
    mass = np.asarray(mass, dtype=np.float64).reshape(n)
    if mom is None:
        vel = np.zeros((n, 3)) if vel is None else np.asarray(vel, dtype=np.float64)
        mom = vel * mass[:, None]
    if pos is None:
        pos = np.zeros((n, 3))
    if quat is None:
        quat = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
    if ang_mom is None:
        ang_mom = np.zeros((n, 3))

    sim.reset_manager.reset_to_conditions(
        sim.states,
        positions=wp.array(np.asarray(pos, dtype=np.float64), dtype=wp.vec3d, device=sim.device),
        momenta=wp.array(np.asarray(mom, dtype=np.float64), dtype=wp.vec3d, device=sim.device),
        angular_momenta=wp.array(np.asarray(ang_mom, dtype=np.float64), dtype=wp.vec3d, device=sim.device),
        quaternions=wp.array(np.asarray(quat, dtype=np.float64), dtype=wp.quatd, device=sim.device),
        masses=wp.array(mass, dtype=wp.float64, device=sim.device),
    )


def apply_delta_v(sim, dv: np.ndarray):
    """Impulsive maneuver: instantaneously add m*dv to every entity's momentum."""
    s = get_state(sim)
    mom = s["mom"] + s["mass"][:, None] * np.asarray(dv, dtype=np.float64)
    sim.reset_manager.reset_to_conditions(
        sim.states,
        positions=wp.array(s["pos"], dtype=wp.vec3d, device=sim.device),
        momenta=wp.array(mom, dtype=wp.vec3d, device=sim.device),
        angular_momenta=wp.array(s["ang_mom"], dtype=wp.vec3d, device=sim.device),
        quaternions=wp.array(s["quat"], dtype=wp.quatd, device=sim.device),
        masses=wp.array(s["mass"], dtype=wp.float64, device=sim.device),
    )

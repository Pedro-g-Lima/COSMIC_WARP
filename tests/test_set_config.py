"""
The broadcast setters (vehicle, thruster and wheel set_config) write only the
fields given: an omitted argument leaves that field unchanged for the
selected entities, it is not reset to a default.
"""
import numpy as np
import pytest
import warp as wp

from cosmic_warp.actuators.reaction_wheels import DEFAULT_SPIN_AXES
from cosmic_warp.actuators.thrusters import DEFAULT_DIRECTIONS, DEFAULT_POSITIONS
from cosmic_warp.core.config import create_simulator_from_config
from cosmic_warp.core.pipeline import SpacecraftSimulator

INERTIA = np.diag([2.0, 3.0, 4.0])


def _cfg(manager):
    return manager.configs.numpy()


def test_fresh_simulator_defaults(device):
    sim = SpacecraftSimulator(num_envs=2, num_bodies=2, device=device)
    v = _cfg(sim.vehicle)
    np.testing.assert_array_equal(v["mass_dry"], 1.0)
    np.testing.assert_array_equal(v["inertia"], np.broadcast_to(np.eye(3) / 6.0, (4, 3, 3)))
    np.testing.assert_allclose(v["inertia_inv"], np.broadcast_to(6.0 * np.eye(3), (4, 3, 3)),
                               rtol=1e-15)
    np.testing.assert_array_equal(v["area_drag"], 1.0)
    np.testing.assert_array_equal(v["cd"], 2.2)
    np.testing.assert_array_equal(v["magnetic_dipole"], 0.0)
    np.testing.assert_array_equal(sim.vehicle.mass_dry_host, 1.0)
    np.testing.assert_array_equal(sim.vehicle.inertia_host, v["inertia"])

    t = _cfg(sim.thruster_manager)
    np.testing.assert_array_equal(t["max_thrust"], 10.0)
    np.testing.assert_array_equal(t["isp"], 0.0)
    layout = np.zeros((8, 3))
    layout[:6] = DEFAULT_POSITIONS
    np.testing.assert_array_equal(t["positions"], np.broadcast_to(layout, (4, 8, 3)))
    np.testing.assert_array_equal(t["directions"], np.broadcast_to(layout, (4, 8, 3)))
    np.testing.assert_array_equal(t["active"], np.broadcast_to([1] * 6 + [0] * 2, (4, 8)))

    w = _cfg(sim.rw_manager)
    np.testing.assert_array_equal(w["max_torque"], 0.1)
    np.testing.assert_array_equal(w["max_momentum"], 0.0)
    np.testing.assert_array_equal(w["wheel_inertia"], 0.01)
    np.testing.assert_array_equal(w["viscous_damping"], 0.0)
    for i, axis in enumerate(list(DEFAULT_SPIN_AXES) + [(0.0, 0.0, 0.0)]):
        np.testing.assert_array_equal(w[f"spin_axis_{i}"], np.broadcast_to(axis, (4, 3)))


def test_vehicle_omitted_fields_unchanged(device):
    """set_config(magnetic_dipole=...) used to reset the dry mass to 1 kg and
    the inertia to the default cube."""
    sim = SpacecraftSimulator(num_envs=2, device=device)
    sim.vehicle.set_config(mass_dry=10.0, inertia=INERTIA, area_drag=0.5, cd=2.0)
    version = sim.vehicle.inertia_version
    sim.vehicle.set_config(magnetic_dipole=[0.01, 0.0, 0.0])
    v = _cfg(sim.vehicle)
    np.testing.assert_array_equal(v["mass_dry"], 10.0)
    np.testing.assert_array_equal(v["inertia"], np.broadcast_to(INERTIA, (2, 3, 3)))
    np.testing.assert_allclose(v["inertia_inv"],
                               np.broadcast_to(np.linalg.inv(INERTIA), (2, 3, 3)), rtol=1e-15)
    np.testing.assert_array_equal(v["area_drag"], 0.5)
    np.testing.assert_array_equal(v["cd"], 2.0)
    np.testing.assert_array_equal(v["magnetic_dipole"][:, 0], 0.01)
    np.testing.assert_array_equal(sim.vehicle.mass_dry_host, 10.0)
    np.testing.assert_array_equal(sim.vehicle.inertia_host, v["inertia"])
    assert sim.vehicle.inertia_version == version  # nominal inertia not re-uploaded

    # mass_dry alone no longer derives a cube inertia from it
    sim.vehicle.set_config(mass_dry=20.0)
    np.testing.assert_array_equal(_cfg(sim.vehicle)["inertia"], np.broadcast_to(INERTIA, (2, 3, 3)))
    np.testing.assert_array_equal(sim.vehicle.mass_dry_host, 20.0)

    # inertia alone refreshes the inverse and the host copy, bumps the version
    sim.vehicle.set_config(inertia=2.0 * INERTIA)
    v = _cfg(sim.vehicle)
    np.testing.assert_allclose(v["inertia_inv"][0], np.linalg.inv(2.0 * INERTIA), rtol=1e-15)
    np.testing.assert_array_equal(sim.vehicle.inertia_host, v["inertia"])
    np.testing.assert_array_equal(v["mass_dry"], 20.0)
    assert sim.vehicle.inertia_version == version + 1


def test_vehicle_selection_and_per_entity_values_preserved(device):
    sim = SpacecraftSimulator(num_envs=2, num_bodies=2, device=device)
    masses = np.array([2.0, 3.0, 4.0, 5.0])
    inertias = np.array([k * INERTIA for k in (1.0, 2.0, 3.0, 4.0)])
    sim.vehicle.set_config_per_entity(mass_dry=masses, inertia=inertias)
    sim.vehicle.set_config(cd=1.5, body_ids=[1])
    v = _cfg(sim.vehicle)
    np.testing.assert_array_equal(v["mass_dry"], masses)
    np.testing.assert_array_equal(v["inertia"], inertias)
    np.testing.assert_array_equal(v["cd"], [2.2, 1.5, 2.2, 1.5])
    np.testing.assert_array_equal(sim.vehicle.mass_dry_host, masses)
    np.testing.assert_array_equal(sim.vehicle.inertia_host, inertias)

    sim.vehicle.set_config(mass_dry=9.0, entity_ids=[2])
    np.testing.assert_array_equal(_cfg(sim.vehicle)["mass_dry"], [2.0, 3.0, 9.0, 5.0])
    np.testing.assert_array_equal(sim.vehicle.mass_dry_host, [2.0, 3.0, 9.0, 5.0])
    np.testing.assert_array_equal(_cfg(sim.vehicle)["inertia"], inertias)


@pytest.mark.parametrize("kwargs, match", [
    ({"mass_dry": np.array([True, True, True, True])}, "mass_dry must hold numbers, got dtype bool"),
    ({"cd": ["1.0", "2.0", "1.0", "2.0"]}, "cd must hold numbers, got '1.0'"),
    ({"cd": [1.0, True, 1.0, 1.0]}, "cd must hold numbers, got True"),
    ({"area_drag": 1.0}, r"area_drag must have shape \(4,\), got \(\)"),
    ({"inertia": np.ones((4, 3))}, r"inertia must have shape \(4, 3, 3\), got \(4, 3\)"),
    ({"magnetic_dipole": np.zeros((3, 3))}, r"magnetic_dipole must have shape \(4, 3\)"),
    ({"cd": lambda d: wp.ones(4, dtype=wp.float32, device=d)},
     r"cd must be a Warp array of float64 with shape \(4,\)"),
    ({"inertia": lambda d: wp.zeros((4, 3, 3), dtype=wp.float64, device=d)},
     "inertia must be a Warp array of mat33d"),
    ({"mass_dry": np.array([1.0, 1.0, 0.0, 1.0])}, "mass_dry must be > 0.0"),
    ({"cd": [1.0, 1.0, 1.0, float("nan")]}, "cd must be finite"),
])
def test_vehicle_per_entity_rejects_before_writing(device, kwargs, match):
    """Booleans and strings are rejected, not converted to numbers; a wrong
    shape, dtype or value raises before any field is written."""
    sim = SpacecraftSimulator(num_envs=2, num_bodies=2, device=device)
    kwargs = {k: v(device) if callable(v) else v for k, v in kwargs.items()}
    before = _cfg(sim.vehicle)
    with pytest.raises(ValueError, match=match):
        sim.vehicle.set_config_per_entity(**{"area_drag": np.full(4, 0.5), **kwargs})
    np.testing.assert_array_equal(_cfg(sim.vehicle), before)


def test_vehicle_randomized_nominal_preserved(device):
    """A set_config of other fields keeps the randomizer's nominal inertia."""
    sim = SpacecraftSimulator(num_envs=4, device=device)
    sim.vehicle.set_config_per_entity(inertia=np.array([k * INERTIA for k in (1.0, 2.0, 3.0, 4.0)]))
    sim.randomizer.set_bounds(inertia_scale=(1.0, 1.0))
    sim.vehicle.set_config(area_drag=0.3)
    sim.reset()
    np.testing.assert_allclose(_cfg(sim.vehicle)["inertia"],
                               [k * INERTIA for k in (1.0, 2.0, 3.0, 4.0)], rtol=1e-15)


def test_thruster_omitted_fields_unchanged(device):
    sim = SpacecraftSimulator(num_envs=3, device=device)
    positions, directions = [(0.0, 0.5, 0.0)], [(2.0, 0.0, 0.0)]
    sim.thruster_manager.set_config(max_thrust=1.0, isp=220.0,
                                    positions=positions, directions=directions)
    before = _cfg(sim.thruster_manager)
    sim.thruster_manager.set_config(max_thrust=2.0, env_ids=[1])
    t = _cfg(sim.thruster_manager)
    np.testing.assert_array_equal(t["max_thrust"], [1.0, 2.0, 1.0])
    np.testing.assert_array_equal(t["isp"], 220.0)
    for key in ("positions", "directions", "active"):
        np.testing.assert_array_equal(t[key], before[key])
    np.testing.assert_array_equal(t["directions"][:, 0], [[1.0, 0.0, 0.0]] * 3)

    # A layout without max_thrust/isp keeps them
    sim.thruster_manager.set_config(positions=DEFAULT_POSITIONS, directions=DEFAULT_DIRECTIONS,
                                    env_ids=[0])
    t = _cfg(sim.thruster_manager)
    np.testing.assert_array_equal(t["max_thrust"], [1.0, 2.0, 1.0])
    np.testing.assert_array_equal(t["isp"], 220.0)
    np.testing.assert_array_equal(t["active"][:, :6].sum(axis=1), [6, 1, 1])


@pytest.mark.parametrize("given", ["positions", "directions"])
def test_thruster_layout_given_together(device, given):
    sim = SpacecraftSimulator(num_envs=1, device=device)
    with pytest.raises(ValueError, match="given together"):
        sim.thruster_manager.set_config(**{given: [(1.0, 0.0, 0.0)]})


def test_wheel_omitted_fields_unchanged(device):
    sim = SpacecraftSimulator(num_envs=2, num_bodies=2, device=device)
    axes = [(1.0, 1.0, 0.0), (0.0, 0.0, 3.0)]
    sim.rw_manager.set_config(max_torque=0.02, max_momentum=1.0, wheel_inertia=0.005,
                              viscous_damping=1e-5, spin_axes=axes)
    sim.rw_manager.set_config(max_momentum=0.4, body_ids=[0])
    w = _cfg(sim.rw_manager)
    np.testing.assert_array_equal(w["max_momentum"], [0.4, 1.0, 0.4, 1.0])
    np.testing.assert_array_equal(w["max_torque"], 0.02)
    np.testing.assert_array_equal(w["wheel_inertia"], 0.005)
    np.testing.assert_array_equal(w["viscous_damping"], 1e-5)
    np.testing.assert_allclose(w["spin_axis_0"], [[2 ** -0.5, 2 ** -0.5, 0.0]] * 4, rtol=1e-15)
    np.testing.assert_array_equal(w["spin_axis_1"], [[0.0, 0.0, 1.0]] * 4)
    np.testing.assert_array_equal(w["spin_axis_2"], 0.0)

    # The momentum check uses the max_momentum just set
    with pytest.raises(ValueError, match="exceeds max_momentum"):
        sim.set_rw_angular_momentum_state([0.5])
    sim.rw_manager.set_config(max_momentum=0.5)
    sim.set_rw_angular_momentum_state([0.5])


@pytest.mark.parametrize("setter, kwargs, match", [
    ("vehicle", {"mass_dry": 0.0}, "mass_dry must be > 0"),
    ("vehicle", {"magnetic_dipole": [0.0, float("nan"), 0.0]}, "must be finite"),
    ("thruster_manager", {"isp": -1.0}, "isp must be >= 0"),
    ("rw_manager", {"wheel_inertia": 0.0}, "wheel_inertia must be > 0"),
    ("rw_manager", {"spin_axes": [(1.0, 0.0, 0.0)] * 5}, "at most 4"),
])
def test_invalid_given_field_writes_nothing(device, setter, kwargs, match):
    sim = SpacecraftSimulator(num_envs=2, device=device)
    manager = getattr(sim, setter)
    before = _cfg(manager)
    with pytest.raises(ValueError, match=match):
        manager.set_config(**kwargs, env_ids=[0])
    assert _cfg(manager).tobytes() == before.tobytes()


@pytest.mark.parametrize("setter", ["vehicle", "thruster_manager", "rw_manager"])
def test_bad_selection_still_rejected(device, setter):
    sim = SpacecraftSimulator(num_envs=2, device=device)
    with pytest.raises(ValueError, match="out of range"):
        getattr(sim, setter).set_config(env_ids=[2])


def test_yaml_omitted_keys_take_defaults(device):
    """The YAML loader still gives every omitted key its default, including
    the cube inertia derived from the YAML mass_dry."""
    sim = create_simulator_from_config({
        "simulation": {"num_envs": 1, "num_bodies": 2, "device": device},
        "reset": {"mass_bounds": [20.0, 21.0]},
        "body": {
            "defaults": {"vehicle": {"mass_dry": 12.0, "cd": 2.0},
                         "actuators": {"thrusters": {"max_thrust": 3.0},
                                       "reaction_wheels": {"max_torque": 0.5}}},
            "per_body": [{"body_id": 1,
                          "vehicle": {"magnetic_dipole": [0.1, 0.0, 0.0]},
                          "actuators": {"thrusters": {"directions": [(0.0, 1.0, 0.0)] * 6},
                                        "reaction_wheels": {"spin_axes": [(0.0, 0.0, 1.0)]}}}],
        },
    })
    v = _cfg(sim.vehicle)
    np.testing.assert_array_equal(v["mass_dry"], 12.0)
    np.testing.assert_array_equal(v["inertia"], np.broadcast_to(2.0 * np.eye(3), (2, 3, 3)))
    np.testing.assert_array_equal(v["cd"], 2.0)
    np.testing.assert_array_equal(v["area_drag"], 1.0)
    np.testing.assert_array_equal(v["magnetic_dipole"], [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    t = _cfg(sim.thruster_manager)
    np.testing.assert_array_equal(t["max_thrust"], 3.0)
    np.testing.assert_array_equal(t["positions"][:, :6], np.broadcast_to(DEFAULT_POSITIONS, (2, 6, 3)))
    np.testing.assert_array_equal(t["directions"][0, :6], DEFAULT_DIRECTIONS)
    np.testing.assert_array_equal(t["directions"][1, :6], [[0.0, 1.0, 0.0]] * 6)
    w = _cfg(sim.rw_manager)
    np.testing.assert_array_equal(w["max_torque"], 0.5)
    np.testing.assert_array_equal(w["wheel_inertia"], 0.01)
    np.testing.assert_array_equal(w["spin_axis_0"], [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_array_equal(w["spin_axis_1"], [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

"""
Domain randomization and batched per-entity vehicle configuration.

Checks:
  1. VehicleManager.set_config_per_entity writes per-entity values and computes
     the inverse inertia in-kernel.
  2. DomainRandomizer samples within bounds with the physical couplings:
     state mass = mass_dry + propellant, inertia = scale * nominal tensor.
  3. Parameters are resampled on every reset (full, partial and terminated,
     including inside CUDA graph capture).
  4. The YAML `randomization:` section (global + per_body) and its validation.
"""
import numpy as np
import pytest

from conftest import get_state
from cosmic_warp.core.config import create_simulator_from_config


def make_config(device, **overrides):
    """Minimal quiet config (no J2, no termination) for randomization tests."""
    config = {
        "simulation": {"num_envs": 64, "num_bodies": 2, "device": device},
        "physics": {"j2_perturbation": False},
        "termination": {"check_bounds": False},
    }
    config.update(overrides)
    return config


# ---------------------------------------------------------------------------
# Batched per-entity vehicle config (Monte Carlo setters)
# ---------------------------------------------------------------------------

def test_set_config_per_entity(make_sim):
    sim = make_sim(num_envs=8)
    n = sim.num_entities

    mass_dry = np.linspace(1.0, 8.0, n)
    inertia = np.zeros((n, 3, 3))
    for i in range(n):
        inertia[i] = np.diag([1.0 + i, 2.0 + i, 3.0 + i])

    sim.vehicle.set_config_per_entity(mass_dry=mass_dry, inertia=inertia)
    cfg = sim.vehicle.configs.numpy()

    np.testing.assert_array_equal(cfg["mass_dry"], mass_dry)
    np.testing.assert_array_equal(cfg["inertia"], inertia)
    # Inverse computed in-kernel: within an ulp of NumPy's
    np.testing.assert_allclose(cfg["inertia_inv"], np.linalg.inv(inertia), rtol=1e-15, atol=1e-16)
    # Untouched fields keep the constructor defaults
    np.testing.assert_array_equal(cfg["cd"], 2.2)


def test_set_config_per_entity_wrong_length(make_sim):
    sim = make_sim(num_envs=8)
    with pytest.raises(ValueError, match="mass_dry"):
        sim.vehicle.set_config_per_entity(mass_dry=np.ones(3))


# ---------------------------------------------------------------------------
# DomainRandomizer sampling and couplings
# ---------------------------------------------------------------------------

def test_randomizer_couplings(make_sim):
    sim = make_sim(num_envs=256, gravity=False)
    nominal = np.diag([10.0, 20.0, 30.0])
    sim.vehicle.set_config(mass_dry=70.0, inertia=nominal.tolist())

    sim.randomizer.set_bounds(
        mass_dry=(60.0, 80.0), propellant=(5.0, 15.0), inertia_scale=(0.8, 1.2)
    )
    sim.reset()

    cfg = sim.vehicle.configs.numpy()
    mass = get_state(sim)["mass"]

    dry = cfg["mass_dry"]
    assert (dry >= 60.0).all() and (dry <= 80.0).all()
    assert dry.std() > 0.0  # actually dispersed

    # Total mass = dry + propellant, never below dry
    propellant = mass - dry
    assert (propellant >= 5.0).all() and (propellant <= 15.0).all()

    # Inertia = scale * nominal: one scalar per entity, shape preserved
    scale = cfg["inertia"][:, 0, 0] / nominal[0, 0]
    assert (scale >= 0.8).all() and (scale <= 1.2).all()
    # (scale recovered by division: an ulp of rounding)
    np.testing.assert_allclose(cfg["inertia"], scale[:, None, None] * nominal, rtol=1e-15)
    # Inverse stays consistent
    np.testing.assert_allclose(
        np.einsum("nij,njk->nik", cfg["inertia"], cfg["inertia_inv"]),
        np.broadcast_to(np.eye(3), cfg["inertia"].shape), rtol=0, atol=1e-15,
    )


def test_randomizer_resamples_each_reset(make_sim):
    sim = make_sim(num_envs=64, gravity=False)
    sim.vehicle.set_config(mass_dry=70.0)
    sim.randomizer.set_bounds(propellant=(0.0, 50.0))

    sim.reset()
    first = get_state(sim)["mass"]
    sim.reset()
    second = get_state(sim)["mass"]
    assert not np.allclose(first, second)


def test_randomizer_partial_reset(make_sim):
    sim = make_sim(num_envs=8, gravity=False)
    sim.vehicle.set_config(mass_dry=70.0)
    sim.randomizer.set_bounds(propellant=(0.0, 50.0))

    sim.reset()
    before = get_state(sim)["mass"]
    sim.reset(env_ids=[0])
    after = get_state(sim)["mass"]

    assert after[0] != before[0]
    np.testing.assert_array_equal(after[1:], before[1:])


def test_randomizer_mass_dry_requires_propellant(make_sim):
    sim = make_sim(num_envs=4)
    with pytest.raises(ValueError, match="propellant"):
        sim.randomizer.set_bounds(mass_dry=(60.0, 80.0))
    with pytest.raises(ValueError, match="min"):
        sim.randomizer.set_bounds(propellant=(10.0, 5.0))
    with pytest.raises(ValueError, match="positive"):
        sim.randomizer.set_bounds(inertia_scale=(0.0, 1.2))


def test_randomizer_inside_graph_capture(make_sim):
    """auto_reset_terminated resamples parameters inside the captured graph."""
    # Default reset bounds put entities near LEO radius, so a tight pos_limit
    # terminates every environment each step and forces auto resets
    sim = make_sim(num_envs=32, check_bounds=True, pos_limit=1.0)
    sim.vehicle.set_config(mass_dry=70.0)
    sim.randomizer.set_bounds(propellant=(5.0, 15.0), inertia_scale=(0.9, 1.1))

    sim.reset()
    first = get_state(sim)["mass"]
    sim.simulate(5, auto_reset_terminated=True)
    mass = get_state(sim)["mass"]

    assert not np.allclose(mass, first)  # resampled during the run
    assert (mass >= 75.0).all() and (mass <= 85.0).all()


# ---------------------------------------------------------------------------
# YAML spec
# ---------------------------------------------------------------------------

def test_yaml_randomization_per_body(device):
    config = make_config(
        device,
        body={
            "defaults": {"vehicle": {"mass_dry": 70.0}},
            "per_body": [{"body_id": 1, "vehicle": {"mass_dry": 80.0}}],
        },
        randomization={
            "propellant": [10.0, 20.0],
            "per_body": [{"body_id": 1, "propellant": [30.0, 40.0]}],
        },
    )
    sim = create_simulator_from_config(config)
    sim.reset()
    mass = get_state(sim)["mass"].reshape(64, 2)

    # Body 0: global bounds on top of dry 70; body 1: override on top of dry 80
    assert (mass[:, 0] >= 80.0).all() and (mass[:, 0] <= 90.0).all()
    assert (mass[:, 1] >= 110.0).all() and (mass[:, 1] <= 120.0).all()


def test_yaml_randomization_inertia_uses_body_config(device):
    nominal = [[10.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 30.0]]
    config = make_config(
        device,
        body={"defaults": {"vehicle": {"mass_dry": 70.0, "inertia": nominal}}},
        reset={"mass_bounds": [80.0, 120.0]},
        randomization={"inertia_scale": [0.5, 0.6]},
    )
    sim = create_simulator_from_config(config)
    sim.reset()

    scale = sim.vehicle.configs.numpy()["inertia"][:, 0, 0] / 10.0
    assert (scale >= 0.5).all() and (scale <= 0.6).all()
    np.testing.assert_allclose(sim.vehicle.configs.numpy()["inertia"],
                               scale[:, None, None] * np.asarray(nominal), rtol=1e-15)


@pytest.mark.parametrize("bounds", [("5", 10.0), (True, 10.0), (float("nan"), 1.0), (1.0,)])
def test_randomization_bounds_must_be_numbers(make_sim, bounds):
    sim = make_sim()
    with pytest.raises(ValueError, match="propellant"):
        sim.randomizer.set_bounds(propellant=bounds)


@pytest.mark.parametrize("seed", ["42", 4.7, True, -1, 2**64])
def test_randomization_seed_rejected(make_sim, seed):
    sim = make_sim()
    with pytest.raises(ValueError, match="randomization seed"):
        sim.randomizer.reseed(seed)


def test_yaml_quoted_randomization_bounds_rejected(device):
    with pytest.raises(ValueError, match="propellant"):
        create_simulator_from_config(
            make_config(device, randomization={"propellant": ["5", "10"]}))


def test_yaml_randomization_validation(device):
    with pytest.raises(ValueError, match="propelant"):
        create_simulator_from_config(
            make_config(device, randomization={"propelant": [1.0, 2.0]})
        )

    with pytest.raises(ValueError, match="propellant"):
        create_simulator_from_config(
            make_config(device, randomization={"mass_dry": [60.0, 80.0]})
        )

    # Prescribed initial mass must stay above the largest sampled dry mass
    config = make_config(
        device,
        body={"defaults": {"vehicle": {"mass_dry": 70.0}}},
        randomization={"mass_dry": [60.0, 80.0], "propellant": [5.0, 10.0]},
        initial_conditions={
            "per_body": [
                {"body_id": 0, "mass": 75.0},
                {"body_id": 1, "mass": 100.0},
            ]
        },
    )
    with pytest.raises(ValueError, match="mass_dry"):
        create_simulator_from_config(config)


def test_inertia_scaling_keeps_per_entity_and_later_inertia(make_sim):
    """Inertia scaling applies to each entity's own configured tensor. It used
    to capture env 0's tensor once at set_bounds and write it to every entity,
    overwriting per-env inertia and any later vehicle.set_config."""
    sim = make_sim(num_envs=2, gravity=False)
    sim.vehicle.set_config_per_entity(inertia=np.array([np.eye(3), 50.0 * np.eye(3)]))
    sim.randomizer.set_bounds(inertia_scale=(1.0, 1.0))
    sim.reset()
    I = sim.vehicle.configs.numpy()["inertia"]
    np.testing.assert_array_equal(I[:, 0, 0], [1.0, 50.0])

    sim.vehicle.set_config(inertia=np.diag([3.0, 4.0, 5.0]))
    sim.reset()
    v = sim.vehicle.configs.numpy()
    np.testing.assert_array_equal(v["inertia"][1], np.diag([3.0, 4.0, 5.0]))
    np.testing.assert_allclose(v["inertia_inv"][1], np.diag([1 / 3, 1 / 4, 1 / 5]), rtol=1e-15)


def test_get_bounds_reports_what_was_set(make_sim):
    sim = make_sim(num_envs=2, num_bodies=2)
    assert sim.randomizer.get_bounds(1) == {}
    sim.randomizer.set_bounds(propellant=(1.0, 2.0), inertia_scale=(0.9, 1.1), body_ids=[1])
    assert sim.randomizer.get_bounds(0) == {}
    assert sim.randomizer.get_bounds(1) == {"propellant": (1.0, 2.0), "inertia_scale": (0.9, 1.1)}
    sim.randomizer.get_bounds(1)["propellant"] = (5.0, 6.0)      # a copy
    assert sim.randomizer.get_bounds(1)["propellant"] == (1.0, 2.0)
    with pytest.raises(ValueError, match="body_id"):
        sim.randomizer.get_bounds(2)


def test_reset_terminated_uses_the_latest_inertia(make_sim):
    """A set_config(inertia=...) after the last sim.reset() is the nominal
    reset_terminated() scales. It used to keep the nominal of the last
    reset/set_bounds call, here the default 1 kg cube's."""
    # A tight pos_limit terminates every environment on its first step
    sim = make_sim(num_envs=4, check_bounds=True, pos_limit=1.0)
    sim.randomizer.set_bounds(inertia_scale=(1.0, 1.0))   # scale 1: inertia = nominal
    sim.reset()
    sim.vehicle.set_config(inertia=np.diag([2.0, 3.0, 4.0]))
    sim.step()
    assert sim.get_terminated().numpy()["terminated"].all()

    sim.reset_terminated()
    v = sim.vehicle.configs.numpy()
    np.testing.assert_array_equal(v["inertia"], np.broadcast_to(np.diag([2.0, 3.0, 4.0]), (4, 3, 3)))
    np.testing.assert_allclose(v["inertia_inv"][0], np.diag([1 / 2, 1 / 3, 1 / 4]), rtol=1e-15)

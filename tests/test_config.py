"""
YAML/dict configuration loader validation and per-body features.

Checks:
  1. Unknown keys are rejected (typos fail loudly).
  2. Mass cross-checks: reset mass_bounds and initial masses must respect
     each body's mass_dry.
  3. Per-body reset bounds: each body samples from its own box.
  4. Per-body initial conditions broadcast across environments, with
     velocity -> momentum conversion.
  5. Orbital-element ICs (YAML orbit spec and the Python helpers) produce
     physically consistent Cartesian states.
"""
import math

import numpy as np
import pytest

from conftest import get_state
from cosmic_warp.core.pipeline import SpacecraftSimulator
from cosmic_warp.core.config import create_simulator_from_config, create_simulator_from_yaml
from cosmic_warp.physics.gravity import MU_EARTH, RE_EARTH
from cosmic_warp.physics.orbits import elements_to_state, circular_orbit_state


def make_config(device, **overrides):
    """Minimal quiet config (no J2, no termination) for loader tests."""
    config = {
        "simulation": {"num_envs": 4, "num_bodies": 2, "device": device},
        "physics": {"j2_perturbation": False},
        "termination": {"check_bounds": False},
    }
    config.update(overrides)
    return config


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_unknown_top_level_key_rejected(device):
    with pytest.raises(ValueError, match="physcs"):
        create_simulator_from_config(make_config(device, physcs={"gravity": True}))


def test_unknown_nested_key_rejected(device):
    config = make_config(device)
    config["termination"]["pos_limitt"] = 1.0e8
    with pytest.raises(ValueError, match="pos_limitt"):
        create_simulator_from_config(config)

    config = make_config(device, body={"defaults": {"vehicle": {"dry_mass": 10.0}}})
    with pytest.raises(ValueError, match="dry_mass"):
        create_simulator_from_config(config)


def test_reset_mass_below_dry_mass_rejected(device):
    config = make_config(
        device,
        body={"defaults": {"vehicle": {"mass_dry": 70.0}}},
        reset={"mass_bounds": [50.0, 100.0]},
    )
    with pytest.raises(ValueError, match="mass_dry"):
        create_simulator_from_config(config)

    # A per-body override above the dry mass fixes body 1 but not body 0
    config["reset"]["per_body"] = [{"body_id": 1, "mass_bounds": [80.0, 100.0]}]
    with pytest.raises(ValueError, match="body 0"):
        create_simulator_from_config(config)


def test_ic_mass_below_dry_mass_rejected(device):
    config = make_config(
        device,
        body={"defaults": {"vehicle": {"mass_dry": 70.0}}},
        initial_conditions={
            "per_body": [
                {"body_id": 0, "mass": 100.0},
                {"body_id": 1, "mass": 50.0},
            ]
        },
    )
    with pytest.raises(ValueError, match="mass_dry"):
        create_simulator_from_config(config)


def test_mixed_ic_forms_rejected(device):
    config = make_config(
        device,
        initial_conditions={
            "per_body": [{"body_id": 0, "mass": 1.0}, {"body_id": 1, "mass": 1.0}],
            "masses": [1.0] * 8,
        },
    )
    with pytest.raises(ValueError, match="not both"):
        create_simulator_from_config(config)


def test_velocity_requires_mass(device):
    config = make_config(
        device,
        initial_conditions={
            "per_body": [
                {"body_id": 0, "velocity": [0.0, 1.0, 0.0]},
                {"body_id": 1, "velocity": [0.0, 2.0, 0.0]},
            ]
        },
    )
    with pytest.raises(ValueError, match="mass"):
        create_simulator_from_config(config)


# ---------------------------------------------------------------------------
# Per-body reset bounds
# ---------------------------------------------------------------------------

def test_per_body_reset_bounds(device):
    config = make_config(device)
    config["simulation"]["num_envs"] = 256
    config["physics"]["gravity"] = False
    config["reset"] = {
        "pos_bounds": [[100.0, -1.0, -1.0], [200.0, 1.0, 1.0]],
        "mass_bounds": [1.0, 2.0],
        "per_body": [
            {"body_id": 1, "pos_bounds": [[300.0, -1.0, -1.0], [400.0, 1.0, 1.0]]},
        ],
    }
    sim = create_simulator_from_config(config)
    sim.reset()
    pos = get_state(sim)["pos"].reshape(256, 2, 3)

    # Body 0 keeps the global box, body 1 samples from its own
    assert (pos[:, 0, 0] >= 100.0).all() and (pos[:, 0, 0] <= 200.0).all()
    assert (pos[:, 1, 0] >= 300.0).all() and (pos[:, 1, 0] <= 400.0).all()


def test_loader_initializes_states_without_initial_conditions(device):
    """
    A loaded simulator must never sit at the zero state (gravity is singular
    at the origin): with no initial_conditions section, states are drawn
    from the reset bounds at load time.
    """
    config = make_config(device)
    sim = create_simulator_from_config(config)
    pos = get_state(sim)["pos"]
    assert (np.linalg.norm(pos, axis=1) > 1.0e6).all()


def test_yaml_device_override(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("simulation: {num_envs: 2, device: cuda}\n"
                    "termination: {check_bounds: false}\n")
    sim = create_simulator_from_yaml(str(path), device="cpu")
    assert str(sim.states.device) == "cpu"


# ---------------------------------------------------------------------------
# Per-body initial conditions
# ---------------------------------------------------------------------------

def test_per_body_ic_broadcast(device):
    config = make_config(device)
    config["initial_conditions"] = {
        "per_body": [
            {"body_id": 0, "position": [1.0, 2.0, 3.0],
             "velocity": [0.0, 10.0, 0.0], "mass": 2.0},
            {"body_id": 1, "position": [4.0, 5.0, 6.0],
             "velocity": [0.0, -10.0, 0.0], "mass": 4.0},
        ]
    }
    sim = create_simulator_from_config(config)
    s = get_state(sim)

    pos = s["pos"].reshape(4, 2, 3)
    mom = s["mom"].reshape(4, 2, 3)
    mass = s["mass"].reshape(4, 2)

    # Same conditions in every environment
    assert (pos[:, 0] == [1.0, 2.0, 3.0]).all() and (pos[:, 1] == [4.0, 5.0, 6.0]).all()
    assert (mass[:, 0] == 2.0).all() and (mass[:, 1] == 4.0).all()
    # momentum = mass * velocity
    assert (mom[:, 0] == [0.0, 20.0, 0.0]).all() and (mom[:, 1] == [0.0, -40.0, 0.0]).all()


@pytest.mark.parametrize("motion", ["velocities", "momenta"])
def test_per_entity_ic_lists(device, motion):
    """initial_conditions as one row per entity (environment-major), with
    momentum = mass * velocity; quaternions normalized on load."""
    n = 8
    rng = np.random.default_rng(1)
    pos = (7.0e6 + rng.normal(size=(n, 3))).tolist()
    vel = rng.normal(size=(n, 3)).tolist()
    masses = (2.0 + rng.random(n)).tolist()
    ang_mom = rng.normal(size=(n, 3)).tolist()
    quat = [[0.0, 0.0, 2.0, 0.0]] * n
    ic = {"positions": pos, "masses": masses, "angular_momenta": ang_mom, "quaternions": quat,
          motion: vel}
    s = get_state(create_simulator_from_config(make_config(device, initial_conditions=ic)))
    np.testing.assert_array_equal(s["pos"], pos)
    np.testing.assert_array_equal(s["mass"], masses)
    np.testing.assert_array_equal(s["ang_mom"], ang_mom)
    np.testing.assert_array_equal(s["quat"], [[0.0, 0.0, 1.0, 0.0]] * n)
    expected = np.asarray(masses)[:, None] * vel if motion == "velocities" else vel
    np.testing.assert_array_equal(s["mom"], expected)


@pytest.mark.parametrize("initial_conditions, match", [
    ({"velocities": [[0.0, 1.0, 0.0]] * 8}, "'velocities' requires 'masses'"),
    ({"velocities": [[0.0, 1.0, 0.0]] * 8, "momenta": [[0.0, 1.0, 0.0]] * 8,
      "masses": [1.0] * 8}, "'velocities' or 'momenta', not both"),
    ({"positions": [[7.0e6, 0.0, 0.0]] * 7}, "positions must have length 8"),
])
def test_per_entity_ic_lists_rejected(device, initial_conditions, match):
    with pytest.raises(ValueError, match=match):
        create_simulator_from_config(make_config(device, initial_conditions=initial_conditions))


@pytest.mark.parametrize("overrides, match", [
    ({"physics": [True]}, "physics must be a mapping"),
    ({"reset": {"orbit": 7.0e6}}, "reset.orbit must be a mapping"),
    ({"reset": {"per_body": {"body_id": 0}}}, "reset.per_body must be a list of mappings"),
    ({"body": {"per_body": [0]}}, "body.per_body must be a list of mappings"),
    ({"initial_conditions": {"masses": 2.0}}, "initial_conditions.masses must be a list"),
    ({"randomization": {"propellant": 1.0}}, "randomization propellant: expected"),
    ({"simulation": {"num_bodies": "2"}}, "simulation.num_bodies must be a positive integer"),
])
def test_malformed_sections_name_the_key(device, overrides, match):
    """Malformed structure raises ValueError naming the key; it used to
    escape as TypeError or AttributeError from deep inside the loader."""
    config = make_config(device)
    for section, value in overrides.items():
        config[section] = {**config.get(section, {}), **value} if isinstance(value, dict) else value
    with pytest.raises(ValueError, match=match):
        create_simulator_from_config(config)


def test_orbit_ic_gives_circular_orbit(device):
    sma = 7.0e6
    inclination = math.radians(51.6)
    config = make_config(device)
    config["simulation"]["num_bodies"] = 1
    config["initial_conditions"] = {
        "per_body": [
            {"body_id": 0, "mass": 10.0,
             "orbit": {"semi_major_axis": sma, "inclination": inclination}},
        ]
    }
    sim = create_simulator_from_config(config)
    s = get_state(sim)

    r = np.linalg.norm(s["pos"], axis=1)
    v = np.linalg.norm(s["mom"] / s["mass"][:, None], axis=1)
    np.testing.assert_array_equal(r, sma)
    np.testing.assert_array_equal(v, math.sqrt(MU_EARTH / sma))

    # Orbit plane inclination: angle between h = r x v and +Z
    h = np.cross(s["pos"][0], s["mom"][0] / s["mass"][0])
    inc = math.acos(h[2] / np.linalg.norm(h))
    assert abs(inc - inclination) < 1e-12


def test_orbit_ic_mean_anomaly(device):
    """An initial-condition orbit given by mean anomaly lands where Kepler's
    equation puts it (as reset.orbit does); giving both anomalies raises."""
    sma, ecc, mean_anomaly = 8.0e6, 0.3, 1.0
    config = make_config(device)
    config["simulation"]["num_bodies"] = 1
    config["initial_conditions"] = {
        "per_body": [
            {"body_id": 0, "mass": 10.0,
             "orbit": {"semi_major_axis": sma, "eccentricity": ecc,
                       "mean_anomaly": mean_anomaly}},
        ]
    }
    s = get_state(create_simulator_from_config(config))
    pos, vel = s["pos"][0], s["mom"][0] / s["mass"][0]

    # Recover M from the state: true anomaly from the conic equation (outbound
    # since r.v > 0), then eccentric and mean anomaly
    r = np.linalg.norm(pos)
    assert np.dot(pos, vel) > 0.0
    nu = math.acos((sma * (1.0 - ecc**2) / r - 1.0) / ecc)
    e_anom = 2.0 * math.atan(math.sqrt((1.0 - ecc) / (1.0 + ecc)) * math.tan(nu / 2.0))
    assert abs(e_anom - ecc * math.sin(e_anom) - mean_anomaly) < 1e-9

    config["initial_conditions"]["per_body"][0]["orbit"]["true_anomaly"] = 0.5
    with pytest.raises(ValueError, match="at most one of true_anomaly or mean_anomaly"):
        create_simulator_from_config(config)


# ---------------------------------------------------------------------------
# Orbital-element helpers (host-side, no simulator)
# ---------------------------------------------------------------------------

def test_elements_to_state_invariants():
    sma, ecc = 8.0e6, 0.2
    inc, raan, argp, nu = map(math.radians, (35.0, 40.0, 25.0, 120.0))
    pos, vel = elements_to_state(sma, ecc, inc, raan, argp, nu)

    # Radius from the conic equation
    p = sma * (1.0 - ecc**2)
    r_expected = p / (1.0 + ecc * math.cos(nu))
    assert abs(np.linalg.norm(pos) - r_expected) < 1e-6

    # Vis-viva
    v_expected = math.sqrt(MU_EARTH * (2.0 / r_expected - 1.0 / sma))
    assert abs(np.linalg.norm(vel) - v_expected) < 1e-9

    # Specific angular momentum: magnitude sqrt(mu*p), tilted by inc from +Z
    h = np.cross(pos, vel)
    assert abs(np.linalg.norm(h) - math.sqrt(MU_EARTH * p)) < 1e-3
    assert abs(math.acos(h[2] / np.linalg.norm(h)) - inc) < 1e-12


def test_elements_to_state_orientation_conventions():
    """
    Pin the sign conventions of the perifocal -> ECI rotation. The invariant
    checks (radius, vis-viva, |h|, inclination) all pass under a RAAN or
    true-anomaly sign flip, so the node direction and motion direction must
    be asserted explicitly.
    """
    sma = 7.0e6
    raan = math.radians(40.0)
    inc = math.radians(60.0)

    # At argp = 0, nu = 0 the satellite sits on the ascending node line
    # [cos(raan), sin(raan), 0] and is heading north (ascending: v_z > 0)
    pos, vel = elements_to_state(sma, 0.0, inc, raan, 0.0, 0.0)
    node = np.array([math.cos(raan), math.sin(raan), 0.0])
    assert np.allclose(pos / np.linalg.norm(pos), node, atol=1e-12)
    assert vel[2] > 0.0

    # Argument of periapsis measured from the node along the motion: for a
    # polar orbit with argp = 90 deg, periapsis (nu = 0) is over the north pole
    pos, _ = elements_to_state(sma, 0.1, math.radians(90.0), 0.0,
                               math.radians(90.0), 0.0)
    assert np.allclose(pos / np.linalg.norm(pos), [0.0, 0.0, 1.0], atol=1e-12)

    # True anomaly advances along the direction of motion
    pos0, vel0 = elements_to_state(sma, 0.0, inc, raan, 0.0, 0.0)
    pos1, _ = elements_to_state(sma, 0.0, inc, raan, 0.0, 1e-4)
    assert np.dot(pos1 - pos0, vel0) > 0.0


def test_circular_orbit_state_periapsis():
    pos, vel = circular_orbit_state(altitude=400.0e3)
    r = np.linalg.norm(pos)
    assert abs(np.dot(pos, vel)) < 1e-6                      # circular: r . v = 0
    assert abs(np.linalg.norm(vel) - math.sqrt(MU_EARTH / r)) < 1e-9


def test_orbit_spec_rejects_bad_input(device):
    config = make_config(device)
    config["simulation"]["num_bodies"] = 1
    config["initial_conditions"] = {
        "per_body": [
            {"body_id": 0, "mass": 1.0,
             "orbit": {"semi_major_axis": 7.0e6, "altitude": 400.0e3}},
        ]
    }
    with pytest.raises(ValueError, match="exactly one"):
        create_simulator_from_config(config)

    with pytest.raises(ValueError, match="eccentricity"):
        elements_to_state(7.0e6, eccentricity=1.2)


@pytest.mark.parametrize("inertia, match", [
    ([[1, 5, 0], [0, 1, 0], [0, 0, 1]], "symmetric"),
    ([[1, 0, 0], [0, 1, 0], [0, 0, -1]], "positive definite"),
    ([[1, 0, 0], [0, 1, 0], [0, 0, 5]], "triangle inequality"),
    ([[1, 0, 0], [0, float("nan"), 0], [0, 0, 1]], "finite"),
])
def test_unphysical_inertia_rejected(device, inertia, match):
    """Python API: inertia must be a physical rigid-body tensor, in both the
    broadcast and the batched setter."""
    sim = SpacecraftSimulator(num_envs=2, device=device)
    with pytest.raises(ValueError, match=match):
        sim.vehicle.set_config(inertia=inertia)
    with pytest.raises(ValueError, match=match):
        sim.vehicle.set_config_per_entity(inertia=np.array([np.eye(3), inertia]))


@pytest.mark.parametrize("kwargs, match", [
    ({"mass_dry": float("nan")}, "mass_dry must be finite"),
    ({"mass_dry": 0.0}, "mass_dry must be > 0"),
    ({"cd": -1.0}, "cd must be >= 0"),
    ({"area_drag": -1.0}, "area_drag must be >= 0"),
])
def test_invalid_vehicle_scalars_rejected(device, kwargs, match):
    sim = SpacecraftSimulator(num_envs=2, device=device)
    with pytest.raises(ValueError, match=match):
        sim.vehicle.set_config(**kwargs)
    batched = {k: np.full(2, v) for k, v in kwargs.items()}
    with pytest.raises(ValueError, match=match.split(" must")[0]):
        sim.vehicle.set_config_per_entity(**batched)


def test_default_mass_bounds_below_dry_rejected(device):
    """A heavy vehicle with reset mass_bounds left at the 1 kg default is
    rejected even though mass_bounds never appears in the YAML."""
    config = make_config(device, body={"defaults": {"vehicle": {"mass_dry": 100.0}}})
    with pytest.raises(ValueError, match="mass_dry"):
        create_simulator_from_config(config)


@pytest.mark.parametrize("section", ["reset", "initial_conditions"])
def test_orbit_specs_take_no_mu(device, section):
    """Orbits are about Earth, the only central body the dynamics know: an
    orbit spec's mu is an unknown key."""
    orbit = {"altitude": 4.0e5, "mu": 3.986004415e14}
    spec = ({"orbit": orbit} if section == "reset"
            else {"per_body": [{"body_id": b, "orbit": orbit} for b in range(2)]})
    with pytest.raises(ValueError, match=r"Unknown key\(s\) in .*orbit: \['mu'\]"):
        create_simulator_from_config(make_config(device, **{section: spec}))


def test_omitted_initial_conditions_keep_drawn_state(device):
    """Fields left out of initial_conditions keep the randomized draw. They
    used to fall back to the reset-bounds center, which put a 70 kg vehicle
    with randomized propellant at 1 kg (below dry mass) without any error."""
    pos = [7.0e6, 0.0, 0.0]
    config = make_config(
        device,
        body={"defaults": {"vehicle": {"mass_dry": 70.0}}},
        randomization={"propellant": [5.0, 10.0]},
        initial_conditions={"per_body": [{"body_id": b, "position": pos} for b in range(2)]},
    )
    sim = create_simulator_from_config(config)
    s = get_state(sim)
    np.testing.assert_array_equal(s["pos"], np.tile(pos, (8, 1)))
    assert np.all((s["mass"] >= 75.0) & (s["mass"] <= 80.0))

    # A prescribed mass keeps the drawn velocity (momentum is rescaled)
    config["initial_conditions"] = {"per_body": [
        {"body_id": b, "position": pos, "mass": 90.0} for b in range(2)]}
    sim = create_simulator_from_config(config)
    v = np.linalg.norm(get_state(sim)["mom"], axis=1) / 90.0
    v_circ = np.sqrt(MU_EARTH / 6.778137e6)
    np.testing.assert_allclose(v, v_circ, atol=2.0)


def test_orbit_mode_initial_conditions_need_position_and_momentum(device):
    """With reset.orbit sampling, a prescribed position used to be combined
    with a velocity drawn elsewhere on the orbit, giving an eccentric orbit
    instead of the configured circular one."""
    pos = [0.0, RE_EARTH + 4.0e5, 0.0]
    config = make_config(device, reset={"orbit": {"altitude": [4.0e5, 4.0e5]}},
                         initial_conditions={"per_body": [
                             {"body_id": b, "position": pos} for b in range(2)]})
    with pytest.raises(ValueError, match="positions and momenta/velocities must be given together"):
        create_simulator_from_config(config)

    # Both halves omitted: the drawn state is kept, and it is circular
    config["initial_conditions"] = {"per_body": [
        {"body_id": b, "angular_momentum": [0.0, 0.0, 0.0]} for b in range(2)]}
    s = get_state(create_simulator_from_config(config))
    r = np.linalg.norm(s["pos"], axis=1)
    v = np.linalg.norm(s["mom"] / s["mass"][:, None], axis=1)
    np.testing.assert_allclose(r, RE_EARTH + 4.0e5, rtol=1e-12)
    np.testing.assert_allclose(v, np.sqrt(MU_EARTH / r), rtol=1e-12)


def test_yaml_floats_follow_yaml_1_2(tmp_path, device):
    """PyYAML (YAML 1.1) read 1.0e1 and 1e5 as strings: mass_dry: 1.0e1 used
    to crash with a TypeError deep in the loader."""
    path = tmp_path / "c.yaml"
    path.write_text(
        f"simulation: {{num_envs: 2, device: {device}, dt: 1e-2}}\n"
        "physics: {gravity: false}\n"
        "termination: {check_bounds: false}\n"
        "body: {defaults: {vehicle: {mass_dry: 1.0e1}}}\n"
        "reset: {mass_bounds: [1.0e1, 2e1]}\n"
    )
    sim = create_simulator_from_yaml(str(path))
    assert sim.dt == 0.01
    np.testing.assert_array_equal(sim.vehicle.configs.numpy()["mass_dry"], 10.0)


@pytest.mark.parametrize("initial_conditions, match", [
    ({"masses": ["5.0"] * 8}, "must be a number"),
    ({"positions": [["7.0e6", "0", "0"]] * 8}, "must be a number"),
    ({"per_body": [{"body_id": b, "position": ["7.0e6", 0.0, 0.0]} for b in range(2)]},
     "must be a number"),
    ({"per_body": [{"body_id": b, "orbit": {"altitude": "4.0e5"}} for b in range(2)]},
     "must be a number"),
    ({"per_body": [{"body_id": b, "orbit": {"altitude": 4.0e5, "raan": "1.0"}}
                   for b in range(2)]}, "must be a number"),
    # Malformed rows used to be reshaped: 16 rows of 2 components silently
    # became the 8 expected quaternions
    ({"quaternions": [[0.0, 1.0]] * 16}, "4-component rows"),
    ({"positions": [[7.0e6, 0.0]] * 8}, "3-component rows"),
])
def test_initial_conditions_reject_quoted_numbers_and_bad_rows(device, initial_conditions, match):
    """initial_conditions and orbit specs converted quoted numbers with float()
    or np.asarray, although the loader rejects quoted numbers elsewhere."""
    config = make_config(device, initial_conditions=initial_conditions)
    with pytest.raises(ValueError, match=match):
        create_simulator_from_config(config)


def test_reset_orbit_and_attitude_reject_quoted_numbers(device):
    for reset in ({"orbit": {"altitude": [4.0e5, 4.0e5], "raan": "1.0"}},
                  {"att_max_angle": "0.5"}):
        with pytest.raises(ValueError, match="must be a number"):
            create_simulator_from_config(make_config(device, reset=reset))


@pytest.mark.parametrize("section, key", [("physics", "gravity"), ("termination", "check_bounds")])
def test_quoted_boolean_rejected(device, section, key):
    """A quoted "false" is a non-empty string and used to enable the flag."""
    config = make_config(device)
    config.setdefault(section, {})[key] = "false"
    with pytest.raises(ValueError, match=f"{section}.{key} must be true or false"):
        create_simulator_from_config(config)


@pytest.mark.parametrize("section", ["body", "reset", "randomization"])
def test_duplicate_per_body_entries_rejected(device, section):
    """Duplicate body_id entries used to be silently last-wins."""
    entry = {"body": {"body_id": 0, "vehicle": {"mass_dry": 5.0}},
             "reset": {"body_id": 0, "mass_bounds": [5.0, 5.0]},
             "randomization": {"body_id": 0, "propellant": [0.0, 1.0]}}[section]
    config = make_config(device, **{section: {"per_body": [entry, dict(entry)]}})
    with pytest.raises(ValueError, match=f"{section}.per_body: duplicate body_id 0"):
        create_simulator_from_config(config)


@pytest.mark.parametrize("value, match", [
    (["40", 50.0], "mass_bounds must be a number"),
    (5.0, r"mass_bounds must be \(min, max\)"),
])
def test_malformed_bounds_rejected(device, value, match):
    config = make_config(device, reset={"mass_bounds": value})
    with pytest.raises(ValueError, match=match):
        create_simulator_from_config(config)


def test_initial_quaternions_normalized_and_zero_rejected(device):
    entries = [{"body_id": b, "position": [7.0e6, 0.0, 0.0], "quaternion": [0.0, 0.0, 0.0, 2.0]}
               for b in range(2)]
    sim = create_simulator_from_config(make_config(device, initial_conditions={"per_body": entries}))
    np.testing.assert_array_equal(get_state(sim)["quat"], np.tile([0.0, 0.0, 0.0, 1.0], (8, 1)))

    entries[0]["quaternion"] = [0.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="nonzero"):
        create_simulator_from_config(make_config(device, initial_conditions={"per_body": entries}))

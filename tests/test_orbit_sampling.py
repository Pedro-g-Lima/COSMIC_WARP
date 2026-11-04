"""
Orbital-element reset sampling.

Checks:
  1. state_to_elements inverts elements_to_state (host round-trip).
  2. Degenerate orbit bounds reproduce the host conversion exactly.
  3. Sampled states are dynamically consistent (vis-viva, mom = mass * v).
  4. Recovered elements are uniform within the configured bounds.
  5. Mean-anomaly mode is time-uniform (Kepler solve inverted correctly).
  6. Per-body mode mixing, determinism, CUDA-graph capture, YAML loading,
     and configuration-time validation.
"""
import math

import numpy as np
import pytest

from conftest import get_state
from cosmic_warp.core.config import create_simulator_from_config
from cosmic_warp.physics.gravity import MU_EARTH, RE_EARTH
from cosmic_warp.physics.orbits import elements_to_state, state_to_elements

ELEMS = dict(semi_major_axis=7.0e6, eccentricity=0.15, inclination=0.9,
             raan=1.2, arg_periapsis=2.1, true_anomaly=0.7)


def recovered_elements(state):
    """Per-entity classical elements recovered from a sampled state dict."""
    vel = state["mom"] / state["mass"][:, None]
    return [state_to_elements(state["pos"][i], vel[i])
            for i in range(len(state["mass"]))]


@pytest.mark.parametrize("pos, vel, match", [
    ([7.0e6, 0.0, 0.0], [1000.0, 0.0, 0.0], "zero angular momentum"),   # radial
    ([0.0, 0.0, 0.0], [0.0, 7500.0, 0.0], "zero angular momentum"),     # at the centre
    ([7.0e6, 0.0, 0.0], [0.0, 12000.0, 0.0], "unbound orbit"),          # hyperbolic
])
def test_state_to_elements_rejects_undefined_elements(pos, vel, match):
    """These used to return NaN or a negative semi-major axis silently."""
    with pytest.raises(ValueError, match=match):
        state_to_elements(pos, vel)


def test_state_to_elements_roundtrip():
    pos, vel = elements_to_state(**ELEMS)
    rec = state_to_elements(pos, vel)
    for key, value in ELEMS.items():
        assert abs(rec[key] - value) < 1e-9 * max(1.0, abs(value)), key


@pytest.mark.parametrize("inclination", [1e-8, math.pi - 1e-8])
def test_state_to_elements_near_equatorial_inclination(inclination):
    """acos(h_z / |h|) lost half the digits near i = 0 and pi: i = 1e-8 came
    back as 1.49e-8. atan2(hypot(h_x, h_y), h_z) is accurate to rounding."""
    pos, vel = elements_to_state(**{**ELEMS, "inclination": inclination})
    rec = state_to_elements(pos, vel)
    assert abs(rec["inclination"] - inclination) < 1e-15


@pytest.mark.parametrize("inclination", [-1e-3, math.pi + 1e-3, math.nan])
def test_elements_to_state_rejects_inclination_outside_range(inclination):
    with pytest.raises(ValueError, match="inclination must"):
        elements_to_state(**{**ELEMS, "inclination": inclination})


def test_fixed_elements_match_host_conversion(make_sim):
    """Degenerate bounds must reproduce elements_to_state on every entity."""
    sim = make_sim(num_envs=16, gravity=False)
    sim.reset_manager.set_orbit_bounds(
        semi_major_axis_bounds=(7.0e6, 7.0e6), eccentricity_bounds=(0.15, 0.15), inclination_bounds=(0.9, 0.9),
        raan_bounds=(1.2, 1.2), arg_periapsis_bounds=(2.1, 2.1),
        true_anomaly_bounds=(0.7, 0.7))
    sim.reset()
    s = get_state(sim)

    pos_ref, vel_ref = elements_to_state(**ELEMS)
    assert np.allclose(s["pos"], pos_ref, rtol=1e-12, atol=1e-5)
    assert np.allclose(s["mom"], vel_ref * s["mass"][:, None], rtol=1e-12, atol=1e-8)


def test_sampled_states_dynamically_consistent(make_sim):
    """Circular band: radius in bounds, vis-viva speed, radial velocity zero."""
    sim = make_sim(num_envs=4096, gravity=False,
                   reset_config={"mass_bounds": (2.0, 4.0)})
    sim.reset_manager.set_orbit_bounds(
        semi_major_axis_bounds=(6.8e6, 7.2e6), eccentricity_bounds=(0.0, 0.0),
        inclination_bounds=(0.0, math.pi), raan_bounds=(0.0, 2.0 * math.pi))
    sim.reset()
    s = get_state(sim)

    r = np.linalg.norm(s["pos"], axis=1)
    assert (r >= 6.8e6 - 1.0).all() and (r <= 7.2e6 + 1.0).all()

    # Momentum is mass * velocity with the entity's own sampled mass
    assert (s["mass"] >= 2.0).all() and (s["mass"] <= 4.0).all()
    vel = s["mom"] / s["mass"][:, None]
    speed = np.linalg.norm(vel, axis=1)
    assert np.abs(speed - np.sqrt(MU_EARTH / r)).max() < 1e-6 * speed.mean()
    radial = np.abs(np.einsum("ij,ij->i", s["pos"], vel))
    assert radial.max() < 1e-6 * (r * speed).mean()


def test_elements_uniform_within_bounds(make_sim):
    """Recovered elements stay inside bounds and are uniformly distributed."""
    bounds = {
        "semi_major_axis": (6.8e6, 7.3e6),
        "eccentricity": (0.01, 0.1),
        "inclination": (0.3, 1.2),
        "raan": (0.5, 2.5),
        "arg_periapsis": (1.0, 3.0),
        "true_anomaly": (0.0, 2.0 * math.pi),
    }
    sim = make_sim(num_envs=4096, gravity=False, reset_config={"seed": 11})
    sim.reset_manager.set_orbit_bounds(
        semi_major_axis_bounds=bounds["semi_major_axis"], eccentricity_bounds=bounds["eccentricity"],
        inclination_bounds=bounds["inclination"], raan_bounds=bounds["raan"],
        arg_periapsis_bounds=bounds["arg_periapsis"],
        true_anomaly_bounds=bounds["true_anomaly"])
    sim.reset()
    rec = recovered_elements(get_state(sim))

    for key, (lo, hi) in bounds.items():
        values = np.array([r[key] for r in rec])
        tol = 1e-6 * max(1.0, hi)
        assert values.min() > lo - tol and values.max() < hi + tol, key
        # Uniformity: mean and variance of U[lo, hi] (4-sigma bands, N=4096)
        u = (values - lo) / (hi - lo)
        assert abs(u.mean() - 0.5) < 0.02, key
        assert abs(u.var() - 1.0 / 12.0) < 0.01, key


def test_mean_anomaly_sampling_is_time_uniform(make_sim):
    """Uniform mean anomaly must come out uniform when converted back."""
    ecc = 0.5
    sim = make_sim(num_envs=4096, gravity=False, reset_config={"seed": 3})
    sim.reset_manager.set_orbit_bounds(
        semi_major_axis_bounds=(1.0e7, 1.0e7), eccentricity_bounds=(ecc, ecc),
        mean_anomaly_bounds=(0.0, 2.0 * math.pi))
    sim.reset()
    rec = recovered_elements(get_state(sim))

    nu = np.array([r["true_anomaly"] for r in rec])
    e_anom = 2.0 * np.arctan2(np.sqrt(1.0 - ecc) * np.sin(nu / 2.0),
                              np.sqrt(1.0 + ecc) * np.cos(nu / 2.0))
    mean_anom = (e_anom - ecc * np.sin(e_anom)) % (2.0 * math.pi)

    u = mean_anom / (2.0 * math.pi)
    assert abs(u.mean() - 0.5) < 0.02
    assert abs(u.var() - 1.0 / 12.0) < 0.01
    # Time bias sanity: with e=0.5 the spacecraft spends most of the period
    # on the apoapsis half of the orbit (true anomaly in (pi/2, 3pi/2))
    assert (np.abs(nu - math.pi) < math.pi / 2.0).mean() > 0.6


def test_orbit_consistency_survives_mass_randomization(make_sim):
    """
    Propellant randomization overwrites the sampled mass after the orbit
    draw; it must rescale the momentum so the velocity (and therefore the
    sampled orbit) is unchanged by the loaded mass.
    """
    sim = make_sim(num_envs=2048, gravity=False)
    sim.vehicle.set_config(mass_dry=50.0)
    sim.reset_manager.set_orbit_bounds(
        semi_major_axis_bounds=(7.0e6, 7.0e6), eccentricity_bounds=(0.0, 0.0),
        inclination_bounds=(0.0, math.pi), raan_bounds=(0.0, 2.0 * math.pi))
    sim.randomizer.set_bounds(propellant=(5.0, 20.0))
    sim.reset()
    s = get_state(sim)

    assert (s["mass"] >= 55.0).all() and (s["mass"] <= 70.0).all()
    speed = np.linalg.norm(s["mom"] / s["mass"][:, None], axis=1)
    v_circ = math.sqrt(MU_EARTH / 7.0e6)
    assert np.abs(speed - v_circ).max() < 1e-6 * v_circ


def test_mixed_modes_per_body(make_sim):
    """Orbit mode for one body must not disturb Cartesian sampling of another."""
    sim = make_sim(num_envs=32, num_bodies=2, gravity=False)
    sim.reset_manager.set_orbit_bounds(
        semi_major_axis_bounds=(7.5e6, 7.5e6), eccentricity_bounds=(0.0, 0.0), body_ids=[0])
    sim.reset()
    pos = get_state(sim)["pos"].reshape(32, 2, 3)

    r0 = np.linalg.norm(pos[:, 0], axis=1)
    assert np.abs(r0 - 7.5e6).max() < 1.0  # body 0: sampled on the orbit
    # Body 1 keeps the default Cartesian box around 400 km LEO (x ~ 6.778e6)
    assert (np.abs(pos[:, 1, 0] - (RE_EARTH + 400.0e3)) < 1.0e3 + 1.0).all()
    assert (np.abs(pos[:, 1, 1:]) < 1.0e3 + 1.0).all()


def test_set_sample_mode_switches_back(make_sim):
    sim = make_sim(num_envs=16, gravity=False)
    sim.reset_manager.set_orbit_bounds(semi_major_axis_bounds=(7.5e6, 7.5e6))
    sim.reset_manager.set_sample_mode("cartesian")
    sim.reset()
    pos = get_state(sim)["pos"]
    assert (np.abs(pos[:, 0] - (RE_EARTH + 400.0e3)) < 1.0e3 + 1.0).all()


def test_orbit_seed_determinism(make_sim):
    def draw(seed):
        sim = make_sim(num_envs=64, gravity=False, reset_config={"seed": seed})
        sim.reset_manager.set_orbit_bounds(
            semi_major_axis_bounds=(6.8e6, 7.2e6), eccentricity_bounds=(0.0, 0.3),
            inclination_bounds=(0.0, 1.0), mean_anomaly_bounds=(0.0, 2.0 * math.pi))
        sim.reset()
        return get_state(sim)["pos"]

    assert np.array_equal(draw(7), draw(7))
    assert not np.array_equal(draw(7), draw(8))


def test_orbit_mode_graph_capture(make_sim):
    """Orbit-mode resets inside the captured loop (auto-reset) draw fresh,
    consistent orbits. Entities above the position limit terminate on their
    first step and are redrawn until they land below it; it used to run
    with bounds off, so no reset ever happened inside the graph."""
    limit = RE_EARTH + 450.0e3
    sim = make_sim(num_envs=64, dt=1.0, check_bounds=True, pos_limit=limit)
    sim.reset_manager.set_orbit_bounds(altitude_bounds=(400.0e3, 500.0e3))
    sim.reset()
    sim.simulate(num_steps=10, auto_reset_terminated=True)

    steps = sim.get_episode_steps().numpy()
    assert (steps < 10).any() and (steps == 10).any(), "expected some in-loop resets"
    s = get_state(sim)
    r = np.linalg.norm(s["pos"], axis=1)
    v = np.linalg.norm(s["mom"] / s["mass"][:, None], axis=1)
    assert (r < limit).all()
    # Circular orbits: speed matches the radius the sampler drew
    np.testing.assert_allclose(v, np.sqrt(MU_EARTH / r), rtol=1e-6)


def test_orbit_bounds_validation(make_sim):
    sim = make_sim(num_envs=2, gravity=False)
    with pytest.raises(ValueError, match="eccentricity_bounds"):
        sim.reset_manager.set_orbit_bounds(eccentricity_bounds=(0.5, 1.1))
    with pytest.raises(ValueError, match="semi_major_axis_bounds"):
        sim.reset_manager.set_orbit_bounds(semi_major_axis_bounds=(-1.0, 7.0e6))
    with pytest.raises(ValueError, match="at most one of semi_major_axis_bounds"):
        sim.reset_manager.set_orbit_bounds(
            semi_major_axis_bounds=(7.0e6, 7.0e6), altitude_bounds=(400.0e3, 400.0e3))
    with pytest.raises(ValueError, match="at most one of true_anomaly_bounds"):
        sim.reset_manager.set_orbit_bounds(
            true_anomaly_bounds=(0.0, 1.0), mean_anomaly_bounds=(0.0, 1.0))
    with pytest.raises(ValueError, match="mode"):
        sim.reset_manager.set_sample_mode("keplerian")


def test_reset_rejects_perigee_below_planet_radius(make_sim):
    """Python API raises like the YAML loader, on every reset."""
    sim = make_sim(num_envs=2, gravity=False,
                   check_planet_impact=True, planet_radius=RE_EARTH)
    sim.reset_manager.set_orbit_bounds(
        altitude_bounds=(100.0e3, 100.0e3), eccentricity_bounds=(0.5, 0.5))
    for _ in range(2):
        with pytest.raises(ValueError, match="perigee"):
            sim.reset()


def _yaml_config(device, **overrides):
    config = {
        "simulation": {"num_envs": 16, "num_bodies": 1, "device": device},
        "physics": {"gravity": False, "j2_perturbation": False},
        "termination": {"check_bounds": False},
    }
    config.update(overrides)
    return config


def test_yaml_orbit_bounds(device):
    sim = create_simulator_from_config(_yaml_config(
        device,
        reset={"orbit": {"altitude": [400.0e3, 400.0e3], "eccentricity": 0.0}}))
    sim.reset()
    r = np.linalg.norm(sim.states.numpy()["pos"], axis=1)
    assert np.abs(r - (RE_EARTH + 400.0e3)).max() < 1.0


def test_yaml_orbit_per_body_override(device):
    config = _yaml_config(
        device,
        reset={
            "orbit": {"semi_major_axis": [7.0e6, 7.0e6], "eccentricity": 0.0},
            "per_body": [{"body_id": 1, "orbit": {"semi_major_axis": [8.0e6, 8.0e6]}}],
        })
    config["simulation"]["num_bodies"] = 2
    sim = create_simulator_from_config(config)
    sim.reset()
    r = np.linalg.norm(sim.states.numpy()["pos"], axis=1).reshape(16, 2)
    assert np.abs(r[:, 0] - 7.0e6).max() < 1.0
    assert np.abs(r[:, 1] - 8.0e6).max() < 1.0


def test_yaml_orbit_validation_errors(device):
    with pytest.raises(ValueError, match="planet_radius"):
        create_simulator_from_config(_yaml_config(
            device,
            termination={"check_planet_impact": True, "planet_radius": RE_EARTH},
            reset={"orbit": {"altitude": [100.0e3, 100.0e3], "eccentricity": 0.5}}))
    with pytest.raises(ValueError, match="eccentricity"):
        create_simulator_from_config(_yaml_config(
            device, reset={"orbit": {"eccentricity": [0.5, 1.2]}}))
    with pytest.raises(ValueError, match="at most one of semi_major_axis"):
        create_simulator_from_config(_yaml_config(
            device, reset={"orbit": {"semi_major_axis": 7.0e6, "altitude": 400.0e3}}))
    with pytest.raises(ValueError, match="at most one of true_anomaly"):
        create_simulator_from_config(_yaml_config(
            device, reset={"orbit": {"true_anomaly": [0.0, 1.0], "mean_anomaly": [0.0, 1.0]}}))
    with pytest.raises(ValueError, match="min <= max"):
        create_simulator_from_config(_yaml_config(
            device, reset={"orbit": {"inclination": [1.0, 0.5]}}))
    with pytest.raises(ValueError, match=r"reset.orbit.inclination must be in \[0, pi\]"):
        create_simulator_from_config(_yaml_config(
            device, reset={"orbit": {"inclination": [0.5, 3.5]}}))
    with pytest.raises(ValueError, match=r"orbit.inclination must be in \[0, pi\]"):
        create_simulator_from_config(_yaml_config(
            device, initial_conditions={"per_body": [
                {"body_id": 0, "orbit": {"altitude": 4.0e5, "inclination": -0.2},
                 "mass": 1.0}]}))
    with pytest.raises(ValueError, match=r"att_max_angle must be in \[0, pi\]"):
        create_simulator_from_config(_yaml_config(device, reset={"att_max_angle": 4.0}))


def test_reset_to_conditions_rejects_orbit_mode_defaults(make_sim):
    """Orbit-mode bodies must get explicit positions and momenta (the
    Cartesian bound centers used as defaults are not on the orbits)."""
    import warp as wp

    sim = make_sim(num_envs=4, num_bodies=2, gravity=False)
    sim.reset_manager.set_orbit_bounds(semi_major_axis_bounds=(7.0e6, 7.0e6), body_ids=[1])

    with pytest.raises(ValueError, match="orbital-element sampling"):
        sim.reset_to_conditions()

    # Selecting only the Cartesian body still uses the defaults fine
    sim.reset_to_conditions(body_ids=[0])

    # Explicit positions + momenta are accepted for the orbit-mode body
    n = sim.num_entities
    pos = wp.array(np.tile([7.0e6, 0.0, 0.0], (n, 1)), dtype=wp.vec3d, device=sim.device)
    mom = wp.array(np.tile([0.0, 7.5e3, 0.0], (n, 1)), dtype=wp.vec3d, device=sim.device)
    sim.reset_to_conditions(positions=pos, momenta=mom)
    np.testing.assert_array_equal(get_state(sim)["pos"][:, 0], 7.0e6)

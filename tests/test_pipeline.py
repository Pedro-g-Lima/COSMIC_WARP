"""
Pipeline plumbing checks: trajectory/time-array alignment, reset selection,
reset-mass sanity checks, constructor validation.
"""
import warnings

import numpy as np
import pytest
import warp as wp

from cosmic_warp.core.pipeline import SpacecraftSimulator
from cosmic_warp.physics.gravity import MU_EARTH, RE_EARTH
from conftest import get_state, set_state
from orbit_utils import elliptical_state, circular_velocity


def test_time_array_matches_stored_states(make_sim):
    """
    results['time'][k] must be the epoch of the state stored at index k
    (states are stored post-step, so time[k] = (k+1)*dt). A one-step offset
    on a LEO orbit displaces the analytic position by v*dt ~ 7.5 km, so this
    comparison pins the alignment unambiguously.
    """
    r0 = 7.0e6
    v0 = circular_velocity(MU_EARTH, r0)
    dt = 1.0

    sim = make_sim(num_envs=1, integrator="rk4", dt=dt)
    set_state(sim, pos=[[r0, 0.0, 0.0]], vel=[[0.0, v0, 0.0]])

    result = sim.simulate(num_steps=100, save_states=True)
    time = result["time"].numpy()
    pos = result["positions"].numpy()[:, 0, :]

    np.testing.assert_array_equal(time, (np.arange(100) + 1) * dt)

    for k in (0, 49, 99):
        r_ref, _ = elliptical_state(r0, 0.0, MU_EARTH, time[k])
        err = np.linalg.norm(pos[k] - r_ref)
        assert err < 10.0, (
            f"state at index {k} does not correspond to time[k]={time[k]} s "
            f"(position error {err:.1f} m)"
        )


def test_reset_body_ids_only(make_sim):
    """reset(body_ids=[...]) must reset only those bodies, not everything."""
    sim = make_sim(num_envs=4, num_bodies=2, gravity=False)
    set_state(sim)  # all entities at the origin

    sim.reset(body_ids=[1])
    pos = get_state(sim)["pos"].reshape(4, 2, 3)

    # Body 0 untouched at the origin; body 1 resampled (default bounds are
    # centered on a ~7e6 m LEO radius) in every environment
    np.testing.assert_array_equal(pos[:, 0], 0.0)
    assert (np.linalg.norm(pos[:, 1], axis=1) > 1.0e6).all()


def test_partial_reset_mask_not_sticky(make_sim):
    """Consecutive partial resets must not leak the previous selection."""
    sim = make_sim(num_envs=4, gravity=False)
    set_state(sim)  # all entities at the origin

    sim.reset(env_ids=[1])
    pos_after_first = get_state(sim)["pos"]
    assert np.linalg.norm(pos_after_first[1]) > 1.0e6

    sim.reset(env_ids=[2])
    pos = get_state(sim)["pos"]
    assert np.linalg.norm(pos[2]) > 1.0e6
    # env 1 keeps its state from the first reset; envs 0 and 3 stay untouched
    assert np.array_equal(pos[1], pos_after_first[1])
    np.testing.assert_array_equal(pos[[0, 3]], 0.0)


def test_reset_to_conditions_clears_termination(make_sim):
    """sim.reset_to_conditions must clear termination status like reset()."""
    sim = make_sim(num_envs=2, gravity=False, check_bounds=True, pos_limit=1.0)
    set_state(sim, pos=np.tile([10.0, 0.0, 0.0], (2, 1)))
    sim.step()
    assert sim.get_terminated().numpy()["terminated"].all()

    sim.reset_to_conditions()  # defaults: center of bounds
    assert not sim.get_terminated().numpy()["terminated"].any()


def test_reset_to_conditions_subset(make_sim):
    """Subset selection resets only the chosen environments."""
    sim = make_sim(num_envs=4, gravity=False)
    set_state(sim)  # all entities at the origin

    sim.reset_to_conditions(env_ids=[1, 3])
    pos = get_state(sim)["pos"]
    # Selected envs move to the bounds center (~7e6 m LEO radius by default)
    assert (np.linalg.norm(pos[[1, 3]], axis=1) > 1.0e6).all()
    np.testing.assert_array_equal(pos[[0, 2]], 0.0)


def test_reset_to_conditions_mass_defaults_to_bounds_center(make_sim):
    sim = make_sim(num_envs=3, gravity=False,
                   reset_config={"mass_bounds": (2.0, 4.0)})
    sim.reset_to_conditions()
    np.testing.assert_array_equal(get_state(sim)["mass"], 3.0)


def test_episode_steps_track_partial_resets(make_sim):
    """Per-entity counters count steps since each entity's own last reset."""
    sim = make_sim(num_envs=4, gravity=False)
    sim.reset()
    assert np.array_equal(sim.get_episode_steps().numpy(), [0, 0, 0, 0])

    for _ in range(5):
        sim.step()
    assert np.array_equal(sim.get_episode_steps().numpy(), [5, 5, 5, 5])

    sim.reset(env_ids=[1])
    assert np.array_equal(sim.get_episode_steps().numpy(), [5, 0, 5, 5])

    for _ in range(2):
        sim.step()
    assert np.array_equal(sim.get_episode_steps().numpy(), [7, 2, 7, 7])

    sim.reset_to_conditions(env_ids=[0])
    assert np.array_equal(sim.get_episode_steps().numpy(), [0, 2, 7, 7])

    sim.reset()
    assert np.array_equal(sim.get_episode_steps().numpy(), [0, 0, 0, 0])


def test_episode_steps_advance_in_simulate(make_sim):
    """The captured CUDA-graph path must advance the counters too."""
    sim = make_sim(num_envs=2, gravity=False)
    sim.reset()
    sim.simulate(num_steps=10)
    assert np.array_equal(sim.get_episode_steps().numpy(), [10, 10])


def test_reset_rejects_mass_bounds_below_dry(make_sim):
    """
    A heavy vehicle with the default 1 kg reset mass bounds would spawn
    entities below dry mass. The Python API raises like the YAML loader, on
    every reset, so a vehicle changed after the first reset is caught too.
    """
    sim = make_sim(num_envs=2, gravity=False)
    sim.vehicle.set_config(mass_dry=70.0)
    with pytest.raises(ValueError, match="mass_dry"):
        sim.reset()

    sim.reset_manager.set_bounds(mass_bounds=(80.0, 120.0))
    sim.reset()
    # A dry mass above a current mass is rejected by the setter itself; one
    # below the current masses but above the bounds' minimum, by reset()
    sim.reset_to_conditions(masses=wp.full(2, 100.0, dtype=wp.float64, device=sim.device))
    sim.vehicle.set_config_per_entity(mass_dry=np.array([70.0, 90.0]))
    with pytest.raises(ValueError, match=r"mass_dry \(90.0\)"):
        sim.reset()


def test_reset_no_warning_when_bounds_or_randomization_fix_it(make_sim):
    sim = make_sim(num_envs=2, gravity=False)
    sim.vehicle.set_config(mass_dry=70.0)
    sim.reset_manager.set_bounds(mass_bounds=(80.0, 120.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sim.reset()

    # Propellant randomization overwrites the sampled mass with dry + propellant
    sim2 = make_sim(num_envs=2, gravity=False)
    sim2.vehicle.set_config(mass_dry=70.0)
    sim2.randomizer.set_bounds(propellant=(5.0, 15.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sim2.reset()


def test_metadata_is_complete_and_serializable(make_sim):
    """metadata() must reflect the constructor settings and survive json.dumps."""
    import json

    import cosmic_warp

    sim = make_sim(num_envs=3, num_bodies=2, integrator="rk4", dt=2.5, drag=True)
    meta = sim.metadata()

    assert meta["cosmic_warp_version"] == cosmic_warp.__version__
    assert meta["num_envs"] == 3
    assert meta["num_bodies"] == 2
    assert meta["integrator"] == "rk4"
    assert meta["dt"] == 2.5
    assert meta["drag"] is True
    assert meta["j2_perturbation"] is False  # make_sim default
    json.dumps(meta)  # must not raise


@pytest.mark.parametrize("kwargs, match", [
    ({"dt": 0.0}, "dt must be > 0"),
    ({"dt": float("nan")}, "dt must be finite"),
    ({"num_envs": 0}, "num_envs must be a positive integer"),
    ({"num_bodies": 2.5}, "num_bodies must be a positive integer"),
    ({"pos_limit": -1.0}, "pos_limit must be > 0"),
    ({"mass_limit": -1.0}, "mass_limit must be >= 0"),
    ({"gravity": False, "j2_perturbation": True}, "requires gravity=True"),
])
def test_invalid_constructor_arguments_rejected(device, kwargs, match):
    with pytest.raises(ValueError, match=match):
        SpacecraftSimulator(device=device, **kwargs)


def test_j2_follows_gravity_by_default(device):
    """gravity=False no longer leaves J2 nominally on (it was silently ignored)."""
    sim = SpacecraftSimulator(device=device, gravity=False)
    assert not sim.flags.numpy()["j2_perturbation"][0]
    sim = SpacecraftSimulator(device=device)
    assert sim.flags.numpy()["j2_perturbation"][0]


def test_mass_limit_at_dry_mass_terminates_when_propellant_runs_out(make_sim):
    """Mass is clamped at mass_dry, so a strict '<' test could never fire for
    the natural setting mass_limit = mass_dry."""
    sim = make_sim(gravity=False, check_bounds=True, mass_limit=1.0, dt=0.1)
    sim.thruster_manager.set_config(max_thrust=10.0, isp=100.0)
    sim.set_thrust_commands([1.0, 0.0, 0.0, 0.0])
    set_state(sim, mass=[1.05])
    sim.simulate(200)
    status = sim.get_terminated().numpy()
    assert status["terminated"][0] and status["reason"][0] == 4


@pytest.mark.parametrize("integrator", ["euler", "rk4", "symplectic"])
@pytest.mark.parametrize("dry, thrust, isp, dt", [
    (2.9, 10.0, 300.0, 0.1),
    (1.3, 0.7, 65.0, 1.3),
])
def test_mass_limit_at_dry_mass_fires_on_every_burnout_step(make_sim, integrator,
                                                            dry, thrust, isp, dt):
    """The symplectic integrator rounded the mass twice per step (two half
    kicks), so on the burnout step it could stop one ulp above dry; the
    leftover flow then rounded away and mass_limit = mass_dry never fired
    (about a quarter of entities in these two settings). Every integrator now
    stores the step-end mass rounded once, and exactly mass_dry on the
    burnout step: each entity terminates with reason 4 on its burnout step."""
    n = 4096
    sim = make_sim(num_envs=n, integrator=integrator, gravity=False, check_bounds=True,
                   mass_limit=dry, dt=dt)
    sim.vehicle.set_config(mass_dry=dry, inertia=np.eye(3) * dry / 6.0)
    sim.thruster_manager.set_config(max_thrust=thrust, isp=isp, positions=[(0.0, 0.0, 0.0)],
                                    directions=[(1.0, 0.0, 0.0)])
    sim.set_thrust_commands([1.0, 0.0, 0.0, 0.0])

    # Random loads lasting k + f steps, f kept away from a step boundary so
    # the burnout step k + 1 is unambiguous
    rng = np.random.default_rng(3)
    burn_steps = rng.integers(0, 30, n) + rng.uniform(0.05, 0.95, n)
    step_flow = thrust / (isp * 9.80665) * dt
    set_state(sim, mass=dry + burn_steps * step_flow)

    first = np.zeros(n, dtype=int)
    for step in range(1, 33):
        sim.step()
        status = sim.get_terminated().numpy()
        first[(first == 0) & status["terminated"]] = step
    np.testing.assert_array_equal(first, np.floor(burn_steps).astype(int) + 1)
    np.testing.assert_array_equal(status["reason"], 4)
    np.testing.assert_array_equal(get_state(sim)["mass"], dry)


def test_planet_radius_defaults_to_earth(make_sim):
    """planet_radius defaulted to 0, so check_planet_impact=True alone never
    fired; it now defaults to Earth's equatorial radius, in Python and YAML."""
    sim = make_sim(num_envs=2, gravity=False, check_planet_impact=True)
    assert sim.termination_checker.planet_radius == RE_EARTH
    set_state(sim, pos=[[RE_EARTH - 1.0, 0.0, 0.0], [RE_EARTH + 1.0e5, 0.0, 0.0]])
    sim.step()
    status = sim.get_terminated().numpy()
    np.testing.assert_array_equal(status["terminated"], [True, False])
    assert status["reason"][0] == 5
    from cosmic_warp.core.config import create_simulator_from_config
    yaml_sim = create_simulator_from_config({"simulation": {"device": sim.device}})
    assert yaml_sim.termination_checker.planet_radius == RE_EARTH


def test_default_limits_do_not_depend_on_the_vehicle(make_sim):
    """A heavy, spinning spacecraft in LEO runs with the default limits: the
    momentum limits are off until set (they scale with mass and inertia)."""
    sim = make_sim(num_envs=2, check_bounds=True)
    sim.vehicle.set_config(mass_dry=500.0, inertia=np.diag([100.0, 120.0, 140.0]))
    sim.reset_manager.set_bounds(mass_bounds=(500.0, 500.0),
                                 ang_mom_bounds=([50.0] * 3, [50.0] * 3))
    sim.reset()
    sim.simulate(num_steps=10)
    assert not sim.get_terminated().numpy()["terminated"].any()
    with pytest.raises(ValueError, match="pos_limit must be finite"):
        make_sim(pos_limit=float("nan"))

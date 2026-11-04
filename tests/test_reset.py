"""
Reset / RNG statistical validation and determinism.

Checks:
  1. Sampled initial conditions are statistically independent across state
     components (guards against the LCG state-threading bug fixed in 2026-07,
     where pos.y and mom.x were drawn from the same underlying value).
  2. Samples are uniform within the configured bounds.
  3. Same seed -> identical draws; different seed -> different draws.
  4. Repeated resets draw fresh values.
  5. Deterministic propagation: identical runs produce bit-identical states.
"""
import numpy as np
import pytest
import warp as wp

from conftest import get_state, set_state
from cosmic_warp.physics.gravity import MU_EARTH, RE_EARTH


BOUNDS = {
    "pos_bounds": ((-100.0, -100.0, -100.0), (100.0, 100.0, 100.0)),
    "mom_bounds": ((-10.0, -10.0, -10.0), (10.0, 10.0, 10.0)),
    "ang_mom_bounds": ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
    "mass_bounds": (1.0, 2.0),
}


@pytest.fixture
def sampled_states(make_sim):
    sim = make_sim(num_envs=4096, gravity=False,
                   reset_config={**BOUNDS, "seed": 123})
    sim.reset()
    return get_state(sim)


def test_no_cross_component_correlation(sampled_states):
    """All pairs of sampled components must be uncorrelated."""
    s = sampled_states
    columns = [s["pos"][:, i] for i in range(3)] \
        + [s["mom"][:, i] for i in range(3)] \
        + [s["ang_mom"][:, i] for i in range(3)] \
        + [s["mass"]]
    corr = np.corrcoef(np.vstack(columns))
    off_diag = corr[~np.eye(len(columns), dtype=bool)]
    # 4/sqrt(N) ~ 0.06 is the ~4-sigma band for N=4096 independent samples
    assert np.abs(off_diag).max() < 0.08, (
        f"correlated initial conditions (max |corr| = {np.abs(off_diag).max():.3f})"
    )


def test_uniformity_within_bounds(sampled_states):
    s = sampled_states
    lo, hi = np.array(BOUNDS["pos_bounds"][0]), np.array(BOUNDS["pos_bounds"][1])
    assert (s["pos"] >= lo).all() and (s["pos"] <= hi).all()
    # Mean and variance of a uniform distribution on [lo, hi]
    u = (s["pos"] - lo) / (hi - lo)
    assert np.abs(u.mean(axis=0) - 0.5).max() < 0.02
    assert np.abs(u.var(axis=0) - 1.0 / 12.0).max() < 0.01

    assert (s["mass"] >= 1.0).all() and (s["mass"] <= 2.0).all()

    # Quaternions must be unit and cover both hemispheres (uniform on S3)
    norms = np.linalg.norm(s["quat"], axis=1)
    assert np.abs(norms - 1.0).max() < 1e-12
    assert 0.4 < (s["quat"][:, 3] > 0).mean() < 0.6


def test_fixed_attitude(make_sim):
    """att_max_angle=0 must always yield exactly att_nominal (normalized)."""
    q_nom = np.array([0.5, 0.5, 0.5, 0.5])
    sim = make_sim(num_envs=64, gravity=False,
                   reset_config={"att_nominal": tuple(q_nom), "att_max_angle": 0.0})
    sim.reset()
    q = get_state(sim)["quat"]
    assert np.allclose(q, q_nom, atol=1e-12)


def test_attitude_cone(make_sim):
    """Sampled attitudes stay within the cone and follow the Haar density."""
    max_angle = 0.2
    q_nom = np.array([0.0, 0.0, np.sin(0.3), np.cos(0.3)])  # 0.6 rad about z
    sim = make_sim(num_envs=4096, gravity=False,
                   reset_config={"att_nominal": tuple(q_nom),
                                 "att_max_angle": max_angle, "seed": 5})
    sim.reset()
    q = get_state(sim)["quat"]

    assert np.abs(np.linalg.norm(q, axis=1) - 1.0).max() < 1e-12
    angles = 2.0 * np.arccos(np.clip(np.abs(q @ q_nom), -1.0, 1.0))
    assert angles.max() <= max_angle + 1e-9
    assert angles.max() > 0.95 * max_angle, "cone not filled"
    # Haar-uniform in a small ball: density ~ theta^2, so E[theta] = 3/4 max.
    # A uniform-angle (non sin^2-weighted) draw would give 1/2 max instead.
    assert abs(angles.mean() - 0.75 * max_angle) < 0.01


def test_attitude_per_body_override(make_sim):
    """set_bounds(att_...) applies to the selected body only."""
    sim = make_sim(num_envs=32, num_bodies=2, gravity=False)
    q_nom = (0.0, 0.0, 0.0, 1.0)
    sim.reset_manager.set_bounds(att_nominal=q_nom, att_max_angle=0.0, body_ids=[1])
    sim.reset()
    q = get_state(sim)["quat"].reshape(32, 2, 4)
    assert np.allclose(q[:, 1], np.array(q_nom), atol=1e-12)
    # Body 0 keeps the default uniform SO(3) sampling
    assert not np.allclose(q[:, 0], np.array(q_nom), atol=1e-3)


def test_attitude_validation(make_sim):
    sim = make_sim(num_envs=2, gravity=False)
    with pytest.raises(ValueError, match="att_max_angle"):
        sim.reset_manager.set_bounds(att_max_angle=-0.1)
    with pytest.raises(ValueError, match="att_nominal"):
        sim.reset_manager.set_bounds(att_nominal=(0.0, 0.0, 0.0, 0.0))


def test_seed_determinism(make_sim):
    def draw(seed):
        sim = make_sim(num_envs=64, gravity=False,
                       reset_config={**BOUNDS, "seed": seed})
        sim.reset()
        return get_state(sim)["pos"]

    assert np.array_equal(draw(7), draw(7)), "same seed must give identical draws"
    assert not np.array_equal(draw(7), draw(8)), "different seeds must differ"


def test_repeated_resets_fresh(make_sim):
    sim = make_sim(num_envs=64, gravity=False, reset_config=dict(BOUNDS))
    sim.reset()
    p1 = get_state(sim)["pos"]
    sim.reset()
    p2 = get_state(sim)["pos"]
    assert not np.array_equal(p1, p2), "second reset repeated the first draw"


def test_propagation_determinism(make_sim):
    """Two identical runs must produce bit-identical final states."""
    def run():
        sim = make_sim(num_envs=8, integrator="symplectic", dt=1.0)
        pos = np.tile([7.0e6, 0.0, 0.0], (8, 1))
        vel = np.tile([0.0, 7546.0, 0.0], (8, 1))
        set_state(sim, pos=pos, vel=vel, ang_mom=np.full((8, 3), 0.01))
        for _ in range(500):
            sim.step()
        return get_state(sim)

    a, b = run(), run()
    for key in a:
        assert np.array_equal(a[key], b[key]), f"non-deterministic field: {key}"


@pytest.mark.parametrize("kwargs, match", [
    ({"env_ids": [3], "entity_ids": [0]}, "cannot be combined"),
    ({"env_ids": [2]}, r"env_ids entry 2 out of range \[0, 2\)"),
    ({"env_ids": [-1]}, "out of range"),
    ({"body_ids": [3]}, "body_ids entry 3 out of range"),
    ({"entity_ids": [6]}, "entity_ids entry 6 out of range"),
    ({"env_ids": [0.5]}, "must contain integers"),
    # Float, string and boolean ids used to be converted: a boolean mask
    # selected environments 0 and 1 by value, not where it was True
    ({"env_ids": [1.0]}, "env_ids must contain integers, got 1.0"),
    ({"env_ids": ["1"]}, "env_ids must contain integers, got '1'"),
    ({"env_ids": [True]}, "env_ids must contain integers, got True"),
    ({"env_ids": np.array([False, True])}, "env_ids must be a 1-D integer array"),
    ({"env_ids": np.array([1.0])}, "env_ids must be a 1-D integer array"),
    ({"body_ids": 1}, "body_ids must be a list of integers, got int"),
    ({"entity_ids": "12"}, "entity_ids must be a list of integers, got str"),
])
def test_invalid_selection_rejected(make_sim, kwargs, match):
    """Selections are checked once, for every manager: combining entity_ids
    with env/body ids, indexing out of range or anything but a list of
    integers raises instead of silently selecting something else (negative
    indices used to wrap around)."""
    sim = make_sim(num_envs=2, num_bodies=3)
    with pytest.raises(ValueError, match=match):
        sim.reset(**kwargs)
    with pytest.raises(ValueError, match=match):
        sim.vehicle.set_config(**kwargs)


def test_integer_arrays_and_ranges_select(make_sim):
    """NumPy integer arrays and ranges are selections like lists; a Warp
    array (on the device) is not."""
    sim = make_sim(num_envs=4)
    sim.vehicle.set_config(cd=1.0, env_ids=np.array([1, 3], dtype=np.int64))
    sim.vehicle.set_config(cd=2.0, env_ids=range(1))
    np.testing.assert_array_equal(sim.vehicle.configs.numpy()["cd"], [2.0, 1.0, 2.2, 1.0])
    with pytest.raises(ValueError, match="env_ids must be a list of integers, got array"):
        sim.reset(env_ids=wp.array([0], dtype=wp.int32, device=sim.device))


def test_reset_bounds_body_ids_checked(make_sim):
    sim = make_sim(num_bodies=2)
    with pytest.raises(ValueError, match="body_ids entry -1 out of range"):
        sim.reset_manager.set_bounds(mass_bounds=(1.0, 2.0), body_ids=[-1])
    with pytest.raises(ValueError, match="body_ids entry 2 out of range"):
        sim.randomizer.set_bounds(propellant=(0.0, 1.0), body_ids=[2])


def test_reset_to_conditions_rejects_short_arrays(make_sim):
    """Condition arrays are indexed by flat entity id; a short array used to
    read out of bounds (garbage or NaN states for the other entities)."""
    sim = make_sim(num_envs=4)
    short = wp.array([[7.0e6, 0.0, 0.0]], dtype=wp.vec3d, device=sim.device)
    with pytest.raises(ValueError, match=r"positions must have shape \(num_entities,\) = \(4,\)"):
        sim.reset_to_conditions(positions=short)
    wrong_dtype = wp.zeros(4, dtype=wp.float64, device=sim.device)
    with pytest.raises(ValueError, match="momenta must be a Warp array of vec3d"):
        sim.reset_to_conditions(momenta=wrong_dtype)


def test_default_reset_velocity_holds_for_any_mass(make_sim):
    """The default reset draws a velocity box around circular LEO speed and
    sets momentum = mass * velocity, so a heavy vehicle still starts in orbit
    (the former momentum box gave 100 kg about 1/100 of orbital speed)."""
    sim = make_sim(num_envs=8)
    sim.vehicle.set_config(mass_dry=100.0)
    sim.reset_manager.set_bounds(mass_bounds=(100.0, 150.0))
    sim.reset()
    s = get_state(sim)
    v = np.linalg.norm(s["mom"] / s["mass"][:, None], axis=1)
    v_circ = np.sqrt(MU_EARTH / (RE_EARTH + 400.0e3))
    np.testing.assert_allclose(v, v_circ, atol=2.0)

    # The reset_to_conditions default (bound centers) follows the same rule
    sim.reset_to_conditions(masses=wp.full(8, 120.0, dtype=wp.float64, device=sim.device))
    s = get_state(sim)
    np.testing.assert_allclose(s["mom"][:, 1], 120.0 * v_circ, rtol=1e-12)


def test_reset_to_conditions_default_mass_below_dry_rejected(make_sim):
    """Without masses, reset_to_conditions writes the mass_bounds center with no
    randomization, so a center below dry mass would spawn without propellant."""
    sim = make_sim(num_envs=2)
    sim.vehicle.set_config(mass_dry=12.0)
    with pytest.raises(ValueError, match="mass_bounds center"):
        sim.reset_to_conditions()
    sim.reset_to_conditions(masses=wp.full(2, 15.0, dtype=wp.float64, device=sim.device))
    np.testing.assert_array_equal(get_state(sim)["mass"], 15.0)


def test_reset_to_conditions_masses_below_dry_rejected(make_sim):
    """Explicit masses below dry mass (or non-finite) are rejected, not
    silently clamped by the next step; the check uses the device mass_dry,
    which randomization may have redrawn."""
    sim = make_sim(num_envs=2)
    sim.vehicle.set_config(mass_dry=12.0)
    with pytest.raises(ValueError, match=r"masses\[1\] = 11.0"):
        sim.reset_to_conditions(masses=wp.array([15.0, 11.0], dtype=wp.float64, device=sim.device))
    with pytest.raises(ValueError, match="not finite"):
        sim.reset_to_conditions(masses=wp.array([15.0, np.nan], dtype=wp.float64, device=sim.device))
    # A partial reset checks only the selected entities
    sim.reset_to_conditions(masses=wp.array([15.0, 11.0], dtype=wp.float64, device=sim.device),
                            env_ids=[0])


@pytest.mark.parametrize("seed", ["42", 4.7, True, -1, 2**64])
def test_reset_seed_rejected(make_sim, seed):
    sim = make_sim()
    with pytest.raises(ValueError, match="reset seed"):
        sim.reset_manager.reseed(seed)


def test_momentum_and_velocity_bounds_exclusive(make_sim):
    sim = make_sim()
    with pytest.raises(ValueError, match="at most one of mom_bounds / vel_bounds"):
        sim.reset_manager.set_bounds(mom_bounds=((0,) * 3, (1,) * 3),
                                     vel_bounds=((0,) * 3, (1,) * 3))
    # Setting momentum bounds switches the body back to momentum sampling
    sim.reset_manager.set_bounds(mom_bounds=((2.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
                                 mass_bounds=(4.0, 4.0))
    sim.reset()
    np.testing.assert_array_equal(get_state(sim)["mom"][0], [2.0, 0.0, 0.0])


@pytest.mark.parametrize("kwargs, match", [
    ({"mass_bounds": (5.0, 1.0)}, "min <= max"),
    ({"mass_bounds": (-1.0, 1.0)}, "mass_bounds must be > 0"),
    ({"pos_bounds": ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))}, "pos_bounds must satisfy min <= max"),
    ({"vel_bounds": ((0.0, 0.0), (1.0, 1.0))}, "3 components"),
    ({"ang_mom_bounds": ((0.0, 0.0, float("nan")), (1.0, 1.0, 1.0))}, "must be finite"),
])
def test_invalid_reset_bounds_rejected(make_sim, kwargs, match):
    sim = make_sim()
    with pytest.raises(ValueError, match=match):
        sim.reset_manager.set_bounds(**kwargs)


def test_rejected_bounds_leave_all_bodies_unchanged(make_sim):
    """set_bounds used to store each body's new bounds before validating them:
    a rejected call kept failing every later call, and the dry-mass check read
    the stored minimum while the device still sampled the old bounds."""
    sim = make_sim(num_envs=2, num_bodies=2)
    sim.vehicle.set_config(mass_dry=5.0)
    sim.reset_manager.set_bounds(mass_bounds=(5.0, 6.0), body_ids=[0])
    with pytest.raises(ValueError, match="min <= max"):
        sim.reset_manager.set_bounds(mass_bounds=(5.0, 1.0))
    with pytest.raises(ValueError, match="mass_dry"):
        sim.reset()   # body 1 still has the default 1 kg bounds
    sim.reset_manager.set_bounds(ang_mom_bounds=((0.0,) * 3, (0.0,) * 3))
    with pytest.raises(ValueError, match="semi_major_axis_bounds"):
        sim.reset_manager.set_orbit_bounds(semi_major_axis_bounds=(7.0e6, 6.0e6))
    sim.reset_manager.set_bounds(mass_bounds=(5.0, 6.0), body_ids=[1])
    sim.reset()   # neither failed call switched a body to orbit sampling
    mass = get_state(sim)["mass"]
    assert np.all((mass >= 5.0) & (mass <= 6.0))


def test_array_bounds_accepted(make_sim):
    sim = make_sim()
    sim.reset_manager.set_bounds(mass_bounds=np.array([2.0, 2.0]),
                                 pos_bounds=np.full((2, 3), 7.0e6),
                                 vel_bounds=np.zeros((2, 3)))
    sim.reset()
    s = get_state(sim)
    np.testing.assert_array_equal(s["mass"], 2.0)
    np.testing.assert_array_equal(s["pos"], 7.0e6)


def test_orbit_bounds_take_no_mu(make_sim):
    """The dynamics always use Earth's gravity, so orbit specs have no mu: one
    with another value would start entities on an orbit they cannot follow."""
    sim = make_sim()
    with pytest.raises(TypeError, match="mu"):
        sim.reset_manager.set_orbit_bounds(altitude_bounds=(4e5, 4e5), mu=MU_EARTH)


@pytest.mark.parametrize("integrator", ["euler", "rk4", "symplectic"])
@pytest.mark.parametrize("w", [2.0, 0.5])
def test_reset_to_conditions_normalizes_quaternions(make_sim, integrator, w):
    """A non-unit quaternion was stored as given and rotated the thrust
    unnormalized on the first step: q = (0, 0, 0, 2) gave a step-1 delta-v of
    3.5, 1.0 and 2.0 m/s (Euler, RK4, symplectic) for 0.5 expected, and
    q = (0, 0, 0, 0.5) reversed the thrust under Euler. The reset now
    normalizes it, so step 1 already gives F dt / m."""
    sim = make_sim(num_envs=1, integrator=integrator, gravity=False, dt=1.0)
    sim.vehicle.set_config(inertia=np.eye(3) / 3.0)    # dry 1 kg: 1 kg of propellant
    sim.thruster_manager.set_config(max_thrust=1.0, positions=[(0.0, 0.0, 0.0)],
                                    directions=[(1.0, 0.0, 0.0)])

    def vec3(v):
        return wp.array([v], dtype=wp.vec3d, device=sim.device)

    sim.reset_to_conditions(positions=vec3((7.0e6, 0.0, 0.0)), momenta=vec3((0.0, 0.0, 0.0)),
                            angular_momenta=vec3((0.0, 0.0, 0.0)),
                            quaternions=wp.array([(0.0, 0.0, 0.0, w)], dtype=wp.quatd,
                                                 device=sim.device),
                            masses=wp.array([2.0], dtype=wp.float64, device=sim.device))
    np.testing.assert_array_equal(get_state(sim)["quat"][0], [0.0, 0.0, 0.0, 1.0])
    sim.set_thrust_commands([1.0, 0.0, 0.0, 0.0])   # the reset cleared the commands
    sim.step()
    np.testing.assert_allclose(sim.get_velocities().numpy()[0], [0.5, 0.0, 0.0], atol=1e-15)


@pytest.mark.parametrize("q", [(0.0, 0.0, 0.0, 0.0), (0.0, np.nan, 0.0, 1.0),
                               (0.0, 0.0, np.inf, 1.0)])
def test_reset_to_conditions_rejects_zero_or_non_finite_quaternion(make_sim, q):
    """Normalization cannot recover a direction from these; only the selected
    entities are checked."""
    sim = make_sim(num_envs=2)
    quats = wp.array([(0.0, 0.0, 0.0, 1.0), q], dtype=wp.quatd, device=sim.device)
    with pytest.raises(ValueError, match=r"quaternions\[1\].*finite nonzero"):
        sim.reset_to_conditions(quaternions=quats)
    sim.reset_to_conditions(quaternions=quats, env_ids=[0])


def test_dry_mass_above_current_mass_rejected(make_sim):
    """set_config(mass_dry=...) after a reset left the state below its dry
    mass: gravity used m while the velocity used max(m, dry), and the next
    update raised the mass to dry while keeping the momentum (free-fall from
    rest came out 4.07, 7.46, 6.10 m/s for 8.13). It is now rejected, for the
    selected entities only, by both setters; before the first reset (masses
    still 0) nothing is checked."""
    sim = make_sim(num_envs=2)
    sim.vehicle.set_config(mass_dry=1.5)            # before any reset: allowed
    sim.vehicle.set_config(mass_dry=1.0)
    sim.reset_to_conditions(masses=wp.array([1.0, 3.0], dtype=wp.float64, device=sim.device))
    with pytest.raises(ValueError, match="mass_dry 2.0 exceeds the current mass 1.0 of entity 0"):
        sim.vehicle.set_config(mass_dry=2.0)
    with pytest.raises(ValueError, match="of entity 0"):
        sim.vehicle.set_config_per_entity(mass_dry=np.array([2.0, 2.0]))
    np.testing.assert_array_equal(sim.vehicle.configs.numpy()["mass_dry"], 1.0)
    sim.vehicle.set_config(mass_dry=3.0, env_ids=[1])   # at the current mass
    sim.vehicle.set_config_per_entity(mass_dry=np.array([1.0, 2.5]))


@pytest.mark.parametrize("path", ["reset_terminated", "auto_reset", "vecenv"])
def test_device_resets_never_spawn_below_dry_mass(make_sim, path):
    """sim.reset() rejects mass_bounds below the dry mass, but the device
    reset paths cannot raise and skipped that check: bounds lowered after the
    first reset respawned terminated entities below dry mass. The reset
    kernels now floor the sampled mass at the entity's dry mass; the velocity
    box still holds (momentum = floored mass * sampled velocity)."""
    # Spawned beyond pos_limit, so every entity terminates on its first step
    sim = make_sim(num_envs=4, gravity=False, check_bounds=True, pos_limit=1.0e7)
    sim.vehicle.set_config(mass_dry=2.0)
    sim.reset_manager.set_bounds(mass_bounds=(2.0, 3.0), pos_bounds=((2.0e7,) * 3, (2.0e7,) * 3),
                                 vel_bounds=((5.0,) * 3, (5.0,) * 3))
    if path == "vecenv":
        from cosmic_warp.rl.env import VecEnv
        from rl_tasks import DetumbleTask
        env = VecEnv(sim, DetumbleTask(), max_episode_steps=100)
        env.reset()
    else:
        sim.reset()
    sim.reset_manager.set_bounds(mass_bounds=(0.5, 1.0))
    if path == "reset_terminated":
        sim.step()
        sim.reset_terminated()
    elif path == "auto_reset":
        sim.simulate(num_steps=1, auto_reset_terminated=True)
    else:
        env.actions.fill_(0.0)
        env.step()
    s = get_state(sim)
    np.testing.assert_array_equal(s["mass"], 2.0)
    np.testing.assert_allclose(s["mom"], 2.0 * 5.0, rtol=1e-15)


def test_reset_to_conditions_floors_mass_at_dry(make_sim):
    """The manager-level reset (no host checks) floors the mass too."""
    sim = make_sim(num_envs=2)
    sim.vehicle.set_config(mass_dry=2.0)
    sim.reset_manager.reset_to_conditions(
        sim.states, masses=wp.array([1.0, 3.0], dtype=wp.float64, device=sim.device))
    np.testing.assert_array_equal(get_state(sim)["mass"], [2.0, 3.0])


@pytest.mark.parametrize("kwargs, match", [
    ({"att_max_angle": 3.2}, r"att_max_angle must be in \[0, pi\]"),
    ({"att_max_angle": -0.1}, r"att_max_angle must be in \[0, pi\]"),
])
def test_attitude_cone_angle_range(make_sim, kwargs, match):
    """A cone half-angle beyond pi used to be accepted (silently uniform)."""
    sim = make_sim()
    with pytest.raises(ValueError, match=match):
        sim.reset_manager.set_bounds(**kwargs)
    with pytest.raises(ValueError, match=match):
        make_sim(reset_config=kwargs)
    sim.reset_manager.set_bounds(att_max_angle=np.pi)


@pytest.mark.parametrize("inc", [(-0.1, 0.5), (0.5, 3.2)])
def test_inclination_bounds_range(make_sim, inc):
    """Inclinations outside [0, pi] describe an orbit already covered inside
    it (with another node and periapsis): rejected rather than reinterpreted."""
    sim = make_sim()
    with pytest.raises(ValueError, match=r"inclination_bounds must be in \[0, pi\]"):
        sim.reset_manager.set_orbit_bounds(inclination_bounds=inc)
    sim.reset_manager.set_orbit_bounds(inclination_bounds=(0.0, np.pi))

"""
Public simulator API: getters, per-entity setters, termination reasons,
argument checks and the shipped YAML configs.
"""
import inspect
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from cosmic_warp.core.config import create_simulator_from_yaml
from cosmic_warp.core.reset import ResetManager
from cosmic_warp.physics.gravity import RE_EARTH
from conftest import get_state, set_state

INERTIA = np.diag([2.0, 3.0, 4.0])


def _sim(make_sim, **kwargs):
    kwargs.setdefault("num_envs", 3)
    kwargs.setdefault("gravity", False)
    sim = make_sim(dt=0.1, **kwargs)
    sim.vehicle.set_config(mass_dry=1.0, inertia=INERTIA)
    return sim


def test_getters_match_state(make_sim):
    sim = _sim(make_sim)
    rng = np.random.default_rng(0)
    q = rng.normal(size=(3, 4))
    q /= np.linalg.norm(q, axis=1)[:, None]
    set_state(sim, pos=rng.normal(size=(3, 3)) * 7e6, vel=rng.normal(size=(3, 3)),
              mass=[2.0, 3.0, 4.0], quat=q, ang_mom=rng.normal(size=(3, 3)))
    s = get_state(sim)
    np.testing.assert_array_equal(sim.get_positions().numpy(), s["pos"])
    np.testing.assert_allclose(sim.get_velocities().numpy(), s["mom"] / s["mass"][:, None],
                               rtol=1e-15)
    np.testing.assert_array_equal(sim.get_quaternions().numpy(), s["quat"])
    np.testing.assert_allclose(sim.get_angular_velocities().numpy(),
                               s["ang_mom"] @ np.linalg.inv(INERTIA), rtol=1e-14)
    np.testing.assert_array_equal(sim.get_masses().numpy(), s["mass"])


def test_per_entity_commands_reach_only_their_entity(make_sim):
    sim = _sim(make_sim)
    sim.thruster_manager.set_config(max_thrust=1.0, isp=0.0, positions=[(0.0, 0.0, 0.0)],
                                    directions=[(1.0, 0.0, 0.0)])
    sim.rw_manager.set_config(max_torque=0.1, max_momentum=1.0, wheel_inertia=0.01)
    set_state(sim, pos=[[7e6, 0.0, 0.0]] * 3, mass=[2.0] * 3)
    thrust = np.zeros((3, 8)); thrust[1, 0] = 1.0
    torque = np.zeros((3, 4)); torque[2, 0] = 0.01
    sim.set_thrust_commands_per_entity(wp.array(thrust, dtype=wp.float64, device=sim.device))
    sim.set_rw_torque_commands_per_entity(wp.array(torque, dtype=wp.float64, device=sim.device))
    sim.step()
    s = get_state(sim)
    assert s["mom"][1][0] > 0.0 and s["mom"][0][0] == 0.0 and s["mom"][2][0] == 0.0
    assert np.linalg.norm(s["ang_mom"][2]) > 0.0
    assert not s["ang_mom"][0].any() and not s["ang_mom"][1].any()


def test_per_entity_wheel_state(make_sim):
    sim = _sim(make_sim)
    sim.rw_manager.set_config(max_momentum=1.0)
    h = np.zeros((3, 4)); h[:, 0] = [0.1, 0.2, 0.3]
    sim.set_rw_angular_momentum_state_per_entity(wp.array(h, dtype=wp.float64, device=sim.device))
    np.testing.assert_array_equal(sim.actuators.states.numpy()["rw_ang_mom"], h)


@pytest.mark.parametrize("call", [
    lambda sim: sim.set_thrust_commands_per_entity(wp.zeros((2, 8), dtype=wp.float64, device=sim.device)),
    lambda sim: sim.set_thrust_commands_per_entity(wp.zeros((3, 4), dtype=wp.float64, device=sim.device)),
    lambda sim: sim.set_rw_torque_commands_per_entity(wp.zeros((3, 3), dtype=wp.float64, device=sim.device)),
    lambda sim: sim.set_rw_torque_commands_per_entity(wp.zeros((3, 4), dtype=wp.float32, device=sim.device)),
    lambda sim: sim.set_rw_angular_momentum_state_per_entity(np.zeros((3, 4))),
])
def test_per_entity_setters_reject_wrong_arrays(make_sim, call):
    """The kernels read these arrays in place: a short one was read past its
    end and a float32 one reinterpreted."""
    with pytest.raises(ValueError, match="must be a float64 Warp array of shape"):
        call(_sim(make_sim))


@pytest.mark.parametrize("call, match", [
    (lambda sim: sim.set_thrust_commands([0.5] * 9), "thrust_cmd: at most 8 values"),
    (lambda sim: sim.set_rw_torque_commands([0.0] * 5), "rw_torque_cmds: at most 4 values"),
    (lambda sim: sim.set_rw_angular_momentum_state([0.0] * 5), "rw_ang_mom: at most 4 values"),
    (lambda sim: sim.set_thrust_commands(["0.5"]), "thrust_cmd must be a number"),
    (lambda sim: sim.set_thrust_commands(0.5), "thrust_cmd must be a list"),
    (lambda sim: sim.set_rw_torque_commands([float("nan")]), "rw_torque_cmds must be finite"),
    (lambda sim: sim.thruster_manager.set_config(positions=1, directions=1),
     "positions must be a list"),
    (lambda sim: sim.rw_manager.set_config(spin_axes=1), "spin_axes must be a list"),
    (lambda sim: sim.randomizer.set_bounds(propellant=1.0), "randomization propellant: expected"),
    (lambda sim: sim.vehicle.set_config(inertia="eye"), "inertia must be a 3x3 matrix"),
    (lambda sim: sim.reset_manager.set_bounds(att_nominal=(0.0, 0.0, 1.0)),
     "att_nominal must have 4 components"),
])
def test_setters_reject_bad_values_by_name(make_sim, call, match):
    """A ValueError naming the argument; a fifth wheel value used to be
    dropped silently, and a scalar layout raised TypeError."""
    with pytest.raises(ValueError, match=match):
        call(_sim(make_sim))


def test_termination_reasons(make_sim):
    sim = _sim(make_sim, num_envs=2, num_bodies=2, check_bounds=True, pos_limit=1.0e7)
    set_state(sim, pos=[[2.0e7, 0.0, 0.0]] + [[7.0e6, 0.0, 0.0]] * 3)
    sim.step()
    assert sim.termination_checker.get_termination_reasons() == {0: 1}   # per environment
    single = _sim(make_sim, num_envs=2, check_bounds=True, pos_limit=1.0e7)
    set_state(single, pos=[[7.0e6, 0.0, 0.0], [2.0e7, 0.0, 0.0]])
    single.step()
    assert single.termination_checker.get_termination_reasons() == {1: 1}


@pytest.mark.parametrize("path", sorted(Path(__file__).parents[1].glob("examples/*.yaml")),
                         ids=lambda p: p.name)
def test_shipped_yaml_configs_load_and_step(device, path):
    sim = create_simulator_from_yaml(str(path), device=device)
    sim.simulate(5)
    assert np.isfinite(sim.states.numpy()["pos"]).all()


def test_actuator_config_per_environment(make_sim):
    """Thruster and wheel configs take env_ids, like vehicle.set_config."""
    sim = _sim(make_sim)
    sim.thruster_manager.set_config(max_thrust=5.0, env_ids=[1])
    sim.rw_manager.set_config(max_torque=0.3, env_ids=[2])
    np.testing.assert_array_equal(sim.thruster_manager.configs.numpy()["max_thrust"], [10.0, 5.0, 10.0])
    np.testing.assert_array_equal(sim.rw_manager.configs.numpy()["max_torque"], [0.1, 0.1, 0.3])


def test_reset_reseed_repeats_the_draw(make_sim):
    sim = make_sim(num_envs=8)
    sim.reset_manager.reseed(7)
    sim.reset()
    first = get_state(sim)["pos"]
    sim.reset()
    assert not np.array_equal(get_state(sim)["pos"], first)
    sim.reset_manager.reseed(7)
    sim.reset()
    np.testing.assert_array_equal(get_state(sim)["pos"], first)


@pytest.mark.parametrize("kwargs,match", [
    ({"integrator": "leapfrog"}, "integrator must be"),
    ({"integrator": 4}, "integrator must be"),
    ({"gravity": "false"}, "gravity must be True or False"),
    ({"drag": 1}, "drag must be True or False"),
    ({"j2_perturbation": "yes"}, "j2_perturbation must be True or False"),
    ({"reset_config": {"pos_bound": 1.0}}, r"Unknown key\(s\) in reset_config: \['pos_bound'\]"),
    ({"reset_config": [("seed", 1)]}, "reset_config must be a dict"),
    ({"device": "tpu"}, "device 'tpu'"),
])
def test_constructor_rejects_bad_settings(make_sim, kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_sim(**kwargs)


def test_thrust_commands_fill_eight_slots(make_sim):
    """One command of up to 8 throttles, omitted trailing slots zero; the
    per-entity array has one column per slot."""
    sim = _sim(make_sim)
    sim.set_thrust_commands([0.1, 0.2, 0.3, 0.4, 0.5])
    cmd = sim.actuators.commands.numpy()
    np.testing.assert_array_equal(cmd["thrust_cmd"], [[0.1, 0.2, 0.3, 0.4]] * 3)
    np.testing.assert_array_equal(cmd["thrust_cmd_ext"], [[0.5, 0.0, 0.0, 0.0]] * 3)
    per_entity = np.arange(24.0).reshape(3, 8) / 24.0
    sim.set_thrust_commands_per_entity(wp.array(per_entity, dtype=wp.float64, device=sim.device))
    cmd = sim.actuators.commands.numpy()
    np.testing.assert_array_equal(np.hstack([cmd["thrust_cmd"], cmd["thrust_cmd_ext"]]),
                                  per_entity)


def test_rw_angular_momenta_getter(make_sim):
    """Wheel momenta read back per entity, into a reused buffer."""
    sim = _sim(make_sim)
    sim.rw_manager.set_config(max_momentum=1.0)
    h = np.arange(12.0).reshape(3, 4) / 20.0
    sim.set_rw_angular_momentum_state_per_entity(wp.array(h, dtype=wp.float64, device=sim.device))
    out = sim.get_rw_angular_momenta()
    assert out.dtype == wp.vec4d and out.shape == (3,)
    np.testing.assert_array_equal(out.numpy(), h)
    assert sim.get_rw_angular_momenta() is out


def test_reset_bounds_read_back(make_sim):
    """get_bounds returns one body's bounds keyed by the setters' argument
    names, so they round-trip; it is a copy."""
    sim = make_sim(num_envs=2, num_bodies=2)
    rm = sim.reset_manager
    rm.set_bounds(mass_bounds=(1.0, 2.0), mom_bounds=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                  body_ids=[1])
    rm.set_orbit_bounds(altitude_bounds=(4.0e5, 5.0e5), mean_anomaly_bounds=(0.0, 1.0),
                        body_ids=[1])
    b0, b1 = rm.get_bounds(0), rm.get_bounds(1)
    assert b0["sample_mode"] == "cartesian" and "vel_bounds" in b0 and "true_anomaly_bounds" in b0
    assert b1["sample_mode"] == "orbit" and "mom_bounds" in b1 and "vel_bounds" not in b1
    assert b1["mass_bounds"] == (1.0, 2.0)
    assert b1["semi_major_axis_bounds"] == (RE_EARTH + 4.0e5, RE_EARTH + 5.0e5)
    assert b1["mean_anomaly_bounds"] == (0.0, 1.0) and "true_anomaly_bounds" not in b1

    cartesian = set(inspect.signature(ResetManager.set_bounds).parameters)
    copy = make_sim(num_envs=2, num_bodies=2).reset_manager
    copy.set_bounds(**{k: v for k, v in b1.items() if k in cartesian}, body_ids=[1])
    copy.set_orbit_bounds(**{k: v for k, v in b1.items()
                             if k not in cartesian and k != "sample_mode"}, body_ids=[1])
    assert copy.get_bounds(1) == b1

    b1["mass_bounds"] = (5.0, 6.0)
    assert rm.get_bounds(1)["mass_bounds"] == (1.0, 2.0)
    with pytest.raises(ValueError, match="body_id entry 2 out of range"):
        rm.get_bounds(2)


@pytest.mark.parametrize("name", [
    "num_envs", "num_bodies", "num_entities", "dt", "device", "differentiable", "states",
    "vehicle", "thruster_manager", "rw_manager", "reset_manager", "randomizer",
    "termination_checker", "actuators",
])
def test_simulator_attributes_are_read_only(make_sim, name):
    """Kernels, graphs and the managers hold these arrays and values: a
    rebound attribute would silently be ignored (or desynchronize dt)."""
    sim = make_sim(num_envs=2)
    with pytest.raises(AttributeError):
        setattr(sim, name, getattr(sim, name))


def test_arrays_on_another_device_rejected(make_sim, device):
    """Warp arrays on another device are rejected by name before any launch,
    so the simulator (and its CUDA context) keeps working."""
    if not wp.is_cuda_available():
        pytest.skip("needs a CUDA device next to the CPU")
    other = "cpu" if "cuda" in str(device) else "cuda"
    sim = _sim(make_sim)
    set_state(sim, pos=[[7.0e6, 0.0, 0.0]] * 3)
    before = get_state(sim)
    with pytest.raises(ValueError, match=f"masses is on {other}.*, the simulator on"):
        sim.reset_to_conditions(masses=wp.ones(3, dtype=wp.float64, device=other))
    with pytest.raises(ValueError, match=f"cd is on {other}.*, the simulator on"):
        sim.vehicle.set_config_per_entity(cd=wp.ones(3, dtype=wp.float64, device=other))
    with pytest.raises(ValueError, match=r"thrust_cmds must be a float64 Warp array of shape \(3, 8\)"):
        sim.set_thrust_commands_per_entity(wp.zeros((3, 8), dtype=wp.float64, device=other))
    for key, value in get_state(sim).items():
        np.testing.assert_array_equal(value, before[key], err_msg=key)
    np.testing.assert_array_equal(sim.vehicle.configs.numpy()["cd"], 2.2)
    sim.step()
    assert np.isfinite(get_state(sim)["pos"]).all()


@pytest.mark.parametrize("call, match", [
    (lambda sim: sim.simulate(0), "num_steps must be a positive integer"),
    (lambda sim: sim.simulate(2.0), "num_steps must be a positive integer"),
    (lambda sim: sim.simulate(2, save_states=1), "save_states must be True or False"),
    (lambda sim: sim.simulate(2, auto_reset_terminated="yes"),
     "auto_reset_terminated must be True or False"),
    (lambda sim: sim.simulate_monte_carlo(-1), "num_steps must be a positive integer"),
])
def test_simulate_rejects_before_acting(make_sim, call, match):
    """A rejected call steps nothing: state, step_count and episode counters
    stay as they were."""
    sim = _sim(make_sim)
    set_state(sim, pos=[[7.0e6, 0.0, 0.0]] * 3, vel=[[0.0, 1.0, 0.0]] * 3)
    sim.step()
    before, count = get_state(sim), sim.step_count
    with pytest.raises(ValueError, match=match):
        call(sim)
    assert sim.step_count == count
    np.testing.assert_array_equal(sim.get_episode_steps().numpy(), 1)
    for key, value in get_state(sim).items():
        np.testing.assert_array_equal(value, before[key], err_msg=key)


def test_time_only_alongside_a_recording(make_sim):
    """simulate() allocates nothing without a recording; with one, time[k]
    is the epoch of row k."""
    sim = _sim(make_sim)
    set_state(sim, pos=[[7.0e6, 0.0, 0.0]] * 3)
    assert sim.simulate(3) == {}
    results = sim.simulate(3, save_rw_states=True)
    assert set(results) == {"time", "rw_angular_velocities"}
    np.testing.assert_array_equal(results["time"].numpy(), (np.arange(3) + 1) * 0.1)

"""Thruster configuration and wrench tests."""
import numpy as np
import pytest
import warp as wp

from cosmic_warp.actuators.base import ActuatorCommand
from cosmic_warp.actuators.thrusters import (
    G0_STD,
    MAX_THRUSTERS,
    ThrusterConfig,
    ThrusterManager,
    thruster_wrench,
)
from cosmic_warp.core.state import State
from cosmic_warp.core.vehicle import VehicleConfig


@wp.kernel
def _wrench_kernel(
    configs: wp.array(dtype=ThrusterConfig),
    cmds: wp.array(dtype=ActuatorCommand),
    states: wp.array(dtype=State),
    vehicles: wp.array(dtype=VehicleConfig),
    out: wp.array(dtype=wp.float64, ndim=2),
):
    tid = wp.tid()
    w = thruster_wrench(configs[tid], cmds[tid], states[tid], vehicles[tid], wp.float64(1.0e-3))
    out[tid, 0] = w.force[0]
    out[tid, 1] = w.force[1]
    out[tid, 2] = w.force[2]
    out[tid, 3] = w.mdot


def _wrench(device, directions, thrust_cmd, max_thrust=10.0, isp=200.0):
    """Configure one entity through ThrusterManager and evaluate the wrench."""
    mgr = ThrusterManager(num_envs=1, device=device)
    mgr.set_config(max_thrust=max_thrust, isp=isp,
                   positions=[(0.0, 0.0, 0.0)] * len(directions),
                   directions=directions)

    cmds = wp.zeros(1, dtype=ActuatorCommand, device=device)
    u = cmds.numpy()
    u["thrust_cmd"][0] = thrust_cmd[:4]
    u["thrust_cmd_ext"][0] = thrust_cmd[4:]
    cmds.assign(u)

    states = wp.zeros(1, dtype=State, device=device)
    s = states.numpy()
    s["mass"][:] = 5.0
    states.assign(s)

    vehicles = wp.zeros(1, dtype=VehicleConfig, device=device)
    v = vehicles.numpy()
    v["mass_dry"][:] = 1.0
    vehicles.assign(v)

    out = wp.zeros((1, 4), dtype=wp.float64, device=device)
    wp.launch(_wrench_kernel, dim=1,
              inputs=[mgr.configs, cmds, states, vehicles, out], device=device)
    r = out.numpy()[0]
    return r[:3], r[3]


@pytest.mark.parametrize("direction", [
    (2.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.5, 0.0, 0.0),
    (-3.0, 4.0, 0.0),
])
def test_directions_are_normalized(device, direction):
    """A non-unit direction must not scale thrust: max_thrust sets magnitude."""
    mgr = ThrusterManager(num_envs=1, device=device)
    mgr.set_config(positions=[(0.0, 0.0, 0.0)], directions=[direction])

    stored = mgr.configs.numpy()["directions"][0][0]
    assert np.linalg.norm(stored) == pytest.approx(1.0, abs=1e-12)
    # same heading as requested
    expected = np.asarray(direction, float)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(stored, expected, atol=1e-12)


def test_thrust_magnitude_independent_of_direction_scale(device):
    """|F| depends on max_thrust and the command only, never on |direction|."""
    cmd = [1.0] + [0.0] * 7
    ref, ref_mdot = _wrench(device, [(1.0, 0.0, 0.0)], cmd)
    for scale in (0.5, 2.0, 10.0):
        f, mdot = _wrench(device, [(scale, 0.0, 0.0)], cmd)
        np.testing.assert_allclose(f, ref, rtol=1e-14)
        assert mdot == pytest.approx(ref_mdot, rel=1e-14)


def test_effective_isp_matches_configured_isp(device):
    """|F| / (mdot * g0) must equal the configured Isp for any direction.

    Before directions were normalized, a non-unit direction scaled force but
    not mass flow, silently multiplying Isp by |direction|.
    """
    cmd = [1.0] + [0.0] * 7
    for direction in [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 1.0, 0.0)]:
        f, mdot = _wrench(device, [direction], cmd, max_thrust=10.0, isp=200.0)
        assert np.linalg.norm(f) / (mdot * G0_STD) == pytest.approx(200.0, rel=1e-12)


def test_zero_direction_slot_is_inert(device):
    """An unused slot produces no force and burns no propellant."""
    mgr_dirs = [(1.0, 0.0, 0.0)]            # slots 1-7 pad to (0,0,0)

    real = [1.0] + [0.0] * 7
    inert = [0.0, 1.0] + [0.0] * 6          # command the padded slot 1
    both = [1.0] * MAX_THRUSTERS

    f_real, mdot_real = _wrench(device, mgr_dirs, real)
    f_inert, mdot_inert = _wrench(device, mgr_dirs, inert)
    f_both, mdot_both = _wrench(device, mgr_dirs, both)

    assert np.linalg.norm(f_inert) == 0.0
    assert mdot_inert == 0.0
    # commanding every slot costs exactly the one real thruster
    np.testing.assert_allclose(f_both, f_real, rtol=1e-14)
    assert mdot_both == pytest.approx(mdot_real, rel=1e-14)


def test_degenerate_direction_raises(device):
    """Non-zero but unnormalizable directions are an error, not a guess."""
    mgr = ThrusterManager(num_envs=1, device=device)
    with pytest.raises(ValueError, match="too small to normalize"):
        mgr.set_config(positions=[(1.0, 0.0, 0.0)],
                       directions=[(1e-20, 0.0, 0.0)])


def test_default_layout_is_unit_and_padded(device):
    """A new manager holds the default layout: six unit directions and two
    zero slots."""
    mgr = ThrusterManager(num_envs=1, device=device)

    dirs = mgr.configs.numpy()["directions"][0]
    norms = np.linalg.norm(dirs, axis=1)
    np.testing.assert_allclose(norms[:6], 1.0, atol=1e-12)
    np.testing.assert_allclose(norms[6:], 0.0, atol=1e-12)


@pytest.mark.parametrize("kwargs, match", [
    ({"positions": [(1.0, 0.0, 0.0)] * 9, "directions": [(1.0, 0.0, 0.0)] * 9},
     "at most 8 thrusters"),
    ({"positions": [(1.0, 0.0, 0.0)] * 3, "directions": [(1.0, 0.0, 0.0)] * 2},
     "same length"),
    ({"positions": [(1.0, 0.0, 0.0)], "directions": [(float("nan"), 0.0, 0.0)]},
     "must be finite"),
    ({"positions": [(1.0, 0.0)], "directions": [(1.0, 0.0, 0.0)]}, "3 components"),
    ({"max_thrust": -1.0}, "max_thrust must be >= 0"),
    ({"isp": -10.0}, "isp must be >= 0"),
])
def test_invalid_config_raises(device, kwargs, match):
    """Malformed layouts and parameters are rejected, never truncated or padded."""
    mgr = ThrusterManager(num_envs=1, device=device)
    with pytest.raises(ValueError, match=match):
        mgr.set_config(**kwargs)

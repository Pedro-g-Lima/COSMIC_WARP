"""
Minimal RL task for the VecEnv, adapter and termination tests.

The package ships the ``Task`` protocol, not tasks (the reference attitude
task lives in ``examples/``), so the tests bring their own: per-body wheel
torque actions, a ``[quat(4), angvel(3), rw_ang_mom(4)]`` observation and the
detumble reward ``-(w_rate*|omega|^2 + w_ctrl*|action|^2)``. Observation and
reward equal ``examples/attitude_task.AttitudeTask`` with ``w_att = 0``
(``tests/test_example_attitude_task.py``), so the PPO learning tests train on
the same problem as examples 14-16.
"""
import warp as wp

from cosmic_warp.core.state import State
from cosmic_warp.core.vehicle import VehicleConfig
from cosmic_warp.actuators.base import ActuatorCommand, ActuatorState
from cosmic_warp.actuators.reaction_wheels import ReactionWheelConfig

ACT_PER_BODY = wp.constant(3)
OBS_PER_BODY = wp.constant(11)


class DetumbleTask:
    """Wheel-torque detumble task (actions scaled by ``max_torque``)."""

    def __init__(self, w_rate: float = 1.0, w_ctrl: float = 1.0e-3):
        self.w_rate = float(w_rate)
        self.w_ctrl = float(w_ctrl)

    def bind(self, env) -> None:
        self._sim = env.sim
        self.act_dim = env.sim.num_bodies * ACT_PER_BODY
        self.obs_dim = env.sim.num_bodies * OBS_PER_BODY

    def ingest_actions(self, action: wp.array) -> None:
        sim = self._sim
        wp.launch(_ingest_kernel, dim=sim.num_entities,
                  inputs=[action, sim.rw_manager.configs, sim.actuators.commands,
                          sim.num_bodies],
                  device=sim.device)

    def write_obs(self, obs: wp.array, reset_env: wp.array, masked: int) -> None:
        sim = self._sim
        wp.launch(_obs_kernel, dim=sim.num_envs,
                  inputs=[sim.states, sim.vehicle.configs, sim.actuators.states,
                          sim.num_bodies, reset_env, int(masked), obs],
                  device=sim.device)

    def write_reward(self, reward: wp.array, action: wp.array) -> None:
        sim = self._sim
        wp.launch(_reward_kernel, dim=sim.num_envs,
                  inputs=[sim.states, sim.vehicle.configs, action, sim.num_bodies,
                          wp.float64(self.w_rate), wp.float64(self.w_ctrl), reward],
                  device=sim.device)


@wp.func
def _unit_clamp(a: wp.float32) -> wp.float64:
    """Clamp to [-1, 1], passing a NaN or Inf on to the commands."""
    x = wp.float64(a)
    if wp.isfinite(x):
        x = wp.clamp(x, wp.float64(-1.0), wp.float64(1.0))
    return x


@wp.kernel
def _ingest_kernel(
    action: wp.array(dtype=wp.float32, ndim=2),
    rw_configs: wp.array(dtype=ReactionWheelConfig),
    commands: wp.array(dtype=ActuatorCommand),
    num_bodies: int,
):
    e = wp.tid()
    env = e // num_bodies
    base = (e - env * num_bodies) * ACT_PER_BODY
    max_torque = rw_configs[e].max_torque
    commands[e].rw_torque_cmd = wp.vec4d(
        _unit_clamp(action[env, base + 0]) * max_torque,
        _unit_clamp(action[env, base + 1]) * max_torque,
        _unit_clamp(action[env, base + 2]) * max_torque,
        wp.float64(0.0))


@wp.kernel
def _obs_kernel(
    states: wp.array(dtype=State),
    vehicles: wp.array(dtype=VehicleConfig),
    act_states: wp.array(dtype=ActuatorState),
    num_bodies: int,
    reset_env: wp.array(dtype=wp.bool),
    masked: int,
    obs: wp.array(dtype=wp.float32, ndim=2),
):
    env = wp.tid()
    if masked == 1 and (not reset_env[env]):
        return
    for b in range(num_bodies):
        e = env * num_bodies + b
        base = b * OBS_PER_BODY
        q = states[e].quat
        # q_w >= 0 hemisphere, as in the example task
        s = wp.float64(1.0)
        if q[3] < wp.float64(0.0):
            s = wp.float64(-1.0)
        omega = wp.mul(vehicles[e].inertia_inv, states[e].ang_mom)
        h = act_states[e].rw_ang_mom
        for k in range(4):
            obs[env, base + k] = wp.float32(s * q[k])
            obs[env, base + 7 + k] = wp.float32(h[k])
        for k in range(3):
            obs[env, base + 4 + k] = wp.float32(omega[k])


@wp.kernel
def _reward_kernel(
    states: wp.array(dtype=State),
    vehicles: wp.array(dtype=VehicleConfig),
    action: wp.array(dtype=wp.float32, ndim=2),
    num_bodies: int,
    w_rate: wp.float64,
    w_ctrl: wp.float64,
    reward: wp.array(dtype=wp.float32),
):
    env = wp.tid()
    total = wp.float64(0.0)
    for b in range(num_bodies):
        e = env * num_bodies + b
        omega = wp.mul(vehicles[e].inertia_inv, states[e].ang_mom)
        ctrl_sq = wp.float64(0.0)
        for k in range(ACT_PER_BODY):
            a = wp.float64(action[env, b * ACT_PER_BODY + k])
            ctrl_sq += a * a
        total += w_rate * wp.dot(omega, omega) + w_ctrl * ctrl_sq
    reward[env] = wp.float32(-total)

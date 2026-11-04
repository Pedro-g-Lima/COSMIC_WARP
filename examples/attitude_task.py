"""Reaction-wheel attitude task

The reference RL task shared by examples 13-16: reaction-wheel attitude
control (detumble, optionally repoint to the identity attitude). It is a
module imported by those scripts, not a script to run.

A task supplies the three environment-specific pieces of an RL problem (the
`cosmic_warp.rl.Task` protocol): the action mapping (normalized policy output
to actuator commands), the observation packing, and the reward. Termination
and truncation are handled generically by `cosmic_warp.rl.VecEnv`, from the
simulator's own termination checker and episode-step counters.

- **Action** `[N, 3 * num_bodies]` in [-1, 1]: torque commands for wheels
  0-2 of each body, scaled by the wheel assembly's `max_torque`.
- **Observation** `[N, 11 * num_bodies]`: per body the quaternion (4, sign
  fixed so that `q_w >= 0`), body angular velocity (3) and wheel angular
  momenta (4), in raw units.
- **Reward**: `-(w_att * angle^2 + w_rate * |ω|^2 + w_ctrl * |action|^2)`
  summed over bodies, where `angle` is the attitude error from the identity
  quaternion; always <= 0. With `w_att = 0` the task is pure detumbling.

All packing and reward math runs in fused Warp kernels on the device;
nothing touches the host. Copy and adapt it for your own task: the package
ships the environment, not tasks. The scripts import it from their own
directory:

```python
from attitude_task import AttitudeTask   # examples/ is the script's directory

env = VecEnv(sim, AttitudeTask(w_att=0.0, w_rate=1.0), max_episode_steps=200)
```
"""
import warnings

import warp as wp

from cosmic_warp.core.state import State
from cosmic_warp.core.vehicle import VehicleConfig
from cosmic_warp.actuators.base import ActuatorCommand, ActuatorState
from cosmic_warp.actuators.reaction_wheels import ReactionWheelConfig


class AttitudeTask:
    """Reaction-wheel attitude control (detumble / repoint to identity).

    - **Action** `[N, 3*num_bodies]` in `[-1, 1]`: per-body torque commands
      for wheels 0-2, each scaled by the assembly's `max_torque`. With the
      default orthogonal wheel layout these are torques about body x/y/z; a
      custom `spin_axes` layout is commanded along its own axes, and a
      configured 4th wheel is observed but never commanded (`bind` warns).
    - **Observation** `[N, 11*num_bodies]`: per body `[quat(4), angvel(3),
      rw_ang_mom(4)]` (raw units; normalization is left to the trainer).
    - **Reward**: `-(w_att * angle^2 + w_rate * |omega|^2 + w_ctrl *
      |action|^2)` summed over bodies (`action` unclamped: outputs beyond
      +-1 are penalized but not applied), where `angle` is the attitude error
      from the identity quaternion. Rewards are dense and always <= 0.
    """

    ACT_PER_BODY = 3
    OBS_PER_BODY = 11

    def __init__(self, w_att: float = 1.0, w_rate: float = 0.1, w_ctrl: float = 0.01):
        self.w_att = float(w_att)
        self.w_rate = float(w_rate)
        self.w_ctrl = float(w_ctrl)

    def bind(self, env) -> None:
        self._sim = env.sim
        self._num_bodies = env.sim.num_bodies
        self.act_dim = env.sim.num_bodies * self.ACT_PER_BODY
        self.obs_dim = env.sim.num_bodies * self.OBS_PER_BODY
        if (env.sim.rw_manager.configs.numpy()["spin_axis_3"] != 0.0).any():
            warnings.warn(
                "AttitudeTask commands only wheels 0-2; the configured 4th "
                "reaction wheel is observed but never commanded.",
                stacklevel=2,
            )

    def ingest_actions(self, action: wp.array) -> None:
        sim = self._sim
        wp.launch(
            kernel=_attitude_ingest_kernel,
            dim=sim.num_entities,
            inputs=[
                action,
                sim.rw_manager.configs,
                sim.actuators.commands,
                self._num_bodies,
                self.ACT_PER_BODY,
            ],
            device=sim.device,
        )

    def write_obs(self, obs: wp.array, reset_env: wp.array, masked: int) -> None:
        sim = self._sim
        wp.launch(
            kernel=_attitude_obs_kernel,
            dim=sim.num_envs,
            inputs=[
                sim.states,
                sim.vehicle.configs,
                sim.actuators.states,
                self._num_bodies,
                self.OBS_PER_BODY,
                reset_env,
                int(masked),
                obs,
            ],
            device=sim.device,
        )

    def write_reward(self, reward: wp.array, action: wp.array) -> None:
        sim = self._sim
        wp.launch(
            kernel=_attitude_reward_kernel,
            dim=sim.num_envs,
            inputs=[
                sim.states,
                sim.vehicle.configs,
                action,
                self._num_bodies,
                self.ACT_PER_BODY,
                wp.float64(self.w_att),
                wp.float64(self.w_rate),
                wp.float64(self.w_ctrl),
                reward,
            ],
            device=sim.device,
        )


@wp.func
def _unit_clamp(a: wp.float32) -> wp.float64:
    """An action clamped to [-1, 1], except a NaN or Inf, which is passed on:
    wp.clamp would turn it into a finite command, hiding a diverged policy
    from the simulator's non-finite check (reason 6)."""
    x = wp.float64(a)
    if wp.isfinite(x):
        x = wp.clamp(x, wp.float64(-1.0), wp.float64(1.0))
    return x


@wp.kernel
def _attitude_ingest_kernel(
    action: wp.array(dtype=wp.float32, ndim=2),   # [N, act_dim]
    rw_configs: wp.array(dtype=ReactionWheelConfig),
    commands: wp.array(dtype=ActuatorCommand),
    num_bodies: int,
    act_per_body: int,
):
    """Map normalized per-body actions to clamped, scaled RW torque commands
    (non-finite actions pass through, so they terminate the environment)."""
    e = wp.tid()
    env = e // num_bodies
    body = e - env * num_bodies
    base = body * act_per_body

    max_torque = rw_configs[e].max_torque
    tx = _unit_clamp(action[env, base + 0]) * max_torque
    ty = _unit_clamp(action[env, base + 1]) * max_torque
    tz = _unit_clamp(action[env, base + 2]) * max_torque

    commands[e].rw_torque_cmd = wp.vec4d(tx, ty, tz, wp.float64(0.0))


@wp.kernel
def _attitude_obs_kernel(
    states: wp.array(dtype=State),
    vehicles: wp.array(dtype=VehicleConfig),
    act_states: wp.array(dtype=ActuatorState),
    num_bodies: int,
    obs_per_body: int,
    reset_env: wp.array(dtype=wp.bool),
    masked: int,
    obs: wp.array(dtype=wp.float32, ndim=2),      # [N, obs_dim]
):
    """Pack [quat(4), angvel(3), rw_ang_mom(4)] per body into the obs row."""
    env = wp.tid()
    if masked == 1 and (not reset_env[env]):
        return

    for b in range(num_bodies):
        e = env * num_bodies + b
        base = b * obs_per_body

        q = states[e].quat
        # Canonicalize to the q_w >= 0 hemisphere: q and -q are the same
        # attitude (quaternion double cover), so flipping sign gives the policy
        # a continuous observation instead of a jump as q_w crosses zero.
        s = wp.float64(1.0)
        if q[3] < wp.float64(0.0):
            s = wp.float64(-1.0)
        omega = wp.mul(vehicles[e].inertia_inv, states[e].ang_mom)
        h = act_states[e].rw_ang_mom

        obs[env, base + 0] = wp.float32(s * q[0])
        obs[env, base + 1] = wp.float32(s * q[1])
        obs[env, base + 2] = wp.float32(s * q[2])
        obs[env, base + 3] = wp.float32(s * q[3])
        obs[env, base + 4] = wp.float32(omega[0])
        obs[env, base + 5] = wp.float32(omega[1])
        obs[env, base + 6] = wp.float32(omega[2])
        obs[env, base + 7] = wp.float32(h[0])
        obs[env, base + 8] = wp.float32(h[1])
        obs[env, base + 9] = wp.float32(h[2])
        obs[env, base + 10] = wp.float32(h[3])


@wp.kernel
def _attitude_reward_kernel(
    states: wp.array(dtype=State),
    vehicles: wp.array(dtype=VehicleConfig),
    action: wp.array(dtype=wp.float32, ndim=2),   # [N, act_dim]
    num_bodies: int,
    act_per_body: int,
    w_att: wp.float64,
    w_rate: wp.float64,
    w_ctrl: wp.float64,
    reward: wp.array(dtype=wp.float32),           # [N]
):
    """Dense attitude-regulation reward (<= 0), summed over an env's bodies."""
    env = wp.tid()
    total = wp.float64(0.0)

    for b in range(num_bodies):
        e = env * num_bodies + b

        # Attitude error from identity: angle = 2*acos(|q_w|) in [0, pi]
        q = wp.normalize(states[e].quat)
        qw = wp.min(wp.abs(q[3]), wp.float64(1.0))
        angle = wp.float64(2.0) * wp.acos(qw)

        omega = wp.mul(vehicles[e].inertia_inv, states[e].ang_mom)
        rate_sq = wp.dot(omega, omega)

        ctrl_sq = wp.float64(0.0)
        for k in range(act_per_body):
            a = wp.float64(action[env, b * act_per_body + k])
            ctrl_sq += a * a

        total += w_att * angle * angle + w_rate * rate_sq + w_ctrl * ctrl_sq

    reward[env] = wp.float32(-total)

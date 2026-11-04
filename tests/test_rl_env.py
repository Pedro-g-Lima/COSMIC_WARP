"""
Tests for the RL vectorized environment (cosmic_warp.rl).

Validate the step contract: observation packing matches the simulator's own
getters, truncation fires exactly at the time limit, terminated environments
auto-reset, the terminal observation is the pre-reset state, and actions map
to actuator torques. Everything runs on device; nothing here reads state
mid-step. The task is the test-local ``DetumbleTask`` (tests/rl_tasks.py); the
example ``AttitudeTask`` has its own tests (test_example_attitude_task.py).
"""
import numpy as np
import pytest
import warp as wp

from cosmic_warp.core.pipeline import SpacecraftSimulator
from cosmic_warp.rl import VecEnv
from rl_tasks import DetumbleTask


def _make_sim(device, num_envs=16, num_bodies=1, max_ang_mom=1.0e8,
              check_bounds=False, seed=42):
    sim = SpacecraftSimulator(
        num_envs=num_envs,
        num_bodies=num_bodies,
        integrator="symplectic",
        dt=0.1,
        device=device,
        gravity=True,
        j2_perturbation=False,
        gravity_gradient=False,
        check_bounds=check_bounds,
        ang_mom_limit=max_ang_mom,
        reset_config={"seed": seed},
    )
    sim.rw_manager.set_config(
        max_torque=0.02, max_momentum=0.5, wheel_inertia=0.005, viscous_damping=1.0e-5
    )
    return sim


def _zero_action(env):
    return wp.zeros((env.num_envs, env.act_dim), dtype=wp.float32, device=env.device)


def test_dims(device):
    """obs/act dimensions scale with the number of bodies."""
    env1 = VecEnv(_make_sim(device, num_bodies=1), DetumbleTask(), max_episode_steps=100)
    assert env1.act_dim == 3 and env1.obs_dim == 11
    env2 = VecEnv(_make_sim(device, num_bodies=2), DetumbleTask(), max_episode_steps=100)
    assert env2.act_dim == 6 and env2.obs_dim == 22


def test_obs_matches_getters(device):
    """Packed observation equals the simulator's own state getters (no reset)."""
    sim = _make_sim(device, check_bounds=False)
    env = VecEnv(sim, DetumbleTask(), max_episode_steps=10**9)
    env.reset()
    env.step(_zero_action(env))  # one step, far from the time limit -> no reset

    obs = env.obs.numpy()
    quat = sim.get_quaternions().numpy().reshape(sim.num_entities, 4)
    angvel = sim.get_angular_velocities().numpy().reshape(sim.num_entities, 3)

    # The obs canonicalizes the quaternion to the q_w >= 0 hemisphere.
    quat = quat * np.where(quat[:, 3:4] < 0.0, -1.0, 1.0)
    # float32 casts of the float64 state: half an ulp (measured 2.9e-8)
    np.testing.assert_allclose(obs[:, 0:4], quat, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(obs[:, 4:7], angvel, rtol=1e-6, atol=1e-7)
    # No wheel command was issued, so wheel momenta stay zero.
    np.testing.assert_allclose(obs[:, 7:11], 0.0, atol=1e-6)


def test_reward_non_positive(device):
    """The detumble reward is a penalty: always <= 0."""
    env = VecEnv(_make_sim(device), DetumbleTask(), max_episode_steps=10**9)
    env.reset()
    _, reward, _, _ = env.step(_zero_action(env))
    assert (reward.numpy() <= 1e-6).all()


def test_reward_matches_formula(device):
    """The reward is computed on the post-step state (the returned obs) and
    the ingested action: -(w_rate*|omega|^2 + w_ctrl*|a|^2)."""
    task = DetumbleTask(w_rate=0.5, w_ctrl=0.1)
    env = VecEnv(_make_sim(device), task, max_episode_steps=10**9)
    env.reset()
    a = 0.25
    act = wp.full((env.num_envs, env.act_dim), a, dtype=wp.float32, device=device)
    _, reward, _, _ = env.step(act)  # no terminations: obs is the reward state

    obs = env.obs.numpy().astype(np.float64)
    rate_sq = (obs[:, 4:7] ** 2).sum(axis=1)
    expected = -(0.5 * rate_sq + 0.1 * env.act_dim * a**2)
    np.testing.assert_allclose(reward.numpy(), expected, rtol=1e-6, atol=1e-7)  # float32


def test_truncation_timing(device):
    """truncated flips to True exactly on the step that hits max_episode_steps."""
    K = 8
    env = VecEnv(_make_sim(device, check_bounds=False), DetumbleTask(), max_episode_steps=K)
    env.reset()
    act = _zero_action(env)

    for _ in range(K - 1):
        _, _, _, truncated = env.step(act)
    assert not truncated.numpy().any()  # still before the limit

    _, _, terminated, truncated = env.step(act)  # step K
    assert truncated.numpy().all()
    assert not terminated.numpy().any()
    # Auto-reset zeroed the per-entity step counters.
    assert (env.sim.episode_steps.numpy() == 0).all()


def test_terminal_obs_is_pre_reset_state(device):
    """terminal_obs at truncation equals a non-truncated reference at that step."""
    K = 10
    seed = 7
    env = VecEnv(_make_sim(device, check_bounds=False, seed=seed),
                 DetumbleTask(), max_episode_steps=K)
    ref = VecEnv(_make_sim(device, check_bounds=False, seed=seed),
                 DetumbleTask(), max_episode_steps=10**9)
    env.reset()
    ref.reset()
    act = _zero_action(env)

    for _ in range(K):
        env.step(act)
        ref.step(act)  # identical initial state + actions, never truncates

    terminal = env.terminal_obs.numpy()
    reference = ref.obs.numpy()  # ref did not reset, so obs is the pre-reset state
    # Same initial state, actions and kernels: bitwise identical
    np.testing.assert_array_equal(terminal, reference)

    # The env's live obs, by contrast, is the fresh post-reset episode.
    assert not np.allclose(env.obs.numpy(), terminal, atol=1e-4)


def test_termination_wins_over_truncation(device):
    """When an env terminates on the same step it hits the time limit,
    terminated is True and truncated stays False (true terminal for
    bootstrapping)."""
    sim = _make_sim(device, check_bounds=True, max_ang_mom=1.0e-9)
    env = VecEnv(sim, DetumbleTask(), max_episode_steps=1)
    env.reset()
    # Step 1: any initial tumble exceeds the limit, and episode_steps hits 1.
    _, _, terminated, truncated = env.step(_zero_action(env))
    assert terminated.numpy().all()
    assert not truncated.numpy().any()


def test_reset_on_termination(device):
    """A tight angular-momentum limit terminates tumbling envs, which auto-reset."""
    sim = _make_sim(device, check_bounds=True, max_ang_mom=0.05)
    env = VecEnv(sim, DetumbleTask(), max_episode_steps=10**9)
    env.reset()
    act = _zero_action(env)

    saw_termination = False
    for _ in range(20):
        _, _, terminated, _ = env.step(act)
        if terminated.numpy().any():
            saw_termination = True
    assert saw_termination, "expected some environments to exceed the ang_mom limit"


@pytest.mark.parametrize("shape, dtype", [((8, 3), wp.float32), ((16, 2), wp.float32),
                                          ((16, 3), wp.float64)])
def test_wrong_action_rejected(device, shape, dtype):
    """step() used to copy a smaller or differently typed action silently,
    leaving part of the action buffer stale."""
    env = VecEnv(_make_sim(device), DetumbleTask(), max_episode_steps=100)
    env.reset()
    with pytest.raises(ValueError, match="action must be a float32 Warp array"):
        env.step(wp.zeros(shape, dtype=dtype, device=device))


@pytest.mark.parametrize("limit, match", [
    (0, "must be a positive integer"), (-1, "must be a positive integer"),
    (1.5, "must be a positive integer"), (2**31, "must be <= 2147483647"),
])
def test_bad_episode_limit_rejected(device, limit, match):
    """max_episode_steps = 0 truncated every step; 1.5 was silently floored;
    2**31 overflowed the int32 compare in the step kernel."""
    with pytest.raises(ValueError, match=f"max_episode_steps {match}"):
        VecEnv(_make_sim(device), DetumbleTask(), max_episode_steps=limit)
    env = VecEnv(_make_sim(device), DetumbleTask(), max_episode_steps=5)
    with pytest.raises(ValueError, match=f"max_episode_steps {match}"):
        env.max_episode_steps = limit
    assert env.max_episode_steps == 5


def test_action_scaling_spins_wheels(device):
    """A saturated +1 action drives the reaction wheels up (obs momentum grows)."""
    env = VecEnv(_make_sim(device, check_bounds=False), DetumbleTask(), max_episode_steps=10**9)
    env.reset()
    act = wp.full((env.num_envs, env.act_dim), 1.0, dtype=wp.float32, device=device)
    for _ in range(10):
        env.step(act)
    wheels = env.obs.numpy()[:, 7:10]  # wheels 0-2
    assert (wheels > 0.0).all()


def test_cuda_graph_matches_eager(device):
    """Graph replay (cuda) produces the same trajectory as eager stepping."""
    if "cuda" not in str(device):
        pytest.skip("graph capture only exercised on cuda")

    env_g = VecEnv(_make_sim(device, check_bounds=False, seed=3),
                   DetumbleTask(), max_episode_steps=10**9)
    env_e = VecEnv(_make_sim(device, check_bounds=False, seed=3),
                   DetumbleTask(), max_episode_steps=10**9)

    env_g.reset()
    env_e.reset()
    act = _zero_action(env_g)
    for _ in range(15):
        env_g.step(act)
        env_e._step_body()  # eager path, same sequence

    # Same kernels in the same order: bitwise identical
    np.testing.assert_array_equal(env_g.obs.numpy(), env_e.obs.numpy())


def test_torch_views_zero_copy(device):
    """Torch views share memory with the Warp buffers (skipped without torch)."""
    torch = pytest.importorskip("torch")
    env = VecEnv(_make_sim(device), DetumbleTask(), max_episode_steps=10**9)
    env.reset()
    obs_t = env.observations_torch()
    assert tuple(obs_t.shape) == (env.num_envs, env.obs_dim)
    # Writing through the torch action view must reach the Warp buffer.
    act_t = env.actions_torch()
    act_t.fill_(0.5)
    wp.synchronize()
    np.testing.assert_array_equal(env.actions.numpy(), 0.5)


def test_auto_reset_uses_the_latest_inertia(device):
    """A set_config(inertia=...) between steps is the nominal the auto-reset's
    inertia randomization scales. The auto-reset used to keep the nominal of
    the last sim.reset(), undoing the set_config at the first episode end."""
    sim = _make_sim(device, num_envs=4)
    sim.vehicle.set_config(inertia=np.diag([0.4, 0.4, 0.3]))
    sim.randomizer.set_bounds(inertia_scale=(1.0, 1.0))   # scale 1: inertia = nominal
    env = VecEnv(sim, DetumbleTask(), max_episode_steps=3)
    env.reset()
    act = _zero_action(env)
    env.step(act)                                          # captures the graph on CUDA

    new = np.diag([2.0, 2.0, 2.0])
    sim.vehicle.set_config(inertia=new)
    env.step(act)
    _, _, _, truncated = env.step(act)                     # step 3: auto-reset
    assert truncated.numpy().all()
    np.testing.assert_array_equal(sim.vehicle.configs.numpy()["inertia"],
                                  np.broadcast_to(new, (4, 3, 3)))


def test_max_episode_steps_setter_recaptures(device):
    """Setting the limit mid-run applies from the next step, also through
    the captured CUDA graph (which bakes the limit in)."""
    env = VecEnv(_make_sim(device), DetumbleTask(), max_episode_steps=100)
    env.reset()
    act = _zero_action(env)
    env.step(act)                                     # captures with limit 100
    env.max_episode_steps = 3
    _, _, _, truncated = env.step(act)
    assert not truncated.numpy().any()
    _, _, _, truncated = env.step(act)                # step 3 of the episode
    assert truncated.numpy().all()


def test_invalidate_graph_applies_a_task_change(device):
    """A task weight is passed by value, so the captured graph keeps the old
    one until invalidate_graph(); the eager CPU step applies it at once."""
    def rewards(new_weight, invalidate):
        task = DetumbleTask(w_ctrl=1.0e-3)
        env = VecEnv(_make_sim(device, num_envs=4, seed=5), task, max_episode_steps=100)
        env.reset()
        act = wp.full((env.num_envs, env.act_dim), 0.5, dtype=wp.float32, device=device)
        env.step(act)
        task.w_ctrl = new_weight
        if invalidate:
            env.invalidate_graph()
        return env.step(act)[1].numpy()

    old, new = rewards(1.0e-3, False), rewards(10.0, True)
    assert (new < old - 1.0).all()                    # the larger control penalty
    stale = rewards(10.0, False)
    np.testing.assert_array_equal(stale, old if "cuda" in str(device) else new)


def test_auto_reset_redraws_randomized_parameters(device):
    """Truncated environments reset with fresh randomized propellant, inside
    the captured step on CUDA."""
    sim = _make_sim(device, num_envs=32)
    sim.randomizer.set_bounds(propellant=(1.0, 2.0))
    env = VecEnv(sim, DetumbleTask(), max_episode_steps=2)
    env.reset()
    before = sim.get_masses().numpy().copy()
    for _ in range(2):
        _, _, _, truncated = env.step(_zero_action(env))
    assert truncated.numpy().all()
    after = sim.get_masses().numpy()
    assert ((after >= 2.0) & (after <= 3.0)).all()     # mass_dry 1 + propellant
    assert not np.array_equal(after, before)


def test_terminal_observations_torch_view(device):
    """The terminal-observation view is zero-copy, like the other views."""
    pytest.importorskip("torch")
    env = VecEnv(_make_sim(device), DetumbleTask(), max_episode_steps=2)
    env.reset()
    view = env.terminal_observations_torch()
    for _ in range(2):
        env.step(_zero_action(env))
    wp.synchronize()
    np.testing.assert_array_equal(view.cpu().numpy(), env.terminal_obs.numpy())
    assert view.data_ptr() == env.terminal_obs.ptr


@pytest.mark.parametrize("name", ["actions", "obs", "terminal_obs", "reward",
                                  "terminated", "truncated"])
def test_buffers_are_read_only(device, name):
    """The captured graph and the torch views hold the buffers themselves: a
    rebound attribute would silently be ignored by both."""
    env = VecEnv(_make_sim(device), DetumbleTask(), max_episode_steps=10)
    with pytest.raises(AttributeError):
        setattr(env, name, getattr(env, name))


def test_nan_action_terminates_its_environment(device):
    """The test task passes a NaN action on to the commands, so its
    environment terminates with reason 6 before the physics step (the
    terminal observation is the unchanged state); the others step on."""
    env = VecEnv(_make_sim(device, num_envs=4), DetumbleTask(), max_episode_steps=100)
    obs = env.reset().numpy().copy()
    act = np.zeros((env.num_envs, env.act_dim), dtype=np.float32)
    act[2, 1] = np.nan
    _, _, terminated, truncated = env.step(wp.array(act, dtype=wp.float32, device=device))
    np.testing.assert_array_equal(terminated.numpy(), [False, False, True, False])
    assert not truncated.numpy().any()
    np.testing.assert_array_equal(env.terminal_obs.numpy()[2], obs[2])

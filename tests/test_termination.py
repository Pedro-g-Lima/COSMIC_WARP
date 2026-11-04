"""Whole-environment termination must be independent of integration order."""

import numpy as np
import pytest
import warp as wp

from conftest import get_state, set_state
from cosmic_warp.rl.env import VecEnv
from rl_tasks import DetumbleTask


INTEGRATORS = ["euler", "rk4", "symplectic"]
DT = 0.1


@pytest.mark.parametrize("integrator", INTEGRATORS)
@pytest.mark.parametrize("offender", [0, 2])
@pytest.mark.parametrize("runner", ["step", "simulate", "monte_carlo"])
def test_terminating_step_advances_all_bodies(make_sim, integrator, offender, runner):
    sim = make_sim(num_envs=2, num_bodies=3, integrator=integrator, dt=DT,
                   gravity=False, check_bounds=True, pos_limit=1.0)
    pos = np.zeros((6, 3))
    pos[offender, 0] = 2.0
    set_state(sim, pos=pos, vel=[[0.0, 1.0, 0.0]] * 6)

    if runner == "step":
        sim.step()
    elif runner == "simulate":
        result = sim.simulate(1, save_states=True)
        np.testing.assert_array_equal(result["positions"].numpy()[0, :, 1], DT)
    else:
        result = sim.simulate_monte_carlo(1)
        np.testing.assert_array_equal(result["final_positions"].numpy()[:, 1], DT)

    np.testing.assert_array_equal(get_state(sim)["pos"][:, 1], DT)
    np.testing.assert_array_equal(sim.get_terminated().numpy()["terminated"],
                                  [True] * 3 + [False] * 3)
    np.testing.assert_array_equal(sim.get_episode_steps().numpy(), [1] * 6)
    frozen = sim.states.numpy()[:3].copy()
    frozen_wheels = sim.actuators.states.numpy()[:3].copy()

    sim.simulate(2)
    np.testing.assert_array_equal(sim.states.numpy()[:3], frozen)
    np.testing.assert_array_equal(sim.actuators.states.numpy()[:3], frozen_wheels)
    np.testing.assert_array_equal(get_state(sim)["pos"][3:, 1], 3 * DT)
    # Preserve attempted-step counting, including calls for frozen entities.
    np.testing.assert_array_equal(sim.get_episode_steps().numpy(), [3] * 6)
    assert sim.step_count == 3

    sim.reset(env_ids=[0])
    np.testing.assert_array_equal(sim.get_episode_steps().numpy(), [0] * 3 + [3] * 3)
    assert not sim.get_terminated().numpy()["terminated"].any()
    assert sim.step_count == 3
    sim.reset()
    np.testing.assert_array_equal(sim.get_episode_steps().numpy(), [0] * 6)
    assert sim.step_count == 0


@pytest.mark.parametrize("integrator", INTEGRATORS)
@pytest.mark.parametrize("reasons", [(1, 2), (2, 3), (3, 4), (4, 5), (5, 5)])
def test_simultaneous_reasons_do_not_depend_on_body_order(make_sim, integrator, reasons):
    for order in (reasons, reasons[::-1]):
        sim = make_sim(num_bodies=2, integrator=integrator, dt=DT,
                       gravity=False, check_bounds=True, check_planet_impact=True,
                       pos_limit=100.0, mom_limit=0.5, ang_mom_limit=0.2,
                       mass_limit=1.5, planet_radius=0.1)
        pos = np.tile([1.0, 0.0, 0.0], (2, 1))
        mom = np.zeros((2, 3))
        ang_mom = np.zeros((2, 3))
        mass = np.full(2, 2.0)
        for body, reason in enumerate(order):
            if reason == 1:
                pos[body, 0] = 101.0
            elif reason == 2:
                mom[body, 1] = 1.0
            elif reason == 3:
                ang_mom[body, 2] = 0.3
            elif reason == 4:
                mass[body] = 1.1
            else:
                pos[body] = 0.0
        set_state(sim, pos=pos, mom=mom, ang_mom=ang_mom, mass=mass)
        sim.step()
        status = sim.get_terminated().numpy()
        assert status["terminated"].all()
        np.testing.assert_array_equal(status["reason"], [min(reasons)] * 2)
        sim.step()
        np.testing.assert_array_equal(sim.get_terminated().numpy(), status)


@pytest.mark.parametrize("integrator", INTEGRATORS)
def test_auto_reset_stores_full_terminating_step(make_sim, integrator):
    sim = make_sim(num_envs=2, num_bodies=2, integrator=integrator, dt=DT,
                   gravity=False, check_bounds=True, pos_limit=1.0)
    sim.reset_manager.set_bounds(
        pos_bounds=((0.0,) * 3, (0.0,) * 3),
        mom_bounds=((0.0,) * 3, (0.0,) * 3),
        ang_mom_bounds=((0.0,) * 3, (0.0,) * 3),
        mass_bounds=(1.0, 1.0), att_max_angle=0.0,
    )
    set_state(sim, pos=[[2.0, 0.0, 0.0]] + [[0.0] * 3] * 3,
              vel=[[0.0, 1.0, 0.0]] * 4)
    sim.set_rw_torque_commands([0.01, 0.0, 0.0, 0.0])
    result = sim.simulate(1, save_states=True, auto_reset_terminated=True)

    np.testing.assert_array_equal(result["positions"].numpy()[0, :, 1], DT)
    np.testing.assert_array_equal(get_state(sim)["pos"][:2], 0.0)
    np.testing.assert_array_equal(get_state(sim)["pos"][2:, 1], DT)
    np.testing.assert_array_equal(sim.get_episode_steps().numpy(), [0, 0, 1, 1])
    assert not sim.get_terminated().numpy()["terminated"].any()
    np.testing.assert_array_equal(sim.actuators.states.numpy()["rw_ang_mom"][:2], 0.0)
    np.testing.assert_array_equal(sim.actuators.commands.numpy()["rw_torque_cmd"][:2], 0.0)
    np.testing.assert_array_equal(sim.actuators.states.numpy()["rw_ang_mom"][2:, 0], 0.01 * DT)


@pytest.mark.parametrize("integrator", INTEGRATORS)
@pytest.mark.parametrize("offender", [0, 1])
def test_vecenv_terminal_observation_contains_full_step(make_sim, integrator, offender):
    envs = []
    for checks in (True, False):
        sim = make_sim(num_envs=2, num_bodies=2, integrator=integrator, dt=DT,
                       gravity=False, check_bounds=checks, ang_mom_limit=0.2)
        sim.reset_manager.set_bounds(ang_mom_bounds=((0.0,) * 3, (0.0,) * 3),
                                     att_max_angle=0.0)
        env = VecEnv(sim, DetumbleTask(), max_episode_steps=100)
        env.reset()
        ang_mom = np.zeros((4, 3))
        ang_mom[offender, 2] = 0.3
        set_state(sim, ang_mom=ang_mom)
        env.actions.fill_(0.5)
        env.step()
        envs.append(env)

    terminated, reference = envs
    np.testing.assert_array_equal(terminated.terminated.numpy(), [True, False])
    assert not terminated.truncated.numpy().any()
    np.testing.assert_array_equal(terminated.terminal_obs.numpy(), reference.obs.numpy())
    np.testing.assert_array_equal(terminated.reward.numpy(), reference.reward.numpy())
    np.testing.assert_array_equal(terminated.obs.numpy()[1], reference.obs.numpy()[1])
    np.testing.assert_array_equal(terminated.sim.get_episode_steps().numpy(), [0, 0, 1, 1])
    np.testing.assert_array_equal(terminated.sim.actuators.states.numpy()["rw_ang_mom"][:2], 0.0)


@pytest.mark.parametrize("integrator", INTEGRATORS)
def test_multibody_rollout_preserves_simulator_and_requires_checks_off(make_sim, integrator):
    sim = make_sim(num_envs=2, num_bodies=2, integrator=integrator, dt=DT,
                   gravity=False, differentiable=True)
    set_state(sim, vel=[[0.0, 1.0, 0.0]] * 4)
    sim.step()
    state = sim.states.numpy().copy()
    wheels = sim.actuators.states.numpy().copy()
    status = sim.get_terminated().numpy().copy()
    counters = sim.get_episode_steps().numpy().copy()
    rollout = sim.differentiable_rollout(2)
    with rollout.tape:
        rollout.forward()

    np.testing.assert_array_equal(rollout.final_states.numpy()["pos"][:, 1], 3 * DT)
    np.testing.assert_array_equal(sim.states.numpy(), state)
    np.testing.assert_array_equal(sim.actuators.states.numpy(), wheels)
    np.testing.assert_array_equal(sim.get_terminated().numpy(), status)
    np.testing.assert_array_equal(sim.get_episode_steps().numpy(), counters)
    assert sim.step_count == 1
    for flag in ("check_bounds", "check_planet_impact"):
        checked = make_sim(num_bodies=2, integrator=integrator, gravity=False,
                           differentiable=True, **{flag: True})
        with pytest.raises(RuntimeError, match=flag):
            checked.differentiable_rollout(2)


@pytest.mark.parametrize("integrator", INTEGRATORS)
@pytest.mark.parametrize("auto_reset", [False, True])
def test_multibody_cuda_graph_matches_eager(make_sim, device, integrator, auto_reset):
    if "cuda" not in str(device):
        pytest.skip("multi-body graph/eager equivalence requires CUDA")

    sims = []
    for _ in range(2):
        sim = make_sim(num_envs=2, num_bodies=3, integrator=integrator, dt=DT,
                       gravity=False, check_bounds=True, pos_limit=1.0)
        set_state(sim, pos=[[2.0, 0.0, 0.0]] + [[0.0] * 3] * 5,
                  vel=[[0.0, 1.0, 0.0]] * 6)
        sims.append(sim)
    captured, eager = sims
    result = captured.simulate(3, save_states=True, auto_reset_terminated=auto_reset)
    positions = []
    for _ in range(3):
        eager.step()
        positions.append(get_state(eager)["pos"])
        if auto_reset:
            eager.reset_terminated()

    np.testing.assert_allclose(result["positions"].numpy(), positions, rtol=1e-12)
    for field, values in get_state(eager).items():
        np.testing.assert_allclose(get_state(captured)[field], values, rtol=1e-12)
    np.testing.assert_array_equal(captured.actuators.states.numpy(), eager.actuators.states.numpy())
    np.testing.assert_array_equal(captured.actuators.commands.numpy(), eager.actuators.commands.numpy())
    np.testing.assert_array_equal(captured.get_terminated().numpy(), eager.get_terminated().numpy())
    np.testing.assert_array_equal(captured.get_episode_steps().numpy(), eager.get_episode_steps().numpy())
    assert captured.step_count == eager.step_count == 3


def test_partial_reset_of_terminated_env_rejected(make_sim):
    """Resetting only some bodies of a terminated environment used to clear
    their status, which the next step re-flagged from the still-terminated
    siblings: the reset bodies advanced one step and froze again."""
    sim = make_sim(num_envs=2, num_bodies=2, gravity=False, check_bounds=True, pos_limit=1.0)
    set_state(sim, pos=[[2.0, 0.0, 0.0]] + [[0.0] * 3] * 3)
    sim.step()
    with pytest.raises(ValueError, match="environment 0 is terminated"):
        sim.reset(body_ids=[0])
    sim.reset(env_ids=[1], body_ids=[0])   # env 1 is running: partial is fine
    sim.reset(env_ids=[0])
    assert not sim.get_terminated().numpy()["terminated"].any()


def test_partial_reset_to_conditions_of_terminated_env_rejected(make_sim):
    """reset_to_conditions follows the same rule as reset(): a partial reset of
    a terminated environment would re-freeze the reset bodies."""
    sim = make_sim(num_envs=2, num_bodies=2, gravity=False, check_bounds=True, pos_limit=1.0)
    set_state(sim, pos=[[2.0, 0.0, 0.0]] + [[0.0] * 3] * 3)
    sim.step()
    zeros = wp.zeros(4, dtype=wp.vec3d, device=sim.device)
    with pytest.raises(ValueError, match="environment 0 is terminated"):
        sim.reset_to_conditions(positions=zeros, momenta=zeros, body_ids=[0])
    sim.reset_to_conditions(positions=zeros, momenta=zeros, env_ids=[0])
    assert not sim.get_terminated().numpy()["terminated"].any()


@pytest.mark.parametrize("integrator", ["euler", "rk4", "symplectic"])
@pytest.mark.parametrize("check_bounds", [False, True])
def test_non_finite_state_terminates_its_environment(make_sim, integrator, check_bounds):
    """A NaN passes every limit check (comparisons with NaN are false), so it
    used to propagate silently. It now terminates the environment with
    reason 6, whether or not bounds are checked."""
    sim = make_sim(num_envs=2, num_bodies=2, integrator=integrator, gravity=False,
                   check_bounds=check_bounds)
    set_state(sim, pos=[[7.0e6, 0.0, 0.0]] * 4, mom=[[np.nan, 0.0, 0.0]] + [[0.0] * 3] * 3)
    sim.step()
    status = sim.get_terminated().numpy()
    np.testing.assert_array_equal(status["terminated"], [True, True, False, False])
    np.testing.assert_array_equal(status["reason"], [6, 6, 0, 0])


@pytest.mark.parametrize("integrator", ["euler", "rk4", "symplectic"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_command_terminates_its_environment(make_sim, integrator, bad):
    """wp.clamp turned a NaN command into a finite actuation: a NaN throttle
    into no thrust, a NaN wheel torque into the full negative torque, with
    nothing reaching the state, so a diverged policy went unnoticed. A
    non-finite thrust or wheel command now terminates its environment with
    reason 6 before the step, the state and wheel momenta carried over."""
    sim = make_sim(num_envs=3, integrator=integrator, dt=DT, gravity=False, check_bounds=False)
    sim.thruster_manager.set_config(isp=200.0)
    set_state(sim, pos=[[7.0e6, 0.0, 0.0]] * 3, mass=[2.0] * 3)
    before = get_state(sim)

    thrust = np.zeros((3, 8))
    thrust[0, 5] = bad                       # env 0: one thruster command (slot 5)
    torque = np.full((3, 4), 0.01)
    torque[1, 2] = bad                       # env 1: one wheel command
    sim.set_thrust_commands_per_entity(wp.array(thrust, dtype=wp.float64, device=sim.device))
    sim.set_rw_torque_commands_per_entity(wp.array(torque, dtype=wp.float64, device=sim.device))
    for _ in range(2):
        sim.step()

    status = sim.get_terminated().numpy()
    np.testing.assert_array_equal(status["terminated"], [True, True, False])
    np.testing.assert_array_equal(status["reason"], [6, 6, 0])
    after = get_state(sim)
    for key in before:
        np.testing.assert_array_equal(after[key][:2], before[key][:2], err_msg=key)
    h = sim.actuators.states.numpy()["rw_ang_mom"]
    np.testing.assert_array_equal(h[:2], 0.0)
    np.testing.assert_allclose(h[2, :3], 0.01 * 2 * DT, rtol=1e-12)

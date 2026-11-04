"""
Tests for the differentiable-RL layer of the package (torch bridge, DiffVecEnv).

The bridge exposes one sim step as a torch autograd node; these tests pin the
contract first-order RL builds on: forward equals the warp-native rollout,
gradcheck passes on a single step (exact — everything is float64), chained-step
gradients match finite differences, and the policy-in-the-loop cross-terms
(obs_t -> action_t -> state_{t+1}) are live — the property the per-step
architecture exists for. On top: windows chain/reset correctly, and gradients
flow through a full env+task rollout. The task is a test-local DiffTask; the
example task and the SHAC trainer are tested in test_example_shac.py. Skipped
without torch.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cosmic_warp.core.autodiff import command_sequence
from cosmic_warp.physics.gravity import MU_EARTH
from cosmic_warp.rl.diff_bridge import DiffSimStepper, U_DIM
from cosmic_warp.rl.diff_env import DiffVecEnv
from conftest import get_state, set_state
from orbit_utils import elliptical_state

A = 7.5e6      # semi-major axis [m]
ECC = 0.1
DT = 1.0
MASS0 = 2.0    # strictly above the default mass_dry (1.0): mass floor inactive
H = 4          # chained-step horizon


def _make_orbit_sim(make_sim, integrator):
    """Physical scenario matching tests/test_gradients.py."""
    sim = make_sim(num_envs=2, integrator=integrator, dt=DT, differentiable=True)
    r0, v0 = elliptical_state(A, ECC, MU_EARTH, 0.0)
    set_state(sim, pos=[r0, r0], vel=[v0, v0], mass=[MASS0, MASS0],
              ang_mom=[[0.02, -0.01, 0.03]] * 2)
    return sim


def _make_toy_sim(make_sim, integrator="symplectic"):
    """Well-conditioned O(1) dynamics for gradcheck: no gravity, pure
    rotational dynamics plus drift, all state components of similar scale."""
    sim = make_sim(num_envs=2, integrator=integrator, dt=0.05,
                   differentiable=True, gravity=False)
    set_state(sim, pos=[[1.0, -2.0, 0.5]] * 2, vel=[[0.1, 0.05, -0.02]] * 2,
              mass=[MASS0] * 2, ang_mom=[[0.05, -0.03, 0.02]] * 2)
    sim.actuators.set_rw_state([0.02, -0.01, 0.015, 0.0])
    return sim


def _zeros_u(sim):
    return torch.zeros((sim.num_entities, U_DIM), dtype=torch.float64,
                       device=str(sim.device))


def _fd_vs_grad(analytic, fd, rtol=1e-4):
    scale = max(abs(fd), abs(analytic))
    if scale == 0.0:
        return
    rel = abs(fd - analytic) / scale
    assert rel < rtol, f"autograd {analytic:.9e} vs FD {fd:.9e} (rel {rel:.2e})"


def test_pack_unpack_roundtrip(make_sim):
    """read_state matches the sim's structs; write_state -> read_state is identity."""
    sim = _make_toy_sim(make_sim)
    stepper = DiffSimStepper(sim)

    x = stepper.read_state()
    s = sim.states.numpy()
    xn = x.cpu().numpy()
    np.testing.assert_array_equal(xn[:, 0:3], s["pos"])
    np.testing.assert_array_equal(xn[:, 3:6], s["mom"])
    np.testing.assert_array_equal(xn[:, 6], s["mass"])
    np.testing.assert_array_equal(xn[:, 7:11], s["quat"])
    np.testing.assert_array_equal(xn[:, 11:14], s["ang_mom"])
    np.testing.assert_array_equal(
        xn[:, 14:18], sim.actuators.states.numpy()["rw_ang_mom"])

    stepper.write_state(torch.zeros_like(x))
    assert float(np.abs(sim.states.numpy()["pos"]).max()) == 0.0
    stepper.write_state(x)
    torch.testing.assert_close(stepper.read_state(), x)


@pytest.mark.parametrize("integrator", ["euler", "rk4", "symplectic"])
def test_step_matches_differentiable_rollout(make_sim, integrator):
    """Chained bridge steps reproduce the warp-native rollout trajectory."""
    sim = _make_orbit_sim(make_sim, integrator)
    stepper = DiffSimStepper(sim)

    torque = [0.004, -0.003, 0.002]
    cmds = command_sequence(sim, H)
    for t in range(H):
        c = cmds[t].numpy()
        c["rw_torque_cmd"][:, :3] = torque
        cmds[t].assign(c)
    rollout = sim.differentiable_rollout(H, commands=cmds)
    rollout.forward()
    ref = rollout.final_states.numpy()

    x = stepper.read_state()
    u = _zeros_u(sim)
    u[:, 8:11] = torch.tensor(torque, dtype=torch.float64)
    with torch.no_grad():
        for _ in range(H):
            x = stepper.step(x, u)
    xn = x.cpu().numpy()
    np.testing.assert_allclose(xn[:, 0:3], ref["pos"], rtol=1e-13)
    np.testing.assert_allclose(xn[:, 7:11], ref["quat"], rtol=1e-13)
    np.testing.assert_allclose(xn[:, 11:14], ref["ang_mom"], rtol=1e-13)


@pytest.mark.parametrize("integrator", ["euler", "rk4", "symplectic"])
def test_simstep_gradcheck(make_sim, integrator):
    """torch.autograd.gradcheck on one step — the load-bearing bridge test.

    Everything on the diff path is float64, so gradcheck is exact; it catches
    any non-adjoint-safe pack/unpack kernel (silently zeroed gradients).
    """
    sim = _make_toy_sim(make_sim, integrator)
    stepper = DiffSimStepper(sim)

    x0 = stepper.read_state().requires_grad_(True)
    # Interior points of every actuator range: thrusters at 0.3 throttle
    # (0 is a clamp kink), wheel torques well inside the +-0.1 limit.
    u0 = _zeros_u(sim)
    u0[:, 0:8] = 0.3
    u0[:, 8:12] = 0.004
    u0.requires_grad_(True)

    assert torch.autograd.gradcheck(stepper.step, (x0, u0),
                                    eps=1e-6, atol=1e-5, rtol=1e-3)


def test_double_backward_raises(make_sim):
    """The backward is a Warp adjoint, not a torch graph. When the loss also
    has a torch-side path (here x0**3), a second derivative used to succeed
    while treating the step Jacobian as constant: silently wrong values."""
    sim = _make_toy_sim(make_sim)
    stepper = DiffSimStepper(sim)
    x0 = stepper.read_state().requires_grad_(True)
    loss = (stepper.step(x0, _zeros_u(sim)) ** 2).sum() + (x0 ** 3).sum()
    (g,) = torch.autograd.grad(loss, x0, create_graph=True)
    with pytest.raises(RuntimeError, match="once_differentiable|double backward|derivative"):
        g.sum().backward()


@pytest.mark.parametrize("integrator", ["euler", "rk4", "symplectic"])
def test_chained_grads_match_fd(make_sim, integrator):
    """d(loss)/dx0 and d(loss)/du over an H-step chain match central FD."""
    sim = _make_orbit_sim(make_sim, integrator)
    stepper = DiffSimStepper(sim)
    dev = str(sim.device)
    target = torch.tensor([0.0, A, 0.0], dtype=torch.float64, device=dev)

    def loss_fn(x0, u):
        x = x0
        for _ in range(H):
            x = stepper.step(x, u)
        d = x[:, 0:3] - target
        return (d * d).sum()

    x0 = stepper.read_state().requires_grad_(True)
    u = _zeros_u(sim)
    u[:, 0] = 0.3  # thruster 0 on: control affects the position loss
    u.requires_grad_(True)

    loss_fn(x0, u).backward()
    gx = x0.grad.clone()
    gu = u.grad.clone()
    assert float(gx.abs().max()) > 0.0 and float(gu.abs().max()) > 0.0

    # In-plane position/momentum of entity 0 (same FD steps as test_gradients),
    # and thruster 0 throttle.
    with torch.no_grad():
        for col, h in [(0, 1.0), (1, 1.0), (3, 1.0), (4, 1.0)]:
            losses = []
            for sign in (+1.0, -1.0):
                xp = x0.detach().clone()
                xp[0, col] += sign * h
                losses.append(float(loss_fn(xp, u.detach())))
            _fd_vs_grad(float(gx[0, col]), (losses[0] - losses[1]) / (2.0 * h))

        h = 1e-3
        losses = []
        for sign in (+1.0, -1.0):
            up = u.detach().clone()
            up[0, 0] += sign * h
            losses.append(float(loss_fn(x0.detach(), up)))
        _fd_vs_grad(float(gu[0, 0]), (losses[0] - losses[1]) / (2.0 * h))


def test_policy_in_loop_cross_terms(make_sim):
    """The closed-loop gradient (policy reads the evolving state) matches FD
    and differs from the open-loop gradient (policy inputs detached) — the
    obs_t -> action_t -> state_{t+1} cross-terms are live."""
    sim = _make_toy_sim(make_sim)
    stepper = DiffSimStepper(sim)
    E = sim.num_entities
    dev = str(sim.device)

    torch.manual_seed(0)
    W0 = -0.5 * torch.eye(3, dtype=torch.float64, device=dev) \
        + 0.05 * torch.randn(3, 3, dtype=torch.float64, device=dev)
    x0 = stepper.read_state()

    def rollout_loss(W, closed):
        x = x0
        for _ in range(H):
            obs = x if closed else x.detach()
            torque = obs[:, 11:14] @ W.T  # detumble-like feedback on ang_mom
            zeros8 = torch.zeros((E, 8), dtype=torch.float64, device=dev)
            zeros1 = torch.zeros((E, 1), dtype=torch.float64, device=dev)
            u = torch.cat([zeros8, torque, zeros1], dim=1)
            x = stepper.step(x, u)
        return (x[:, 11:14] ** 2).sum()

    W = W0.clone().requires_grad_(True)
    g_closed = torch.autograd.grad(rollout_loss(W, closed=True), W)[0]
    g_open = torch.autograd.grad(rollout_loss(W, closed=False), W)[0]

    h = 1e-6
    with torch.no_grad():
        for i in range(3):
            for j in range(3):
                Wp = W0.clone()
                Wp[i, j] += h
                Wm = W0.clone()
                Wm[i, j] -= h
                fd = float(rollout_loss(Wp, True) - rollout_loss(Wm, True)) / (2.0 * h)
                _fd_vs_grad(float(g_closed[i, j]), fd)

    assert not torch.allclose(g_closed, g_open, rtol=1e-3), (
        "closed-loop and open-loop gradients coincide: the policy-in-the-loop "
        "path is not being differentiated"
    )


def test_repeated_backward_consistent(make_sim):
    """Backward with retain_graph=True gives the same gradients twice (the
    per-call tape is re-zeroed and re-seeded, never accumulated)."""
    sim = _make_toy_sim(make_sim)
    stepper = DiffSimStepper(sim)

    x0 = stepper.read_state().requires_grad_(True)
    u = _zeros_u(sim)
    u[:, 8:11] = 0.004
    u.requires_grad_(True)

    loss = (stepper.step(x0, u) ** 2).sum()
    g1 = torch.autograd.grad(loss, (x0, u), retain_graph=True)
    g2 = torch.autograd.grad(loss, (x0, u))
    torch.testing.assert_close(g1[0], g2[0])
    torch.testing.assert_close(g1[1], g2[1])


def test_taped_steps_leave_sim_untouched(make_sim):
    """Bridge steps never mutate the simulator's state or episode counters."""
    sim = _make_toy_sim(make_sim)
    stepper = DiffSimStepper(sim)
    before = get_state(sim)

    x = stepper.read_state()
    u = _zeros_u(sim)
    u[:, 8:11] = 0.01
    with torch.no_grad():
        for _ in range(3):
            x = stepper.step(x, u)

    assert (sim.episode_steps.numpy() == 0).all()
    after = get_state(sim)
    for field in before:
        np.testing.assert_array_equal(before[field], after[field])


def test_nonfinite_step_leaves_sim_status_untouched(make_sim):
    """The always-on non-finite check fires on the stepper's private status,
    never on the simulator's."""
    sim = _make_toy_sim(make_sim)
    stepper = DiffSimStepper(sim)
    x = stepper.read_state()
    x[1, :] = float("nan")
    with torch.no_grad():
        x_next = stepper.step(x, _zeros_u(sim))
    status = sim.ctx.termination_status.numpy()
    assert not status["terminated"].any()
    assert torch.isfinite(x_next[0]).all() and not torch.isfinite(x_next[1]).all()


def test_stepper_guards(make_sim):
    """Constructor and step() validate their contract."""
    with pytest.raises(RuntimeError, match="differentiable"):
        DiffSimStepper(make_sim(num_envs=2))
    with pytest.raises(RuntimeError, match="check_bounds"):
        DiffSimStepper(make_sim(num_envs=2, differentiable=True, check_bounds=True))

    sim = _make_toy_sim(make_sim)
    stepper = DiffSimStepper(sim)
    x = stepper.read_state()
    u = _zeros_u(sim)
    with pytest.raises(TypeError, match="float64"):
        stepper.step(x.float(), u)
    with pytest.raises(ValueError, match="shape"):
        stepper.step(x, torch.zeros((sim.num_entities, U_DIM + 1),
                                    dtype=torch.float64, device=str(sim.device)))


def test_no_grad_fast_path_matches_taped(make_sim):
    """Under no_grad (or with detached inputs) step() skips taping and
    per-call allocation but produces the identical next state."""
    sim = _make_toy_sim(make_sim)
    stepper = DiffSimStepper(sim)
    x = stepper.read_state()
    u = _zeros_u(sim)
    u[:, 8:11] = 0.004

    y_taped = stepper.step(x.clone().requires_grad_(True), u)
    assert y_taped.requires_grad
    assert stepper._eval_buf is None          # the taped path never allocates it

    with torch.no_grad():
        y_fast = stepper.step(x, u)
    assert stepper._eval_buf is not None
    assert not y_fast.requires_grad
    torch.testing.assert_close(y_fast, y_taped, rtol=0.0, atol=0.0)

    y_detached = stepper.step(x, u)           # grad enabled, nothing requires it
    assert not y_detached.requires_grad
    torch.testing.assert_close(y_detached, y_taped.detach(), rtol=0.0, atol=0.0)


# --------------------------------------------------------------------------
# DiffVecEnv windows
# --------------------------------------------------------------------------

class _SpinDownTask:
    """Test-local DiffTask (single body): actions torque wheels 0-2, the
    observation is ``[quat(4), ang_mom(3), rw_ang_mom(4)]`` and the reward
    penalizes body angular momentum and control effort."""

    act_dim = 3
    obs_dim = 11
    TORQUE = 0.05   # [N m] per unit action, inside the default wheel limit

    def bind(self, sim):
        assert sim.num_bodies == 1

    def commands(self, action):
        zeros8 = torch.zeros((action.shape[0], 8), dtype=torch.float64, device=action.device)
        zeros1 = torch.zeros((action.shape[0], 1), dtype=torch.float64, device=action.device)
        return torch.cat([zeros8, self.TORQUE * action, zeros1], dim=1)

    def observation(self, x):
        return x[:, 7:18]

    def reward(self, x, action):
        return -((x[:, 11:14] ** 2).sum(dim=1) + 1e-3 * (action ** 2).sum(dim=1))


def test_window_chaining_and_reset(make_sim):
    """Carried state is detached between windows; the episode time limit
    triggers a full reset; taped windows never touch sim.episode_steps."""
    sim = _make_toy_sim(make_sim)
    with pytest.raises(ValueError, match="multiple"):
        DiffVecEnv(sim, _SpinDownTask(), horizon=4, max_episode_steps=10)
    # horizon = 0 used to raise ZeroDivisionError from the modulo check
    with pytest.raises(ValueError, match="horizon must be a positive integer"):
        DiffVecEnv(sim, _SpinDownTask(), horizon=0, max_episode_steps=10)

    env = DiffVecEnv(sim, _SpinDownTask(), horizon=5, max_episode_steps=10)
    env.reset()
    x0 = env.state.clone()
    zero_policy = lambda obs: torch.zeros(
        (env.num_envs, env.act_dim), dtype=torch.float64, device=env.device)

    window = env.rollout(zero_policy)
    assert len(window.obs) == 6 and len(window.rewards) == 5
    assert not env.end_window()                      # mid-episode: no reset
    assert env.state.requires_grad is False             # BPTT truncated
    assert not torch.equal(env.state, x0)               # dynamics advanced

    env.rollout(zero_policy)
    assert env.end_window()                          # hit the time limit
    assert env._steps_since_reset == 0
    assert (sim.episode_steps.numpy() == 0).all()


def test_window_calls_out_of_order_rejected(make_sim):
    """end_window() needs a pending rollout(). Without one it used to fail
    with an AttributeError on None, or, after a reset, carry the discarded
    window's final state over the reset (H27)."""
    env = DiffVecEnv(_make_toy_sim(make_sim), _SpinDownTask(), horizon=2,
                     max_episode_steps=4)
    zero_policy = lambda obs: torch.zeros(
        (env.num_envs, env.act_dim), dtype=torch.float64, device=env.device)

    with pytest.raises(RuntimeError, match="rollout"):
        env.end_window()                             # nothing rolled out yet
    env.rollout(zero_policy)
    env.end_window()
    with pytest.raises(RuntimeError, match="rollout"):
        env.end_window()                             # window already closed
    env.rollout(zero_policy)
    x_reset = env.reset()
    with pytest.raises(RuntimeError, match="rollout"):
        env.end_window()                             # reset discarded it
    torch.testing.assert_close(env.task.observation(env.state), x_reset,
                               rtol=0.0, atol=0.0)


def test_env_rollout_grad_matches_fd(make_sim):
    """d(loss)/d(policy params) through DiffVecEnv + a task matches
    central FD — the full task/env stack preserves the bridge's gradient."""
    sim = _make_toy_sim(make_sim)
    env = DiffVecEnv(sim, _SpinDownTask(), horizon=H, max_episode_steps=4 * H)
    env.reset()
    dev = env.device

    torch.manual_seed(0)
    W0 = 0.1 * torch.randn(env.act_dim, env.obs_dim, dtype=torch.float64, device=dev)

    def loss_fn(W):
        window = env.rollout(lambda obs: torch.tanh(obs @ W.T))
        return -torch.stack(window.rewards).sum()

    W = W0.clone().requires_grad_(True)
    g = torch.autograd.grad(loss_fn(W), W)[0]
    assert float(g.abs().max()) > 0.0

    h = 1e-5
    with torch.no_grad():
        for i, j in [(0, 0), (1, 4), (2, 8), (0, 10)]:
            Wp = W0.clone()
            Wp[i, j] += h
            Wm = W0.clone()
            Wm[i, j] -= h
            fd = float(loss_fn(Wp) - loss_fn(Wm)) / (2.0 * h)
            _fd_vs_grad(float(g[i, j]), fd)


# --------------------------------------------------------------------------
# Backward against the forward's parameters
# --------------------------------------------------------------------------

def _make_randomized_sim(make_sim):
    """Inertia randomized at every reset, so a reset between a step and its
    backward changes the parameters the adjoint would read."""
    sim = make_sim(num_envs=2, integrator="rk4", dt=0.05, differentiable=True,
                   gravity=False, reset_config={"seed": 0})
    sim.reset_manager.set_bounds(ang_mom_bounds=((-0.2,) * 3, (0.2,) * 3),
                                 mass_bounds=(2.0, 2.0))
    sim.randomizer.set_bounds(inertia_scale=(0.2, 5.0))
    return sim


def _window_grad(make_sim, end_window_first):
    """dL/dW of one DiffVecEnv window that ends its episode, with end_window()
    (a reset that redraws the inertia) before or after backward()."""
    env = DiffVecEnv(_make_randomized_sim(make_sim), _SpinDownTask(),
                     horizon=H, max_episode_steps=H)
    env.reset()
    torch.manual_seed(0)
    W = (0.1 * torch.randn(env.act_dim, env.obs_dim, dtype=torch.float64,
                           device=env.device)).requires_grad_(True)
    snapshots = []
    snapshot = env.stepper._snapshot_ctx
    env.stepper._snapshot_ctx = lambda: snapshots.append(1) or snapshot()
    window = env.rollout(lambda obs: torch.tanh(obs @ W.T))
    assert len(snapshots) == 1                # one parameter copy per window
    loss = -torch.stack(window.rewards).sum()
    inertia = env.sim.vehicle.configs.numpy()["inertia"].copy()
    if end_window_first:
        assert env.end_window()
        assert not np.allclose(env.sim.vehicle.configs.numpy()["inertia"], inertia)
        loss.backward()
    else:
        loss.backward()
        assert env.end_window()
    return W.grad


def test_window_grad_independent_of_end_window_order(make_sim):
    """The backward replays the parameters of the forward. It used to read
    the simulator's live configuration, so end_window() before backward()
    at the episode limit gave the gradient of the redrawn inertia (off by
    up to 390% in the review's repro)."""
    g_after = _window_grad(make_sim, end_window_first=False)
    g_before = _window_grad(make_sim, end_window_first=True)
    assert float(g_after.abs().max()) > 0.0
    torch.testing.assert_close(g_before, g_after, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("frozen", [False, True])
def test_step_grad_independent_of_later_set_config(make_sim, frozen):
    """A set_config between taped steps and backward() does not change the
    gradient, per step or inside frozen_params()."""
    def grads(change_before_backward):
        sim = _make_toy_sim(make_sim, "rk4")
        sim.vehicle.set_config(inertia=np.diag([0.4, 0.5, 0.3]))
        stepper = DiffSimStepper(sim)
        x0 = stepper.read_state().requires_grad_(True)
        u = _zeros_u(sim)
        u[:, 8:11] = 0.004
        u.requires_grad_(True)
        x = x0
        if frozen:
            with stepper.frozen_params():
                for _ in range(H):
                    x = stepper.step(x, u)
        else:
            for _ in range(H):
                x = stepper.step(x, u)
        loss = (x[:, 7:14] ** 2).sum()
        if change_before_backward:
            sim.vehicle.set_config(inertia=np.diag([2.0, 1.5, 3.0]))
        return torch.autograd.grad(loss, (x0, u))

    ref, changed = grads(False), grads(True)
    for g_ref, g in zip(ref, changed):
        assert float(g_ref.abs().max()) > 0.0
        torch.testing.assert_close(g, g_ref, rtol=1e-12, atol=0.0)

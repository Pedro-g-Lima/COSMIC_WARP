"""
Differentiable-rollout validation.

Checks, per integrator:
  1. Tape gradients of a terminal loss w.r.t. the initial state match central
     finite differences (the same style of ground truth the physics suite
     uses, applied to the adjoint instead of the propagator).
  2. Control gradients: thruster and reaction-wheel command sequences receive
     correct sensitivities.
  3. Vehicle-parameter gradients (system identification): drag, thruster,
     inertia, magnetic-dipole and reaction-wheel parameters match finite
     differences.
  4. End-to-end signal: plain gradient descent on the initial velocity
     retargets the trajectory, and recovers an unknown drag coefficient
     (validates tape/parameter resync via rollout.reset()).
"""
import numpy as np
import pytest
import warp as wp

from cosmic_warp.core.state import State
from cosmic_warp.core.autodiff import command_sequence
from cosmic_warp.physics.gravity import MU_EARTH
from conftest import set_state
from orbit_utils import elliptical_state


A = 7.5e6      # semi-major axis [m]
ECC = 0.1
NUM_STEPS = 20
DT = 1.0
MASS0 = 2.0    # initial mass [kg], strictly above the default mass_dry (1.0)
               # so thrusters have propellant and the mass floor is inactive


@wp.kernel
def _pos_loss_kernel(
    states: wp.array(dtype=State),
    target: wp.vec3d,
    loss: wp.array(dtype=wp.float64),
):
    """Sum of squared final-position errors to a fixed target."""
    tid = wp.tid()
    d = states[tid].pos - target
    wp.atomic_add(loss, 0, wp.dot(d, d))


@wp.kernel
def _ang_mom_loss_kernel(
    states: wp.array(dtype=State),
    loss: wp.array(dtype=wp.float64),
):
    """Sum of squared final body angular momenta."""
    tid = wp.tid()
    am = states[tid].ang_mom
    wp.atomic_add(loss, 0, wp.dot(am, am))


def _make_diff_sim(make_sim, integrator, ang_mom=None, **kwargs):
    sim = make_sim(num_envs=2, integrator=integrator, dt=DT,
                   differentiable=True, **kwargs)
    r0, v0 = elliptical_state(A, ECC, MU_EARTH, 0.0)
    set_state(sim, pos=[r0, r0], vel=[v0, v0], mass=[MASS0, MASS0],
              ang_mom=None if ang_mom is None else [ang_mom, ang_mom])
    return sim


def _run_loss(sim, rollout, loss_kernel, loss_args, taped):
    """One forward pass + loss evaluation; returns the scalar loss value."""
    rollout.reset()
    loss = wp.zeros(1, dtype=wp.float64, device=sim.device, requires_grad=True)
    if taped:
        with rollout.tape:
            rollout.forward()
            wp.launch(loss_kernel, dim=sim.num_entities,
                      inputs=[rollout.final_states, *loss_args, loss],
                      device=sim.device)
        rollout.tape.backward(loss)
    else:
        rollout.forward()
        wp.launch(loss_kernel, dim=sim.num_entities,
                  inputs=[rollout.final_states, *loss_args, loss],
                  device=sim.device)
    return float(loss.numpy()[0])


def _fd_vs_tape(analytic, fd, rtol=1e-4):
    scale = max(abs(fd), abs(analytic))
    if scale == 0.0:
        return  # both exactly zero (e.g. out-of-plane component of a planar orbit)
    rel = abs(fd - analytic) / scale
    assert rel < rtol, f"adjoint {analytic:.9e} vs FD {fd:.9e} (rel {rel:.2e})"


@pytest.mark.parametrize("integrator", ["euler", "rk4", "symplectic"])
def test_nonfinite_entity_freezes_inside_rollout(make_sim, integrator):
    """A non-finite entity terminates inside the rollout and is carried over
    unchanged by later steps, without touching the simulator's own status."""
    sim = _make_diff_sim(make_sim, integrator)
    s = sim.states.numpy()
    s["mom"][1] = np.nan
    sim.states.assign(s)
    rollout = sim.differentiable_rollout(3)
    rollout.forward()

    status = sim.ctx.termination_status.numpy()
    assert not status["terminated"].any() and not status["reason"].any()
    # Entity 1: NaN from step 1 on, frozen (not zeroed) afterwards
    frozen = [rollout.states[t].numpy()[1] for t in (1, 2, 3)]
    for t in (2, 3):
        for field in ("pos", "mom", "quat", "ang_mom", "mass"):
            np.testing.assert_array_equal(frozen[t - 1][field], frozen[0][field])
    assert np.isfinite(rollout.final_states.numpy()["pos"][0]).all()


@pytest.mark.parametrize("integrator", ["euler", "rk4", "symplectic"])
def test_initial_state_gradients_match_fd(make_sim, integrator):
    """dloss/dx0 from the tape matches central finite differences."""
    sim = _make_diff_sim(make_sim, integrator)
    target = wp.vec3d(0.0, A, 0.0)
    rollout = sim.differentiable_rollout(NUM_STEPS)

    _run_loss(sim, rollout, _pos_loss_kernel, [target], taped=True)
    # Copy: on CPU .numpy() is a zero-copy view, and the untaped FD runs
    # below zero the grad buffer via rollout.reset()
    g = rollout.initial_states.grad.numpy().copy()

    s0 = sim.states.numpy().copy()
    # (field, component, fd step): in-plane position [m] and momentum [kg*m/s]
    for field, comp, h in [("pos", 0, 1.0), ("pos", 1, 1.0),
                           ("mom", 0, 1.0), ("mom", 1, 1.0)]:
        losses = []
        for sign in (+1.0, -1.0):
            s = s0.copy()
            s[field][0][comp] += sign * h
            sim.states.assign(s)
            losses.append(_run_loss(sim, rollout, _pos_loss_kernel, [target], taped=False))
        sim.states.assign(s0)
        fd = (losses[0] - losses[1]) / (2.0 * h)
        _fd_vs_tape(g[field][0][comp], fd)

    # Entity 1 sees an identical orbit: gradients must match entity 0
    np.testing.assert_allclose(g["pos"][1], g["pos"][0], rtol=1e-12)


@pytest.mark.parametrize("integrator", ["euler", "rk4", "symplectic"])
def test_full_state_gradients_match_fd(make_sim, integrator):
    """dloss/dx0 for every state block, with J2, drag (low orbit), gravity
    gradient and magnetic torque on, a non-identity attitude and a tumbling
    body: the initial-state test above covers only planar pos/mom with
    point-mass gravity."""
    r = 6.678e6   # ~300 km: drag is not negligible
    sim = make_sim(num_envs=2, integrator=integrator, dt=DT, differentiable=True,
                   j2_perturbation=True, drag=True, gravity_gradient=True,
                   magnetic_torque=True)
    sim.vehicle.set_config(mass_dry=1.0, inertia=np.diag([0.15, 0.2, 0.3]),
                           area_drag=0.5, magnetic_dipole=[0.1, 0.05, 0.0])
    v = np.sqrt(MU_EARTH / r)
    q = np.array([0.1, -0.2, 0.3, 0.9]) / np.linalg.norm([0.1, -0.2, 0.3, 0.9])
    set_state(sim, pos=[[r, 0.0, 0.0]] * 2, vel=[[0.0, 0.8 * v, 0.6 * v]] * 2,
              mass=[MASS0] * 2, quat=[q] * 2, ang_mom=[[1e-2, -2e-2, 3e-2]] * 2)
    target = wp.vec3d(0.0, r, 0.0)
    rollout = sim.differentiable_rollout(NUM_STEPS)
    s0 = sim.states.numpy().copy()

    # (loss, field, component, fd step)
    cases = [(_pos_loss_kernel, [target], "pos", 2, 1.0),
             (_pos_loss_kernel, [target], "mom", 2, 1e-2),
             (_ang_mom_loss_kernel, [], "quat", 0, 1e-6),
             (_ang_mom_loss_kernel, [], "quat", 3, 1e-6),
             (_ang_mom_loss_kernel, [], "ang_mom", 1, 1e-6)]
    for loss_kernel, loss_args, field, comp, h in cases:
        _run_loss(sim, rollout, loss_kernel, loss_args, taped=True)
        g = float(rollout.initial_states.grad.numpy()[field][0][comp])
        losses = []
        for sign in (+1.0, -1.0):
            st = s0.copy()
            st[field][0][comp] += sign * h
            sim.states.assign(st)
            losses.append(_run_loss(sim, rollout, loss_kernel, loss_args, taped=False))
        sim.states.assign(s0)
        fd = (losses[0] - losses[1]) / (2.0 * h)
        assert fd != 0.0, f"{field}[{comp}] does not affect the loss"
        _fd_vs_tape(g, fd)


@pytest.mark.parametrize("integrator", ["euler", "rk4", "symplectic"])
def test_thruster_command_gradients_match_fd(make_sim, integrator):
    """dloss/du_t for a thruster throttle matches central finite differences."""
    sim = _make_diff_sim(make_sim, integrator)
    target = wp.vec3d(0.0, A, 0.0)

    cmds = command_sequence(sim, NUM_STEPS)
    u = cmds[0].numpy()
    u["thrust_cmd"][:, 0] = 0.5   # thruster 0 firing on the first step
    cmds[0].assign(u)
    u0 = cmds[0].numpy().copy()

    rollout = sim.differentiable_rollout(NUM_STEPS, commands=cmds)
    _run_loss(sim, rollout, _pos_loss_kernel, [target], taped=True)
    g = cmds[0].grad.numpy()["thrust_cmd"][0][0]

    h = 1e-4
    losses = []
    for sign in (+1.0, -1.0):
        u = u0.copy()
        u["thrust_cmd"][0][0] += sign * h
        cmds[0].assign(u)
        losses.append(_run_loss(sim, rollout, _pos_loss_kernel, [target], taped=False))
    cmds[0].assign(u0)
    fd = (losses[0] - losses[1]) / (2.0 * h)
    _fd_vs_tape(g, fd)


@pytest.mark.parametrize("integrator", ["euler", "rk4", "symplectic"])
def test_burnout_step_gradients_match_fd(make_sim, integrator):
    """Through a step where the propellant runs out 40 % in, the burn fraction
    makes the impulse a smooth function of the propellant left: dloss/dm0
    matches finite differences."""
    sim = _make_diff_sim(make_sim, integrator)
    sim.thruster_manager.set_config(max_thrust=10.0, isp=200.0,
                                    positions=[(0.0, 0.0, 0.0)], directions=[(0.0, 1.0, 0.0)])
    mdot = 10.0 / (200.0 * 9.80665)
    s0 = sim.states.numpy().copy()
    s0["mass"][:] = 1.0 + 0.4 * mdot * DT
    sim.states.assign(s0)
    target = wp.vec3d(0.0, A, 0.0)

    cmds = command_sequence(sim, NUM_STEPS)
    u = cmds[0].numpy()
    u["thrust_cmd"][:, 0] = 1.0
    cmds[0].assign(u)

    rollout = sim.differentiable_rollout(NUM_STEPS, commands=cmds)
    _run_loss(sim, rollout, _pos_loss_kernel, [target], taped=True)
    g = rollout.initial_states.grad.numpy()["mass"][0]

    h = 1e-6 * mdot * DT   # stays inside the burnout regime
    losses = []
    for sign in (+1.0, -1.0):
        s = s0.copy()
        s["mass"][0] += sign * h
        sim.states.assign(s)
        losses.append(_run_loss(sim, rollout, _pos_loss_kernel, [target], taped=False))
    sim.states.assign(s0)
    fd = (losses[0] - losses[1]) / (2.0 * h)
    assert fd != 0.0
    _fd_vs_tape(g, fd)


@pytest.mark.parametrize("integrator", ["euler", "rk4", "symplectic"])
def test_rw_command_gradients_match_fd(make_sim, integrator):
    """dloss/d(rw torque) matches finite differences through wheel dynamics."""
    sim = _make_diff_sim(make_sim, integrator)

    cmds = command_sequence(sim, NUM_STEPS)
    for t in range(NUM_STEPS):
        u = cmds[t].numpy()
        u["rw_torque_cmd"][:, 0] = 1e-3   # constant torque on wheel 0
        cmds[t].assign(u)
    u5 = cmds[5].numpy().copy()

    rollout = sim.differentiable_rollout(NUM_STEPS, commands=cmds)
    _run_loss(sim, rollout, _ang_mom_loss_kernel, [], taped=True)
    g = cmds[5].grad.numpy()["rw_torque_cmd"][0][0]

    h = 1e-6
    losses = []
    for sign in (+1.0, -1.0):
        u = u5.copy()
        u["rw_torque_cmd"][0][0] += sign * h
        cmds[5].assign(u)
        losses.append(_run_loss(sim, rollout, _ang_mom_loss_kernel, [], taped=False))
    cmds[5].assign(u5)
    fd = (losses[0] - losses[1]) / (2.0 * h)
    _fd_vs_tape(g, fd)


@pytest.mark.parametrize("integrator", ["euler", "rk4", "symplectic"])
def test_drag_param_gradients_match_fd(make_sim, integrator):
    """dloss/d(cd, area_drag) matches finite differences through drag."""
    sim = _make_diff_sim(make_sim, integrator, drag=True)
    sim.vehicle.set_config(mass_dry=1.0, area_drag=100.0, cd=2.2)
    target = wp.vec3d(0.0, A, 0.0)
    rollout = sim.differentiable_rollout(NUM_STEPS)

    _run_loss(sim, rollout, _pos_loss_kernel, [target], taped=True)
    g = rollout.vehicle_configs.grad.numpy().copy()

    v0 = sim.vehicle.configs.numpy().copy()
    for field, h in [("cd", 0.02), ("area_drag", 1.0)]:
        assert abs(g[field][0]) > 0.0
        losses = []
        for sign in (+1.0, -1.0):
            v = v0.copy()
            v[field][0] += sign * h
            sim.vehicle.configs.assign(v)
            losses.append(_run_loss(sim, rollout, _pos_loss_kernel, [target], taped=False))
        sim.vehicle.configs.assign(v0)
        fd = (losses[0] - losses[1]) / (2.0 * h)
        _fd_vs_tape(g[field][0], fd)


@pytest.mark.parametrize("integrator", ["euler", "rk4", "symplectic"])
def test_thruster_param_gradients_match_fd(make_sim, integrator):
    """dloss/d(max_thrust, isp) matches FD, including the mass-depletion path."""
    sim = _make_diff_sim(make_sim, integrator)
    sim.thruster_manager.set_config(max_thrust=1.0, isp=200.0)
    target = wp.vec3d(0.0, A, 0.0)

    cmds = command_sequence(sim, NUM_STEPS)
    for t in range(NUM_STEPS):
        u = cmds[t].numpy()
        u["thrust_cmd"][:, 0] = 0.5   # thruster 0 at half throttle, all steps
        cmds[t].assign(u)

    rollout = sim.differentiable_rollout(NUM_STEPS, commands=cmds)
    _run_loss(sim, rollout, _pos_loss_kernel, [target], taped=True)
    g = rollout.thruster_configs.grad.numpy().copy()

    t0 = sim.thruster_manager.configs.numpy().copy()
    for field, h in [("max_thrust", 1e-3), ("isp", 1.0)]:
        assert abs(g[field][0]) > 0.0
        losses = []
        for sign in (+1.0, -1.0):
            tc = t0.copy()
            tc[field][0] += sign * h
            sim.thruster_manager.configs.assign(tc)
            losses.append(_run_loss(sim, rollout, _pos_loss_kernel, [target], taped=False))
        sim.thruster_manager.configs.assign(t0)
        fd = (losses[0] - losses[1]) / (2.0 * h)
        _fd_vs_tape(g[field][0], fd)


@pytest.mark.parametrize("integrator", ["euler", "rk4", "symplectic"])
def test_inertia_gradients_match_fd(make_sim, integrator):
    """rollout.inertia_grad() (direct + precomputed-inverse path) matches the
    total FD derivative taken through set_config_per_entity, which recomputes
    inertia_inv from the perturbed tensor."""
    sim = _make_diff_sim(make_sim, integrator, ang_mom=[1e-2, 2e-2, 3e-2],
                         gravity_gradient=True)
    # Strictly inside the physical region, so FD perturbations stay valid
    inertia0 = np.tile(np.diag([0.15, 0.2, 0.3]), (sim.num_entities, 1, 1))
    sim.vehicle.set_config_per_entity(inertia=inertia0)
    rollout = sim.differentiable_rollout(NUM_STEPS)

    _run_loss(sim, rollout, _ang_mom_loss_kernel, [], taped=True)
    g = rollout.inertia_grad()

    # Inertia tensors are symmetric, so an off-diagonal perturbation moves
    # both mirrored entries; its derivative is the sum of their gradients
    h = 1e-5
    for row, col in [(0, 0), (0, 1)]:
        tape = g[0][row][col] + (g[0][col][row] if row != col else 0.0)
        assert abs(tape) > 0.0
        losses = []
        for sign in (+1.0, -1.0):
            inertia = inertia0.copy()
            inertia[0][row][col] += sign * h
            if row != col:
                inertia[0][col][row] += sign * h
            sim.vehicle.set_config_per_entity(inertia=inertia)
            losses.append(_run_loss(sim, rollout, _ang_mom_loss_kernel, [], taped=False))
        sim.vehicle.set_config_per_entity(inertia=inertia0)
        fd = (losses[0] - losses[1]) / (2.0 * h)
        _fd_vs_tape(tape, fd)


@pytest.mark.parametrize("integrator", ["euler", "rk4", "symplectic"])
def test_thruster_geometry_gradients_match_fd(make_sim, integrator):
    """dloss/d(thruster position/direction) matches FD (torque/force paths)."""
    sim = _make_diff_sim(make_sim, integrator)
    sim.thruster_manager.set_config(
        max_thrust=0.01, isp=0.0,
        positions=[(0.0, 1.0, 0.0)], directions=[(1.0, 0.0, 0.0)],
    )
    target = wp.vec3d(0.0, A, 0.0)

    cmds = command_sequence(sim, NUM_STEPS)
    for t in range(NUM_STEPS):
        u = cmds[t].numpy()
        u["thrust_cmd"][:, 0] = 0.5
        cmds[t].assign(u)

    rollout = sim.differentiable_rollout(NUM_STEPS, commands=cmds)
    t0 = sim.thruster_manager.configs.numpy().copy()
    h = 1e-4

    # Position enters through the torque arm: angular-momentum loss.
    # Direction enters through the applied force: position loss.
    cases = [("positions", 1, _ang_mom_loss_kernel, []),
             ("directions", 0, _pos_loss_kernel, [target])]
    for field, comp, loss_kernel, loss_args in cases:
        _run_loss(sim, rollout, loss_kernel, loss_args, taped=True)
        g = float(rollout.thruster_configs.grad.numpy()[field][0][0][comp])
        assert abs(g) > 0.0
        losses = []
        for sign in (+1.0, -1.0):
            tc = t0.copy()
            tc[field][0][0][comp] += sign * h
            sim.thruster_manager.configs.assign(tc)
            losses.append(_run_loss(sim, rollout, loss_kernel, loss_args, taped=False))
        sim.thruster_manager.configs.assign(t0)
        fd = (losses[0] - losses[1]) / (2.0 * h)
        _fd_vs_tape(g, fd)


@pytest.mark.parametrize("integrator", ["euler", "rk4", "symplectic"])
def test_magnetic_dipole_gradients_match_fd(make_sim, integrator):
    """dloss/d(magnetic_dipole) matches FD through the magnetic torque."""
    sim = _make_diff_sim(make_sim, integrator, magnetic_torque=True)
    sim.vehicle.set_config(mass_dry=1.0, magnetic_dipole=[0.1, 0.05, 0.0])
    rollout = sim.differentiable_rollout(NUM_STEPS)

    _run_loss(sim, rollout, _ang_mom_loss_kernel, [], taped=True)
    g = float(rollout.vehicle_configs.grad.numpy()["magnetic_dipole"][0][0])
    assert abs(g) > 0.0

    v0 = sim.vehicle.configs.numpy().copy()
    h = 1e-3
    losses = []
    for sign in (+1.0, -1.0):
        v = v0.copy()
        v["magnetic_dipole"][0][0] += sign * h
        sim.vehicle.configs.assign(v)
        losses.append(_run_loss(sim, rollout, _ang_mom_loss_kernel, [], taped=False))
    sim.vehicle.configs.assign(v0)
    fd = (losses[0] - losses[1]) / (2.0 * h)
    _fd_vs_tape(g, fd)


@pytest.mark.parametrize("integrator", ["euler", "rk4", "symplectic"])
def test_rw_param_gradients_match_fd(make_sim, integrator):
    """dloss/d(wheel_inertia, viscous_damping) matches FD (damped wheels)."""
    sim = _make_diff_sim(make_sim, integrator)
    sim.rw_manager.set_config(wheel_inertia=0.01, viscous_damping=1e-3)

    cmds = command_sequence(sim, NUM_STEPS)
    for t in range(NUM_STEPS):
        u = cmds[t].numpy()
        u["rw_torque_cmd"][:, 0] = 1e-3   # constant torque on wheel 0
        cmds[t].assign(u)

    rollout = sim.differentiable_rollout(NUM_STEPS, commands=cmds)
    _run_loss(sim, rollout, _ang_mom_loss_kernel, [], taped=True)
    g = rollout.rw_configs.grad.numpy().copy()

    r0 = sim.rw_manager.configs.numpy().copy()
    for field, h in [("wheel_inertia", 1e-5), ("viscous_damping", 1e-6)]:
        assert abs(g[field][0]) > 0.0
        losses = []
        for sign in (+1.0, -1.0):
            rc = r0.copy()
            rc[field][0] += sign * h
            sim.rw_manager.configs.assign(rc)
            losses.append(_run_loss(sim, rollout, _ang_mom_loss_kernel, [], taped=False))
        sim.rw_manager.configs.assign(r0)
        fd = (losses[0] - losses[1]) / (2.0 * h)
        _fd_vs_tape(g[field][0], fd)


def test_gradient_descent_recovers_drag_coefficient(make_sim):
    """System identification: recover an unknown drag coefficient from an
    observed final position (validates parameter resync via rollout.reset())."""
    sim = _make_diff_sim(make_sim, "symplectic", drag=True)
    sim.vehicle.set_config(mass_dry=1.0, area_drag=100.0, cd=2.2)
    rollout = sim.differentiable_rollout(100)

    # Observed final position under the true drag coefficient
    _run_loss(sim, rollout, _pos_loss_kernel, [wp.vec3d(0.0, 0.0, 0.0)], taped=False)
    target = wp.vec3d(*rollout.final_states.numpy()["pos"][0])

    # Start from a wrong guess and descend on dloss/dcd
    sim.vehicle.set_config(mass_dry=1.0, area_drag=100.0, cd=1.2)
    for _ in range(25):
        loss = _run_loss(sim, rollout, _pos_loss_kernel, [target], taped=True)
        g = rollout.vehicle_configs.grad.numpy()["cd"].copy()
        if loss == 0.0 or abs(g[0]) == 0.0:
            break
        # Newton step for the 1-D residual: loss = n*(s*dcd)^2 and
        # g = 2*s^2*dcd per entity, so dcd = 2*loss / (n*g)
        v = sim.vehicle.configs.numpy().copy()
        v["cd"] -= 2.0 * loss / (sim.num_entities * g)
        sim.vehicle.configs.assign(v)

    np.testing.assert_allclose(sim.vehicle.configs.numpy()["cd"], 2.2, rtol=1e-2)


def test_gradient_descent_retargets_trajectory(make_sim):
    """Optimizing the initial velocity by plain gradient descent moves the
    final position onto a shifted target (end-to-end adjoint signal)."""
    sim = _make_diff_sim(make_sim, "symplectic")
    num_steps, t_final = 100, 100 * DT
    rollout = sim.differentiable_rollout(num_steps)

    # Baseline final position, then aim 10 km further along-track
    base = _run_loss(sim, rollout, _pos_loss_kernel,
                     [wp.vec3d(0.0, 0.0, 0.0)], taped=False)
    p_final = rollout.final_states.numpy()["pos"][0]
    target = wp.vec3d(p_final[0], p_final[1] + 1e4, p_final[2])

    lr = 0.25 * MASS0**2 / t_final**2   # GD contraction for the ballistic sensitivity t/m
    s0 = sim.states.numpy().copy()
    initial_loss = None
    for _ in range(40):
        sim.states.assign(s0)
        loss = _run_loss(sim, rollout, _pos_loss_kernel, [target], taped=True)
        if initial_loss is None:
            initial_loss = loss
        g = rollout.initial_states.grad.numpy()["mom"]
        s0["mom"] = s0["mom"] - lr * g

    assert initial_loss > 0.0
    assert loss < 1e-4 * initial_loss, (
        f"gradient descent failed to converge: {initial_loss:.3e} -> {loss:.3e}"
    )


@pytest.mark.parametrize("num_steps", [0, -3, 2.5, True])
def test_rollout_rejects_bad_num_steps(make_sim, num_steps):
    """num_steps = 0 used to build an empty rollout whose final state is the
    input; a negative or float count failed deep inside list building."""
    sim = make_sim(differentiable=True)
    with pytest.raises(ValueError, match="num_steps must be a positive integer"):
        sim.differentiable_rollout(num_steps)
    with pytest.raises(ValueError, match="num_steps must be a positive integer"):
        command_sequence(sim, num_steps)


def test_rollout_requires_differentiable_flag(make_sim):
    sim = make_sim(num_envs=1)
    with pytest.raises(RuntimeError, match="differentiable=True"):
        sim.differentiable_rollout(5)


def test_rollout_rejects_termination_checks(make_sim):
    sim = make_sim(num_envs=1, differentiable=True, check_bounds=True)
    with pytest.raises(RuntimeError, match="check_bounds"):
        sim.differentiable_rollout(5)


def test_forward_requires_reset_between_runs(make_sim):
    sim = _make_diff_sim(make_sim, "symplectic")
    rollout = sim.differentiable_rollout(5)
    rollout.forward()
    with pytest.raises(RuntimeError, match="reset"):
        rollout.forward()
    rollout.reset()
    rollout.forward()  # fine after reset


def test_rollout_does_not_touch_episode_steps(make_sim):
    """Rollout steps must not advance the simulator's episode counters."""
    sim = _make_diff_sim(make_sim, "symplectic")
    sim.step()
    rollout = sim.differentiable_rollout(5)
    rollout.forward()
    assert np.array_equal(sim.get_episode_steps().numpy(),
                          np.full(sim.num_entities, 1))


def test_backward_independent_of_later_simulator_changes(make_sim):
    """The backward reads only the rollout's copies: a reset (which zeroes the
    commands) or a set_config between forward() and backward() leaves the
    gradients unchanged. With commands=None the backward used to read the
    simulator's live commands."""
    def grads(change_before_backward):
        sim = _make_diff_sim(make_sim, "rk4", ang_mom=[0.02, -0.01, 0.03])
        rollout = sim.differentiable_rollout(5)
        sim.set_rw_torque_commands([0.004, -0.003, 0.002])   # held by forward()
        loss = wp.zeros(1, dtype=wp.float64, device=sim.device, requires_grad=True)
        with rollout.tape:
            rollout.forward()
            wp.launch(_ang_mom_loss_kernel, dim=sim.num_entities,
                      inputs=[rollout.final_states, loss], device=sim.device)
        if change_before_backward:
            sim.reset()
            sim.vehicle.set_config(inertia=np.diag([2.0, 1.5, 3.0]))
        rollout.tape.backward(loss)
        return (rollout.initial_states.grad.numpy()["ang_mom"],
                rollout.act_states[0].grad.numpy()["rw_ang_mom"],
                rollout.inertia_grad())

    for g_ref, g in zip(grads(False), grads(True)):
        assert np.abs(g_ref).max() > 0.0
        np.testing.assert_array_equal(g, g_ref)

"""
Warp <-> torch autograd bridge: differentiable sim steps as torch autograd nodes.

:class:`DiffSimStepper` exposes one simulator integration step as a
``torch.autograd.Function``: flat float64 torch tensors in (state ``x``,
actuator command ``u``), flat float64 torch tensor out (next state), with the
step's warp kernels recorded on a per-call ``wp.Tape`` whose adjoint runs in
the node's backward. Chaining ``step()`` calls under torch autograd yields the
exact interleaved gradient through dynamics *and* anything torch-side in the
loop (a policy ``a_t = pi(o_t)``, a reward, ...) — the foundation for
first-order RL (SHAC).

    sim = SpacecraftSimulator(..., differentiable=True, check_bounds=False)
    stepper = DiffSimStepper(sim)
    x = stepper.read_state()                     # [E, X_DIM] f64, detached
    x.requires_grad_(True)
    for t in range(horizon):
        u = policy(x)                            # [E, U_DIM] f64, on-graph
        x = stepper.step(x, u)                   # differentiable sim step
    loss = objective(x)
    loss.backward()                              # adjoints through all steps

Each differentiable ``step()`` call allocates its own tape and warp buffers
(freed with the autograd graph), so any autograd pattern is safe: repeated
backward with ``retain_graph=True``, ``torch.autograd.gradcheck``'s
interleaved numeric and analytic passes, and slot-free window chaining. When
no gradient can flow (``torch.no_grad()`` or detached inputs), ``step()``
switches to an untaped fast path with reused buffers — evaluation and
inference loops pay no taping cost. Buffer pooling for the differentiable
path is a later optimization if per-step allocation ever shows up in profiles.

Each differentiable step runs on a snapshot of the vehicle, thruster and
wheel configurations, so its backward replays the parameters its forward
used: resetting the simulator (which redraws randomized parameters) or
calling ``set_config`` between a step and its backward does not change the
gradient. A taped step copies the three arrays; inside
:meth:`DiffSimStepper.frozen_params` the steps share one copy instead.

Gradients are first order only: the backward pass is a Warp adjoint, not a
torch graph, so double backward (Hessian-vector products, gradient
penalties) raises instead of treating the step Jacobian as a constant.

Same requirements as :class:`~cosmic_warp.core.autodiff.DifferentiableRollout`:
the simulator must be built with ``differentiable=True`` and termination checks
disabled, everything is float64, and taped steps cannot be CUDA-graph captured.
"""
from contextlib import contextmanager
from dataclasses import replace

import torch
import warp as wp

from cosmic_warp.core.pipeline import SpacecraftSimulator
from cosmic_warp.core.state import State
from cosmic_warp.actuators.base import ActuatorCommand, ActuatorState

# Kernels in this module need adjoints (they sit on the tape).
wp.set_module_options({"enable_backward": True})

#: Flat state layout [E, X_DIM] (f64):
#: pos(3), mom(3), mass(1), quat(4, xyzw), ang_mom(3)  -- State
#: rw_ang_mom(4)                                        -- ActuatorState
X_DIM = 18
#: Flat command layout [E, U_DIM] (f64, physical units):
#: thrust throttles(8, slots 0-7), rw_torque_cmd(4)    -- ActuatorCommand
U_DIM = 12


@wp.kernel
def _unpack_x_kernel(
    x: wp.array(dtype=wp.float64, ndim=2),        # [E, X_DIM]
    states: wp.array(dtype=State),
    act_states: wp.array(dtype=ActuatorState),
):
    """Flat state row -> State/ActuatorState structs (per-field, adjoint-safe)."""
    tid = wp.tid()
    states[tid].pos = wp.vec3d(x[tid, 0], x[tid, 1], x[tid, 2])
    states[tid].mom = wp.vec3d(x[tid, 3], x[tid, 4], x[tid, 5])
    states[tid].mass = x[tid, 6]
    states[tid].quat = wp.quatd(x[tid, 7], x[tid, 8], x[tid, 9], x[tid, 10])
    states[tid].ang_mom = wp.vec3d(x[tid, 11], x[tid, 12], x[tid, 13])
    act_states[tid].rw_ang_mom = wp.vec4d(x[tid, 14], x[tid, 15], x[tid, 16], x[tid, 17])


@wp.kernel
def _pack_x_kernel(
    states: wp.array(dtype=State),
    act_states: wp.array(dtype=ActuatorState),
    x: wp.array(dtype=wp.float64, ndim=2),        # [E, X_DIM]
):
    """State/ActuatorState structs -> flat state row (per-field, adjoint-safe)."""
    tid = wp.tid()
    p = states[tid].pos
    m = states[tid].mom
    q = states[tid].quat
    L = states[tid].ang_mom
    h = act_states[tid].rw_ang_mom
    x[tid, 0] = p[0]
    x[tid, 1] = p[1]
    x[tid, 2] = p[2]
    x[tid, 3] = m[0]
    x[tid, 4] = m[1]
    x[tid, 5] = m[2]
    x[tid, 6] = states[tid].mass
    x[tid, 7] = q[0]
    x[tid, 8] = q[1]
    x[tid, 9] = q[2]
    x[tid, 10] = q[3]
    x[tid, 11] = L[0]
    x[tid, 12] = L[1]
    x[tid, 13] = L[2]
    x[tid, 14] = h[0]
    x[tid, 15] = h[1]
    x[tid, 16] = h[2]
    x[tid, 17] = h[3]


@wp.kernel
def _unpack_u_kernel(
    u: wp.array(dtype=wp.float64, ndim=2),        # [E, U_DIM]
    commands: wp.array(dtype=ActuatorCommand),
):
    """Flat command row -> ActuatorCommand structs (per-field, adjoint-safe)."""
    tid = wp.tid()
    commands[tid].thrust_cmd = wp.vec4d(u[tid, 0], u[tid, 1], u[tid, 2], u[tid, 3])
    commands[tid].thrust_cmd_ext = wp.vec4d(u[tid, 4], u[tid, 5], u[tid, 6], u[tid, 7])
    commands[tid].rw_torque_cmd = wp.vec4d(u[tid, 8], u[tid, 9], u[tid, 10], u[tid, 11])


class _StepBuffers:
    """Warp buffers for one step launch sequence (flat in/out + struct scratch)."""

    def __init__(self, num_entities: int, device, requires_grad: bool):
        n, d, g = num_entities, device, requires_grad
        self.x_in = wp.zeros((n, X_DIM), dtype=wp.float64, device=d, requires_grad=g)
        self.x_out = wp.zeros((n, X_DIM), dtype=wp.float64, device=d, requires_grad=g)
        self.u = wp.zeros((n, U_DIM), dtype=wp.float64, device=d, requires_grad=g)
        self.state_in = wp.zeros(n, dtype=State, device=d, requires_grad=g)
        self.state_out = wp.zeros(n, dtype=State, device=d, requires_grad=g)
        self.act_in = wp.zeros(n, dtype=ActuatorState, device=d, requires_grad=g)
        self.act_out = wp.zeros(n, dtype=ActuatorState, device=d, requires_grad=g)
        self.cmd = wp.zeros(n, dtype=ActuatorCommand, device=d, requires_grad=g)


class _StepRecord(_StepBuffers):
    """Per-call tape and warp buffers for one recorded step.

    Allocated fresh in every differentiable forward so the tape never
    references a buffer that a later step overwrites (the invariant
    DifferentiableRollout keeps with distinct per-step arrays); freed with
    the autograd graph.
    """

    def __init__(self, num_entities: int, device, ctx):
        super().__init__(num_entities, device, requires_grad=True)
        self.tape = wp.Tape()
        self.ctx = ctx   # the context the step ran on, parameter snapshot included


class _SimStep(torch.autograd.Function):
    """One integrator step: (x_t, u_t) -> x_{t+1}, adjoint via the warp tape."""

    @staticmethod
    def forward(ctx, x, u, stepper):
        rec = _StepRecord(stepper.num_entities, stepper.device, stepper._taped_ctx())

        wp.to_torch(rec.x_in).copy_(x.detach())
        wp.to_torch(rec.u).copy_(u.detach())

        with rec.tape:
            stepper._launch_sequence(rec, rec.ctx)

        ctx.rec = rec
        return wp.to_torch(rec.x_out).clone()

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_x_next):
        rec = ctx.rec
        # Zero all tape adjoints so repeated backward (retain_graph=True,
        # gradcheck) does not accumulate stale seeds, then seed the output.
        rec.tape.zero()
        wp.to_torch(rec.x_out.grad).copy_(grad_x_next)
        rec.tape.backward(grads={rec.x_out: rec.x_out.grad})
        grad_x = wp.to_torch(rec.x_in.grad).clone()
        grad_u = wp.to_torch(rec.u.grad).clone()
        return grad_x, grad_u, None


class DiffSimStepper:
    """Differentiable per-step interface to a simulator (torch boundary).

    Args:
        sim: A :class:`~cosmic_warp.core.pipeline.SpacecraftSimulator` built
            with ``differentiable=True`` and termination checks disabled
            (``check_bounds=False``, ``check_planet_impact=False``): the
            integrator freezes terminated entities, which would silently kill
            their gradient paths.

    The stepper never mutates the simulator: taped steps use a scratch
    ``episode_steps`` counter and their own state buffers. Use
    :meth:`read_state` / :meth:`write_state` to move state between the
    simulator (resets, evaluation) and the differentiable chain.

    A taped step reads the vehicle, thruster and wheel configurations from a
    copy taken at its forward, which its backward reuses; the simulator may
    be reset or reconfigured before ``backward()``.
    """

    def __init__(self, sim: SpacecraftSimulator):
        if not getattr(sim, "differentiable", False):
            raise RuntimeError(
                "Simulator was not created with differentiable=True; adjoint "
                "kernels for its integrator are not compiled."
            )
        if sim._settings["check_bounds"] or sim._settings["check_planet_impact"]:
            raise RuntimeError(
                "DiffSimStepper requires check_bounds=False and "
                "check_planet_impact=False: a terminated entity is frozen by "
                "the integrator and its gradients silently vanish."
            )
        self.sim = sim
        self.num_entities = sim.num_entities
        self.device = sim.device

        # Scratch episode counters and termination status: the integrator
        # kernels write both (the non-finite check is always on), and taped
        # steps must never mutate the simulator's own.
        self._ctx = replace(
            sim.ctx,
            episode_steps=wp.zeros(self.num_entities, dtype=wp.int32, device=self.device),
            termination_status=wp.empty_like(sim.ctx.termination_status),
        )
        # Parameter snapshot shared by the taped steps inside frozen_params()
        self._frozen_ctx = None
        # Reused (non-taped) buffer for read_state/write_state.
        self._x_io = wp.zeros(
            (self.num_entities, X_DIM), dtype=wp.float64, device=self.device
        )
        # Reused untaped buffers for the no-grad fast path (lazy: only
        # evaluation/inference loops pay for them).
        self._eval_buf = None

    def step(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Advance one step: ``(x, u) -> x_next``, differentiable w.r.t. both.

        When no gradient can flow — under ``torch.no_grad()``, or when neither
        input requires grad — the step skips taping and per-call allocation
        entirely (same kernels, reused buffers), so evaluation and inference
        loops run at plain simulation cost.

        Args:
            x: State ``[num_entities, X_DIM]`` float64 (see :data:`X_DIM` layout).
            u: Actuator commands ``[num_entities, U_DIM]`` float64, physical
                units (see :data:`U_DIM` layout).
        """
        self._check("x", x, X_DIM)
        self._check("u", u, U_DIM)
        if torch.is_grad_enabled() and (x.requires_grad or u.requires_grad):
            return _SimStep.apply(x, u, self)
        return self._step_no_grad(x, u)

    @contextmanager
    def frozen_params(self):
        """Share one parameter snapshot across the taped steps in the block.

        Every taped :meth:`step` otherwise copies the vehicle, thruster and
        wheel configurations for its backward. Inside the block they share
        one copy, taken on entry: cheaper for a multi-step rollout, provided
        nothing changes the configuration inside the block. Nested blocks
        share the outer snapshot.
        """
        if self._frozen_ctx is not None:
            yield
            return
        self._frozen_ctx = self._snapshot_ctx()
        try:
            yield
        finally:
            self._frozen_ctx = None

    def _snapshot_ctx(self):
        """The scratch context with copies of the configuration arrays (the
        parameters a step's adjoint reads), fresh so later writes to the
        simulator's arrays cannot reach a recorded tape."""
        ctx = self._ctx
        params = {}
        for name in ("vehicles", "thruster_configs", "rw_configs"):
            live = getattr(self.sim.ctx, name)
            copy = wp.empty_like(live)
            wp.copy(copy, live)
            params[name] = copy
        return replace(ctx, **params)

    def _taped_ctx(self):
        """The context a taped step runs on."""
        return self._frozen_ctx if self._frozen_ctx is not None else self._snapshot_ctx()

    def _launch_sequence(self, b: "_StepBuffers", ctx) -> None:
        """The step's kernel sequence: unpack x/u -> integrate -> pack x."""
        # Each step starts from the simulator's status (resets clear it)
        wp.copy(ctx.termination_status, self.sim.ctx.termination_status)
        wp.launch(_unpack_x_kernel, dim=self.num_entities,
                  inputs=[b.x_in, b.state_in, b.act_in], device=self.device)
        wp.launch(_unpack_u_kernel, dim=self.num_entities,
                  inputs=[b.u, b.cmd], device=self.device)
        self.sim.integrator.launch_step(
            ctx, self.sim.dt,
            b.state_in, b.state_out,
            b.act_in, b.act_out,
            b.cmd,
        )
        wp.launch(_pack_x_kernel, dim=self.num_entities,
                  inputs=[b.state_out, b.act_out, b.x_out], device=self.device)

    def _step_no_grad(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Untaped fast path: identical kernel sequence, reused buffers."""
        if self._eval_buf is None:
            self._eval_buf = _StepBuffers(self.num_entities, self.device,
                                          requires_grad=False)
        b = self._eval_buf
        wp.to_torch(b.x_in).copy_(x.detach())
        wp.to_torch(b.u).copy_(u.detach())
        self._launch_sequence(b, self._ctx)   # no backward: live parameters
        return wp.to_torch(b.x_out).clone()

    def read_state(self) -> torch.Tensor:
        """The simulator's current state as a detached ``[E, X_DIM]`` tensor."""
        wp.launch(_pack_x_kernel, dim=self.num_entities,
                  inputs=[self.sim.ctx.states, self.sim.ctx.actuator_states, self._x_io],
                  device=self.device)
        return wp.to_torch(self._x_io).clone()

    def write_state(self, x: torch.Tensor) -> None:
        """Write a (detached) ``[E, X_DIM]`` tensor into the simulator's state."""
        self._check("x", x, X_DIM)
        wp.to_torch(self._x_io).copy_(x.detach())
        wp.launch(_unpack_x_kernel, dim=self.num_entities,
                  inputs=[self._x_io, self.sim.ctx.states, self.sim.ctx.actuator_states],
                  device=self.device)

    def _check(self, name: str, t: torch.Tensor, dim: int) -> None:
        if t.dtype != torch.float64:
            raise TypeError(f"{name} must be float64 (physics precision), got {t.dtype}")
        if t.shape != (self.num_entities, dim):
            raise ValueError(
                f"{name} must have shape ({self.num_entities}, {dim}), got {tuple(t.shape)}"
            )

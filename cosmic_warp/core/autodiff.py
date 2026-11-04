"""
Differentiable rollouts: reverse-mode gradients through the dynamics.

Records a multi-step simulation on a ``wp.Tape`` and backpropagates a scalar
loss to the initial state, the vehicle parameters (system identification) and
(optionally) a per-step control sequence.
Enabled per simulator with ``SpacecraftSimulator(differentiable=True)``;
the default (non-differentiable) path is untouched — no adjoint kernels are
compiled and no extra memory is allocated unless opted in.

Usage::

    sim = SpacecraftSimulator(num_envs=4, differentiable=True,
                              check_bounds=False)
    sim.reset_to_conditions(positions=..., momenta=...)

    commands = command_sequence(sim, num_steps=100)   # optional controls
    rollout = sim.differentiable_rollout(num_steps=100, commands=commands)

    loss = wp.zeros(1, dtype=wp.float64, device=sim.device, requires_grad=True)
    with rollout.tape:
        rollout.forward()
        wp.launch(my_loss_kernel, dim=sim.num_entities,
                  inputs=[rollout.final_states, loss], device=sim.device)
    rollout.tape.backward(loss)

    dl_dx0 = rollout.initial_states.grad.numpy()          # structured (pos/mom/...)
    dl_du0 = commands[0].grad.numpy()                     # structured (thrust_cmd/...)
    dl_dcd = rollout.vehicle_configs.grad.numpy()["cd"]   # vehicle parameters

Memory: the rollout keeps ``num_steps + 1`` copies of the dynamic and
actuator state (plus matching gradient buffers), i.e. roughly
``2 * (num_steps + 1) * num_entities * 144`` bytes on the device, plus
three horizon-independent parameter shadow arrays (negligible). For long
horizons split the trajectory into segments and chain the gradients
(checkpointing) — not provided here.

Requirements:

- Termination checks must be disabled (``check_bounds=False``,
  ``check_planet_impact=False``): the fused termination write would alias
  across taped launches and freeze gradient flow for terminated entities.
- The rollout never mutates the simulator, and its backward reads only the
  rollout's own copies (state, parameters, and the held commands when
  ``commands`` is None), so the simulator may be reset or reconfigured
  between ``forward()`` and ``backward()``.
"""
import sys
from dataclasses import replace

import numpy as np
import warp as wp

from cosmic_warp.core import validation
from cosmic_warp.core.state import State
from cosmic_warp.core.vehicle import VehicleConfig
from cosmic_warp.actuators.base import ActuatorCommand, ActuatorState
from cosmic_warp.actuators.thrusters import ThrusterConfig
from cosmic_warp.actuators.reaction_wheels import ReactionWheelConfig


def enable_backward_for(integrator) -> None:
    """Enable adjoint code generation for the integrator's kernel module.

    Warp then (re)compiles the module with its adjoint, roughly doubling that
    module's one-off JIT cost. The option is module-wide and stays on for the
    rest of the process, so later simulators using the same integrator get
    the adjoint too; modules of unused integrators are unaffected.
    """
    module = sys.modules[type(integrator).__module__]
    wp.set_module_options({"enable_backward": True}, module=module)


def command_sequence(sim, num_steps: int) -> list:
    """Allocate a differentiable per-step control sequence.

    Returns a list of ``num_steps`` zero-initialized ``ActuatorCommand``
    arrays (one per step, each of length ``num_entities``) with
    ``requires_grad=True``. Populate them via their structured numpy view::

        cmds = command_sequence(sim, 50)
        u = cmds[0].numpy()
        u["thrust_cmd"][:, 0] = 0.5    # thruster 0 at half throttle, step 0
        cmds[0].assign(u)

    After ``tape.backward(loss)`` each ``cmds[t].grad`` holds dloss/du_t.
    """
    num_steps = validation.positive_int("num_steps", num_steps)
    return [
        wp.zeros(sim.num_entities, dtype=ActuatorCommand, device=sim.device,
                 requires_grad=True)
        for _ in range(num_steps)
    ]


class DifferentiableRollout:
    """A taped multi-step rollout with per-step state history.

    Allocated once and reusable across optimization iterations via
    :meth:`reset`. The forward pass reads the simulator's *current* state as
    the initial condition; the simulator itself is never modified.
    """

    def __init__(self, sim, num_steps: int, commands: list = None):
        if not getattr(sim, "differentiable", False):
            raise RuntimeError(
                "Simulator was not created with differentiable=True; adjoint "
                "kernels for its integrator are not compiled."
            )
        if sim._settings["check_bounds"] or sim._settings["check_planet_impact"]:
            raise RuntimeError(
                "Differentiable rollouts require check_bounds=False and "
                "check_planet_impact=False: termination writes alias across "
                "taped steps and corrupt the backward pass."
            )
        num_steps = validation.positive_int("num_steps", num_steps)
        if commands is not None and len(commands) != num_steps:
            raise ValueError(
                f"commands must have one entry per step "
                f"({len(commands)} given, {num_steps} steps)"
            )

        self.sim = sim
        self.num_steps = num_steps
        self.commands = commands
        self.tape = wp.Tape()

        n, dev = sim.num_entities, sim.device
        # Per-step state history: distinct buffers so the tape never sees an
        # array overwritten after being read (which corrupts adjoints).
        self.states = [
            wp.zeros(n, dtype=State, device=dev, requires_grad=True)
            for _ in range(num_steps + 1)
        ]
        self.act_states = [
            wp.zeros(n, dtype=ActuatorState, device=dev, requires_grad=True)
            for _ in range(num_steps + 1)
        ]

        # Differentiable shadow copies of the vehicle-parameter arrays
        # (system identification): after backward, each array's .grad holds
        # dloss/d(parameter), accumulated over all steps.
        self.vehicle_configs = wp.zeros(
            n, dtype=VehicleConfig, device=dev, requires_grad=True
        )
        self.thruster_configs = wp.zeros(
            n, dtype=ThrusterConfig, device=dev, requires_grad=True
        )
        self.rw_configs = wp.zeros(
            n, dtype=ReactionWheelConfig, device=dev, requires_grad=True
        )
        # Without a command sequence, a copy of the simulator's commands taken
        # by forward() and held for every step, so the backward does not read
        # commands a reset (which zeroes them) or a setter changed meanwhile
        self._held_commands = (
            wp.zeros(n, dtype=ActuatorCommand, device=dev) if commands is None else None
        )

        # Private context with scratch episode counters and termination status
        # (the integrator kernels write both, and the always-on non-finite
        # check can fire; the rollout must never mutate the simulator's own
        # state) and the differentiable parameter shadows.
        self._ctx = replace(
            sim.ctx,
            episode_steps=wp.zeros(n, dtype=wp.int32, device=dev),
            termination_status=wp.empty_like(sim.ctx.termination_status),
            vehicles=self.vehicle_configs,
            thruster_configs=self.thruster_configs,
            rw_configs=self.rw_configs,
        )

        self._forward_done = False
        self._load_initial_state()

    def _load_initial_state(self) -> None:
        """Copy the simulator's current state and parameters into the rollout."""
        wp.copy(self.states[0], self.sim.states)
        wp.copy(self.act_states[0], self.sim.actuators.states)
        wp.copy(self._ctx.termination_status, self.sim.ctx.termination_status)
        wp.copy(self.vehicle_configs, self.sim.ctx.vehicles)
        wp.copy(self.thruster_configs, self.sim.ctx.thruster_configs)
        wp.copy(self.rw_configs, self.sim.ctx.rw_configs)

    def reset(self) -> None:
        """Prepare for another taped forward pass (optimization loops).

        Clears the tape, zeroes all gradients and re-reads the simulator's
        current state and vehicle parameters — update parameters on the
        simulator (e.g. ``sim.vehicle.set_config(...)``) between iterations.
        """
        self.tape.zero()
        self.tape.reset()
        self._forward_done = False
        self._load_initial_state()

    def forward(self) -> None:
        """Run the rollout, filling the per-step state history.

        Call inside a ``with rollout.tape:`` block so the launches are
        recorded; launch any loss kernels in the same block.
        """
        if self._forward_done:
            raise RuntimeError(
                "forward() already ran on this tape; call reset() before "
                "re-running (the tape would otherwise record the rollout twice)."
            )
        sim = self.sim
        if self.commands is None:
            wp.copy(self._held_commands, sim.ctx.actuator_commands)
        for t in range(self.num_steps):
            cmd = self.commands[t] if self.commands is not None else self._held_commands
            sim.integrator.launch_step(
                self._ctx, sim.dt,
                self.states[t], self.states[t + 1],
                self.act_states[t], self.act_states[t + 1],
                cmd,
            )
        self._forward_done = True

    def inertia_grad(self) -> np.ndarray:
        """Total dloss/d(inertia), shape ``(num_entities, 3, 3)``.

        The kernels read both ``inertia`` and the precomputed ``inertia_inv``
        field, whose inversion happens at config-write time — outside the
        tape. This combines the direct ``inertia`` gradient with the path
        through the inverse using the analytic adjoint of matrix inversion,
        dL/dI -= I^-T (dL/dI^-1) I^-T.
        """
        v = self.vehicle_configs.numpy()
        g = self.vehicle_configs.grad.numpy()
        i_inv = v["inertia_inv"]
        return g["inertia"] - np.einsum(
            "nji,njk,nlk->nil", i_inv, g["inertia_inv"], i_inv
        )

    @property
    def initial_states(self) -> wp.array:
        """State structs at step 0; ``.grad`` holds dloss/dx0 after backward."""
        return self.states[0]

    @property
    def final_states(self) -> wp.array:
        """State structs after the last step (input for loss kernels)."""
        return self.states[self.num_steps]

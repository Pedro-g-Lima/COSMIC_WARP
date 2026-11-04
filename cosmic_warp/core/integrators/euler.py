"""
Forward Euler integrator implementation.
"""
import warp as wp

# Adjoint codegen is skipped by default (halves JIT compile cost of the fused
# megakernels); SpacecraftSimulator(differentiable=True) re-enables it for
# this module before the first launch (see core/autodiff.py).
wp.set_module_options({"enable_backward": False})

from cosmic_warp.core.integrators.base import (
    Integrator,
    compute_state_derivative,
    state_add_scaled,
    store_state,
    State,
    VehicleConfig,
    TerminationConditions,
    TerminationStatus,
    REASON_NONFINITE,
    SimulationFlags,
    ActuatorCommand,
    ActuatorState,
    command_is_finite,
    ReactionWheelConfig,
    ThrusterConfig,
    thruster_wrench,
    compute_forces,
    compute_torques,
    integrate_reaction_wheels,
    compute_rw_momentum_delta,
    update_termination_status
)


class EulerIntegrator(Integrator):
    """
    Forward Euler integrator (first-order).

    Simple and fast but least accurate. Suitable for:
    - Debugging and prototyping
    - Very small time steps
    - Non-stiff systems
    """

    def launch_step(self, ctx, dt, states_in, states_out, act_states_in, act_states_out, commands):
        wp.launch(
            kernel=euler_step_kernel_masked,
            dim=self.num_entities,
            inputs=[states_in, states_out, ctx.vehicles, commands,
                    act_states_in, act_states_out,
                    ctx.thruster_configs, ctx.rw_configs, ctx.flags, dt,
                    ctx.termination_status, ctx.termination_conditions,
                    ctx.episode_steps],
            device=self.device
        )

    @property
    def order(self) -> int:
        """First-order method."""
        return 1

    @property
    def name(self) -> str:
        """Method name."""
        return "Forward Euler"


@wp.kernel
def euler_step_kernel_masked(
    states_in: wp.array(dtype=State),
    states_out: wp.array(dtype=State),
    vehicles: wp.array(dtype=VehicleConfig),
    actuator_commands: wp.array(dtype=ActuatorCommand),
    act_states_in: wp.array(dtype=ActuatorState),
    act_states_out: wp.array(dtype=ActuatorState),
    thruster_configs: wp.array(dtype=ThrusterConfig),
    rw_configs: wp.array(dtype=ReactionWheelConfig),
    flags: wp.array(dtype=SimulationFlags),
    dt: wp.float64,
    status: wp.array(dtype=TerminationStatus),
    conditions: wp.array(dtype=TerminationConditions),
    episode_steps: wp.array(dtype=wp.int32),
):
    tid = wp.tid()

    # Episode step counter fused here (saves a dedicated kernel launch);
    # counts all entities, matching the previous separate-kernel behavior
    episode_steps[tid] = episode_steps[tid] + 1

    if status[tid].terminated:
        # Frozen until reset: carry the state over (a no-op when stepping in
        # place; rollouts and the torch bridge use separate output buffers)
        states_out[tid] = states_in[tid]
        act_states_out[tid] = act_states_in[tid]
        return

    # A NaN or Inf command (e.g. a diverged policy) would be clamped to a
    # finite actuation: terminate with the non-finite reason instead, the
    # state carried over unchanged
    if not command_is_finite(actuator_commands[tid]):
        states_out[tid] = states_in[tid]
        act_states_out[tid] = act_states_in[tid]
        status[tid].terminated = True
        status[tid].reason = REASON_NONFINITE
        return

    # --- Load (structs are read-only locals from here on) ---
    state = states_in[tid]
    vehicle = vehicles[tid]
    flag = flags[0]
    act_cmd = actuator_commands[tid]
    act_state = act_states_in[tid]
    thr_config = thruster_configs[tid]
    rw_config = rw_configs[tid]

    # ============================================================
    # External forces/torques (thruster wrench computed in one pass)
    # ============================================================

    thr = thruster_wrench(thr_config, act_cmd, state, vehicle, dt)
    F_tot = compute_forces(state, vehicle, flag, thr.force)
    M_ext = compute_torques(state, vehicle, flag, thr.torque)

    # ============================================================
    # Reaction wheel update (analytical)
    # ============================================================

    # Update all 4 wheels independently (unused wheels have zero spin axes)
    L_rw_new = integrate_reaction_wheels(rw_config, act_cmd, act_state, dt)
    dL_rw = compute_rw_momentum_delta(L_rw_new, act_state, rw_config)
    M_rw_equiv = -dL_rw / dt

    # ============================================================
    # Rigid-body derivatives
    # ============================================================

    total_torque = M_ext + M_rw_equiv

    deriv = compute_state_derivative(
        state,
        vehicle,
        F_tot,
        total_torque,
        thr.mdot,
        act_state,
        rw_config
    )

    # ============================================================
    # Euler update
    # ============================================================

    stepped = state_add_scaled(
        state,
        vehicle,
        deriv.mom_dot,
        deriv.pos_dot,
        deriv.ang_mom_dot,
        deriv.quat_dot,
        deriv.mass_dot,
        dt
    )

    # The mass takes its exact step-end value (dry exactly on burnout)
    new_state = State(stepped.pos, stepped.mom, thr.mass_end, stepped.quat, stepped.ang_mom)

    store_state(states_out, tid, new_state)
    act_states_out[tid].rw_ang_mom = L_rw_new

    # Only flag this entity; Integrator.step() reduces multi-body status later.
    update_termination_status(new_state, conditions[0], flag, status, tid)

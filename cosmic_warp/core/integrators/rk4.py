"""
Runge-Kutta 4th order (RK4) integrator implementation.
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


class RK4Integrator(Integrator):
    """
    Classic 4th-order Runge-Kutta integrator (RK4).

    Good balance of accuracy and computational cost.
    Uses 4 function evaluations per step.

    The method computes:
        k1 = f(t, y)
        k2 = f(t + dt/2, y + dt/2 * k1)
        k3 = f(t + dt/2, y + dt/2 * k2)
        k4 = f(t + dt, y + dt * k3)
        y(t+dt) = y(t) + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    Reaction wheel momenta h(t) follow a closed-form solution within the
    step, so each stage evaluates them at its own time (0, dt/2, dt). The RK
    stages advance the body-frame total angular momentum L + h, whose rate
    tau_ext - omega x (L + h) needs no wheel torque; L is recovered by
    subtracting h at the stage time. This keeps the wheel/body coupling
    fourth-order.
    """

    def launch_step(self, ctx, dt, states_in, states_out, act_states_in, act_states_out, commands):
        wp.launch(
            kernel=rk4_step_kernel_masked,
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
        """Fourth-order method."""
        return 4

    @property
    def name(self) -> str:
        """Method name."""
        return "Runge-Kutta 4"


@wp.kernel
def rk4_step_kernel_masked(
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
    state0 = states_in[tid]
    vehicle = vehicles[tid]
    flag = flags[0]
    act_cmd = actuator_commands[tid]
    act_state = act_states_in[tid]
    thr_config = thruster_configs[tid]
    rw_config = rw_configs[tid]

    half_dt = wp.float64(0.5) * dt
    sixth = wp.float64(1.0) / wp.float64(6.0)

    # ============================================================
    # Reaction wheel momenta at the stage times (closed form)
    # ============================================================

    # Unused wheels have zero spin axes and contribute nothing
    L_rw_half = integrate_reaction_wheels(rw_config, act_cmd, act_state, half_dt)
    L_rw_new = integrate_reaction_wheels(rw_config, act_cmd, act_state, dt)
    act_half = ActuatorState(L_rw_half)
    act_new = ActuatorState(L_rw_new)

    # Wheel momentum gained since the step start, body frame. The body loses
    # exactly this much (total L + h is advanced by the stages).
    dL_rw_half = compute_rw_momentum_delta(L_rw_half, act_state, rw_config)
    dL_rw = compute_rw_momentum_delta(L_rw_new, act_state, rw_config)

    # Body-frame thrust, torque and mass flow are constant over the step
    # (scaled by the burn fraction on the burnout step), so the stages see a
    # smooth right-hand side
    thr = thruster_wrench(thr_config, act_cmd, state0, vehicle, dt)

    # ============================================================
    # k1
    # ============================================================

    F1 = compute_forces(state0, vehicle, flag, thr.force)
    M1 = compute_torques(state0, vehicle, flag, thr.torque)
    k1 = compute_state_derivative(state0, vehicle, F1, M1, thr.mdot, act_state, rw_config)

    # ============================================================
    # k2
    # ============================================================

    s2_tot = state_add_scaled(
        state0,
        vehicle,
        k1.mom_dot,
        k1.pos_dot,
        k1.ang_mom_dot,
        k1.quat_dot,
        k1.mass_dot,
        half_dt
    )
    s2 = State(s2_tot.pos, s2_tot.mom, s2_tot.mass, s2_tot.quat, s2_tot.ang_mom - dL_rw_half)

    F2 = compute_forces(s2, vehicle, flag, thr.force)
    M2 = compute_torques(s2, vehicle, flag, thr.torque)
    k2 = compute_state_derivative(s2, vehicle, F2, M2, thr.mdot, act_half, rw_config)

    # ============================================================
    # k3
    # ============================================================

    s3_tot = state_add_scaled(
        state0,
        vehicle,
        k2.mom_dot,
        k2.pos_dot,
        k2.ang_mom_dot,
        k2.quat_dot,
        k2.mass_dot,
        half_dt
    )
    s3 = State(s3_tot.pos, s3_tot.mom, s3_tot.mass, s3_tot.quat, s3_tot.ang_mom - dL_rw_half)

    F3 = compute_forces(s3, vehicle, flag, thr.force)
    M3 = compute_torques(s3, vehicle, flag, thr.torque)
    k3 = compute_state_derivative(s3, vehicle, F3, M3, thr.mdot, act_half, rw_config)

    # ============================================================
    # k4
    # ============================================================

    s4_tot = state_add_scaled(
        state0,
        vehicle,
        k3.mom_dot,
        k3.pos_dot,
        k3.ang_mom_dot,
        k3.quat_dot,
        k3.mass_dot,
        dt
    )
    s4 = State(s4_tot.pos, s4_tot.mom, s4_tot.mass, s4_tot.quat, s4_tot.ang_mom - dL_rw)

    F4 = compute_forces(s4, vehicle, flag, thr.force)
    M4 = compute_torques(s4, vehicle, flag, thr.torque)
    k4 = compute_state_derivative(s4, vehicle, F4, M4, thr.mdot, act_new, rw_config)

    # ============================================================
    # Final RK4 combination
    # ============================================================

    two = wp.float64(2.0)

    new_tot = state_add_scaled(
        state0,
        vehicle,
        (k1.mom_dot + two*k2.mom_dot + two*k3.mom_dot + k4.mom_dot) * sixth,
        (k1.pos_dot + two*k2.pos_dot + two*k3.pos_dot + k4.pos_dot) * sixth,
        (k1.ang_mom_dot + two*k2.ang_mom_dot + two*k3.ang_mom_dot + k4.ang_mom_dot) * sixth,
        wp.quatd(
            (k1.quat_dot[0] + two*k2.quat_dot[0] + two*k3.quat_dot[0] + k4.quat_dot[0]) * sixth,
            (k1.quat_dot[1] + two*k2.quat_dot[1] + two*k3.quat_dot[1] + k4.quat_dot[1]) * sixth,
            (k1.quat_dot[2] + two*k2.quat_dot[2] + two*k3.quat_dot[2] + k4.quat_dot[2]) * sixth,
            (k1.quat_dot[3] + two*k2.quat_dot[3] + two*k3.quat_dot[3] + k4.quat_dot[3]) * sixth,
        ),
        (k1.mass_dot + two*k2.mass_dot + two*k3.mass_dot + k4.mass_dot) * sixth,
        dt
    )

    # The mass takes its exact step-end value (dry exactly on burnout)
    new_state = State(new_tot.pos, new_tot.mom, thr.mass_end, new_tot.quat,
                      new_tot.ang_mom - dL_rw)

    store_state(states_out, tid, new_state)
    act_states_out[tid].rw_ang_mom = L_rw_new

    # Only flag this entity; Integrator.step() reduces multi-body status later.
    update_termination_status(new_state, conditions[0], flag, status, tid)

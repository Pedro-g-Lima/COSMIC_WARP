"""
Symplectic integrator implementation for Hamiltonian systems.

This integrator preserves the symplectic structure of Hamiltonian mechanics,
making it ideal for long-term orbital simulations and energy conservation.
"""
import warp as wp

# Adjoint codegen is skipped by default (halves JIT compile cost of the fused
# megakernels); SpacecraftSimulator(differentiable=True) re-enables it for
# this module before the first launch (see core/autodiff.py).
wp.set_module_options({"enable_backward": False})

from cosmic_warp.core.integrators.base import (
    Integrator,
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
    compute_torques,
    integrate_reaction_wheels,
    compute_rw_momentum_delta,
    compute_rw_momentum_vec,
    update_termination_status
)
from cosmic_warp.physics.aggregation import compute_nongravity_forces
from cosmic_warp.physics.gravity import gravity_force


class SymplecticIntegrator(Integrator):
    """
    Symplectic Integrator (second-order, kick-drift-kick).

    More accurate and stable for long-term simulations of mechanical systems. Suitable for:
    - Long-duration orbital simulations
    - Systems with conserved quantities (e.g., energy, momentum)
    - Scenarios where stability over many time steps is critical

    Accuracy notes:
    - The scheme is symplectic only for constant mass. While thrusters fire,
      propellant consumption makes the system non-Hamiltonian and energy is
      (correctly) not conserved. Each kick solves the constant-force
      variable-mass flow exactly, so burns stay second-order.
    - The rotational drift is the explicit midpoint rule on the rotation
      group: omega is taken at the half-step attitude. Total angular momentum
      is conserved exactly in the inertial frame; the attitude error is
      second-order in dt.
    - Velocity-dependent forces (drag) are evaluated at the start of each
      kick, which is first-order in their (usually tiny) contribution.
    """

    def launch_step(self, ctx, dt, states_in, states_out, act_states_in, act_states_out, commands):
        wp.launch(
            kernel=symplectic_step_kernel_masked,
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
        """Second-order method."""
        return 2

    @property
    def name(self) -> str:
        """Method name."""
        return "Symplectic Integrator"


@wp.kernel
def symplectic_step_kernel_masked(
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
    # Kick-drift-kick in single-assignment form: every intermediate quantity
    # gets its own phase-suffixed local (_k1 first kick, _d drift, _k2 second
    # kick) and structs are only built via their constructors — struct field
    # mutation breaks Warp's adjoint generation.
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
    s = states_in[tid]
    vehicle = vehicles[tid]
    flag = flags[0]
    act_cmd = actuator_commands[tid]
    act_state = act_states_in[tid]
    thr_config = thruster_configs[tid]
    rw_config = rw_configs[tid]

    half_dt = wp.float64(0.5) * dt

    # Body-frame thrust, torque and mass flow are constant over the step
    # (scaled by the burn fraction on the burnout step)
    thr = thruster_wrench(thr_config, act_cmd, s, vehicle, dt)

    # ============================================================
    # FIRST HALF KICK (forces + torques)
    # ============================================================

    # Gravity scales with mass; thrust and drag do not (see kick_momentum).
    F_grav_1 = gravity_vector(s, flag)
    F_other_1 = compute_nongravity_forces(s, vehicle, flag, thr.force)
    M_ext_1 = compute_torques(s, vehicle, flag, thr.torque)

    # Momentum and mass half-kick. The gyroscopic term is geometric and
    # handled in the Drift phase. The burn fraction keeps the mass at or above
    # dry; the clamp only absorbs rounding.
    m_0 = wp.max(s.mass, vehicle.mass_dry)
    mass_k1 = wp.max(m_0 - half_dt * thr.mdot, vehicle.mass_dry)
    mom_k1 = kick_momentum(s.mom, m_0, mass_k1, F_grav_1, F_other_1, half_dt)

    # --- Reaction wheel coupling (momentum exchange for all wheels) ---
    L_rw_k1 = integrate_reaction_wheels(rw_config, act_cmd, act_state, half_dt)
    dL_rw_k1 = compute_rw_momentum_delta(L_rw_k1, act_state, rw_config)
    act_state_k1 = ActuatorState(L_rw_k1)

    ang_mom_k1 = s.ang_mom + half_dt * M_ext_1 - dL_rw_k1   # equal and opposite RW exchange

    # ============================================================
    # DRIFT
    # ============================================================

    # Translation (inertial) — use current (kicked) mass, not nominal vehicle mass
    m_d = wp.max(mass_k1, vehicle.mass_dry)  # guard against zero/negative mass
    pos_d = s.pos + dt * mom_k1 / m_d

    # Rotation: free rigid-body flow (wheel momenta held fixed between the
    # kicks). Total angular momentum is constant in the inertial frame, so in
    # the body frame it rotates backwards by the incremental rotation.
    rw_momentum_vec = compute_rw_momentum_vec(act_state_k1, rw_config)
    H_total = ang_mom_k1 + rw_momentum_vec

    # Explicit midpoint on the rotation group: predict the half-step attitude
    # with the initial omega, then rotate the full step with the midpoint omega
    omega_0 = wp.mul(vehicle.inertia_inv, ang_mom_k1)
    dq_half = quat_exp(omega_0, half_dt)
    H_total_half = wp.quat_rotate(quat_conj(dq_half), H_total)
    omega_d = wp.mul(vehicle.inertia_inv, H_total_half - rw_momentum_vec)

    dq = quat_exp(omega_d, dt)
    quat_d = wp.normalize(s.quat * dq)
    H_total_new = wp.quat_rotate(quat_conj(dq), H_total)

    # Extract the new spacecraft body angular momentum
    ang_mom_d = H_total_new - rw_momentum_vec

    # ============================================================
    # SECOND HALF KICK (forces + torques)
    # ============================================================

    s_mid = State(pos_d, mom_k1, mass_k1, quat_d, ang_mom_d)

    F_grav_2 = gravity_vector(s_mid, flag)
    F_other_2 = compute_nongravity_forces(s_mid, vehicle, flag, thr.force)
    M_ext_2 = compute_torques(s_mid, vehicle, flag, thr.torque)

    # Second momentum and mass half-kick, from the mid-step mass. The end
    # mass is the exact step-end value, rounded once (two half-kick roundings
    # could stop one ulp above dry, where the leftover flow rounds away)
    mass_k2 = thr.mass_end
    mom_k2 = kick_momentum(mom_k1, mass_k1, mass_k2, F_grav_2, F_other_2, half_dt)

    # Reaction wheel second half (all 4 wheels)
    L_rw_k2 = integrate_reaction_wheels(rw_config, act_cmd, act_state_k1, half_dt)
    dL_rw_k2 = compute_rw_momentum_delta(L_rw_k2, act_state_k1, rw_config)

    ang_mom_k2 = ang_mom_d + half_dt * M_ext_2 - dL_rw_k2

    # --- Store ---
    new_state = State(pos_d, mom_k2, mass_k2, quat_d, ang_mom_k2)
    store_state(states_out, tid, new_state)
    act_states_out[tid].rw_ang_mom = L_rw_k2

    # Only flag this entity; Integrator.step() reduces multi-body status later.
    update_termination_status(new_state, conditions[0], flag, status, tid)


@wp.func
def gravity_vector(state: State, flags: SimulationFlags) -> wp.vec3d:
    """Gravitational force [N], zero when gravity is disabled."""
    F = wp.vec3d()
    if flags.gravity:
        F = gravity_force(state, flags)
    return F


@wp.func
def kick_momentum(
    mom: wp.vec3d,
    m_start: wp.float64,
    m_end: wp.float64,
    F_grav: wp.vec3d,
    F_other: wp.vec3d,
    h: wp.float64,
) -> wp.vec3d:
    """
    Exact momentum kick over time h with forces held fixed and mass falling
    linearly from m_start to m_end (the burn fraction keeps the flow on for
    the whole kick, so the mass never sits at a dry clamp within it).

    With expelled propellant leaving at the vehicle velocity (Meshchersky),
    m dv/dt = F. Gravity is proportional to mass (constant acceleration);
    thrust and drag are not, so they act through the integral of 1/m(t):
        v1 = v0 + h F_grav/m_start + F_other * int_0^h dt/m(t)
    """
    v0 = mom / m_start
    # int_0^h dt/m(t) = h/m_start * (-ln(1-x)/x), x = burned fraction of
    # m_start; series for small x (including x = 0) avoids cancellation
    x = (m_start - m_end) / m_start
    log_ratio = wp.float64(1.0) + x * (wp.float64(0.5) + x * (
        wp.float64(1.0) / wp.float64(3.0) + x * wp.float64(0.25)))
    if x > wp.float64(1.0e-4):
        log_ratio = -wp.log(wp.float64(1.0) - x) / x
    v1 = v0 + h * F_grav / m_start + h / m_start * log_ratio * F_other
    return m_end * v1


@wp.func
def quat_conj(q: wp.quatd) -> wp.quatd:
    return wp.quatd(-q[0], -q[1], -q[2], q[3])


@wp.func
def quat_exp(omega: wp.vec3d, dt: wp.float64) -> wp.quatd:
    omega_len = wp.length(omega)
    theta = omega_len * dt
    if theta < wp.float64(1e-9):
        half_dt = wp.float64(0.5) * dt
        return wp.normalize(
            wp.quatd(
                omega[0]*half_dt,
                omega[1]*half_dt,
                omega[2]*half_dt,
                wp.float64(1.0)
            )
        )

    axis = omega / omega_len
    half = wp.float64(0.5) * theta
    s = wp.sin(half)
    return wp.quatd(axis[0]*s, axis[1]*s, axis[2]*s, wp.cos(half))

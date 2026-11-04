"""
Base integrator interface and common utilities.

This module provides the abstract base class for all integrators and re-exports
common dependencies to maintain clean separation of concerns.
"""
import warp as wp
from abc import ABC, abstractmethod

# Core types - re-exported for integrator implementations
from cosmic_warp.core.state import State
from cosmic_warp.core.vehicle import VehicleConfig
from cosmic_warp.core.termination import (
    TerminationConditions,
    TerminationStatus,
    REASON_NONFINITE,
    update_termination_status,
    reduce_environment_termination_kernel,
)
from cosmic_warp.core.flags import SimulationFlags
from cosmic_warp.core.context import SimContext

# Physics computation - re-exported for integrator implementations
from cosmic_warp.physics.aggregation import compute_forces, compute_torques

# Actuator types and functions - re-exported for integrator implementations
from cosmic_warp.actuators.base import ActuatorCommand, ActuatorState, command_is_finite
from cosmic_warp.actuators.reaction_wheels import ReactionWheelConfig, compute_rw_angular_momentum
from cosmic_warp.actuators.thrusters import ThrusterConfig, thruster_wrench


@wp.struct
class StateDerivative:
    mom_dot: wp.vec3d
    pos_dot: wp.vec3d
    ang_mom_dot: wp.vec3d
    quat_dot: wp.quatd
    mass_dot: wp.float64


class Integrator(ABC):
    """Abstract base class for numerical integrators."""

    def __init__(self, num_envs: int, device: str = "cuda", num_bodies: int = 1):
        """
        Initialize integrator.

        Args:
            num_envs: Number of parallel environments
            device: Device to run on
            num_bodies: Number of bodies per environment
        """
        self.num_envs = num_envs
        self.num_bodies = num_bodies
        self.num_entities = num_envs * num_bodies
        self.device = device

    @abstractmethod
    def launch_step(
        self,
        ctx: SimContext,
        dt: float,
        states_in: wp.array,
        states_out: wp.array,
        act_states_in: wp.array,
        act_states_out: wp.array,
        commands: wp.array,
    ) -> None:
        """
        Launch one integration step reading from the *_in buffers and writing
        the *_out buffers (single fused kernel). The kernel also increments
        ctx.episode_steps and checks termination for each entity independently.
        Use step() for in-place stepping with whole-environment termination;
        it reduces multi-body status after this kernel completes.

        The in/out buffers may alias (states_in is states_out): each thread
        fully reads its own element before writing it. Differentiable rollouts
        pass distinct per-step buffers so the tape never sees an overwrite.

        Args:
            ctx: Simulation context (vehicle/actuator configs, flags, status)
            dt: Time step
            states_in: State structs read [num_entities]
            states_out: State structs written [num_entities]
            act_states_in: ActuatorState structs read [num_entities]
            act_states_out: ActuatorState structs written [num_entities]
            commands: ActuatorCommand structs for this step [num_entities]
        """
        pass

    def step(self, ctx: SimContext, dt: float) -> None:
        """
        Perform one in-place integration step (updates ctx.states and
        ctx.actuator_states, flags newly terminated entities in
        ctx.termination_status, increments ctx.episode_steps).

        Args:
            ctx: Simulation context (state, config and status device arrays)
            dt: Time step
        """
        self.launch_step(
            ctx, dt,
            ctx.states, ctx.states,
            ctx.actuator_states, ctx.actuator_states,
            ctx.actuator_commands,
        )
        if self.num_bodies > 1:
            wp.launch(
                kernel=reduce_environment_termination_kernel,
                dim=self.num_envs,
                inputs=[ctx.termination_status, self.num_bodies],
                device=self.device,
            )

    @property
    @abstractmethod
    def order(self) -> int:
        """Return the order of the integration method."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the integration method."""
        pass


@wp.func
def compute_rw_momentum_vec(
    act_state: ActuatorState,
    rw_config: ReactionWheelConfig
) -> wp.vec3d:
    """Total reaction wheel angular momentum vector in the body frame
    (sum of all 4 wheels; unused wheels have zero spin axes)."""
    return (
        act_state.rw_ang_mom[0] * rw_config.spin_axis_0 +
        act_state.rw_ang_mom[1] * rw_config.spin_axis_1 +
        act_state.rw_ang_mom[2] * rw_config.spin_axis_2 +
        act_state.rw_ang_mom[3] * rw_config.spin_axis_3
    )


@wp.func
def compute_state_derivative(
    state: State,
    vehicle: VehicleConfig,
    force: wp.vec3d,
    torque: wp.vec3d,
    mdot: wp.float64,
    act_state: ActuatorState,
    rw_config: ReactionWheelConfig
) -> StateDerivative:
    """
    Compute state derivatives given current state and forces/torques.

    Frame conventions:
        - position, linear momentum: inertial
        - angular momentum, rw momentum, omega: body
        - forces: inertial
        - torques: body
        - mdot: total propellant mass flow rate [kg/s] (>= 0, from
          thruster_wrench for the current step)

    Adjoint-safe: single-assignment locals, struct built once via its
    constructor (field-wise struct mutation breaks Warp's autodiff).
    """

    # ============================================================
    # Mass dynamics — propellant mass flow
    # ============================================================

    # mass_dot = -T_total / (Isp * g0)  (always <= 0)
    mass_dot = -mdot

    # ============================================================
    # Translational dynamics (inertial)
    # ============================================================

    # Use current (state) mass so velocity is correct as propellant is consumed
    m = wp.max(state.mass, vehicle.mass_dry)   # guard against zero/negative mass
    pos_dot = state.mom / m

    # Variable-mass (Meshchersky) form: expelled propellant leaves at the
    # vehicle velocity, so p_dot = F + m_dot * v (thrust is already in F)
    mom_dot = force + mass_dot * pos_dot

    # ============================================================
    # Rotational dynamics (body frame)
    # ============================================================

    # Angular velocity from body angular momentum
    omega = wp.mul(vehicle.inertia_inv, state.ang_mom)

    # Total angular momentum in body frame
    H_total = state.ang_mom + compute_rw_momentum_vec(act_state, rw_config)

    # Euler equation in body frame
    ang_mom_dot = torque - wp.cross(omega, H_total)

    # ============================================================
    # Quaternion kinematics
    # ============================================================

    q = wp.normalize(state.quat)

    # Body angular velocity quaternion
    omega_quat = wp.quatd(omega[0], omega[1], omega[2], wp.float64(0.0))

    # q_dot = 0.5 * q ⊗ omega_quat  (body -> inertial convention)
    quat_dot = wp.float64(0.5) * wp.mul(q, omega_quat)

    return StateDerivative(mom_dot, pos_dot, ang_mom_dot, quat_dot, mass_dot)


@wp.func
def state_add_scaled(
    state: State,
    vehicle: VehicleConfig,
    mom_dot: wp.vec3d,
    pos_dot: wp.vec3d,
    ang_mom_dot: wp.vec3d,
    quat_dot: wp.quatd,
    mass_dot: wp.float64,
    scale: wp.float64,
) -> State:
    """
    Add scaled derivatives to state (for RK methods).

    Args:
        state: Base state
        vehicle: Vehicle configuration (its dry mass floors the new mass)
        mom_dot: Momentum derivative
        pos_dot: Position derivative
        ang_mom_dot: Angular momentum derivative
        quat_dot: Quaternion derivative
        mass_dot: Mass rate-of-change [kg/s] (negative = consuming propellant)
        scale: Scaling factor (usually dt or fraction of dt)

    Returns:
        New state with scaled derivatives added (constructed once — adjoint-safe)
    """
    new_pos = state.pos + scale * pos_dot
    new_mom = state.mom + scale * mom_dot
    new_ang_mom = state.ang_mom + scale * ang_mom_dot
    new_mass = wp.max(state.mass + scale * mass_dot, vehicle.mass_dry)   # clamp to dry mass

    # Quaternion update
    new_quat = wp.normalize(wp.quatd(
        state.quat[0] + scale * quat_dot[0],
        state.quat[1] + scale * quat_dot[1],
        state.quat[2] + scale * quat_dot[2],
        state.quat[3] + scale * quat_dot[3]
    ))

    return State(new_pos, new_mom, new_mass, new_quat, new_ang_mom)


@wp.func
def store_state(states_out: wp.array(dtype=State), tid: int, s: State):
    """Write a state into the output array field-by-field.

    Whole-struct stores of locally built structs break Warp's adjoint
    generation (silently zeroed gradients); per-field stores are the
    autodiff-safe pattern shared by every integrator kernel.
    """
    states_out[tid].pos = s.pos
    states_out[tid].mom = s.mom
    states_out[tid].mass = s.mass
    states_out[tid].quat = s.quat
    states_out[tid].ang_mom = s.ang_mom


@wp.func
def integrate_reaction_wheels(
    rw_config: ReactionWheelConfig,
    act_cmd: ActuatorCommand,
    act_state: ActuatorState,
    dt: wp.float64
) -> wp.vec4d:
    """Update all 4 reaction wheels and return their new angular momenta.

    This centralizes the repeated pattern used across integrators so kernels
    stay concise and maintainable.
    """
    return wp.vec4d(
        compute_rw_angular_momentum(rw_config, act_cmd, act_state, 0, dt),
        compute_rw_angular_momentum(rw_config, act_cmd, act_state, 1, dt),
        compute_rw_angular_momentum(rw_config, act_cmd, act_state, 2, dt),
        compute_rw_angular_momentum(rw_config, act_cmd, act_state, 3, dt)
    )


@wp.func
def compute_rw_momentum_delta(
    L_new: wp.vec4d,
    act_state: ActuatorState,
    rw_config: ReactionWheelConfig
) -> wp.vec3d:
    """Compute the total delta angular-momentum vector transferred from
    reaction wheels to the body (sum of wheel deltas projected on spin axes).
    """
    dL_rw = (
        (L_new[0] - act_state.rw_ang_mom[0]) * rw_config.spin_axis_0 +
        (L_new[1] - act_state.rw_ang_mom[1]) * rw_config.spin_axis_1 +
        (L_new[2] - act_state.rw_ang_mom[2]) * rw_config.spin_axis_2 +
        (L_new[3] - act_state.rw_ang_mom[3]) * rw_config.spin_axis_3
    )
    return dL_rw


class IntegratorFactory:
    """Factory for creating integrators."""

    @staticmethod
    def create(method: str, num_envs: int, device: str = "cuda", num_bodies: int = 1) -> "Integrator":
        """
        Create an integrator instance.

        Args:
            method: Integration method ("euler", "rk4", "symplectic")
            num_envs: Number of parallel environments
            device: Device to run on
            num_bodies: Number of bodies per environment

        Returns:
            Integrator instance
        """
        from cosmic_warp.core.integrators.euler import EulerIntegrator
        from cosmic_warp.core.integrators.rk4 import RK4Integrator
        from cosmic_warp.core.integrators.symplectic import SymplecticIntegrator

        if not isinstance(method, str) or method.lower() not in ("euler", "rk4", "symplectic"):
            raise ValueError(
                f"integrator must be 'euler', 'rk4' or 'symplectic', got {method!r}")
        method = method.lower()

        if method == "euler":
            return EulerIntegrator(num_envs, device, num_bodies)
        elif method == "rk4":
            return RK4Integrator(num_envs, device, num_bodies)
        return SymplecticIntegrator(num_envs, device, num_bodies)

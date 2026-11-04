"""
Termination condition checking for simulation environments.
"""
import numpy as np
import warp as wp

from cosmic_warp.core.state import State
from cosmic_warp.core.flags import SimulationFlags

# Termination reason codes (TerminationStatus.reason). When several bodies of
# an environment trip on the same step, all of them get the lowest code.
REASON_NONE = wp.constant(0)
REASON_POSITION = wp.constant(1)
REASON_MOMENTUM = wp.constant(2)
REASON_ANG_MOM = wp.constant(3)
REASON_MASS = wp.constant(4)
REASON_IMPACT = wp.constant(5)
REASON_NONFINITE = wp.constant(6)

#: Code -> (name, condition), the reference for ``get_termination_reasons()``.
REASONS = {
    REASON_POSITION: ("position", "`|pos| > pos_limit`, with `check_bounds`"),
    REASON_MOMENTUM: ("momentum", "`|mom| > mom_limit`, with `check_bounds`"),
    REASON_ANG_MOM: ("angular momentum", "`|ang_mom| > ang_mom_limit`, with `check_bounds`"),
    REASON_MASS: ("mass", "`mass <= mass_limit`, with `check_bounds`"),
    REASON_IMPACT: ("planet impact", "`|pos| < planet_radius`, with `check_planet_impact`"),
    REASON_NONFINITE: ("non-finite state", "a NaN or Inf in any state component, or in an "
                       "actuator command (the step is then skipped), always checked"),
}


@wp.struct
class TerminationConditions:
    """Stores termination limits and thresholds."""
    # Position bounds [m]
    pos_limit: wp.float64
    # Momentum bounds [kg*m/s]
    mom_limit: wp.float64
    # Angular momentum bounds [kg*m^2/s]
    ang_mom_limit: wp.float64
    # Mass limit [kg]
    mass_limit: wp.float64
    # Central body impact radius [m]
    planet_radius: wp.float64


@wp.struct
class TerminationStatus:
    """Stores termination status for each entity.

    Termination is env-level: when any body violates a condition, every
    body of that environment is flagged (and later reset) together.
    """

    terminated: wp.bool      # Is entity's environment terminated?
    reason: wp.int32          # Termination reason code: 0 or one of REASONS


class TerminationChecker:
    """Manages termination checking for simulation entities."""

    def __init__(
        self,
        num_envs: int,
        pos_limit: float,
        mom_limit: float,
        ang_mom_limit: float,
        mass_limit: float,
        planet_radius: float,
        device: str = "cuda",
        num_bodies: int = 1
    ):
        """
        Initialize termination checker.

        Args:
            num_envs: Number of parallel environments
            pos_limit: Maximum position magnitude [m]
            mom_limit: Maximum momentum magnitude [kg*m/s]
            ang_mom_limit: Maximum angular momentum magnitude [kg*m^2/s]
            mass_limit: Minimum mass [kg]
            planet_radius: Central body impact radius [m]
            device: Device to run on
            num_bodies: Number of bodies per environment
        """
        self.num_envs = num_envs
        self.num_bodies = num_bodies
        self.num_entities = num_envs * num_bodies
        self.device = device
        # Host-side copy for configuration-time sanity checks
        self.planet_radius = float(planet_radius)

        # Create termination conditions
        self.conditions = wp.zeros(1, dtype=TerminationConditions, device=device)

        # Initialize conditions on-device to ensure the values are visible
        # to GPU kernels (avoids host->device view issues).

        wp.launch(
            kernel=_init_conditions_kernel,
            dim=1,
            inputs=[
                self.conditions,
                float(pos_limit),
                float(mom_limit),
                float(ang_mom_limit),
                float(planet_radius),
                float(mass_limit)
            ],
            device=device,
        )

        # Create termination status array
        self.status = wp.zeros(self.num_entities, dtype=TerminationStatus, device=device)

    def extract_termination_mask(self, out_mask: wp.array(dtype=wp.bool)) -> None:
        """
        Write a boolean reset mask into a preallocated output array.

        Args:
            out_mask: WARP bool array where True indicates environment needs reset
        """
        wp.launch(
            kernel=extract_termination_mask_kernel,
            dim=self.num_entities,
            inputs=[self.status, out_mask],
            device=self.device
        )

    def reset_status(self) -> None:
        """Reset termination status for all entities (use reset_status_masked
        for a subset)."""
        wp.launch(
            kernel=reset_all_status_kernel,
            dim=self.num_entities,
            inputs=[self.status],
            device=self.device
        )

    def reset_status_masked(self, mask: wp.array) -> None:
        """
        Reset termination status for entities where mask is True.

        Args:
            mask: WARP bool array [num_entities]
        """
        wp.launch(
            kernel=reset_masked_status_kernel,
            dim=self.num_entities,
            inputs=[self.status, mask],
            device=self.device
        )

    def clear_terminated_status(self) -> None:
        """
        Clear the termination status flags for all terminated environments.

        Sets terminated flag back to false and reason to 0 for all environments
        that were marked as terminated.
        """
        wp.launch(
            kernel=clear_terminated_status_kernel,
            dim=self.num_entities,
            inputs=[self.status],
            device=self.device
        )

    def get_termination_reasons(self) -> dict:
        """
        Get the termination reason of every terminated environment.

        Termination is environment wide (every body of a terminated
        environment carries its reason), so the result is keyed by
        environment, whatever num_bodies. Reads the status to the host.

        Returns:
            {env_id: reason code} for the terminated environments (codes in REASONS)
        """
        status = self.status.numpy().reshape(self.num_envs, self.num_bodies)
        terminated = status["terminated"].any(axis=1)
        reasons = np.where(status["terminated"], status["reason"], 0).max(axis=1)
        return {int(env): int(reasons[env]) for env in np.flatnonzero(terminated)}


@wp.kernel
def _init_conditions_kernel(conds: wp.array(dtype=TerminationConditions),
                            pos_limit: wp.float64,
                            mom_limit: wp.float64,
                            ang_mom_limit: wp.float64,
                            planet_radius: wp.float64,
                            mass_limit: wp.float64) -> None:
    """Kernel to initialize termination conditions on-device."""
    conds[0].pos_limit = pos_limit
    conds[0].mom_limit = mom_limit
    conds[0].ang_mom_limit = ang_mom_limit
    conds[0].planet_radius = planet_radius
    conds[0].mass_limit = mass_limit

@wp.func
def update_termination_status(
    state: State,
    cond: TerminationConditions,
    flag: SimulationFlags,
    status: wp.array(dtype=TerminationStatus),
    tid: int,
):
    """
    Evaluate a freshly integrated state and flag only this entity.

    Called from the integrator kernels' tail so the state is still in
    registers. Writing sibling status here would race with their initial
    status reads and prevent some bodies from completing the step. Multi-body
    environments are reduced after integration by Integrator.step(). Only
    writes status on termination; the running path costs no global store.
    """
    terminated = False
    reason = REASON_NONE

    if flag.check_bounds:
        pos_mag = wp.length(state.pos)
        if pos_mag > cond.pos_limit:
            terminated = True
            reason = REASON_POSITION
        elif wp.length(state.mom) > cond.mom_limit:
            terminated = True
            reason = REASON_MOMENTUM
        elif wp.length(state.ang_mom) > cond.ang_mom_limit:
            terminated = True
            reason = REASON_ANG_MOM
        elif state.mass <= cond.mass_limit:
            terminated = True
            reason = REASON_MASS

    if not terminated and flag.check_planet_impact:
        if wp.length(state.pos) < cond.planet_radius:
            terminated = True
            reason = REASON_IMPACT

    # Always on: a NaN passes every bounds check above (comparisons with NaN
    # are false), and would otherwise propagate silently
    if not terminated:
        if not (wp.isfinite(state.pos) and wp.isfinite(state.mom) and wp.isfinite(state.mass)
                and wp.isfinite(state.quat) and wp.isfinite(state.ang_mom)):
            terminated = True
            reason = REASON_NONFINITE

    if terminated:
        status[tid].terminated = True
        status[tid].reason = reason


@wp.kernel
def reduce_environment_termination_kernel(
    status: wp.array(dtype=TerminationStatus),
    num_bodies: int,
) -> None:
    """Propagate the lowest nonzero reason after all bodies complete the step.

    One thread owns each environment's status range. Environments do not
    overlap, and all status reads finish before this thread writes its range.
    """
    env_start = wp.tid() * num_bodies
    reason = int(0)
    for body in range(num_bodies):
        entity = env_start + body
        if status[entity].terminated:
            candidate = status[entity].reason
            if reason == 0 or candidate < reason:
                reason = candidate

    if reason != 0:
        for body in range(num_bodies):
            status[env_start + body].terminated = True
            status[env_start + body].reason = reason


@wp.kernel
def reset_all_status_kernel(status: wp.array(dtype=TerminationStatus)) -> None:
    """Reset all termination statuses."""
    tid = wp.tid()
    status[tid].terminated = False
    status[tid].reason = 0

@wp.kernel
def reset_masked_status_kernel(
    status: wp.array(dtype=TerminationStatus),
    mask: wp.array(dtype=wp.bool)
) -> None:
    """Reset termination status where mask is True."""
    tid = wp.tid()
    if mask[tid]:
        status[tid].terminated = False
        status[tid].reason = 0

@wp.kernel
def extract_termination_mask_kernel(
    status: wp.array(dtype=TerminationStatus),
    mask: wp.array(dtype=wp.bool)
) -> None:
    """Extracts a boolean mask of environments that need resetting."""
    tid = wp.tid()
    mask[tid] = status[tid].terminated

@wp.kernel
def clear_terminated_status_kernel(
    status: wp.array(dtype=TerminationStatus)
) -> None:
    """Clears the termination flags."""
    tid = wp.tid()
    if status[tid].terminated:
        status[tid].terminated = False
        status[tid].reason = 0

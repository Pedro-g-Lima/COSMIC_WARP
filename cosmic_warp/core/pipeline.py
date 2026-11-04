"""
Main simulation pipeline orchestrating all components.
"""
import math

import numpy as np

import warp as wp
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from cosmic_warp.core.autodiff import DifferentiableRollout

from cosmic_warp.core import validation
from cosmic_warp.core.entities import expand_entity_ids
from cosmic_warp.core.state import State, allocate_state
from cosmic_warp.core.flags import create_flags
from cosmic_warp.core.context import SimContext
from cosmic_warp.core.reset import RESET_CONFIG_KEYS, ResetManager
from cosmic_warp.core.randomization import DomainRandomizer
from cosmic_warp.core.termination import TerminationChecker
from cosmic_warp.core.vehicle import VehicleManager, VehicleConfig
from cosmic_warp.core.integrators.base import IntegratorFactory, Integrator
from cosmic_warp.physics.gravity import RE_EARTH

# Actuator imports
from cosmic_warp.actuators.base import (
    MAX_WHEELS,
    ActuatorState,
    ActuatorManager,
    _padded,
)
from cosmic_warp.actuators.thrusters import ThrusterManager
from cosmic_warp.actuators.reaction_wheels import ReactionWheelManager, ReactionWheelConfig

def _read_only(name: str) -> property:
    """A read-only attribute backed by self._<name>: rebinding it would
    detach the simulator from the arrays its kernels and graphs hold."""
    return property(lambda self: getattr(self, "_" + name))


class SpacecraftSimulator:
    """
    Main simulation pipeline for parallel spacecraft dynamics.

    This class orchestrates all simulation components including:
    - State management
    - Force/torque computation
    - Numerical integration
    - Termination checking
    - Environment resets
    """

    # Read-only attributes (documented by the string after each)
    num_envs = _read_only("num_envs")
    """Number of parallel environments."""
    num_bodies = _read_only("num_bodies")
    """Number of bodies per environment."""
    num_entities = _read_only("num_entities")
    """num_envs * num_bodies."""
    dt = _read_only("dt")
    """Time step [s]."""
    device = _read_only("device")
    """The device the simulator was built on, as given."""
    differentiable = _read_only("differentiable")
    """Built with adjoint kernels for differentiable_rollout()."""
    states = _read_only("states")
    """The State array, one row per entity: assign to it or write it from a
    kernel, never rebind it."""
    vehicle = _read_only("vehicle")
    """The VehicleManager."""
    thruster_manager = _read_only("thruster_manager")
    """The ThrusterManager."""
    rw_manager = _read_only("rw_manager")
    """The ReactionWheelManager."""
    reset_manager = _read_only("reset_manager")
    """The ResetManager."""
    randomizer = _read_only("randomizer")
    """The DomainRandomizer."""
    termination_checker = _read_only("termination_checker")
    """The TerminationChecker."""
    actuators = _read_only("actuators")
    """The ActuatorManager: its commands and states arrays, one row per entity."""

    def __init__(
        self,
        *,
        num_envs: int = 1,
        integrator: str = "rk4",
        dt: float = 0.01,
        device: str = "cuda",
        num_bodies: int = 1,

        # Physics flags
        gravity: bool = True,
        j2_perturbation: Optional[bool] = None,
        gravity_gradient: bool = False,
        magnetic_torque: bool = False,
        drag: bool = False,

        # Simulation checks
        check_bounds: bool = True,
        check_planet_impact: bool = False,

        # Simulation settings
        pos_limit: float = 1.0e8,
        mom_limit: float = math.inf,
        ang_mom_limit: float = math.inf,
        mass_limit: float = 0.0,
        planet_radius: float = RE_EARTH,

        # Reset bounds
        reset_config: Optional[Dict[str, Any]] = None,

        # Differentiable rollouts (compiles adjoint kernels for the integrator)
        differentiable: bool = False
    ):
        """
        Initialize spacecraft simulator.

        Args:
            num_envs: Number of parallel environments
            integrator: Integration method ("euler", "rk4", "symplectic"); default "rk4"
            dt: Time step [s]
            device: Device to run on ("cuda" or "cpu")
            num_bodies: Number of bodies per environment
            gravity: Enable gravitational forces
            j2_perturbation: Enable J2 perturbation; requires gravity=True.
                Default (None): follows gravity
            gravity_gradient: Enable gravity gradient torque
            magnetic_torque: Enable magnetic torque (set vehicle.magnetic_dipole [A*m^2])
            drag: Enable atmospheric drag (uses vehicle.area_drag [m^2] and vehicle.cd)
            check_bounds: Terminate on the position, momentum, angular-momentum
                and mass limits below
            check_planet_impact: Enable impact detection with the central body
            pos_limit: Maximum position bound [m]; math.inf disables
            mom_limit: Maximum momentum bound [kg*m/s]; no limit by default
                (momentum scales with the vehicle's mass)
            ang_mom_limit: Maximum angular momentum bound [kg*m^2/s]; no
                limit by default (it scales with the vehicle's inertia)
            mass_limit: Terminate once mass <= mass_limit [kg]; 0 disables.
                mass_limit = mass_dry terminates when propellant runs out
            planet_radius: Central body impact radius [m], used by
                check_planet_impact; default Earth's equatorial radius
                (RE_EARTH)
            reset_config: Configuration for environment resets
            differentiable: Compile adjoint kernels for the chosen integrator
                so differentiable_rollout() can backpropagate through the
                dynamics (roughly doubles that module's one-off JIT cost;
                no effect on stepping performance)
        """
        wp.init()

        try:
            wp.get_device(device)
        except ValueError as e:   # an unknown name (no CUDA raises RuntimeError)
            raise ValueError(f"device {device!r}: {e}") from None
        num_envs = validation.positive_int("num_envs", num_envs)
        num_bodies = validation.positive_int("num_bodies", num_bodies)
        dt = validation.scalar("dt", dt, positive=True)
        gravity, gravity_gradient, magnetic_torque, drag, check_bounds, \
            check_planet_impact, differentiable = (
                validation.flag(name, value) for name, value in (
                    ("gravity", gravity), ("gravity_gradient", gravity_gradient),
                    ("magnetic_torque", magnetic_torque), ("drag", drag),
                    ("check_bounds", check_bounds),
                    ("check_planet_impact", check_planet_impact),
                    ("differentiable", differentiable)))
        if j2_perturbation is not None:
            j2_perturbation = validation.flag("j2_perturbation", j2_perturbation)
        pos_limit = validation.scalar("pos_limit", pos_limit, positive=True, allow_inf=True)
        mom_limit = validation.scalar("mom_limit", mom_limit, positive=True, allow_inf=True)
        ang_mom_limit = validation.scalar("ang_mom_limit", ang_mom_limit, positive=True,
                                          allow_inf=True)
        mass_limit = validation.scalar("mass_limit", mass_limit, minimum=0.0)
        planet_radius = validation.scalar("planet_radius", planet_radius, minimum=0.0)
        if reset_config is None:
            reset_config = {}
        elif not isinstance(reset_config, dict):
            raise ValueError(f"reset_config must be a dict, got {type(reset_config).__name__}")
        unknown = set(reset_config) - RESET_CONFIG_KEYS
        if unknown:
            raise ValueError(f"Unknown key(s) in reset_config: {sorted(unknown)}. "
                             f"Allowed: {sorted(RESET_CONFIG_KEYS)}.")
        if j2_perturbation is None:
            j2_perturbation = gravity
        elif j2_perturbation and not gravity:
            raise ValueError("j2_perturbation=True requires gravity=True")

        self._num_envs = num_envs
        self._num_bodies = num_bodies
        self._num_entities = num_envs * num_bodies
        self._dt = dt
        self._device = device

        # Initialize state
        self._states = allocate_state(num_envs, num_bodies, device)

        # Vehicle and actuator managers start with their default
        # configuration (1 kg cube, 6-thruster layout, 3 orthogonal wheels)
        self._vehicle = VehicleManager(num_envs, device=device, num_bodies=num_bodies,
                                       states=self.states)

        # Initialize flags
        self.flags = create_flags(
            gravity=gravity,
            j2_perturbation=j2_perturbation,
            gravity_gradient=gravity_gradient,
            magnetic_torque=magnetic_torque,
            drag=drag,
            check_bounds=check_bounds,
            check_planet_impact=check_planet_impact,
            device=device
        )

        # Initialize integrator
        self.integrator: Integrator = IntegratorFactory.create(
            integrator, num_envs, device=device, num_bodies=num_bodies
        )

        # Differentiable rollouts: compile the integrator module with its
        # adjoint (must happen before its first kernel launch)
        self._differentiable = differentiable
        if differentiable:
            from cosmic_warp.core.autodiff import enable_backward_for
            enable_backward_for(self.integrator)

        # Initialize reset manager
        self._reset_manager = ResetManager(num_envs, device=device, num_bodies=num_bodies,
                                           **reset_config)
        # Every reset floors the mass at the entity's (current) dry mass
        self.reset_manager.vehicles = self.vehicle.configs
        # Per-entity steps since last reset (zeroed by the reset kernels,
        # incremented each step)
        self.episode_steps = self.reset_manager.episode_steps

        # Domain randomization (inactive until randomizer.set_bounds is called)
        self._randomizer = DomainRandomizer(
            num_envs, self.vehicle, device=device, num_bodies=num_bodies
        )

        # Initialize termination checker
        self._termination_checker = TerminationChecker(
            num_envs,
            pos_limit=pos_limit,
            mom_limit=mom_limit,
            ang_mom_limit=ang_mom_limit,
            mass_limit=mass_limit,
            planet_radius=planet_radius,
            device=device,
            num_bodies=num_bodies
        )

        self.termination_mask = wp.zeros(self.num_entities, dtype=wp.bool, device=device)

        # Centralized actuator command/state management
        self._actuators = ActuatorManager(num_envs, device=device, num_bodies=num_bodies)

        # Initialize actuator managers
        self._thruster_manager = ThrusterManager(num_envs, device=device, num_bodies=num_bodies)
        self._rw_manager = ReactionWheelManager(num_envs, device=device, num_bodies=num_bodies,
                                                actuator_states=self.actuators.states)

        # Shared simulation context handed to the integrator each step
        self.ctx = SimContext(
            states=self.states,
            vehicles=self.vehicle.configs,
            actuator_commands=self.actuators.commands,
            actuator_states=self.actuators.states,
            thruster_configs=self.thruster_manager.configs,
            rw_configs=self.rw_manager.configs,
            flags=self.flags,
            termination_conditions=self.termination_checker.conditions,
            termination_status=self.termination_checker.status,
            episode_steps=self.episode_steps,
        )

        # Device-side trajectory step counter (allows CUDA graph capture of
        # the whole step + store sequence)
        self._traj_step = wp.zeros(1, dtype=wp.int32, device=device)

        # Preallocated buffers for state getters (reused between calls)
        self._positions_buf = wp.zeros(self.num_entities, dtype=wp.vec3d, device=device)
        self._velocities_buf = wp.zeros(self.num_entities, dtype=wp.vec3d, device=device)
        self._quaternions_buf = wp.zeros(self.num_entities, dtype=wp.quatd, device=device)
        self._angular_velocities_buf = wp.zeros(self.num_entities, dtype=wp.vec3d, device=device)
        self._masses_buf = wp.zeros(self.num_entities, dtype=wp.float64, device=device)
        self._rw_angular_momenta_buf = wp.zeros(self.num_entities, dtype=wp.vec4d, device=device)

        # Statistics
        self.step_count = 0

        # Constructor settings snapshot for metadata()
        self._settings = {
            "num_envs": num_envs,
            "num_bodies": num_bodies,
            "integrator": integrator,
            "dt": dt,
            "gravity": gravity,
            "j2_perturbation": j2_perturbation,
            "gravity_gradient": gravity_gradient,
            "magnetic_torque": magnetic_torque,
            "drag": drag,
            "check_bounds": check_bounds,
            "check_planet_impact": check_planet_impact,
            "differentiable": differentiable,
        }

    def metadata(self) -> Dict[str, Any]:
        """
        Provenance snapshot of the software and the constructor settings.

        Returns a JSON-serializable dict with the package, warp-lang and
        Python versions, the device, and the constructor settings
        (dimensions, integrator, dt, physics flags, termination switches).
        It does not cover termination limits or anything configured after
        construction: vehicle and actuator settings, reset and randomization
        bounds and seeds, initial conditions and commands. Save those too (e.g. the YAML
        config) to reproduce a run.
        """
        import platform

        from cosmic_warp import __version__

        return {
            "cosmic_warp_version": __version__,
            "warp_version": wp.config.version,
            "python_version": platform.python_version(),
            "device": str(wp.get_device(self.device)),
            **self._settings,
        }

    @property
    def _use_cuda_graph(self) -> bool:
        return "cuda" in str(self.device)

    def _check_reset_bounds(self) -> None:
        """
        Reject reset bounds that cannot produce a valid initial state, on
        every Python-side reset (host data only: no device read, no sync).

        - mass_bounds below a body's dry mass would spawn entities with less
          than dry mass, i.e. no propellant. Bodies with randomized propellant
          are exempt: their mass is dry + propellant by construction.
        - With check_planet_impact, orbit bounds whose worst-case perigee is
          at or below the impact radius would spawn entities on crashing orbits.
        """
        dry = self.vehicle.mass_dry_host.reshape(self.num_envs, self.num_bodies).max(axis=0)
        radius = self.termination_checker.planet_radius
        check_perigee = self._settings["check_planet_impact"] and radius > 0.0
        for body_id in range(self.num_bodies):
            b = self.reset_manager._bounds[body_id]
            mass_min = float(b["mass_bounds"][0])
            if "propellant" not in self.randomizer._bounds[body_id] and mass_min < dry[body_id]:
                raise ValueError(
                    f"reset mass_bounds minimum ({mass_min}) is below mass_dry "
                    f"({dry[body_id]}) of body {body_id}: resets would spawn "
                    f"entities with less than dry mass (zero propellant). Set "
                    f"reset_manager.set_bounds(mass_bounds=...) or enable propellant "
                    f"randomization."
                )
            if check_perigee and b["sample_mode"] == 1:
                perigee = (float(b["semi_major_axis_bounds"][0])
                           * (1.0 - float(b["eccentricity_bounds"][1])))
                if perigee <= radius:
                    raise ValueError(
                        f"orbit reset bounds of body {body_id} allow a perigee of "
                        f"{perigee:.6e} m, at or below the impact radius "
                        f"({radius:.6e} m): resets would spawn entities on crashing "
                        f"orbits. Raise semi_major_axis_bounds or lower eccentricity_bounds "
                        f"via set_orbit_bounds."
                    )

    def _check_default_mass(self) -> None:
        """reset_to_conditions without masses writes the mass_bounds center and
        applies no randomization: reject a center below a body's dry mass."""
        dry = self.vehicle.mass_dry_host.reshape(self.num_envs, self.num_bodies).max(axis=0)
        for body_id in range(self.num_bodies):
            lo, hi = self.reset_manager._bounds[body_id]["mass_bounds"]
            center = 0.5 * (float(lo) + float(hi))
            if center < dry[body_id]:
                raise ValueError(
                    f"reset_to_conditions without masses uses the mass_bounds center "
                    f"({center}), below mass_dry ({dry[body_id]}) of body {body_id}. "
                    f"Pass masses or raise reset_manager.set_bounds(mass_bounds=...)."
                )

    def _check_masses(self, masses, env_ids, body_ids, entity_ids) -> None:
        """Reject explicit masses that are non-finite or below the entity's dry
        mass (the step would otherwise clamp them silently)."""
        ids = expand_entity_ids(self.num_envs, self.num_bodies, env_ids, body_ids, entity_ids)
        ids = np.arange(self.num_entities) if ids is None else np.asarray(ids)
        # The device copy: randomization may have redrawn mass_dry
        m = masses.numpy()[ids]
        dry = self.vehicle.configs.numpy()["mass_dry"][ids]
        bad = ~np.isfinite(m) | (m < dry)
        if bad.any():
            k = int(np.flatnonzero(bad)[0])
            raise ValueError(
                f"reset_to_conditions masses[{int(ids[k])}] = {m[k]} is not finite "
                f"or below mass_dry ({dry[k]}) of entity {int(ids[k])}.")

    def _check_quaternions(self, quaternions, env_ids, body_ids, entity_ids) -> None:
        """Reject explicit quaternions that are zero or non-finite: the reset
        normalizes them, which those cannot survive."""
        ids = expand_entity_ids(self.num_envs, self.num_bodies, env_ids, body_ids, entity_ids)
        ids = np.arange(self.num_entities) if ids is None else np.asarray(ids)
        q = np.asarray(quaternions.numpy(), dtype=np.float64).reshape(-1, 4)[ids]
        bad = ~np.isfinite(q).all(axis=1) | (np.linalg.norm(q, axis=1) == 0.0)
        if bad.any():
            k = int(np.flatnonzero(bad)[0])
            raise ValueError(
                f"reset_to_conditions quaternions[{int(ids[k])}] = {q[k].tolist()} is "
                f"not a finite nonzero (x, y, z, w) quaternion.")

    def _check_partial_reset(self, env_ids, body_ids, entity_ids) -> None:
        """Reject a reset that would leave a terminated environment partly
        frozen: the next step would re-flag the reset bodies from their
        still-terminated siblings. Reads the status only for partial
        selections."""
        ids = expand_entity_ids(self.num_envs, self.num_bodies, env_ids, body_ids, entity_ids)
        if ids is None or self.num_bodies == 1:
            return
        selected = np.zeros(self.num_entities, dtype=bool)
        selected[ids] = True
        per_env = selected.reshape(self.num_envs, self.num_bodies)
        partial = per_env.any(axis=1) & ~per_env.all(axis=1)
        if not partial.any():
            return
        terminated = self.termination_checker.status.numpy()["terminated"]
        bad = partial & terminated.reshape(self.num_envs, self.num_bodies).any(axis=1)
        if bad.any():
            raise ValueError(
                f"environment {int(np.flatnonzero(bad)[0])} is terminated: reset all of "
                f"its bodies (termination is environment wide)"
            )

    def reset(
        self,
        *,
        env_ids: Optional[list] = None,
        body_ids: Optional[list] = None,
        entity_ids: Optional[list] = None
    ) -> None:
        """
        Reset environments to initial conditions.

        Args:
            env_ids: List of environment indices to reset (None for all)
            body_ids: List of body indices to reset (None for all bodies)
            entity_ids: Optional flat entity indices (env_id * num_bodies + body_id)

        Raises:
            ValueError: if the selection covers only some bodies of an
                environment that is terminated. Termination is environment
                wide, so such an environment must be reset whole.
        """
        self._check_reset_bounds()
        self.randomizer.sync_nominal()
        self._check_partial_reset(env_ids, body_ids, entity_ids)

        if env_ids is None and body_ids is None and entity_ids is None:
            self.reset_manager.reset_all(self.states)
            if self.randomizer.enabled:
                self.randomizer.randomize_all(self.states)
            self.termination_checker.reset_status()
            self.actuators.reset()
            self.step_count = 0  # Only reset when resetting all
        else:
            # Build the selection mask once and reuse it for every component
            mask = self.reset_manager.build_mask(env_ids, body_ids, entity_ids)
            if mask is None:
                return
            self.reset_manager.reset_masked(self.states, mask)
            if self.randomizer.enabled:
                self.randomizer.randomize_masked(self.states, mask)
            self.termination_checker.reset_status_masked(mask)
            self.actuators.reset_masked(mask)
            # Don't reset step_count for partial resets

    def reset_to_conditions(
        self,
        *,
        positions: wp.array = None,
        momenta: wp.array = None,
        angular_momenta: wp.array = None,
        quaternions: wp.array = None,
        masses: wp.array = None,
        env_ids: Optional[list] = None,
        body_ids: Optional[list] = None,
        entity_ids: Optional[list] = None
    ) -> None:
        """
        Reset environments to explicit initial conditions.

        Like reset(), this also clears termination status and actuator
        commands/states for the selected entities; unlike reset(), the
        dynamic state is set to the given values instead of being sampled
        (and domain randomization is not applied).

        The condition arrays are always indexed by flat entity id, so they
        keep the full [num_entities] length even when resetting a subset.

        Args:
            positions: WARP array of positions [num_entities] (vec3d), None uses ResetConfig center
            momenta: WARP array of momenta [num_entities] (vec3d), None uses ResetConfig center
            angular_momenta: WARP array of angular momenta [num_entities] (vec3d), None uses ResetConfig center
            quaternions: WARP array of quaternions [num_entities] (quatd),
                normalized on write; None uses identity
            masses: WARP array of masses [num_entities] (float64), None uses ResetConfig center
            env_ids: List of environment indices to reset (None for all)
            body_ids: List of body indices to reset (None for all bodies)
            entity_ids: Optional flat entity indices (env_id * num_bodies + body_id)

        Raises:
            ValueError: if an array is not a Warp array of the listed dtype,
                of length num_entities, on the simulator's device; if the
                selection covers only some bodies of a
                terminated environment (as reset()), if masses is omitted
                and the mass_bounds center is below a body's dry mass, or if
                a selected mass is below its dry mass or a selected
                quaternion is zero or non-finite.
        """
        self.reset_manager.check_conditions(positions=positions, momenta=momenta,
                                            angular_momenta=angular_momenta,
                                            quaternions=quaternions, masses=masses)
        if positions is None or momenta is None:
            self.reset_manager.check_orbit_defaults(env_ids, body_ids, entity_ids)
        if masses is None:
            self._check_default_mass()
        else:
            self._check_masses(masses, env_ids, body_ids, entity_ids)
        if quaternions is not None:
            self._check_quaternions(quaternions, env_ids, body_ids, entity_ids)
        self._check_partial_reset(env_ids, body_ids, entity_ids)

        if env_ids is None and body_ids is None and entity_ids is None:
            self.reset_manager.reset_to_conditions(
                self.states, positions, momenta, angular_momenta, quaternions, masses
            )
            self.termination_checker.reset_status()
            self.actuators.reset()
            self.step_count = 0
        else:
            # Build the selection mask once and reuse it for every component
            mask = self.reset_manager.build_mask(env_ids, body_ids, entity_ids)
            if mask is None:
                return
            self.reset_manager.reset_to_conditions(
                self.states, positions, momenta, angular_momenta, quaternions, masses,
                mask=mask
            )
            self.termination_checker.reset_status_masked(mask)
            self.actuators.reset_masked(mask)

    def reset_terminated(self) -> None:
        """
        Reset environments that have met termination conditions entirely on the GPU.
        """
        # 0. A set_config(inertia=...) since the last call becomes the nominal
        # inertia_scale applies to: a host-side version check, skipped while
        # capturing (a host upload cannot be recorded; simulate() syncs before
        # its capture, and a replayed graph does not run Python)
        if not wp.get_device(self.device).is_capturing:
            self.randomizer.sync_nominal()

        # 1. Flag environments that have terminated into the preallocated mask
        self.termination_checker.extract_termination_mask(self.termination_mask)

        # 2. Reset dynamic states, resampling randomized vehicle parameters
        self.reset_manager.reset_masked(self.states, self.termination_mask)
        if self.randomizer.enabled:
            self.randomizer.randomize_masked(self.states, self.termination_mask)

        # 3. Reset actuator states and commands
        self.actuators.reset_masked(self.termination_mask)

        # 4. Clear their termination status back to false
        self.termination_checker.clear_terminated_status()

    def step(self) -> None:
        """
        Advance simulation by one time step.

        Force/torque computation, numerical integration, per-entity termination
        checking and episode step counters are fused into one kernel. Multi-body
        status is then reduced so all bodies complete the terminating step.
        """
        self.integrator.step(self.ctx, self.dt)
        self.step_count += 1

    def set_thrust_commands(self, thrust_cmd: list) -> None:
        """
        Set thruster commands for all entities (broadcast same command).

        Args:
            thrust_cmd: Up to 8 throttles in [0, 1], one per thruster slot;
                omitted trailing slots are zero
        """
        self.actuators.set_thrust_commands(thrust_cmd)

    def set_thrust_commands_per_entity(self, thrust_cmds: wp.array) -> None:
        """
        Set thruster commands per entity (different command for each body).

        Args:
            thrust_cmds: float64 Warp array of shape (num_envs * num_bodies, 8)
                on the simulator's device: one throttle in [0, 1] per thruster
                slot (columns of unused slots have no effect)
        """
        self.actuators.set_thrust_commands_per_entity(thrust_cmds)

    def set_rw_torque_commands(self, rw_torque_cmds: list) -> None:
        """
        Set reaction wheel torque commands for all entities (broadcast same command).

        Args:
            rw_torque_cmds: Up to 4 RW torques [N*m]; omitted trailing
                wheels are zero
        """
        self.actuators.set_rw_torque_commands(rw_torque_cmds)

    def set_rw_torque_commands_per_entity(self, rw_torque_cmds: wp.array) -> None:
        """
        Set reaction wheel torque commands per entity (different command for each body).

        Args:
            rw_torque_cmds: float64 Warp array of shape (num_envs * num_bodies, 4)
                on the simulator's device, RW torques [N*m]
        """
        self.actuators.set_rw_torque_commands_per_entity(rw_torque_cmds)

    def set_rw_angular_momentum_state(self, rw_ang_mom: list) -> None:
        """
        Set reaction wheel angular momentum state for all entities (broadcast).

        Args:
            rw_ang_mom: Up to 4 RW angular momenta [kg*m^2/s]; omitted
                trailing wheels are zero. Each must be finite and within the
                wheel's max_momentum.
        """
        h = np.zeros((self.num_entities, 4))
        h[:] = _padded("rw_ang_mom", rw_ang_mom, MAX_WHEELS)
        self._check_rw_momentum(h)
        self.actuators.set_rw_state(rw_ang_mom)

    def set_rw_angular_momentum_state_per_entity(self, rw_ang_mom: wp.array) -> None:
        """
        Set reaction wheel angular momentum state per entity.

        Args:
            rw_ang_mom: float64 Warp array of shape (num_envs * num_bodies, 4)
                on the simulator's device [kg*m^2/s]. Each entry must be
                finite and within the wheel's max_momentum.
        """
        self.actuators._check_per_entity("rw_ang_mom", rw_ang_mom, MAX_WHEELS)
        self._check_rw_momentum(rw_ang_mom.numpy())
        self.actuators.set_rw_state_per_entity(rw_ang_mom)

    def _check_rw_momentum(self, h: np.ndarray) -> None:
        """Reject wheel momenta a wheel cannot hold (configuration time only).

        A wheel above its max_momentum would be clamped on the next step and
        the excess dumped into the body as a spurious impulse.
        """
        if not np.all(np.isfinite(h)):
            raise ValueError("reaction wheel momenta must be finite")
        limit = self.rw_manager.configs.numpy()["max_momentum"][:, None]
        over = (limit > 0.0) & (np.abs(h) > limit)
        if over.any():
            entity, wheel = np.argwhere(over)[0]
            raise ValueError(
                f"wheel {wheel} of entity {entity}: momentum {h[entity, wheel]} exceeds "
                f"max_momentum {limit[entity, 0]}"
            )

    def get_positions(self) -> wp.array:
        """
        Get positions of all entities.

        Returns:
            WARP array of positions [num_envs * num_bodies, 3].
            The returned array is a preallocated buffer reused between calls.
        """
        wp.launch(
            kernel=extract_positions_kernel,
            dim=self.num_entities,
            inputs=[self.states, self._positions_buf],
            device=self.device
        )
        return self._positions_buf

    def get_velocities(self) -> wp.array:
        """
        Get velocities of all entities.

        Returns:
            WARP array of velocities [num_envs * num_bodies, 3].
            The returned array is a preallocated buffer reused between calls.
        """
        wp.launch(
            kernel=extract_velocities_kernel,
            dim=self.num_entities,
            inputs=[self.states, self.vehicle.configs, self._velocities_buf],
            device=self.device
        )
        return self._velocities_buf

    def get_quaternions(self) -> wp.array:
        """
        Get attitude quaternions of all entities.

        Returns:
            WARP array of quaternions [num_envs * num_bodies, 4].
            The returned array is a preallocated buffer reused between calls.
        """
        wp.launch(
            kernel=extract_quaternions_kernel,
            dim=self.num_entities,
            inputs=[self.states, self._quaternions_buf],
            device=self.device
        )
        return self._quaternions_buf

    def get_angular_velocities(self) -> wp.array:
        """
        Get angular velocities of all entities.

        Returns:
            WARP array of angular velocities [num_envs * num_bodies, 3].
            The returned array is a preallocated buffer reused between calls.
        """
        wp.launch(
            kernel=extract_angular_velocities_kernel,
            dim=self.num_entities,
            inputs=[self.states, self.vehicle.configs, self._angular_velocities_buf],
            device=self.device
        )
        return self._angular_velocities_buf

    def get_masses(self) -> wp.array:
        """
        Get masses of all entities.

        Returns:
            WARP array of masses [num_envs * num_bodies].
            The returned array is a preallocated buffer reused between calls.
        """
        wp.launch(
            kernel=extract_masses_kernel,
            dim=self.num_entities,
            inputs=[self.states, self._masses_buf],
            device=self.device
        )
        return self._masses_buf

    def get_rw_angular_momenta(self) -> wp.array:
        """
        Get reaction wheel angular momenta of all entities.

        Returns:
            WARP array of vec4d [num_envs * num_bodies]: the momentum of
            wheels 0-3 about their spin axes [kg*m^2/s].
            The returned array is a preallocated buffer reused between calls.
        """
        wp.launch(
            kernel=extract_rw_angular_momenta_kernel,
            dim=self.num_entities,
            inputs=[self.actuators.states, self._rw_angular_momenta_buf],
            device=self.device
        )
        return self._rw_angular_momenta_buf

    def get_terminated(self) -> wp.array:
        """
        Get termination status for all entities.

        Returns:
            WARP array of TerminationStatus structs [num_envs * num_bodies]
        """
        return self.termination_checker.status

    def get_episode_steps(self) -> wp.array:
        """
        Get per-entity episode step counters.

        Counts steps since each entity's last reset (every reset path zeroes
        the counter of the entities it resets), so per-environment episode
        lengths stay correct across partial resets and reset_terminated.
        Counts attempted steps even while a terminated entity's state is frozen.

        Returns:
            WARP array of int32 [num_envs * num_bodies]
        """
        return self.episode_steps

    def _launch_store_state(self, positions, velocities, quaternions, angular_velocities, masses) -> None:
        """Launch the state-storing kernel using the device-side step counter."""
        wp.launch(
            kernel=store_trajectory_kernel,
            dim=self.num_entities,
            inputs=[self.states, self.vehicle.configs, positions, velocities, quaternions, angular_velocities, masses, self._traj_step],
            device=self.device
        )

    def _launch_store_rw(self, rw_angular_velocities) -> None:
        """Launch the RW-storing kernel using the device-side step counter."""
        wp.launch(
            kernel=store_rw_trajectory_kernel,
            dim=self.num_entities,
            inputs=[self.actuators.states, self.rw_manager.configs, rw_angular_velocities,
                    self._traj_step],
            device=self.device
        )

    def _run_steps(self, step_body, num_steps: int) -> None:
        """Run step_body num_steps times, via CUDA graph capture when available.

        On CUDA the kernel sequence of one step is captured once (recorded
        without executing) and replayed, removing per-kernel launch overhead;
        step_count bookkeeping is patched up around the capture.
        """
        if self._use_cuda_graph:
            step_count_before = self.step_count
            with wp.ScopedCapture(device=self.device) as capture:
                step_body()
            self.step_count = step_count_before

            for _ in range(num_steps):
                wp.capture_launch(capture.graph)
            self.step_count += num_steps
        else:
            for _ in range(num_steps):
                step_body()

    def simulate(
        self,
        num_steps: int,
        *,
        save_states: bool = False,
        save_rw_states: bool = False,
        auto_reset_terminated: bool = False
    ) -> Dict[str, wp.array]:
        """
        Run simulation for specified number of steps.

        On CUDA devices the whole per-step kernel sequence is captured once
        into a CUDA graph and replayed, removing per-kernel launch overhead.

        Args:
            num_steps: Number of time steps to simulate
            save_states: Whether to save the state trajectory (positions,
                velocities, etc.) at every step. Off by default: it allocates
                num_steps x num_entities device memory, so enable it only
                when the time history is actually needed (e.g. plotting)
            save_rw_states: Whether to save reaction wheel state trajectory
            auto_reset_terminated: Whether to automatically reset terminated environments or keep them terminated
        Returns:
            Dictionary of WARP arrays: the recorded trajectories, and with
            them "time", time[k] = (k + 1) * dt, the epoch of row k (each row
            holds the state after its step). Empty without save_states or
            save_rw_states.
        """
        num_steps = validation.positive_int("num_steps", num_steps)
        save_states, save_rw_states, auto_reset_terminated = (
            validation.flag(name, value) for name, value in (
                ("save_states", save_states), ("save_rw_states", save_rw_states),
                ("auto_reset_terminated", auto_reset_terminated)))
        self.randomizer.sync_nominal()   # before any graph capture
        results = {}

        # Pre-allocate trajectory storage on device (only if requested)
        if save_states:
            positions = wp.zeros((num_steps, self.num_entities), dtype=wp.vec3d, device=self.device)
            velocities = wp.zeros((num_steps, self.num_entities), dtype=wp.vec3d, device=self.device)
            quaternions = wp.zeros((num_steps, self.num_entities), dtype=wp.quatd, device=self.device)
            angular_velocities = wp.zeros((num_steps, self.num_entities), dtype=wp.vec3d, device=self.device)
            masses = wp.zeros((num_steps, self.num_entities), dtype=wp.float64, device=self.device)

        if save_rw_states:
            rw_angular_velocities = wp.zeros((num_steps, self.num_entities), dtype=wp.vec4d, device=self.device)

        def step_body():
            self.step()
            if save_states:
                self._launch_store_state(positions, velocities, quaternions, angular_velocities, masses)
            if save_rw_states:
                self._launch_store_rw(rw_angular_velocities)
            if auto_reset_terminated:
                self.reset_terminated()
            wp.launch(
                kernel=increment_step_kernel,
                dim=1,
                inputs=[self._traj_step],
                device=self.device
            )

        self._traj_step.zero_()
        self._run_steps(step_body, num_steps)

        # The epochs of the recorded rows, only alongside a recording
        if save_states or save_rw_states:
            results['time'] = wp.empty(num_steps, dtype=wp.float64, device=self.device)
            wp.launch(
                kernel=create_time_array_kernel,
                dim=num_steps,
                inputs=[results['time'], self.dt],
                device=self.device
            )

        # Add state data to results if saved
        if save_states:
            results['positions'] = positions
            results['velocities'] = velocities
            results['quaternions'] = quaternions
            results['angular_velocities'] = angular_velocities
            results['masses'] = masses

        # Add RW data to results if saved
        if save_rw_states:
            results['rw_angular_velocities'] = rw_angular_velocities

        return results

    def differentiable_rollout(self, num_steps: int, *,
                               commands: list = None) -> "DifferentiableRollout":
        """
        Create a differentiable rollout starting from the current state.

        Gradients flow to the initial state, the vehicle/actuator parameters
        (system identification) and, if given, the per-step control sequence.
        Requires the simulator to be constructed with differentiable=True
        (and termination checks disabled). See
        :class:`cosmic_warp.core.autodiff.DifferentiableRollout` for the
        taping/backward workflow and memory footprint.

        Args:
            num_steps: Number of time steps to roll out
            commands: Optional per-step control sequence from
                :func:`cosmic_warp.core.autodiff.command_sequence`; None uses
                the simulator's current actuator commands for every step
                (controls are then not differentiated)

        Returns:
            DifferentiableRollout: rollout bound to this simulator
        """
        from cosmic_warp.core.autodiff import DifferentiableRollout
        return DifferentiableRollout(self, num_steps, commands)

    def simulate_monte_carlo(self, num_steps: int) -> Dict[str, wp.array]:
        """
        Run simulation and return only final states (for Monte Carlo).

        Memory efficient - doesn't store trajectory, only final state and statistics.

        Args:
            num_steps: Number of time steps to simulate

        Returns:
            Dictionary with final state and termination info
        """
        num_steps = validation.positive_int("num_steps", num_steps)
        self._run_steps(self.step, num_steps)

        # Extract final states
        final_positions = self.get_positions()
        final_velocities = self.get_velocities()
        final_quaternions = self.get_quaternions()
        final_angular_velocities = self.get_angular_velocities()
        final_masses = self.get_masses()

        return {
            'final_positions': final_positions,
            'final_velocities': final_velocities,
            'final_quaternions': final_quaternions,
            'final_angular_velocities': final_angular_velocities,
            'final_masses': final_masses,
            'termination_status': self.termination_checker.status,
            'sim_time': num_steps * self.dt
        }


@wp.kernel
def extract_positions_kernel(
    states: wp.array(dtype=State),
    positions: wp.array(dtype=wp.vec3d)
):
    """Extract positions from states."""
    tid = wp.tid()
    positions[tid] = states[tid].pos


@wp.kernel
def extract_velocities_kernel(
    states: wp.array(dtype=State),
    vehicles: wp.array(dtype=VehicleConfig),
    velocities: wp.array(dtype=wp.vec3d)
):
    """Extract velocities from states."""
    tid = wp.tid()
    m = wp.max(states[tid].mass, vehicles[tid].mass_dry)
    velocities[tid] = states[tid].mom / m


@wp.kernel
def extract_quaternions_kernel(
    states: wp.array(dtype=State),
    quaternions: wp.array(dtype=wp.quatd)
):
    """Extract quaternions from states."""
    tid = wp.tid()
    quaternions[tid] = states[tid].quat


@wp.kernel
def extract_angular_velocities_kernel(
    states: wp.array(dtype=State),
    vehicles: wp.array(dtype=VehicleConfig),
    angular_velocities: wp.array(dtype=wp.vec3d)
):
    """Extract angular velocities from states."""
    tid = wp.tid()
    angular_velocities[tid] = wp.mul(vehicles[tid].inertia_inv, states[tid].ang_mom)


@wp.kernel
def extract_masses_kernel(
    states: wp.array(dtype=State),
    masses: wp.array(dtype=wp.float64)
):
    """Extract masses from states."""
    tid = wp.tid()
    masses[tid] = states[tid].mass


@wp.kernel
def extract_rw_angular_momenta_kernel(
    actuator_states: wp.array(dtype=ActuatorState),
    rw_ang_mom: wp.array(dtype=wp.vec4d)
):
    """Extract reaction wheel angular momenta from actuator states."""
    tid = wp.tid()
    rw_ang_mom[tid] = actuator_states[tid].rw_ang_mom


@wp.kernel
def store_trajectory_kernel(
    states: wp.array(dtype=State),
    vehicles: wp.array(dtype=VehicleConfig),
    positions: wp.array(dtype=wp.vec3d, ndim=2),
    velocities: wp.array(dtype=wp.vec3d, ndim=2),
    quaternions: wp.array(dtype=wp.quatd, ndim=2),
    angular_velocities: wp.array(dtype=wp.vec3d, ndim=2),
    masses: wp.array(dtype=wp.float64, ndim=2),
    step: wp.array(dtype=wp.int32)
):
    """Store current state in trajectory arrays."""
    tid = wp.tid()
    s = step[0]
    m = wp.max(states[tid].mass, vehicles[tid].mass_dry)
    positions[s, tid] = states[tid].pos
    velocities[s, tid] = states[tid].mom / m
    quaternions[s, tid] = states[tid].quat
    angular_velocities[s, tid] = wp.mul(vehicles[tid].inertia_inv, states[tid].ang_mom)
    masses[s, tid] = states[tid].mass


@wp.kernel
def store_rw_trajectory_kernel(
    actuator_states: wp.array(dtype=ActuatorState),
    rw_configs: wp.array(dtype=ReactionWheelConfig),
    rw_angular_velocities: wp.array(dtype=wp.vec4d, ndim=2),
    step: wp.array(dtype=wp.int32)
):
    """Store individual reaction wheel angular velocities in trajectory arrays."""
    tid = wp.tid()
    s = step[0]

    # Compute individual wheel angular velocities (scalar per wheel)
    if rw_configs[tid].wheel_inertia > wp.float64(0.0):
        inv_inertia = wp.float64(1.0) / rw_configs[tid].wheel_inertia
        rw_omegas = wp.vec4d(
            actuator_states[tid].rw_ang_mom[0] * inv_inertia,
            actuator_states[tid].rw_ang_mom[1] * inv_inertia,
            actuator_states[tid].rw_ang_mom[2] * inv_inertia,
            actuator_states[tid].rw_ang_mom[3] * inv_inertia
        )
    else:
        rw_omegas = wp.vec4d()

    rw_angular_velocities[s, tid] = rw_omegas


@wp.kernel
def increment_step_kernel(step: wp.array(dtype=wp.int32)):
    """Advance the device-side trajectory step counter."""
    step[0] = step[0] + 1


@wp.kernel
def create_time_array_kernel(
    time: wp.array(dtype=wp.float64),
    dt: wp.float64
):
    """Create time array: time[k] = (k+1)*dt (states are stored post-step)."""
    tid = wp.tid()
    time[tid] = wp.float64(tid + 1) * dt


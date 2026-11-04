"""
Domain randomization of per-entity vehicle parameters.

Parameters are resampled entirely on the GPU whenever entities are reset,
from counter-based per-entity streams like ResetManager's (on their own
stream id, cosmic_warp.core.rng), so randomization stays valid inside CUDA
graph captures (reset_terminated /
simulate(auto_reset_terminated=True)).

Sampling respects the physical couplings:
  - Propellant is sampled, not total mass: state mass = mass_dry + propellant,
    so entities can never spawn below dry mass (which the dynamics would
    silently clamp to "out of propellant").
  - Mass randomization preserves the velocity, not the momentum: the state's
    momentum is rescaled by new_mass / old_mass, so the sampled trajectory
    (in particular an orbit drawn in orbital-element reset mode) is unchanged
    by how much propellant the vehicle carries.
  - The inertia tensor is randomized through a scalar scale factor applied to
    the body's nominal tensor, preserving symmetry, positive definiteness and
    the triangle inequality of the principal moments.
"""
import numpy as np
import warp as wp

from cosmic_warp.core import validation
from cosmic_warp.core.entities import check_ids
from cosmic_warp.core.state import State
from cosmic_warp.core.vehicle import VehicleConfig, VehicleManager
from cosmic_warp.core.rng import (RNG_STATE, STREAM_RANDOMIZE, advance, draw_uniform,
                                  sample_uniform, seed_states)


@wp.struct
class RandomizationConfig:
    """Per-body uniform sampling bounds. A field is active when its flag is nonzero."""

    randomize_mass_dry: wp.int32
    mass_dry_min: wp.float64
    mass_dry_max: wp.float64

    randomize_propellant: wp.int32
    propellant_min: wp.float64
    propellant_max: wp.float64

    randomize_inertia: wp.int32
    inertia_scale_min: wp.float64
    inertia_scale_max: wp.float64


class DomainRandomizer:
    """
    Resamples vehicle parameters (and propellant load) per entity on reset.

    Bounds are per body and broadcast across environments; every entity draws
    its own values from its body's bounds. Inertia scaling applies to each
    entity's own configured (nominal) tensor, so per-environment inertia is
    kept; a later vehicle.set_config takes effect at the next sim.reset(),
    reset_terminated(), simulate() or RL VecEnv.step().
    """

    def __init__(
        self,
        num_envs: int,
        vehicle: VehicleManager,
        device: str = "cuda",
        num_bodies: int = 1,
        seed: int = 1234
    ):
        """
        Args:
            num_envs: Number of parallel environments
            vehicle: The simulator's VehicleManager
            device: Device to run on
            num_bodies: Number of bodies per environment
            seed: RNG seed for parameter sampling
        """
        self.num_envs = num_envs
        self.num_bodies = num_bodies
        self.num_entities = num_envs * num_bodies
        self.device = device
        self._vehicle = vehicle
        self._vehicle_configs = vehicle.configs

        # Host-side per-body bounds mirror (merged on partial overrides).
        # Keys: "mass_dry", "propellant", "inertia_scale" -> (min, max).
        self._bounds = [dict() for _ in range(num_bodies)]

        # Per-entity nominal inertia (and inverse) that inertia_scale applies
        # to, refreshed from the vehicle's host mirror when it changes
        self._inertia_nominal = wp.zeros(self.num_entities, dtype=wp.mat33d, device=device)
        self._inertia_inv_nominal = wp.zeros(self.num_entities, dtype=wp.mat33d, device=device)
        self._nominal_version = None

        # All flags zero-initialized = nothing randomized until set_bounds
        self.config = wp.zeros(num_bodies, dtype=RandomizationConfig, device=device)
        self.enabled = False

        self.rng_state = wp.empty(self.num_entities, dtype=RNG_STATE, device=device)
        self.reseed(seed)

    def reseed(self, seed: int) -> None:
        """Restart the per-entity RNG streams at draw 0 of ``seed`` (an
        integer in [0, 2**64))."""
        seed_states(self.rng_state, validation.seed("randomization seed", seed))

    def set_bounds(
        self,
        *,
        mass_dry: tuple = None,
        propellant: tuple = None,
        inertia_scale: tuple = None,
        body_ids: list = None
    ) -> None:
        """
        Enable randomization of vehicle parameters for specific bodies.

        Args:
            mass_dry: (min, max) dry mass [kg]. Requires `propellant` for the
                same bodies so total mass stays above dry mass.
            propellant: (min, max) propellant mass [kg], sampled on top of the
                (possibly randomized) dry mass: state mass = dry + propellant.
            inertia_scale: (min, max) scale factor applied to each entity's
                configured (nominal) inertia tensor.
            body_ids: List of body indices to configure (None = all bodies)
        """
        body_ids = (range(self.num_bodies) if body_ids is None
                    else check_ids("body_ids", body_ids, self.num_bodies))

        if mass_dry is not None:
            mass_dry = _checked_bounds(mass_dry, "mass_dry", positive=True)
        if propellant is not None:
            propellant = _checked_bounds(propellant, "propellant")
        if inertia_scale is not None:
            inertia_scale = _checked_bounds(inertia_scale, "inertia_scale", positive=True)

        # Validate the merged bounds of every body before mutating anything
        merged = {}
        for body_id in body_ids:
            b = dict(self._bounds[body_id])
            if mass_dry is not None:
                b["mass_dry"] = mass_dry
            if propellant is not None:
                b["propellant"] = propellant
            if inertia_scale is not None:
                b["inertia_scale"] = inertia_scale
            if "mass_dry" in b and "propellant" not in b:
                raise ValueError(
                    f"randomization of mass_dry (body {body_id}) requires "
                    "propellant bounds too, so total mass = dry + propellant "
                    "stays above dry mass."
                )
            merged[body_id] = b

        for body_id, b in merged.items():
            self._bounds[body_id] = b
            self._write_config(body_id)

        self.enabled = any(self._bounds)
        self.sync_nominal()

    def get_bounds(self, body_id: int = 0) -> dict:
        """The bounds set for one body, as {name: (min, max)}; a parameter
        absent from the dict is not randomized. A copy: edit with set_bounds."""
        (body_id,) = check_ids("body_id", [body_id], self.num_bodies)
        return dict(self._bounds[body_id])

    def sync_nominal(self) -> None:
        """Upload the configured inertia as the nominal for inertia scaling,
        if it changed since the last upload (host-side check, no device read).
        Called by set_bounds, sim.reset(), reset_terminated() (outside a
        capture), simulate() and the RL VecEnv.step()."""
        if self._nominal_version == self._vehicle.inertia_version:
            return
        nominal = self._vehicle.inertia_host
        self._inertia_nominal.assign(nominal)
        self._inertia_inv_nominal.assign(np.linalg.inv(nominal) if nominal.any()
                                         else nominal)
        self._nominal_version = self._vehicle.inertia_version

    def _write_config(self, body_id: int) -> None:
        """Write the host-side bounds of one body to its device config slot."""
        b = self._bounds[body_id]
        mass_dry = b.get("mass_dry")
        propellant = b.get("propellant")
        inertia_scale = b.get("inertia_scale")

        wp.launch(
            kernel=_init_randomization_config_kernel,
            dim=1,
            inputs=[
                self.config,
                int(body_id),
                int(mass_dry is not None),
                float(mass_dry[0]) if mass_dry else 0.0,
                float(mass_dry[1]) if mass_dry else 0.0,
                int(propellant is not None),
                float(propellant[0]) if propellant else 0.0,
                float(propellant[1]) if propellant else 0.0,
                int(inertia_scale is not None),
                float(inertia_scale[0]) if inertia_scale else 0.0,
                float(inertia_scale[1]) if inertia_scale else 0.0,
            ],
            device=self.device
        )

    def randomize_all(self, states: wp.array) -> None:
        """Resample randomized parameters for all entities."""
        wp.launch(
            kernel=randomize_kernel,
            dim=self.num_entities,
            inputs=[states, self._vehicle_configs, self.config, self.rng_state,
                    self._inertia_nominal, self._inertia_inv_nominal, self.num_bodies],
            device=self.device
        )

    def randomize_masked(self, states: wp.array, mask: wp.array) -> None:
        """Resample randomized parameters for masked entities (graph-safe)."""
        wp.launch(
            kernel=randomize_masked_kernel,
            dim=self.num_entities,
            inputs=[states, self._vehicle_configs, self.config, self.rng_state,
                    self._inertia_nominal, self._inertia_inv_nominal, mask, self.num_bodies],
            device=self.device
        )


def _checked_bounds(bounds, name: str, positive: bool = False) -> tuple:
    """Validate a (min, max) pair: two finite numbers, ordered, and
    nonnegative (or positive)."""
    if (isinstance(bounds, (str, bytes)) or not hasattr(bounds, "__len__")
            or len(bounds) != 2):
        raise ValueError(f"randomization {name}: expected (min, max), got {bounds!r}.")
    lo = validation.scalar(f"randomization {name} min", bounds[0])
    hi = validation.scalar(f"randomization {name} max", bounds[1])
    if lo > hi:
        raise ValueError(f"randomization {name}: min ({lo}) exceeds max ({hi}).")
    if lo < 0.0 or (positive and lo == 0.0):
        kind = "positive" if positive else "nonnegative"
        raise ValueError(f"randomization {name}: bounds must be {kind} (got min {lo}).")
    return (lo, hi)


@wp.kernel
def _init_randomization_config_kernel(
    config: wp.array(dtype=RandomizationConfig),
    idx: wp.int32,
    randomize_mass_dry: wp.int32,
    mass_dry_min: wp.float64, mass_dry_max: wp.float64,
    randomize_propellant: wp.int32,
    propellant_min: wp.float64, propellant_max: wp.float64,
    randomize_inertia: wp.int32,
    inertia_scale_min: wp.float64, inertia_scale_max: wp.float64,
):
    """Initialize the randomization config of one body on device."""
    config[idx].randomize_mass_dry = randomize_mass_dry
    config[idx].mass_dry_min = mass_dry_min
    config[idx].mass_dry_max = mass_dry_max
    config[idx].randomize_propellant = randomize_propellant
    config[idx].propellant_min = propellant_min
    config[idx].propellant_max = propellant_max
    config[idx].randomize_inertia = randomize_inertia
    config[idx].inertia_scale_min = inertia_scale_min
    config[idx].inertia_scale_max = inertia_scale_max


# Every randomization consumes RANDOMIZE_DRAWS draws of the entity's stream at
# fixed slots (0 dry mass, 1 inertia scale, 2 propellant), whichever are enabled.
RANDOMIZE_DRAWS = wp.constant(3)


@wp.func
def _randomize_entity(
    cfg: RandomizationConfig,
    states: wp.array(dtype=State),
    vehicles: wp.array(dtype=VehicleConfig),
    rng_state: wp.array(dtype=RNG_STATE),
    inertia_nominal: wp.array(dtype=wp.mat33d),
    inertia_inv_nominal: wp.array(dtype=wp.mat33d),
    tid: int
):
    """Resample one entity's randomized parameters, advancing its RNG."""
    st = rng_state[tid]

    if cfg.randomize_mass_dry != 0:
        vehicles[tid].mass_dry = sample_uniform(
            draw_uniform(st, 0, tid, STREAM_RANDOMIZE), cfg.mass_dry_min, cfg.mass_dry_max)

    if cfg.randomize_inertia != 0:
        scale = sample_uniform(draw_uniform(st, 1, tid, STREAM_RANDOMIZE),
                               cfg.inertia_scale_min, cfg.inertia_scale_max)
        vehicles[tid].inertia = inertia_nominal[tid] * scale
        vehicles[tid].inertia_inv = inertia_inv_nominal[tid] * (wp.float64(1.0) / scale)

    # Total mass = dry + propellant (uses the freshly sampled dry mass).
    # Momentum is rescaled so the velocity is preserved: the sampled
    # trajectory (position, velocity) is a property of the orbit, not of the
    # loaded mass — without this, overwriting the mass would silently change
    # the velocity (and break orbit-mode dynamic consistency, mom = mass * v).
    if cfg.randomize_propellant != 0:
        old_mass = states[tid].mass
        new_mass = vehicles[tid].mass_dry + sample_uniform(
            draw_uniform(st, 2, tid, STREAM_RANDOMIZE), cfg.propellant_min, cfg.propellant_max
        )
        if old_mass > wp.float64(0.0):
            states[tid].mom = states[tid].mom * (new_mass / old_mass)
        states[tid].mass = new_mass

    rng_state[tid] = advance(st, RANDOMIZE_DRAWS)


@wp.kernel
def randomize_kernel(
    states: wp.array(dtype=State),
    vehicles: wp.array(dtype=VehicleConfig),
    config: wp.array(dtype=RandomizationConfig),
    rng_state: wp.array(dtype=RNG_STATE),
    inertia_nominal: wp.array(dtype=wp.mat33d),
    inertia_inv_nominal: wp.array(dtype=wp.mat33d),
    num_bodies: wp.int32
) -> None:
    """Resample randomized parameters for all entities."""
    tid = wp.tid()
    _randomize_entity(config[tid % num_bodies], states, vehicles, rng_state,
                      inertia_nominal, inertia_inv_nominal, tid)


@wp.kernel
def randomize_masked_kernel(
    states: wp.array(dtype=State),
    vehicles: wp.array(dtype=VehicleConfig),
    config: wp.array(dtype=RandomizationConfig),
    rng_state: wp.array(dtype=RNG_STATE),
    inertia_nominal: wp.array(dtype=wp.mat33d),
    inertia_inv_nominal: wp.array(dtype=wp.mat33d),
    mask: wp.array(dtype=wp.bool),
    num_bodies: wp.int32
) -> None:
    """Resample randomized parameters for masked entities only."""
    tid = wp.tid()
    if mask[tid]:
        _randomize_entity(config[tid % num_bodies], states, vehicles, rng_state,
                          inertia_nominal, inertia_inv_nominal, tid)

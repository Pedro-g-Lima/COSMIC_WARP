"""
Reset functionality for initializing/resetting simulation environments.

Random sampling draws from counter-based per-entity streams kept in device
memory (cosmic_warp.core.rng), so repeated resets draw fresh values and
remain valid inside CUDA graph captures.
"""
import math

import warp as wp
from cosmic_warp.core import validation
from cosmic_warp.core.state import State
from cosmic_warp.core.vehicle import VehicleConfig
from cosmic_warp.core.entities import check_ids, expand_entity_ids, init_mask_kernel
from cosmic_warp.core.rng import (RNG_STATE, STREAM_RESET, advance, draw_uniform,
                                  sample_uniform, seed_states)
from cosmic_warp.physics.gravity import MU_EARTH, RE_EARTH
from cosmic_warp.physics.orbits import elements_to_rv, mean_to_true_anomaly


@wp.struct
class ResetConfig:
    """Configuration for environment reset (dynamic state only)."""

    # Position bounds (uniform random sampling)
    pos_min: wp.vec3d
    pos_max: wp.vec3d

    # Momentum bounds, or velocity bounds when mom_is_velocity == 1 (the
    # sampled vector is then multiplied by the sampled mass)
    mom_min: wp.vec3d
    mom_max: wp.vec3d
    mom_is_velocity: wp.int32

    # Angular momentum bounds
    ang_mom_min: wp.vec3d
    ang_mom_max: wp.vec3d

    # Mass bounds
    mass_min: wp.float64
    mass_max: wp.float64

    # Attitude sampling: rotations uniform (Haar) within a cone of half-angle
    # att_max_angle (in [0, pi]) around att_nominal. att_max_angle = pi is
    # uniform SO(3); att_max_angle = 0 always yields exactly att_nominal.
    att_nominal: wp.quatd
    att_max_angle: wp.float64

    # Sampling mode: 0 = Cartesian box (pos/mom bounds), 1 = orbital elements
    sample_mode: wp.int32

    # Orbital-element bounds (used when sample_mode == 1; angles in radians).
    # anomaly bounds are true anomaly, or mean anomaly when anomaly_is_mean.
    sma_min: wp.float64
    sma_max: wp.float64
    ecc_min: wp.float64
    ecc_max: wp.float64
    inc_min: wp.float64
    inc_max: wp.float64
    raan_min: wp.float64
    raan_max: wp.float64
    argp_min: wp.float64
    argp_max: wp.float64
    anom_min: wp.float64
    anom_max: wp.float64
    anomaly_is_mean: wp.int32


#: The SpacecraftSimulator(reset_config=...) keys: the constructor's bounds
RESET_CONFIG_KEYS = frozenset({"pos_bounds", "vel_bounds", "mom_bounds", "ang_mom_bounds",
                               "mass_bounds", "att_nominal", "att_max_angle", "seed"})


class ResetManager:
    """Manages entity resets with configurable initial conditions."""

    def __init__(
        self,
        num_envs: int,
        device: str = "cuda",
        num_bodies: int = 1,
        pos_bounds: tuple = None,
        mom_bounds: tuple = None,
        ang_mom_bounds: tuple = None,
        mass_bounds: tuple = None,
        att_nominal: tuple = None,
        att_max_angle: float = None,
        seed: int = 42,
        vel_bounds: tuple = None
    ):
        """
        Initialize reset manager.

        The bounds are used for random sampling (overridden by
        reset_to_conditions).

        Args:
            num_envs: Number of parallel environments
            device: Device to run on
            num_bodies: Number of bodies per environment
            pos_bounds: (min, max) position bounds [m]
            mom_bounds: (min, max) momentum bounds [kg*m/s]
            vel_bounds: (min, max) velocity bounds [m/s], alternative to
                mom_bounds: momentum = sampled mass * sampled velocity, so the
                bounds hold for any mass. Default: circular 400 km LEO speed
                along +Y, +/- 1 m/s
            ang_mom_bounds: (min, max) angular momentum bounds [kg*m^2/s]
            mass_bounds: (min, max) mass bounds [kg]
            att_nominal: nominal attitude quaternion (x, y, z, w), default identity
            att_max_angle: attitude cone half-angle [rad] around att_nominal,
                in [0, pi]. Default pi = uniform over SO(3); 0 = always exactly
                att_nominal
            seed: RNG seed for random initial conditions
        """
        self.num_envs = num_envs
        self.num_bodies = num_bodies
        self.num_entities = num_envs * num_bodies
        self.device = device

        # Defaults: a 1 kg spacecraft near a circular 400 km LEO orbit,
        # consistent with the simulator's default gravity + J2 physics.
        r0 = RE_EARTH + 400.0e3
        v0 = (MU_EARTH / r0) ** 0.5

        if pos_bounds is None:
            pos_bounds = ((r0 - 1.0e3, -1.0e3, -1.0e3), (r0 + 1.0e3, 1.0e3, 1.0e3))

        if mom_bounds is not None and vel_bounds is not None:
            raise ValueError("provide at most one of mom_bounds / vel_bounds")
        if mom_bounds is None and vel_bounds is None:
            # Circular orbital velocity along +Y, small dispersion; a velocity
            # box keeps the orbit valid whatever mass is configured
            vel_bounds = ((-1.0, v0 - 1.0, -1.0), (1.0, v0 + 1.0, 1.0))

        if ang_mom_bounds is None:
            ang_mom_bounds = ((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1))

        if mass_bounds is None:
            mass_bounds = (1.0, 1.0)

        if att_nominal is None:
            att_nominal = (0.0, 0.0, 0.0, 1.0)

        if att_max_angle is None:
            att_max_angle = math.pi

        # Per-body reset configs on device (all bodies start with the same
        # bounds; override per body via set_bounds / set_orbit_bounds).
        # Host-side copies are kept so partial overrides merge with the
        # current values. Orbit defaults match the Cartesian ones: a circular
        # 400 km LEO orbit, uniform in anomaly.
        self._bounds = [
            {
                "pos_bounds": pos_bounds,
                "mom_bounds": mom_bounds,
                "vel_bounds": vel_bounds,
                "ang_mom_bounds": ang_mom_bounds,
                "mass_bounds": mass_bounds,
                "att_nominal": att_nominal,
                "att_max_angle": att_max_angle,
                "sample_mode": 0,
                "semi_major_axis_bounds": (r0, r0),
                "eccentricity_bounds": (0.0, 0.0),
                "inclination_bounds": (0.0, 0.0),
                "raan_bounds": (0.0, 0.0),
                "arg_periapsis_bounds": (0.0, 0.0),
                "anomaly_bounds": (0.0, 2.0 * math.pi),
                "anomaly_is_mean": False,
            }
            for _ in range(num_bodies)
        ]
        self.config = wp.zeros(num_bodies, dtype=ResetConfig, device=device)
        self._update(range(num_bodies), {})

        # Vehicle configs whose dry masses floor every reset mass. The
        # simulator binds its own; a standalone manager's zeros floor nothing.
        self.vehicles = wp.zeros(self.num_entities, dtype=VehicleConfig, device=device)

        # Persistent per-entity RNG state
        self.rng_state = wp.empty(self.num_entities, dtype=RNG_STATE, device=device)
        self.reseed(seed)

        # Steps since each entity's last reset. Zeroed inside the reset
        # kernels themselves so state and counter can never drift apart;
        # incremented by the simulator each step.
        self.episode_steps = wp.zeros(self.num_entities, dtype=wp.int32, device=device)

        # Reusable selection buffers for build_mask (avoids per-call device
        # allocations)
        self._mask_buf = wp.zeros(self.num_entities, dtype=wp.bool, device=device)
        self._ids_buf = wp.zeros(self.num_entities, dtype=wp.int32, device=device)
        self._empty_mask = wp.empty(0, dtype=wp.bool, device=device)

        # Empty placeholders for omitted reset_to_conditions arrays (the
        # kernel treats shape 0 as "use the config default")
        self._empty_vec3 = wp.empty(0, dtype=wp.vec3d, device=device)
        self._empty_quat = wp.empty(0, dtype=wp.quatd, device=device)
        self._empty_f64 = wp.empty(0, dtype=wp.float64, device=device)

    def _update(self, body_ids, changes: dict) -> None:
        """Apply changes to the bounds of the selected bodies and write them to
        their device config slots. Every updated body is validated before any
        is stored, so a rejected call leaves all bounds unchanged."""
        updated = {body_id: _checked({**self._bounds[body_id], **changes})
                   for body_id in body_ids}
        for body_id, b in updated.items():
            self._bounds[body_id] = b
            wp.launch(
                kernel=_init_reset_config_kernel,
                dim=1,
                inputs=[self.config, int(body_id), *_config_inputs(b)],
                device=self.device
            )

    def reseed(self, seed: int) -> None:
        """Restart the per-entity RNG streams of the reset sampler at draw 0
        of ``seed`` (an integer in [0, 2**64))."""
        seed_states(self.rng_state, validation.seed("reset seed", seed))

    def set_bounds(
        self,
        *,
        pos_bounds: tuple = None,
        vel_bounds: tuple = None,
        mom_bounds: tuple = None,
        ang_mom_bounds: tuple = None,
        mass_bounds: tuple = None,
        att_nominal: tuple = None,
        att_max_angle: float = None,
        body_ids: list = None
    ) -> None:
        """
        Override reset sampling bounds for specific bodies.

        Args:
            pos_bounds: (min, max) position bounds [m] (None = keep current)
            vel_bounds: (min, max) velocity bounds [m/s]; momentum = sampled
                mass * sampled velocity. Setting one of mom_bounds/vel_bounds
                replaces the other (None = keep current)
            mom_bounds: (min, max) momentum bounds [kg*m/s] (None = keep current)
            ang_mom_bounds: (min, max) angular momentum bounds [kg*m^2/s] (None = keep current)
            mass_bounds: (min, max) mass bounds [kg] (None = keep current)
            att_nominal: nominal attitude quaternion (x, y, z, w) (None = keep current)
            att_max_angle: attitude cone half-angle [rad] around att_nominal,
                in [0, pi]; pi = uniform over SO(3), 0 = always exactly
                att_nominal (None = keep current)
            body_ids: List of body indices to configure (None = all bodies)
        """
        body_ids = (range(self.num_bodies) if body_ids is None
                    else check_ids("body_ids", body_ids, self.num_bodies))
        if mom_bounds is not None and vel_bounds is not None:
            raise ValueError("provide at most one of mom_bounds / vel_bounds")
        changes = {key: value for key, value in (
            ("pos_bounds", pos_bounds), ("ang_mom_bounds", ang_mom_bounds),
            ("mass_bounds", mass_bounds), ("att_nominal", att_nominal),
            ("att_max_angle", att_max_angle),
        ) if value is not None}
        if mom_bounds is not None:
            changes.update(mom_bounds=mom_bounds, vel_bounds=None)
        if vel_bounds is not None:
            changes.update(mom_bounds=None, vel_bounds=vel_bounds)
        self._update(body_ids, changes)

    def set_orbit_bounds(
        self,
        *,
        semi_major_axis_bounds: tuple = None,
        altitude_bounds: tuple = None,
        eccentricity_bounds: tuple = None,
        inclination_bounds: tuple = None,
        raan_bounds: tuple = None,
        arg_periapsis_bounds: tuple = None,
        true_anomaly_bounds: tuple = None,
        mean_anomaly_bounds: tuple = None,
        body_ids: list = None
    ) -> None:
        """
        Sample orbital elements instead of Cartesian boxes for specific bodies.

        Enables orbit sampling mode: on every reset each element is drawn
        uniformly within its bounds and converted to an ECI state entirely
        on device, so position and momentum are always dynamically consistent
        (momentum = mass * velocity). pos_bounds and mom_bounds are ignored
        in this mode; angular momentum, mass and attitude sampling apply
        unchanged. Angles in radians; the orbits are about Earth (MU_EARTH,
        the value the dynamics use).

        Unspecified elements keep their current values (defaults: a circular
        400 km LEO orbit, uniform true anomaly). Use set_sample_mode to
        switch a body back to Cartesian sampling.

        Args:
            semi_major_axis_bounds: (min, max) semi-major axis [m]
            altitude_bounds: (min, max) altitude above the Earth equatorial
                radius [m] (alternative to semi_major_axis_bounds)
            eccentricity_bounds: (min, max) eccentricity, within [0, 1)
            inclination_bounds: (min, max) inclination [rad], within [0, pi]
            raan_bounds: (min, max) right ascension of the ascending node [rad]
            arg_periapsis_bounds: (min, max) argument of periapsis [rad]
            true_anomaly_bounds: (min, max) true anomaly [rad]
            mean_anomaly_bounds: (min, max) mean anomaly [rad] — uniform in
                time along the orbit (alternative to true_anomaly_bounds)
            body_ids: List of body indices to configure (None = all bodies)
        """
        if semi_major_axis_bounds is not None and altitude_bounds is not None:
            raise ValueError(
                "provide at most one of semi_major_axis_bounds / altitude_bounds")
        if true_anomaly_bounds is not None and mean_anomaly_bounds is not None:
            raise ValueError(
                "provide at most one of true_anomaly_bounds / mean_anomaly_bounds"
            )
        if altitude_bounds is not None:
            semi_major_axis_bounds = tuple(
                RE_EARTH + validation.scalar("altitude_bounds", v)
                for v in _pair("altitude_bounds", altitude_bounds))

        body_ids = (range(self.num_bodies) if body_ids is None
                    else check_ids("body_ids", body_ids, self.num_bodies))
        changes = {key: value for key, value in (
            ("semi_major_axis_bounds", semi_major_axis_bounds),
            ("eccentricity_bounds", eccentricity_bounds),
            ("inclination_bounds", inclination_bounds), ("raan_bounds", raan_bounds),
            ("arg_periapsis_bounds", arg_periapsis_bounds),
        ) if value is not None}
        changes["sample_mode"] = 1
        if true_anomaly_bounds is not None:
            changes.update(anomaly_bounds=true_anomaly_bounds, anomaly_is_mean=False)
        if mean_anomaly_bounds is not None:
            changes.update(anomaly_bounds=mean_anomaly_bounds, anomaly_is_mean=True)
        self._update(body_ids, changes)

    def set_sample_mode(self, mode: str, *, body_ids: list = None) -> None:
        """
        Switch bodies between "cartesian" (box) and "orbit" (element) sampling.

        Args:
            mode: "cartesian" or "orbit"
            body_ids: List of body indices to configure (None = all bodies)
        """
        modes = {"cartesian": 0, "orbit": 1}
        if mode not in modes:
            raise ValueError(f"mode must be one of {sorted(modes)}, got {mode!r}")
        body_ids = (range(self.num_bodies) if body_ids is None
                    else check_ids("body_ids", body_ids, self.num_bodies))
        self._update(body_ids, {"sample_mode": modes[mode]})

    def get_bounds(self, body_id: int = 0) -> dict:
        """
        The reset bounds of one body, keyed by the set_bounds and
        set_orbit_bounds argument names, plus "sample_mode" ("cartesian" or
        "orbit"). The momentum box is under vel_bounds or mom_bounds,
        whichever is set, and the anomaly under true_anomaly_bounds or
        mean_anomaly_bounds; an altitude is reported as its
        semi_major_axis_bounds. Bounds are tuples of floats. A copy: edit
        with set_bounds, set_orbit_bounds and set_sample_mode.
        """
        (body_id,) = check_ids("body_id", [body_id], self.num_bodies)
        b = self._bounds[body_id]
        momentum = "mom_bounds" if b["vel_bounds"] is None else "vel_bounds"
        anomaly = "mean_anomaly_bounds" if b["anomaly_is_mean"] else "true_anomaly_bounds"
        return {
            "sample_mode": "orbit" if b["sample_mode"] == 1 else "cartesian",
            **{key: b[key] for key in (
                "pos_bounds", momentum, "ang_mom_bounds", "mass_bounds", "att_nominal",
                "att_max_angle", "semi_major_axis_bounds", "eccentricity_bounds",
                "inclination_bounds", "raan_bounds", "arg_periapsis_bounds")},
            anomaly: b["anomaly_bounds"],
        }

    def build_mask(
        self,
        env_ids: list = None,
        body_ids: list = None,
        entity_ids: list = None
    ) -> wp.array:
        """
        Fill the reusable selection mask from id selections.

        Reuses preallocated buffers, so no device memory is allocated. The
        returned array is shared between calls: consume it before calling
        build_mask again.

        Args:
            env_ids: List of environment indices (None = no selection)
            body_ids: List of body indices (None = all bodies)
            entity_ids: Optional flat entity indices (env_id * num_bodies + body_id)

        Returns:
            Bool mask array [num_entities], or None when the selection is empty
        """
        entity_ids = expand_entity_ids(self.num_envs, self.num_bodies, env_ids, body_ids, entity_ids)
        if entity_ids is None or len(entity_ids) == 0:
            return None

        self._mask_buf.zero_()
        ids_host = wp.array(entity_ids, dtype=wp.int32, device="cpu")
        wp.copy(self._ids_buf, ids_host, count=len(entity_ids))
        wp.launch(
            kernel=init_mask_kernel,
            dim=len(entity_ids),
            inputs=[self._mask_buf, self._ids_buf],
            device=self.device
        )
        return self._mask_buf

    def reset_all(self, states: wp.array(dtype=State)) -> None:
        """
        Reset all environments to random initial conditions.

        Args:
            states: WARP array of State structs to reset
        """
        wp.launch(
            kernel=reset_random_kernel,
            dim=self.num_entities,
            inputs=[states, self.config, self.vehicles, self.rng_state, self.episode_steps,
                    self.num_bodies],
            device=self.device
        )

    def reset_masked(self, states: wp.array(dtype=State), mask: wp.array(dtype=wp.bool)) -> None:
        """
        Reset specific entities using a boolean mask.

        Args:
            states: WARP array of State structs
            mask: WARP array of bools where True indicates entity to reset
        """
        wp.launch(
            kernel=reset_masked_kernel,
            dim=self.num_entities,
            inputs=[states, self.config, self.vehicles, self.rng_state, mask,
                    self.episode_steps, self.num_bodies],
            device=self.device
        )

    def check_orbit_defaults(
        self,
        env_ids: list = None,
        body_ids: list = None,
        entity_ids: list = None
    ) -> None:
        """Reject default positions/momenta for the selected orbit-sampled
        bodies.

        The reset_to_conditions defaults are the Cartesian bound centers,
        which for a body configured via set_orbit_bounds are stale leftovers
        rather than a state on the configured orbits.
        """
        ids = expand_entity_ids(self.num_envs, self.num_bodies, env_ids, body_ids, entity_ids)
        selected = (range(self.num_bodies) if ids is None
                    else sorted({i % self.num_bodies for i in ids}))
        for body_id in selected:
            if self._bounds[body_id]["sample_mode"] == 1:
                raise ValueError(
                    f"body {body_id} uses orbital-element sampling: "
                    "reset_to_conditions requires explicit positions and "
                    "momenta for orbit-mode bodies (the Cartesian bound "
                    "centers used as defaults would not lie on the "
                    "configured orbits)."
                )

    def check_conditions(self, **arrays) -> None:
        """Check reset_to_conditions arrays on their metadata (no device
        read): a Warp array of the field's dtype, of length num_entities, on
        this manager's device. A kernel would read a short array past its
        end, and an array on another device cannot be launched with."""
        dtypes = {"positions": wp.vec3d, "momenta": wp.vec3d, "angular_momenta": wp.vec3d,
                  "quaternions": wp.quatd, "masses": wp.float64}
        device = self.config.device
        for name, array in arrays.items():
            if array is None:
                continue
            dtype = dtypes[name]
            if not isinstance(array, wp.array) or array.dtype != dtype:
                got = (f"a Warp array of {array.dtype.__name__}"
                       if isinstance(array, wp.array) else type(array).__name__)
                raise ValueError(f"{name} must be a Warp array of {dtype.__name__}, got {got}")
            if array.shape != (self.num_entities,):
                raise ValueError(
                    f"{name} must have shape (num_entities,) = ({self.num_entities},), "
                    f"got {array.shape} (arrays are indexed by flat entity id, "
                    f"even when resetting a subset)"
                )
            if array.device != device:
                raise ValueError(f"{name} is on {array.device}, the simulator on {device}")

    def reset_to_conditions(
        self,
        states: wp.array(dtype=State),
        positions: wp.array = None,
        momenta: wp.array = None,
        angular_momenta: wp.array = None,
        quaternions: wp.array = None,
        masses: wp.array = None,
        mask: wp.array = None
    ) -> None:
        """
        Reset entities to specific conditions (dynamic state only).

        The condition arrays are always indexed by flat entity id, so they
        keep the full [num_entities] length even when resetting a subset.
        Only the arrays' metadata is checked here; sim.reset_to_conditions
        also checks the values, the selection and orbit-mode bodies.

        Args:
            states: WARP array of State structs
            positions: WARP array of positions [num_entities] (vec3d), None uses ResetConfig center
            momenta: WARP array of momenta [num_entities] (vec3d), None uses ResetConfig center
            angular_momenta: WARP array of angular momenta [num_entities] (vec3d), None uses ResetConfig center
            quaternions: WARP array of quaternions [num_entities] (quatd),
                normalized on write; None uses identity
            masses: WARP array of masses [num_entities] (float64), None uses ResetConfig center
            mask: Bool mask [num_entities] selecting the entities to reset;
                None resets all

        Raises:
            ValueError: if an array is not a Warp array of the listed dtype, of
                length num_entities, on this manager's device.
        """
        self.check_conditions(positions=positions, momenta=momenta,
                              angular_momenta=angular_momenta, quaternions=quaternions,
                              masses=masses)
        if mask is None:
            mask = self._empty_mask  # empty mask = all entities

        # Handle None by passing empty arrays (kernel checks shape[0])
        if positions is None:
            positions = self._empty_vec3
        if momenta is None:
            momenta = self._empty_vec3
        if angular_momenta is None:
            angular_momenta = self._empty_vec3
        if quaternions is None:
            quaternions = self._empty_quat
        if masses is None:
            masses = self._empty_f64

        wp.launch(
            kernel=reset_to_conditions_kernel,
            dim=self.num_entities,
            inputs=[
                states,
                self.config,
                self.vehicles,
                positions,
                momenta,
                angular_momenta,
                quaternions,
                masses,
                mask,
                self.episode_steps,
                self.num_bodies
            ],
            device=self.device
        )


@wp.kernel
def _init_reset_config_kernel(
    config: wp.array(dtype=ResetConfig),
    idx: wp.int32,
    pos_min: wp.vec3d, pos_max: wp.vec3d,
    mom_min: wp.vec3d, mom_max: wp.vec3d,
    mom_is_velocity: wp.int32,
    ang_mom_min: wp.vec3d, ang_mom_max: wp.vec3d,
    mass_min: wp.float64,
    mass_max: wp.float64,
    att_nominal: wp.quatd,
    att_max_angle: wp.float64,
    sample_mode: wp.int32,
    sma_min: wp.float64, sma_max: wp.float64,
    ecc_min: wp.float64, ecc_max: wp.float64,
    inc_min: wp.float64, inc_max: wp.float64,
    raan_min: wp.float64, raan_max: wp.float64,
    argp_min: wp.float64, argp_max: wp.float64,
    anom_min: wp.float64, anom_max: wp.float64,
    anomaly_is_mean: wp.int32
):
    """Initialize the reset config of one body on device."""
    config[idx].pos_min = pos_min
    config[idx].pos_max = pos_max
    config[idx].mom_min = mom_min
    config[idx].mom_max = mom_max
    config[idx].mom_is_velocity = mom_is_velocity
    config[idx].ang_mom_min = ang_mom_min
    config[idx].ang_mom_max = ang_mom_max
    config[idx].mass_min = mass_min
    config[idx].mass_max = mass_max
    config[idx].att_nominal = att_nominal
    config[idx].att_max_angle = att_max_angle
    config[idx].sample_mode = sample_mode
    config[idx].sma_min = sma_min
    config[idx].sma_max = sma_max
    config[idx].ecc_min = ecc_min
    config[idx].ecc_max = ecc_max
    config[idx].inc_min = inc_min
    config[idx].inc_max = inc_max
    config[idx].raan_min = raan_min
    config[idx].raan_max = raan_max
    config[idx].argp_min = argp_min
    config[idx].argp_max = argp_max
    config[idx].anom_min = anom_min
    config[idx].anom_max = anom_max
    config[idx].anomaly_is_mean = anomaly_is_mean


# Every reset consumes RESET_DRAWS draws of its entity's stream, whatever the
# sample mode, at fixed slots: 0-5 the orbital elements (orbit mode) or the
# position and momentum/velocity boxes; 6-8 angular momentum; 9-11 attitude;
# 12 mass.
RESET_DRAWS = wp.constant(13)


@wp.func
def _u(st: RNG_STATE, slot: int, tid: int) -> wp.float64:
    """Uniform [0, 1) draw ``slot`` of this reset."""
    return draw_uniform(st, slot, tid, STREAM_RESET)


@wp.func
def _sample_state(
    cfg: ResetConfig,
    mass_dry: wp.float64,
    rng_state: wp.array(dtype=RNG_STATE),
    tid: int
) -> State:
    """Draw a random State within the configured bounds, advancing the RNG.
    The mass is floored at mass_dry (see reset_masked_kernel)."""
    st = rng_state[tid]

    s = State()
    vel = wp.vec3d()  # orbit-mode velocity, turned into momentum once mass is drawn
    if cfg.sample_mode == 1:
        # Orbital-element sampling: uniform draws on each element, converted
        # to an ECI state on device, so pos/mom are dynamically consistent
        sma = sample_uniform(_u(st, 0, tid), cfg.sma_min, cfg.sma_max)
        ecc = sample_uniform(_u(st, 1, tid), cfg.ecc_min, cfg.ecc_max)
        inc = sample_uniform(_u(st, 2, tid), cfg.inc_min, cfg.inc_max)
        raan = sample_uniform(_u(st, 3, tid), cfg.raan_min, cfg.raan_max)
        argp = sample_uniform(_u(st, 4, tid), cfg.argp_min, cfg.argp_max)
        anom = sample_uniform(_u(st, 5, tid), cfg.anom_min, cfg.anom_max)
        if cfg.anomaly_is_mean == 1:
            anom = mean_to_true_anomaly(anom, ecc)
        rv = elements_to_rv(sma, ecc, inc, raan, argp, anom, wp.float64(MU_EARTH))
        s.pos = wp.spatial_top(rv)
        vel = wp.spatial_bottom(rv)
    else:
        s.pos = wp.vec3d(
            sample_uniform(_u(st, 0, tid), cfg.pos_min[0], cfg.pos_max[0]),
            sample_uniform(_u(st, 1, tid), cfg.pos_min[1], cfg.pos_max[1]),
            sample_uniform(_u(st, 2, tid), cfg.pos_min[2], cfg.pos_max[2])
        )
        s.mom = wp.vec3d(
            sample_uniform(_u(st, 3, tid), cfg.mom_min[0], cfg.mom_max[0]),
            sample_uniform(_u(st, 4, tid), cfg.mom_min[1], cfg.mom_max[1]),
            sample_uniform(_u(st, 5, tid), cfg.mom_min[2], cfg.mom_max[2])
        )
    s.ang_mom = wp.vec3d(
        sample_uniform(_u(st, 6, tid), cfg.ang_mom_min[0], cfg.ang_mom_max[0]),
        sample_uniform(_u(st, 7, tid), cfg.ang_mom_min[1], cfg.ang_mom_max[1]),
        sample_uniform(_u(st, 8, tid), cfg.ang_mom_min[2], cfg.ang_mom_max[2])
    )

    ua = _u(st, 9, tid)
    ub = _u(st, 10, tid)
    uc = _u(st, 11, tid)
    two_pi = wp.float64(2.0) * wp.float64(wp.PI)
    if cfg.att_max_angle >= wp.float64(wp.PI):
        # Uniform random unit quaternion (Shoemake's method)
        sqrt1_ua = wp.sqrt(wp.float64(1.0) - ua)
        sqrtua = wp.sqrt(ua)
        s.quat = wp.normalize(wp.quatd(
            sqrt1_ua * wp.sin(two_pi * ub),
            sqrt1_ua * wp.cos(two_pi * ub),
            sqrtua * wp.sin(two_pi * uc),
            sqrtua * wp.cos(two_pi * uc)
        ))
    else:
        # Haar-uniform rotation within a cone of half-angle att_max_angle
        # around att_nominal: uniform axis on the sphere, rotation angle with
        # density ~ sin^2(theta/2) (CDF ~ theta - sin(theta), inverted by
        # fixed-iteration bisection so it stays CUDA-graph safe)
        z = wp.float64(2.0) * ua - wp.float64(1.0)
        r = wp.sqrt(wp.max(wp.float64(0.0), wp.float64(1.0) - z * z))
        phi = two_pi * ub

        theta_max = cfg.att_max_angle
        c = uc * (theta_max - wp.sin(theta_max))
        lo = wp.float64(0.0)
        hi = theta_max
        for _ in range(32):
            mid = (lo + hi) * wp.float64(0.5)
            if mid - wp.sin(mid) < c:
                lo = mid
            else:
                hi = mid
        half = (lo + hi) * wp.float64(0.25)

        sin_half = wp.sin(half)
        perturbation = wp.quatd(
            r * wp.cos(phi) * sin_half,
            r * wp.sin(phi) * sin_half,
            z * sin_half,
            wp.cos(half)
        )
        s.quat = wp.normalize(wp.mul(cfg.att_nominal, perturbation))

    s.mass = wp.max(sample_uniform(_u(st, 12, tid), cfg.mass_min, cfg.mass_max), mass_dry)
    if cfg.sample_mode == 1:
        s.mom = vel * s.mass
    elif cfg.mom_is_velocity == 1:
        s.mom = s.mom * s.mass   # the Cartesian draw was a velocity

    rng_state[tid] = advance(st, RESET_DRAWS)
    return s


@wp.kernel
def reset_random_kernel(
    states: wp.array(dtype=State),
    config: wp.array(dtype=ResetConfig),
    vehicles: wp.array(dtype=VehicleConfig),
    rng_state: wp.array(dtype=RNG_STATE),
    episode_steps: wp.array(dtype=wp.int32),
    num_bodies: wp.int32
) -> None:
    """Reset all environments with random initial conditions (dynamic state only)."""
    tid = wp.tid()
    states[tid] = _sample_state(config[tid % num_bodies], vehicles[tid].mass_dry, rng_state, tid)
    episode_steps[tid] = 0


@wp.kernel
def reset_masked_kernel(
    states: wp.array(dtype=State),
    config: wp.array(dtype=ResetConfig),
    vehicles: wp.array(dtype=VehicleConfig),
    rng_state: wp.array(dtype=RNG_STATE),
    mask: wp.array(dtype=wp.bool),
    episode_steps: wp.array(dtype=wp.int32),
    num_bodies: wp.int32
) -> None:
    """Reset only masked environments (dynamic state only).

    Also the device path of reset_terminated, simulate(auto_reset_terminated)
    and the RL VecEnv, none of which can raise: a mass_bounds minimum below
    the dry mass (rejected by sim.reset(), but possible after a later
    set_bounds) spawns at the dry mass rather than below it.
    """
    tid = wp.tid()

    if mask[tid]:
        states[tid] = _sample_state(config[tid % num_bodies], vehicles[tid].mass_dry,
                                    rng_state, tid)
        episode_steps[tid] = 0


@wp.kernel
def reset_to_conditions_kernel(
    states: wp.array(dtype=State),
    config: wp.array(dtype=ResetConfig),
    vehicles: wp.array(dtype=VehicleConfig),
    positions: wp.array(dtype=wp.vec3d),
    momenta: wp.array(dtype=wp.vec3d),
    angular_momenta: wp.array(dtype=wp.vec3d),
    quaternions: wp.array(dtype=wp.quatd),
    masses: wp.array(dtype=wp.float64),
    mask: wp.array(dtype=wp.bool),
    episode_steps: wp.array(dtype=wp.int32),
    num_bodies: wp.int32
) -> None:
    """Reset to specific or default conditions (dynamic state only).

    An empty mask (shape 0) selects all entities.
    """
    tid = wp.tid()
    if mask.shape[0] > 0 and not mask[tid]:
        return
    cfg = config[tid % num_bodies]

    # Position: use provided or default to center of bounds
    if positions.shape[0] > 0:
        pos = positions[tid]
    else:
        pos = (cfg.pos_min + cfg.pos_max) * wp.float64(0.5)


    # Angular momentum: use provided or default to center of bounds
    if angular_momenta.shape[0] > 0:
        ang_mom = angular_momenta[tid]
    else:
        ang_mom = (cfg.ang_mom_min + cfg.ang_mom_max) * wp.float64(0.5)

    # Quaternion: use provided (normalized, so the first step already rotates
    # with a unit quaternion) or default to identity
    if quaternions.shape[0] > 0:
        quat = wp.normalize(quaternions[tid])
    else:
        quat = wp.quat_identity(dtype=wp.float64)

    # Mass: use provided or default to center of bounds, floored at the dry
    # mass (sim.reset_to_conditions rejects either below it beforehand)
    if masses.shape[0] > 0:
        mass = masses[tid]
    else:
        mass = (cfg.mass_min + cfg.mass_max) * wp.float64(0.5)
    mass = wp.max(mass, vehicles[tid].mass_dry)

    # Momentum: use provided or default to center of bounds (center
    # velocity times mass for velocity bounds)
    if momenta.shape[0] > 0:
        mom = momenta[tid]
    else:
        mom = (cfg.mom_min + cfg.mom_max) * wp.float64(0.5)
        if cfg.mom_is_velocity == 1:
            mom = mom * mass

    # Set state (only dynamic variables)
    states[tid].pos = pos
    states[tid].mom = mom
    states[tid].quat = quat
    states[tid].ang_mom = ang_mom
    states[tid].mass = mass
    episode_steps[tid] = 0


def _checked(b: dict) -> dict:
    """Validate one body's bounds; return them as tuples of floats (the
    attitude as given, normalized only on the device)."""
    att = tuple(float(v) for v in validation.array("att_nominal", b["att_nominal"]))
    if len(att) != 4:
        raise ValueError(f"att_nominal must have 4 components (x, y, z, w), got {len(att)}")
    if not any(att):
        raise ValueError(
            f"att_nominal must be a nonzero (x, y, z, w) quaternion, got {b['att_nominal']}"
        )

    def pair(key):
        return tuple(validation.scalar(key, v) for v in _pair(key, b[key]))

    sma = pair("semi_major_axis_bounds")
    ecc = pair("eccentricity_bounds")
    if not 0.0 < sma[0] <= sma[1]:
        raise ValueError(f"semi_major_axis_bounds must satisfy 0 < min <= max, got {sma}")
    if not 0.0 <= ecc[0] <= ecc[1] < 1.0:
        raise ValueError(
            f"eccentricity_bounds must satisfy 0 <= min <= max < 1 (bound orbits only), "
            f"got {ecc}"
        )
    angles = {key: pair(key) for key in ("inclination_bounds", "raan_bounds",
                                         "arg_periapsis_bounds", "anomaly_bounds")}
    for key, (lo, hi) in angles.items():
        if lo > hi:
            raise ValueError(f"{key} must satisfy min <= max, got {b[key]}")
    for v in angles["inclination_bounds"]:
        validation.polar_angle("inclination_bounds", v)
    velocity_mode = b["vel_bounds"] is not None
    momentum = "vel_bounds" if velocity_mode else "mom_bounds"
    mass = tuple(validation.scalar("mass_bounds", v, positive=True)
                 for v in _pair("mass_bounds", b["mass_bounds"]))
    if mass[0] > mass[1]:
        raise ValueError(f"mass_bounds must be (min, max) with min <= max, got {mass}")
    return {
        **b,
        **angles,
        "pos_bounds": _box("pos_bounds", b["pos_bounds"]),
        momentum: _box(momentum, b[momentum]),
        "ang_mom_bounds": _box("ang_mom_bounds", b["ang_mom_bounds"]),
        "mass_bounds": mass,
        "att_nominal": att,
        "att_max_angle": validation.polar_angle("att_max_angle", b["att_max_angle"]),
        "semi_major_axis_bounds": sma,
        "eccentricity_bounds": ecc,
        "anomaly_is_mean": bool(b["anomaly_is_mean"]),
    }


def _config_inputs(b: dict) -> list:
    """One body's checked bounds as _init_reset_config_kernel inputs."""
    norm = math.sqrt(sum(v * v for v in b["att_nominal"]))
    velocity_mode = b["vel_bounds"] is not None
    mom = b["vel_bounds"] if velocity_mode else b["mom_bounds"]
    return [
        wp.vec3d(*b["pos_bounds"][0]), wp.vec3d(*b["pos_bounds"][1]),
        wp.vec3d(*mom[0]), wp.vec3d(*mom[1]),
        int(velocity_mode),
        wp.vec3d(*b["ang_mom_bounds"][0]), wp.vec3d(*b["ang_mom_bounds"][1]),
        b["mass_bounds"][0], b["mass_bounds"][1],
        wp.quatd(*[v / norm for v in b["att_nominal"]]),
        b["att_max_angle"],
        int(b["sample_mode"]),
        *b["semi_major_axis_bounds"], *b["eccentricity_bounds"],
        *b["inclination_bounds"], *b["raan_bounds"],
        *b["arg_periapsis_bounds"], *b["anomaly_bounds"],
        int(b["anomaly_is_mean"]),
    ]


def _pair(name: str, bounds) -> tuple:
    """Unpack a (min, max) pair given as any two-element sequence or array."""
    if isinstance(bounds, (str, bytes)) or not hasattr(bounds, "__len__") or len(bounds) != 2:
        raise ValueError(f"{name} must be (min, max), got {bounds!r}")
    return bounds[0], bounds[1]


def _box(name: str, bounds) -> tuple:
    """Validate a (min, max) pair of 3-vectors with min <= max componentwise."""
    lo, hi = _pair(name, bounds)
    lo, hi = validation.vec3(f"{name} min", lo), validation.vec3(f"{name} max", hi)
    if any(a > b for a, b in zip(lo, hi)):
        raise ValueError(f"{name} must satisfy min <= max componentwise, got {lo} and {hi}")
    return lo, hi

"""YAML configuration loader for SpacecraftSimulator.

The configuration format is documented in docs/guides/configuration.md. Unknown keys
are rejected so typos fail loudly instead of silently falling back to defaults.
"""

import re
from typing import Any, Dict, List

import numpy as np
import warp as wp
import yaml

from cosmic_warp.core import validation
from cosmic_warp.actuators.reaction_wheels import DEFAULT_SPIN_AXES
from cosmic_warp.actuators.thrusters import DEFAULT_DIRECTIONS, DEFAULT_POSITIONS
from cosmic_warp.core.pipeline import SpacecraftSimulator
from cosmic_warp.physics.gravity import RE_EARTH
from cosmic_warp.physics.orbits import elements_to_state, mean_to_true_anomaly


# ---------------------------------------------------------------------------
# Schema: allowed keys per section (unknown keys raise ValueError)
# ---------------------------------------------------------------------------
_TOP_KEYS = {"simulation", "physics", "termination", "reset", "body", "initial_conditions",
             "randomization"}
_SIMULATION_KEYS = {"num_envs", "num_bodies", "integrator", "dt", "device"}
_PHYSICS_KEYS = {"gravity", "j2_perturbation", "gravity_gradient", "magnetic_torque", "drag"}
_TERMINATION_KEYS = {"check_bounds", "check_planet_impact", "pos_limit", "mom_limit",
                     "ang_mom_limit", "mass_limit", "planet_radius"}
_RESET_KEYS = {"pos_bounds", "mom_bounds", "vel_bounds", "ang_mom_bounds", "mass_bounds",
               "att_nominal", "att_max_angle", "orbit", "seed", "per_body"}
_RESET_PER_BODY_KEYS = {"body_id", "pos_bounds", "mom_bounds", "vel_bounds", "ang_mom_bounds",
                        "mass_bounds",
                        "att_nominal", "att_max_angle", "orbit"}
_RESET_ORBIT_KEYS = {"semi_major_axis", "altitude", "eccentricity", "inclination",
                     "raan", "arg_periapsis", "true_anomaly", "mean_anomaly"}
_BODY_KEYS = {"defaults", "per_body"}
_BODY_DEFAULTS_KEYS = {"vehicle", "actuators"}
_BODY_PER_BODY_KEYS = {"body_id", "vehicle", "actuators"}
_ACTUATORS_KEYS = {"thrusters", "reaction_wheels"}
_VEHICLE_KEYS = {"mass_dry", "inertia", "area_drag", "cd", "magnetic_dipole"}
_THRUSTER_KEYS = {"max_thrust", "isp", "positions", "directions"}
_RW_KEYS = {"max_torque", "max_momentum", "wheel_inertia", "viscous_damping", "spin_axes"}
_IC_KEYS = {"positions", "velocities", "momenta", "angular_momenta", "quaternions",
            "masses", "per_body"}
_IC_PER_BODY_KEYS = {"body_id", "position", "velocity", "momentum", "angular_momentum",
                     "quaternion", "mass", "orbit"}
_ORBIT_KEYS = {"semi_major_axis", "altitude", "eccentricity", "inclination", "raan",
               "arg_periapsis", "true_anomaly", "mean_anomaly"}
_RANDOMIZATION_KEYS = {"seed", "mass_dry", "propellant", "inertia_scale", "per_body"}
_RANDOMIZATION_PER_BODY_KEYS = {"body_id", "mass_dry", "propellant", "inertia_scale"}


class _Loader(yaml.SafeLoader):
    """SafeLoader with YAML 1.2 floats: PyYAML follows YAML 1.1, where an
    exponent needs a dot and a sign, so `1e5` or `1.0e5` would load as
    strings."""


# Copy the resolver table (PyYAML shares the per-character lists with the
# base class) before adding the YAML 1.2 float pattern
_Loader.yaml_implicit_resolvers = {
    k: list(v) for k, v in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
                |[-+]?\.(?:inf|Inf|INF)
                |\.(?:nan|NaN|NAN))$""", re.X),
    list("-+0123456789."),
)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.load(handle, Loader=_Loader) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping.")
    return data


def _check_keys(cfg: Dict[str, Any], allowed: set, name: str) -> None:
    if not isinstance(cfg, dict):
        raise ValueError(f"{name} must be a mapping, got {cfg!r}")
    unknown = set(cfg) - allowed
    if unknown:
        raise ValueError(
            f"Unknown key(s) in {name}: {sorted(unknown)}. Allowed: {sorted(allowed)}."
        )


def _check_flags(cfg: Dict[str, Any], keys: set, name: str) -> None:
    """Flags must be YAML booleans: a quoted "false" is a non-empty string,
    which would silently enable the feature."""
    for key in keys & set(cfg):
        if not isinstance(cfg[key], bool):
            raise ValueError(f"{name}.{key} must be true or false, got {cfg[key]!r}")


def _entries(entries, name: str) -> list:
    """A per_body section: a list of mappings (None: no entries)."""
    if entries is None:
        return []
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        raise ValueError(f"{name} must be a list of mappings, got {entries!r}")
    return entries


def _check_unique_body_ids(entries, name: str) -> None:
    seen = set()
    for entry in _entries(entries, name):
        body_id = entry.get("body_id")
        if body_id in seen:
            raise ValueError(f"{name}: duplicate body_id {body_id}")
        seen.add(body_id)


def _get_body_id(entry: Dict[str, Any], num_bodies: int, name: str) -> int:
    body_id = entry.get("body_id")
    if body_id is None:
        raise ValueError(f"{name} entries must include body_id.")
    if isinstance(body_id, bool) or not isinstance(body_id, int):
        raise ValueError(f"{name}: body_id must be an integer, got {body_id!r}.")
    if not 0 <= body_id < num_bodies:
        raise ValueError(f"{name}: body_id {body_id} out of range [0, {num_bodies - 1}].")
    return body_id


def _ensure_length(values: List[Any], expected: int, name: str) -> None:
    if len(values) != expected:
        raise ValueError(f"{name} must have length {expected}.")


def _vec3_array(values, device: str, name: str, count: int) -> wp.array:
    arr = validation.array(name, values, 3)
    _ensure_length(arr, count, name)
    return wp.array(arr, dtype=wp.vec3d, device=device)


def _unit_quaternions(values, name: str) -> np.ndarray:
    """Normalize [qx, qy, qz, qw] rows on load; reject zero or non-finite."""
    arr = validation.array(name, values, 4)
    norm = np.linalg.norm(arr, axis=1)
    if np.any(norm == 0.0):
        raise ValueError(f"{name} must be nonzero [qx, qy, qz, qw] quaternions.")
    return arr / norm[:, None]


def _quat_array(values, device: str, name: str, count: int) -> wp.array:
    arr = _unit_quaternions(values, name)
    _ensure_length(arr, count, name)
    return wp.array(arr, dtype=wp.quatd, device=device)


def _float_array(values, device: str, name: str, count: int) -> wp.array:
    arr = validation.array(name, values)
    _ensure_length(arr, count, name)
    return wp.array(arr, dtype=wp.float64, device=device)


def _pair(value, name: str) -> tuple:
    """Normalize a scalar or [min, max] list to an ordered (min, max) tuple."""
    if not isinstance(value, (list, tuple)):
        x = validation.scalar(name, value)
        return (x, x)
    if len(value) != 2:
        raise ValueError(f"{name} must be a scalar or a [min, max] pair.")
    values = [validation.scalar(name, v) for v in value]
    if values[0] > values[1]:
        raise ValueError(f"{name} must satisfy min <= max, got {values}.")
    return (values[0], values[1])


def _orbit_bounds_kwargs(spec: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Convert a reset orbit spec into ResetManager.set_orbit_bounds kwargs."""
    _check_keys(spec, _RESET_ORBIT_KEYS, name)
    if "semi_major_axis" in spec and "altitude" in spec:
        raise ValueError(f"{name}: provide at most one of semi_major_axis or altitude.")
    if "true_anomaly" in spec and "mean_anomaly" in spec:
        raise ValueError(f"{name}: provide at most one of true_anomaly or mean_anomaly.")

    # Each key maps to the set_orbit_bounds argument <key>_bounds
    kwargs = {f"{key}_bounds": _pair(value, f"{name}.{key}") for key, value in spec.items()}
    if "altitude" in spec:
        alt = kwargs.pop("altitude_bounds")
        kwargs["semi_major_axis_bounds"] = (RE_EARTH + alt[0], RE_EARTH + alt[1])
    if "inclination" in spec:
        for v in kwargs["inclination_bounds"]:
            validation.polar_angle(f"{name}.inclination", v)
    return kwargs


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _dry_masses(body_cfg: Dict[str, Any], num_bodies: int) -> List[float]:
    """Effective per-body dry mass after defaults + per_body overrides."""
    defaults = (body_cfg.get("defaults") or {}).get("vehicle") or {}
    dry = [validation.scalar("body.defaults.vehicle.mass_dry",
                             defaults.get("mass_dry", 1.0))] * num_bodies
    for entry in body_cfg.get("per_body") or []:
        vehicle = entry.get("vehicle") or {}
        if "mass_dry" in vehicle:
            dry[int(entry["body_id"])] = validation.scalar("body.per_body vehicle.mass_dry",
                                                           vehicle["mass_dry"])
    return dry


def _rand_bounds(rand_cfg: Dict[str, Any], num_bodies: int) -> List[Dict[str, Any]]:
    """Effective per-body randomization bounds after per_body overrides."""
    per_body = {int(entry["body_id"]): entry for entry in rand_cfg.get("per_body") or []}
    effective = []
    for body_id in range(num_bodies):
        entry = {k: rand_cfg[k] for k in ("mass_dry", "propellant", "inertia_scale")
                 if k in rand_cfg}
        override = per_body.get(body_id) or {}
        entry.update({k: v for k, v in override.items() if k != "body_id"})
        effective.append(entry)
    return effective


def _validate_reset_orbit(
    reset_cfg: Dict[str, Any],
    termination: Dict[str, Any],
    num_bodies: int
) -> None:
    """
    Validate reset orbit specs: element sanity plus the perigee cross-check.

    The worst-case perigee (sma_min * (1 - ecc_max)) of every orbit-sampled
    body must clear termination.planet_radius when planet-impact checking is
    enabled — otherwise resets could spawn entities on crashing orbits.
    """
    global_spec = reset_cfg.get("orbit")
    per_body = {
        int(entry["body_id"]): entry["orbit"]
        for entry in reset_cfg.get("per_body") or []
        if "orbit" in entry
    }
    if global_spec is None and not per_body:
        return

    global_kwargs = _orbit_bounds_kwargs(global_spec, "reset.orbit") if global_spec else {}
    check_impact = bool(termination.get("check_planet_impact", False))
    radius = validation.scalar("termination.planet_radius",
                               termination.get("planet_radius", RE_EARTH))
    default_sma = RE_EARTH + 400.0e3  # ResetManager's orbit default

    for body_id in range(num_bodies):
        if global_spec is None and body_id not in per_body:
            continue
        kwargs = dict(global_kwargs)
        if body_id in per_body:
            kwargs.update(_orbit_bounds_kwargs(
                per_body[body_id], f"reset.per_body (body {body_id}).orbit"))
        sma = kwargs.get("semi_major_axis_bounds", (default_sma, default_sma))
        ecc = kwargs.get("eccentricity_bounds", (0.0, 0.0))
        if sma[0] <= 0.0:
            raise ValueError(
                f"reset orbit of body {body_id}: semi-major axis must be positive."
            )
        if not 0.0 <= ecc[0] <= ecc[1] < 1.0:
            raise ValueError(
                f"reset orbit of body {body_id}: eccentricity must be within "
                f"[0, 1) (bound orbits only), got {list(ecc)}."
            )
        if check_impact and radius > 0.0:
            perigee = sma[0] * (1.0 - ecc[1])
            if perigee <= radius:
                raise ValueError(
                    f"reset orbit of body {body_id}: minimum perigee "
                    f"({perigee:.6e} m) is at or below termination.planet_radius "
                    f"({radius:.6e} m); resets could spawn entities on crashing "
                    f"orbits. Raise semi_major_axis/altitude or lower eccentricity."
                )


def _validate_config(config: Dict[str, Any]) -> None:
    _check_keys(config, _TOP_KEYS, "config")

    sim_cfg = config.get("simulation") or {}
    _check_keys(sim_cfg, _SIMULATION_KEYS, "simulation")
    _check_keys(config.get("physics") or {}, _PHYSICS_KEYS, "physics")
    _check_keys(config.get("termination") or {}, _TERMINATION_KEYS, "termination")
    _check_flags(config.get("physics") or {}, _PHYSICS_KEYS, "physics")
    _check_flags(config.get("termination") or {}, {"check_bounds", "check_planet_impact"},
                 "termination")

    num_bodies = validation.positive_int("simulation.num_bodies", sim_cfg.get("num_bodies", 1))

    def check_actuators(act: Dict[str, Any], name: str) -> None:
        _check_keys(act, _ACTUATORS_KEYS, name)
        _check_keys(act.get("thrusters") or {}, _THRUSTER_KEYS, f"{name}.thrusters")
        _check_keys(act.get("reaction_wheels") or {}, _RW_KEYS, f"{name}.reaction_wheels")

    body_cfg = config.get("body") or {}
    _check_keys(body_cfg, _BODY_KEYS, "body")
    defaults = body_cfg.get("defaults") or {}
    _check_keys(defaults, _BODY_DEFAULTS_KEYS, "body.defaults")
    _check_keys(defaults.get("vehicle") or {}, _VEHICLE_KEYS, "body.defaults.vehicle")
    check_actuators(defaults.get("actuators") or {}, "body.defaults.actuators")
    _check_unique_body_ids(body_cfg.get("per_body"), "body.per_body")
    for entry in body_cfg.get("per_body") or []:
        _check_keys(entry, _BODY_PER_BODY_KEYS, "body.per_body entry")
        _get_body_id(entry, num_bodies, "body.per_body")
        _check_keys(entry.get("vehicle") or {}, _VEHICLE_KEYS, "body.per_body.vehicle")
        check_actuators(entry.get("actuators") or {}, "body.per_body.actuators")

    reset_cfg = config.get("reset") or {}
    _check_keys(reset_cfg, _RESET_KEYS, "reset")
    _check_unique_body_ids(reset_cfg.get("per_body"), "reset.per_body")
    for entry in reset_cfg.get("per_body") or []:
        _check_keys(entry, _RESET_PER_BODY_KEYS, "reset.per_body entry")
        _get_body_id(entry, num_bodies, "reset.per_body")
    _validate_reset_orbit(reset_cfg, config.get("termination") or {}, num_bodies)

    ic_cfg = config.get("initial_conditions") or {}
    _check_keys(ic_cfg, _IC_KEYS, "initial_conditions")
    for entry in _entries(ic_cfg.get("per_body"), "initial_conditions.per_body"):
        _check_keys(entry, _IC_PER_BODY_KEYS, "initial_conditions.per_body entry")
        _get_body_id(entry, num_bodies, "initial_conditions.per_body")

    rand_cfg = config.get("randomization") or {}
    _check_keys(rand_cfg, _RANDOMIZATION_KEYS, "randomization")
    _check_unique_body_ids(rand_cfg.get("per_body"), "randomization.per_body")
    for entry in rand_cfg.get("per_body") or []:
        _check_keys(entry, _RANDOMIZATION_PER_BODY_KEYS, "randomization.per_body entry")
        _get_body_id(entry, num_bodies, "randomization.per_body")
    rand = _rand_bounds(rand_cfg, num_bodies)

    # Cross-check: prescribed initial masses must not fall below the dry mass
    # of the corresponding body (such entities would silently spawn "out of
    # propellant" because state mass is clamped to mass_dry). When
    # randomization is active, mass_dry may itself be sampled, so they are
    # checked against the largest possible dry mass. Reset mass_bounds are
    # checked by sim.reset() for every configuration path.
    dry = _dry_masses(body_cfg, num_bodies)
    dry_max = [
        max(dry[b], _pair(rand[b]["mass_dry"], "randomization.mass_dry")[1])
        if "mass_dry" in rand[b] else dry[b]
        for b in range(num_bodies)
    ]

    if "masses" in ic_cfg:
        for i, mass in enumerate(validation.array("initial_conditions.masses", ic_cfg["masses"])):
            if mass < dry_max[i % num_bodies]:
                raise ValueError(
                    f"initial_conditions.masses[{i}] = {mass} is below mass_dry "
                    f"({dry_max[i % num_bodies]}) of body {i % num_bodies}."
                )
    for entry in ic_cfg.get("per_body") or []:
        body_id = int(entry["body_id"])
        if ("mass" in entry and validation.scalar(
                f"initial_conditions.per_body body {body_id} mass", entry["mass"]) < dry_max[body_id]):
            raise ValueError(
                f"initial_conditions.per_body: mass {entry['mass']} of body "
                f"{body_id} is below its mass_dry ({dry_max[body_id]})."
            )


# ---------------------------------------------------------------------------
# Body configuration
# ---------------------------------------------------------------------------

def _apply_body_config(sim: SpacecraftSimulator, cfg: Dict[str, Any]) -> None:
    """Configure bodies. Each setter is given every field (it leaves omitted
    ones unchanged), so a key omitted in the YAML takes its default."""
    if not cfg:
        return

    base_vehicle = {
        "mass_dry": 1.0,
        "inertia": None,          # cube: mass_dry / 6 on the diagonal
        "area_drag": 1.0,
        "cd": 2.2,
        "magnetic_dipole": [0.0, 0.0, 0.0],
    }
    base_thrusters = {
        "max_thrust": 10.0,
        "isp": 0.0,
        "positions": DEFAULT_POSITIONS,
        "directions": DEFAULT_DIRECTIONS,
    }
    base_reaction_wheels = {
        "max_torque": 0.1,
        "max_momentum": 0.0,
        "wheel_inertia": 0.01,
        "viscous_damping": 0.0,
        "spin_axes": DEFAULT_SPIN_AXES,
    }

    def full(base: Dict[str, Any], *layers: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        for layer in layers:
            merged.update(layer)
        for key, value in merged.items():
            if value is None and key == "inertia":
                mass_dry = validation.scalar("mass_dry", merged["mass_dry"], positive=True)
                merged[key] = np.eye(3) * (mass_dry / 6.0)
            elif value is None and isinstance(base[key], tuple | list):
                merged[key] = base[key]
            elif value is None:
                validation.scalar(key, value)  # a null number: rejected
        return merged

    defaults = cfg.get("defaults") or {}
    default_vehicle = defaults.get("vehicle") or {}
    default_actuators = defaults.get("actuators") or {}
    default_thrusters = default_actuators.get("thrusters") or {}
    default_reaction_wheels = default_actuators.get("reaction_wheels") or {}

    if default_vehicle:
        sim.vehicle.set_config(**full(base_vehicle, default_vehicle))
    if default_thrusters:
        sim.thruster_manager.set_config(**full(base_thrusters, default_thrusters))
    if default_reaction_wheels:
        sim.rw_manager.set_config(**full(base_reaction_wheels, default_reaction_wheels))

    for entry in cfg.get("per_body") or []:
        body_id = int(entry["body_id"])

        vehicle = entry.get("vehicle")
        if vehicle:
            sim.vehicle.set_config(**full(base_vehicle, default_vehicle, vehicle),
                                   body_ids=[body_id])

        actuators = entry.get("actuators") or {}
        thrusters = actuators.get("thrusters")
        if thrusters:
            sim.thruster_manager.set_config(
                **full(base_thrusters, default_thrusters, thrusters), body_ids=[body_id])

        reaction_wheels = actuators.get("reaction_wheels")
        if reaction_wheels:
            sim.rw_manager.set_config(
                **full(base_reaction_wheels, default_reaction_wheels, reaction_wheels),
                body_ids=[body_id])


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

def _orbit_to_state(spec: Dict[str, Any], name: str):
    """Expand an orbital-element spec into (position, velocity). Angles in radians."""
    _check_keys(spec, _ORBIT_KEYS, name)
    if ("semi_major_axis" in spec) == ("altitude" in spec):
        raise ValueError(f"{name}: provide exactly one of semi_major_axis or altitude.")
    if "true_anomaly" in spec and "mean_anomaly" in spec:
        raise ValueError(f"{name}: provide at most one of true_anomaly or mean_anomaly.")

    def number(key, default=0.0):
        return validation.scalar(f"{name}.{key}", spec.get(key, default))

    sma = (number("semi_major_axis") if "semi_major_axis" in spec
           else RE_EARTH + number("altitude"))
    ecc = number("eccentricity")
    nu = (float(mean_to_true_anomaly(wp.float64(number("mean_anomaly")), wp.float64(ecc)))
          if "mean_anomaly" in spec else number("true_anomaly"))
    return elements_to_state(
        sma,
        eccentricity=ecc,
        inclination=validation.polar_angle(f"{name}.inclination", spec.get("inclination", 0.0)),
        raan=number("raan"),
        arg_periapsis=number("arg_periapsis"),
        true_anomaly=nu,
    )


def _expand_per_body_conditions(
    entries: List[Dict[str, Any]],
    num_envs: int,
    num_bodies: int,
    device: str,
) -> Dict[str, wp.array]:
    """Broadcast per-body initial conditions across all environments."""
    by_body: Dict[int, Dict[str, Any]] = {}
    for entry in entries:
        body_id = int(entry["body_id"])
        if body_id in by_body:
            raise ValueError(f"initial_conditions.per_body: duplicate body_id {body_id}.")
        by_body[body_id] = dict(entry)

    if sorted(by_body) != list(range(num_bodies)):
        raise ValueError(
            "initial_conditions.per_body must contain every body_id from 0 to "
            f"num_bodies-1 ({num_bodies - 1}) exactly once."
        )

    # Expand orbit specs into explicit position + velocity
    for body_id, entry in by_body.items():
        if "orbit" in entry:
            for key in ("position", "velocity", "momentum"):
                if key in entry:
                    raise ValueError(
                        f"initial_conditions.per_body body {body_id}: 'orbit' "
                        f"cannot be combined with '{key}'."
                    )
            pos, vel = _orbit_to_state(
                entry.pop("orbit"), f"initial_conditions.per_body body {body_id} orbit"
            )
            entry["position"] = pos.tolist()
            entry["velocity"] = vel.tolist()

    def field(name: str):
        present = [name in by_body[b] for b in range(num_bodies)]
        if not any(present):
            return None
        if not all(present):
            raise ValueError(
                f"initial_conditions.per_body: '{name}' must be given for all "
                f"bodies or none (missing for body "
                f"{present.index(False)}). Note that 'orbit' expands to "
                f"position + velocity."
            )
        return [by_body[b][name] for b in range(num_bodies)]

    def numbers(key, width=None):
        values = field(key)
        return (None if values is None
                else validation.array(f"initial_conditions.per_body {key}", values, width))

    positions = numbers("position", 3)
    velocities = numbers("velocity", 3)
    momenta = numbers("momentum", 3)
    masses = numbers("mass")

    if velocities is not None and momenta is not None:
        raise ValueError("initial_conditions.per_body: provide 'velocity' or 'momentum', not both.")
    if velocities is not None:
        if masses is None:
            raise ValueError(
                "initial_conditions.per_body: 'velocity' (or 'orbit') requires "
                "'mass' on every body (momentum = mass * velocity)."
            )
        momenta = masses[:, None] * velocities

    angular_momenta = numbers("angular_momentum", 3)
    quaternions = field("quaternion")
    if quaternions is not None:
        quaternions = _unit_quaternions(quaternions, "initial_conditions.per_body quaternion")

    def tile_rows(arr, dtype):
        if arr is None:
            return None
        return wp.array(np.tile(arr, (num_envs, 1)), dtype=dtype, device=device)

    return {
        "positions": tile_rows(positions, wp.vec3d),
        "momenta": tile_rows(momenta, wp.vec3d),
        "angular_momenta": tile_rows(angular_momenta, wp.vec3d),
        "quaternions": tile_rows(quaternions, wp.quatd),
        "masses": None if masses is None else wp.array(
            np.tile(masses, num_envs),
            dtype=wp.float64, device=device,
        ),
    }


def _build_per_entity_conditions(
    cfg: Dict[str, Any], count: int, device: str
) -> Dict[str, wp.array]:
    """Build initial-condition arrays from explicit per-entity lists."""
    momenta = None
    if "velocities" in cfg:
        if "momenta" in cfg:
            raise ValueError("initial_conditions: provide 'velocities' or 'momenta', not both.")
        if "masses" not in cfg:
            raise ValueError(
                "initial_conditions: 'velocities' requires 'masses' "
                "(momentum = mass * velocity)."
            )
        velocities = validation.array("velocities", cfg["velocities"], 3)
        _ensure_length(velocities, count, "velocities")
        masses = validation.array("masses", cfg["masses"])
        _ensure_length(masses, count, "masses")
        momenta = wp.array(masses[:, None] * velocities, dtype=wp.vec3d, device=device)
    elif "momenta" in cfg:
        momenta = _vec3_array(cfg["momenta"], device, "momenta", count)

    return {
        "positions": _vec3_array(cfg["positions"], device, "positions", count)
        if "positions" in cfg else None,
        "momenta": momenta,
        "angular_momenta": _vec3_array(cfg["angular_momenta"], device, "angular_momenta", count)
        if "angular_momenta" in cfg else None,
        "quaternions": _quat_array(cfg["quaternions"], device, "quaternions", count)
        if "quaternions" in cfg else None,
        "masses": _float_array(cfg["masses"], device, "masses", count)
        if "masses" in cfg else None,
    }


def _apply_initial_conditions(sim: SpacecraftSimulator, cfg: Dict[str, Any]) -> None:
    if not cfg:
        return

    per_body = cfg.get("per_body")
    per_entity_keys = set(cfg) - {"per_body"}
    if per_body and per_entity_keys:
        raise ValueError(
            "initial_conditions: use either 'per_body' or per-entity lists "
            f"({sorted(per_entity_keys)}), not both."
        )

    if per_body:
        arrays = _expand_per_body_conditions(per_body, sim.num_envs, sim.num_bodies, sim.device)
    else:
        arrays = _build_per_entity_conditions(cfg, sim.num_entities, sim.device)

    # A drawn orbit-mode state is a consistent (position, velocity) pair:
    # replacing only one half would put the body on another orbit.
    if (arrays["positions"] is None) != (arrays["momenta"] is None):
        orbit_bodies = [b for b in range(sim.num_bodies)
                        if sim.reset_manager._bounds[b]["sample_mode"] == 1]
        if orbit_bodies:
            raise ValueError(
                f"initial_conditions: body {orbit_bodies[0]} uses reset.orbit sampling, "
                f"so positions and momenta/velocities must be given together (the "
                f"drawn half would not lie on the same orbit)."
            )

    # Omitted fields keep the state drawn by the preceding sim.reset() (with
    # randomization applied). A prescribed mass keeps the drawn velocity, so
    # the momentum is rescaled rather than reused.
    drawn = sim.states.numpy()
    fill = {"positions": ("pos", wp.vec3d), "angular_momenta": ("ang_mom", wp.vec3d),
            "quaternions": ("quat", wp.quatd), "masses": ("mass", wp.float64)}
    if arrays["momenta"] is None:
        mom = drawn["mom"]
        if arrays["masses"] is not None:
            mom = mom / drawn["mass"][:, None] * arrays["masses"].numpy()[:, None]
        arrays["momenta"] = wp.array(mom, dtype=wp.vec3d, device=sim.device)
    for key, (field, dtype) in fill.items():
        if arrays[key] is None:
            arrays[key] = wp.array(drawn[field], dtype=dtype, device=sim.device)

    sim.reset_to_conditions(**arrays)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def create_simulator_from_config(config: Dict[str, Any]) -> SpacecraftSimulator:
    _validate_config(config)

    sim_cfg = config.get("simulation") or {}
    physics = config.get("physics") or {}
    termination = config.get("termination") or {}
    reset_cfg = dict(config.get("reset") or {})
    per_body_reset = reset_cfg.pop("per_body", None) or []
    orbit_spec = reset_cfg.pop("orbit", None)

    # The section keys are the constructor's argument names: omitted keys
    # take the constructor's defaults, which are defined there only
    sim = SpacecraftSimulator(**sim_cfg, **physics, **termination,
                              reset_config=reset_cfg or None)

    _apply_body_config(sim, config.get("body") or {})

    if orbit_spec:
        sim.reset_manager.set_orbit_bounds(
            **_orbit_bounds_kwargs(orbit_spec, "reset.orbit"))

    for entry in per_body_reset:
        body_id = int(entry["body_id"])
        bounds = {k: v for k, v in entry.items() if k not in ("body_id", "orbit")}
        if bounds:
            sim.reset_manager.set_bounds(**bounds, body_ids=[body_id])
        if "orbit" in entry:
            sim.reset_manager.set_orbit_bounds(
                **_orbit_bounds_kwargs(entry["orbit"], f"reset.per_body (body {body_id}).orbit"),
                body_ids=[body_id])

    # Randomization is configured after the body config so nominal inertia
    # tensors are captured correctly, and before the initial conditions so the
    # initial population is already randomized.
    rand_cfg = config.get("randomization") or {}
    if rand_cfg:
        if "seed" in rand_cfg:
            sim.randomizer.reseed(rand_cfg["seed"])
        for body_id, bounds in enumerate(_rand_bounds(rand_cfg, sim.num_bodies)):
            if bounds:
                sim.randomizer.set_bounds(**bounds, body_ids=[body_id])

    # A loaded simulator always starts initialized: draw random states from
    # the reset bounds (applying randomization), then let any explicit
    # initial conditions overwrite the draw.
    sim.reset()
    _apply_initial_conditions(sim, config.get("initial_conditions") or {})

    return sim


def create_simulator_from_yaml(path: str, *, device: str = None) -> SpacecraftSimulator:
    """Load a simulator from a YAML file. device, if given, overrides
    simulation.device, so a scenario written for CUDA also runs on CPU."""
    config = _load_yaml(path)
    if device is not None:
        config["simulation"] = {**(config.get("simulation") or {}), "device": device}
    return create_simulator_from_config(config)

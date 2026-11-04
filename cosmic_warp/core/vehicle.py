"""
Vehicle configuration and properties.
"""
import numpy as np
import warp as wp

from cosmic_warp.core import validation
from cosmic_warp.core.entities import expand_entity_ids

@wp.struct
class VehicleConfig:
    """Physical properties of the spacecraft."""
    # Mass properties
    mass_dry: wp.float64          # Dry mass (without propellant) [kg]
    inertia: wp.mat33d            # Inertia tensor [kg*m^2]
    inertia_inv: wp.mat33d        # Inverse inertia tensor

    # Geometry
    area_drag: wp.float64         # Cross-sectional area for drag [m^2]
    cd: wp.float64                # Drag coefficient

    # Magnetic properties
    magnetic_dipole: wp.vec3d     # Residual magnetic dipole moment in body frame [A*m^2]


@wp.kernel
def _set_config_kernel(
    configs: wp.array(dtype=VehicleConfig),
    entity_ids: wp.array(dtype=wp.int32),
    set_mass_dry: wp.int32,
    mass_dry: wp.float64,
    set_inertia: wp.int32,
    inertia: wp.mat33d,
    inertia_inv: wp.mat33d,
    set_area_drag: wp.int32,
    area_drag: wp.float64,
    set_cd: wp.int32,
    cd: wp.float64,
    set_magnetic_dipole: wp.int32,
    magnetic_dipole: wp.vec3d
) -> None:
    """Write the flagged fields of the vehicle config of the given entities."""
    tid = wp.tid()
    entity_id = entity_ids[tid]
    if set_mass_dry != 0:
        configs[entity_id].mass_dry = mass_dry
    if set_inertia != 0:
        configs[entity_id].inertia = inertia
        configs[entity_id].inertia_inv = inertia_inv
    if set_area_drag != 0:
        configs[entity_id].area_drag = area_drag
    if set_cd != 0:
        configs[entity_id].cd = cd
    if set_magnetic_dipole != 0:
        configs[entity_id].magnetic_dipole = magnetic_dipole


@wp.kernel
def _set_config_per_entity_kernel(
    configs: wp.array(dtype=VehicleConfig),
    mass_dry: wp.array(dtype=wp.float64),
    inertia: wp.array(dtype=wp.mat33d),
    area_drag: wp.array(dtype=wp.float64),
    cd: wp.array(dtype=wp.float64),
    magnetic_dipole: wp.array(dtype=wp.vec3d)
) -> None:
    """Kernel to set vehicle config per entity (empty arrays leave fields unchanged)."""
    tid = wp.tid()
    if mass_dry.shape[0] > 0:
        configs[tid].mass_dry = mass_dry[tid]
    if inertia.shape[0] > 0:
        configs[tid].inertia = inertia[tid]
        configs[tid].inertia_inv = wp.inverse(inertia[tid])
    if area_drag.shape[0] > 0:
        configs[tid].area_drag = area_drag[tid]
    if cd.shape[0] > 0:
        configs[tid].cd = cd[tid]
    if magnetic_dipole.shape[0] > 0:
        configs[tid].magnetic_dipole = magnetic_dipole[tid]


class VehicleManager:
    """Manages vehicle configurations for parallel entities (envs * bodies)."""

    def __init__(self, num_envs: int, device: str = "cuda", num_bodies: int = 1,
                 states: wp.array = None):
        """
        Args:
            num_envs: Number of parallel environments
            device: Compute device
            num_bodies: Number of bodies per environment
            states: The simulator's State array, so a new mass_dry can be
                checked against the current masses (None: no check)
        """
        self.num_envs = num_envs
        self.num_bodies = num_bodies
        self.num_entities = num_envs * num_bodies
        self.device = device
        self._states = states
        self.configs = wp.zeros(self.num_entities, dtype=VehicleConfig, device=device)
        # Host mirrors of the configured dry masses and inertia tensors, so
        # configuration checks and the randomizer's nominal values need no
        # device read. Randomized values are not mirrored. inertia_version
        # changes whenever the configured inertia does.
        self.mass_dry_host = np.zeros(self.num_entities)
        self.inertia_host = np.zeros((self.num_entities, 3, 3))
        self.inertia_version = 0
        # Default: a 1 kg cube (inertia mass/6 per axis), no magnetic dipole
        self.set_config(mass_dry=1.0, inertia=np.eye(3) / 6.0, area_drag=1.0,
                        cd=2.2, magnetic_dipole=[0.0, 0.0, 0.0])

    def set_config(
        self,
        *,
        mass_dry: float = None,
        inertia: list = None,
        area_drag: float = None,
        cd: float = None,
        magnetic_dipole: list = None,
        env_ids: list = None,
        body_ids: list = None,
        entity_ids: list = None
    ) -> None:
        """
        Set vehicle configuration for specific or all entities.

        Only the fields given are checked and written; None leaves a field
        unchanged for the selected entities. A new simulator starts with a
        1 kg cube: mass_dry 1, inertia diag(1/6), area_drag 1, cd 2.2, no
        magnetic dipole.

        Args:
            mass_dry: Dry mass (without propellant) [kg], > 0 and at most the
                current mass of every selected entity that has been reset
            inertia: Inertia tensor [kg*m^2] as 3x3 list/tuple; must be
                     symmetric, positive definite and satisfy the triangle
                     inequality of a rigid body
            area_drag: Cross-sectional area for drag [m^2], >= 0
            cd: Drag coefficient, >= 0
            magnetic_dipole: Residual magnetic dipole moment in body frame [A*m^2]
                             as [mx, my, mz]
            env_ids: List of environment indices to configure (None = all)
            body_ids: List of body indices to configure (None = all bodies)
            entity_ids: Optional flat entity indices (env_id * num_bodies + body_id)
        """
        if mass_dry is not None:
            mass_dry = validation.scalar("mass_dry", mass_dry, positive=True)
        if area_drag is not None:
            area_drag = validation.scalar("area_drag", area_drag, minimum=0.0)
        if cd is not None:
            cd = validation.scalar("cd", cd, minimum=0.0)
        if inertia is not None:
            inertia = validation.inertia("inertia", inertia)
            inertia_mat = wp.mat33d(*inertia.ravel())
            inertia_inv = wp.mat33d(*np.linalg.inv(inertia).ravel())
        else:
            inertia_mat = inertia_inv = wp.mat33d()
        if magnetic_dipole is not None:
            mag_dipole_vec = wp.vec3d(*validation.vec3("magnetic_dipole", magnetic_dipole))
        else:
            mag_dipole_vec = wp.vec3d()

        entity_ids = expand_entity_ids(self.num_envs, self.num_bodies, env_ids, body_ids, entity_ids)
        if entity_ids is None:
            entity_ids = list(range(self.num_entities))
        if len(entity_ids) == 0:
            return

        if mass_dry is not None:
            self._check_state_masses(np.full(len(entity_ids), mass_dry), entity_ids)
            self.mass_dry_host[entity_ids] = mass_dry
        if inertia is not None:
            self.inertia_host[entity_ids] = inertia
            self.inertia_version += 1

        wp.launch(
            kernel=_set_config_kernel,
            dim=len(entity_ids),
            inputs=[
                self.configs,
                wp.array(entity_ids, dtype=wp.int32, device=self.device),
                int(mass_dry is not None),
                float(mass_dry or 0.0),
                int(inertia is not None),
                inertia_mat,
                inertia_inv,
                int(area_drag is not None),
                float(area_drag or 0.0),
                int(cd is not None),
                float(cd or 0.0),
                int(magnetic_dipole is not None),
                mag_dipole_vec
            ],
            device=self.device
        )

    def set_config_per_entity(
        self,
        *,
        mass_dry: "np.ndarray | wp.array | None" = None,
        inertia: "np.ndarray | wp.array | None" = None,
        area_drag: "np.ndarray | wp.array | None" = None,
        cd: "np.ndarray | wp.array | None" = None,
        magnetic_dipole: "np.ndarray | wp.array | None" = None
    ) -> None:
        """
        Set vehicle configuration per entity from batched arrays (e.g. for
        Monte Carlo parameter dispersions).

        Each argument holds one value per entity, num_envs * num_bodies of
        them: a NumPy array or nested list of numbers, or a Warp array of the
        field's dtype on the simulator's device (used as-is; anything else is
        copied to the device). None leaves that field unchanged. The inverse
        inertia tensor is computed in-kernel from `inertia`. Values are
        checked as in set_config (Warp arrays are copied to the host for the
        check) before anything is written.

        Args:
            mass_dry: Dry masses [kg], shape (num_entities,); Warp: float64
            inertia: Inertia tensors [kg*m^2], shape (num_entities, 3, 3);
                Warp: mat33d of shape (num_entities,)
            area_drag: Drag areas [m^2], shape (num_entities,); Warp: float64
            cd: Drag coefficients, shape (num_entities,); Warp: float64
            magnetic_dipole: Dipole moments [A*m^2], shape (num_entities, 3);
                Warp: vec3d of shape (num_entities,)
        """
        n = self.num_entities
        fields = {   # name: (values, Warp dtype, host row shape, value check)
            "mass_dry": (mass_dry, wp.float64, (), _at_least("mass_dry", 0.0, positive=True)),
            "inertia": (inertia, wp.mat33d, (3, 3),
                        lambda h: validation.inertia("inertia", h)),
            "area_drag": (area_drag, wp.float64, (), _at_least("area_drag", 0.0)),
            "cd": (cd, wp.float64, (), _at_least("cd", 0.0)),
            "magnetic_dipole": (magnetic_dipole, wp.vec3d, (3,), _finite("magnetic_dipole")),
        }
        device = self.configs.device
        host, arrays = {}, {}
        for name, (values, dtype, row, check) in fields.items():
            if values is None:
                arrays[name] = wp.empty(0, dtype=dtype, device=device)
                continue
            if isinstance(values, wp.array):
                if values.dtype != dtype or values.shape != (n,):
                    raise ValueError(
                        f"{name} must be a Warp array of {dtype.__name__} with shape "
                        f"({n},), got {values.dtype.__name__} {values.shape}")
                if values.device != device:
                    raise ValueError(f"{name} is on {values.device}, the simulator on {device}")
                host[name] = values.numpy().astype(np.float64).reshape((n, *row))
                arrays[name] = values
            else:
                host[name] = validation.numeric_array(name, values, (n, *row))
                arrays[name] = None   # uploaded once every field has passed
            check(host[name])
        if "mass_dry" in host:
            self._check_state_masses(host["mass_dry"], np.arange(n))
        for name, array in arrays.items():
            if array is None:
                arrays[name] = wp.array(host[name], dtype=fields[name][1], device=device)

        wp.launch(
            kernel=_set_config_per_entity_kernel,
            dim=n,
            inputs=[self.configs, *arrays.values()],
            device=self.device
        )
        if "mass_dry" in host:
            self.mass_dry_host[:] = host["mass_dry"]
        if "inertia" in host:
            self.inertia_host[:] = host["inertia"]
            self.inertia_version += 1

    def _check_state_masses(self, mass_dry: np.ndarray, entity_ids) -> None:
        """Reject a dry mass above the current mass of an entity: the state
        would hold less than dry mass (negative propellant), which the step
        does not model. Entities never reset (mass 0) are not checked: every
        reset path floors the mass at the dry mass. Configuration time only
        (reads the state to the host)."""
        if self._states is None:
            return
        ids = np.asarray(entity_ids)
        mass = self._states.numpy()["mass"][ids]
        bad = (mass > 0.0) & (mass < mass_dry)
        if bad.any():
            k = int(np.flatnonzero(bad)[0])
            raise ValueError(
                f"mass_dry {mass_dry[k]} exceeds the current mass {mass[k]} of entity "
                f"{int(ids[k])}: configure mass_dry (and mass_bounds) before "
                f"sim.reset(), or first reset the entity to a mass of at least "
                f"mass_dry"
            )


def _finite(name: str):
    """A host-array check: every value finite."""
    def check(host):
        if not np.all(np.isfinite(host)):
            raise ValueError(f"{name} must be finite")
    return check


def _at_least(name: str, minimum: float, positive: bool = False):
    """A host-array check: every value finite and >= minimum (> with positive)."""
    def check(host):
        _finite(name)(host)
        low = host <= minimum if positive else host < minimum
        if np.any(low):
            bound = f"> {minimum}" if positive else f">= {minimum}"
            raise ValueError(f"{name} must be {bound}, got {host[low][0]}")
    return check

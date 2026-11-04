"""
Simulation flags for enabling/disabling physics modules and features.
"""
import warp as wp


@wp.struct
class SimulationFlags:
    """Flags to control which physics modules are active in the simulation."""

    # Physics modules
    gravity: wp.bool            # Enable gravitational forces
    j2_perturbation: wp.bool    # Enable J2 perturbation (requires gravity=True)
    gravity_gradient: wp.bool   # Enable gravity gradient torque
    magnetic_torque: wp.bool    # Enable magnetic torque
    drag: wp.bool               # Enable atmospheric drag force

    # Termination conditions
    check_bounds: wp.bool         # Check position bounds
    check_planet_impact: wp.bool  # Check for impact with the central body


@wp.kernel
def _init_flags_kernel(flags: wp.array(dtype=SimulationFlags),
                        gravity: bool,
                        j2_perturbation: bool,
                        gravity_gradient: bool,
                        magnetic_torque: bool,
                        drag: bool,
                        check_bounds: bool,
                        check_planet_impact: bool):
    # always use index 0 for the shared flags
    flags[0].gravity = gravity
    flags[0].j2_perturbation = j2_perturbation
    flags[0].gravity_gradient = gravity_gradient
    flags[0].magnetic_torque = magnetic_torque
    flags[0].drag = drag
    flags[0].check_bounds = check_bounds
    flags[0].check_planet_impact = check_planet_impact

def create_flags(
    gravity: bool,
    j2_perturbation: bool,
    gravity_gradient: bool,
    magnetic_torque: bool,
    drag: bool,
    check_bounds: bool,
    check_planet_impact: bool,
    device: str = "cuda"
) -> wp.array:
    """
    Create simulation flags array for multiple environments.

    Args:
        gravity: Enable gravitational forces
        j2_perturbation: Enable J2 perturbation (requires gravity=True)
        gravity_gradient: Enable gravity gradient torque
        magnetic_torque: Enable magnetic torque (requires vehicle.magnetic_dipole set)
        drag: Enable atmospheric drag force (uses vehicle.area_drag and vehicle.cd)
        check_bounds: Check position bounds
        check_planet_impact: Check for impact with the central body
        device: Device to allocate on ("cuda" or "cpu")

    Returns:
        WARP array with SimulationFlags struct
    """
    # Allocate on device and initialize via a small kernel to ensure values
    # are written on the device-side struct (avoids host->device view issues).
    flags = wp.zeros(1, dtype=SimulationFlags, device=device)

    wp.launch(
        kernel=_init_flags_kernel,
        dim=1,
        inputs=[
            flags,
            bool(gravity),
            bool(j2_perturbation),
            bool(gravity_gradient),
            bool(magnetic_torque),
            bool(drag),
            bool(check_bounds),
            bool(check_planet_impact),
        ],
        device=device,
    )

    return flags

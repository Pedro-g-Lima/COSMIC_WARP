import warp as wp

@wp.struct
class State:
    """Dynamic state variables for spacecraft states (float64)."""
    # Translational state (inertial frame)
    pos: wp.vec3d       # position [m]
    mom: wp.vec3d       # linear momentum [kg*m/s]
    mass: wp.float64    # current total mass [kg] (decreases as propellant is consumed)

    # Rotational state (body frame)
    quat: wp.quatd      # attitude quaternion
    ang_mom: wp.vec3d   # angular momentum [kg*m^2/s]


def allocate_state(num_envs, num_bodies: int = 1, device: str = "cuda"):
    """
    Allocate state storage for all environments and bodies.

    States are stored in a flat array of length num_envs * num_bodies.
    Use entity_id = env_id * num_bodies + body_id for indexing.
    """
    return wp.zeros(num_envs * num_bodies, dtype=State, device=device)

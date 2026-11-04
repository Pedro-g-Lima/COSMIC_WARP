"""Shared entity indexing helpers for managers operating on flat (env, body) arrays."""
import numpy as np
import warp as wp


@wp.kernel
def init_mask_kernel(mask: wp.array(dtype=wp.bool), entity_ids: wp.array(dtype=wp.int32)) -> None:
    """Initialize mask for specific entities."""
    i = wp.tid()
    entity_id = entity_ids[i]
    mask[entity_id] = True


def expand_entity_ids(
    num_envs: int,
    num_bodies: int,
    env_ids: list | None = None,
    body_ids: list | None = None,
    entity_ids: list | None = None
) -> list | None:
    """
    Expand (env_ids, body_ids) selections into flat entity indices.

    entity_id = env_id * num_bodies + body_id

    Returns entity_ids directly if given (it cannot be combined with env_ids
    or body_ids). Returns None when no selection is made (callers treat None
    as "all entities"). Every index must be an integer within range.
    """
    if entity_ids is not None:
        if env_ids is not None or body_ids is not None:
            raise ValueError("entity_ids cannot be combined with env_ids or body_ids")
        return check_ids("entity_ids", entity_ids, num_envs * num_bodies)
    if env_ids is not None:
        env_ids = check_ids("env_ids", env_ids, num_envs)
    if body_ids is not None:
        body_ids = check_ids("body_ids", body_ids, num_bodies)
    if env_ids is None and body_ids is not None:
        env_ids = list(range(num_envs))
    if env_ids is None:
        return None
    if body_ids is None:
        body_ids = list(range(num_bodies))
    return [env_id * num_bodies + body_id
            for env_id in env_ids
            for body_id in body_ids]


def check_ids(name: str, ids, count: int) -> list:
    """
    Return ids as a list of ints, each in [0, count). ids is a list, tuple or
    range of integers, or a 1-D integer NumPy array. Booleans (a mask is not
    a selection), floats, strings, scalars and Warp arrays are rejected.
    """
    if isinstance(ids, np.ndarray):
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError(
                f"{name} must be a 1-D integer array, got shape {ids.shape} {ids.dtype}")
        ids = ids.tolist()
    elif not isinstance(ids, (list, tuple, range)):
        raise ValueError(
            f"{name} must be a list of integers, got {type(ids).__name__} {ids!r}")
    out = []
    for i in ids:
        if isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)):
            raise ValueError(f"{name} must contain integers, got {i!r}")
        i = int(i)
        if not 0 <= i < count:
            raise ValueError(f"{name} entry {i} out of range [0, {count})")
        out.append(i)
    return out

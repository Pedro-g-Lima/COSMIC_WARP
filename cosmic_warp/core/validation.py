"""
Host-side input validation shared by the configuration APIs.

Every check runs at configuration time on host values and raises ValueError
with the offending name and value. Nothing here is called from the step loop.
"""
import math
import numbers

import numpy as np


def scalar(name: str, value, minimum: float = None, positive: bool = False,
           allow_inf: bool = False) -> float:
    """Return value as a finite float (or +inf with allow_inf, for limits
    that can be disabled), optionally >= minimum or > 0."""
    if isinstance(value, (str, bytes, bool)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(x) and not (allow_inf and x == math.inf):
        raise ValueError(f"{name} must be finite, got {x}")
    if positive and x <= 0.0:
        raise ValueError(f"{name} must be > 0, got {x}")
    if minimum is not None and x < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {x}")
    return x


def polar_angle(name: str, value) -> float:
    """Return value as a finite angle in [0, pi] [rad]: an inclination or a
    cone half-angle, both defined only on that range."""
    x = scalar(name, value)
    if not 0.0 <= x <= math.pi:
        raise ValueError(f"{name} must be in [0, pi] rad, got {x}")
    return x


def positive_int(name: str, value, maximum: int = None) -> int:
    """Return value as an int >= 1 (step counts, horizons, episode limits),
    optionally <= maximum."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {value}")
    return int(value)


def describe(value) -> str:
    """Type, shape, dtype and device of an array argument, for error messages."""
    dtype = getattr(value, "dtype", "")
    parts = [type(value).__name__, str(getattr(value, "shape", "")),
             getattr(dtype, "__name__", str(dtype)), str(getattr(value, "device", ""))]
    return " ".join(p for p in parts if p)


def flag(name: str, value) -> bool:
    """Return value as a bool; only True/False are accepted ("false" is truthy)."""
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be True or False, got {value!r}")
    return bool(value)


def seed(name: str, value) -> int:
    """Return value as an RNG seed, the 64-bit Philox key: an int in [0, 2**64)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) \
            or not 0 <= value < 2**64:
        raise ValueError(f"{name} must be an integer in [0, 2**64), got {value!r}")
    return int(value)


def sequence(name: str, value) -> list:
    """Return value as a list: a list, tuple or array, not a string or scalar."""
    if (isinstance(value, (str, bytes, dict)) or not hasattr(value, "__len__")
            or (isinstance(value, np.ndarray) and value.ndim == 0)):
        raise ValueError(f"{name} must be a list, got {value!r}")
    return list(value)


def vec3(name: str, value) -> tuple:
    """Return value as a tuple of three finite floats."""
    try:
        if isinstance(value, (str, bytes)) or any(isinstance(c, (str, bytes, bool)) for c in value):
            raise ValueError
        v = [float(c) for c in value]
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a sequence of 3 numbers, got {value!r}") from None
    if len(v) != 3:
        raise ValueError(f"{name} must have 3 components, got {len(v)}")
    if not all(math.isfinite(c) for c in v):
        raise ValueError(f"{name} must be finite, got {tuple(v)}")
    return tuple(v)


def array(name: str, values, width: int = None) -> np.ndarray:
    """
    Return a list of numbers ([N]) or of width-component rows ([N, width]) as
    a finite float64 array. Strings and booleans are rejected, not converted,
    and rows of another width are rejected, not reshaped.
    """
    arr = np.asarray(values, dtype=object)
    if width is None:
        if arr.ndim != 1:
            raise ValueError(f"{name} must be a list of numbers, got shape {arr.shape}")
    elif arr.ndim != 2 or arr.shape[1] != width:
        raise ValueError(f"{name} must be a list of {width}-component rows, got shape {arr.shape}")
    return np.array([scalar(name, v) for v in arr.flat], dtype=np.float64).reshape(arr.shape)


def numeric_array(name: str, values, shape: tuple) -> np.ndarray:
    """
    Return values as a float64 array of the given shape. A NumPy array must
    have an integer or float dtype; any other sequence is checked element by
    element. Booleans and strings are rejected, not converted. Finiteness is
    left to the caller.
    """
    if isinstance(values, np.ndarray):
        arr = values
        if arr.dtype.kind not in "iuf":
            raise ValueError(f"{name} must hold numbers, got dtype {arr.dtype}")
    else:
        arr = np.asarray(values, dtype=object)
    if arr.shape != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {arr.shape}")
    if arr.dtype == object:
        for v in arr.flat:
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real):
                raise ValueError(f"{name} must hold numbers, got {v!r}")
    return arr.astype(np.float64)


def inertia(name: str, value) -> np.ndarray:
    """
    Check one (3, 3) or a batch (N, 3, 3) of inertia tensors: finite,
    symmetric, positive definite, and satisfying the triangle inequality on
    the principal moments (a physical rigid body). Returns a float64 array.
    """
    try:
        I = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a 3x3 matrix of numbers, got {value!r}") from None
    if I.shape[-2:] != (3, 3):
        raise ValueError(f"{name} must be 3x3, got shape {I.shape}")
    if not np.all(np.isfinite(I)):
        raise ValueError(f"{name} must be finite")
    batch = I.reshape(-1, 3, 3)
    scale = np.max(np.abs(batch), axis=(1, 2))
    asym = np.max(np.abs(batch - batch.transpose(0, 2, 1)), axis=(1, 2))
    bad = np.flatnonzero(asym > 1e-9 * scale)
    if bad.size:
        raise ValueError(f"{name} must be symmetric (entity {bad[0]}: {batch[bad[0]].tolist()})")
    moments = np.linalg.eigvalsh(batch)   # ascending principal moments
    bad = np.flatnonzero(moments[:, 0] <= 0.0)
    if bad.size:
        raise ValueError(
            f"{name} must be positive definite (entity {bad[0]}: principal moments "
            f"{moments[bad[0]].tolist()})"
        )
    tol = 1e-9 * moments[:, 2]
    bad = np.flatnonzero(moments[:, 0] + moments[:, 1] < moments[:, 2] - tol)
    if bad.size:
        raise ValueError(
            f"{name} violates the triangle inequality I1 + I2 >= I3 (entity {bad[0]}: "
            f"principal moments {moments[bad[0]].tolist()}); no rigid body has it"
        )
    return I

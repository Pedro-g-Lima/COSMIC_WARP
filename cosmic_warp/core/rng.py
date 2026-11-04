"""
Graph-safe, counter-based per-entity random streams.

Every draw is Philox4x32-10 (Salmon et al., SC'11; the generator behind
cuRAND and Random123) of a counter that names it exactly: the entity's draw
count, the entity index and a stream id, under a key that is the seed.
Distinct entities, streams or draws are distinct counters, so no two streams
ever overlap (unlike iterating a 32-bit state per entity, whose streams share
cycles). A uniform draw is float64 with 53 random bits.

A stream's state is one ``RNG_STATE`` row, ``(seed, draws so far)``, in
device memory. Kernels advance it in place, so they are CUDA-graph-safe:
replaying a graph gives fresh draws, and the same seed replays the same
ones. ``tests/test_rng.py`` checks the generator against Random123's
published known-answer vectors.
"""
import warp as wp

# One entity's stream state: (key = seed, number of draws so far).
RNG_STATE = wp.vec2ul

# Stream ids, the counter's last word: the reset sampler and the domain
# randomizer never share a draw, even under equal seeds. 0 is unused.
STREAM_RESET = 1
STREAM_RANDOMIZE = 2

_M0 = wp.constant(wp.uint32(0xD2511F53))
_M1 = wp.constant(wp.uint32(0xCD9E8D57))
_W0 = wp.constant(wp.uint32(0x9E3779B9))
_W1 = wp.constant(wp.uint32(0xBB67AE85))
_LOW = wp.constant(wp.uint64(0xFFFFFFFF))
_SHIFT = wp.constant(wp.uint64(32))
_TO_UNIT = wp.constant(wp.float64(2.0**-53))


@wp.func
def philox4x32(c0: wp.uint32, c1: wp.uint32, c2: wp.uint32, c3: wp.uint32,
               k0: wp.uint32, k1: wp.uint32) -> wp.vec4ui:
    """Philox4x32-10 of counter (c0, c1, c2, c3) under key (k0, k1)."""
    for r in range(10):
        if r > 0:
            k0 = k0 + _W0
            k1 = k1 + _W1
        p0 = wp.uint64(_M0) * wp.uint64(c0)
        p1 = wp.uint64(_M1) * wp.uint64(c2)
        hi0 = wp.uint32(p0 >> _SHIFT)
        lo0 = wp.uint32(p0 & _LOW)
        hi1 = wp.uint32(p1 >> _SHIFT)
        lo1 = wp.uint32(p1 & _LOW)
        n0 = hi1 ^ c1 ^ k0
        n2 = hi0 ^ c3 ^ k1
        c0 = n0
        c1 = lo1
        c2 = n2
        c3 = lo0
    return wp.vec4ui(c0, c1, c2, c3)


@wp.func
def draw_uniform(state: RNG_STATE, offset: int, entity: int, stream: int) -> wp.float64:
    """Uniform [0, 1) float64 (53 random bits) of draw ``state[1] + offset``
    of this entity's stream, without advancing it: a kernel loads the state
    once, draws at offsets 0, 1, ... and stores ``state[1] + count`` back."""
    count = state[1] + wp.uint64(offset)
    key = state[0]
    bits = philox4x32(wp.uint32(count & _LOW), wp.uint32(count >> _SHIFT),
                      wp.uint32(entity), wp.uint32(stream),
                      wp.uint32(key & _LOW), wp.uint32(key >> _SHIFT))
    hi = wp.uint64(bits[0] >> wp.uint32(5))     # 27 bits
    lo = wp.uint64(bits[1] >> wp.uint32(6))     # 26 bits
    return wp.float64((hi << wp.uint64(26)) | lo) * _TO_UNIT


@wp.func
def advance(state: RNG_STATE, count: int) -> RNG_STATE:
    """``state`` after ``count`` draws."""
    return RNG_STATE(state[0], state[1] + wp.uint64(count))


@wp.func
def sample_uniform(u: wp.float64, lo: wp.float64, hi: wp.float64) -> wp.float64:
    """Map a uniform [0, 1) draw onto [lo, hi)."""
    return lo + (hi - lo) * u


def seed_states(states: wp.array, seed: int) -> None:
    """Restart every row of ``states`` at draw 0 of the stream keyed by
    ``seed`` (an integer in [0, 2**64), checked by the caller)."""
    states.fill_(RNG_STATE(seed, 0))

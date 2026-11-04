"""
Counter-based random streams (cosmic_warp.core.rng): the generator against
Random123's published known-answer vectors, the float64 draw construction,
stream independence across entities and between the reset and randomization
streams, uniform moments, 64-bit seeds and per-entity draw counts.
"""
import numpy as np
import pytest
import warp as wp

from cosmic_warp.core.rng import (RNG_STATE, STREAM_RANDOMIZE, STREAM_RESET, draw_uniform,
                                  philox4x32, seed_states)
from cosmic_warp.core.reset import RESET_DRAWS
from conftest import get_state

MASK32 = 0xFFFFFFFF


def philox_reference(counter, key):
    """Philox4x32-10 in plain Python, after Salmon et al. (SC'11)."""
    c0, c1, c2, c3 = counter
    k0, k1 = key
    for r in range(10):
        if r > 0:
            k0 = (k0 + 0x9E3779B9) & MASK32
            k1 = (k1 + 0xBB67AE85) & MASK32
        p0 = 0xD2511F53 * c0
        p1 = 0xCD9E8D57 * c2
        c0, c1, c2, c3 = ((p1 >> 32) ^ c1 ^ k0, p1 & MASK32,
                          (p0 >> 32) ^ c3 ^ k1, p0 & MASK32)
    return c0, c1, c2, c3


def uniform_reference(seed, draw, entity, stream):
    """Draw ``draw`` of an entity's stream: 53 bits of the first two words."""
    w = philox_reference((draw & MASK32, draw >> 32, entity, stream), (seed & MASK32, seed >> 32))
    return ((w[0] >> 5) * 2**26 + (w[1] >> 6)) / 2.0**53


@wp.kernel
def _philox(c: wp.array(dtype=wp.vec4ui), key: wp.array(dtype=wp.vec2ui),
            out: wp.array(dtype=wp.vec4ui)):
    i = wp.tid()
    out[i] = philox4x32(c[i][0], c[i][1], c[i][2], c[i][3], key[i][0], key[i][1])


@wp.kernel
def _draws(states: wp.array(dtype=RNG_STATE), stream: int, out: wp.array2d(dtype=wp.float64)):
    i = wp.tid()
    for k in range(out.shape[1]):
        out[i, k] = draw_uniform(states[i], k, i, stream)


def draws(device, seed, n, k, stream=STREAM_RESET, count=0):
    states = wp.empty(n, dtype=RNG_STATE, device=device)
    seed_states(states, seed)
    if count:
        states.fill_(RNG_STATE(seed, count))
    out = wp.empty((n, k), dtype=wp.float64, device=device)
    wp.launch(_draws, dim=n, inputs=[states, stream, out], device=device)
    return out.numpy()


@pytest.mark.parametrize("on", ["cpu", "test-device"])
def test_philox_matches_the_random123_known_answers(device, on):
    # Random123's kat_vectors, philox4x32 R=10: zeros, all ones, and the
    # digits-of-pi counter and key
    counters = [[0, 0, 0, 0], [MASK32] * 4, [0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344]]
    keys = [[0, 0], [MASK32] * 2, [0xA4093822, 0x299F31D0]]
    expected = [[0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8],
                [0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD],
                [0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1]]
    on = "cpu" if on == "cpu" else device
    out = wp.zeros(3, dtype=wp.vec4ui, device=on)
    wp.launch(_philox, dim=3, device=on, inputs=[
        wp.array(np.array(counters, np.uint32), dtype=wp.vec4ui, device=on),
        wp.array(np.array(keys, np.uint32), dtype=wp.vec2ui, device=on), out])
    np.testing.assert_array_equal(out.numpy(), np.array(expected, np.uint32))
    # The plain-Python reference the other tests use agrees
    assert [list(philox_reference(c, k)) for c, k in zip(counters, keys)] == expected


def test_draws_are_53_bit_uniforms_of_the_counter(device):
    """Draw = Philox(counter (draw count, entity, stream), key seed), with
    64-bit seeds and counts: high words of both matter."""
    seed, count = 3 + 2**40, 2**32 - 2   # the count carries into its high word
    got = draws(device, seed, 3, 4, STREAM_RANDOMIZE, count=count)
    for i in range(3):
        for k in range(4):
            assert got[i, k] == uniform_reference(seed, count + k, i, STREAM_RANDOMIZE)
    x = draws(device, 5, 64, 64).ravel()
    assert np.all((x >= 0.0) & (x < 1.0))
    # 53 random bits: every draw is a multiple of 2^-53, and the low bits
    # below float32 resolution are populated
    np.testing.assert_array_equal(x * 2.0**53, np.round(x * 2.0**53))
    assert np.mean((x * 2.0**53) % 2**29 != 0) > 0.99


def test_entity_streams_never_overlap(device):
    """The defect this generator replaced: 32-bit per-entity states on shared
    cycles let one entity's stream replay another's, shifted. Distinct
    counters cannot: over 4096 entities × 64 draws no value repeats."""
    n, k = 4096, 64
    x = draws(device, 42, n, k)
    assert np.unique(x).size == n * k
    # Neighbouring entities are uncorrelated: |r| < 4/√N
    r = np.corrcoef(x[:-1].ravel(), x[1:].ravel())[0, 1]
    assert abs(r) < 4 / np.sqrt(x[:-1].size)


def test_reset_and_randomization_streams_are_independent(device):
    """Equal seeds, entities and counts on the two streams share no draw."""
    reset = draws(device, 9, 1024, 16, STREAM_RESET)
    randomize = draws(device, 9, 1024, 16, STREAM_RANDOMIZE)
    assert np.intersect1d(reset, randomize).size == 0
    r = np.corrcoef(reset.ravel(), randomize.ravel())[0, 1]
    assert abs(r) < 4 / np.sqrt(reset.size)


def test_uniform_moments(device):
    x = draws(device, 3, 1024, 256).ravel()
    assert abs(x.mean() - 0.5) < 5 * np.sqrt(1 / 12 / x.size)
    assert x.var() == pytest.approx(1 / 12, rel=0.01)


def test_reset_draws_follow_the_counter_layout(make_sim):
    """A reset of entity i draws its position from slots 0-2 of its own
    reset stream: the kernel and the reference agree bitwise."""
    sim = make_sim(num_envs=4, reset_config={"seed": 2**63 + 5})
    sim.reset_manager.set_bounds(pos_bounds=((0.0, 0.0, 0.0), (1.0, 2.0, 4.0)))
    before = sim.reset_manager.rng_state.numpy()[:, 1].copy()
    sim.reset()
    pos = get_state(sim)["pos"]
    for i in range(4):
        expected = [uniform_reference(2**63 + 5, int(before[i]) + k, i, STREAM_RESET) * w
                    for k, w in enumerate((1.0, 2.0, 4.0))]
        np.testing.assert_array_equal(pos[i], expected)


def test_seeds_are_64_bit_keys(make_sim):
    """3 and 3 + 2**32 are different keys; the largest seed is accepted."""
    pos = []
    for seed in (3, 3 + 2**32, 2**64 - 1, np.uint64(2**64 - 1)):
        sim = make_sim(num_envs=8)
        sim.reset_manager.reseed(seed)
        sim.reset()
        pos.append(get_state(sim)["pos"])
    assert not np.array_equal(pos[0], pos[1])
    np.testing.assert_array_equal(pos[2], pos[3])
    np.testing.assert_array_equal(sim.reset_manager.rng_state.numpy()[:, 0], 2**64 - 1)


def test_draw_counts_advance_per_reset_entity(make_sim):
    """Each reset advances only the reset entities' counts, by a fixed number
    of draws on each stream, so an entity's draws do not depend on when the
    others reset."""
    sim = make_sim(num_envs=3, num_bodies=2)
    sim.randomizer.set_bounds(propellant=(0.1, 0.2))
    reset0 = sim.reset_manager.rng_state.numpy()[:, 1].copy()
    rand0 = sim.randomizer.rng_state.numpy()[:, 1].copy()
    sim.reset(env_ids=[1])
    selected = np.array([0, 0, 1, 1, 0, 0])
    np.testing.assert_array_equal(sim.reset_manager.rng_state.numpy()[:, 1],
                                  reset0 + RESET_DRAWS * selected)
    np.testing.assert_array_equal(sim.randomizer.rng_state.numpy()[:, 1],
                                  rand0 + 3 * selected)
    sim.reset()
    np.testing.assert_array_equal(sim.reset_manager.rng_state.numpy()[:, 1],
                                  reset0 + RESET_DRAWS * (selected + 1))

# Execution

CUDA graphs, termination, resets, random streams, input checks and the YAML
loader: the machinery around the physics must not change the answer. Rows
marked *CUDA only* skip in CI and are verified on a local GPU; see
[CI and local GPU checks](index.md#ci-and-local-gpu-checks).

On CUDA, `simulate()` captures one step into a CUDA graph and replays it; on
CPU it runs the same kernels eagerly. Every test that calls `simulate()`
therefore exercises graph replay when run with `--device cuda`.

## CUDA graphs

`tests/test_termination.py`, `tests/test_hot_loop.py`, `tests/test_pipeline.py`

| Claim | Referent | Tolerance |
| --- | --- | --- |
| Multi-body `simulate()` with an environment terminating on the first step, 3 steps, all integrators, with and without auto-reset (*CUDA only*) | the same steps run eagerly, with `reset_terminated()` for auto-reset | positions and state 1e-12 relative; wheel states, commands, termination status, episode counters bitwise; same `step_count` |
| `simulate()` captures with every per-step feature on: J2, gravity gradient, magnetic torque, drag, bounds and planet-impact checks, an orbit-mode body, randomization, thrusters, wheels and auto-reset; 20 steps, all integrators (*CUDA only*) | capture raises on any host synchronization or device-to-host read inside the step | captures, finite state |
| Episode counters advance inside `simulate()` | the step count | exact |

`VecEnv.step()` replay is covered on the [learning page](learning.md#vectorized-environment).

## Termination

`tests/test_termination.py`, `tests/test_pipeline.py`, `tests/test_api.py`:
termination is per environment: when one body violates a limit, every body
of that environment finishes the step and then freezes.

| Claim | Referent | Tolerance |
| --- | --- | --- |
| With the default limits, a 500 kg spacecraft with 50 kg·m²/s of angular momentum in LEO runs (the momentum limits are off until set); a NaN limit is rejected | specified defaults | exact |
| A terminating step advances every body of the environment, offender first or last, through `step()`, `simulate()` and `simulate_monte_carlo()`, all integrators | the full step v·dt for every body | 1e-7 relative |
| Terminated environments stay frozen; running ones continue; episode counters count attempted steps | state and wheel states before the extra steps | bitwise |
| Partial and full reset afterwards | counters and `step_count` | exact |
| Two bodies violating different limits at once, all five reason pairs, both body orders, all integrators | the smallest reason code, independent of body order, unchanged on the next step | exact |
| Auto-reset stores the full terminating step in the saved trajectory, then resets the environment's state, wheels and commands | the step v·dt; zeros | 1e-7 relative; exactly zero |
| A NaN in the state terminates its environment with reason 6, with or without bounds checks, all integrators | NaN passes every comparison, so it has its own check | exact |
| `mass_limit` equal to the dry mass fires when propellant runs out | mass is clamped at dry mass, so the check is inclusive | reason 4 |
| … on the burnout step itself: 4096 random propellant loads, two thruster settings, all integrators | the step-end mass is exactly dry on that step (two roundings per symplectic step left a quarter of entities one ulp above dry, never terminating) | step and reason exact; mass bitwise |
| A NaN or Inf thrust or wheel-torque command terminates its environment with reason 6 before the step, all integrators | the clamps would turn it into a finite actuation | exact; state and wheel momenta unchanged bitwise |
| `check_planet_impact` with the default `planet_radius`, from Python and YAML | Earth's equatorial radius, `RE_EARTH`: 1 m below it is an impact | reason 5 |
| Termination reasons reported per entity and per (env, body) | the violated limit | exact |
| `reset_to_conditions()` clears termination like `reset()` | specified semantics | exact |
| Resetting only some bodies of a terminated environment, through `reset()` or `reset_to_conditions()` | the siblings would re-terminate it on the next step | raises |
| A differentiable rollout of a multi-body simulator leaves the simulator's state, wheels, status, counters and `step_count` untouched, and refuses termination checks | specified semantics | bitwise; raises |

## Resets

`tests/test_pipeline.py`, `tests/test_reset.py`

| Claim | Referent | Tolerance |
| --- | --- | --- |
| `reset(body_ids=…)` and `reset(env_ids=…)` reset only the selection; a second partial reset does not reuse the first selection | untouched entities keep their state | bitwise |
| `reset_to_conditions(env_ids=…)` resets only the selection | untouched entities keep their state | bitwise |
| Per-entity episode counters through partial resets, `reset_to_conditions` and full resets | hand-computed counts | exact |
| Without `masses`, `reset_to_conditions` writes the `mass_bounds` centre | the bounds | bitwise |
| The default reset velocity is circular LEO speed for any mass (100–150 kg) | √(μ/r) at 400 km; momentum = m v | 2 m/s; `reset_to_conditions` default 1e-12 relative |
| Reset mass bounds (or their default centre) below the dry mass, checked on every reset | rejected, also after a later per-env vehicle change | raises |
| Bounds or propellant randomization that keep the mass above dry | accepted | no warning |
| A rejected `set_bounds` or `set_orbit_bounds` call | leaves every body's bounds and sampling mode unchanged | exact |
| Setting momentum bounds after velocity bounds switches back to momentum sampling | the bound | 1e-7 relative |
| `reset_to_conditions` with q = (0, 0, 0, 2) or (0, 0, 0, ½), 1 N of thrust on 2 kg, all integrators | stored normalized; step-1 Δv = F dt / m | bitwise; 1e-15 m/s |
| `reset_terminated()`, `simulate(auto_reset_terminated=True)` and the VecEnv after the mass bounds were lowered below the dry mass; `reset_manager.reset_to_conditions` below it | the mass floored at the dry mass; momentum = that mass × the sampled velocity | bitwise; 1e-15 relative |

## Random streams and determinism

`tests/test_rng.py`: the generator. `tests/test_reset.py`, `tests/test_api.py`:
reset sampling on the device, with 4096 environments for the statistics.

| Claim | Referent | Tolerance |
| --- | --- | --- |
| Philox4x32-10, CPU and CUDA | Random123's known-answer vectors | bitwise |
| A draw is the 53-bit float64 uniform of counter (draw count, entity, stream) under the 64-bit seed, including high words of seed and count | a plain-Python Philox | bitwise |
| 4096 entities × 64 draws; neighbouring entities | no value repeats; uncorrelated | none repeats; \|r\| < 4/√N |
| The reset and randomization streams under one seed | no shared draw; uncorrelated | none shared; \|r\| < 4/√N |
| Uniform moments, 262,144 draws | mean ½, variance 1/12 | 5σ; 1 % |
| Seeds 3 and 3 + 2³²; seed 2⁶⁴ − 1 | different draws; accepted | bitwise |
| A partial reset advances only its entities' draw counts, by a fixed number per reset on each stream | specified semantics | exact |
| No correlation between any two of the ten sampled components (position, momentum, angular momentum, mass) | independent draws | \|r\| < 0.08 (the 4σ band is 0.06) |
| Positions uniform within their bounds; masses inside theirs | U[lo, hi] | inside; normalized mean within 0.02 of 1/2, variance within 0.01 of 1/12 |
| Uniform attitudes | unit quaternions covering both hemispheres | norm 1e-12; q_w > 0 in 40–60 % of draws |
| Attitude inside a cone of 0.2 rad | Haar density, mean angle ¾ of the cone ([[21]](../references.md#ref-21)); a uniform-angle draw would give ½ | inside to 1e-9; fills past 95 %; mean within 0.01 rad |
| `att_max_angle = 0` | exactly the nominal attitude | 1e-12 |
| A per-body attitude override | applies to that body only | 1e-12 |
| Same seed, different seed; repeated resets; `reseed()` | identical draws; different draws; fresh draws; the original draw again | bitwise |
| Two identical 500-step runs | each other | bitwise |

## Input validation

Invalid input raises `ValueError` (or `RuntimeError` for a misuse of state)
with a message naming the problem, before anything changes.

| Claim | Test |
| --- | --- |
| Constructor: non-positive or non-finite `dt`, non-integer counts, non-positive limits, J2 without gravity; J2 follows `gravity` by default | `tests/test_pipeline.py` |
| Constructor: an unknown or non-string integrator name; physics and termination switches that are not `True`/`False` (`"false"` is truthy); unknown `reset_config` keys; an unknown device name | `tests/test_api.py` |
| Rebinding a simulator attribute (`states`, `dt`, the counts, the device, the managers): read-only | `tests/test_api.py` |
| `simulate()` and `simulate_monte_carlo()` with a non-positive or non-integer `num_steps` or a non-boolean flag: rejected before any step, state, `step_count` and episode counters unchanged; `time` returned only alongside a recording | `tests/test_api.py` |
| Warp arrays on another device (`reset_to_conditions`, `set_config_per_entity`, the per-entity command setters): rejected by name before any launch, the simulator stepping on (*needs CUDA*) | `tests/test_api.py` |
| Selections: `entity_ids` combined with env/body ids, out-of-range or negative ids, non-integers; anything but a list, tuple, range or 1-D integer array of integers (booleans, floats, strings, a scalar, a Warp array) — for resets and every config setter | `tests/test_reset.py` |
| Reset bounds: inverted, non-positive mass, wrong component count, non-finite; an `att_max_angle` or `inclination_bounds` outside [0, π]; condition arrays of the wrong length or dtype; explicit `reset_to_conditions` masses that are non-finite or below the (possibly randomized) dry mass, and quaternions that are zero or non-finite, checked only for the selected entities; seeds that are not integers in [0, 2⁶⁴) | `tests/test_reset.py` |
| Orbit bounds: e ≥ 1, negative semi-major axis, conflicting size or anomaly specs, unknown sample mode, orbit-mode bodies without explicit positions and momenta in `reset_to_conditions`; inclinations outside [0, π] (also `elements_to_state` and the YAML orbit specs) | `tests/test_orbit_sampling.py` |
| Vehicle: non-symmetric, non-positive-definite, triangle-inequality-violating or non-finite inertia (broadcast and batched setters); non-finite or non-positive mass; negative C_D or area; a `mass_dry` above a reset entity's current mass (both setters, selected entities only); batched values that are booleans or strings, or of the wrong shape or Warp dtype, rejected before any field is written | `tests/test_config.py`, `tests/test_randomization.py`, `tests/test_reset.py`, `tests/test_set_config.py` |
| Thrusters: more than 8, mismatched lists, non-finite values, wrong component count, negative thrust or Isp, an unnormalizable direction | `tests/test_thrusters.py` |
| Wheels: more than 4, unnormalizable or non-finite axes, non-positive inertia, negative limits or damping, wheel momentum set above capacity, capacity lowered below a stored wheel momentum (selected entities only) | `tests/test_rw.py` |
| Command setters: more than 8 thrust or 4 wheel values, a scalar, non-numbers, non-finite values; per-entity arrays of the wrong shape or dtype. Config setters given a scalar for a thruster layout, wheel axes or randomization bounds, a string inertia or a 3-component nominal attitude: a `ValueError` naming the argument | `tests/test_api.py` |
| Randomization: dry-mass dispersion without propellant, inverted or non-positive ranges; bounds that are not two finite numbers (quoted, boolean, NaN, from Python or YAML); invalid seeds | `tests/test_randomization.py` |

## YAML loader

`tests/test_config.py`, `tests/test_api.py`, `tests/test_randomization.py`,
`tests/test_orbit_sampling.py`: the loader is fail-closed, and the
[configuration guide](../guides/configuration.md) is checked against it.

| Claim | Referent | Tolerance |
| --- | --- | --- |
| Unknown keys, top-level or nested | typos fail loudly | raises, naming the key |
| `1.0e1` and `1e5` read as numbers (YAML 1.2), not strings | the number | exact |
| Quoted numbers and booleans (`"false"` would enable a flag) | rejected | raises |
| Duplicate `per_body` entries; malformed bounds; malformed condition rows (16 rows of 2 are not 8 quaternions) | rejected, not reshaped or last-wins | raises |
| A section that is not a mapping, a `per_body` that is not a list of mappings, a scalar where a list belongs | rejected naming the key, not a `TypeError` | raises |
| Per-entity condition lists (`positions`, `velocities` or `momenta`, `masses`, ...) | the state, one row per entity; momentum = mass × velocity | bitwise |
| Reset and initial masses below a body's dry mass, including the 1 kg default bounds and the largest randomized dry mass | rejected at load | raises |
| Per-body reset bounds and initial conditions | each body samples its own box; conditions broadcast to every environment, momentum = mass × velocity | inside bounds; conditions and momenta bitwise |
| Fields left out of `initial_conditions` | keep the drawn state; a prescribed mass keeps the drawn velocity | 2 m/s of circular speed |
| Initial conditions giving a position but no velocity under orbit-mode resets | rejected; with neither, the drawn circular state is kept | raises; radius and speed 1e-12 relative |
| Initial quaternions | normalized; a zero quaternion is rejected | 1e-7 relative; raises |
| No `initial_conditions` | states drawn at load, never the zero state (gravity is singular there) | \|r\| > 1000 km |
| `device=` passed to the loader | overrides the file | exact |
| Orbit specs: altitude bounds, per-body override | the requested radius | 1 m |
| Every shipped `examples/*.yaml` | loads and steps 5 times | finite state |
| The configuration guide lists exactly the keys of `simulation`, `physics`, `termination`, `reset` and `reset.orbit` (`tests/test_docs.py`) | the loader's schema | exact |
| The defaults the guide states | the simulator's constructor defaults | exact |

## API and outputs

`tests/test_api.py`, `tests/test_pipeline.py`

| Claim | Referent | Tolerance |
| --- | --- | --- |
| Getters for position, quaternion and mass | the state arrays | bitwise |
| Velocity and angular-velocity getters | p/m and I⁻¹L | 1e-15 and 1e-14 relative |
| Per-entity thrust, wheel-torque and wheel-state setters | reach only their entity | exactly zero elsewhere |
| Thruster and wheel configs with `env_ids` | change only those environments | exact |
| `results["time"][k]` is the epoch of stored state k: (k + 1) dt | the analytic orbit at that time (a one-step offset would be 7.5 km off) | 10 m |
| `metadata()` reflects the constructor and serializes to JSON | the constructor arguments and package version | exact |

## Documentation

`tests/test_docs.py`

| Claim | Referent | Tolerance |
| --- | --- | --- |
| Every Python snippet in the README, Getting started, Troubleshooting, the guides and the theory pages runs, page by page, on the test device | the page's own code | runs; pages needing a missing optional dependency skip |
| The validation pages cite only test files that exist, and every section with a table cites one | the `tests/` directory | exact |
| `CITATION.cff` version | `cosmic_warp.__version__` | exact |
| Every section of claims on these pages appears in the index summary | the summary's links | exact |

## Examples

`tests/test_examples.py`: the gallery pages are generated from the example
scripts, so every script runs, as a user runs it.

| Claim | Referent | Tolerance |
| --- | --- | --- |
| Each of the 17 example scripts, and `run_config.py` on each scenario YAML, runs with small arguments on the test device, by path, from a scratch directory | exit status, the script's closing output, the plots it is meant to write | exit 0 within a 60 s budget (120 s for training); exactly the expected PNGs, written to the working directory; nothing written into the repository |
| Every example script and scenario is covered | the files in `examples/` | exact |

## API surface and kernel hygiene

`tests/test_stability.py`, `tests/test_kernel_hygiene.py`

| Claim | Referent | Tolerance |
| --- | --- | --- |
| Every name the [API stability](../reference/stability.md) page promises resolves at its import path, with the stated arguments and struct fields | the code | exact |
| Every name `cosmic_warp` and `cosmic_warp.rl` export is on that page | `__all__` | exact |
| Every float literal (or name resolving to a float) in a kernel or `wp.func` of the package and the examples is a direct argument of a dtype constructor: a bare literal is float32 in Warp and rounds silently in comparisons | the parsed source; a synthetic negative control; a device check that `x == 0.1` differs from `x == wp.float64(0.1)` | exact |

## Benchmark gate

`tests/test_bench.py`: the performance regression gate in
`benchmarks/bench.py`. The measured numbers are on the
[benchmarks page](../benchmarks.md); they are not test-checked.

| Claim | Referent | Tolerance |
| --- | --- | --- |
| A run within the 15 % threshold passes; a 20 % drop fails | the stored baseline | exit code 0 / 1 |
| A metric missing from the run, or an empty run, fails | the stored baseline | exit code 1 |

# Validation

Every claim on these pages is checked by a test, and each table names the
file under [`tests/`](https://github.com/Pedro-g-Lima/COSMIC_WARP/tree/main/tests)
that runs it. Tolerances are the bounds the tests assert; "measured" values
are those recorded in the test file. [Benchmarks](../benchmarks.md) is the
exception: measured throughput, not test-checked claims.

| Question | Checked against |
| --- | --- |
| Are the models and integrators right? | closed forms (Kepler, J2 secular rates, torque-free precession, the rocket equation, Clohessy–Wiltshire); otherwise invariants, an independent fine-step integration, or the integrators against each other |
| Does the machinery change the answer? | eager execution for graph replay; the specified semantics for termination and resets; the distributions for random draws |
| Are the learning interfaces exact? | NumPy recomputation and twin environments; central finite differences for gradients |

None of this validates a real spacecraft: the models are checked against the
equations they implement, and the parameters are yours. All physics checks
run in float64.

## Summary

| Area | Checked against | In CI |
| --- | --- | --- |
| [Two-body motion](physics.md#two-body-motion) | the analytic Kepler solution | yes |
| [J2 oblateness](physics.md#j2-oblateness) | first-order secular rates, within 3 % | yes |
| [Rigid-body rotation](physics.md#rigid-body-rotation) and [environmental torques](physics.md#environmental-torques) | closed-form precession, gyroscopic, gravity-gradient and dipole torques | yes |
| [Atmospheric drag](physics.md#atmospheric-drag) | the analytic decay rate, within 10 % | yes |
| [Thrusters](physics.md#thrusters-and-the-rocket-equation) | the rocket equation; the exact burnout error | yes |
| [Reaction wheels](physics.md#reaction-wheels) | momentum conservation; an independent fine-step integration | yes |
| [Hohmann transfer](physics.md#impulsive-manoeuvres) and [relative motion](physics.md#relative-motion) | vis-viva; the Clohessy–Wiltshire solution | yes |
| [Orbital elements](physics.md#orbital-elements) and [their sampling](physics.md#orbital-element-sampling) | conic geometry, Kepler's equation, uniform distributions | yes |
| [Order of convergence](numerics.md#order-of-convergence) | the analytic solution and self-convergence: 1, 2, 4, capped at 2 at wheel saturation | yes |
| [Integrator agreement](numerics.md#integrator-agreement) | a fine RK4 reference against each integrator's Richardson estimate | yes |
| [Conserved quantities](numerics.md#conserved-quantities) | energy, angular momentum, eccentricity vector | yes |
| [Golden trajectories](numerics.md#golden-trajectories) | a committed bit-exact fixture per device | CPU fixture; GPU fixture local |
| [CUDA graphs](execution.md#cuda-graphs) | eager execution | no — local GPU |
| [Termination](execution.md#termination) and [resets](execution.md#resets) | specified semantics | yes |
| [Random streams](execution.md#random-streams-and-determinism) | Random123's known answers; uniform and Haar distributions; seeded replay | yes |
| [Input validation](execution.md#input-validation) and [YAML loader](execution.md#yaml-loader) | fail-closed rules; the configuration guide | yes |
| [API](execution.md#api-and-outputs), [documentation](execution.md#documentation) and [benchmark gate](execution.md#benchmark-gate) | the state arrays; the pages' own code; a stored baseline | yes; torch snippets in the RL job |
| [Examples](execution.md#examples) | each script's own output | yes; RL examples in the RL job |
| [API surface and kernel hygiene](execution.md#api-surface-and-kernel-hygiene) | the stability page; the parsed kernel source | yes |
| [Vectorized environment](learning.md#vectorized-environment) and [attitude task](learning.md#example-attitude-task) | NumPy recomputation; twin environments | CPU rows |
| [Trainer adapters](learning.md#trainer-adapters) | each library's interface; short PPO training | interfaces (RL job); training no — local GPU |
| [Differentiable rollouts](learning.md#differentiable-rollouts) | central finite differences | yes |
| [Torch bridge](learning.md#torch-bridge) and [differentiable environment](learning.md#differentiable-environment) | `gradcheck`, finite differences, the Warp-native rollout | RL job |
| [SHAC example](learning.md#shac-example) | the Warp task; a naive TD(λ); short training | RL job; training no — local GPU |
| [Domain randomization](learning.md#domain-randomization) | bounds and physical couplings | yes |

## CI and local GPU checks

GitHub's runners have no GPU, so CI runs on Warp's CPU backend
(`.github/workflows/tests.yml`):

| Job | Runs |
| --- | --- |
| Physics | `pytest --device cpu` on Python 3.12 and 3.14 with the `dev` group's pinned tools, without torch |
| Dependency range | the same suite at both ends of the requirements (Python 3.12 with Warp 1.15.0, NumPy 2.0.2, PyYAML 6.0.1; Python 3.14 with the newest), golden trajectories deselected |
| Wheel | the built wheel, installed alone, run from outside the checkout (`tests/wheel_smoke.py`) |
| RL | CPU torch with rsl_rl, skrl and rl_games on Python 3.12: `tests/test_rl_env.py`, `tests/test_rl_diff_env.py`, `tests/test_example_shac.py`, the three adapter files, `tests/test_docs.py`, `tests/test_examples.py`, skip reasons printed |

*Locally verified* on a GPU, not continuously (they skip on CPU):

- multi-body graph replay against eager execution (`tests/test_termination.py`);
- capture of `simulate()` with every feature on (`tests/test_hot_loop.py`);
- `VecEnv` graph replay against eager stepping (`tests/test_rl_env.py`);
- the four training runs, marked `slow`: PPO through each adapter and SHAC.

On CPU `simulate()` runs eagerly, so every test that calls it (orbits, J2,
drag, wheels, auto-reset sampling) checks graph replay only on the GPU.

```bash
pytest --device cuda                  # or --device cpu
pytest --device cuda -m "not slow"    # without the training runs
```

## Not yet validated

| Gap | Status |
| --- | --- |
| User forces and torques; custom integrators | not supported yet ([roadmap](../roadmap.md)) |
| Central bodies other than Earth | not configurable: orbit specs take no `mu` |
| Events inside a step | not located: wheel saturation caps every integrator at [order 2](numerics.md#order-of-convergence); burnout leaves an [O(dt²) position error](physics.md#thrusters-and-the-rocket-equation). Both measured, neither removed |
| Gradients at actuator limits | checks stay inside the actuator ranges; the throttle clamp and wheel saturation are kinks ([smoothness caveats](../theory/differentiable.md#smoothness-caveats)) |
| Magnetic field | only the aligned dipole the code implements; no IGRF comparison |
| Density table | read from the code by the test, so a wrong table value would pass |
| J2 beyond first-order secular rates | short-period terms and higher harmonics not compared |
| Long horizons | invariants checked over one orbit or a few seconds |
| Distribution of randomized vehicle parameters | only bounds and couplings (distributions are tested for reset states) |
| Learning beyond detumble | only short, seeded detumble runs on a GPU, each required to cut the body rate by at least 40 % |
| Multi-body trainer adapters | not supported: they raise `NotImplementedError` |
| Throughput on full-rate FP64 GPUs | not measured ([benchmarks](../benchmarks.md)) |

# COSMIC WARP

<p align="center"><img src="https://raw.githubusercontent.com/Pedro-g-Lima/COSMIC_WARP/main/docs/assets/banner.png" alt="COSMIC WARP" width="900" /></p>

<p align="center">
<a href="https://github.com/Pedro-g-Lima/COSMIC_WARP/actions/workflows/tests.yml"><img src="https://github.com/Pedro-g-Lima/COSMIC_WARP/actions/workflows/tests.yml/badge.svg" alt="Tests" /></a>
<a href="https://pedro-g-lima.github.io/COSMIC_WARP/"><img src="https://img.shields.io/badge/docs-GitHub%20Pages-blue" alt="Documentation" /></a>
<a href="https://pypi.org/project/cosmic-warp/"><img src="https://img.shields.io/pypi/v/cosmic-warp" alt="PyPI" /></a>
<img src="https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue" alt="Python 3.12-3.14" />
<a href="https://github.com/Pedro-g-Lima/COSMIC_WARP/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green" alt="License: Apache-2.0" /></a>
</p>

COSMIC WARP is a GPU-parallel spacecraft dynamics simulator built on
[NVIDIA Warp](https://github.com/NVIDIA/warp). It propagates thousands to
millions of independent spacecraft environments at once — 6-DOF orbital and
attitude dynamics in float64, with thrusters, reaction wheels and
environmental perturbations — for Monte Carlo studies, trajectory and
parameter optimization with exact gradients, and reinforcement learning.
`simulate()` and the RL environment replay each step as a CUDA graph, with
no host round-trip inside the loop.

**[Documentation](https://pedro-g-lima.github.io/COSMIC_WARP/)** — installation, guides,
examples, theory and API reference.

> **Status: 0.1 beta.** The public API may change before 1.0; every breaking
> change is listed in the [changelog](https://github.com/Pedro-g-Lima/COSMIC_WARP/blob/main/CHANGELOG.md).

## Highlights

- **GPU-parallel** — physics, integration and termination fused into one
  kernel per step for every environment; `simulate()` replays the step as a
  CUDA graph
- **Float64 throughout** — all dynamic state and physics in double precision
- **Integrators** — Euler, RK4 (default) and a symplectic kick–drift–kick scheme, each
  verified to reach its nominal order of convergence
- **Physics** — point-mass gravity, J2, atmospheric drag, gravity-gradient
  and magnetic torques, each switchable; variable mass from thruster
  propellant use
- **Actuators** — up to 8 thrusters (Isp, propellant use) and 4 reaction
  wheels (momentum saturation, viscous friction) per body
- **Multi-body** — several spacecraft per environment, configured per body
  and terminated together (no forces between them)
- **Differentiable** — reverse-mode gradients through the dynamics w.r.t.
  initial state, control sequences and vehicle parameters, checked against
  finite differences
- **Reinforcement learning** — on-device vectorized environment with
  auto-reset and zero-copy torch views, adapters for rsl_rl, skrl and
  rl_games, and a differentiable environment for first-order methods; a
  reference task and a minimal SHAC trainer in `examples/`
- **Domain randomization and reset sampling** — on-GPU, including orbital
  element sampling
- **Validated** — tests against analytic results: Kepler orbits, J2 secular
  rates, drag decay, torque-free rigid-body motion, Hohmann transfers, the
  rocket equation, Clohessy–Wiltshire relative motion, closed-form torques

## Inside a step

One fused kernel per step integrates every entity: one GPU thread each, with
the forces, torques, actuators, integrator update and termination check
inside. `simulate()` records the step's kernels once as a CUDA graph and
replays them.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Pedro-g-Lima/COSMIC_WARP/main/docs/assets/step-dark.svg">
  <img alt="One simulate() step: the step kernel, the multi-body reduction, the optional trajectory store and auto-reset, and the trajectory index. Inside the step kernel, per entity: episode counter, terminated check, thruster wrench, integration stages (forces, torques, wheel momenta), store, termination check." src="https://raw.githubusercontent.com/Pedro-g-Lima/COSMIC_WARP/main/docs/assets/step-light.svg">
</picture>

## Learning through the dynamics

`DiffVecEnv` runs short windows of steps on torch's autograd graph, so a
loss built from the rewards differentiates through the policy and the
dynamics at every step — the gradient a first-order method such as SHAC
uses (example 17).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Pedro-g-Lima/COSMIC_WARP/main/docs/assets/learning-dark.svg">
  <img alt="One DiffVecEnv window: H steps of one environment from the carried, detached state; a policy shared by every step maps observations to actions; rewards go into the trainer's loss, and backward runs through every step back to the start of the window." src="https://raw.githubusercontent.com/Pedro-g-Lima/COSMIC_WARP/main/docs/assets/learning-light.svg">
</picture>

## Installation

```bash
pip install cosmic-warp
```

Optional extras (quote the brackets):

| Extra | Adds |
| --- | --- |
| `torch` | PyTorch: torch views, differentiable RL |
| `rsl-rl`, `skrl`, `rl-games` | the trainer adapters (each pulls in torch) |
| `examples` | matplotlib for the example scripts |

```bash
pip install "cosmic-warp[torch,skrl]"
```

The examples and tests live in the repository; to run them, install from a
clone:

```bash
git clone https://github.com/Pedro-g-Lima/COSMIC_WARP.git
cd COSMIC_WARP
pip install -U pip    # the dependency groups below need pip 25.1
pip install -e ".[examples]"
```

A CUDA GPU is the intended target; everything also runs on the CPU
(`device="cpu"`; the default is `"cuda"`, with no automatic fallback). See
[Getting started](https://pedro-g-lima.github.io/COSMIC_WARP/getting-started/).

## Quick start

```python
from cosmic_warp.core.pipeline import SpacecraftSimulator

# 1000 spacecraft in low Earth orbit, drawn from the default reset bounds
sim = SpacecraftSimulator(num_envs=1000, integrator="rk4", dt=1.0)
sim.reset()

trajectory = sim.simulate(num_steps=600, save_states=True)
positions = trajectory["positions"].numpy()   # [num_steps, num_envs, 3]
```

Whole scenarios (physics, per-body vehicles and actuators, reset bounds,
initial conditions) can be defined in YAML (this file ships with the
repository; run from its root):

```python
from cosmic_warp.core.config import create_simulator_from_yaml

sim = create_simulator_from_yaml("examples/multibody_orbit.yaml")
```

Continue with [Getting started](https://pedro-g-lima.github.io/COSMIC_WARP/getting-started/)
and the [YAML reference](https://pedro-g-lima.github.io/COSMIC_WARP/guides/configuration/).

## Examples

Seventeen scripts in [`examples/`](https://github.com/Pedro-g-Lima/COSMIC_WARP/tree/main/examples/), run from the repository root
(`python examples/<name>.py --help`):

| Topic | Examples |
| --- | --- |
| Orbits and integrators | 01 basic orbit, 02 integrator comparison, 05 energy conservation, 08 J2 |
| Batches | 03 Monte Carlo, 04 scalability, 09 multi-body, 10 domain randomization |
| Actuators | 06 thrusters, 07 reaction wheels |
| Gradients | 11 trajectory optimization, 12 system identification |
| Reinforcement learning | 13 environment, 14–16 PPO with rsl_rl / skrl / rl_games, 17 SHAC |

The [examples gallery](https://pedro-g-lima.github.io/COSMIC_WARP/examples/) shows each
script with its plots and explanation.

## Documentation

| Section | Contents |
| --- | --- |
| [Getting started](https://pedro-g-lima.github.io/COSMIC_WARP/getting-started/) | Install, verify Warp, first orbit |
| [Guides](https://pedro-g-lima.github.io/COSMIC_WARP/guides/) | Configuration, vehicles, actuators, resets, physics, running, scaling, RL, differentiable simulation |
| [Examples](https://pedro-g-lima.github.io/COSMIC_WARP/examples/) | One page per example script |
| [Theory](https://pedro-g-lima.github.io/COSMIC_WARP/theory/) | Conventions, dynamics, gravity and J2, environment, actuators, integrators, sampling, GPU execution, gradients |
| [Validation](https://pedro-g-lima.github.io/COSMIC_WARP/validation/) | What the test suite checks, against which results |
| [Benchmarks](https://pedro-g-lima.github.io/COSMIC_WARP/benchmarks/) | Measured throughput per GPU |
| [API reference](https://pedro-g-lima.github.io/COSMIC_WARP/reference/api/cosmic_warp/) | Generated from the docstrings |
| [Roadmap](https://pedro-g-lima.github.io/COSMIC_WARP/roadmap/) | Planned for 1.x and beyond |

## Testing and benchmarks

```bash
pip install -e . --group dev   # pinned test tools (pip >= 25.1)
pytest                    # full suite on CUDA
pytest --device cpu       # the same suite without a GPU
pytest -m "not slow"      # skip the end-to-end training tests

python benchmarks/bench.py --save   # record this GPU's baseline (first run)
python benchmarks/bench.py          # compare; exit 1 on a >15 % regression
```

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](https://github.com/Pedro-g-Lima/COSMIC_WARP/blob/main/CONTRIBUTING.md). For
questions, bugs and feature requests, open an
[issue](https://github.com/Pedro-g-Lima/COSMIC_WARP/issues).

## Citation

If you use COSMIC WARP in your research, please cite it (GitHub's "Cite this
repository" uses [`CITATION.cff`](https://github.com/Pedro-g-Lima/COSMIC_WARP/blob/main/CITATION.cff)):

```bibtex
@software{cosmic_warp,
  title   = {COSMIC WARP: GPU-Accelerated Spacecraft Simulator},
  author  = {Lima, Pedro and Arora, Aman},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/Pedro-g-Lima/COSMIC_WARP}
}
```

Please also cite the tools it builds on, in particular NVIDIA Warp. BibTeX
for these and for every source behind the physics models is on the
[references page](https://pedro-g-lima.github.io/COSMIC_WARP/references/).

## License

[Apache License 2.0](https://github.com/Pedro-g-Lima/COSMIC_WARP/blob/main/LICENSE).

## Acknowledgements

COSMIC WARP is developed by the SpaceR research group at SnT, University of
Luxembourg.

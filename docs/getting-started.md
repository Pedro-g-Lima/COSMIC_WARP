# Getting started

Install COSMIC WARP, propagate a batch of spacecraft, fire a thruster, and
load a scenario from YAML.

## Requirements

<!-- generated: requirements -->

An NVIDIA GPU with CUDA is the target; everything also runs on the CPU,
more slowly.

**Python older than 3.12?** Ubuntu 22.04, for one, ships 3.10. Leave the
system Python alone and create the environment with a newer one:

```bash
conda create -n cosmic-warp python=3.12 && conda activate cosmic-warp
# or with uv, then `uv pip install` wherever this page says `pip install`
uv venv --python 3.12 && source .venv/bin/activate
```

## Install

```bash
pip install cosmic-warp
```

To run the examples and the tests, install from a clone instead:

```bash
git clone https://github.com/Pedro-g-Lima/COSMIC_WARP.git
cd COSMIC_WARP
python -m venv .venv && source .venv/bin/activate
pip install -U pip    # `pip install --group`, below, needs pip 25.1
pip install -e ".[examples]"
```

Optional extras, combined as `pip install "cosmic-warp[torch,skrl]"`:

<!-- generated: extras -->

## Verify

```bash
python -c "import warp as wp; wp.init()"
```

`wp.init()` lists the devices Warp found. Without a CUDA device, pass
`device="cpu"` to every simulator: the default is `"cuda"`, with no
automatic fallback ([Troubleshooting](troubleshooting.md#cuda-not-available)).

From a clone, run the test suite:

```bash
pip install -e . --group dev
pytest --device cpu    # or --device cuda (the default)
```

## First simulation

```python
import numpy as np
from cosmic_warp.core.pipeline import SpacecraftSimulator
from cosmic_warp.physics.gravity import RE_EARTH

sim = SpacecraftSimulator(num_envs=8, dt=1.0)
sim.reset()
results = sim.simulate(num_steps=600, save_states=True)

positions = results["positions"].numpy()   # (600, 8, 3): step, environment, xyz
altitude = np.linalg.norm(positions[-1], axis=1) - RE_EARTH
print((altitude / 1e3).round(1))           # km, after 10 minutes
```

One altitude per environment, a little below the 400 km they started at:

```text
[396.8 396.9 398.5 398.  397.5 399.1 398.9 397.5]
```

| Line | What it does |
| --- | --- |
| `SpacecraftSimulator(...)` | 8 independent environments, one 1 kg cube each, RK4 at 1 s, gravity and J2 on. Kernels compile on first use ([slow first run](troubleshooting.md#slow-first-run)). |
| `sim.reset()` | Samples each initial state: by default within ±1 km and ±1 m/s of a circular 400 km orbit, random attitude ([resets](guides/reset.md)). |
| `sim.simulate(...)` | Runs the steps on the device, as one CUDA graph on a GPU; `save_states=True` records the state after every step. |
| `.numpy()` | Copies a device array to the host. |

## The state

The getters return one row per entity, `N = num_envs * num_bodies`:

| Getter | Shape | Meaning |
| --- | --- | --- |
| `get_positions()` | (N, 3) | position in ECI, m |
| `get_velocities()` | (N, 3) | velocity in ECI, m/s |
| `get_quaternions()` | (N, 4) | attitude, body → ECI, stored `(x, y, z, w)` |
| `get_angular_velocities()` | (N, 3) | body rates, rad/s |
| `get_masses()` | (N,) | total mass (dry + propellant), kg |

The state holds momenta; the getters convert. Each getter returns a buffer
its next call overwrites, so call `.numpy()` first. Frames and units:
[Frames, state and units](theory/conventions.md).

## Fire a thruster

Thrust needs propellant: the reset mass must exceed the dry mass.

```python
sim = SpacecraftSimulator(num_envs=4, dt=0.1)
sim.vehicle.set_config(mass_dry=10.0, inertia=np.diag([0.5, 0.6, 0.7]))
sim.thruster_manager.set_config(max_thrust=1.0, isp=220.0)  # default six-thruster layout
sim.reset_manager.set_bounds(mass_bounds=(12.0, 12.0))      # 2 kg of propellant
sim.reset()                    # also zeroes the commands, so command after it
sim.set_thrust_commands([1.0]) # thruster 0, pointing +X, at full thrust
sim.simulate(num_steps=600)    # one minute
print(sim.get_masses().numpy())
```

Each spacecraft burned `1 N × 60 s / (220 s × g0)` = 0.0278 kg:

```text
[11.97218956 11.97218956 11.97218956 11.97218956]
```

More in [Vehicles](guides/vehicle.md) and [Actuators](guides/actuators.md).

## Multi-body

An environment can hold several spacecraft, such as a chaser and a target:

```python
sim = SpacecraftSimulator(num_envs=2, num_bodies=3)
sim.reset()
positions = sim.get_positions().numpy()  # (6, 3)
env, body = 1, 2
print(positions[env * sim.num_bodies + body])
```

Rows are environment-major: `entity = env * num_bodies + body`. Bodies exert
no forces on each other; they share termination and reset. Configuration
calls take `body_ids=[...]` to configure one body in every environment.

## From a YAML file

A whole scenario fits in one file. This one ships with the repository (run
from its root):

```python
from cosmic_warp.core.config import create_simulator_from_yaml

sim = create_simulator_from_yaml("examples/multibody_orbit.yaml")
sim.simulate(num_steps=100)
```

The loader checks every key and value and returns a reset simulator;
`device="cpu"` overrides the file's device. Format:
[Configure a scenario](guides/configuration.md).

## Common tasks

| Task | Call | Guide |
| --- | --- | --- |
| Pick the integrator | `SpacecraftSimulator(integrator="symplectic")` | [Integrators](guides/integrators.md) |
| Switch physics models | `SpacecraftSimulator(drag=True, j2_perturbation=False)` | [Physics models](guides/physics.md) |
| Set mass, inertia, drag area | `sim.vehicle.set_config(...)` | [Vehicles](guides/vehicle.md) |
| Configure thrusters and wheels | `sim.thruster_manager.set_config(...)`, `sim.rw_manager.set_config(...)` | [Actuators](guides/actuators.md) |
| Command actuators | `sim.set_thrust_commands(...)`, `sim.set_rw_torque_commands(...)` | [Actuators](guides/actuators.md) |
| Set reset bounds or exact states | `sim.reset_manager.set_bounds(...)`, `sim.reset_to_conditions(...)` | [Initial conditions and resets](guides/reset.md) |
| Randomize vehicle parameters | `sim.randomizer.set_bounds(...)` | [Domain randomization](guides/randomization.md) |
| Record, terminate, auto-reset | `sim.simulate(..., auto_reset_terminated=True)` | [Run, record, terminate](guides/running.md) |

## Next

[Guides](guides/index.md) · [Examples](examples/index.md), starting with the
[basic orbit](examples/01_basic_orbit.md) · [Validation](validation/index.md) ·
[API reference](reference/api/cosmic_warp/index.md)

# Getting started

Install COSMIC WARP, propagate a batch of spacecraft, fire a thruster, and
load a scenario from YAML.

## Requirements

<!-- generated: requirements -->

An NVIDIA GPU with CUDA is the intended target. Everything also runs on the
CPU, more slowly.

## Install

From PyPI:

```bash
pip install cosmic-warp
```

The examples and the test suite live in the repository. To run them, install
from a clone instead:

```bash
git clone https://github.com/Pedro-g-Lima/COSMIC_WARP.git
cd COSMIC_WARP
python -m venv .venv && source .venv/bin/activate
pip install -U pip    # `pip install --group`, below, needs pip 25.1
pip install -e ".[examples]"
```

Optional extras:

<!-- generated: extras -->

Combine them as needed, quoting the brackets: `pip install "cosmic-warp[torch,skrl]"`.

## Verify

```bash
python -c "import warp as wp; wp.init()"
```

`wp.init()` prints the devices Warp found. If no CUDA device is listed, pass
`device="cpu"` to every simulator — the default device is `"cuda"`, and there
is no automatic fallback (see [Troubleshooting](troubleshooting.md#cuda-not-available)).

From a clone, install the pinned test tools and run the suite on either
device:

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

It prints one altitude per environment, a little below the 400 km they
started at:

```text
[396.8 396.9 398.5 398.  397.5 399.1 398.9 397.5]
```

What each line does:

1. **Create the simulator.** Eight independent environments, one spacecraft
   each, stepped with RK4 (the default integrator) at 1 s. Gravity and J2 are
   on; the vehicle is a 1 kg cube. Kernels compile on first use, so the first
   run takes longer (see [Troubleshooting](troubleshooting.md#slow-first-run)).
2. **Reset.** `reset()` samples every environment's initial state: by
   default within ±1 km and ±1 m/s of a circular 400 km orbit, with a random
   attitude. [Initial conditions and resets](guides/reset.md) sets other
   bounds or exact states.
3. **Simulate.** `simulate(num_steps)` runs the steps on the device — on a GPU,
   replayed as one CUDA graph. With `save_states=True` it also records the
   state after every step.
4. **Read back.** The results are Warp arrays on the device; `.numpy()`
   copies one to the host.

## The state

The state of each spacecraft lives on the device. The getters return one row
per entity:

| Getter | Shape | Meaning |
| --- | --- | --- |
| `get_positions()` | (N, 3) | position in ECI, m |
| `get_velocities()` | (N, 3) | velocity in ECI, m/s |
| `get_quaternions()` | (N, 4) | attitude, body → ECI, stored `(x, y, z, w)` |
| `get_angular_velocities()` | (N, 3) | body rates, rad/s |
| `get_masses()` | (N,) | total mass (dry + propellant), kg |

`N = num_envs * num_bodies`. Internally the state holds linear and angular
momenta, not velocities; the getters convert. Each getter returns a buffer
that the next call overwrites, so call `.numpy()` before calling it again.
Frames and units are defined in [Frames, state and units](theory/conventions.md).

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

[Vehicles](guides/vehicle.md) and [Actuators](guides/actuators.md) have the
details.

## Multi-body

An environment can hold several spacecraft — a chaser and a target, say:

```python
sim = SpacecraftSimulator(num_envs=2, num_bodies=3)
sim.reset()
positions = sim.get_positions().numpy()  # (6, 3)
env, body = 1, 2
print(positions[env * sim.num_bodies + body])
```

Rows are environment-major: `entity = env * num_bodies + body`. Bodies of one
environment do not exert forces on each other; they share termination and
reset. Every configuration call takes `body_ids=[...]` to configure one body
across all environments.

## From a YAML file

A whole scenario — simulator, physics, termination, vehicles, actuators,
reset bounds and initial conditions — fits in one file. This one ships with
the repository; run from its root:

```python
from cosmic_warp.core.config import create_simulator_from_yaml

sim = create_simulator_from_yaml("examples/multibody_orbit.yaml")
sim.simulate(num_steps=100)
```

The loader checks every key and value and resets the simulator, so it is
ready to step. Pass `device="cpu"` to override the file's device. The format
is in [Configure a scenario](guides/configuration.md).

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

## Next steps

- [Guides](guides/index.md) — one task per page.
- [Examples](examples/index.md) — complete, runnable scripts, starting with
  the [basic orbit](examples/01_basic_orbit.md).
- [Validation](validation/index.md) — what is tested, against what.
- [API reference](reference/api/cosmic_warp/index.md).

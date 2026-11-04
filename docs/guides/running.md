# Run, record, terminate

`step()` advances one step per Python call; `simulate(num_steps)` runs many
in one call. Both integrate every environment with the commands currently
set, and neither copies to the host.

## Step by step

```python
from cosmic_warp.core.pipeline import SpacecraftSimulator

sim = SpacecraftSimulator(num_envs=64, dt=1.0)   # LEO reset bounds by default
sim.reset()

for _ in range(10):
    sim.step()
```

Each `step()` launches the step kernel (plus a reduction when
`num_bodies > 1`). Use it when Python changes commands between steps; for
long runs at fixed commands use `simulate()`, and for closed-loop control at
full speed see [Scale up](scaling.md#capture-your-own-loop).

## Record a trajectory

```python
results = sim.simulate(num_steps=600, save_states=True)

t = results["time"].numpy()                # [600]
positions = results["positions"].numpy()   # [600, num_entities, 3]
```

On CUDA, `simulate()` captures one step into a CUDA graph and replays it
`num_steps` times; commands stay as set before the call.

| Key | Present | Type, shape | Content |
| --- | --- | --- | --- |
| `time` | `save_states` or `save_rw_states` | float64 `[num_steps]` | `(k + 1) * dt`: row `k` is the state **after** step `k + 1` of this call |
| `positions` | `save_states=True` | vec3d `[num_steps, num_entities]` | inertial position [m] |
| `velocities` | `save_states=True` | vec3d | inertial velocity, `mom / max(mass, mass_dry)` [m/s] |
| `quaternions` | `save_states=True` | quatd | attitude `(x, y, z, w)`, body to inertial |
| `angular_velocities` | `save_states=True` | vec3d | body rate, `inertia_inv @ ang_mom` [rad/s] |
| `masses` | `save_states=True` | float64 | total mass [kg] |
| `rw_angular_velocities` | `save_rw_states=True` | vec4d | wheel speeds, `rw_ang_mom / wheel_inertia` [rad/s] |

Each call allocates its arrays on the device; without a recording it
allocates nothing and returns `{}`. `save_states` costs 112 bytes per entity
per step, `save_rw_states` 32. A multi-body trajectory reshapes to
`positions.reshape(num_steps, sim.num_envs, sim.num_bodies, 3)`.

## Final states only

```python
final = sim.simulate_monte_carlo(num_steps=600)

final_positions = final["final_positions"].numpy()   # [num_entities, 3]
terminated = final["termination_status"].numpy()["terminated"]
```

`simulate_monte_carlo()` stores no history and has no auto-reset. It returns
`final_positions`, `final_velocities`, `final_quaternions`,
`final_angular_velocities`, `final_masses`, `termination_status` and
`sim_time` (`num_steps * dt`). The arrays are the getters' buffers and the
live status: copy what you need before the simulator runs again.

## Read the current state

| Getter | Returns, per entity |
| --- | --- |
| `get_positions()` | vec3d, inertial [m] |
| `get_velocities()` | vec3d, inertial [m/s] |
| `get_quaternions()` | quatd `(x, y, z, w)` |
| `get_angular_velocities()` | vec3d, body frame [rad/s] |
| `get_masses()` | float64 [kg] |
| `get_rw_angular_momenta()` | vec4d, wheels 0–3 about their axes [kg·m²/s] |
| `get_terminated()` | `TerminationStatus` structs: `terminated`, `reason` |
| `get_episode_steps()` | int32, steps since the entity's last reset |

A getter fills a preallocated buffer that its next call overwrites.
`sim.states` holds the raw `State` structs
([fields](../theory/conventions.md#state)). It, `dt`, the counts, the device
and the managers are read-only attributes: assign into the arrays, never
rebind them. `.numpy()` copies and synchronizes; keep it out of hot loops.

## Termination

Each step checks every entity against limits fixed at construction (or the
[`termination` YAML section](configuration.md#termination)):

```python
from cosmic_warp.physics.gravity import RE_EARTH

sim = SpacecraftSimulator(
    num_envs=64,
    dt=1.0,
    check_bounds=True,        # position, momentum, angular-momentum and mass limits
    pos_limit=1.0e8,          # |pos| [m]
    mom_limit=1.0e6,          # |mom| [kg*m/s]
    ang_mom_limit=10.0,       # |ang_mom| [kg*m^2/s]
    mass_limit=0.0,           # terminate once mass <= this [kg]; 0 disables
    check_planet_impact=True,
    planet_radius=RE_EARTH,   # |pos| below this is an impact [m]; the default
)
sim.reset()
sim.simulate(num_steps=100)

status = sim.get_terminated().numpy()       # structured: "terminated", "reason"
reasons = sim.termination_checker.get_termination_reasons()   # {env: code}, host read
```

The defaults are in the [`termination` table](configuration.md#termination).
The momentum limits are absolute, not velocities (at LEO speed, 1.0e6 allows
about 130 kg). `get_termination_reasons()` is keyed by environment:
termination is environment-wide.

A terminated environment stays frozen until reset:

```python
sim.reset_terminated()                                   # on the device, every terminated env
sim.simulate(num_steps=100, auto_reset_terminated=True)  # the same, after every step
sim.reset(env_ids=[0])                                   # explicit; resets all bodies of env 0
```

Reason codes, precedence, the always-on non-finite check and the counters:
[GPU execution](../theory/execution.md#termination).

## Reproducibility

`sim.metadata()` is a JSON-serializable provenance snapshot: the
cosmic_warp, warp-lang and Python versions, the device, and the
constructor's dimensions, integrator, `dt`, physics flags, termination
switches and `differentiable` flag.

```python
import json

provenance = json.dumps(sim.metadata(), indent=2)   # store it with your results
```

It omits the termination limits and everything configured after
construction (vehicles, actuators, bounds, seeds, initial conditions,
commands): save the scenario YAML or script with it, and a `pip freeze`.

| Same machine and versions | Other GPU or warp-lang version |
| --- | --- |
| bitwise identical ([checked](../validation/execution.md#random-streams-and-determinism)): one thread per entity, no atomics, fixed default seeds | close, not bit-identical: generated code changes, and float64 differences grow over long runs |

API: [`SpacecraftSimulator`](../reference/api/cosmic_warp/core/pipeline.md) ·
[`TerminationChecker`](../reference/api/cosmic_warp/core/termination.md)

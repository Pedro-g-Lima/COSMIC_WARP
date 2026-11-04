# Domain randomization

`sim.randomizer.set_bounds(...)` makes every reset draw new vehicle
parameters per entity, on the device: dry mass, propellant load, inertia
scale. Draws are uniform; bounds are per body, shared by all environments.

```python
from cosmic_warp.core.pipeline import SpacecraftSimulator

sim = SpacecraftSimulator(num_envs=64)
sim.vehicle.set_config(mass_dry=70.0)
sim.randomizer.set_bounds(
    mass_dry=(60.0, 80.0),      # kg
    propellant=(5.0, 15.0),     # kg, on top of the drawn dry mass
    inertia_scale=(0.8, 1.2),   # multiplier on the configured inertia
)
sim.reset()                                          # first draw
sim.simulate(num_steps=100, auto_reset_terminated=True)
masses = sim.get_masses().numpy()                    # between 65 and 95 kg
```

Every reset path draws — `sim.reset()` (whole or partial),
`reset_terminated()`, auto-reset in `simulate()` and the
[RL environment](rl.md) — inside the captured graph, with no host
synchronization. `reset_to_conditions` does not randomize.

## Bounds

| Bound | Meaning | Accepted values |
| --- | --- | --- |
| `mass_dry` | dry mass, kg | `0 < min ≤ max`; needs `propellant` for the same body |
| `propellant` | propellant mass, kg; reset mass = `mass_dry + propellant` | `0 ≤ min ≤ max` |
| `inertia_scale` | one factor per entity on its configured inertia tensor | `0 < min ≤ max` |

`body_ids` limits a call to some bodies; a bound you do not pass keeps its
value, so calls can be layered; equal `min` and `max` fix a value. Bounds
cannot be removed: build a new simulator for nominal runs. YAML: the
[`randomization` section](configuration.md#randomization).

## Physical couplings

- **Propellant, not total mass, is drawn**, so an entity never spawns below
  its dry mass. Hence a randomized `mass_dry` needs `propellant` bounds, and
  a body with `propellant` bounds ignores its reset `mass_bounds`.
- **Velocity is kept, not momentum**: the momentum is rescaled with the new
  mass, so a drawn orbit ([orbital-element sampling](reset.md#orbital-element-sampling))
  does not depend on the propellant load.
- **The inertia is scaled whole**, which keeps it physical. The scale is
  drawn independently of the masses.

## The nominal inertia

`inertia_scale` multiplies each entity's configured tensor (from
`set_config` or `set_config_per_entity`), never a previous draw. A newly
configured tensor becomes the nominal at the next `set_bounds`,
`sim.reset()`, `reset_terminated()`, `simulate()` or RL `env.step()` — each
compares a version number on the host and uploads only on change. A CUDA
graph you capture yourself does not run that check: after a `set_config`
between replays, its resets keep the old nominal until one of those calls
runs outside the graph.

## Seeds and streams

The draws use their own per-entity streams, with a different stream id from
the reset sampler's, so the two never share a draw, even under equal seeds
([random streams](../theory/sampling.md#random-streams)). Seeds are integers
in [0, 2⁶⁴); `sim.randomizer.reseed(seed)` or `randomization.seed` in YAML
restarts the streams. An entity's draws advance only when it resets, so its
parameters do not depend on when other environments terminate.

## Other distributions

Draw values on the host and write them with
[`set_config_per_entity`](vehicle.md#per-entity); they stay fixed across
resets.

API: [`DomainRandomizer`](../reference/api/cosmic_warp/core/randomization.md)

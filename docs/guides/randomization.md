# Domain randomization

`sim.randomizer.set_bounds(...)` makes every reset draw new vehicle
parameters — dry mass, propellant load, inertia scale — per entity, on the
device. Draws are uniform; bounds are per body and shared by all
environments.

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

The draws happen on every reset path: `sim.reset()` (whole or partial),
`reset_terminated()`, `simulate(auto_reset_terminated=True)` and the
[RL environment](rl.md)'s auto-resets. They run inside the captured CUDA
graph, with no host synchronization. `reset_to_conditions` does not
randomize.

## Bounds

| Bound | Meaning | Accepted values |
| --- | --- | --- |
| `mass_dry` | dry mass, kg | `0 < min ≤ max`; needs `propellant` for the same body |
| `propellant` | propellant mass, kg; the reset mass is `mass_dry + propellant` | `0 ≤ min ≤ max` |
| `inertia_scale` | one factor per entity on its configured inertia tensor | `0 < min ≤ max` |

`body_ids` limits a call to some bodies. A bound you do not pass keeps its
current value, so calls can be layered per body; an equal `min` and `max`
fixes the value. Bounds cannot be removed once set — build a new simulator
for nominal runs. In YAML: the [`randomization` section](configuration.md#randomization).

## Physical couplings

- **Propellant, not total mass, is drawn.** The reset mass is written as
  `mass_dry + propellant`, so an entity never spawns below its dry mass. For
  the same reason a randomized `mass_dry` needs `propellant` bounds, and a
  body with `propellant` bounds ignores its reset `mass_bounds`.
- **Velocity is kept, not momentum.** Overwriting the mass rescales the
  momentum by `new_mass / old_mass`, so the drawn trajectory — an orbit from
  [orbital-element sampling](reset.md#orbital-element-sampling), say — does
  not depend on the propellant load.
- **The inertia is scaled whole.** One factor multiplies the tensor, which
  keeps it symmetric, positive definite and physical. The scale is drawn
  independently of the masses: a heavier draw is not a larger inertia.

## The nominal inertia

`inertia_scale` multiplies each entity's configured tensor — from
`vehicle.set_config` or, per entity, `set_config_per_entity` — never a previous
draw. A new configured tensor becomes the nominal at the next `set_bounds`,
`sim.reset()`, `reset_terminated()`, `simulate()` or RL `env.step()` call,
so every later draw scales it. These calls compare a version number on the
host and upload only when the inertia changed. A CUDA graph you capture
yourself does not run them on replay: after a `set_config` between replays,
its resets keep the old nominal until one of those calls runs outside the
graph.

## Seeds and streams

The parameter draws have their own per-entity random streams, separate
from the reset sampler's: the generator is counter-based, and the two use
different stream ids, so they never share a draw, even under equal seeds
([random streams](../theory/sampling.md#random-streams)). A seed is an
integer in [0, 2⁶⁴). `sim.randomizer.reseed(seed)` restarts the streams;
`randomization.seed` does it from YAML. Each entity's draws advance only when
it resets, so the parameters an environment sees do not depend on when the
others terminate.

## Other distributions

For a distribution other than uniform, draw the values on the host and write
them with [`set_config_per_entity`](vehicle.md#per-entity). Those values stay
fixed across resets.

API: [`DomainRandomizer`](../reference/api/cosmic_warp/core/randomization.md)

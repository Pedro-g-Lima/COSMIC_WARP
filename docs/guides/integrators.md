# Integrators

Three fixed-step schemes, chosen at construction:

```python
from cosmic_warp.core.pipeline import SpacecraftSimulator

sim = SpacecraftSimulator(num_envs=4, integrator="symplectic", dt=1.0)
sim.reset()
sim.simulate(num_steps=600)
```

| `integrator` | Order | Force evaluations per step | Use for |
| --- | --- | --- | --- |
| `"rk4"` (default) | 4 | 4 | most work: the most accurate per step |
| `"symplectic"` | 2 | 2 | long orbit propagation: bounded energy error instead of drift |
| `"euler"` | 1 | 1 | debugging, very small steps |

Cost scales roughly with force evaluations ([Benchmarks](../benchmarks.md));
each order is [checked by a convergence test](../validation/numerics.md#order-of-convergence).

## Choosing the step

There is no adaptive step and no stability guard. Pick `dt` for the fastest
dynamics in the scenario — the orbit, a spin, a wheel's friction time
constant, a burn — then halve it and compare the quantity you care about.

## Where the order drops

| Situation | Effect |
| --- | --- |
| A reaction wheel saturates inside a step | every scheme drops to second order around it (events are not located) |
| Propellant runs out inside a step | delta-v exact; an O(dt²) position error, once per burnout ([burnout](../theory/dynamics.md#variable-mass)) |
| Thrust or drag with `"symplectic"` | no longer symplectic (it is for the orbit at constant mass under gravity and J2); drag enters to first order |
| Thrust within any step | thrust, torque and mass flow are evaluated once per step; only their rotation follows the attitude |

Derivations: [Numerical integration](../theory/integrators.md). Comparisons
on orbits: [integrator comparison](../examples/02_integrator_comparison.md),
[energy conservation](../examples/05_energy_conservation.md). YAML:
`simulation.integrator`.

API: [`Integrator` and the factory](../reference/api/cosmic_warp/core/integrators/base.md) ·
[Euler](../reference/api/cosmic_warp/core/integrators/euler.md) ·
[RK4](../reference/api/cosmic_warp/core/integrators/rk4.md) ·
[Symplectic](../reference/api/cosmic_warp/core/integrators/symplectic.md)

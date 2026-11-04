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
| `"symplectic"` | 2 | 2 | long orbit propagation: the energy error stays bounded instead of drifting |
| `"euler"` | 1 | 1 | debugging, or very small steps |

The step costs roughly in proportion to its force evaluations — see
[Benchmarks](../benchmarks.md) for measured rates. Each order is verified by
a convergence test; the [Validation](../validation/index.md) page lists them.

## Choosing the step

There is no adaptive step and no stability guard: `dt` is yours to choose.
Pick it for the fastest dynamics in the scenario — the orbit, a spin, a
reaction wheel's friction time constant, the length of a burn — then check
convergence: halve `dt` and compare the quantity you care about.

## Where the order drops

- **Events inside a step.** No scheme locates events. A reaction wheel
  saturating mid-step limits every scheme to second order around it. On the
  step where the propellant runs out, the remaining propellant is spread
  over the whole step: the delta-v is exact, the position error is
  O(dt²), once per burnout.
- **Symplectic, beyond gravity.** The scheme is symplectic for the orbit at
  constant mass under gravity and J2. Thrust and drag break that structure,
  and drag enters to first order in its (usually small) contribution.
- **Thrust within a step.** Thrust, its torque and the mass flow are
  evaluated once per step from the commands and propellant at the step
  start; only their rotation into the inertial frame follows the attitude
  within the step.

The schemes are derived in [Numerical integration](../theory/integrators.md).
[Integrator comparison](../examples/02_integrator_comparison.md) and
[Energy conservation](../examples/05_energy_conservation.md) compare them on
orbits. In YAML: `simulation.integrator`.

API: [`Integrator` and the factory](../reference/api/cosmic_warp/core/integrators/base.md) ·
[Euler](../reference/api/cosmic_warp/core/integrators/euler.md) ·
[RK4](../reference/api/cosmic_warp/core/integrators/rk4.md) ·
[Symplectic](../reference/api/cosmic_warp/core/integrators/symplectic.md)

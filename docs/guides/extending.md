# Extending

## What you can extend in 0.1

Everything below goes through the public API, with no change to the
library:

- **Scenarios:** vehicles, actuators, reset and randomization bounds,
  termination limits and initial conditions, from Python or
  [YAML](configuration.md).
- **Control:** per-step thruster and reaction-wheel commands, from Python
  or from your own kernels ([Scale up](scaling.md#capture-your-own-loop)),
  and differentiable command sequences for
  [optimization](differentiable.md).
- **Learning tasks:** observations, rewards and action mappings, through the
  `Task` protocol ([`VecEnv`](rl.md#write-a-task)) or the `DiffTask`
  protocol ([`DiffVecEnv`](differentiable-rl.md)).

## Known limitations (planned for 1.x)

- **Custom integrators.** An integrator kernel must re-implement force
  accumulation, mass flow, termination and the episode counters, so there is
  no supported way to add one yet.
- **Custom forces and torques.** There is no per-entity external force or
  torque input: disturbances or new force models (solar radiation pressure,
  third bodies, contact) need changes to the step kernels. A differentiable
  external-wrench input is planned.

Both may change the integrator interface before 1.0; the
[changelog](../reference/changelog.md) will list it.

## Kernel conventions

These apply to code inside the simulator, and to your own kernels that read
its state.

All state and physics are **float64**. A Python float literal in a Warp
kernel is float32, so spell every literal out:

```python
import warp as wp

MU = wp.constant(wp.float64(3.986004418e14))   # module-level constants


@wp.func
def accel(pos: wp.vec3d, mu: wp.float64) -> wp.vec3d:
    r = wp.length(pos)
    if r > wp.float64(0.0):                    # every literal as wp.float64
        return -mu / (r * r * r) * pos
    return wp.vec3d()                          # zero vector
```

A bare literal in float64 arithmetic is a compile error (`Input types must
be the same`). A bare literal in a comparison compiles, but compares
against the literal's float32 value — `x == 0.1` is false for a float64
`x = 0.1` — so the explicit form matters there too.

Two more conventions from the step kernels:

- **Adjoint off by default.** The integrator modules set
  `wp.set_module_options({"enable_backward": False})`, because backward
  code generation adds to their compile time.
  `SpacecraftSimulator(differentiable=True)` turns it back on for the
  selected integrator's module, so integrator kernels must stay
  adjoint-safe even though they normally compile without the adjoint.
- **Adjoint-safe kernels.** [Differentiable rollouts](differentiable.md)
  backpropagate through the step kernels, and Warp's autodiff produces wrong
  gradients for some struct patterns without an error. The integrators
  follow these rules: load each struct into a local once and treat it as
  read-only (`state0 = states_in[tid]`); keep intermediates in
  single-assignment locals; build new structs through their constructors
  (`State(pos, mom, mass, quat, ang_mom)`), never by mutating a field of a
  local; and write results field by field (`store_state()`), never as a
  whole-struct store of a mutated local. `tests/test_gradients.py` compares
  every integrator's gradients with finite differences and catches
  violations.

## Contributing

Bug reports, fixes and new physics or actuator models are welcome; the
1.x extension points above, solar radiation pressure and third-body
gravity are the main gaps. See the [roadmap](../roadmap.md) and
[CONTRIBUTING.md](https://github.com/Pedro-g-Lima/COSMIC_WARP/blob/main/CONTRIBUTING.md)
for the development install, the tests, the performance gate and what a
pull request needs. For larger changes, open an
[issue](https://github.com/Pedro-g-Lima/COSMIC_WARP/issues) first.

# Extending

## Through the public API

No library change needed for:

- **Scenarios**: vehicles, actuators, reset and randomization bounds,
  termination limits and initial conditions, from Python or
  [YAML](configuration.md).
- **Control**: per-step thruster and wheel commands, from Python or your own
  kernels ([Scale up](scaling.md#capture-your-own-loop)), and differentiable
  command sequences ([optimization](differentiable.md)).
- **Learning tasks**: observations, rewards and action mappings, through the
  `Task` protocol ([`VecEnv`](rl.md#write-a-task)) or `DiffTask`
  ([`DiffVecEnv`](differentiable-rl.md)).

## Known limitations

- **No custom integrators.** An integrator kernel must re-implement force
  accumulation, mass flow, termination and the episode counters; there is no
  supported interface yet.
- **No custom forces or torques.** There is no per-entity external force or
  torque input, so new force models or disturbances (solar radiation
  pressure, third bodies, contact) need changes to the step kernels.

Both are planned ([roadmap](../roadmap.md)) and may change the integrator
interface before 1.0.

## Kernel conventions

For code inside the simulator and for your own kernels that read its state.

**Float64 literals.** All state and physics are float64, and a Python float
literal in a Warp kernel is float32, so spell every literal out:

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

In float64 arithmetic a bare literal is a compile error (`Input types must
be the same`); in a comparison it compiles but compares against the float32
value (`x == 0.1` is false for a float64 `x = 0.1`).

**Adjoint off by default.** The integrator modules set
`wp.set_module_options({"enable_backward": False})` to save compile time;
`SpacecraftSimulator(differentiable=True)` turns it on for the selected
integrator. Integrator kernels must therefore stay adjoint-safe.

**Adjoint-safe kernels.** Warp's autodiff produces wrong gradients for some
struct patterns without an error. The integrators load each struct into a
local once and treat it as read-only (`state0 = states_in[tid]`); keep
intermediates in single-assignment locals; build new structs through their
constructors (`State(pos, mom, mass, quat, ang_mom)`), never by mutating a
local's field; and write results field by field (`store_state()`).
`tests/test_gradients.py` compares every integrator's gradients with finite
differences and catches violations.

## Contributing

Bug reports, fixes and new physics or actuator models are welcome; the
limitations above, solar radiation pressure and third-body gravity are the
main gaps. See the [roadmap](../roadmap.md) and
[CONTRIBUTING.md](https://github.com/Pedro-g-Lima/COSMIC_WARP/blob/main/CONTRIBUTING.md)
(development install, tests, performance gate, pull requests). For larger
changes, open an [issue](https://github.com/Pedro-g-Lima/COSMIC_WARP/issues)
first.

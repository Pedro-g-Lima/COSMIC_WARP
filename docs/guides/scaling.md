# Scale up

The same simulator runs one environment or a million. Getting the speed
takes three things: enough environments to fill the GPU, the step replayed
as a CUDA graph, and no host copies inside the loop.

## A complete loop

A rate-damping controller closes the loop on the device: a kernel reads the
state and writes per-entity wheel torques, the simulator steps, and a second
kernel accumulates a diagnostic. The whole iteration is captured once and
replayed.

```python
import numpy as np
import warp as wp

from cosmic_warp.core.pipeline import SpacecraftSimulator
from cosmic_warp.core.state import State
from cosmic_warp.core.vehicle import VehicleConfig


@wp.kernel
def damp_rates(states: wp.array(dtype=State), vehicles: wp.array(dtype=VehicleConfig),
               gain: wp.float64, torque: wp.array(dtype=wp.float64, ndim=2)):
    tid = wp.tid()
    omega = vehicles[tid].inertia_inv * states[tid].ang_mom    # body rate [rad/s]
    for k in range(3):
        torque[tid, k] = gain * omega[k]                       # wheel k on body axis k


@wp.kernel
def sum_rates(states: wp.array(dtype=State), vehicles: wp.array(dtype=VehicleConfig),
              total: wp.array(dtype=wp.float64)):
    tid = wp.tid()
    omega = vehicles[tid].inertia_inv * states[tid].ang_mom
    wp.atomic_add(total, 0, wp.length(omega))


sim = SpacecraftSimulator(num_envs=1024, dt=0.1, check_bounds=False)
sim.vehicle.set_config(mass_dry=10.0, inertia=np.diag([0.3, 0.3, 0.2]))
sim.reset_manager.set_bounds(mass_bounds=(10.0, 10.0))
sim.rw_manager.set_config(max_torque=0.01, max_momentum=0.5, wheel_inertia=0.002)
sim.reset()

torque = wp.zeros((sim.num_entities, 4), dtype=wp.float64, device=sim.device)
total = wp.zeros(1, dtype=wp.float64, device=sim.device)


def control_step():
    wp.launch(damp_rates, dim=sim.num_entities,
              inputs=[sim.states, sim.vehicle.configs, 0.05, torque], device=sim.device)
    sim.set_rw_torque_commands_per_entity(torque)
    sim.step()
    wp.launch(sum_rates, dim=sim.num_entities,
              inputs=[sim.states, sim.vehicle.configs, total], device=sim.device)


if wp.get_device(sim.device).is_cuda:
    with wp.ScopedCapture(device=sim.device) as capture:   # records, does not run
        control_step()
    for _ in range(200):
        wp.capture_launch(capture.graph)
else:
    for _ in range(200):
        control_step()

mean_rate = total.numpy()[0] / (200 * sim.num_entities)   # one host copy, at the end
```

On a CPU the same loop runs eagerly; capture needs a CUDA device.

## Choose the batch size

The [Benchmarks](../benchmarks.md) table gives the measured rates:

- **Up to roughly ten thousand environments** on the measured GPU, adding
  environments is nearly free: one thread integrates one entity, and until the GPU is full
  the time per step barely grows with the batch. At the smallest sizes the
  step is set by launch overhead and kernel latency, not by the batch.
- **Beyond that**, throughput levels off at the GPU's float64 arithmetic
  rate (see [below](#float64)).

The simulator itself needs about 1.3 kB of device memory per entity.
Trajectories are larger: `save_states=True` stores 112 bytes per entity per
step, so a million entities over a thousand steps take 112 GB. Record only
what you need — final states with `simulate_monte_carlo()`, or your own
on-device reduction as in the loop above.

## `simulate()`, `step()` and your own graph

**`simulate(num_steps)`** captures one step (with the optional trajectory
writes and auto-reset) at the start of every call and replays it
`num_steps` times. Commands are fixed for the call, so prefer a few long
calls to many short ones.

**`step()`** launches its kernels from Python on every call. That is fine
for prototyping and when commands come from host code, but at small batches
the launch overhead dominates the step itself.
[Example 04](../examples/04_scalability.md) times this path.

### Capture your own loop

For closed-loop control at graph speed, capture your kernels and `step()`
together, as in the loop above. What the graph fixes and what stays live:

- **Fixed:** scalars passed by value to a kernel (`dt`, the gain above) and
  the identity of every array. Write into existing arrays (`assign`,
  `copy`, a kernel); an array you replace is not seen by the graph.
- **Live:** array contents — state, commands, and the vehicle, thruster and
  wheel configurations, which `set_config` writes in place. Change them
  between replays without recapturing.
- **Not allowed inside:** host reads and synchronization — any `.numpy()`,
  and `sim.reset()`, which validates on the host. `sim.reset_terminated()` runs on the device and can be captured.
- `sim.step_count` counts Python calls, so it does not advance on replay;
  `get_episode_steps()` does.

The RL environment captures its own step; see
[RL environments](rl.md#cuda-graphs).

## Keep data on the device

Every `.numpy()` copies to the host and waits for the GPU. Inside a loop
that cost dominates. Instead:

- **Command** with per-entity device arrays
  (`set_thrust_commands_per_entity`, `set_rw_torque_commands_per_entity`), written
  by a kernel or a zero-copy torch view, not a Python loop over entities.
- **Record** with `simulate(save_states=True)`, or your own kernel writing
  to a preallocated array.
- **Reduce** on the device — sum a cost, count terminations — and copy the
  result once.
- **Read** getters and termination reasons after the loop.

## Float64

All state and physics are float64; there is no float32 mode. On consumer
GPUs, which run float64 at a small fraction of their float32 rate, that
arithmetic sets the ceiling — see
[why float64 limits the step](../benchmarks.md#why-float64-limits-the-step-on-consumer-gpus).
The same kernels run on the CPU (`device="cpu"`), which is useful for
debugging and CI.

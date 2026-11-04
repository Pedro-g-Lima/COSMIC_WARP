# Scale up

The same simulator runs one environment or a million. Speed needs three
things: enough environments to fill the GPU, the step replayed as a CUDA
graph, and no host copies inside the loop.

## A complete loop

A rate-damping controller closes the loop on the device: one kernel reads
the state and writes per-entity wheel torques, the simulator steps, a second
kernel accumulates a diagnostic. The iteration is captured once and
replayed (on a CPU it runs eagerly).

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

## Choose the batch size

| Batch | Behaviour on the measured GPU ([Benchmarks](../benchmarks.md)) |
| --- | --- |
| up to ~10⁴ environments | nearly free: one thread per entity, and the time per step barely grows; the smallest batches are set by launch overhead |
| beyond | throughput levels off at the GPU's float64 rate ([why](../benchmarks.md#why-float64-limits-the-step)) |

The simulator needs about 1.3 kB of device memory per entity. Trajectories
are larger: `save_states=True` stores 112 bytes per entity per step (a
million entities over a thousand steps is 112 GB). Record only what you
need: final states with `simulate_monte_carlo()`, or an on-device reduction
as above.

## `simulate()`, `step()` and your own graph

- **`simulate(num_steps)`** captures one step (with the optional trajectory
  writes and auto-reset) at the start of every call and replays it.
  Commands are fixed for the call: prefer a few long calls.
- **`step()`** launches its kernels from Python each call; at small batches
  launch overhead dominates ([example 04](../examples/04_scalability.md)
  times it).

### Capture your own loop

For closed-loop control at graph speed, capture your kernels and `step()`
together, as above.

| Fixed by the capture | Live |
| --- | --- |
| scalars passed by value (`dt`, the gain above); the identity of every array — write into existing arrays (`assign`, `copy`, a kernel), never replace them | array contents: state, commands, and the vehicle, thruster and wheel configurations, which `set_config` writes in place |

- **Not allowed inside:** host reads and synchronization — any `.numpy()`,
  and `sim.reset()`, which validates on the host. `sim.reset_terminated()`
  runs on the device and can be captured.
- `sim.step_count` counts Python calls and does not advance on replay;
  `get_episode_steps()` does.

The RL environment captures its own step ([CUDA graphs](rl.md#cuda-graphs)).

## Keep data on the device

Every `.numpy()` copies and waits for the GPU. Instead:

- **Command** with per-entity device arrays
  (`set_thrust_commands_per_entity`, `set_rw_torque_commands_per_entity`)
  written by a kernel or a zero-copy torch view.
- **Record** with `simulate(save_states=True)` or your own kernel.
- **Reduce** on the device and copy the result once.
- **Read** getters and termination reasons after the loop.

All state and physics are float64; there is no float32 mode. The same
kernels run on the CPU (`device="cpu"`), for debugging and CI.

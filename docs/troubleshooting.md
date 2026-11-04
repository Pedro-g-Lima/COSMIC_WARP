# Troubleshooting

## `import warp` fails

Check that the virtual environment you installed into is active, then that
`python -c "import warp"` succeeds. Reinstalling the package reinstalls a
compatible `warp-lang` (the supported range is in the
[requirements](getting-started.md#requirements)).

## CUDA not available

The default device is `"cuda"`, and there is no automatic fallback. Without
a CUDA device, Warp raises `RuntimeError: "cuda" device requested but CUDA
is not supported by the hardware or driver`. Pass the device explicitly:

```python
from cosmic_warp.core.pipeline import SpacecraftSimulator

sim = SpacecraftSimulator(device="cpu")
```

For a YAML scenario, `create_simulator_from_yaml(path, device="cpu")`
overrides the file's `simulation.device`. The test suite also defaults to
CUDA; run it with `pytest --device cpu`.

## Every environment terminates on the first step

A terminated environment stays frozen until it is reset. By default only the
position limit (100 000 km) and the non-finite check are active; the
momentum and angular-momentum limits are off until you set them, and they
are absolute values, so a limit sized for one vehicle can end every episode
of a heavier one. Read the reasons:

```python
sim = SpacecraftSimulator(num_envs=2, mom_limit=1.0e5)
sim.vehicle.set_config(mass_dry=20.0)
sim.reset_manager.set_bounds(mass_bounds=(25.0, 25.0))
sim.reset()
sim.step()
print(sim.termination_checker.get_termination_reasons())  # {0: 2, 1: 2}: momentum
```

Size the limits for the vehicle — `SpacecraftSimulator(mom_limit=..., ang_mom_limit=...)`
or the [`termination` YAML section](guides/configuration.md#termination) —
or turn them off with `check_bounds=False`. The reason codes are listed in
[Termination](theory/execution.md#termination).

## Thrusters do nothing

- **No propellant.** A thruster produces no force while the mass is at or
  below the dry mass, even with `isp=0`. The default reset mass equals the
  default dry mass (1 kg), so the default vehicle carries none. Set
  `mass_bounds` above `mass_dry`, or randomize the propellant load.
- **Commands set before the reset.** Every reset zeroes the actuator
  commands of the environments it resets. Command after `sim.reset()`.
- **A negative command.** Thrust commands are clamped to `[0, 1]`: a
  thruster only pushes along its direction.
- **An unused slot.** A thruster whose direction is `(0, 0, 0)` is inert.

## `ValueError` at configuration time

Every configuration call checks its values on the host and raises
`ValueError` naming the field: a non-finite or negative value, an inertia
tensor that no rigid body has, an unknown YAML key, a quoted YAML number.
Two checks catch most first scenarios:

- `sim.reset()` rejects reset `mass_bounds` below a body's `mass_dry`: such
  entities would spawn without propellant. After raising `mass_dry`, raise
  `mass_bounds` too — see [Vehicles](guides/vehicle.md#mass-and-propellant).
- With `check_planet_impact`, `sim.reset()` rejects orbit bounds whose lowest
  perigee is at or below the impact radius (`planet_radius`, Earth's radius
  by default).

## The state becomes NaN or Inf

A NaN or Inf in any state component terminates its environment with reason
6, whatever the limits; the state is then frozen until reset. Usual causes:
a step too large for the fastest dynamics (fast spin, stiff wheel friction),
or a spacecraft placed at Earth's centre — reset boxes are in ECI, so a box
around the origin puts it there. Halve `dt` and compare, and check the
initial conditions. A NaN or Inf actuator command (often a diverged policy)
also ends the environment with reason 6, before the step, with the state
still finite: check the commands too.

## Slow first run

The first run compiles the kernels (the `Module ... load on device`
messages). Compiled modules are cached under `~/.cache/warp/<version>`, so
later runs start in seconds. The cache is per Warp version: expect one slow
run after upgrading `warp-lang`. To silence the messages, set
`wp.config.log_level = wp.LOG_WARNING` after `import warp as wp`.

## Slow stepping

- Run on a CUDA device. On the CPU, everything works but slowly.
- Use `simulate()` rather than a Python loop over `step()`: on a GPU it
  replays the step as one CUDA graph.
- Raise the batch size: throughput rises with the number of environments up
  to the GPU's capacity — see [Benchmarks](benchmarks.md).
- Keep data on the device; [Scale up](guides/scaling.md) has the rules.

## Getting help

Search or open an issue on
[GitHub](https://github.com/Pedro-g-Lima/COSMIC_WARP/issues), with the output of
`sim.metadata()` and a minimal script.

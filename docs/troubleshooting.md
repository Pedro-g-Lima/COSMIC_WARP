# Troubleshooting

## `import warp` fails

Check that the environment you installed into is active and that
`python -c "import warp"` works. Reinstalling the package reinstalls a
compatible `warp-lang` ([requirements](getting-started.md#requirements)).

## CUDA not available

The default device is `"cuda"`, with no automatic fallback. Without a CUDA
device Warp raises `RuntimeError: "cuda" device requested but CUDA is not
supported by the hardware or driver`. Pass the device:

```python
from cosmic_warp.core.pipeline import SpacecraftSimulator

sim = SpacecraftSimulator(device="cpu")
```

`create_simulator_from_yaml(path, device="cpu")` overrides a scenario's
device; `pytest --device cpu` runs the tests.

## Every environment terminates on the first step

By default only the position limit (100 000 km) and the non-finite check are
active. The momentum and angular-momentum limits are absolute values, so a
limit sized for one vehicle can end every episode of a heavier one. Read the
reasons:

```python
sim = SpacecraftSimulator(num_envs=2, mom_limit=1.0e5)
sim.vehicle.set_config(mass_dry=20.0)
sim.reset_manager.set_bounds(mass_bounds=(25.0, 25.0))
sim.reset()
sim.step()
print(sim.termination_checker.get_termination_reasons())  # {0: 2, 1: 2}: momentum
```

Size the limits for the vehicle (constructor or the
[`termination` YAML section](guides/configuration.md#termination)), or turn
them off with `check_bounds=False`. Reason codes:
[Termination](theory/execution.md#termination).

## Thrusters do nothing

- **No propellant.** A thruster produces nothing while the mass is at or
  below the dry mass, even with `isp=0`, and the default vehicle carries
  none. Set `mass_bounds` above `mass_dry`, or randomize the propellant.
- **Commands set before the reset.** Every reset zeroes the commands.
- **A negative command.** Thrust commands are clamped to `[0, 1]`.
- **An unused slot.** A thruster with direction `(0, 0, 0)` is inert.

## `ValueError` at configuration time

Every configuration call checks its values and names the bad field. The two
most common in a first scenario:

- `sim.reset()` rejects reset `mass_bounds` below a body's `mass_dry`. After
  raising `mass_dry`, raise `mass_bounds` too
  ([Vehicles](guides/vehicle.md#mass-and-propellant)).
- With `check_planet_impact`, `sim.reset()` rejects orbit bounds whose
  lowest perigee does not clear `planet_radius`.

## The state becomes NaN or Inf

The environment terminates with reason 6 and freezes until reset. Usual
causes: a step too large for the fastest dynamics (fast spin, stiff wheel
friction), a spacecraft placed at Earth's centre (reset boxes are in ECI),
or a NaN or Inf actuator command, often from a diverged policy. Halve `dt`
and compare; check the initial conditions and the commands.

## Slow first run

The first run compiles the kernels (the `Module ... load on device`
messages). The cache in `~/.cache/warp/<version>` makes later runs start in
seconds; expect one slow run after upgrading `warp-lang`. To silence the
messages, set `wp.config.log_level = wp.LOG_WARNING`.

## Slow stepping

- Run on a CUDA device.
- Use `simulate()`, not a Python loop over `step()`.
- Raise the batch size ([Benchmarks](benchmarks.md)).
- Keep data on the device ([Scale up](guides/scaling.md)).

## Getting help

Open an issue on [GitHub](https://github.com/Pedro-g-Lima/COSMIC_WARP/issues)
with the output of `sim.metadata()` and a minimal script.

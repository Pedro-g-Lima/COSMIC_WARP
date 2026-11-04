"""YAML scenario runner

Loads any YAML scenario file with `create_simulator_from_yaml`, runs it for a
number of steps, and prints the simulator metadata and summary statistics of
the final state: wall time and entity-steps per second, radius, speed and
mass ranges, and how many entities terminated, by reason.

The scenario files next to this script cover the configuration system; the
[configuration guide](../guides/configuration.md) documents every key.

| File | Scenario |
| --- | --- |
| `minimal_orbit.yaml` | one spacecraft on a fixed 400 km circular orbit |
| `monte_carlo_dispersion.yaml` | 4096 spacecraft on randomized LEO orbits |
| `multibody_orbit.yaml` | four bodies per environment on phased circular orbits |
| `rendezvous_chaser_target.yaml` | a passive target and a thruster-equipped chaser |
| `rl_attitude_training.yaml` | reaction-wheel attitude control, 2048 environments |

```bash
python examples/run_config.py examples/minimal_orbit.yaml --steps 5000
python examples/run_config.py examples/monte_carlo_dispersion.yaml --steps 50000 --monte-carlo
python examples/run_config.py examples/rl_attitude_training.yaml --steps 2000 --auto-reset
python examples/run_config.py examples/minimal_orbit.yaml --device cpu
```

Options: `config` (path to the YAML file), `--steps` (default 1000),
`--monte-carlo` (use `simulate_monte_carlo`, final states only),
`--auto-reset` (reset terminated environments inside the loop),
`--device` (overrides `simulation.device` in the file).
"""

import argparse
import os
import time


import numpy as np
import warp as wp

from cosmic_warp.core.config import create_simulator_from_yaml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a YAML scenario config")
    parser.add_argument("config", help="Path to YAML config")
    parser.add_argument("--steps", type=int, default=1000, help="Simulation steps")
    parser.add_argument("--monte-carlo", action="store_true",
                        help="Use simulate_monte_carlo (final states only)")
    parser.add_argument("--auto-reset", action="store_true",
                        help="Automatically reset terminated environments")
    parser.add_argument("--device", default=None,
                        help="Override simulation.device from the config (e.g. 'cpu')")
    return parser.parse_args()


def main() -> None:
    wp.init()
    args = _parse_args()

    sim = create_simulator_from_yaml(args.config, device=args.device)

    print("=" * 60)
    print(f"COSMIC WARP - {os.path.basename(args.config)}")
    print("=" * 60)
    for key, value in sim.metadata().items():
        print(f"  {key}: {value}")
    print(f"  steps: {args.steps}")

    start = time.perf_counter()
    if args.monte_carlo:
        sim.simulate_monte_carlo(args.steps)
    else:
        sim.simulate(args.steps, auto_reset_terminated=args.auto_reset)
    wp.synchronize()
    elapsed = time.perf_counter() - start

    states = sim.states.numpy()
    radius = np.linalg.norm(states["pos"], axis=1)
    speed = np.linalg.norm(states["mom"] / states["mass"][:, None], axis=1)
    status = sim.get_terminated().numpy()
    terminated = status["terminated"]

    print("-" * 60)
    print(f"  wall time: {elapsed:.2f} s "
          f"({args.steps * sim.num_entities / elapsed:.3e} entity-steps/s)")
    print(f"  sim time: {sim.step_count * sim.dt:.1f} s")
    print(f"  radius [m]: min {radius.min():.4e}  mean {radius.mean():.4e}  "
          f"max {radius.max():.4e}")
    print(f"  speed [m/s]: min {speed.min():.1f}  mean {speed.mean():.1f}  "
          f"max {speed.max():.1f}")
    print(f"  mass [kg]: min {states['mass'].min():.2f}  "
          f"max {states['mass'].max():.2f}")
    if args.auto_reset:
        # Terminated entities were reset inside the loop, so the final
        # status cannot tell how many terminated along the way
        print("  terminated: not counted with --auto-reset (reset in the loop)")
    else:
        print(f"  terminated: {int(terminated.sum())}/{sim.num_entities}")
    if terminated.any():
        reasons = status["reason"][terminated]
        names = {1: "position", 2: "momentum", 3: "angular momentum",
                 4: "mass", 5: "planet impact", 6: "non-finite state"}
        for code in np.unique(reasons):
            print(f"    reason {names.get(int(code), code)}: "
                  f"{int((reasons == code).sum())}")
    print("=" * 60)


if __name__ == "__main__":
    main()

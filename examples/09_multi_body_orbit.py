"""Multi-body orbit

Runs a multi-spacecraft scenario defined entirely in a YAML file and plots
every body of one environment: radius and speed over time, and the 3D orbits.
The script only loads the file with `create_simulator_from_yaml` and runs it;
dimensions, physics, per-body vehicles and initial orbits all come from the
configuration (see the [configuration guide](../guides/configuration.md)).

The default file, `examples/multibody_orbit.yaml`, defines:

- 3 environments x 4 bodies, RK4 with 0.01 s steps, gravity on, J2 off
- per-body dry masses (70-100 kg) and per-body reset mass bounds
- per-body circular equatorial initial orbits given as orbital elements
  (semi-major axes 7000-7060 km, phased 0.1 rad apart) and initial masses
  (100-130 kg), the same in every environment

```bash
python examples/09_multi_body_orbit.py
python examples/09_multi_body_orbit.py --config examples/rendezvous_chaser_target.yaml --steps 20000
```

Options: `--config`, `--env` (environment to plot), `--steps`, `--device`
(overrides `simulation.device` in the file), `--no-show`. The plots are
shown in windows and not saved, so with `--no-show` the script only prints
the run summary.

**What to look for:** each body keeps a constant radius and speed on its own
circular orbit; the outer bodies move more slowly, so the phasing between
them changes over time.
"""

import argparse
import os
import time


import numpy as np
import warp as wp
import matplotlib.pyplot as plt

from cosmic_warp.core.config import create_simulator_from_yaml


def _entity_id(env_id: int, body_id: int, num_bodies: int) -> int:
    return env_id * num_bodies + body_id


def _parse_args() -> argparse.Namespace:
    default_config = os.path.join(os.path.dirname(__file__), "multibody_orbit.yaml")

    parser = argparse.ArgumentParser(description="Multi-body orbit visualization")
    parser.add_argument("--config", default=default_config, help="Path to YAML config")
    parser.add_argument("--env", type=int, default=0, help="Environment index to plot")
    parser.add_argument("--steps", type=int, default=50000, help="Simulation steps")
    parser.add_argument("--device", default=None,
                        help="Override simulation.device from the config (e.g. 'cpu')")
    parser.add_argument("--no-show", action="store_true",
                         help="Skip interactive plot windows (useful for headless/CI runs)")
    return parser.parse_args()


def main() -> None:
    wp.init()
    args = _parse_args()

    sim = create_simulator_from_yaml(args.config, device=args.device)

    if args.env < 0 or args.env >= sim.num_envs:
        raise ValueError(f"env must be in [0, {sim.num_envs - 1}]")

    num_envs = sim.num_envs
    num_bodies = sim.num_bodies

    print("=" * 60)
    print("COSMIC WARP - Multi-Body Orbit Visualization")
    print("=" * 60)
    print(f"  Environments: {num_envs}")
    print(f"  Bodies per env: {num_bodies}")
    print(f"  Plot environment: {args.env}")
    print(f"  Time step: {sim.dt} s")
    print(f"  Steps: {args.steps}")

    start = time.time()
    trajectory = sim.simulate(num_steps=args.steps, save_states=True)
    elapsed = time.time() - start

    print(f"  Runtime: {elapsed:.2f} s")

    pos = trajectory["positions"].numpy()
    vel = trajectory["velocities"].numpy()
    t = trajectory["time"].numpy()

    body_indices = [_entity_id(args.env, body_id, num_bodies) for body_id in range(num_bodies)]

    # Plot altitude and speed for each body in the selected environment.
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    for body_id, idx in enumerate(body_indices):
        r_mag = np.linalg.norm(pos[:, idx, :], axis=1) / 1e3
        v_mag = np.linalg.norm(vel[:, idx, :], axis=1)
        ax1.plot(t, r_mag, label=f"body {body_id}")
        ax2.plot(t, v_mag, label=f"body {body_id}")

    ax1.set_title("Radius vs Time")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Radius [km]")
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Speed vs Time")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Speed [m/s]")
    ax2.grid(True, alpha=0.3)

    ax2.legend(loc="upper right", fontsize=8)

    # 3D trajectories for all bodies in the selected environment.
    fig2 = plt.figure(figsize=(7, 6))
    ax3 = fig2.add_subplot(111, projection="3d")

    for body_id, idx in enumerate(body_indices):
        ax3.plot(
            pos[:, idx, 0] / 1e6,
            pos[:, idx, 1] / 1e6,
            pos[:, idx, 2] / 1e6,
            linewidth=1.5,
            label=f"body {body_id}",
        )

    ax3.set_xlabel("X [1000 km]")
    ax3.set_ylabel("Y [1000 km]")
    ax3.set_zlabel("Z [1000 km]")
    ax3.set_title(f"3D Orbits (env {args.env})")
    ax3.set_box_aspect([1, 1, 1])
    ax3.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

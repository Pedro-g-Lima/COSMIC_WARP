r"""Domain randomization

Resamples vehicle parameters on the device at every reset with
`sim.randomizer`, then shows the physical effect of the spread: the same
thrust command produces a mass-dependent delta-v across the environments.

- 4096 environments with an 8 kg dry mass; each reset draws a uniform
  propellant load in [2, 6] kg (total mass 10-14 kg) and an inertia scale in
  [0.8, 1.2]
- two consecutive resets draw two independent mass populations
- every environment is then placed on the same circular 7000 km orbit and
  fires the same 1 N along-track thrust for 100 s (2000 steps of 0.05 s,
  symplectic integrator)

```bash
python examples/10_domain_randomization.py
python examples/10_domain_randomization.py --seed 1 --device cpu --num-envs 512
```

Options: `--num-envs`, `--num-steps`, `--dt`, `--device`, `--seed` (seeds the
randomizer). The plot is saved to the working directory; no window is opened.

![Domain randomization](../assets/examples/domain_randomization.png)

**What to look for:** the mass histograms of the two resets differ
(independent draws from the same bounds), and the delta-v of the identical
burn follows $\Delta v = F\,t/m$ across the sampled masses, from about
7.1 m/s at 14 kg to 10 m/s at 10 kg, within 0.2 % of the formula.
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from cosmic_warp.core.pipeline import SpacecraftSimulator
from cosmic_warp.physics.gravity import MU_EARTH
import warp as wp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="On-GPU domain randomization demonstration")
    parser.add_argument("--num-envs", type=int, default=4096, help="Number of parallel environments")
    parser.add_argument("--num-steps", type=int, default=2000, help="Simulation steps for the burn")
    parser.add_argument("--dt", type=float, default=0.05, help="Time step [s]")
    parser.add_argument("--device", default="cuda", help="Warp device (e.g. 'cuda' or 'cpu')")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the randomizer RNG")
    return parser.parse_args()


def main():
    """Run the domain randomization demonstration."""
    args = _parse_args()
    wp.init()
    print("=" * 70)
    print("COSMIC WARP - Domain Randomization")
    print("=" * 70)

    num_envs = args.num_envs
    dt = args.dt
    num_steps = args.num_steps  # burn duration = num_steps * dt
    mass_dry = 8.0    # kg

    sim = SpacecraftSimulator(
        num_envs=num_envs,
        integrator="symplectic",
        dt=dt,
        device=args.device,
        gravity=True,
        mom_limit=1.0e6,  # default 1e5 would terminate a 14 kg body in LEO
    )
    sim.vehicle.set_config(mass_dry=mass_dry)

    # Randomize propellant load and inertia at every reset
    sim.randomizer.set_bounds(
        propellant=(2.0, 6.0),      # kg -> total mass in [10, 14] kg
        inertia_scale=(0.8, 1.2),
    )
    if args.seed is not None:
        sim.randomizer.reseed(args.seed)

    # Two consecutive resets draw two independent mass populations
    sim.reset()
    masses_first = sim.states.numpy()["mass"].copy()
    sim.reset()
    masses = sim.states.numpy()["mass"].copy()
    print(f"\nSampled total mass: [{masses.min():.2f}, {masses.max():.2f}] kg "
          f"(dry {mass_dry} kg + propellant [2, 6] kg)")

    # Place every entity on the same circular orbit. Momenta scale with the
    # sampled masses; passing the masses back keeps the randomized values.
    r_orbit = 7.0e6
    mu = MU_EARTH
    v_circ = np.sqrt(mu / r_orbit)
    positions = np.tile([r_orbit, 0.0, 0.0], (num_envs, 1))
    velocities = np.tile([0.0, v_circ, 0.0], (num_envs, 1))
    momenta = masses[:, None] * velocities

    sim.reset_to_conditions(
        positions=wp.array(positions, dtype=wp.vec3d, device=sim.device),
        momenta=wp.array(momenta, dtype=wp.vec3d, device=sim.device),
        angular_momenta=wp.zeros(num_envs, dtype=wp.vec3d, device=sim.device),
        masses=wp.array(masses, dtype=wp.float64, device=sim.device),
    )

    # Single 1 N thruster at the CoM firing along-track (+y, identity
    # attitude). Same command everywhere; only the sampled mass differs.
    sim.thruster_manager.set_config(
        max_thrust=1.0,
        positions=[(0.0, 0.0, 0.0)] * 8,
        directions=[(0.0, 1.0, 0.0)] * 8,
    )
    sim.set_thrust_commands([1.0, 0.0, 0.0, 0.0])

    print(f"Burning 1 N along-track for {num_steps * dt:.0f} s "
          f"across {num_envs} environments...")
    sim.simulate(num_steps=num_steps)

    states = sim.states.numpy()
    v_final = np.linalg.norm(states["mom"], axis=1) / states["mass"]
    delta_v = v_final - v_circ
    burn_time = num_steps * dt

    # Plot: resampled mass populations + mass-dependent delta-v spread
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bins = np.linspace(mass_dry + 2.0, mass_dry + 6.0, 40)
    ax1.hist(masses_first, bins=bins, alpha=0.6, label="Reset 1")
    ax1.hist(masses, bins=bins, alpha=0.6, label="Reset 2")
    ax1.set_xlabel("Total mass [kg]")
    ax1.set_ylabel("Environments")
    ax1.set_title("Mass resampled at every reset")
    ax1.legend()
    ax1.grid(alpha=0.3)

    m_line = np.linspace(masses.min(), masses.max(), 100)
    ax2.scatter(masses, delta_v, s=2, alpha=0.3, label="Simulated")
    ax2.plot(m_line, 1.0 * burn_time / m_line, "r--", lw=2,
             label=r"$\Delta v = F\,t\,/\,m$")
    ax2.set_xlabel("Total mass [kg]")
    ax2.set_ylabel(r"$\Delta v$ [m/s]")
    ax2.set_title(f"Identical 1 N burn, {burn_time:.0f} s")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(os.getcwd(), 'domain_randomization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

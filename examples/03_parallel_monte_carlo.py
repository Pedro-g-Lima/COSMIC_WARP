"""Parallel Monte Carlo

Propagates 100 000 spacecraft at once from dispersed initial conditions and
summarizes where they end up. The initial states are drawn with NumPy on the
host; the propagation of the whole population is a single batched call,
`sim.simulate_monte_carlo`, which keeps only the final states (no trajectory
storage).

- Nominal orbit: 7000 km radius, circular speed, 100 kg
- Dispersions: radius 500 km and speed 100 m/s (1-sigma), starting latitude
  uniform in +/-30 deg, a small out-of-plane velocity
- Physics: gravity with J2, Euler integrator, termination on impact with a
  sphere 100 km above the Earth's surface
- Output: termination count, final radius/speed statistics, position and
  velocity dispersion, and throughput in spacecraft-steps per second

```bash
python examples/03_parallel_monte_carlo.py
python examples/03_parallel_monte_carlo.py --num-envs 10000 --seed 0
```

Options: `--num-envs`, `--num-steps`, `--dt`, `--device`, `--seed`
(seeds the NumPy initial-condition draw), `--no-show`.

![Parallel Monte Carlo](../assets/examples/monte_carlo_analysis.png)

**What to look for:** about 18 % of the population terminates. Most of these
start below the impact radius (a 500 km radial spread puts about 15 % of the
draws under it) and stay frozen where they terminated, which gives the spike
at about 6470 km in the radius histogram. The default run is 5e9
spacecraft-steps; on a desktop GPU (RTX 4070 Ti SUPER) it takes about 14 s.
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import time

from cosmic_warp.core.pipeline import SpacecraftSimulator
from cosmic_warp.physics.gravity import MU_EARTH
import warp as wp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel Monte Carlo orbital analysis")
    parser.add_argument("--num-envs", type=int, default=100000, help="Number of parallel spacecraft")
    parser.add_argument("--num-steps", type=int, default=50000, help="Simulation steps")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step [s]")
    parser.add_argument("--device", default="cuda", help="Warp device (e.g. 'cuda' or 'cpu')")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the initial-condition RNG")
    parser.add_argument("--no-show", action="store_true",
                         help="Skip interactive plot window (plot is still saved to disk)")
    return parser.parse_args()


def main():
    """Run Monte Carlo orbital analysis."""
    args = _parse_args()
    wp.init()
    print("=" * 70)
    print("COSMIC WARP - Parallel Monte Carlo Simulation")
    print("=" * 70)

    if args.seed is not None:
        np.random.seed(args.seed)

    # Configuration
    num_envs = args.num_envs
    num_steps = args.num_steps
    dt = args.dt

    print(f"\nConfiguration:")
    print(f"  Number of parallel environments: {num_envs}")
    print(f"  Simulation steps: {num_steps}")
    print(f"  Time step: {dt} s")
    print(f"  Total simulated time: {num_steps * dt} s")
    print(f"  Integrator: Euler")
    
    # Create simulator
    print(f"\nInitializing simulator...")
    sim = SpacecraftSimulator(
        num_envs=num_envs,
        integrator="euler",
        dt=dt,
        device=args.device,
        gravity=True,
        check_planet_impact=True,
        pos_limit=1.0e8,  # 100,000 km
        mom_limit=1.0e7,   # For 100 kg spacecraft: 100 km/s limit
        planet_radius=1.0e5 + 6.371e6  # 100 km + Earth radius
    )
    
    # Set up initial conditions with variations
    print(f"\nGenerating initial conditions...")
    
    # Nominal circular orbit
    r_orbit = 7.0e6  # 7000 km
    mu = MU_EARTH
    v_circular = np.sqrt(mu / r_orbit)
    mass = 100.0
    
    print(f"  Nominal orbit radius: {r_orbit/1e3:.1f} km")
    print(f"  Nominal circular velocity: {v_circular:.1f} m/s")
    
    # Generate random initial conditions
    positions = np.zeros((num_envs, 3))
    momenta = np.zeros((num_envs, 3))
    
    # Random angles
    theta = np.random.uniform(0, 2*np.pi, num_envs)
    phi = np.random.uniform(-np.pi/6, np.pi/6, num_envs)  # ±30° starting latitude
    
    # Radius variations: 500 km (1-sigma)
    r_variations = np.random.normal(0, 500e3, num_envs)
    r = r_orbit + r_variations
    
    # Velocity variations: 100 m/s (1-sigma)
    v_variations = np.random.normal(0, 100, num_envs)
    v = v_circular + v_variations
    
    # Set positions
    positions[:, 0] = r * np.cos(theta) * np.cos(phi)
    positions[:, 1] = r * np.sin(theta) * np.cos(phi)
    positions[:, 2] = r * np.sin(phi)
    
    # Set velocities (perpendicular to position)
    momenta[:, 0] = -v * np.sin(theta)
    momenta[:, 1] = v * np.cos(theta)
    momenta[:, 2] = np.random.normal(0, 10, num_envs) # Small z-velocity

    masses = np.full(num_envs, mass)
    
    print(f"  Position std dev: {np.std(r_variations)/1e3:.1f} km")
    print(f"  Velocity std dev: {np.std(v_variations):.1f} m/s")
    
    # Reset simulation with custom conditions
    sim.reset()
    # Convert to WARP arrays on the simulator device
    positions_wp = wp.array(positions, dtype=wp.vec3d, device=sim.device)
    momenta_wp = wp.array(momenta * mass, dtype=wp.vec3d, device=sim.device)
    mass_wp = wp.array(masses, dtype=wp.float64, device=sim.device)

    sim.reset_to_conditions(
        positions=positions_wp,
        momenta=momenta_wp,
        masses=mass_wp
    )
    
    # Run simulation with timing
    print(f"\nRunning simulation...")
    print(f"  Simulating {num_envs} spacecraft in parallel...")
    
    start_time = time.time()
    results = sim.simulate_monte_carlo(num_steps=num_steps)
    elapsed_time = time.time() - start_time
    
    print(f"\n✓ Simulation complete!")
    print(f"  Execution time: {elapsed_time:.2f} s")
    print(f"  Steps/second: {num_steps/elapsed_time:.0f}")
    print(f"  Env-steps/second: {num_envs * num_steps / elapsed_time:.0f}")
    print(f"  Speed vs realtime: {num_steps * dt / elapsed_time:.1f}x")
    
    # Extract final results
    pos_final = results['final_positions'].numpy()  # [num_envs, 3]
    vel_final = results['final_velocities'].numpy()
    termination_status = results['termination_status'].numpy()
    
    # Compute statistics
    print(f"\nPerforming statistical analysis...")
    
    # Count failures
    num_terminated = np.sum(termination_status['terminated'])
    failure_rate = num_terminated / num_envs * 100
    
    print(f"\nTermination Statistics:")
    print(f"  Total terminated: {num_terminated}/{num_envs} ({failure_rate:.1f}%)")
    
    # Final state statistics
    r_final = np.linalg.norm(pos_final, axis=1)
    v_final = np.linalg.norm(vel_final, axis=1)
    
    print(f"\nFinal State Statistics:")
    print(f"  Orbit radius:")
    print(f"    Mean: {np.mean(r_final)/1e3:.1f} km")
    print(f"    Std dev: {np.std(r_final)/1e3:.1f} km")
    print(f"    Min: {np.min(r_final)/1e3:.1f} km")
    print(f"    Max: {np.max(r_final)/1e3:.1f} km")
    print(f"\n  Velocity:")
    print(f"    Mean: {np.mean(v_final):.1f} m/s")
    print(f"    Std dev: {np.std(v_final):.1f} m/s")
    
    # Position dispersion
    pos_std = np.std(pos_final, axis=0)
    print(f"\n  Position Dispersion (1-sigma):")
    print(f"    X: {pos_std[0]/1e3:.1f} km")
    print(f"    Y: {pos_std[1]/1e3:.1f} km")
    print(f"    Z: {pos_std[2]/1e3:.1f} km")
    
    # Velocity dispersion
    vel_std = np.std(vel_final, axis=0)
    print(f"\n  Velocity Dispersion (1-sigma):")
    print(f"    Vx: {vel_std[0]:.1f} m/s")
    print(f"    Vy: {vel_std[1]:.1f} m/s")
    print(f"    Vz: {vel_std[2]:.1f} m/s")
    
    # Visualization
    print(f"\nGenerating visualizations...")
    
    fig = plt.figure(figsize=(12, 8))
    
    # Plot 1: Final position distribution (XY plane)
    ax1 = fig.add_subplot(221)
    
    # Separate terminated and active
    active_mask = ~termination_status['terminated']
    terminated_mask = termination_status['terminated']
    
    if np.any(active_mask):
        ax1.scatter(pos_final[active_mask, 0]/1e6, pos_final[active_mask, 1]/1e6,
                   s=5, alpha=0.6, c='green', label=f'Active ({np.sum(active_mask)})')
    if np.any(terminated_mask):
        ax1.scatter(pos_final[terminated_mask, 0]/1e6, pos_final[terminated_mask, 1]/1e6,
                   s=5, alpha=0.6, c='red', label=f'Terminated ({np.sum(terminated_mask)})')
    
    # Earth
    circle = plt.Circle((0, 0), 6.371, color='blue', alpha=0.2)
    ax1.add_patch(circle)
    
    ax1.set_xlabel('X [1000 km]')
    ax1.set_ylabel('Y [1000 km]')
    ax1.set_title(f'Final Position Distribution (n={num_envs})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Orbit radius histogram
    ax2 = fig.add_subplot(222)
    ax2.hist(r_final/1e3, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
    ax2.axvline(np.mean(r_final)/1e3, color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(r_final)/1e3:.1f} km')
    ax2.set_xlabel('Orbit radius [km]')
    ax2.set_ylabel('Count')
    ax2.set_title('Final Orbit Radius Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Velocity histogram
    ax3 = fig.add_subplot(223)
    ax3.hist(v_final, bins=50, alpha=0.7, edgecolor='black', color='orange')
    ax3.axvline(np.mean(v_final), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(v_final):.1f} m/s')
    ax3.set_xlabel('Velocity [m/s]')
    ax3.set_ylabel('Count')
    ax3.set_title('Final Velocity Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Position dispersion (3D scatter)
    from mpl_toolkits.mplot3d import Axes3D
    ax4 = fig.add_subplot(224, projection='3d')
    
    if np.any(active_mask):
        ax4.scatter(pos_final[active_mask, 0]/1e6, pos_final[active_mask, 1]/1e6, pos_final[active_mask, 2]/1e6,
                   s=2, alpha=0.5, c='green', label='Active')
    if np.any(terminated_mask):
        ax4.scatter(pos_final[terminated_mask, 0]/1e6, pos_final[terminated_mask, 1]/1e6, pos_final[terminated_mask, 2]/1e6,
                   s=2, alpha=0.5, c='red', label='Terminated')
    
    # Earth
    u = np.linspace(0, 2*np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = 6.371 * np.outer(np.cos(u), np.sin(v))
    y = 6.371 * np.outer(np.sin(u), np.sin(v))
    z = 6.371 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax4.plot_surface(x, y, z, color='blue', alpha=0.2)
    
    ax4.set_xlabel('X [1000 km]')
    ax4.set_ylabel('Y [1000 km]')
    ax4.set_zlabel('Z [1000 km]')
    ax4.set_title('3D Final Position Scatter')
    ax4.legend()
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(os.getcwd(), 'monte_carlo_analysis.png')
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")

    if not args.no_show:
        plt.show()
    
    # Performance summary
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"\nTotal computations: {num_envs * num_steps:,} spacecraft-steps")
    print(f"Wall time: {elapsed_time:.2f} seconds")
    print(f"Throughput: {num_envs * num_steps / elapsed_time:,.0f} spacecraft-steps/second")
    
    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

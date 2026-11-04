"""Integrator comparison

Runs the same two-body orbit with each integrator (`euler`, `rk4`,
`symplectic`) at the same step size and compares trajectory, radius,
specific-energy error and wall time in one figure.

The spacecraft starts at 7000 km radius, 1000 m/s faster than circular, so
the orbit is eccentric (apogee near 12 500 km). J2 is off, so the specific
orbital energy is an exact invariant and its drift measures the integration
error. The default run is 500 000 steps of 0.01 s (5000 s, about half an
orbit) per integrator.

```bash
python examples/02_integrator_comparison.py
```

Options: `--num-steps`, `--dt`, `--device`, `--no-show`.

![Integrator comparison](../assets/examples/integrator_comparison.png)

**What to look for:** at this step size the three trajectories overlap; the
difference shows in the energy panel. Euler's energy error grows steadily, to
about 0.005 % after 5000 s, while RK4 and symplectic stay at round-off level.
The performance panel shows the cost per step; in the run shown Euler is the
cheapest and RK4 the most expensive.
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import time
import warp as wp

from cosmic_warp.core.pipeline import SpacecraftSimulator
from cosmic_warp.physics.gravity import RE_EARTH
from cosmic_warp.physics.gravity import MU_EARTH


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare numerical integrators")
    parser.add_argument("--num-steps", type=int, default=500000, help="Simulation steps")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step [s], used by all integrators")
    parser.add_argument("--device", default="cuda", help="Warp device (e.g. 'cuda' or 'cpu')")
    parser.add_argument("--no-show", action="store_true",
                         help="Skip interactive plot window (plot is still saved to disk)")
    return parser.parse_args()


def compute_orbital_energy(positions, velocities, mu=MU_EARTH):
    """Compute specific orbital energy."""
    r = np.linalg.norm(positions, axis=-1)
    v = np.linalg.norm(velocities, axis=-1)
    kinetic = 0.5 * v**2
    potential = -mu / r
    return kinetic + potential


def run_simulation(integrator, dt, num_steps, device):
    """Run simulation with specified integrator."""

    sim = SpacecraftSimulator(
        num_envs=1,
        integrator=integrator,
        dt=dt,
        device=device,
        gravity=True,
        # Point-mass energy is conserved only without J2
        j2_perturbation=False,
        pos_limit=1.0e8,  # 100,000 km
        mom_limit=1.0e7   # 100 km/s
    )
    
    # Start at 7000 km, 1000 m/s faster than circular (eccentric orbit)
    r_orbit = 7.0e6
    mu = MU_EARTH
    v_circular = np.sqrt(mu / r_orbit)
    
    positions = np.array([[r_orbit, 0.0, 0.0]])
    momenta = np.array([[0.0, v_circular + 1000.0, 0.0]])
    mass = 100.0

    # Convert to WARP arrays on the simulator device
    positions_wp = wp.array(positions, dtype=wp.vec3d, device=sim.device)
    momenta_wp = wp.array(momenta * mass, dtype=wp.vec3d, device=sim.device)
    mass_wp = wp.array([mass], dtype=wp.float64, device=sim.device)

    sim.reset()
    sim.reset_to_conditions(
        positions=positions_wp,
        momenta=momenta_wp,
        masses=mass_wp
    )

    # Time the simulation
    start_time = time.time()
    trajectory = sim.simulate(num_steps=num_steps, save_states=True)
    elapsed_time = time.time() - start_time

    # Convert trajectory arrays to numpy for downstream analysis/plotting
    trajectory['positions'] = trajectory['positions'].numpy()
    trajectory['velocities'] = trajectory['velocities'].numpy()
    trajectory['time'] = trajectory['time'].numpy()

    info = {
        'dt': sim.dt,
        'integrator_order': {"euler": 1, "symplectic": 2, "rk4": 4}[integrator],
        'sim_time': sim.step_count * sim.dt,
    }
    return trajectory, elapsed_time, info


def main():
    """Compare integrators."""
    args = _parse_args()
    wp.init()
    print("=" * 70)
    print("COSMIC WARP - Integrator Comparison Example")
    print("=" * 70)

    # Configuration
    num_steps = args.num_steps
    dt = args.dt

    integrators = ['euler', 'rk4', 'symplectic']
    results = {}

    print(f"\nRunning simulations with {num_steps} steps...")
    print(f"Note: All integrators use dt={dt}s")
    print("-" * 70)

    # Run simulations
    for integrator in integrators:
        print(f"\nRunning {integrator.upper()} integrator (dt={dt}s)...", end=" ")

        trajectory, elapsed, info = run_simulation(integrator, dt, num_steps, args.device)
        
        print(f"Done! ({elapsed:.3f}s)")
        print(f"  Integration order: {info['integrator_order']}")
        print(f"  Simulated time: {info['sim_time']:.2f}s")
        
        results[integrator] = {
            'trajectory': trajectory,
            'elapsed': elapsed,
            'info': info,
            'dt': dt
        }
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {'euler': 'red', 'rk4': 'blue', 'symplectic': 'green'}
    
    # Plot 1: Trajectories (XY plane)
    ax = axes[0, 0]
    for integrator in integrators:
        traj = results[integrator]['trajectory']
        pos = traj['positions'][:, 0, :]  # First environment
        ax.plot(pos[:, 0]/1e6, pos[:, 1]/1e6, 
                label=integrator.upper(), 
                color=colors[integrator],
                linewidth=2, alpha=0.7)
    
    # Plot Earth
    circle = plt.Circle((0, 0), 6.371, color='blue', alpha=0.2)
    ax.add_patch(circle)
    
    ax.set_xlabel('X [1000 km]')
    ax.set_ylabel('Y [1000 km]')
    ax.set_title('Orbital Trajectories (XY Plane)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Plot 2: Altitude over time
    ax = axes[0, 1]
    for integrator in integrators:
        traj = results[integrator]['trajectory']
        pos = traj['positions'][:, 0, :]
        time_axis = traj['time']
        altitude = (np.sqrt(np.sum(pos**2, axis=1)) - RE_EARTH) / 1e3
        
        ax.plot(time_axis, altitude, 
                label=integrator.upper(),
                color=colors[integrator],
                linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Altitude [km]')
    ax.set_title('Altitude vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Energy conservation
    ax = axes[1, 0]
    for integrator in integrators:
        traj = results[integrator]['trajectory']
        pos = traj['positions'][:, 0, :]
        vel = traj['velocities'][:, 0, :]
        time_axis = traj['time']
        
        energy = compute_orbital_energy(pos, vel)
        energy_error = (energy - energy[0]) / abs(energy[0]) * 100
        
        ax.plot(time_axis, energy_error,
                label=integrator.upper(),
                color=colors[integrator],
                linewidth=2, alpha=0.7)
        
        print(f"\n{integrator.upper()} Energy Conservation:")
        print(f"  Initial energy: {energy[0]:.2f} J/kg")
        print(f"  Final energy: {energy[-1]:.2f} J/kg")
        print(f"  Relative error: {energy_error[-1]:.6f}%")
    
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Energy Error [%]')
    ax.set_title('Energy Conservation Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Performance comparison
    ax = axes[1, 1]
    
    integrator_names = [i.upper() for i in integrators]
    elapsed_times = [results[i]['elapsed'] for i in integrators]
    steps_per_sec = [num_steps / results[i]['elapsed'] for i in integrators]
    
    x = np.arange(len(integrators))
    width = 0.35
    
    ax2 = ax.twinx()
    
    bars1 = ax.bar(x - width/2, elapsed_times, width, 
                   label='Elapsed Time', color='skyblue', alpha=0.7)
    bars2 = ax2.bar(x + width/2, steps_per_sec, width,
                    label='Steps/Second', color='orange', alpha=0.7)
    
    # Add value labels on top of each bar
    for i, (bar, val) in enumerate(zip(bars1, elapsed_times)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}s',
                ha='center', va='bottom', fontsize=9, color='darkblue')
    
    for i, (bar, val) in enumerate(zip(bars2, steps_per_sec)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val:.0f}',
                 ha='center', va='bottom', fontsize=9, color='darkorange')
    
    ax.set_xlabel('Integrator')
    ax.set_ylabel('Elapsed Time [s]', color='skyblue')
    ax2.set_ylabel('Steps/Second', color='orange')
    ax.set_title('Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(integrator_names)
    ax.tick_params(axis='y', labelcolor='skyblue')
    ax2.tick_params(axis='y', labelcolor='orange')
    
    # Add legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    # Performance summary
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    
    for integrator in integrators:
        info = results[integrator]
        print(f"\n{integrator.upper()}:")
        print(f"  Time step: {info['dt']}s")
        print(f"  Elapsed time: {info['elapsed']:.3f}s")
        print(f"  Steps/second: {num_steps/info['elapsed']:.1f}")
        print(f"  Simulated time: {info['info']['sim_time']:.2f}s")
        print(f"  Speed ratio: {info['info']['sim_time']/info['elapsed']:.1f}x realtime")
    
    # Save figure
    output_path = os.path.join(os.getcwd(), 'integrator_comparison.png')
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")

    if not args.no_show:
        plt.show()
    
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print("\n1. Symplectic integrator has excellent long-term energy conservation")
    print("2. RK4 provides high local accuracy but energy may drift over time")
    print("3. Symplectic is recommended for long-duration orbital simulations")
    print("4. RK4 is better for short simulations requiring high accuracy")
    print("5. Euler requires very small dt and is mainly for debugging")
    
    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

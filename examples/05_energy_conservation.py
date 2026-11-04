"""Energy conservation

Propagates one spacecraft on a circular 7000 km two-body orbit (J2 off, so
the total mechanical energy is an exact invariant) with each integrator and
tracks the energy error over a long horizon:

- Euler (1st order)
- RK4 (4th order)
- Symplectic (2nd order, structure-preserving)

The default run is 1 000 000 steps of 1 s per integrator, about 278 hours or
170 orbits. The energy is sampled on the host every `--sample-interval` steps
and compared in four panels: total energy, relative error, a close-up of RK4
against symplectic, and the final drift on a log scale.

```bash
python examples/05_energy_conservation.py
python examples/05_energy_conservation.py --num-steps 100000 --device cpu
```

Options: `--num-steps`, `--dt`, `--device`, `--sample-interval`, `--no-show`.

![Energy conservation](../assets/examples/energy_conservation_comparison.png)

**What to look for:** Euler's energy error grows secularly, to about +50 %
by the end of the run (the orbit spirals outward). RK4 and symplectic both
stay within about 1e-12 relative error, at round-off level: at this step size
neither drifts measurably. The symplectic error oscillates within a band,
the behaviour that keeps it bounded over very long runs; RK4 has no
such guarantee, and its error grows with the step size.
"""

import argparse
import os

import warp as wp
import matplotlib.pyplot as plt
import numpy as np

from cosmic_warp.core.pipeline import SpacecraftSimulator
from cosmic_warp.physics.gravity import MU_EARTH


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Energy conservation comparison across integrators")
    parser.add_argument("--num-steps", type=int, default=1000000, help="Simulation steps per integrator")
    parser.add_argument("--dt", type=float, default=1.0, help="Time step [s]")
    parser.add_argument("--device", default="cuda", help="Warp device (e.g. 'cuda' or 'cpu')")
    parser.add_argument("--sample-interval", type=int, default=100,
                         help="Steps between energy samples")
    parser.add_argument("--no-show", action="store_true",
                         help="Skip interactive plot window (plot is still saved to disk)")
    return parser.parse_args()


def compute_energy(pos, vel, mass=100.0):
    """
    Compute total mechanical energy (kinetic + potential).
    
    Args:
        pos: Position array [N, 3]
        vel: Velocity array [N, 3]
        mass: Spacecraft mass [kg]
    
    Returns:
        Total energy array [N]
    """
    mu = MU_EARTH  # the value the dynamics use
    
    # Kinetic energy: 0.5 * m * v^2
    v_mag_sq = np.sum(vel**2, axis=1)
    KE = 0.5 * mass * v_mag_sq
    
    # Potential energy: -mu * m / r
    r_mag = np.sqrt(np.sum(pos**2, axis=1))
    PE = -mu * mass / r_mag
    
    # Total energy
    E = KE + PE
    
    return E, KE, PE


def main():
    """Run the energy comparison for all integrators."""
    args = _parse_args()

    print("=" * 60)
    print("ENERGY CONSERVATION COMPARISON - All Integrators")
    print("=" * 60)

    # Configuration
    num_envs = 1  # Single spacecraft for clarity
    num_steps = args.num_steps  # Long horizon to expose energy drift
    dt = args.dt
    device = args.device

    # Set up initial conditions
    r_orbit = 7.0e6  # 7000 km orbital radius
    mu = MU_EARTH
    v_circular = np.sqrt(mu / r_orbit)
    mass = 100.0
    
    print(f"\nSimulation setup:")
    print(f"  Time step: {dt} s")
    print(f"  Total steps: {num_steps}")
    print(f"  Total time: {num_steps * dt / 3600:.2f} hours")
    print(f"\nInitial conditions:")
    print(f"  Orbital radius: {r_orbit/1e3:.1f} km")
    print(f"  Circular velocity: {v_circular:.1f} m/s")
    print(f"  Spacecraft mass: {mass} kg")
    
    # Sample interval
    sample_interval = args.sample_interval
    num_samples = num_steps // sample_interval
    
    # Storage for all integrators
    integrators = ['euler', 'rk4', 'symplectic']
    colors = ['red', 'green', 'blue']
    results = {}
    
    # Run simulation for each integrator
    for integrator in integrators:
        print(f"\n" + "-" * 60)
        print(f"Running {integrator.upper()} integrator...")
        print("-" * 60)
        
        # Create simulator
        sim = SpacecraftSimulator(
            num_envs=num_envs,
            integrator=integrator,
            dt=dt,
            gravity=True,
            # Point-mass energy is conserved only without J2
            j2_perturbation=False,
            pos_limit=1.0e10,
            mom_limit=1.0e10,
            ang_mom_limit=1.0e10,
            device=device
        )
        
        # Initialize state
        positions = wp.array([[r_orbit, 0.0, 0.0]], dtype=wp.vec3d, device=device)
        momenta = wp.array([[0.0, v_circular * mass, 0.0]], dtype=wp.vec3d, device=device)
        masses = wp.array([mass], dtype=wp.float64, device=device)
        
        sim.reset()
        sim.reset_to_conditions(positions=positions, momenta=momenta, masses=masses)
        
        # Storage arrays
        energies = np.zeros(num_samples)
        kinetic = np.zeros(num_samples)
        potential = np.zeros(num_samples)
        times = np.zeros(num_samples)
        
        # Run simulation
        for i in range(num_samples):
            # Step forward
            for _ in range(sample_interval):
                sim.step()
            
            # Extract current state
            pos = sim.get_positions().numpy()
            vel = sim.get_velocities().numpy()
            
            # Compute energy
            E, KE, PE = compute_energy(pos, vel, mass=mass)
            
            energies[i] = E[0]
            kinetic[i] = KE[0]
            potential[i] = PE[0]
            times[i] = (i + 1) * sample_interval * dt
        
        # Store results
        results[integrator] = {
            'energies': energies,
            'kinetic': kinetic,
            'potential': potential,
            'times': times
        }
        
        # Analyze energy conservation
        E_initial = energies[0]
        E_final = energies[-1]
        E_drift = E_final - E_initial
        E_drift_percent = (E_drift / abs(E_initial)) * 100
        E_variation = np.max(energies) - np.min(energies)
        E_mean_error = np.mean(energies) - E_initial
        E_std = np.std(energies)
        
        print(f"\nEnergy Analysis ({integrator}):")
        print(f"  Initial energy: {E_initial:.6e} J")
        print(f"  Final energy: {E_final:.6e} J")
        print(f"  Energy drift: {E_drift:.6e} J ({E_drift_percent:.4f}%)")
        print(f"  Energy variation: {E_variation:.6e} J")
        print(f"  Mean absolute error: {E_mean_error:.6e} J")
        print(f"  Energy std dev: {E_std:.6e} J")
        print(f"  Relative error: {E_std / abs(E_initial) * 100:.6f}%")
        
        # Store statistics
        results[integrator]['stats'] = {
            'E_initial': E_initial,
            'E_final': E_final,
            'E_drift': E_drift,
            'E_drift_percent': E_drift_percent,
            'E_variation': E_variation,
            'E_mean_error': E_mean_error,
            'E_std': E_std,
            'relative_error': E_std / abs(E_initial) * 100
        }
    
    # Create comparison plots
    print(f"\n" + "=" * 60)
    print("Creating comparison plots...")
    print("=" * 60)
    
    fig = plt.figure(figsize=(14, 10))
    
    # Plot 1: Total energy comparison
    ax1 = plt.subplot(2, 2, 1)
    for integrator, color in zip(integrators, colors):
        times = results[integrator]['times'] / 3600
        energies = results[integrator]['energies']
        ax1.plot(times, energies, color=color, linewidth=2, label=integrator.upper(), alpha=0.8)
    
    E_initial = results['symplectic']['stats']['E_initial']
    ax1.axhline(y=E_initial, color='black', linestyle='--', alpha=0.5, linewidth=1, label='Initial')
    ax1.set_xlabel('Time [hours]')
    ax1.set_ylabel('Total Energy [J]')
    ax1.set_title('Total Energy vs Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Energy error (relative to initial)
    ax2 = plt.subplot(2, 2, 2)
    for integrator, color in zip(integrators, colors):
        times = results[integrator]['times'] / 3600
        energies = results[integrator]['energies']
        E_init = results[integrator]['stats']['E_initial']
        error_percent = ((energies - E_init) / abs(E_init)) * 100
        ax2.plot(times, error_percent, color=color, linewidth=2, label=integrator.upper(), alpha=0.8)
    
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax2.set_xlabel('Time [hours]')
    ax2.set_ylabel('Energy Error [%]')
    ax2.set_title('Relative Energy Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: RK4 vs Symplectic relative error comparison
    ax3 = plt.subplot(2, 2, 3)
    for integrator, color in zip(['rk4', 'symplectic'], ['green', 'blue']):
        times = results[integrator]['times'] / 3600
        energies = results[integrator]['energies']
        E_init = results[integrator]['stats']['E_initial']
        error_percent = ((energies - E_init) / abs(E_init)) * 100
        ax3.plot(times, error_percent, color=color, linewidth=2, label=integrator.upper(), alpha=0.8)
    
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax3.set_xlabel('Time [hours]')
    ax3.set_ylabel('Energy Error [%]')
    ax3.set_title('RK4 vs Symplectic Error Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Energy drift comparison bar chart
    ax4 = plt.subplot(2, 2, 4)
    drift_values = [abs(results[integ]['stats']['E_drift_percent']) for integ in integrators]
    bars = ax4.bar(integrators, drift_values, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('|Energy Drift| [%]')
    ax4.set_title('Energy Drift Comparison')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, drift_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    output_path = os.path.join(os.getcwd(), 'energy_conservation_comparison.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Summary comparison table
    print(f"\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(f"{'Integrator':<15} {'Drift [%]':<15} {'Rel. Error [%]':<15}")
    print("-" * 60)

    for integrator in integrators:
        stats = results[integrator]['stats']
        drift = abs(stats['E_drift_percent'])
        rel_error = stats['relative_error']
        print(f"{integrator.upper():<15} {drift:<15.6f} {rel_error:<15.6f}")

    print("=" * 60)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

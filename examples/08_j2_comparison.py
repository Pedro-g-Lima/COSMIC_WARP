"""J2 comparison

Propagates the same orbit twice, with and without the J2 term of the Earth's
gravity (its oblateness), and plots the two side by side: 3D and planar views
of the trajectories, altitude, inclination and the distance between the two
runs.

The orbit is circular at 400 km altitude and 45 deg inclination, starting at
the ascending node; the default run is 10 000 steps of 1 s with RK4, about
2.8 hours or 1.8 orbits.

```bash
python examples/08_j2_comparison.py
python examples/08_j2_comparison.py --num-steps 86400
```

Options: `--num-steps`, `--dt`, `--device`, `--no-show`.

![J2 comparison](../assets/examples/j2_comparison.png)

**What to look for:** without J2 the altitude and inclination stay constant.
With J2 the altitude oscillates by about 13 km once per orbit and the
inclination by about 0.04 deg twice per orbit, and the two runs drift apart
by about 150 km by the end. The secular J2 effect, regression of the
ascending node, is too small to see in these plots over 1.8 orbits: it is
about -0.7 deg over the default run (roughly -6 deg per day at this
inclination).
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from cosmic_warp.core.pipeline import SpacecraftSimulator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="J2 perturbation comparison")
    parser.add_argument("--num-steps", type=int, default=10000, help="Simulation steps")
    parser.add_argument("--dt", type=float, default=1.0, help="Time step [s]")
    parser.add_argument("--device", default="cuda", help="Warp device (e.g. 'cuda' or 'cpu')")
    parser.add_argument("--no-show", action="store_true",
                         help="Skip interactive plot window (plot is still saved to disk)")
    return parser.parse_args()


def setup_leo_orbit(sim: SpacecraftSimulator):
    """Set up a Low Earth Orbit with 45-degree inclination."""
    # LEO altitude: 400 km
    r_leo = 6.3781363e6 + 400e3  # Earth radius + altitude [m]

    # Circular velocity
    mu = 3.986004415e14  # Earth's gravitational parameter
    v_circ = np.sqrt(mu / r_leo)

    # Inclination angle (45 degrees for strong J2 effect)
    inclination = np.radians(45.0)

    # Set initial state for all environments
    states = sim.states.numpy()
    for i in range(sim.num_envs):
        # Position: start at ascending node (on equatorial plane)
        states[i]['pos'] = [r_leo, 0.0, 0.0]

        # Velocity: inclined orbit - rotated out of XY plane
        # For an inclined circular orbit starting at ascending node:
        # velocity is perpendicular to position, inclined to equatorial plane
        v_y = v_circ * np.cos(inclination)
        v_z = v_circ * np.sin(inclination)
        states[i]['mom'] = [0.0, 100.0 * v_y, 100.0 * v_z]  # 100kg spacecraft

        # Orientation: identity quaternion
        states[i]['quat'] = [0.0, 0.0, 0.0, 1.0]
        states[i]['ang_mom'] = [0.0, 0.0, 0.0]
        states[i]['mass'] = 100.0  # kg

    # CRITICAL: Copy the modified numpy array back to device
    sim.states.assign(states)


def compute_altitude(pos):
    """Compute altitude above Earth's surface."""
    radius = np.linalg.norm(pos, axis=1)
    return (radius - 6.3781363e6) / 1000.0  # km


def compute_inclination(pos, vel):
    """Compute orbital inclination over time."""
    # Angular momentum vector h = r × v
    h = np.cross(pos, vel)
    h_mag = np.linalg.norm(h, axis=1)
    # Inclination is angle between h and z-axis
    h_z = h[:, 2]
    inc = np.arccos(np.clip(h_z / h_mag, -1, 1))
    return np.degrees(inc)


def main():
    args = _parse_args()

    print("=" * 60)
    print("J2 Perturbation Comparison Example")
    print("=" * 60)

    # Simulation parameters
    num_steps = args.num_steps
    dt = args.dt
    total_time = num_steps * dt

    print(f"\nSimulation parameters:")
    print(f"  Orbit inclination: 45 degrees")
    print(f"  Time step: {dt} s")
    print(f"  Total steps: {num_steps}")
    print(f"  Total time: {total_time/3600:.2f} hours")

    # Run simulation WITHOUT J2
    print("\n[1/2] Running simulation WITHOUT J2 perturbation...")
    sim_no_j2 = SpacecraftSimulator(
        num_envs=1,
        dt=dt,
        device=args.device,
        integrator="rk4",
        gravity=True,
        j2_perturbation=False,  # Disable J2
        check_bounds=False
    )
    # Verify the physics switches
    meta = sim_no_j2.metadata()
    print(f"  Flags: gravity={meta['gravity']}, j2_perturbation={meta['j2_perturbation']}")
    setup_leo_orbit(sim_no_j2)
    traj_no_j2 = sim_no_j2.simulate(num_steps, save_states=True)

    # Run simulation WITH J2
    print("[2/2] Running simulation WITH J2 perturbation...")
    sim_with_j2 = SpacecraftSimulator(
        num_envs=1,
        dt=dt,
        device=args.device,
        integrator="rk4",
        gravity=True,
        j2_perturbation=True,  # Enable J2
        check_bounds=False
    )
    # Verify the physics switches
    meta = sim_with_j2.metadata()
    print(f"  Flags: gravity={meta['gravity']}, j2_perturbation={meta['j2_perturbation']}")
    setup_leo_orbit(sim_with_j2)
    traj_with_j2 = sim_with_j2.simulate(num_steps, save_states=True)

    print("\nSimulations complete!")

    # Extract trajectories
    # Warp arrays with vec3 dtype need special handling
    pos_no_j2_raw = traj_no_j2['positions'].numpy()  # Shape: (steps, envs)
    pos_with_j2_raw = traj_with_j2['positions'].numpy()

    # Convert vec3 to regular numpy arrays (steps, 3)
    pos_no_j2 = np.array([[p[0], p[1], p[2]] for p in pos_no_j2_raw[:, 0]])
    pos_with_j2 = np.array([[p[0], p[1], p[2]] for p in pos_with_j2_raw[:, 0]])

    # Debug: Check if positions are actually different
    print(f"\nDiagnostics:")
    print(f"  First position (no J2): {pos_no_j2[0]}")
    print(f"  First position (with J2): {pos_with_j2[0]}")
    print(f"  Last position (no J2): {pos_no_j2[-1]/1e6} [1000 km]")
    print(f"  Last position (with J2): {pos_with_j2[-1]/1e6} [1000 km]")
    print(f"  Position vector difference at end: {(pos_with_j2[-1] - pos_no_j2[-1])} [m]")

    alt_no_j2 = compute_altitude(pos_no_j2)
    alt_with_j2 = compute_altitude(pos_with_j2)

    # Get velocities for inclination calculation
    vel_no_j2 = np.array([[v[0], v[1], v[2]] for v in traj_no_j2['velocities'].numpy()[:, 0]])
    vel_with_j2 = np.array([[v[0], v[1], v[2]] for v in traj_with_j2['velocities'].numpy()[:, 0]])

    inc_no_j2 = compute_inclination(pos_no_j2, vel_no_j2)
    inc_with_j2 = compute_inclination(pos_with_j2, vel_with_j2)

    # Compute position difference
    pos_diff = np.linalg.norm(pos_with_j2 - pos_no_j2, axis=1)

    print(f"\nResults after {total_time/3600:.2f} hours:")
    print(f"  Position difference: {pos_diff[-1]/1000:.2f} km")
    print(f"  Max position difference: {pos_diff.max()/1000:.2f} km")
    print(f"  Altitude drift (no J2): {alt_no_j2[-1] - alt_no_j2[0]:.3f} km")
    print(f"  Altitude drift (with J2): {alt_with_j2[-1] - alt_with_j2[0]:.3f} km")
    print(f"  Inclination change (no J2): {inc_no_j2[-1] - inc_no_j2[0]:.4f} degrees")
    print(f"  Inclination change (with J2): {inc_with_j2[-1] - inc_with_j2[0]:.4f} degrees")

    # Generate plots
    time_hours = traj_no_j2['time'].numpy() / 3600.0

    fig = plt.figure(figsize=(16, 12))

    # Plot 1: 3D trajectories
    ax = fig.add_subplot(2, 3, 1, projection='3d')
    ax.plot(pos_no_j2[:, 0]/1e6, pos_no_j2[:, 1]/1e6, pos_no_j2[:, 2]/1e6,
            'b-', linewidth=1.5, label='No J2', alpha=0.8)
    ax.plot(pos_with_j2[:, 0]/1e6, pos_with_j2[:, 1]/1e6, pos_with_j2[:, 2]/1e6,
            'r-', linewidth=1.5, label='With J2', alpha=0.8)
    # Add Earth sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_earth = 6.378 * np.outer(np.cos(u), np.sin(v))
    y_earth = 6.378 * np.outer(np.sin(u), np.sin(v))
    z_earth = 6.378 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_earth, y_earth, z_earth, color='cyan', alpha=0.3)
    ax.set_xlabel('X [1000 km]')
    ax.set_ylabel('Y [1000 km]')
    ax.set_zlabel('Z [1000 km]')
    ax.set_title('3D Trajectories (45° Inclination)')
    ax.legend()
    ax.set_box_aspect([1,1,1])

    # Plot 2: Altitude over time
    ax = fig.add_subplot(2, 3, 2)
    ax.plot(time_hours, alt_no_j2, 'b-', linewidth=1.5, label='No J2')
    ax.plot(time_hours, alt_with_j2, 'r-', linewidth=1.5, label='With J2')
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Altitude [km]')
    ax.set_title('Altitude vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Position difference
    ax = fig.add_subplot(2, 3, 3)
    ax.plot(time_hours, pos_diff/1000.0, 'g-', linewidth=2)
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Position Difference [km]')
    ax.set_title('Position Difference (With J2 - No J2)')
    ax.grid(True, alpha=0.3)

    # Plot 4: XY plane view (equatorial)
    ax = fig.add_subplot(2, 3, 4)
    ax.plot(pos_no_j2[:, 0]/1e6, pos_no_j2[:, 1]/1e6, 'b-',
            linewidth=1.5, label='No J2', alpha=0.7)
    ax.plot(pos_with_j2[:, 0]/1e6, pos_with_j2[:, 1]/1e6, 'r-',
            linewidth=1.5, label='With J2', alpha=0.7)
    # Add Earth circle
    circle = plt.Circle((0, 0), 6.378, color='cyan', alpha=0.3)
    ax.add_patch(circle)
    ax.set_xlabel('X [1000 km]')
    ax.set_ylabel('Y [1000 km]')
    ax.set_title('Equatorial Plane View (XY)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Plot 5: XZ plane view (meridional)
    ax = fig.add_subplot(2, 3, 5)
    ax.plot(pos_no_j2[:, 0]/1e6, pos_no_j2[:, 2]/1e6, 'b-',
            linewidth=1.5, label='No J2', alpha=0.7)
    ax.plot(pos_with_j2[:, 0]/1e6, pos_with_j2[:, 2]/1e6, 'r-',
            linewidth=1.5, label='With J2', alpha=0.7)
    # Add Earth circle
    circle = plt.Circle((0, 0), 6.378, color='cyan', alpha=0.3)
    ax.add_patch(circle)
    ax.set_xlabel('X [1000 km]')
    ax.set_ylabel('Z [1000 km]')
    ax.set_title('Meridional Plane View (XZ)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Plot 6: Inclination over time
    ax = fig.add_subplot(2, 3, 6)
    ax.plot(time_hours, inc_no_j2, 'b-', linewidth=1.5, label='No J2')
    ax.plot(time_hours, inc_with_j2, 'r-', linewidth=1.5, label='With J2')
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Inclination [degrees]')
    ax.set_title('Orbital Inclination vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(os.getcwd(), 'j2_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    if not args.no_show:
        plt.show()

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

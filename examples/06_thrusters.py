"""Thrusters

Configures an 8-thruster planar layout, fires one thruster at a constant
command, and plots the resulting motion of the first environment. Gravity is
off, so the only forces and torques come from the thruster.

- Layout: 8 thrusters on the +/-X and +/-Y edges (0.15 m arm), thrusting
  tangentially in the XY plane, which gives force along X and Y and torque
  about Z, and nothing out of plane
- Vehicle: 5.25 kg, isotropic inertia 0.0591 kg m^2, starting at rest
- Command: thruster 0 at full thrust (1.8 N along +Y at x = -0.15 m), the
  same in all 10 000 environments, for 5 s

```bash
python examples/06_thrusters.py
```

Options: `--num-envs`, `--num-steps`, `--dt`, `--device`, `--no-show`.

![Planar trajectory](../assets/examples/planar_thrusters_trajectory.png)

![Linear velocity](../assets/examples/planar_thrusters_velocity.png)

![Angular velocity](../assets/examples/planar_thrusters_angular_velocity.png)

![Mass](../assets/examples/planar_thrusters_mass.png)

**What to look for:** an off-centre thruster gives a force and a torque.
The body spins up about -Z at 0.27 / 0.0591 = 4.57 rad/s^2 (about
-22.8 rad/s after 5 s); the body-fixed thrust rotates with it, so the
trajectory bends into a wobbling drift instead of a straight line. With
`isp = 0` the mass stays at 5.25 kg.
"""
import argparse
import os

import matplotlib.pyplot as plt
import warp as wp
import numpy as np
import time
from cosmic_warp.core.pipeline import SpacecraftSimulator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Planar thruster layout trajectory demo")
    parser.add_argument("--num-envs", type=int, default=10000, help="Number of parallel environments")
    parser.add_argument("--num-steps", type=int, default=500, help="Simulation steps")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step [s]")
    parser.add_argument("--device", default="cuda", help="Warp device (e.g. 'cuda' or 'cpu')")
    parser.add_argument("--no-show", action="store_true",
                         help="Skip interactive plot windows (plots are still saved to disk)")
    return parser.parse_args()


def main():
    args = _parse_args()

    # Create simulator (no gravity: a planar demo)
    print("Initializing simulator...")
    sim = SpacecraftSimulator(
        num_envs=args.num_envs,
        integrator="rk4",
        dt=args.dt,
        device=args.device,
        gravity=False,      # planar demo: no orbit
        check_bounds=False,
    )
    
    # Configure a planar thruster layout: 8 thrusters on the
    # +/-X and +/-Y edges, thrusting tangentially in the XY plane. Gives
    # planar 3-DOF authority (+/-Fx, +/-Fy, +/-Tz) and nothing out of plane.
    print("Setting up the planar thruster layout...")
    arm = 0.15
    sim.thruster_manager.set_config(
        max_thrust=1.8,
        isp=0.0,            # ideal thruster: no propellant use, mass stays constant
        positions=[
            (-arm, 0.0, 0.0), (0.0, -arm, 0.0), (0.0, -arm, 0.0), (arm, 0.0, 0.0),
            (arm, 0.0, 0.0), (0.0, arm, 0.0), (0.0, arm, 0.0), (-arm, 0.0, 0.0),
        ],
        directions=[
            (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0), (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0),
        ],
    )

    J_zz = 0.0591
    J = [[J_zz, 0.0, 0.0], [0.0, J_zz, 0.0], [0.0, 0.0, J_zz]]
    
    # Set vehicle properties (mass_dry = 1.0 kg)
    dry_mass = 1.0
    sim.vehicle.set_config(
        mass_dry=dry_mass, 
        inertia=J
    )
    
    # Reset to initial state
    sim.reset()

    # Reset every environment to rest at the origin (condition arrays are
    # indexed by entity, so they cover all of them)
    n = sim.num_entities
    sim.reset_to_conditions(
        positions=wp.zeros(n, dtype=wp.vec3d, device=sim.device),
        momenta=wp.zeros(n, dtype=wp.vec3d, device=sim.device),
        quaternions=wp.array(np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)), dtype=wp.quatd, device=sim.device),
        angular_momenta=wp.zeros(n, dtype=wp.vec3d, device=sim.device),
        masses=wp.full(n, 5.25, dtype=wp.float64, device=sim.device)
    )
    
    # Constant thrust command: thruster 0 at full throttle (omitted slots are zero)
    sim.set_thrust_commands([1.0])
    
    # Run simulation
    print("Running simulation...")
    num_steps = args.num_steps

    t0 = time.time()
    trajectory = sim.simulate(num_steps, save_states=True)
    elapsed = time.time() - t0
    
    # Convert to numpy for plotting
    print("Extracting trajectory data...")
    positions_np = trajectory['positions'].numpy()  # [num_steps, num_envs, 3]
    velocities_np = trajectory['velocities'].numpy()  # [num_steps, num_envs, 3]
    angular_velocities_np = trajectory['angular_velocities'].numpy()  # [num_steps, num_envs, 3]
    masses_np = trajectory['masses'].numpy() # [num_steps, num_envs]
    time_np = trajectory['time'].numpy()
    
    # Extract x, y positions for environment 0
    x = positions_np[:, 0, 0]  # X coordinate
    y = positions_np[:, 0, 1]  # Y coordinate
    t = time_np
    
    # Extract mass for environment 0
    m = masses_np[:, 0]
    
    # Extract velocities for environment 0
    vx = velocities_np[:, 0, 0]
    vy = velocities_np[:, 0, 1]
    vz = velocities_np[:, 0, 2]
    
    # Extract angular velocities for environment 0
    wx = angular_velocities_np[:, 0, 0]
    wy = angular_velocities_np[:, 0, 1]
    wz = angular_velocities_np[:, 0, 2]
    
    # Create plots
    print("Creating plots...")
    output_dir = os.getcwd()
    fig = plt.figure(figsize=(14, 5))
    
    # Figure 1: X-Y position plot
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(x, y, 'b-', linewidth=2, label='Trajectory')
    ax1.plot(x[0], y[0], 'go', markersize=10, label='Start')
    ax1.plot(x[-1], y[-1], 'ro', markersize=10, label='End')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X Position [m]', fontsize=12)
    ax1.set_ylabel('Y Position [m]', fontsize=12)
    ax1.set_title('Planar Thruster Trajectory (X-Y)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.axis('equal')
    
    # Figure 2: X and Y over time
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(t, x, 'r-', linewidth=2, label='X Position')
    ax2.plot(t, y, 'b-', linewidth=2, label='Y Position')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Time [s]', fontsize=12)
    ax2.set_ylabel('Position [m]', fontsize=12)
    ax2.set_title('Position vs Time', fontsize=14, fontweight='bold')
    ax2.legend()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'planar_thrusters_trajectory.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved as '{output_path}'")
    
    # Figure 3: Velocity components over time
    fig2 = plt.figure(figsize=(14, 5))
    
    ax3 = fig2.add_subplot(1, 2, 1)
    ax3.plot(t, vx, 'r-', linewidth=2, label='Vx')
    ax3.plot(t, vy, 'b-', linewidth=2, label='Vy')
    ax3.plot(t, vz, 'g-', linewidth=2, label='Vz')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('Time [s]', fontsize=12)
    ax3.set_ylabel('Velocity [m/s]', fontsize=12)
    ax3.set_title('Linear Velocity vs Time', fontsize=14, fontweight='bold')
    ax3.legend()
    
    # Velocity magnitude
    v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
    ax4 = fig2.add_subplot(1, 2, 2)
    ax4.plot(t, v_mag, 'k-', linewidth=2, label='|V|')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel('Time [s]', fontsize=12)
    ax4.set_ylabel('Speed [m/s]', fontsize=12)
    ax4.set_title('Velocity Magnitude vs Time', fontsize=14, fontweight='bold')
    ax4.legend()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'planar_thrusters_velocity.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved as '{output_path}'")
    
    # Figure 4: Angular velocity components over time
    fig3 = plt.figure(figsize=(14, 5))
    
    ax5 = fig3.add_subplot(1, 2, 1)
    ax5.plot(t, wx, 'r-', linewidth=2, label='ωx')
    ax5.plot(t, wy, 'b-', linewidth=2, label='ωy')
    ax5.plot(t, wz, 'g-', linewidth=2, label='ωz')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlabel('Time [s]', fontsize=12)
    ax5.set_ylabel('Angular Velocity [rad/s]', fontsize=12)
    ax5.set_title('Angular Velocity vs Time', fontsize=14, fontweight='bold')
    ax5.legend()
    
    # Angular velocity magnitude
    w_mag = np.sqrt(wx**2 + wy**2 + wz**2)
    ax6 = fig3.add_subplot(1, 2, 2)
    ax6.plot(t, w_mag, 'k-', linewidth=2, label='|ω|')
    ax6.grid(True, alpha=0.3)
    ax6.set_xlabel('Time [s]', fontsize=12)
    ax6.set_ylabel('Angular Speed [rad/s]', fontsize=12)
    ax6.set_title('Angular Velocity Magnitude vs Time', fontsize=14, fontweight='bold')
    ax6.legend()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'planar_thrusters_angular_velocity.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved as '{output_path}'")

    # Figure 5: Mass over time
    fig4 = plt.figure(figsize=(7, 5))
    ax7 = fig4.add_subplot(1, 1, 1)
    ax7.plot(t, m, 'm-', linewidth=2, label='Current Mass')
    ax7.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Dry Mass')
    ax7.grid(True, alpha=0.3)
    ax7.set_xlabel('Time [s]', fontsize=12)
    ax7.set_ylabel('Mass [kg]', fontsize=12)
    ax7.set_title('Spacecraft Mass vs Time', fontsize=14, fontweight='bold')
    ax7.legend()

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'planar_thrusters_mass.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved as '{output_path}'")
    
    # Print final stats
    print(f"\nSimulation complete:")
    print(f"  Elapsed time: {elapsed:.2f} s for {num_steps} steps")
    print(f"  Simulation duration: {t[-1]:.2f} s")
    print(f"  Final position: ({x[-1]:.3f}, {y[-1]:.3f}) m")
    print(f"  Distance traveled: {np.sqrt(x[-1]**2 + y[-1]**2):.3f} m")
    print(f"  Final velocity: ({vx[-1]:.3f}, {vy[-1]:.3f}, {vz[-1]:.3f}) m/s")
    print(f"  Final angular velocity: ({wx[-1]:.3f}, {wy[-1]:.3f}, {wz[-1]:.3f}) rad/s")
    print(f"  Initial mass: {m[0]:.3f} kg")
    print(f"  Final mass: {m[-1]:.3f} kg")
    print(f"  Propellant consumed: {m[0]-m[-1]:.3f} kg")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

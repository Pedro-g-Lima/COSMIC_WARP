"""Basic orbit

Propagates ten 100 kg spacecraft in parallel on circular two-body orbits of
7000 km radius (gravity on, J2 off), records the full trajectory, and plots it.
It covers the basic workflow:

- creating a `SpacecraftSimulator`
- setting initial conditions with `reset_to_conditions`
- running `simulate(..., save_states=True)` and timing it
- reading the trajectory back for plotting

The spacecraft start evenly spaced around the equator, each moving due north,
so each flies its own polar orbit. The default run covers 500 s (50 000 steps
of 0.01 s with RK4), a short arc of the roughly 97-minute orbit.

```bash
python examples/01_basic_orbit.py
python examples/01_basic_orbit.py --integrator symplectic --num-steps 600000
```

Options: `--num-envs`, `--num-steps`, `--dt`, `--integrator`
(`euler`, `rk4`, `symplectic`; default `rk4`), `--device`, `--no-show`.
The plots are saved to the working directory; without `--no-show` an attitude
animation of spacecraft 0 is also shown.

![3D trajectories](../assets/examples/orbital_3d.png)

![Time series of spacecraft 0](../assets/examples/orbital_timeseries.png)

![Radius, speed and mass of spacecraft 0](../assets/examples/orbital_altitude_velocity.png)

**What to look for:** each trajectory is an arc of a polar orbit. The orbit
altitude and the speed stay constant to round-off over
the run (within about 1e-7 m and 1e-10 m/s), and the mass stays at 100 kg.
"""

import argparse
import os
import time

import math

import warp as wp
import matplotlib.pyplot as plt
import numpy as np

from cosmic_warp.core.pipeline import SpacecraftSimulator
from cosmic_warp.physics.gravity import RE_EARTH
from cosmic_warp.physics.gravity import MU_EARTH


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Basic orbital dynamics simulation")
    parser.add_argument("--num-envs", type=int, default=10, help="Number of parallel spacecraft")
    parser.add_argument("--num-steps", type=int, default=50000, help="Simulation steps")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step [s]")
    parser.add_argument("--integrator", default="rk4", choices=["euler", "rk4", "symplectic"],
                         help="Integration method")
    parser.add_argument("--device", default="cuda", help="Warp device (e.g. 'cuda' or 'cpu')")
    parser.add_argument("--no-show", action="store_true",
                         help="Skip interactive plot windows (plots are still saved to disk)")
    return parser.parse_args()


def main():
    """Run basic orbital simulation."""
    args = _parse_args()

    wp.init()

    print("=" * 60)
    print("COSMIC WARP - Basic Orbital Simulation Example")
    print("=" * 60)

    # Configuration
    num_envs = args.num_envs
    num_steps = args.num_steps
    dt = args.dt

    integrator = args.integrator
    device = args.device

    # Create simulator
    print(f"\nInitializing simulator...")
    print(f"  Environments: {num_envs}")
    print(f"  Time step: {dt} s")
    print(f"  Integrator: {integrator}")
    
    sim = SpacecraftSimulator(
        num_envs=num_envs,
        device=device,
        integrator=integrator,
        dt=dt,
        gravity=True,
        j2_perturbation=False,  # Disable J2 for this basic example
        magnetic_torque=False,  # Disable magnetic torque for this basic example
        gravity_gradient=False,
        pos_limit=1.0e8,
        mom_limit=1.0e9,
        ang_mom_limit=1.0e8,
    )

    # Set up circular orbit initial conditions
    # Orbital radius: 7000 km (LEO)
    r_orbit = 7.0e6  # meters
    mu = MU_EARTH
    v_circular = math.sqrt(mu / r_orbit)
    
    print(f"\nSetting up circular orbits:")
    print(f"  Orbital radius: {r_orbit/1e3:.1f} km")
    print(f"  Circular velocity: {v_circular:.1f} m/s")
    
    # Use host arrays to set up initial conditions, then copy to device
    pos_host = []
    mom_host = []
    mass_host = []
    
    for i in range(num_envs):
        mass_host.append(100.0)
        # Evenly spaced angle around the equator
        theta = 2 * wp.PI * i / num_envs
        
        # Position on circular orbit (XY plane)
        pos_host.append([
            r_orbit * wp.cos(theta),
            r_orbit * wp.sin(theta),
            0.0
        ])
        
        # Velocity along +Z, perpendicular to position (circular polar orbit)
        mom_host.append([
            0.0,
            0.0,
            v_circular*mass_host[-1]
        ])
    
    # Convert to WARP arrays
    positions = wp.array(pos_host, dtype=wp.vec3d, device=device)
    momenta = wp.array(mom_host, dtype=wp.vec3d, device=device)
    masses = wp.array(mass_host, dtype=wp.float64, device=device)
    
    # Reset with custom conditions
    sim.reset()
    sim.reset_to_conditions(
        positions=positions,
        momenta=momenta,
        masses=masses
    )
    
    # Run simulation with timing
    print(f"\nRunning simulation for {num_steps} steps...")
    print(f"  Total simulation time: {num_steps * dt:.1f} s")
    start_time = time.time()
    trajectory = sim.simulate(num_steps=num_steps, save_states=True)
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\nSimulation complete!")
    print(f"  Total steps: {sim.step_count}")
    print(f"  Simulation time: {sim.step_count * sim.dt:.1f} s")
    print(f"  Execution time: {elapsed:.2f} s")
    print(f"  Time per step: {elapsed / sim.step_count:.6f} s/step")
    print(f"  Steps per second: {sim.step_count / elapsed:.2f} steps/s")
    
    # Extract results (WARP arrays - convert to numpy only for visualization)
    pos_warp = trajectory['positions']  # Shape: [num_steps, num_envs, 3]
    vel_warp = trajectory['velocities']
    time_warp = trajectory['time']
    
    # Convert to numpy only for final analysis and plotting
    pos = pos_warp.numpy()
    vel = vel_warp.numpy()
    time_arr = time_warp.numpy()
    pos_initial = positions.numpy()
    
    # Compute orbital elements for analysis
    print(f"\nComputing final orbital parameters...")

    # Semi-major axis and eccentricity for first spacecraft
    r_final = math.sqrt(pos[-1, 0, 0]**2 + pos[-1, 0, 1]**2 + pos[-1, 0, 2]**2)

    r_initial = math.sqrt(pos_initial[0, 0]**2 + pos_initial[0, 1]**2 + pos_initial[0, 2]**2)
    print(f"  Initial radius: {r_initial/1e3:.1f} km")
    print(f"  Final radius: {r_final/1e3:.1f} km")
    


    ###############################################
    # Visualization
    ###############################################
    print(f"\nGenerating plots...")


    # --- Figure 1: Time series for pos, vel, ang_vel, quats (env 0 only, 2x2 grid) ---
    fig1, axs1 = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    labels = ['x', 'y', 'z']
    # Position components
    for j in range(3):
        axs1[0, 0].plot(time_arr, pos[:, 0, j], label=labels[j], alpha=0.8)
    axs1[0, 0].set_ylabel('Position [m]')
    axs1[0, 0].set_title('Position (env 0)')
    axs1[0, 0].legend(loc='upper right', fontsize=8)
    axs1[0, 0].grid(True, alpha=0.3)

    # Velocity components
    for j in range(3):
        axs1[0, 1].plot(time_arr, vel[:, 0, j], label=labels[j], alpha=0.8)
    axs1[0, 1].set_ylabel('Velocity [m/s]')
    axs1[0, 1].set_title('Velocity (env 0)')
    axs1[0, 1].legend(loc='upper right', fontsize=8)
    axs1[0, 1].grid(True, alpha=0.3)

    # Angular velocity components (if available)
    ang_vel = trajectory.get('angular_velocities', None)
    if ang_vel is not None:
        ang_vel = ang_vel.numpy()
        for j in range(3):
            axs1[1, 0].plot(time_arr, ang_vel[:, 0, j], label=labels[j], alpha=0.8)
        axs1[1, 0].set_ylabel('Ang. Vel. [rad/s]')
        axs1[1, 0].set_title('Angular Velocity (env 0)')
        axs1[1, 0].legend(loc='upper right', fontsize=8)
        axs1[1, 0].grid(True, alpha=0.3)
    else:
        axs1[1, 0].set_visible(False)

    # Quaternion components (if available)
    quats = trajectory.get('quaternions', None)
    if quats is not None:
        quats = quats.numpy()
        quat_labels = ['x', 'y', 'z', 'w']   # Warp order
        for j in range(4):
            axs1[1, 1].plot(time_arr, quats[:, 0, j], label=quat_labels[j], alpha=0.8)
        axs1[1, 1].set_ylabel('Quaternion')
        axs1[1, 1].set_title('Quaternion (env 0)')
        axs1[1, 1].legend(loc='upper right', fontsize=8)
        axs1[1, 1].grid(True, alpha=0.3)
    else:
        axs1[1, 1].set_visible(False)

    for ax in axs1[-1, :]:
        ax.set_xlabel('Time [s]')
    plt.tight_layout()
    output_path1 = os.path.join(os.getcwd(), 'orbital_timeseries.png')
    fig1.savefig(output_path1, dpi=150)
    print(f"\nTime series plot saved to: {output_path1}")

    # --- Figure 2: 3D Trajectory ---
    fig2 = plt.figure(figsize=(7, 6))
    ax2 = fig2.add_subplot(111, projection='3d')
    for i in range(min(100, num_envs)):
        ax2.plot(
            pos[:, i, 0] / 1e6,
            pos[:, i, 1] / 1e6,
            pos[:, i, 2] / 1e6,
            alpha=1.0,
            linewidth=1.5
        )
    # Plot Earth (using numpy only for sphere generation)
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = 6.371 * np.outer(np.cos(u), np.sin(v))
    y = 6.371 * np.outer(np.sin(u), np.sin(v))
    z = 6.371 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax2.plot_surface(x, y, z, color='blue', alpha=0.3)
    ax2.set_xlabel('X [1000 km]')
    ax2.set_ylabel('Y [1000 km]')
    ax2.set_zlabel('Z [1000 km]')
    ax2.set_title('3D Trajectories')
    ax2.set_box_aspect([1, 1, 1])
    output_path2 = os.path.join(os.getcwd(), 'orbital_3d.png')
    fig2.savefig(output_path2, dpi=150)
    print(f"3D trajectory plot saved to: {output_path2}")

    # --- Figure 3: Altitude, Velocity Magnitude, and Mass ---
    fig3, (ax3, ax4, ax5) = plt.subplots(1, 3, figsize=(16, 5))
    # Altitude over time
    altitude = (np.sqrt(np.sum(pos[:, 0, :]**2, axis=1)) - RE_EARTH) / 1e3
    ax3.plot(time_arr, altitude, alpha=0.7, linewidth=1.5)
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Altitude [km]')
    ax3.set_title('Altitude vs Time')
    ax3.grid(True, alpha=0.3)
    # Velocity magnitude over time
    vel_mag = np.sqrt(np.sum(vel[:, 0, :]**2, axis=1))
    ax4.plot(time_arr, vel_mag, alpha=0.7, linewidth=1.5)
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Velocity [m/s]')
    ax4.set_title('Velocity Magnitude vs Time')
    ax4.grid(True, alpha=0.3)
    # Mass over time
    mass_traj = trajectory['masses'].numpy()  # [num_steps, num_envs]
    ax5.plot(time_arr, mass_traj[:, 0], alpha=0.7, linewidth=1.5)
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('Mass [kg]')
    ax5.set_title('Mass vs Time (env 0)')
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path3 = os.path.join(os.getcwd(), 'orbital_altitude_velocity.png')
    fig3.savefig(output_path3, dpi=150)
    print(f"Altitude/velocity/mass plot saved to: {output_path3}")

    # --- Animation: Attitude visualization (env 0, centered at spacecraft) ---
    # Built only when it will be shown
    if quats is not None and not args.no_show:
        from matplotlib.animation import FuncAnimation

        print("\nPreparing attitude animation (close window to exit)...")

        def quat_rotate_vec(q, v):
            """Rotate vector v by unit quaternion q = [x, y, z, w] (Rodrigues formula)."""
            w = q[3]
            u = q[:3]
            return v + 2.0 * w * np.cross(u, v) + 2.0 * np.cross(u, np.cross(u, v))

        # Subsample to at most N_frames for smooth playback
        N_frames = 500
        skip = max(1, num_steps // N_frames)
        q_anim = quats[::skip, 0, :]          # [N_frames, 4]  format: [x, y, z, w]
        t_anim = time_arr[::skip]
        p_anim = pos[::skip, 0, :]             # [N_frames, 3]  ECI positions

        # Normalize quaternions (guard against drift)
        q_anim = q_anim / np.linalg.norm(q_anim, axis=1, keepdims=True)

        # Body-frame unit axis vectors
        e = np.eye(3)   # rows: x, y, z

        # Spacecraft body wireframe in body frame (flat box like a cubesat)
        dx, dy, dz = 0.40, 0.40, 0.18
        corners = np.array([
            [ dx,  dy,  dz], [ dx,  dy, -dz],
            [ dx, -dy,  dz], [ dx, -dy, -dz],
            [-dx,  dy,  dz], [-dx,  dy, -dz],
            [-dx, -dy,  dz], [-dx, -dy, -dz],
        ])
        box_edges = [
            (0,1),(2,3),(4,5),(6,7),   # z-parallel
            (0,2),(1,3),(4,6),(5,7),   # y-parallel
            (0,4),(1,5),(2,6),(3,7),   # x-parallel
        ]

        fig_anim = plt.figure(figsize=(7, 7))
        ax_a = fig_anim.add_subplot(111, projection='3d')
        lim = 0.9
        ax_a.set_xlim(-lim, lim); ax_a.set_ylim(-lim, lim); ax_a.set_zlim(-lim, lim)
        ax_a.set_xlabel('ECI X'); ax_a.set_ylabel('ECI Y'); ax_a.set_zlabel('ECI Z')
        ax_a.set_box_aspect([1, 1, 1])

        # Faint ECI reference frame lines
        for i, (col, lbl) in enumerate([('gray', ''), ('gray', ''), ('gray', '')]):
            ax_a.plot([-lim, lim] if i == 0 else [0, 0],
                      [-lim, lim] if i == 1 else [0, 0],
                      [-lim, lim] if i == 2 else [0, 0],
                      color=col, alpha=0.15, linewidth=0.8)

        def _make_axis_line(tip, color, label):
            line, = ax_a.plot([0, tip[0]], [0, tip[1]], [0, tip[2]],
                              color=color, linewidth=2.5, label=label)
            return line

        def _make_box_lines(q):
            rc = np.array([quat_rotate_vec(q, c) for c in corners])
            lines = []
            for i, j in box_edges:
                l, = ax_a.plot([rc[i,0], rc[j,0]], [rc[i,1], rc[j,1]], [rc[i,2], rc[j,2]],
                               color='#888888', alpha=0.55, linewidth=1.0)
                lines.append(l)
            return lines

        def nadir_eci(frame):
            """Unit vector pointing from spacecraft toward Earth in ECI."""
            r = p_anim[frame]
            return -r / np.linalg.norm(r)

        q0 = q_anim[0]
        tips0 = [quat_rotate_vec(q0, e[i]) * 0.75 for i in range(3)]
        line_x = _make_axis_line(tips0[0], '#e74c3c', 'Body X')
        line_y = _make_axis_line(tips0[1], '#2ecc71', 'Body Y')
        line_z = _make_axis_line(tips0[2], '#3498db', 'Body Z')
        axis_lines = [line_x, line_y, line_z]
        box_lines = _make_box_lines(q0)

        # Nadir arrow: solid line + dashed tail to distinguish from body axes
        nd0 = nadir_eci(0) * 0.82
        nadir_line, = ax_a.plot([0, nd0[0]], [0, nd0[1]], [0, nd0[2]],
                                color='#f39c12', linewidth=2.5,
                                linestyle='--', label='↓ Nadir (Earth)')
        # Arrowhead approximated by a short solid segment at the tip
        nadir_tip, = ax_a.plot([nd0[0]*0.82, nd0[0]], [nd0[1]*0.82, nd0[1]], [nd0[2]*0.82, nd0[2]],
                               color='#f39c12', linewidth=4.0, linestyle='-')

        time_txt = ax_a.text2D(0.03, 0.95, f't = {t_anim[0]:.1f} s',
                               transform=ax_a.transAxes, fontsize=11,
                               bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        ax_a.legend(loc='lower right', fontsize=9)
        ax_a.set_title('Spacecraft Attitude (env 0)', fontsize=11, pad=8)

        def update(frame):
            q = q_anim[frame]
            colors = ['#e74c3c', '#2ecc71', '#3498db']
            for k, (line, col) in enumerate(zip(axis_lines, colors)):
                tip = quat_rotate_vec(q, e[k]) * 0.75
                line.set_data([0, tip[0]], [0, tip[1]])
                line.set_3d_properties([0, tip[2]])
            rc = np.array([quat_rotate_vec(q, c) for c in corners])
            for line, (i, j) in zip(box_lines, box_edges):
                line.set_data([rc[i,0], rc[j,0]], [rc[i,1], rc[j,1]])
                line.set_3d_properties([rc[i,2], rc[j,2]])
            nd = nadir_eci(frame) * 0.82
            nadir_line.set_data([0, nd[0]], [0, nd[1]])
            nadir_line.set_3d_properties([0, nd[2]])
            nadir_tip.set_data([nd[0]*0.82, nd[0]], [nd[1]*0.82, nd[1]])
            nadir_tip.set_3d_properties([nd[2]*0.82, nd[2]])
            time_txt.set_text(f't = {t_anim[frame]:.1f} s')
            return axis_lines + box_lines + [nadir_line, nadir_tip, time_txt]

        anim = FuncAnimation(
            fig_anim, update,
            frames=len(t_anim),
            interval=30,          # ms between frames (~33 fps)
            blit=False            # 3D axes don't support blit
        )
        print(f"  Animating {len(t_anim)} frames "
              f"(every {skip} steps, {t_anim[1]-t_anim[0]:.2f} s / frame)")
    elif quats is None:
        print("\nNo quaternion data — skipping attitude animation.")

    if not args.no_show:
        plt.show()

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

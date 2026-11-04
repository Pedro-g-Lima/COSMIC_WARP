"""Reaction wheels

Spins up a single reaction wheel along body Z with a constant torque command
and compares the result with the analytic solution. The wheel model has
viscous friction:

- wheel: `J_r * dω_r/dt = τ_cmd - b * ω_r`
- body reaction torque: `τ_body = -J_r * dω_r/dt`

With `τ_cmd = 0.1 N m`, `J_r = 1.127e-3 kg m^2` and `b = 5.37e-5 N m s/rad`,
the wheel speed follows `ω_r(t) = (τ_cmd / b) * (1 - exp(-b t / J_r))`: a
time constant of about 21 s and a steady state of 1861 rad/s. The default run
is 200 s (200 000 steps of 1 ms, RK4), gravity off, starting from rest.

```bash
python examples/07_reaction_wheels.py
```

Options: `--num-steps`, `--dt`, `--device`, `--no-show`.

![Wheel speeds](../assets/examples/rw_wheels.png)

![Body angular velocity](../assets/examples/rw_angvel.png)

**What to look for:** the simulated wheel speed lies on the analytic curve
and ends within 0.01 % of the steady state. The wheel plot shows all four
wheel slots; wheels 1-3 are unused and stay at zero. The body counter-rotates
so that the total angular momentum stays zero: with `J_body = 0.0591 kg m^2`
it settles at `-J_r * ω_r / J_body` = -35.5 rad/s about Z.
"""

import argparse
import os

import warp as wp
import numpy as np
import matplotlib.pyplot as plt

from cosmic_warp.core.pipeline import SpacecraftSimulator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reaction wheel dynamics demo")
    parser.add_argument("--num-steps", type=int, default=200000, help="Simulation steps")
    parser.add_argument("--dt", type=float, default=0.001, help="Time step [s]")
    parser.add_argument("--device", default="cuda", help="Warp device (e.g. 'cuda' or 'cpu')")
    parser.add_argument("--no-show", action="store_true",
                         help="Skip interactive plot window (plots are still saved to disk)")
    return parser.parse_args()


def main():
    """Run the reaction wheel example."""
    args = _parse_args()

    wp.init()

    print("=" * 60)
    print("Reaction wheel dynamics")
    print("=" * 60)

    # Create simulator with RW enabled
    sim = SpacecraftSimulator(
        num_envs=1,
        dt=args.dt,
        integrator="rk4",
        device=args.device,
        gravity=False
    )

    sim.vehicle.set_config(mass_dry=5.25, inertia=0.0591 * np.eye(3))
    sim.reset_manager.set_bounds(mass_bounds=(5.25, 5.25))  # no propellant
    
    # Configure a reaction wheel with viscous friction
    J_wheel = 0.00112703295596      # kg*m^2
    max_torque = 0.375    # N*m
    b = 5.372473380648529e-05           # N*m*s/rad (viscous damping)

    # A single wheel along body Z; wheels 1-3 are unused (up to 4 can be given)
    spin_axes = [
        (0.0, 0.0, 1.0)
    ]
    
    sim.rw_manager.set_config(
        max_torque=max_torque,
        wheel_inertia=J_wheel,
        viscous_damping=b,
        spin_axes=spin_axes
    )
    
    print(f"\nReaction Wheel Configuration (1 wheel):")
    print(f"  Wheel inertia: {J_wheel} kg*m^2")
    print(f"  Max torque: {max_torque} N*m")
    print(f"  Viscous damping: {b} N*m*s/rad")
    print(f"  Spin axes:")
    for i, axis in enumerate(spin_axes):
        print(f"    Wheel {i}: {axis}")
    
    # Reset to zero initial conditions
    sim.reset()
    positions = wp.array([(0.0, 0.0, 0.0)], dtype=wp.vec3d, device=args.device)
    momenta = wp.array([(0.0, 0.0, 0.0)], dtype=wp.vec3d, device=args.device)
    sim.reset_to_conditions(positions=positions, momenta=momenta)
    
    # Constant torque command on wheel 0 (unused wheels get 0)
    torque_cmds = [0.1, 0.0, 0.0, 0.0]  # N*m
    sim.set_rw_torque_commands(torque_cmds)
    
    print(f"\nApplying constant torques: {torque_cmds} N*m")
    if b > 0:
        print(f"Expected steady-state wheel speeds:")
        for i, tau in enumerate(torque_cmds):
            print(f"  Wheel {i}: omega_ss = {tau/b:.1f} rad/s")
    else:
        print(f"No damping (b=0): wheels will accelerate indefinitely")
        print(f"Expected acceleration:")
        for i, tau in enumerate(torque_cmds):
            alpha = tau / J_wheel
            print(f"  Wheel {i}: alpha = {alpha:.1f} rad/s^2")
    
    # Simulate
    num_steps = args.num_steps
    results = sim.simulate(num_steps=num_steps, save_states=True, save_rw_states=True)
    
    # Extract angular velocities
    time = results['time'].numpy()
    ang_vel = results['angular_velocities'].numpy()  # [num_steps, num_envs, 3]
    
    # Get wheel velocities from states
    states_host = results['rw_angular_velocities'].numpy()  # [num_steps, num_envs, 4]
    
    print(f"\nFinal state:")
    print(f"  Body angular velocity: {ang_vel[-1, 0, :]} rad/s")
    print(f"  Wheel angular velocities: {states_host[-1, 0, :]} rad/s")
    
    if b > 0:
        print(f"\nTheoretical vs Actual steady-state speeds:")
        for i, tau in enumerate(torque_cmds):
            omega_ss_theory = tau / b
            omega_actual = states_host[-1, 0, i]
            # Avoid division by zero when expected steady-state is zero
            if abs(omega_ss_theory) < 1e-12:
                abs_err = abs(omega_actual - omega_ss_theory)
                print(f"  Wheel {i}: Expected {omega_ss_theory:.3f} rad/s, Actual {omega_actual:.3f} rad/s (abs error: {abs_err:.3e} rad/s)")
            else:
                error = abs(omega_actual - omega_ss_theory) / abs(omega_ss_theory) * 100
                print(f"  Wheel {i}: Expected {omega_ss_theory:.3f} rad/s, Actual {omega_actual:.3f} rad/s (error: {error:.2f}%)")
    else:
        print(f"\nTheoretical vs Actual final speeds (no steady state):")
        for i, tau in enumerate(torque_cmds):
            alpha = tau / J_wheel
            omega_theory = alpha * time[-1]
            omega_actual = states_host[-1, 0, i]
            if abs(omega_theory) < 1e-12:
                abs_err = abs(omega_actual - omega_theory)
                print(f"  Wheel {i}: Expected {omega_theory:.3f} rad/s, Actual {omega_actual:.3f} rad/s (abs error: {abs_err:.3e} rad/s)")
            else:
                error = abs(omega_actual - omega_theory) / abs(omega_theory) * 100
                print(f"  Wheel {i}: Expected {omega_theory:.3f} rad/s, Actual {omega_actual:.3f} rad/s (error: {error:.2f}%)")
    
    # Figure 1: Wheels (2x2)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    wheel_names = ['Wheel 0', 'Wheel 1', 'Wheel 2', 'Wheel 3']

    fig_w, axs_w = plt.subplots(2, 2, figsize=(12, 8))
    axs_w = axs_w.flatten()

    # Theoretical wheel velocity curve
    time_theory = np.linspace(0, time[-1], 200)

    for i in range(4):
        ax = axs_w[i]
        tau = torque_cmds[i]
        
        if b > 0:
            # With damping: ω(t) = (τ_cmd/b) * (1 - exp(-b*t/J))
            omega_ss = tau / b
            omega_theory = omega_ss * (1 - np.exp(-b * time_theory / J_wheel))
            title = f'{wheel_names[i]} (τ={tau:.2f} N·m, ω_ss={omega_ss:.1f} rad/s)'
            ax.axhline(y=omega_ss, color='gray', linestyle=':', alpha=0.4, linewidth=1, label='Steady state')
        else:
            # No damping: ω(t) = (τ_cmd/J) * t
            alpha = tau / J_wheel
            omega_theory = alpha * time_theory
            title = f'{wheel_names[i]} (τ={tau:.2f} N·m, α={alpha:.1f} rad/s²)'

        ax.plot(time, states_host[:, 0, i], '-', color=colors[i], label='Actual', linewidth=2, alpha=0.9)
        ax.plot(time_theory, omega_theory, '--', color='black', label='Theoretical', linewidth=2, alpha=0.6)

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('ω [rad/s]')
        ax.set_title(title)
        ax.legend(loc='lower right' if b > 0 else 'upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    wheels_path = os.path.join(os.getcwd(), 'rw_wheels.png')
    plt.savefig(wheels_path, dpi=150, bbox_inches='tight')
    print(f"\nWheels plot saved to: {wheels_path}")

    # Figure 2: Body angular velocity (2x1) - top: 3 axes, bottom: magnitude
    fig_b, axs_b = plt.subplots(2, 1, figsize=(10, 8))

    axs_b[0].plot(time, ang_vel[:, 0, 0], 'r-', label='ω_x', linewidth=2)
    axs_b[0].plot(time, ang_vel[:, 0, 1], 'g-', label='ω_y', linewidth=2)
    axs_b[0].plot(time, ang_vel[:, 0, 2], 'b-', label='ω_z', linewidth=2)
    axs_b[0].set_xlabel('Time [s]')
    axs_b[0].set_ylabel('Angular Velocity [rad/s]')
    axs_b[0].set_title('Spacecraft Body Angular Velocity (components)')
    axs_b[0].legend(loc='upper right')
    axs_b[0].grid(True, alpha=0.3)

    ang_speed = np.linalg.norm(ang_vel[:, 0, :], axis=1)
    axs_b[1].plot(time, ang_speed, 'k-', label='|ω|', linewidth=2)
    axs_b[1].set_xlabel('Time [s]')
    axs_b[1].set_ylabel('Angular Speed [rad/s]')
    axs_b[1].set_title('Spacecraft Angular Speed (magnitude)')
    axs_b[1].legend(loc='upper right')
    axs_b[1].grid(True, alpha=0.3)

    plt.tight_layout()
    body_path = os.path.join(os.getcwd(), 'rw_angvel.png')
    plt.savefig(body_path, dpi=150, bbox_inches='tight')
    print(f"Body plots saved to: {body_path}")
    if not args.no_show:
        plt.show()


    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)

if __name__ == "__main__":
    main()
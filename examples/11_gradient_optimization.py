r"""Gradient-based trajectory optimization

Backpropagates through a 500-step orbital rollout to retarget a trajectory:
plain gradient descent on the initial velocity moves the end point 50 km
cross-track, out of the orbital plane.

- creates a differentiable simulator (`differentiable=True`) and a rollout
  with `sim.differentiable_rollout`
- records the rollout and a terminal loss (squared miss distance) on the
  rollout's `wp.Tape` and reads the gradient from
  `rollout.initial_states.grad`
- the orbit is circular at about 500 km altitude; the rollout is 500 steps
  of 2 s (1000 s), symplectic integrator, two-body gravity (J2 off)

```bash
python examples/11_gradient_optimization.py
python examples/11_gradient_optimization.py --offset-km 10 --device cpu
```

Options: `--num-steps`, `--dt`, `--iters`, `--offset-km`, `--tol-m`,
`--device`, `--no-show`. See the
[differentiable rollouts guide](../guides/differentiable.md) for the
rollout API.

![Gradient optimization](../assets/examples/gradient_optimization.png)

**What to look for:** the miss distance contracts geometrically, from
50 km to below 1 m in about 30 iterations and below 1 mm by the default 60
(each a forward and an adjoint rollout). The delta-v found matches the
analytic two-body answer $\Delta v = \delta \, n / \sin(nt)$ (offset
$\delta$, mean motion $n$), which the script also prints: 61.883 m/s.
"""

import argparse
import os
import math
import time

import numpy as np
import warp as wp
import matplotlib.pyplot as plt

from cosmic_warp.core.pipeline import SpacecraftSimulator
from cosmic_warp.core.state import State
from cosmic_warp.physics.gravity import MU_EARTH


@wp.kernel
def position_loss_kernel(
    states: wp.array(dtype=State),
    target: wp.vec3d,
    loss: wp.array(dtype=wp.float64),
):
    """Squared distance of the final position to the target."""
    tid = wp.tid()
    d = states[tid].pos - target
    wp.atomic_add(loss, 0, wp.dot(d, d))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradient-based trajectory optimization")
    parser.add_argument("--num-steps", type=int, default=500, help="Rollout steps")
    parser.add_argument("--dt", type=float, default=2.0, help="Time step [s]")
    parser.add_argument("--iters", type=int, default=60, help="Gradient descent iterations")
    parser.add_argument("--offset-km", type=float, default=50.0,
                        help="Cross-track target offset from the ballistic endpoint [km]")
    parser.add_argument("--tol-m", type=float, default=1.0,
                        help="Miss distance counted as converged [m]")
    parser.add_argument("--device", default="cuda", help="Warp device (e.g. 'cuda' or 'cpu')")
    parser.add_argument("--no-show", action="store_true",
                        help="Skip interactive plot windows (plots are still saved to disk)")
    return parser.parse_args()


def rollout_positions(rollout, stride=5):
    """Positions along the recorded trajectory (for plotting)."""
    idx = list(range(0, rollout.num_steps + 1, stride))
    return np.array([rollout.states[t].numpy()["pos"][0] for t in idx])


def main():
    args = _parse_args()

    wp.init()

    print("=" * 60)
    print("COSMIC WARP - Gradient-Based Trajectory Optimization")
    print("=" * 60)

    r_orbit = 6.878e6                      # ~500 km altitude
    v_circ = math.sqrt(MU_EARTH / r_orbit)
    t_final = args.num_steps * args.dt

    sim = SpacecraftSimulator(
        num_envs=1,
        integrator="symplectic",
        dt=args.dt,
        device=args.device,
        j2_perturbation=False,             # two-body, to compare with the analytic dv
        check_bounds=False,
        differentiable=True,
    )
    sim.reset_to_conditions(
        positions=wp.array([[r_orbit, 0.0, 0.0]], dtype=wp.vec3d, device=args.device),
        momenta=wp.array([[0.0, v_circ, 0.0]], dtype=wp.vec3d, device=args.device),  # 1 kg
    )

    rollout = sim.differentiable_rollout(args.num_steps)

    # Ballistic endpoint, then aim `offset_km` off it (cross-track)
    rollout.forward()
    p_ballistic = rollout.final_states.numpy()["pos"][0]
    target_np = p_ballistic + np.array([0.0, 0.0, args.offset_km * 1e3])
    target = wp.vec3d(*target_np)
    baseline_traj = rollout_positions(rollout)

    print(f"Rollout: {args.num_steps} steps x {args.dt} s = {t_final:.0f} s")
    print(f"Target: {args.offset_km:.0f} km cross-track off the ballistic endpoint")

    # Gradient descent on the initial momentum (1 kg body: momentum == velocity).
    # For the ballistic sensitivity dp_final/dv0 ~ t_final, GD contracts for
    # lr < 1/t_final^2.
    lr = 0.25 / t_final**2
    s0 = sim.states.numpy().copy()
    v0_initial = s0["mom"][0].copy()

    losses = []
    t_start = time.time()
    for it in range(args.iters):
        sim.states.assign(s0)
        rollout.reset()
        loss = wp.zeros(1, dtype=wp.float64, device=args.device, requires_grad=True)
        with rollout.tape:
            rollout.forward()
            wp.launch(position_loss_kernel, dim=sim.num_entities,
                      inputs=[rollout.final_states, target, loss], device=args.device)
        rollout.tape.backward(loss)

        g = rollout.initial_states.grad.numpy()["mom"]
        s0["mom"] = s0["mom"] - lr * g
        losses.append(float(loss.numpy()[0]))
        if it % 10 == 0 or it == args.iters - 1:
            print(f"  iter {it:3d}  miss distance = {math.sqrt(losses[-1])/1e3:9.4f} km")
    elapsed = time.time() - t_start

    dv = s0["mom"][0] - v0_initial
    miss = math.sqrt(losses[-1])           # miss of the last evaluated rollout
    status = "Converged" if miss <= args.tol_m else "Not converged"
    print(f"\n{status} (miss {miss:.3f} m, tolerance {args.tol_m} m) after "
          f"{args.iters} forward+backward rollouts of {args.num_steps} steps "
          f"in {elapsed:.2f} s")
    print(f"Delta-v found: [{dv[0]:+.3f}, {dv[1]:+.3f}, {dv[2]:+.3f}] m/s "
          f"(|dv| = {np.linalg.norm(dv):.3f} m/s)")
    # Two-body cross-track answer: dv = delta * n / sin(n t)
    n = math.sqrt(MU_EARTH / r_orbit**3)
    print(f"Analytic cross-track dv: {args.offset_km * 1e3 * n / math.sin(n * t_final):.3f} m/s")
    print(f"Miss distance: {math.sqrt(losses[0])/1e3:.3f} km -> {miss:.3f} m")

    optimized_traj = rollout_positions(rollout)

    # --- Plots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.semilogy(np.sqrt(losses) / 1e3)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Miss distance [km]")
    ax1.set_title("Gradient descent convergence")
    ax1.grid(True, alpha=0.3)

    ax2.plot(baseline_traj[:, 1] / 1e3, baseline_traj[:, 2] / 1e3,
             "--", label="Ballistic")
    ax2.plot(optimized_traj[:, 1] / 1e3, optimized_traj[:, 2] / 1e3,
             label="Optimized")
    ax2.plot(target_np[1] / 1e3, target_np[2] / 1e3, "r*", markersize=14, label="Target")
    ax2.set_xlabel("y [km]")
    ax2.set_ylabel("z [km]")
    ax2.set_title("Trajectory (cross-track plane)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(os.getcwd(), "gradient_optimization.png")
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

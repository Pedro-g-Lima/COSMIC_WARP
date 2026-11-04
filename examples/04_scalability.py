"""Scalability

Times `sim.step()` for increasing numbers of environments (1 to 1 000 000 by
default, in powers of ten) and for each integrator, and prints the total
time, time per step and steps per second for each count. Multiply steps per
second by the environment count for the throughput in env-steps per second.

- Each count gets 10 warm-up steps, then `--steps` timed steps (10 000 by
  default) followed by a device synchronization.
- Each `sim.step()` is a separate Python call, so at small counts the time
  per step is set by launch overhead rather than by the number of
  environments.
- A count that runs out of device memory is reported and skipped; any other
  error is raised.

```bash
python examples/04_scalability.py
python examples/04_scalability.py --env-counts 1000 100000 --steps 1000 --integrators rk4
```

Options: `--env-counts`, `--steps`, `--dt`, `--device`, `--integrators`.

**What to look for:** the time per step grows far more slowly than the
environment count up to about 10 000 environments, then roughly in proportion
to it once the GPU is saturated, so the throughput keeps rising up to the
largest counts. Runtimes depend on the hardware; see the
[Benchmarks](../benchmarks.md) page for measured numbers.
"""

import argparse
import time

import warp as wp

from cosmic_warp.core.pipeline import SpacecraftSimulator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scalability benchmark across env counts")
    parser.add_argument("--env-counts", type=int, nargs="+",
                         default=[10**0, 10**1, 10**2, 10**3, 10**4, 10**5, 10**6],
                         help="Environment counts to benchmark")
    parser.add_argument("--steps", type=int, default=10000, help="Steps timed per env count")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step [s]")
    parser.add_argument("--device", default="cuda", help="Warp device (e.g. 'cuda' or 'cpu')")
    parser.add_argument("--integrators", nargs="+", default=["euler", "rk4", "symplectic"],
                         choices=["euler", "rk4", "symplectic"], help="Integrators to benchmark")
    return parser.parse_args()


def benchmark(integrator: str, env_counts, steps: int = 10000, dt: float = 0.01, device: str = "cuda"):
    wp.init()
    print(f"\nBenchmarking integrator: {integrator}")

    for n in env_counts:
        try:
            print(f"  envs={n:}: creating simulator...", end=" ")
            sim = SpacecraftSimulator(
                num_envs=n,
                integrator=integrator,
                dt=dt,
                device=device,
                gravity=True,
                pos_limit=1e9,
                mom_limit=1e7,
            )
            sim.reset()

            # Warmup 10 steps (kernel compilation may occur here)
            for _ in range(10):
                sim.step()
            wp.synchronize()    
            
            sim.reset()    

            t0 = time.time()
            for _ in range(steps):
                sim.step()
            wp.synchronize()  
            elapsed = time.time() - t0

            print(f"OK  total={elapsed:.4f}s  per_step={elapsed/steps:.8f}s  steps/s={steps/elapsed:.1f}")

            # Clean up
            del sim

        except RuntimeError as e:
            # Only out-of-memory is expected here; anything else is a bug
            if "Failed to allocate" not in str(e):
                raise
            print(f"SKIPPED  out of memory: {e}")


def main():
    args = _parse_args()

    # 1e6 envs may be skipped depending on hardware
    for integrator in args.integrators:
        benchmark(integrator, args.env_counts, steps=args.steps, dt=args.dt, device=args.device)

    print("\nScalability benchmark complete.")


if __name__ == "__main__":
    main()

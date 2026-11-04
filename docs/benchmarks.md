# Benchmarks

Throughput of `simulate()` (CUDA graph replay, float64, gravity + J2,
`dt = 0.01 s`) from `benchmarks/bench.py`. Each cell is steps/s of the whole
batch, with entity-steps/s (steps/s × entities) in parentheses. Single-body
rows run with termination checks off and no thrust.

The tables are read from the committed per-GPU baselines in
`benchmarks/baselines/` when the site is built, so they are the numbers the
regression gate compares against.

<!-- generated: baselines -->

![Environment-steps per second against the number of parallel environments for Euler, symplectic and RK4, log-log](figures/throughput-light.svg#only-light)
![Environment-steps per second against the number of parallel environments for Euler, symplectic and RK4, log-log](figures/throughput-dark.svg#only-dark)

/// caption
Environment-steps per second (steps/s × environments) against the number of
parallel environments, single body, from the committed RTX 4070 Ti SUPER
baseline in the table above: still rising at 16 384 environments, the
largest batch the baseline records. Measured, not test-checked; drawn from
`benchmarks/baselines/` when the site is built.
///

## Why float64 limits the step on consumer GPUs

Each integrator step is one fused kernel per entity, bound by FP64 arithmetic
throughput rather than by memory or register spills. Consumer (GeForce) GPUs
execute FP64 at 1/64 of their FP32 rate: on the RTX 4070 Ti SUPER the
symplectic step at 2²⁰ environments ran at 3.5 × 10⁸ env-steps/s (measured
once, 2026-10-28), close to the card's measured FP64 peak of 0.61 TFLOP/s.
Removing gravity and J2 speeds the step up only in proportion to the FP64
work removed, and Euler (one force evaluation per step) runs about twice as
fast as the symplectic scheme (two). A 16-core CPU running the same kernels
reaches a comparable FP64 rate, so this GPU is only 2–7× faster than such a
CPU. GPUs with full-rate FP64 (datacenter parts) have not been measured.

## Method

```bash
python benchmarks/bench.py --save           # record this GPU's baseline (first run)
python benchmarks/bench.py                  # compare: exit 1 on a >15 % regression
python benchmarks/bench.py --threshold 10   # stricter gate
```

The script warms up first (JIT compilation and graph capture excluded),
brackets each timed region with `wp.synchronize()` and reports the median of
repeated runs. Every GPU gets its own baseline; never compare across GPUs.
The differentiable-window rows need torch and are skipped without it.

To measure your own batch sizes, see the
[scalability example](examples/04_scalability.md).

# Benchmarks

Throughput of `simulate()` (CUDA graph replay, float64, gravity + J2,
`dt = 0.01 s`) from `benchmarks/bench.py`: steps/s of the whole batch, with
entity-steps/s (steps/s × entities) in parentheses. Single-body rows run
with termination checks off and no thrust. The tables are the committed
per-GPU baselines in `benchmarks/baselines/`, the numbers the regression
gate compares against.

<!-- generated: baselines -->

![Environment-steps per second against the number of parallel environments for Euler, symplectic and RK4, log-log](figures/throughput-light.svg#only-light)
![Environment-steps per second against the number of parallel environments for Euler, symplectic and RK4, log-log](figures/throughput-dark.svg#only-dark)

/// caption
Environment-steps per second against parallel environments, single body,
from the RTX 4070 Ti SUPER baseline above: still rising at 16 384, the
largest batch recorded. Drawn from `benchmarks/baselines/` at site build.
///

## Why float64 limits the step

The step is bound by FP64 arithmetic, not by memory or register spills.
Consumer (GeForce) GPUs run FP64 at 1/64 of their FP32 rate: on the RTX
4070 Ti SUPER the symplectic step reaches 3.5 × 10⁸ env-steps/s at 2²⁰
environments, close to the card's measured FP64 peak, and only 2–7× a
16-core CPU running the same kernels. GPUs with full-rate FP64 (datacenter
parts) have not been measured. The measurements behind this are in the
[experiment record](https://github.com/Pedro-g-Lima/COSMIC_WARP/blob/main/docs/data/README.md).

## Method

```bash
python benchmarks/bench.py --save           # record this GPU's baseline (first run)
python benchmarks/bench.py                  # compare: exit 1 on a >15 % regression
python benchmarks/bench.py --threshold 10   # stricter gate
```

The script warms up first (JIT compilation and graph capture excluded),
brackets each timed region with `wp.synchronize()` and reports the median of
repeated runs. Every GPU gets its own baseline; never compare across GPUs or
Warp versions. The differentiable-window rows need torch. To time your own
batch sizes, see the [scalability example](examples/04_scalability.md).

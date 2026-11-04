# Experiment record

One entry per experiment: the question, the answer, its data, how to
reproduce it, and the version and date it was measured. Unless noted, the
hardware is an NVIDIA GeForce RTX 4070 Ti SUPER (driver 580) with an AMD
Ryzen 9 9950X, float64, and the numbers hold only for that setup.

This record is not on the docs site. Claims a test checks live on the
[validation pages](../validation/index.md); measured throughput on
[Benchmarks](../benchmarks.md). This is the history behind them, kept so
nobody reruns an experiment to rediscover its answer.

## Step throughput

- **What limits the step on a consumer GPU?** FP64 arithmetic. GeForce cards
  run FP64 at 1/64 of the FP32 rate; this card's measured FP64 peak is
  3.06 × 10¹¹ DFMA/s (0.61 TFLOP/s, 8 independent FMA chains per thread).
  Symplectic at 2²⁰ environments ran at 3.49 × 10⁸ env-steps/s, about 880
  FP64 instructions per env-step at that peak; the kernel's static SASS has
  about 1530 (891 DFMA, 456 DMUL, 183 DADD, plus division, square-root and
  trigonometric calls on branches not all taken). Removing gravity and J2
  (about 60 FP64 operations) sped the step up 6.6 %; Euler (one force
  evaluation) runs about 2× symplectic (two). Not confirmed with Nsight
  Compute. Faster steps need fewer FP64 operations per step or a full-rate
  FP64 GPU (not measured).
  `python examples/04_scalability.py --env-counts 1048576 --integrators symplectic` · Warp 1.18.0 · 0.1.0 development, 2026-10-28
- **How much faster is the GPU than a CPU running the same kernels?** 2–7×
  a 16-core CPU, which has a comparable FP64 rate. Symplectic, env-steps/s,
  GPU / 16 CPU processes / 1 CPU core: 9.3 × 10⁷ / 4.8 × 10⁷ / 6.0 × 10⁶ at
  4096 environments; 2.7 × 10⁸ / 8.0 × 10⁷ / 5.9 × 10⁶ at 65,536;
  3.8 × 10⁸ / 5.4 × 10⁷ / 5.9 × 10⁶ at 1,048,576. Both devices gave identical
  positions after 200 steps from the same initial states.
  `python examples/04_scalability.py --device cpu --integrators symplectic` (one core; the 16-process column split the batch over 16 processes) · Warp 1.16.0 · 0.1.0 development, 2026-10-20
- **Is host work or a debug setting slowing the GPU loop?** No. The host
  queues 200 steps in 0.4 ms while the GPU runs them in 588 ms (10⁶
  environments); no debug or verify flag was set; the step kernel is 99.9 %
  of GPU time per step.
  `python examples/04_scalability.py --env-counts 1048576` · Warp 1.16.0 · 0.1.0 development, 2026-10-20
- **Do register spills limit the step kernels?** Not any more. All three use
  255 registers per thread, at 17 % occupancy (8 of 48 warps per SM at 256
  threads). Evaluating the thruster wrench once per step instead of per
  force evaluation cut local memory per thread from 504 to 80 B
  (symplectic), 656 to 472 B (RK4) and 8 to 0 B (Euler).
  `cuFuncGetAttribute` (registers, local size) and `cuOccupancyMaxActiveBlocksPerMultiprocessor` on the cached PTX of `cosmic_warp.core.integrators.<scheme>`, through `cuda.bindings` · Warp 1.16.0 → 1.18.0 · 0.1.0 development, 2026-10-20 and 2026-10-28
- **Does capping the registers pay?** Not enough to adopt.
  `@wp.kernel(cuda_max_registers=128)` doubles occupancy but spills more
  (symplectic 592 B, RK4 1192 B): +12 % symplectic, +13 % Euler and ±0 RK4
  at 10⁶ environments; 0 to −10 % at the benchmark sizes (≤ 16,384
  environments, too few blocks for occupancy to matter). Caps of 96 and 168
  are worse, and the option needs Warp ≥ 1.18.
  `python benchmarks/bench.py` with the cap set on the step kernels · Warp 1.18.0 · 0.1.0 development, 2026-10-28
- **Does accumulating the RK4 stage sum incrementally help?** No measurable
  change: the compiler already shortens those lifetimes.
  `python benchmarks/bench.py` · Warp 1.18.0 · 0.1.0 development, 2026-10-28
- **Does the Warp version change the step rate?** Yes: the same code ran
  5–12 % slower on every integrator under Warp 1.18.0 than under 1.15.0, at
  the benchmark sizes, so compare baselines only at one Warp version.
  `python benchmarks/bench.py` under each Warp version · 0.1.0 development, 2026-10-28
- **What does the RL step cost over `simulate()`?** About 20 % more time per
  step: 42 M env-steps/s against 51 M for `simulate()` on the same
  simulator (example 13's configuration: symplectic, J2 and gravity
  gradient, three wheels, 4096 environments). The RL step runs ten launches
  against two — action mapping, reward, done reduction, two observation
  writes, the terminal copy and three masked reset kernels — every step.
  `python examples/13_rl_attitude_env.py` (prints the throughput) · 0.1.0 development, 2026-10-29

## Numerics

- **Why is the step-end mass rounded once?** With `mass_limit = mass_dry`,
  two roundings per symplectic step (one per half kick) left about a quarter
  of 4096 entities one ulp above the dry mass on the burnout step, so they
  never terminated. Every integrator now writes the step-end mass once, and
  exactly `mass_dry` on the burnout step.
  `pytest tests/test_pipeline.py -k burnout_step` · 0.1.0 development
- **Why is the damped wheel's relaxation factor not computed directly?**
  `1 − exp(−bt/J)` cancels to zero for tiny damping: at b = 1e-18 N·m·s the
  wheel stayed at rest. Warp has no `expm1`, so the factor
  `(1 − e^(−x))/x` uses Kahan's form, accurate to a few ulps, and tends to
  the undamped wheel as b → 0.
  `pytest tests/test_rw.py -k any_damping` · 0.1.0 development
- **Why does `state_to_elements` not use `acos` for the inclination?** It
  returned 1.49e-8 rad for an orbit at 1e-8 rad. The `atan2` form recovers
  inclinations within 1e-15 rad at 1e-8 and π − 1e-8.
  `pytest tests/test_orbit_sampling.py -k near_equatorial` · 0.1.0 development

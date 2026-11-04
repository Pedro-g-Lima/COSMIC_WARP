#!/usr/bin/env python
"""
Tier 1 performance regression benchmark.

Measures simulate() throughput (steps/s) across integrators and environment
counts — plus, when torch is installed, differentiable-RL window timings
(a DiffVecEnv window rollout, and rollout + backward, with a linear torch
policy) — and compares against a committed per-GPU baseline. Fails (exit 1)
if any metric regresses beyond the threshold.

Usage:
    python benchmarks/bench.py             # run + compare against this GPU's baseline
    python benchmarks/bench.py --save      # run + (over)write this GPU's baseline
    python benchmarks/bench.py --threshold 20   # regression threshold in percent

Methodology (GPU benchmarking essentials):
  - warmup run first so JIT kernel compilation and graph capture are excluded
  - wp.synchronize() brackets every timed region (launches are async)
  - median of repeats (a single clock-throttle event skews a mean)
  - baselines are keyed by GPU name: numbers are never comparable across GPUs

Caveat for laptop GPUs: boost clocks vary with thermal state between sessions,
so small-workload cells (few envs) can swing +/-20% with unchanged code. Trust
the large-env (compute-bound) cells for regression decisions, or lock clocks
(sudo nvidia-smi -lgc <freq>) before recording a baseline.
"""
import argparse
import datetime
import json
import platform
import re
import statistics
import sys
import time
from pathlib import Path

import warp as wp

import cosmic_warp
from cosmic_warp.core.pipeline import SpacecraftSimulator

INTEGRATORS = ["symplectic", "rk4", "euler"]
ENV_COUNTS = [16, 1024, 16384]
NUM_STEPS = 2000
REPEATS = 5
# Extra cells at 16384 entities: thrusters firing (the default cells sit at
# dry mass, where the thruster wrench returns early) and 4-body environments
# (multi-body termination reduction)
CASE_ENTITIES = 16384
DIFF_ENV_COUNTS = [1024, 4096]
DIFF_HORIZON = 32
BASELINE_DIR = Path(__file__).parent / "baselines"


def gpu_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def measure(integrator: str, num_envs: int, num_bodies: int = 1,
            thrusting: bool = False) -> float:
    """Median simulate() throughput in steps/s."""
    sim = SpacecraftSimulator(num_envs=num_envs, num_bodies=num_bodies,
                              integrator=integrator, dt=0.01,
                              check_bounds=num_bodies > 1)
    if thrusting:
        # 9 kg of propellant outlasts the ~100 s of timed steps at this flow
        sim.thruster_manager.set_config(max_thrust=10.0, isp=200.0)
        sim.reset_manager.set_bounds(mass_bounds=(10.0, 10.0))
    sim.reset()
    if thrusting:
        sim.set_thrust_commands([0.5] * 8)   # reset() clears commands

    # Warmup: kernel compilation + first graph capture
    sim.simulate(20, save_states=False)
    wp.synchronize()

    rates = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        sim.simulate(NUM_STEPS, save_states=False)
        wp.synchronize()
        rates.append(NUM_STEPS / (time.perf_counter() - t0))
    return statistics.median(rates)


def measure_diff_window(num_envs: int) -> dict:
    """Median DiffVecEnv window rates (windows/s): the taped H-step rollout
    alone, and rollout + backward to a linear policy's weights.

    Returns {} when torch is not installed (metrics simply absent).
    """
    try:
        import torch
        from cosmic_warp.rl.diff_env import DiffVecEnv
    except ImportError:
        return {}

    class WheelTask:
        """Trivial DiffTask: wheel torques in, raw rotational state out,
        penalize body angular momentum."""
        act_dim, obs_dim = 3, 11

        def bind(self, sim):
            pass

        def commands(self, action):
            pad8 = torch.zeros((action.shape[0], 8), dtype=action.dtype, device=action.device)
            return torch.cat([pad8, 0.02 * action, pad8[:, :1]], dim=1)

        def observation(self, x):
            return x[:, 7:18]   # quat, ang_mom, wheel momenta

        def reward(self, x, action):
            return -(x[:, 11:14] ** 2).sum(dim=1)

    sim = SpacecraftSimulator(num_envs=num_envs, integrator="symplectic",
                              dt=0.1, check_bounds=False, differentiable=True)
    sim.reset_manager.set_bounds(att_max_angle=0.5236,
                                 ang_mom_bounds=((-0.2,) * 3, (0.2,) * 3))
    env = DiffVecEnv(sim, WheelTask(), horizon=DIFF_HORIZON,
                     max_episode_steps=6 * DIFF_HORIZON)
    torch.manual_seed(0)
    weights = (0.1 * torch.randn(3, 11, dtype=torch.float64, device=env.device)
               ).requires_grad_(True)

    def window_loss():
        window = env.rollout(lambda obs: torch.tanh(obs @ weights.T))
        return -torch.stack(window.rewards).mean()

    # Warmup: adjoint kernel compilation + allocator
    window_loss().backward()
    env.end_window()
    wp.synchronize()

    rollout_rates, window_rates = [], []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        loss = window_loss()
        wp.synchronize()
        t1 = time.perf_counter()
        loss.backward()
        wp.synchronize()
        t2 = time.perf_counter()
        env.end_window()
        rollout_rates.append(1.0 / (t1 - t0))
        window_rates.append(1.0 / (t2 - t0))
    return {
        f"diff_rollout/envs={num_envs}": round(statistics.median(rollout_rates), 2),
        f"diff_rollout_backward/envs={num_envs}": round(statistics.median(window_rates), 2),
    }


def run_benchmarks() -> dict:
    wp.init()
    device = wp.get_device()
    results = {}
    print(f"GPU: {device.name} | warp {wp.__version__} | "
          f"{NUM_STEPS} steps x {REPEATS} repeats (median)\n")
    print(f"{'integrator':<12} {'envs':>7} {'steps/s':>12} {'entity-steps/s':>16}")
    for integrator in INTEGRATORS:
        for num_envs in ENV_COUNTS:
            rate = measure(integrator, num_envs)
            results[f"{integrator}/envs={num_envs}"] = round(rate, 1)
            print(f"{integrator:<12} {num_envs:>7} {rate:>12.0f} {rate * num_envs:>16.3e}")

    print(f"\n{'case':<24} {'entities':>8} {'steps/s':>12} {'entity-steps/s':>16}")
    for integrator in INTEGRATORS:
        for case, kwargs in (("thrusting", {"num_envs": CASE_ENTITIES, "thrusting": True}),
                             ("bodies=4", {"num_envs": CASE_ENTITIES // 4, "num_bodies": 4})):
            rate = measure(integrator, **kwargs)
            key = f"{integrator}/{case}/entities={CASE_ENTITIES}"
            results[key] = round(rate, 1)
            print(f"{integrator + '/' + case:<24} {CASE_ENTITIES:>8} {rate:>12.0f} "
                  f"{rate * CASE_ENTITIES:>16.3e}")

    diff_header_printed = False
    for num_envs in DIFF_ENV_COUNTS:
        diff_results = measure_diff_window(num_envs)
        if not diff_results:
            print("\nDifferentiable window benchmarks skipped (torch not installed)")
            break
        if not diff_header_printed:
            print(f"\n{'window (H=' + str(DIFF_HORIZON) + ')':<18} {'envs':>7} "
                  f"{'/s':>10} {'env-steps/s':>16}")
            diff_header_printed = True
        results.update(diff_results)
        for key, rate in diff_results.items():
            kind = key.split("/")[0].removeprefix("diff_")
            print(f"{kind:<18} {num_envs:>7} {rate:>10.2f} "
                  f"{rate * DIFF_HORIZON * num_envs:>16.3e}")
    return {
        "meta": {
            "gpu": device.name,
            "cuda_driver": ".".join(map(str, wp.get_cuda_driver_version())),
            "warp": wp.__version__,
            "torch": _torch_version(),
            "python": platform.python_version(),
            "date": datetime.date.today().isoformat(),
            "cosmic_warp": cosmic_warp.__version__,
            "num_steps": NUM_STEPS,
            "repeats": REPEATS,
            "diff_horizon": DIFF_HORIZON,
        },
        "results": results,
    }


def _torch_version() -> str:
    try:
        import torch
        return torch.__version__
    except ImportError:
        return "not installed"


def compare(current: dict, baseline: dict, threshold_pct: float) -> int:
    meta = baseline["meta"]
    print(f"\nBaseline: {meta['date']} (cosmic_warp {meta.get('cosmic_warp', '?')}) on {meta['gpu']} "
          f"(threshold {threshold_pct:.0f}%)\n")
    print(f"{'metric':<28} {'baseline':>12} {'current':>12} {'delta':>8}")

    regressions, missing = [], []
    for key, base_rate in baseline["results"].items():
        cur_rate = current["results"].get(key)
        if cur_rate is None:
            print(f"{key:<28} {base_rate:>12.0f} {'missing':>12}")
            missing.append(key)
            continue
        delta = 100.0 * (cur_rate - base_rate) / base_rate
        flag = ""
        if delta < -threshold_pct:
            flag = "  << REGRESSION"
            regressions.append((key, delta))
        print(f"{key:<28} {base_rate:>12.0f} {cur_rate:>12.0f} {delta:>+7.1f}%{flag}")

    if regressions:
        print(f"\nFAIL: {len(regressions)} metric(s) regressed beyond "
              f"{threshold_pct:.0f}%:")
        for key, delta in regressions:
            print(f"  {key}: {delta:+.1f}%")
    # An unmeasured metric is not a passed one: e.g. the diff_* rows need torch
    if missing:
        print(f"\nFAIL: {len(missing)} baseline metric(s) not measured in this run "
              f"(missing dependency?): {', '.join(missing)}")
    if regressions or missing:
        return 1
    print("\nOK: no regressions beyond threshold.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true",
                        help="write results as the baseline for this GPU")
    parser.add_argument("--threshold", type=float, default=15.0,
                        help="regression threshold in percent (default 15)")
    args = parser.parse_args()

    current = run_benchmarks()
    baseline_path = BASELINE_DIR / f"{gpu_slug(current['meta']['gpu'])}.json"

    if args.save:
        BASELINE_DIR.mkdir(exist_ok=True)
        baseline_path.write_text(json.dumps(current, indent=2) + "\n")
        print(f"\nBaseline saved to {baseline_path}")
        return 0

    if not baseline_path.exists():
        print(f"\nNo baseline for this GPU ({current['meta']['gpu']}).")
        print(f"Create one with: python benchmarks/bench.py --save")
        return 2

    baseline = json.loads(baseline_path.read_text())
    return compare(current, baseline, args.threshold)


if __name__ == "__main__":
    sys.exit(main())

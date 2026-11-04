"""
Every example script runs: the docs gallery (docs/examples/) is generated
from examples/*.py, and a page documenting a script that no longer works is
worse than no page.

Each script is run as a user runs it, `python examples/<script> ... --device
<device>`, in a subprocess with the test's tmp_path as the working directory
(the scripts save their plots there) and the non-interactive Agg backend.
The flags are the smallest that still go through every code path of the
script: a few environments, steps and training iterations. Each run must exit
with status 0 within its time budget, print its closing line, write exactly
the PNGs it documents into the working directory, and leave nothing behind in
the repository.

`attitude_task.py` and `shac.py` are modules, not scripts: examples 13-17
import them from the script's directory, so they are exercised here through
those scripts, and their behavior is tested in test_example_attitude_task.py
and test_example_shac.py.

Optional dependencies are checked in this process (`importlib.util.find_spec`)
and a script that needs a missing one is skipped with the reason.
"""
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

import pytest

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"

# Measured wall time of each run below, as `--device cpu` / `--device cuda`
# (pytest --durations, warm Warp kernel cache, Ryzen 9 9950X / RTX 4070 Ti SUPER):
#   01 0.9/1.1 s, 02 0.8/1.0, 03 0.8/0.9, 04 0.4/0.5, 05 0.9/1.0, 06 0.9/1.1,
#   07 0.9/0.9, 08 1.3/1.4, 09 0.5/0.6, 10 0.7/0.9, 11 0.7/0.8, 12 0.7/0.8,
#   13 0.4/0.5, 14 2.0/2.3, 15 2.0/2.3, 16 3.6/4.4, 17 1.9/2.3,
#   run_config 0.3-0.5 s per YAML file; about 22 s for the whole file.
# With an empty kernel cache (cold compile) on CPU the longest were 01 3.8 s,
# 11 2.5 s and 16 8.2 s. The budgets leave a wide margin for slow CI runners;
# none of the runs is long enough to need the `slow` marker.
BUDGET_SECONDS = 60.0
BUDGET_RL_SECONDS = 120.0  # torch import, and rl_games' torch.compile


class Case(NamedTuple):
    script: str
    args: tuple = ()
    needs: tuple = ()      # optional modules, checked with find_spec
    pngs: tuple = ()       # plots the script saves to the working directory
    expect: str = ""       # a line the script prints when it finishes
    budget: float = BUDGET_SECONDS


PLOT = ("matplotlib",)

SCRIPT_CASES = [
    Case("01_basic_orbit.py", ("--num-envs", "2", "--num-steps", "200", "--no-show"), PLOT,
         ("orbital_timeseries.png", "orbital_3d.png", "orbital_altitude_velocity.png"),
         "Example complete!"),
    Case("02_integrator_comparison.py", ("--num-steps", "200", "--no-show"), PLOT,
         ("integrator_comparison.png",), "Example complete!"),
    Case("03_parallel_monte_carlo.py",
         ("--num-envs", "64", "--num-steps", "100", "--seed", "0", "--no-show"), PLOT,
         ("monte_carlo_analysis.png",), "Example complete!"),
    Case("04_scalability.py", ("--env-counts", "1", "8", "--steps", "20"),
         expect="Scalability benchmark complete."),
    Case("05_energy_conservation.py", ("--num-steps", "2000", "--no-show"), PLOT,
         ("energy_conservation_comparison.png",), "Plot saved to:"),
    Case("06_thrusters.py", ("--num-envs", "16", "--num-steps", "50", "--no-show"), PLOT,
         ("planar_thrusters_trajectory.png", "planar_thrusters_velocity.png",
          "planar_thrusters_angular_velocity.png", "planar_thrusters_mass.png"),
         "Simulation complete:"),
    Case("07_reaction_wheels.py", ("--num-steps", "500", "--no-show"), PLOT,
         ("rw_wheels.png", "rw_angvel.png"), "Done."),
    Case("08_j2_comparison.py", ("--num-steps", "500", "--no-show"), PLOT,
         ("j2_comparison.png",), "Example complete!"),
    # Shows its plots instead of saving them, so with --no-show it saves none
    Case("09_multi_body_orbit.py", ("--steps", "100", "--no-show"), PLOT,
         (), "Runtime:"),
    Case("10_domain_randomization.py", ("--num-envs", "64", "--num-steps", "50", "--seed", "0"),
         PLOT, ("domain_randomization.png",), "Example complete!"),
    Case("11_gradient_optimization.py", ("--num-steps", "50", "--iters", "3", "--no-show"),
         PLOT, ("gradient_optimization.png",), "Delta-v found:"),
    Case("12_system_identification.py",
         ("--num-steps", "50", "--iters", "2", "--obs-every", "10", "--no-show"), PLOT,
         ("system_identification.png",), "Recovered cd ="),
    Case("13_rl_attitude_env.py", ("--num-envs", "16", "--steps", "20"),
         expect="final mean rate (detumble)"),
    Case("14_rl_rsl_training.py", ("--num-envs", "16", "--iterations", "2"),
         ("torch", "rsl_rl"), expect="after training", budget=BUDGET_RL_SECONDS),
    Case("15_rl_skrl_training.py", ("--num-envs", "16", "--iterations", "2"),
         ("torch", "skrl"), expect="after training", budget=BUDGET_RL_SECONDS),
    # Writes its checkpoint and summaries under runs/ in the working directory
    Case("16_rl_rlgames_training.py", ("--num-envs", "16", "--epochs", "2"),
         ("torch", "rl_games"), expect="after training", budget=BUDGET_RL_SECONDS),
    Case("17_rl_shac_training.py", ("--num-envs", "16", "--updates", "2", "--horizon", "8"),
         ("torch",), expect="after training", budget=BUDGET_RL_SECONDS),
]

# run_config.py on every YAML scenario, in the mode its header describes
CONFIG_MODES = {
    "minimal_orbit.yaml": (),
    "monte_carlo_dispersion.yaml": ("--monte-carlo",),
    "multibody_orbit.yaml": (),
    "rendezvous_chaser_target.yaml": (),
    "rl_attitude_training.yaml": ("--auto-reset",),
}
CONFIG_CASES = [
    Case("run_config.py", (str(EXAMPLES / name), "--steps", "20", *mode), expect="wall time:")
    for name, mode in CONFIG_MODES.items()
]

CASES = (
    [pytest.param(c, id=Path(c.script).stem) for c in SCRIPT_CASES]
    + [pytest.param(c, id=f"run_config-{Path(c.args[0]).stem}") for c in CONFIG_CASES]
)


def _is_script(path: Path) -> bool:
    return 'if __name__ == "__main__":' in path.read_text(encoding="utf-8")


def _stray_candidates():
    """Untracked and ignored files where a script could write by mistake: the
    repository root (where users run the scripts from), new top-level
    directories (e.g. a trainer's runs/) and examples/ (the script's own
    directory). Bytecode and the tool caches (.pytest_cache/, .cache/) are
    excluded, and so are the other tracked directories, which other tools may
    be editing while the suite runs. None when the tree is not a git checkout."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                              check=True, timeout=60).stdout
    try:
        tracked = {line.split("/", 1)[0] for line in git("ls-files").splitlines() if "/" in line}
        status = git("status", "--porcelain", "--ignored", "--untracked-files=all")
    except (OSError, subprocess.SubprocessError):
        return None
    paths = {line[3:] for line in status.splitlines() if line[:2] in ("??", "!!")}
    return {
        path for path in paths
        if "__pycache__" not in path and not path.endswith(".pyc")
        and not path.startswith((".pytest_cache/", ".cache/"))   # tool caches, other runs
        and (path.split("/", 1)[0] not in tracked or path.startswith("examples/"))
    }


def test_every_example_is_covered():
    """A new example script or YAML scenario cannot be silently left untested."""
    scripts = {p.name for p in EXAMPLES.glob("*.py") if _is_script(p)}
    assert scripts == {c.script for c in SCRIPT_CASES} | {"run_config.py"}
    yamls = {p.name for p in EXAMPLES.glob("*.yaml")}
    assert yamls == set(CONFIG_MODES)
    # The modules the scripts import have their own tests
    modules = {p.stem for p in EXAMPLES.glob("*.py") if not _is_script(p)}
    assert modules == {"attitude_task", "shac"}
    for stem in modules:
        assert (REPO / "tests" / f"test_example_{stem}.py").is_file()


@pytest.mark.parametrize("case", CASES)
def test_example_runs(case, device, tmp_path):
    missing = [m for m in case.needs if importlib.util.find_spec(m) is None]
    if missing:
        pytest.skip(f"{case.script} needs {', '.join(missing)}")

    before = _stray_candidates()
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1"}
    cmd = [sys.executable, str(EXAMPLES / case.script), *case.args, "--device", device]
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True,
                            timeout=2 * case.budget)
    elapsed = time.perf_counter() - start

    log = f"$ {' '.join(cmd)}\n--- stdout\n{result.stdout[-4000:]}\n--- stderr\n{result.stderr[-4000:]}"
    assert result.returncode == 0, log
    assert elapsed < case.budget, f"{case.script} took {elapsed:.1f} s (budget {case.budget:.0f} s)"
    assert case.expect in result.stdout, log
    if case.script == "run_config.py":
        assert "non-finite state" not in result.stdout, log

    # The plots land in the working directory, and only the documented ones
    assert sorted(p.name for p in tmp_path.glob("*.png")) == sorted(case.pngs), log
    for name in case.pngs:
        assert (tmp_path / name).stat().st_size > 0

    after = _stray_candidates()
    if before is not None and after is not None:
        assert after <= before, f"{case.script} wrote into the repository: {sorted(after - before)}"

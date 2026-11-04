"""The docs' validation figures, each drawn from the computation its test checks.

Figures carry no title: the page's caption says what each shows, what to read
from it and which test holds the data, which keeps the text searchable and the
images reusable elsewhere under their own captions.

Every figure's data comes from the test that holds the claim — loaded by path,
since ``tests/`` is not a package — so a figure cannot show something the
suite does not compute. Data is computed once, on CPU, and drawn twice: a
light and a dark SVG, which the pages pair with Material's ``#only-light`` and
``#only-dark``. ``scripts/gen_figures.py`` runs this at docs build time; the
data is cached under ``.cache/figures/``, keyed by every source it depends on,
so ``mkdocs serve`` recomputes only when the code changes.

    python scripts/figures.py [--out DIR] [NAME ...]   # write the SVGs, for a look

Needs the ``dev`` dependency group (pytest, the pinned Warp, matplotlib).
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TESTS = REPO / "tests"
DEVICE = "cpu"
CACHE = REPO / ".cache" / "figures"

# Material's own text colours on its two schemes, and the reference categorical
# palette stepped for each (validated against #ffffff and #1e2129 for three
# series; the series carry direct labels, so colour is never the only key).
THEMES = {
    "light": {"ink": "#212121", "muted": "#737373", "grid": "#e8e8e8",
              "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]},
    "dark": {"ink": "#c8cbd6", "muted": "#8d91a0", "grid": "#33363f",
             "series": ["#3987e5", "#d95926", "#199e70", "#c98500"]},
}
INTEGRATORS = {"euler": "Euler", "symplectic": "symplectic", "rk4": "RK4"}


# -- shared ----------------------------------------------------------------- #

def _load(name):
    """A module of ``tests/`` by path; cached so each loads once. The tests
    import their helpers (``conftest``, ``orbit_utils``) by bare name."""
    if str(TESTS) not in sys.path:
        sys.path.insert(0, str(TESTS))
    key = "figures_" + name
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, TESTS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def _make_sim():
    return _load("conftest").simulator_factory(DEVICE)


def _figure(theme, size=(7.0, 3.6)):
    import matplotlib
    matplotlib.use("svg")
    import matplotlib.pyplot as plt

    t = THEMES[theme]
    plt.rcParams.update({
        "svg.fonttype": "path", "svg.hashsalt": "cosmicwarp", "font.size": 9.5,
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.transparent": True,
        "text.color": t["ink"], "axes.labelcolor": t["ink"],
        "axes.edgecolor": t["muted"], "xtick.color": t["muted"], "ytick.color": t["muted"],
        "xtick.labelcolor": t["ink"], "ytick.labelcolor": t["ink"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": t["grid"], "grid.linewidth": 0.8,
        "lines.linewidth": 1.6, "lines.markersize": 5, "legend.frameon": False,
    })
    fig, ax = plt.subplots(figsize=size, layout="constrained")
    return fig, ax, t


def _svg(fig):
    import matplotlib.pyplot as plt

    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", metadata={"Date": None})
    plt.close(fig)
    return buffer.getvalue()


# -- convergence ------------------------------------------------------------ #

def convergence_data():
    """Position error against the Kepler solution after 2048 s, at the step
    sizes test_convergence_order uses."""
    two_body = _load("test_two_body")
    make = _make_sim()
    return {integrator: (np.array(dts), np.array(two_body.kepler_errors(make, integrator, dts)), order)
            for integrator, dts, order in two_body.CONVERGENCE_CASES}


def convergence_draw(data, theme):
    fig, ax, t = _figure(theme)
    for k, (integrator, (dts, errors, order)) in enumerate(data.items()):
        color = t["series"][k]
        observed = np.mean(np.log2(errors[:-1] / errors[1:]))
        ax.loglog(dts, errors, "o-", color=color)
        # The expected slope, through the finest point, extended a factor 2 finer.
        guide = np.array([dts[-1] / 2, dts[0]])
        ax.loglog(guide, errors[-1] * (guide / dts[-1]) ** order, "--", color=t["muted"], lw=1)
        ax.annotate(f"{INTEGRATORS[integrator]}: observed order {observed:.2f} (expected {order})",
                    (dts[0], errors[0]), xytext=(8, 0), textcoords="offset points", va="center",
                    color=t["ink"])
    ax.set_xlim(0.2, 600)
    ax.set_xlabel("time step dt (s)")
    ax.set_ylabel("position error after 2048 s (m)")
    return _svg(fig)


# -- burnout ---------------------------------------------------------------- #

def burnout_data():
    """The measured burnout position error at the test's three step sizes, and
    near each at burn fractions f across the step."""
    maneuvers = _load("test_maneuvers")
    make = _make_sim()
    out = []
    for k, tested in enumerate(maneuvers.BURNOUT_DTS):
        n = int(maneuvers.BURNOUT_TIME / tested)   # whole steps before burnout, as at the tested dt
        fractions = np.linspace(0.05, 0.95, 10) + 0.03 * (k - 1)   # apart, so every group shows
        dts = np.concatenate([[tested], maneuvers.BURNOUT_TIME / (n + fractions)])
        measured = np.array([maneuvers.burnout_position_error(make, dt)[0] for dt in dts])
        f = maneuvers.BURNOUT_TIME / dts % 1.0
        out.append({"dt": tested, "f": f, "scaled": measured * maneuvers.BURNOUT_M_DRY
                    / (maneuvers.BURNOUT_THRUST * dts**2)})
    return out


def burnout_draw(data, theme):
    fig, ax, t = _figure(theme)
    f = np.linspace(0.0, 1.0, 200)
    ax.plot(f, -f * (1 - f) / 2, color=t["muted"], lw=1.2, zorder=1)
    ax.annotate(r"closed form $-f(1-f)/2$", (0.5, -0.125), xytext=(0, -10), textcoords="offset points",
                ha="center", va="top", color=t["ink"])
    for k, group in enumerate(data):
        color = t["series"][k]
        ax.plot(group["f"][1:], group["scaled"][1:], "o", color=color, mfc="none", mew=1.4,
                label=f"dt ≈ {group['dt']} s")
        ax.plot(group["f"][:1], group["scaled"][:1], "o", color=color, ms=7, zorder=3)
    ax.legend(loc="upper center", ncols=3, title="filled: the step sizes the test checks", title_fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.15, 0.03)
    ax.set_xlabel("fraction f of the burnout step the propellant lasts")
    ax.set_ylabel(r"position error $\times\ m_\mathrm{dry} / (T\,\Delta t^2)$")
    return _svg(fig)


# -- reaction wheels -------------------------------------------------------- #

def wheels_data():
    """The saturating and the damped wheel of test_rw.py, their first 2 s."""
    rw = _load("test_rw")
    make = _make_sim()
    s, d = rw.SATURATION, rw.DAMPED
    _, result = rw.saturating_wheel(make)
    saturating = result["rw_angular_velocities"].numpy()[:, 0, 0] * s["J_w"]
    damped = rw.damped_wheel(make)
    t_sat = (np.arange(len(saturating)) + 1) * s["dt"]
    t_damp = (np.arange(len(damped)) + 1) * d["dt"]
    keep = t_damp <= t_sat[-1] + 1e-9
    fine = np.linspace(0.0, t_sat[-1], 400)
    return {
        "saturating": (t_sat, saturating, np.minimum(s["tau"] * fine, s["L_max"])),
        "damped": (t_damp[keep], damped[keep],
                   d["tau"] * d["J_w"] / d["b"] * (1.0 - np.exp(-d["b"] / d["J_w"] * fine))),
        "fine": fine, "params": (s, d),
    }


def wheels_draw(data, theme):
    fig, ax, t = _figure(theme)
    s, d = data["params"]
    labels = {"saturating": rf"undamped, $\tau$ = {s['tau']} N·m: $\tau t$ up to the capacity {s['L_max']} N·m·s",
              "damped": rf"damped, $J/b$ = {d['J_w'] / d['b']:.2f} s: "
                        r"$\tau (J/b)\,(1 - e^{-bt/J})$"}
    for k, key in enumerate(("saturating", "damped")):
        times, simulated, exact = data[key]
        color = t["series"][k]
        every = max(1, len(times) // 25)
        ax.plot(data["fine"], exact, color=color)
        ax.plot(times[::every], simulated[::every], "o", color=color, mfc="none", mew=1.4)
        ax.annotate(labels[key], (data["fine"][-1], exact[-1]), xytext=(0, 7 if k == 0 else -8),
                    textcoords="offset points", ha="right", va="bottom" if k == 0 else "top", color=t["ink"])
    ax.set_xlim(0, data["fine"][-1])
    ax.set_ylim(0, s["L_max"] * 1.25)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("wheel momentum (N·m·s)")
    return _svg(fig)


# -- J2 --------------------------------------------------------------------- #

def j2_data():
    j2 = _load("test_j2")
    make = _make_sim()
    return {"node": j2.nodal_regression(make), "periapsis": j2.apsidal_rotation(make)}


def j2_draw(data, theme):
    fig, ax, t = _figure(theme)
    day = 86400.0
    for k, (key, name, sym) in enumerate((("node", "node, circular 500 km, i = 51.6°", r"\Omega"),
                                          ("periapsis", "periapsis, e = 0.1, i = 30°", r"\omega"))):
        times, angle, rate = data[key]
        color = t["series"][k]
        fitted = np.polyfit(times, angle, 1)[0]
        hours = times / 3600
        ax.plot(hours, np.degrees(angle - angle[0]), color=color, lw=1.2)
        ax.plot(hours, np.degrees(rate * (times - times[0])), "--", color=t["muted"], lw=1)
        ax.text(0.02, 0.95 if rate > 0 else 0.06,
                rf"{name}: fitted $\dot{{{sym}}}$ = {np.degrees(fitted) * day:.2f} °/day, "
                f"theory {np.degrees(rate) * day:.2f}", transform=ax.transAxes,
                va="top" if rate > 0 else "bottom", color=t["ink"])
    ax.axhline(0, color=t["muted"], lw=0.8)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    ax.set_xlabel("time (h)")
    ax.set_ylabel("change in angle (°)")
    return _svg(fig)


# -- throughput ------------------------------------------------------------- #

BASELINE = "benchmarks/baselines/nvidia_geforce_rtx_4070_ti_super.json"


def throughput_data():
    results = json.loads((REPO / BASELINE).read_text(encoding="utf-8"))["results"]
    out = {}
    for integrator in INTEGRATORS:
        points = sorted((int(k.split("envs=")[1]), v) for k, v in results.items()
                        if k.startswith(f"{integrator}/envs="))
        out[integrator] = np.array(points, dtype=float)
    return out


def throughput_draw(data, theme):
    fig, ax, t = _figure(theme)
    for k, integrator in enumerate(("euler", "symplectic", "rk4")):
        n, steps = data[integrator].T
        ax.loglog(n, steps * n, "o-", color=t["series"][k])
        ax.annotate(INTEGRATORS[integrator], (n[-1], steps[-1] * n[-1]), xytext=(6, (0, 3, -3)[k]),
                    textcoords="offset points", va="center", color=t["ink"])
    ax.set_xlim(right=ax.get_xlim()[1] * 3)
    ax.set_xlabel("parallel environments")
    ax.set_ylabel("environment-steps per second")
    return _svg(fig)


FIGURES = {
    "convergence": (convergence_data, convergence_draw),
    "burnout": (burnout_data, burnout_draw),
    "wheels": (wheels_data, wheels_draw),
    "j2": (j2_data, j2_draw),
    "throughput": (throughput_data, throughput_draw),
}


# -- cache and output -------------------------------------------------------- #

def sources_key():
    """Every input a figure's data depends on: the package, the tests it loads,
    the baselines, this file, and the library versions."""
    digest = hashlib.sha256()
    for pattern in ["cosmic_warp/**/*.py", "tests/*.py", "benchmarks/baselines/*.json", "scripts/figures.py"]:
        for path in sorted(REPO.glob(pattern)):
            digest.update(str(path.relative_to(REPO)).encode())
            digest.update(path.read_bytes())
    for package in ("warp-lang", "numpy"):
        try:
            digest.update(f"{package}={importlib.metadata.version(package)}".encode())
        except importlib.metadata.PackageNotFoundError:
            digest.update(f"{package}=absent".encode())
    return digest.hexdigest()[:16]


def data(name, key=None):
    """A figure's data, from the cache when its sources are unchanged."""
    key = key or sources_key()
    path = CACHE / f"{name}-{key}.pkl"
    if path.is_file():
        return pickle.loads(path.read_bytes())
    value = FIGURES[name][0]()
    CACHE.mkdir(parents=True, exist_ok=True)
    for stale in CACHE.glob(f"{name}-*.pkl"):
        stale.unlink()
    path.write_bytes(pickle.dumps(value))
    return value


def render(name, key=None):
    """{theme: svg text} for one figure."""
    value = data(name, key)
    return {theme: FIGURES[name][1](value, theme) for theme in THEMES}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("names", nargs="*", default=list(FIGURES))
    parser.add_argument("--out", type=Path, default=REPO / ".cache" / "figures-preview")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    key = sources_key()
    for name in args.names:
        for theme, svg in render(name, key).items():
            (args.out / f"{name}-{theme}.svg").write_text(svg, encoding="utf-8")
        print(f"{name}: written to {args.out}")

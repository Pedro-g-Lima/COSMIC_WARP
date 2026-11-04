"""Fill ``<!-- generated: NAME -->`` markers in the docs from the code.

Run by mkdocs-gen-files during the docs build, after the other scripts. Each
marker is replaced by a table built from the source of truth, so the page
cannot drift from it:

- ``requirements`` and ``extras``: ``pyproject.toml`` (each extra's comment
  is its description);
- ``termination-reasons``: ``cosmic_warp.core.termination.REASONS``;
- ``baselines``: every committed benchmark baseline in
  ``benchmarks/baselines/``.

An unknown marker, or a source that cannot produce its table, fails the build.
"""
import json
import re
from pathlib import Path

import mkdocs_gen_files

root = Path(__file__).parent.parent
MARKER = re.compile(r"<!-- generated: ([\w-]+) -->")


def _pyproject_section(name):
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(rf"^\[{re.escape(name)}\]\n(.*?)(?=^\[)", text, re.M | re.S)
    if not match:
        raise SystemExit(f"pyproject.toml has no [{name}] section")
    return match.group(1)


def requirements():
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    python = re.search(r'^requires-python = "([^"]+)"', text, re.M).group(1)
    deps = re.search(r"^dependencies = \[(.*?)\]", text, re.M | re.S).group(1)
    lines = [f"- Python `{python}`"]
    lines += [f"- `{d}`" for d in re.findall(r'"([^"]+)"', deps)]
    return "\n".join(lines)


def extras():
    rows, comment = [], []
    for line in _pyproject_section("project.optional-dependencies").splitlines():
        if line.startswith("# "):
            comment.append(line[2:].strip())
            continue
        match = re.match(r"^([\w-]+) = \[(.*)", line)
        if match:
            name = match.group(1)
            if not comment:
                raise SystemExit(f"pyproject.toml: extra '{name}' has no comment "
                                 "describing it (the docs' extras table needs one)")
            rows.append((name, " ".join(comment)))
        comment = []
    lines = ["| Extra | For |", "| --- | --- |"]
    lines += [f"| `{name}` | {text} |" for name, text in rows]
    return "\n".join(lines)


def termination_reasons():
    from cosmic_warp.core import termination

    lines = ["| Code | Reason | Condition |", "| ---: | --- | --- |"]
    for code, (name, condition) in sorted(termination.REASONS.items()):
        lines.append(f"| {code} | {name} | {condition} |")
    return "\n".join(lines)


INTEGRATORS = ("symplectic", "rk4", "euler")


def _baseline_table(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    results, meta = data.get("results", {}), data.get("meta", {})
    if not results:
        raise SystemExit(f"{path} has no results")

    def row(label, key, entities):
        cells = []
        for integrator in INTEGRATORS:
            rate = results.get(f"{integrator}/{key}")
            cells.append("—" if rate is None else
                         f"{rate:,.0f} ({rate * entities / 1e6:,.0f} M)")
        return f"| {label} | " + " | ".join(cells) + " |"

    lines = [f"**{meta.get('gpu', path.stem)}** — recorded {meta.get('date', '?')}, "
             f"Warp {meta.get('warp', '?')}, CUDA driver {meta.get('cuda_driver', '?')}, "
             f"cosmic_warp {meta.get('cosmic_warp', '?')}, Python {meta.get('python', '?')}.", "",
             "| Workload | " + " | ".join(i.capitalize() if i != "rk4" else "RK4"
                                         for i in INTEGRATORS) + " |",
             "| --- |" + " ---: |" * len(INTEGRATORS)]
    envs = sorted({int(m.group(1)) for k in results
                   for m in [re.fullmatch(r"(?:symplectic|rk4|euler)/envs=(\d+)", k)] if m})
    for n in envs:
        lines.append(row(f"{n:,} environments", f"envs={n}", n))
    for key in sorted({k.split("/", 1)[1] for k in results if "/entities=" in k}):
        case, entities = key.rsplit("/entities=", 1)
        entities = int(entities)
        label = {"thrusting": f"{entities:,} environments, thrusters firing",
                 "bodies=4": f"{entities // 4:,} environments × 4 bodies, "
                             "termination checks on"}.get(case, key)
        lines.append(row(label, key, entities))

    diff = sorted({int(m.group(1)) for k in results
                   for m in [re.fullmatch(r"diff_rollout/envs=(\d+)", k)] if m})
    if diff:
        horizon = meta.get("diff_horizon", "?")
        lines += ["", f"Differentiable windows ({horizon} steps, `DiffVecEnv` with a "
                      "linear torch policy), windows/s:", "",
                  "| Environments | Rollout | Rollout + backward |", "| ---: | ---: | ---: |"]
        for n in diff:
            lines.append(f"| {n:,} | {results[f'diff_rollout/envs={n}']:,.1f} | "
                         f"{results[f'diff_rollout_backward/envs={n}']:,.1f} |")
    return "\n".join(lines)


def baselines():
    paths = sorted((root / "benchmarks" / "baselines").glob("*.json"))
    if not paths:
        raise SystemExit("benchmarks/baselines/ has no baseline")
    return "\n\n".join(_baseline_table(p) for p in paths)


GENERATORS = {
    "requirements": requirements,
    "extras": extras,
    "termination-reasons": termination_reasons,
    "baselines": baselines,
}


def _fill(match):
    name = match.group(1)
    if name not in GENERATORS:
        raise SystemExit(f"unknown docs marker <!-- generated: {name} -->")
    return GENERATORS[name]()


for page in sorted((root / "docs").rglob("*.md")):
    rel = page.relative_to(root / "docs").as_posix()
    text = page.read_text(encoding="utf-8")
    if MARKER.search(text):
        with mkdocs_gen_files.open(rel, "w") as fd:
            fd.write(MARKER.sub(_fill, text))

"""Generate the examples gallery from ``examples/``.

Run by mkdocs-gen-files during the docs build: one page per script, its
module docstring as the prose and its full source below it, plus an index
listing the scripts and the YAML scenarios (described by the first line of
each file's leading comment). Nothing is written to the repo: an example and
the page documenting it are the same file, so they cannot drift apart.

Docstrings are therefore written in **Markdown**: the first line is the page
title, the rest its body. Images are linked as ``../assets/examples/...``.
"""
import ast
import os
import re
from pathlib import Path

import mkdocs_gen_files

root = Path(__file__).parent.parent
src = root / "examples"
docs = root / "docs"

#: Gallery groups in display order. Every script must be listed: an unlisted
#: one fails the build rather than silently missing from the gallery.
GROUPS = {
    "Orbits and integrators": ["01_basic_orbit", "02_integrator_comparison",
                               "05_energy_conservation", "08_j2_comparison"],
    "Batches": ["03_parallel_monte_carlo", "04_scalability", "09_multi_body_orbit",
                "10_domain_randomization"],
    "Actuators": ["06_thrusters", "07_reaction_wheels"],
    "Gradients": ["11_gradient_optimization", "12_system_identification"],
    "Reinforcement learning": ["13_rl_attitude_env", "14_rl_rsl_training",
                               "15_rl_skrl_training", "16_rl_rlgames_training",
                               "17_rl_shac_training"],
    "Shared modules and tools": ["attitude_task", "shac", "run_config"],
}
#: Modules imported by the scripts rather than run.
MODULES = {"attitude_task", "shac"}

group_of = {stem: group for group, stems in GROUPS.items() for stem in stems}
nav = mkdocs_gen_files.Nav()
nav["Overview"] = "index.md"
titles = {}

for path in sorted(src.glob("*.py")):
    source = path.read_text(encoding="utf-8")
    docstring = ast.get_docstring(ast.parse(source))
    if not docstring:
        raise SystemExit(f"{path.relative_to(root)} has no module docstring, so it "
                         "has no gallery page")
    if path.stem not in group_of:
        raise SystemExit(f"{path.relative_to(root)} is in no gallery group: add it to "
                         "GROUPS in scripts/gen_example_pages.py")
    title, _, body = docstring.partition("\n")
    title = title.strip().rstrip(".")
    titles[path.stem] = title
    page = Path("examples", f"{path.stem}.md")
    with mkdocs_gen_files.open(page, "w") as fd:
        fd.write(f"# {title}\n\n{body.strip()}\n\n")
        if path.stem in MODULES:
            fd.write(f"Imported by the scripts next to it: `from {path.stem} import ...`.\n\n")
        # The docstring holds ``` fences: the source fence must be longer
        fence = "`" * (max(map(len, re.findall(r"`+", source)), default=0) + 1)
        fence = max(fence, "```", key=len)
        fd.write(f'## Source\n\n`examples/{path.name}`\n\n{fence}{{ .python linenums="1" }}\n'
                 f"{source}{fence}\n")
    mkdocs_gen_files.set_edit_path(page, os.path.relpath(path, docs))

missing = sorted(set(group_of) - set(titles))
if missing:
    raise SystemExit(f"GROUPS lists examples that do not exist: {missing}")


def yaml_summary(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith("# "):
        raise SystemExit(f"{path.relative_to(root)} needs a leading '# <summary>' comment")
    return lines[0][2:].strip()


with mkdocs_gen_files.open("examples/index.md", "w") as fd:
    fd.write(
        "# Examples\n\n"
        "Complete scripts, one feature each, in `examples/` of the repository. Run "
        "them from the repository root by path, e.g. `python examples/01_basic_orbit.py`; "
        "`--help` lists each script's options. Each page shows the script's source.\n\n"
        "Scripts that plot save their figures as PNG files in the working directory "
        "and open a window unless given `--no-show` (example 10 never opens one).\n")
    for group, stems in GROUPS.items():
        fd.write(f"\n## {group}\n\n")
        for stem in stems:
            fd.write(f"- [{titles[stem]}]({stem}.md) — `{stem}.py`\n")
            nav[(group, titles[stem])] = f"{stem}.md"
    fd.write("\n## YAML scenarios\n\n"
             "Complete scenarios in one [configuration file](../guides/configuration.md), "
             "run with [`run_config.py`](run_config.md):\n\n"
             "```bash\npython examples/run_config.py examples/<config>.yaml --steps 5000\n```\n\n"
             "| Config | Scenario |\n| --- | --- |\n")
    for path in sorted(src.glob("*.yaml")):
        fd.write(f"| `{path.name}` | {yaml_summary(path)} |\n")

with mkdocs_gen_files.open("examples/SUMMARY.md", "w") as fd:
    fd.writelines(nav.build_literate_nav())

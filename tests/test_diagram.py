"""The step diagram in the README and on the GPU-execution page, held to the code.

A hand-drawn diagram drifts from the code silently. Here the order is read
from the source — `simulate()`'s per-step body, `Integrator.step()` and each
integrator kernel — and the stage lists `scripts/diagrams.py` draws from must
match it exactly, and each committed SVG must have been drawn by the current
version of that file. No drawing, so no matplotlib, is needed.
"""
import ast
import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CORE = REPO / "cosmic_warp" / "core"
PAGES = {"README.md": ["docs/assets/step-{}.svg", "docs/assets/learning-{}.svg"],
         "docs/theory/execution.md": ["../assets/step-{}.svg"],
         "docs/guides/differentiable-rl.md": ["../assets/learning-{}.svg"]}
KERNELS = {"euler": "euler_step_kernel_masked", "rk4": "rk4_step_kernel_masked",
           "symplectic": "symplectic_step_kernel_masked"}

spec = importlib.util.spec_from_file_location("diagrams", REPO / "scripts" / "diagrams.py")
diagrams = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagrams)

# simulate()'s per-step body: what each call launches. `self.step()` opens
# into Integrator.step's two launches.
STEP_CALLS = {"_launch_store_state": "store trajectory", "_launch_store_rw": "store trajectory",
              "reset_terminated": "auto-reset", "increment_step_kernel": "trajectory index"}
INTEGRATOR_STEP_CALLS = {"launch_step": "step kernel",
                         "reduce_environment_termination_kernel": "multi-body reduction"}
# Each integrator kernel: the call or statement that starts each phase.
KERNEL_CALLS = {"thruster_wrench": "thruster wrench", "compute_forces": "integration stages",
                "compute_nongravity_forces": "integration stages", "gravity_vector": "integration stages",
                "compute_torques": "integration stages", "store_state": "store state, wheel momenta",
                "update_termination_status": "termination check"}
# What a stage sums into the force and the torque.
FORCE_CALLS = {"gravity_force": "gravity + J2", "drag_force": "drag", "quat_rotate": "thrust"}
TORQUE_CALLS = {"gravity_gradient_torque": "gravity gradient", "magnetic_torque": "magnetic"}


def _function(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(found) == 1, f"expected one `{name}` in {path.name}, found {len(found)}"
    return found[0]


def _called(node):
    """The name a call targets: `f(...)`, `obj.f(...)`, or the kernel of `wp.launch(kernel=f)`."""
    func = node.func
    name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
    if name == "launch":
        for kw in node.keywords:
            if kw.arg == "kernel" and isinstance(kw.value, ast.Name):
                return kw.value.id
        if node.args and isinstance(node.args[0], ast.Name):
            return node.args[0].id
    return name


def _in_order(found):
    """Labels by source position, each kept at its first occurrence."""
    labels = []
    for _, label in sorted(found):
        if label not in labels:
            labels.append(label)
    return labels


def _calls(function, table):
    return [((n.lineno, n.col_offset), table[_called(n)]) for n in ast.walk(function)
            if isinstance(n, ast.Call) and _called(n) in table]


def step_stages():
    """simulate()'s per-step kernels, in launch order."""
    simulate = _function(CORE / "pipeline.py", "simulate")
    body = next(n for n in ast.walk(simulate) if isinstance(n, ast.FunctionDef) and n.name == "step_body")
    found = _calls(body, STEP_CALLS)
    step = [n for n in ast.walk(body) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "step" and isinstance(n.func.value, ast.Name) and n.func.value.id == "self"]
    assert len(step) == 1, "step_body no longer calls self.step() once"
    pos = (step[0].lineno, step[0].col_offset)
    # sim.step() is integrator.step(): its launches take self.step()'s place.
    sim_step = _function(CORE / "pipeline.py", "step")
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "step"
               and isinstance(n.func.value, ast.Attribute) and n.func.value.attr == "integrator"
               for n in ast.walk(sim_step)), "SpacecraftSimulator.step() no longer calls integrator.step()"
    inner = _calls(_function(CORE / "integrators" / "base.py", "step"), INTEGRATOR_STEP_CALLS)
    found += [(pos + sub, label) for sub, label in inner]
    return _in_order(found)


def kernel_stages(integrator):
    """One integrator kernel's phases, in source order."""
    kernel = _function(CORE / "integrators" / f"{integrator}.py", KERNELS[integrator])
    found = _calls(kernel, KERNEL_CALLS)
    for n in ast.walk(kernel):
        pos = (getattr(n, "lineno", 0), getattr(n, "col_offset", 0))
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                                             and t.value.id == "episode_steps" for t in n.targets):
            found.append((pos, "episode counter +1"))
        if isinstance(n, ast.If) and isinstance(n.test, ast.Attribute) and n.test.attr == "terminated":
            found.append((pos, "terminated?"))
    return _in_order(found)


def _names(*functions):
    return {_called(n) for f in functions for n in ast.walk(f) if isinstance(n, ast.Call)}


def test_the_code_has_every_stage():
    """Guards the readers above: a renamed call must fail here, not silently
    shorten the expected order."""
    assert step_stages() == ["step kernel", "multi-body reduction", "store trajectory", "auto-reset",
                             "trajectory index"]
    for integrator in KERNELS:
        assert len(kernel_stages(integrator)) == 6, integrator


def _flat(labels):
    return [label.replace("\n", " ") for label in labels]


def test_step_row_matches_simulate():
    assert _flat(diagrams.STEP) == step_stages(), \
        f"the diagram shows {_flat(diagrams.STEP)}, simulate() runs {step_stages()}"


@pytest.mark.parametrize("integrator", KERNELS)
def test_kernel_row_matches_each_integrator(integrator):
    assert _flat(diagrams.KERNEL) == kernel_stages(integrator), \
        f"the diagram shows {_flat(diagrams.KERNEL)}, the {integrator} kernel runs {kernel_stages(integrator)}"


@pytest.mark.parametrize("integrator", KERNELS)
def test_every_kernel_integrates_the_wheels(integrator):
    """The stages' third card: every kernel advances the wheel momenta."""
    kernel = _function(CORE / "integrators" / f"{integrator}.py", KERNELS[integrator])
    assert "integrate_reaction_wheels" in _names(kernel)


def test_stage_physics_matches_the_force_and_torque_sums():
    physics = REPO / "cosmic_warp" / "physics"
    forces = [_function(physics / "aggregation.py", name) for name in ("compute_forces", "compute_nongravity_forces")]
    assert {FORCE_CALLS[n] for n in _names(*forces) if n in FORCE_CALLS} == set(diagrams.FORCES)
    # J2 is inside the gravity force, switched by its flag.
    assert "gravity_accel_j2" in _names(_function(physics / "gravity.py", "gravity_force"))

    torques = _function(physics / "aggregation.py", "compute_torques")
    shown = {TORQUE_CALLS[n] for n in _names(torques) if n in TORQUE_CALLS}
    if any(isinstance(n, ast.Name) and n.id == "thrust_torque" for n in ast.walk(torques)):
        shown.add("thrust")
    assert shown == set(diagrams.TORQUES)


@pytest.mark.parametrize("page", PAGES)
def test_pages_show_the_drawn_diagrams(page):
    text = (REPO / page).read_text(encoding="utf-8")
    for image in PAGES[page]:
        for theme in diagrams.THEMES:
            assert image.format(theme) in text, f"{page} does not show {image.format(theme)}"


@pytest.mark.parametrize("name", diagrams.DIAGRAMS)
def test_committed_diagrams_are_current(name):
    """Read from the hash each SVG records, so no drawing, and no matplotlib, is needed."""
    for theme in diagrams.THEMES:
        svg = diagrams.path(name, theme).read_text(encoding="utf-8")
        recorded = re.search(r"scripts/diagrams\.py ([0-9a-f]{16})", svg)
        assert recorded and recorded.group(1) == diagrams.source_hash(), \
            f"{diagrams.path(name, theme).relative_to(REPO)} is stale: run python scripts/diagrams.py"

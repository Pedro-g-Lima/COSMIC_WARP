"""docs/reference/stability.md is mechanical: every promised name resolves at
its import path with the listed arguments (in order, ``*`` marking where the
keyword-only ones start), every listed member, field and key exists, and every
name ``cosmic_warp`` and ``cosmic_warp.rl`` export is listed."""
import dataclasses
import importlib
import inspect
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = (ROOT / "docs" / "reference" / "stability.md").read_text(encoding="utf-8")

NAME = re.compile(r"`([A-Za-z_]\w*\*?)(?:\(([^)`]*)\))?`")
# Modules that import an optional dependency: skip them when it is missing.
OPTIONAL = {
    "cosmic_warp.rl.diff_bridge": "torch",
    "cosmic_warp.rl.diff_tasks": "torch",
    "cosmic_warp.rl.diff_env": "torch",
    "cosmic_warp.rl.adapters.rsl_rl": "rsl_rl",
    "cosmic_warp.rl.adapters.skrl": "skrl",
    "cosmic_warp.rl.adapters.rl_games": "rl_games",
}


def promised_rows():
    """(label, module or None, third column) for every row of the table."""
    section = PAGE.split("## Promised", 1)[1].split("\n## ", 1)[0]
    rows = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        assert len(cells) == 3, f"table row without exactly three cells: {line[:80]}"
        if cells[0] == "Surface" or set(cells[0]) == {"-"}:
            continue
        module = re.fullmatch(r"`([\w.]+)`", cells[1])
        assert module or cells[1] == "—", f"bad import path cell: {cells[1]}"
        rows.append((cells[0], module.group(1) if module else None, cells[2]))
    return rows


def names(text):
    """[(name, [args] or None)] for every backticked name in ``text``."""
    return [(m.group(1), None if m.group(2) is None
             else [x.strip() for x in m.group(2).split(",") if x.strip()])
            for m in NAME.finditer(text)]


def listed(third_column):
    """The names a row promises: its third column, up to the first dash."""
    return names(third_column.split(" — ", 1)[0])


def module_rows():
    return [(label, module, rest) for label, module, rest in promised_rows() if module]


def member_rows():
    return [(label, rest) for label, module, rest in promised_rows() if not module]


def promised_at(module):
    return {n for label, path, _ in module_rows() if path == module for n, _ in names(label)}


def import_promised(module):
    if module in OPTIONAL:
        pytest.importorskip(OPTIONAL[module])
    return importlib.import_module(module)


def parameters(obj):
    return [p for p in inspect.signature(obj).parameters if p != "self"]


def signature(obj):
    """The parameters as the page writes them: in order, ``*`` before the
    first keyword-only one."""
    out = []
    for p in inspect.signature(obj).parameters.values():
        if p.name == "self":
            continue
        if p.kind is p.KEYWORD_ONLY and "*" not in out:
            out.append("*")
        out.append(p.name)
    return out


def check_args(obj, name, args, where):
    assert signature(obj) == args, (
        f"{where}.{name}: page lists ({', '.join(args)}), "
        f"code has ({', '.join(signature(obj))})")


def fields_of(obj):
    """Field names of a Warp struct or a dataclass, else None."""
    if dataclasses.is_dataclass(obj):
        return {f.name for f in dataclasses.fields(obj)}
    if hasattr(obj, "vars") and hasattr(obj, "cls"):
        return set(obj.vars)
    return None


def test_the_table_is_parsed():
    assert len(module_rows()) > 20
    assert len(member_rows()) > 15
    assert sum(len(listed(rest)) for _, rest in member_rows()) > 100


@pytest.mark.parametrize("label,module,rest", module_rows(), ids=lambda v: str(v)[:40])
def test_promised_names_resolve_at_their_import_path(label, module, rest):
    target = import_promised(module)
    found = names(label)
    assert found, f"no name in {label!r}"
    for name, args in found:
        if name.endswith("*"):
            assert any(n.startswith(name[:-1]) for n in dir(target)), f"{module}.{name}"
            continue
        assert hasattr(target, name), f"{module}.{name}"
        if args is not None:
            check_args(getattr(target, name), name, args, module)
    if len(found) == 1:
        fields = fields_of(getattr(target, found[0][0]))
        if fields is not None:   # a struct or dataclass row lists its fields
            promised = {n for n, _ in listed(rest)}
            assert promised and promised <= fields, (
                f"{module}.{found[0][0]}: {sorted(promised - fields)} are not fields")


@pytest.fixture(scope="module")
def surfaces(device):
    """What each member row of the page describes, on a tiny simulator."""
    from cosmic_warp.core.config import _TOP_KEYS
    from cosmic_warp.core.pipeline import SpacecraftSimulator
    from cosmic_warp.core.reset import ResetManager
    from cosmic_warp.rl import Task, VecEnv

    class NullTask:
        def bind(self, env):
            self.act_dim, self.obs_dim = 1, 1

    sim = SpacecraftSimulator(num_envs=2, integrator="euler", dt=1.0, device=device,
                              differentiable=True, check_bounds=False)
    sim.reset()
    out = {
        "`sim` methods": sim,
        "`sim` attributes": sim,
        "`rollout`": sim.differentiable_rollout(2),
        "`env`": VecEnv(sim, NullTask(), max_episode_steps=4),
        "`Task` protocol": Task,
        # key sets: exactly the listed keys
        "`simulate()` keys": set(sim.simulate(1, save_states=True, save_rw_states=True)),
        "`simulate_monte_carlo()` keys": set(sim.simulate_monte_carlo(1)),
        "`reset_config` keys": set(parameters(ResetManager)) - {"num_envs", "device", "num_bodies"},
        "YAML scenario": set(_TOP_KEYS),
    }
    for label, _ in member_rows():
        path = re.fullmatch(r"`sim\.(\w+)`", label)
        if path:
            out[label] = getattr(sim, path.group(1))
    try:
        import torch  # noqa: F401
    except ImportError:
        return out
    from cosmic_warp.rl.diff_bridge import DiffSimStepper
    from cosmic_warp.rl.diff_env import DiffVecEnv
    from cosmic_warp.rl.diff_tasks import DiffTask

    class NullDiffTask:
        def bind(self, sim):
            self.act_dim, self.obs_dim = 1, 1

    out.update({
        "`stepper`": DiffSimStepper(sim),
        "`DiffTask` protocol": DiffTask,
        "`diff_env`": DiffVecEnv(sim, NullDiffTask(), horizon=2, max_episode_steps=4),
    })
    return out


TORCH_ROWS = {"`stepper`", "`DiffTask` protocol", "`diff_env`"}


@pytest.mark.parametrize("label,rest", member_rows(), ids=lambda v: str(v)[:30])
def test_promised_members_exist(label, rest, surfaces):
    members = listed(rest)
    if not members:
        return   # a prose row (conventions)
    if label in TORCH_ROWS:
        pytest.importorskip("torch")
    assert label in surfaces, f"no resolver for the row {label!r}: add one to this test"
    target = surfaces[label]
    if isinstance(target, set):
        assert {n for n, _ in members} == target, f"{label}: page and code keys differ"
        return
    present = set(dir(target)) | set(getattr(target, "__annotations__", {}))
    for name, args in members:
        assert name in present, f"{label}: no {name}"
        if args is not None:
            check_args(getattr(target, name), name, args, label)


def test_every_resolver_has_a_row():
    labels = {label for label, _ in member_rows()}
    for label in ("`sim` methods", "`rollout`", "`env`", "`stepper`", "`diff_env`",
                  "`simulate()` keys", "YAML scenario", "`sim.vehicle`"):
        assert label in labels


@pytest.mark.parametrize("package", ["cosmic_warp", "cosmic_warp.rl"])
def test_every_export_is_promised(package):
    module = importlib.import_module(package)
    exported = getattr(module, "__all__", None)
    if exported is None:
        exported = [n for n, v in vars(module).items()
                    if not n.startswith("_") and not inspect.ismodule(v)]
    missing = set(exported) - promised_at(package)
    assert not missing, f"{package} exports {sorted(missing)} without a stability entry"


@pytest.mark.parametrize("label,rest", member_rows(), ids=lambda v: str(v)[:30])
def test_setters_take_optional_arguments_by_keyword(label, rest, surfaces):
    """Every promised set_* method's optional arguments are keyword-only, so a
    reordered or inserted argument can never silently shift a positional one."""
    if label in TORCH_ROWS:
        pytest.importorskip("torch")
    for name, args in listed(rest):
        if not name.startswith("set_") or args is None:
            continue
        for p in inspect.signature(getattr(surfaces[label], name)).parameters.values():
            if p.default is not p.empty:
                assert p.kind is p.KEYWORD_ONLY, f"{label}.{name}: {p.name} is positional"

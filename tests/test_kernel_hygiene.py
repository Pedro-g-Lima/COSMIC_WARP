"""No bare float literals in Warp kernel code.

Warp has no literal promotion: a Python float that reaches kernel code is a
``float32``, whatever the other operand is. What that does depends on where
it sits. Measured on Warp 1.18.0, CPU and CUDA alike, with ``x`` a float64
holding exactly 0.1:

* **Compile error** — arithmetic and in-place arithmetic (``x * 0.1``,
  ``x += 0.1``), builtins with one generic type (``wp.max``, ``wp.min``,
  ``wp.clamp``, ``wp.pow``, ``wp.where``, the ``a if c else b`` expression),
  storing into a float64 array or struct field, passing to or returning from a
  float64 ``@wp.func``, and a bare-literal default for a float64 parameter.
* **Silently float32** — every comparison: ``x == 0.1`` is False,
  ``x < 0.1``, ``0.1 > x``, ``x != 0.1`` and ``-x > -0.1`` are True, and
  ``y == 16777217.0`` misses the exact float64 value 16777217. A literal kept
  in a variable and cast later is rounded too (``y = 0.1; z = y * 1.0;
  wp.float64(z)`` gives 0.10000000149011612), as is a constant expression
  inside a constructor (``wp.float64(1.0 / 3.0)`` gives 0.3333333432674408).
  A module-level Python float (``G = 0.1``, ``G = wp.constant(0.1)``,
  ``wp.PI``) behaves exactly like a literal: ``x < G`` is True.
* **Exact** — a literal, or a negated one, passed *directly* to
  ``wp.float64(...)`` (0.1, -0.1, 6378137.123456789, 1e-300 all round-trip),
  and likewise ``wp.float64(G)`` and ``wp.float64(wp.PI)``. Literal arguments
  of ``wp.vec3d`` / ``wp.quatd`` / ``wp.mat33d`` and ``y = 0.1;
  wp.float64(y)`` also came out exact, but only through Warp's constant
  folding, which the next harmless-looking edit (``y * 1.0``, a branch that
  reassigns ``y``) silently defeats.

Integer literals are not checked. Mixing one with float64 in arithmetic or
in ``wp.max``/``wp.pow`` is a compile error; a comparison (``x > 0``,
``x == 16777217``) or a vector constructor (``wp.vec3d(1, 0, 0)``) converts
int32 to float64, which is exact for every int32.

So the rule: every float literal in a ``@wp.kernel`` / ``@wp.func`` (body
and default arguments), and every name that resolves to a host Python float,
is a direct argument of a scalar dtype constructor — ``wp.float64(...)`` for
the physics, ``wp.float32(...)`` where single precision is intended (the RL
observation/action kernels). There is no
exemption, not even for 0.0: an exact literal in a comparison is harmless
today but reads exactly like the inexact one, and the rule is only checkable
if it has no judgment calls.
"""
import ast
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import warp as wp

import cosmic_warp

PACKAGE = Path(cosmic_warp.__file__).resolve().parent
REPO = PACKAGE.parent
EXAMPLES = REPO / "examples"

#: Decorators (bare or called, ``wp.`` or ``warp.``) that make Warp code.
KERNEL_DECORATORS = {"kernel", "func", "func_native", "func_grad", "func_replay"}
#: Scalar constructors that type a literal exactly (see the docstring).
TYPED_CONSTRUCTORS = {"float64", "float32", "float16"}
WARP_ALIASES = {"wp", "warp"}


def _is_warp_attr(node, names):
    """``node`` is ``wp.<name>`` / ``warp.<name>`` for a name in ``names``."""
    return (isinstance(node, ast.Attribute) and node.attr in names
            and isinstance(node.value, ast.Name) and node.value.id in WARP_ALIASES)


def _warp_defs(tree):
    """Every function in ``tree`` that Warp compiles, at any nesting depth.

    Covers ``@wp.kernel``, ``@wp.kernel(...)`` and the other decorators in
    ``KERNEL_DECORATORS``, and the call form ``wp.kernel(fn)`` used by
    factories that build a kernel from a nested function.
    """
    called = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and _is_warp_attr(node.func, KERNEL_DECORATORS)
                and node.args and isinstance(node.args[0], ast.Name)):
            called.add(node.args[0].id)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        decorators = [d.func if isinstance(d, ast.Call) else d for d in node.decorator_list]
        if node.name in called or any(_is_warp_attr(d, KERNEL_DECORATORS) for d in decorators):
            yield node


def _safe_to_exec(stmt):
    """Top-level statements of an example that only bind names."""
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        return True
    if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is not None:
        return all(not isinstance(n, ast.Call) or _is_warp_attr(n.func, {"constant"} | TYPED_CONSTRUCTORS)
                   for n in ast.walk(stmt.value))
    return False


OPTIONAL_DEPENDENCIES = {"torch", "rsl_rl", "skrl", "rl_games", "tensordict", "matplotlib"}


def _host_namespace(path, tree):
    """The module-level names a kernel in ``path`` can see.

    Package modules are imported. Example scripts are not (they may build
    simulators or parse arguments at import): only their imports and plain
    constant assignments are executed, which is all a kernel can reference.
    """
    if path.is_relative_to(PACKAGE):
        parts = path.relative_to(REPO).with_suffix("").parts
        try:
            return vars(importlib.import_module(".".join(p for p in parts if p != "__init__")))
        except ModuleNotFoundError as e:
            # A module needing an optional dependency (torch) is still checked:
            # its kernels reference only Warp and module constants
            if (e.name or "").split(".")[0] not in OPTIONAL_DEPENDENCIES:
                raise
    ns = {"__name__": f"_kernel_hygiene_{path.stem}", "__file__": str(path)}
    sys.path.insert(0, str(path.parent))
    try:
        for stmt in tree.body:
            if _safe_to_exec(stmt):
                try:
                    exec(compile(ast.Module([stmt], []), str(path), "exec"), ns)
                except Exception:  # an optional dependency missing: name stays unresolved
                    pass
    finally:
        sys.path.remove(str(path.parent))
    return ns


def _local_names(fn):
    """Parameters and every name the function (or a nested scope) assigns."""
    args = fn.args
    names = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
    names |= {a.arg for a in (args.vararg, args.kwarg) if a is not None}
    names |= {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)
              and isinstance(n.ctx, (ast.Store, ast.Del))}
    return names


def _host_float(node, namespace, local):
    """True if ``node`` is a name/attribute that resolves to a host float."""
    if not (isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Load)):
        return False
    root = node
    while isinstance(root, ast.Attribute):
        root = root.value
    if not isinstance(root, ast.Name) or root.id in local or root.id not in namespace:
        return False
    try:
        value = eval(compile(ast.Expression(node), "<kernel-name>", "eval"), dict(namespace))
    except Exception:
        return False
    return isinstance(value, (float, np.floating))


def bare_float_literals(source, filename="<string>", namespace=None, path=None):
    """``(lineno, function, text)`` for every untyped float in Warp code.

    ``namespace`` maps module-level names to their host values; names that
    resolve to a Python float count as literals. Without it only literal
    constants are checked.
    """
    tree = ast.parse(source, filename=filename)
    if namespace is None:
        namespace = _host_namespace(path, tree) if path is not None else {}
    found = []
    for fn in _warp_defs(tree):
        local = _local_names(fn)
        # The code Warp compiles: the body and the default values. Decorators
        # and annotations are host-side types and options.
        scope = fn.body + fn.args.defaults + [d for d in fn.args.kw_defaults if d is not None]
        typed = set()
        for root in scope:
            for node in ast.walk(root):
                if not (isinstance(node, ast.Call) and _is_warp_attr(node.func, TYPED_CONSTRUCTORS)):
                    continue
                for arg in node.args:
                    if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, (ast.USub, ast.UAdd)):
                        arg = arg.operand
                    typed.add(id(arg))
        for root in scope:
            for node in ast.walk(root):
                literal = (isinstance(node, ast.Constant) and isinstance(node.value, float))
                if (literal or _host_float(node, namespace, local)) and id(node) not in typed:
                    found.append((node.lineno, fn.name, ast.unparse(node)))
    return sorted(set(found))


def _kernel_files():
    files = sorted(PACKAGE.rglob("*.py"))
    if EXAMPLES.is_dir():
        files += sorted(EXAMPLES.glob("*.py"))
    out = []
    for path in files:
        source = path.read_text(encoding="utf-8")
        if any(True for _ in _warp_defs(ast.parse(source, filename=str(path)))):
            out.append(path)
    return out


KERNEL_FILES = _kernel_files()


def test_the_scan_finds_kernels():
    """Guard against the AST walk silently matching nothing."""
    defs = [fn for path in KERNEL_FILES
            for fn in _warp_defs(ast.parse(path.read_text(encoding="utf-8")))]
    assert len(KERNEL_FILES) >= 20 and len(defs) >= 70, (
        f"found only {len(defs)} Warp functions in {len(KERNEL_FILES)} files — the walk is broken")
    assert PACKAGE / "core" / "pipeline.py" in KERNEL_FILES


@pytest.mark.parametrize(
    "path", KERNEL_FILES, ids=[str(p.relative_to(REPO)) for p in KERNEL_FILES])
def test_no_bare_float_literals_in_kernels(path):
    found = bare_float_literals(path.read_text(encoding="utf-8"), str(path), path=path)
    where = "\n".join(f"  {path.relative_to(REPO)}:{line} in {fn}(): {text}"
                      for line, fn, text in found)
    assert not found, (
        f"bare float literals in Warp code:\n{where}\n"
        "Warp has no literal promotion: these are float32 (silently so in a "
        "comparison). Pass each one directly to wp.float64(...), or to "
        "wp.float32(...) where single precision is intended.")


NEGATIVE_CONTROL = '''\
import warp as wp

G = 0.1
G64 = wp.constant(wp.float64(0.1))

@wp.func
def f(x: wp.float64, b: wp.float64 = 0.25):                      # 7: default
    return x

@wp.kernel(enable_backward=False)
def k(xs: wp.array(dtype=wp.float64)):
    x = xs[0]
    if x == 0.1:                                                  # 13
        x = wp.float64(-0.5) * wp.float64(G) * G64 * wp.float64(wp.PI)
    v = wp.vec3d(0.3, wp.float64(0.0), wp.float64(0.0))           # 15
    y = wp.float64(1.0 / 3.0)                                     # 16
    if x < G:                                                     # 17
        x = x * wp.float64(2)
    if x > 0:
        x = wp.where(x > wp.float64(1.0), x, wp.float64(-1.0))
    u = wp.randf(wp.uint32(0)) * wp.PI                            # 21

def make():
    def _body(xs: wp.array(dtype=wp.float32)):
        xs[0] = xs[0] * 2.5                                       # 25
    return wp.kernel(_body)

def host_only(x):
    return x * 0.5
'''


def test_checker_flags_a_synthetic_snippet():
    """Negative control: the checker must see each kind of violation."""
    namespace = {"wp": wp, "G": 0.1, "G64": wp.float64(0.1)}
    found = bare_float_literals(NEGATIVE_CONTROL, namespace=namespace)
    assert [(line, text) for line, _, text in found] == [
        (7, "0.25"),
        (13, "0.1"),
        (15, "0.3"),
        (16, "1.0"),
        (16, "3.0"),
        (17, "G"),
        (21, "wp.PI"),
        (25, "2.5"),
    ], found


@wp.kernel
def _literal_probe(xs: wp.array(dtype=wp.float64), out: wp.array(dtype=wp.int32)):
    x = xs[0]
    if x == 0.1:
        out[0] = 1
    if x == wp.float64(0.1):
        out[1] = 1


def test_bare_literal_is_float32_in_comparisons(device):
    """The premise of this file, on the current Warp.

    If this starts failing, Warp has changed how it types literals; the rule
    above is still correct, but the docstring's measurements need redoing.
    """
    xs = wp.array([0.1], dtype=wp.float64, device=device)
    out = wp.zeros(2, dtype=wp.int32, device=device)
    wp.launch(_literal_probe, 1, inputs=[xs, out], device=device)
    bare, typed = out.numpy()
    assert typed == 1
    assert bare == 0, "x == 0.1 matched an exact float64 0.1: bare literals are no longer float32"

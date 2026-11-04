"""
The docs, checked against the code.

- Every Python snippet in the README, Getting started, the guides and the
  theory pages runs, page by page in order, on the test device. A page is one
  workflow: later snippets may use names defined by earlier ones. A snippet
  preceded by ``<!-- docs-test: skip -->`` is not run (it needs a file the
  reader supplies, or trains for minutes); a page whose optional dependency is
  missing is skipped. When the next code block after a snippet is a ``text``
  block, it is the snippet's output, and must be what the snippet prints.
- The validation pages cite only test files that exist, and every section
  holding a table of claims cites one.
- The YAML reference lists exactly the keys the loader accepts, with the
  simulator's defaults.
"""
import contextlib
import importlib.util
import inspect
import io
import re
import sys
from pathlib import Path

import pytest
import warp as wp

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"

SNIPPET_PAGES = sorted(
    [REPO / "README.md", DOCS / "getting-started.md", DOCS / "troubleshooting.md"]
    + list((DOCS / "guides").glob("*.md")) + list((DOCS / "theory").glob("*.md")))
FENCE = re.compile(r"^(<!-- docs-test: skip -->\n)?( *)```(\w*)\n(.*?)^\2```", re.S | re.M)
OPTIONAL = {"torch", "rsl_rl", "skrl", "rl_games", "matplotlib", "tensordict"}


def snippets(page):
    """(line, skip, code, expected output or None) per Python snippet."""
    text = page.read_text(encoding="utf-8")
    blocks = list(FENCE.finditer(text))
    for match, after in zip(blocks, blocks[1:] + [None]):
        skip, indent, language, code = match.groups()
        if language != "python":
            continue
        code = "\n".join(line[len(indent):] for line in code.splitlines())
        line = text[:match.start()].count("\n") + 1
        output = after.group(4) if after is not None and after.group(3) == "text" else None
        yield line, bool(skip), code, output


@pytest.fixture
def on_test_device(device, monkeypatch):
    """Snippets build simulators with the default device (or name "cuda"):
    run them on the test device instead."""
    import cosmic_warp
    from cosmic_warp.core import config, pipeline

    class Simulator(pipeline.SpacecraftSimulator):
        def __init__(self, *args, **kwargs):
            if kwargs.get("device", "cuda") == "cuda":
                kwargs["device"] = device
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(pipeline, "SpacecraftSimulator", Simulator)
    monkeypatch.setattr(config, "SpacecraftSimulator", Simulator)
    monkeypatch.setattr(cosmic_warp, "SpacecraftSimulator", Simulator)
    # Warp's module-load lines would mix into the output the docs show
    monkeypatch.setattr(wp.config, "log_level", wp.LOG_WARNING)
    monkeypatch.chdir(REPO)
    monkeypatch.syspath_prepend(str(REPO / "examples"))
    return device


@pytest.mark.parametrize("page", SNIPPET_PAGES, ids=lambda p: str(p.relative_to(REPO)))
def test_snippets_run(page, on_test_device, tmp_path, monkeypatch):
    blocks = list(snippets(page))
    if not blocks:
        pytest.skip("no Python snippets")
    namespace = {"__name__": "docs_snippet"}
    for i, (line, skip, code, expected) in enumerate(blocks):
        if skip:
            continue
        printed = io.StringIO()
        try:
            if "@wp.kernel" in code or "@wp.func" in code:
                # Warp reads kernel source from a real file
                path = tmp_path / f"snippet_{i}.py"
                path.write_text(code)
                spec = importlib.util.spec_from_file_location(f"docs_snippet_{i}", path)
                module = importlib.util.module_from_spec(spec)
                module.__dict__.update({k: v for k, v in namespace.items()
                                        if not k.startswith("__")})
                monkeypatch.setitem(sys.modules, spec.name, module)
                spec.loader.exec_module(module)
                namespace.update({k: v for k, v in vars(module).items()
                                  if not k.startswith("__")})
            else:
                with contextlib.redirect_stdout(printed):
                    exec(compile(code, f"{page.name}:{line}", "exec"), namespace)
        except ModuleNotFoundError as e:
            if e.name and e.name.split(".")[0] in OPTIONAL:
                pytest.skip(f"optional dependency {e.name} not installed")
            raise
        except Exception as e:
            raise AssertionError(f"{page.relative_to(REPO)}:{line}: snippet failed: "
                                 f"{type(e).__name__}: {e}") from e
        if expected is not None:
            assert printed.getvalue() == expected, (
                f"{page.relative_to(REPO)}:{line}: the snippet prints\n{printed.getvalue()}"
                f"but the page shows\n{expected}")


# ---------------------------------------------------------------- validation

VALIDATION = DOCS / "validation"
CITATION = re.compile(r"`(tests/[\w/]+\.py)`")


def validation_sections():
    sections = []
    for page in sorted(VALIDATION.glob("*.md")):
        parts = re.split(r"^## (.+)$", page.read_text(encoding="utf-8"), flags=re.M)
        sections.append((f"{page.name}: (top)", parts[0]))
        sections += [(f"{page.name}: {h}", b) for h, b in zip(parts[1::2], parts[2::2])]
    return sections


def test_cited_test_files_exist():
    text = "".join(p.read_text(encoding="utf-8") for p in VALIDATION.glob("*.md"))
    cited = sorted(set(CITATION.findall(text)))
    assert cited, "the validation pages cite no test files"
    missing = [c for c in cited if not (REPO / c).is_file()]
    assert not missing, f"docs/validation/ cites missing test files: {missing}"


@pytest.mark.parametrize("heading,body", [s for s in validation_sections()
                                          if not s[0].startswith("index.md")],
                         ids=lambda x: x if isinstance(x, str) and len(x) < 80 else "")
def test_claim_sections_cite_a_test(heading, body):
    has_table = any(line.startswith("|") for line in body.splitlines())
    if has_table:
        assert CITATION.search(body), (
            f"validation section '{heading}' has a table of claims but cites no test file")


# ---------------------------------------------------------------- YAML reference

def yaml_tables():
    """{section heading: {key: default cell}} from the configuration guide."""
    text = (DOCS / "guides" / "configuration.md").read_text(encoding="utf-8")
    tables = {}
    for heading, body in zip(*[iter(re.split(r"^#{2,3} `([\w.]+)`.*$", text, flags=re.M)[1:])] * 2):
        rows = {}
        for line in body.splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if line.startswith("| `") and len(cells) >= 2:
                rows[cells[0].strip("`")] = cells[1]
        tables[heading] = rows
    return tables


@pytest.mark.parametrize("section,schema", [
    ("simulation", "_SIMULATION_KEYS"), ("physics", "_PHYSICS_KEYS"),
    ("termination", "_TERMINATION_KEYS"), ("reset", "_RESET_KEYS"),
    ("reset.orbit", "_RESET_ORBIT_KEYS"),
])
def test_yaml_reference_lists_the_schema(section, schema):
    from cosmic_warp.core import config

    documented = set(yaml_tables().get(section, {}))
    accepted = set(getattr(config, schema))
    assert documented == accepted, (
        f"configuration guide, `{section}`: undocumented keys "
        f"{sorted(accepted - documented)}, unknown keys {sorted(documented - accepted)}")


@pytest.mark.parametrize("section", ["simulation", "physics", "termination"])
def test_yaml_reference_defaults(section):
    """Defaults the guide states for keys that map to simulator arguments."""
    from cosmic_warp.core.pipeline import SpacecraftSimulator

    params = inspect.signature(SpacecraftSimulator.__init__).parameters
    for key, cell in yaml_tables()[section].items():
        if key not in params or not cell.startswith("`"):
            continue
        stated = cell.strip("`").replace(".inf", "inf")  # YAML spelling
        default = params[key].default
        if isinstance(default, bool):
            assert stated == str(default).lower(), f"{section}.{key}: docs {stated}, code {default}"
        elif isinstance(default, (int, float)):
            assert float(stated) == float(default), f"{section}.{key}: docs {stated}, code {default}"
        else:
            assert stated == str(default), f"{section}.{key}: docs {stated}, code {default}"


def test_citation_version_matches_the_package():
    import cosmic_warp

    cff = (REPO / "CITATION.cff").read_text(encoding="utf-8")
    assert re.search(rf'^version: "?{re.escape(cosmic_warp.__version__)}"?$', cff, re.M)


def _slug(heading):
    """Python-Markdown's default heading id, which the site uses."""
    import unicodedata

    text = unicodedata.normalize("NFKD", heading).encode("ascii", "ignore").decode()
    return re.sub(r"[-\s]+", "-", re.sub(r"[^\w\s-]", "", text).strip().lower())


def test_every_claim_section_is_in_the_summary():
    """The index's summary is where readers count the evidence: a section of
    claims missing from it would go uncounted."""
    summary = (VALIDATION / "index.md").read_text(encoding="utf-8")
    links = set(re.findall(r"\]\((\w+\.md(?:#[\w-]+)?)\)", summary))
    missing = []
    for heading, body in validation_sections():
        if heading.startswith("index.md") or not any(
                line.startswith("|") for line in body.splitlines()):
            continue
        page, title = heading.split(": ", 1)
        target = page if title == "(top)" else f"{page}#{_slug(title)}"
        if target not in links:
            missing.append(heading)
    assert not missing, f"validation sections missing from the index summary: {missing}"


# ---------------------------------------------------------------- README

REPO_URL = re.compile(r"https://(?:github\.com/Pedro-g-Lima/COSMIC_WARP/(?:blob|tree)/main/"
                      r"|raw\.githubusercontent\.com/Pedro-g-Lima/COSMIC_WARP/main/)([^#]*)")


def test_readme_links_resolve():
    """The README is also the PyPI page, where a relative link 404s: every link
    into the repository is absolute, and names a file that exists.
    ``mkdocs build --strict`` checks the site's links but never reads the
    README, so a renamed file or a missing diagram would go unnoticed there."""
    text = (REPO / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\]\(([^)\s]+)\)", text) + re.findall(r'(?:src|srcset|href)="([^"]+)"', text)
    relative = sorted(t for t in targets if not re.match(r"[a-z]+:|#", t))
    assert not relative, f"README has relative links, broken on PyPI: {relative}"
    local = {m.group(1) for t in targets if (m := REPO_URL.fullmatch(t))}
    assert local, "README has no links into the repository; update this test with it"
    missing = sorted(t for t in local if not (REPO / t).exists())
    assert not missing, f"README links to missing files: {missing}"


def test_readme_badges_match_pyproject():
    """The Python and license badges are static images, so nothing updates them
    when the classifiers, ``requires-python`` or the license change."""
    import tomllib
    from urllib.parse import unquote

    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    pythons = [c.rsplit(" :: ", 1)[1] for c in project["classifiers"]
               if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", c)]
    lowest = re.fullmatch(r">=\s*(3\.\d+)", project["requires-python"].strip())
    assert lowest and pythons[0] == lowest.group(1), (
        f"requires-python {project['requires-python']} differs from the lowest "
        f"version classifier {pythons[0]}")

    text = (REPO / "README.md").read_text(encoding="utf-8")
    badges = {}
    for path in re.findall(r"https://img\.shields\.io/badge/([^?\"')\s]+)", text):
        # label-message-colour, with "--" an escaped dash
        label, message, _ = (unquote(part).replace("\0", "-") for part in path.replace("--", "\0").split("-"))
        badges[label] = message
    assert badges.get("python") == " | ".join(pythons), (
        f"README python badge {badges.get('python')!r} differs from the classifiers {pythons}")
    assert badges.get("license") == project["license"], (
        f"README license badge {badges.get('license')!r} differs from pyproject.toml {project['license']!r}")
    alt = re.search(r'alt="Python (3\.\d+)-(3\.\d+)"', text)
    assert alt and [alt.group(1), alt.group(2)] == [pythons[0], pythons[-1]], \
        "the python badge's alt text does not match the classifiers"

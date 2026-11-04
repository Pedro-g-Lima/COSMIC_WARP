"""Every docs figure generator runs and draws both themes.

The docs build runs them too, but only in the docs job; this keeps a broken
generator from reaching it. CPU only, like the build, and uncached: a cached
result would skip the computation this is here to exercise.
"""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("docs_figures", REPO / "scripts" / "figures.py")
figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figures)


@pytest.mark.parametrize("name", list(figures.FIGURES))
def test_figure_draws_in_both_themes(name):
    compute, draw = figures.FIGURES[name]
    value = compute()
    for theme in figures.THEMES:
        svg = draw(value, theme)
        assert svg.lstrip().startswith("<?xml") and "</svg>" in svg, f"{name}/{theme} is not an SVG"
    # The two themes share data but not colours: a copy-paste slip that drew one
    # theme twice would pass every other check.
    assert draw(value, "light") != draw(value, "dark")


@pytest.mark.parametrize("name", list(figures.FIGURES))
def test_pages_show_both_themes(name):
    """Each figure is embedded, light and dark, on a validation or benchmarks page."""
    pages = [*(REPO / "docs" / "validation").glob("*.md"), REPO / "docs" / "benchmarks.md"]
    text = "".join(page.read_text(encoding="utf-8") for page in pages)
    for theme in figures.THEMES:
        assert f"figures/{name}-{theme}.svg#only-{theme}" in text, f"no page shows {name}-{theme}"

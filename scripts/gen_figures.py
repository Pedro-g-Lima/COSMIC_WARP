"""Write the validation figures into the docs build (light and dark SVG each).

Run by mkdocs-gen-files. The pages embed ``figures/<name>-light.svg#only-light``
and ``figures/<name>-dark.svg#only-dark``; what each figure shows, and where its
data comes from, is in ``scripts/figures.py``.
"""
import sys
from pathlib import Path

import mkdocs_gen_files

sys.path.insert(0, str(Path(__file__).parent))
import figures  # noqa: E402

key = figures.sources_key()
for name in figures.FIGURES:
    for theme, svg in figures.render(name, key).items():
        with mkdocs_gen_files.open(f"figures/{name}-{theme}.svg", "w") as f:
            f.write(svg)

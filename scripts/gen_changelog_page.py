"""Publish the root ``CHANGELOG.md`` as the site's changelog page.

Run by mkdocs-gen-files during the docs build. The changelog lives at the repo
root, where GitHub readers expect it; links into ``docs/`` are made relative
to the docs directory.
"""
import os
from pathlib import Path

import mkdocs_gen_files

root = Path(__file__).parent.parent
text = (root / "CHANGELOG.md").read_text(encoding="utf-8")

with mkdocs_gen_files.open("reference/changelog.md", "w") as fd:
    fd.write(text.replace("](docs/", "](../"))

mkdocs_gen_files.set_edit_path("reference/changelog.md",
                               os.path.relpath(root / "CHANGELOG.md", root / "docs"))

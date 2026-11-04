"""Generate one API-reference page per cosmic_warp module at build time.

Executed by the mkdocs-gen-files plugin during `mkdocs build`; the pages are
virtual and never written into docs/. Only the cosmic_warp package is
documented.
"""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

for path in sorted(Path("cosmic_warp").rglob("*.py")):
    module_path = path.with_suffix("")
    doc_path = path.with_suffix(".md")
    full_doc_path = Path("reference", "api", doc_path)

    parts = tuple(module_path.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1].startswith("_"):
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"::: {'.'.join(parts)}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("reference/api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())

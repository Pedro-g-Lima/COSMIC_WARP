## What and why

<!-- What does this change, and why? Link the issue it addresses. -->

## Checklist

- [ ] Tests added or updated; a fix comes with a test that fails before it
- [ ] `pytest --device cpu` passes (and `pytest` on CUDA if kernels changed)
- [ ] `mkdocs build --strict` passes if docs changed
- [ ] `benchmarks/bench.py` shows no regression if step kernels changed
- [ ] Docs and `CHANGELOG.md` updated for behaviour changes

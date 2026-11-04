# Contributing to COSMIC WARP

Thanks for your interest! Bug reports, fixes, documentation improvements and
new physics or actuator models are all welcome. For larger changes, open an
issue first so the approach can be discussed before you invest time.

## Development install

```bash
git clone https://github.com/Pedro-g-Lima/COSMIC_WARP.git
cd COSMIC_WARP
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade "pip>=25.1"   # for --group
pip install -e . --group dev
# the RL stack: pip install -e ".[torch,rsl-rl,skrl,rl-games]" --group dev
```

## Tests

```bash
pytest                   # full suite on CUDA
pytest --device cpu      # without a GPU (what CI runs)
pytest -m "not slow"     # skip the end-to-end training tests
```

CI runs the CPU suite on Python 3.12 and 3.14; both ends of the dependency
ranges (Python 3.12 with the lowest supported Warp, NumPy and PyYAML, Python
3.14 with the newest); the built wheel in a fresh environment; and the RL
tests with CPU torch. CUDA-only tests (graph capture, training, the GPU
golden trajectories) must pass on a GPU before a release.

If you change a step kernel, also run the performance gate on your GPU:

```bash
python benchmarks/bench.py --save   # once, on the base commit
python benchmarks/bench.py          # after your change; fails on >15 % regression
```

## Releasing

Set `__version__` in `cosmic_warp/__init__.py` (the package version is read
from it) and in `CITATION.cff`, date the `CHANGELOG.md` entry, then push the
tag `v<version>`: the release workflow builds, checks and publishes to PyPI,
and refuses a tag that does not match `__version__`.

## Docs

```bash
pip install -e . --group docs
mkdocs build --strict    # must pass
mkdocs serve             # preview at http://127.0.0.1:8000
```

## What a pull request needs

- **Correctness first.** Physical accuracy, then differentiability, then
  GPU performance. No silent approximations: state any model limit in the
  docs.
- **A test for every fix**, one that fails before the fix. Physics changes
  are tested against an analytic or known result, and gradients against
  finite differences. Don't loosen tolerances to make a test pass; seed
  randomness; every skip states its reason.
- **Validate at configuration time**, with a `ValueError` that names the
  field. Nothing in the step loop may synchronize with the host.
- **Float64 and adjoint-safe kernels**: see the
  [kernel conventions](https://pedro-g-lima.github.io/COSMIC_WARP/guides/extending/#kernel-conventions).
  Every float literal in a kernel goes through `wp.float64(...)`
  (`tests/test_kernel_hygiene.py` enforces it).
- **Golden trajectories.** `tests/test_golden.py` pins a seeded run bit for
  bit. If your change is meant to alter results, regenerate the fixtures
  (`pytest tests/test_golden.py --device cpu --update-goldens`, and on a GPU
  with `--device cuda`) and say why in the pull request.
- **Scope.** The package is the simulator: controllers, tasks and training
  algorithms belong in `examples/` or in your own project.
- **Docs and changelog.** Update the docs for any behaviour change and add
  an entry to `CHANGELOG.md`.
- Match the surrounding code style; keep changes minimal and focused.

## License

Unless you explicitly state otherwise, any contribution you submit for
inclusion is licensed under the Apache License 2.0, as defined in section 5
of the [license](LICENSE), without additional terms or conditions.

"""The benchmark regression gate (benchmarks/bench.py compare())."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "bench", Path(__file__).resolve().parents[1] / "benchmarks" / "bench.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

BASELINE = {"meta": {"date": "d", "git": "g", "gpu": "gpu"},
            "results": {"rk4/envs=16": 100.0, "euler/envs=16": 200.0}}


@pytest.mark.parametrize("results, code", [
    ({"rk4/envs=16": 95.0, "euler/envs=16": 210.0}, 0),   # within threshold
    ({"rk4/envs=16": 80.0, "euler/envs=16": 200.0}, 1),   # regression
    ({"rk4/envs=16": 100.0}, 1),                          # missing metric
    ({}, 1),                                              # everything missing
])
def test_compare_gate(results, code):
    """A metric missing from the current run used to be skipped, so the gate
    passed even when nothing was measured."""
    assert bench.compare({"results": results}, BASELINE, threshold_pct=15.0) == code

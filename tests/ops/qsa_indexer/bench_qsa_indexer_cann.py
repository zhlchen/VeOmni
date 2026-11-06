# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Op-level benchmark: eager vs ``npu_ascendc`` QSA indexer entry (report-only).

Not collected by default directory scans (``bench_`` prefix) — run it
explicitly when the cards are idle:

    pytest -s -q tests/ops/qsa_indexer/bench_qsa_indexer_cann.py

Methodology (Triton ``do_bench`` / vLLM benchmark style): the same seeded
case list, stub, and inputs as the correctness suite; one warmup pass, then
per-iteration device sync and wall-clock timing; medians are reported with
the eager vs ``npu_ascendc`` ratio. Timing asserts are deliberately absent —
shared cards make wall-clock noisy — but each case first re-checks the
selection parity once so a broken path cannot be benchmarked silently.

The timed region is the full entry under ``torch.no_grad()``: it includes
the per-call TND metadata rebuild, the host-side segment loop, and the
kernel — exactly what a production forward pays — so a slow case here
localizes the overhead the way a RUN-level number cannot.
``QSA_INDEXER_BENCH_WARMUP`` / ``QSA_INDEXER_BENCH_ITERS`` tune the loop
counts (defaults 10 / 100).
"""

from __future__ import annotations

import os
import statistics
import time

import pytest
import torch

from tests.ops.qsa_indexer.test_qsa_indexer_cann import (
    _CASES,
    _CASE_IDS,
    _REQUIRE_NPU,
    _assert_same_selection,
    _build_case,
    _forward,
)
from veomni.utils.device import get_device_type

_WARMUP = int(os.environ.get("QSA_INDEXER_BENCH_WARMUP", "10"))
_ITERS = int(os.environ.get("QSA_INDEXER_BENCH_ITERS", "100"))


def _time_ms(fn, warmup: int, iters: int) -> list[float]:
    """Wall-clock ms per iteration, device-synchronized, after a warmup pass."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


@_REQUIRE_NPU
@pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
def test_bench_eager_vs_lightning(case):
    """Report the eager vs npu_ascendc median entry latency for one case."""
    device = get_device_type()
    case_id, indexer, hidden, positions, cu, packed = _build_case(case, device)

    with torch.no_grad():
        eager_out = _forward(indexer, hidden, positions, cu, packed)
        indexer.qsa_indexer_implementation = "npu_ascendc"
        lightning_out = _forward(indexer, hidden, positions, cu, packed)
    _assert_same_selection(lightning_out, eager_out, f"bench-gate[{case_id}]")

    def run():
        with torch.no_grad():
            _forward(indexer, hidden, positions, cu, packed)

    indexer.qsa_indexer_implementation = "eager"
    eager_ms = _time_ms(run, _WARMUP, _ITERS)
    indexer.qsa_indexer_implementation = "npu_ascendc"
    lightning_ms = _time_ms(run, _WARMUP, _ITERS)

    eager_median = statistics.median(eager_ms)
    lightning_median = statistics.median(lightning_ms)
    print(
        f"[{case_id}] eager {eager_median:8.3f} ms | npu_ascendc {lightning_median:8.3f} ms "
        f"| speedup {eager_median / lightning_median:6.2f}x ({_ITERS} iters, median)"
    )

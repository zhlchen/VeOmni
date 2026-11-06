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

"""QSA indexer selection parity: ``eager`` vs ``npu_ascendc`` through one entry.

Single-op A/B verification: for every case, build a seed-fixed bf16 batch,
call the same generated modeling forward twice — once with
``qsa_indexer_implementation = "eager"``, once with
``"npu_ascendc"`` (the dispatch routes both through the same entry) — and
compare the selections. Each case runs multiple rounds with different seeds
so the comparison is not a single-input coincidence; every round is
reproducible from its seed.

Inputs are bf16 because production feeds bf16 indexer features (fp32 inputs
would put fp32-vs-bf16 topk boundary flips into the comparison). Rows are
compared as sorted id multisets per query — TopK does not specify the order
of equal masked entries — and the failure message names the first offending
rows. The eager modeling path is the reference by definition here; it is
independently covered by ``tests/models/test_qwen4_exp_qsa_topk.py``
against the GPU modeling variant and the causal invariants.

The comparison requires an Ascend NPU with ``cann_ops_transformer``
(the eager leg runs on any device, but the pair only exists on NPU);
entry-level rejection contracts run on any host. LightningIndexerV2
requires ``head_dim == 128``; every case stays inside that envelope (the
production indexer head is 128). ``compress_ratio == 1`` is covered with
the caller dropping the all-zero ``cmp_residual_k`` the kernel refuses in
that mode.
"""

from __future__ import annotations

import importlib.util
import time
from types import SimpleNamespace

import pytest
import torch

from veomni.models.transformers.qwen4_exp.generated.patched_modeling_qwen4_exp_npu import (
    Qwen4ExpTextQSAIndexer,
)
from veomni.utils.device import IS_NPU_AVAILABLE, get_device_type

_REQUIRE_NPU = pytest.mark.skipif(
    not IS_NPU_AVAILABLE or importlib.util.find_spec("cann_ops_transformer") is None,
    reason="CANN Lightning Indexer requires Ascend NPU and cann_ops_transformer",
)

_HIDDEN_DIM = 16
_SEEDS = (5, 6, 7)

# (id, lens, layout, heads, head_dim, compress_ratio, topk, token_budget)
# Layouts spread the entry's three segment-length derivations: packed batches
# described by cu boundaries alone, by cu plus the host-side packed list, and
# equal-length unpacked batches.
_CASES = [
    ("single-segment", [64], "packed-cu", 4, 128, 4, 8, 40),
    ("equal-length-batch", [128, 128], "unpacked", 8, 128, 4, 16, 70),
    ("every-token-blocks", [96], "packed-cu", 8, 128, 1, 24, 30),
    ("wide-blocks", [256], "packed-lens", 4, 128, 32, 4, 140),
    ("packed-varlen", [130, 100, 97], "packed-lens", 4, 128, 4, 512, 520),
    ("prefill-scale", [1002, 998], "packed-cu", 8, 128, 4, 48, 200),
    ("tail-truncation", [64], "packed-cu", 4, 128, 4, 8, 20),
    ("no-complete-block", [4, 4], "packed-lens", 4, 128, 8, 4, 12),
]
_CASE_IDS = [case[0] for case in _CASES]


def _layout(lens: list[int], layout: str):
    if layout == "unpacked":
        return len(lens), lens[0], None, None
    boundaries = [0]
    for length in lens:
        boundaries.append(boundaries[-1] + length)
    cu = torch.tensor(boundaries, dtype=torch.int32)
    if layout == "packed-cu":
        return 1, sum(lens), cu, None
    if layout == "packed-lens":
        return 1, sum(lens), cu, list(lens)
    raise ValueError(f"unknown layout {layout}")


def _build_case(case: tuple, device, seed: int = 5):
    """Deterministic bf16 stub batch for one case and seed."""
    case_id, lens, layout, heads, head_dim, rate, topk, budget = case
    batch, seq_len, cu, packed = _layout(lens, layout)
    torch.manual_seed(seed)
    hidden = torch.randn(batch, seq_len, _HIDDEN_DIM).to(device=device, dtype=torch.bfloat16)
    proj = torch.nn.Linear(_HIDDEN_DIM, heads * head_dim + head_dim, bias=False).to(
        device=device, dtype=torch.bfloat16
    )
    indexer = SimpleNamespace(
        index_kv_heads=1,
        index_n_heads=heads,
        index_head_dim=head_dim,
        compress_ratio=rate,
        block_topk=topk,
        token_budget=budget,
        index_qk_proj=proj,
        q_layernorm=torch.nn.Identity(),
        k_layernorm=torch.nn.Identity(),
        veomni_rope=lambda query, key, cos, sin, **kwargs: (query, key),
    )
    positions = (
        torch.ones(batch, seq_len, head_dim, device=device),
        torch.zeros(batch, seq_len, head_dim, device=device),
    )
    return case_id, indexer, hidden, positions, cu, packed


def _forward(indexer, hidden, positions, cu, packed):
    """Run the generated modeling forward — the dispatch routes on the stub attr."""
    return Qwen4ExpTextQSAIndexer.forward(
        indexer,
        hidden,
        positions,
        None,
        None,
        cu_seq_lens_q=cu,
        packed_seq_lens=packed,
    )


def _assert_same_selection(actual: torch.Tensor, expected: torch.Tensor, case_id: str):
    assert actual.shape == expected.shape, f"[{case_id}] shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    assert actual.dtype == torch.int32, f"[{case_id}] dtype {actual.dtype} != torch.int32"
    actual_rows = actual.sort(dim=-1).values
    expected_rows = expected.sort(dim=-1).values
    mismatched = (actual_rows != expected_rows).any(dim=-1).nonzero()
    if mismatched.numel():
        details = []
        for batch, local in mismatched[:3].tolist():
            got = actual[batch, local]
            want = expected[batch, local]
            details.append(
                f"row (b={batch}, l={local}): "
                f"actual={sorted(got[got >= 0].tolist())} expected={sorted(want[want >= 0].tolist())}"
            )
        raise AssertionError(f"[{case_id}] {mismatched.shape[0]} rows differ; " + "; ".join(details))


@_REQUIRE_NPU
@pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
def test_npu_ascendc_matches_eager(case):
    """Same entry, same fixed input, ``eager`` vs ``npu_ascendc``; multiple seeded rounds."""
    device = get_device_type()
    for seed in _SEEDS:
        case_id, indexer, hidden, positions, cu, packed = _build_case(case, device, seed)
        indexer.qsa_indexer_implementation = "eager"
        torch.npu.synchronize()
        _t_eager_start = time.perf_counter()
        eager = _forward(indexer, hidden, positions, cu, packed)
        torch.npu.synchronize()
        _t_eager = (time.perf_counter() - _t_eager_start) * 1e3
        indexer.qsa_indexer_implementation = "npu_ascendc"
        torch.npu.synchronize()
        _t_light_start = time.perf_counter()
        lightning = _forward(indexer, hidden, positions, cu, packed)
        torch.npu.synchronize()
        _t_lightning = (time.perf_counter() - _t_light_start) * 1e3
        print(
            f"[timing] {case_id} seed{seed}: eager {_t_eager:.3f} ms | npu_ascendc {_t_lightning:.3f} ms",
            flush=True,
        )
        _assert_same_selection(lightning, eager, f"npu_ascendc_vs_eager[{case_id}-seed{seed}]")


# =======================
# |  ENTRY CONTRACTS    |
# =======================


def test_forward_lightning_rejects_non_npu(monkeypatch: pytest.MonkeyPatch):
    from veomni.ops.kernels.qsa.ascendc import qsa_indexer as cann_indexer

    monkeypatch.setattr(cann_indexer, "IS_NPU_AVAILABLE", False)

    with pytest.raises(ValueError, match="requires an Ascend NPU"):
        cann_indexer.qwen4_exp_qsa_indexer_forward_lightning(
            SimpleNamespace(), None, (None, None), None, None
        )


def test_forward_lightning_rejects_sequence_parallel(monkeypatch: pytest.MonkeyPatch):
    import veomni.distributed.parallel_state as parallel_state
    from veomni.ops.kernels.qsa.ascendc import qsa_indexer as cann_indexer

    monkeypatch.setattr(cann_indexer, "IS_NPU_AVAILABLE", True)
    monkeypatch.setattr(
        parallel_state,
        "get_parallel_state",
        lambda: SimpleNamespace(ulysses_enabled=True, cp_enabled=False),
    )

    with pytest.raises(ValueError, match="Ulysses/CP"):
        cann_indexer.qwen4_exp_qsa_indexer_forward_lightning(
            SimpleNamespace(), None, (None, None), None, None
        )

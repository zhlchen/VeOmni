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

"""CANN pool_key_indexer backend for the Qwen4-Exp QSA block selection.

Inline dispatch from the generated modeling forward: this replaces only the
eager TopK loop and returns its ``selected_local`` PRE-tail intermediate
(``int32`` token columns, ``-1`` padding); the shared eager
tail/concat/pad/truncation finishes the row for both implementations. The
op's calling contract — alignment, tail lanes, dtypes — is documented at
the call sites. MindSpeed-MM's integration pointed at the fused-op family.
"""

from __future__ import annotations

import torch
from torch import Tensor

from veomni.utils.device import IS_NPU_AVAILABLE

if IS_NPU_AVAILABLE:
    from cann_ops_transformer.ops import pool_key_indexer

_WEIGHT_CACHE: dict[tuple, Tensor] = {}
_TAIL_K_CACHE: dict[tuple, Tensor] = {}

# Segment-causal floor rule: block j is visible to query row t iff
# j * cr + cr - 1 <= t — the eager segment loop's rule, and what the
# op's mask_mode 3 implements.
_SEGMENT_CAUSAL_MASK_MODE = 3


@torch.no_grad()
def qsa_poolkey_selected_token_ids(
    index_queries: Tensor,
    pooled_keys: Tensor,
    *,
    token_budget: int,
    compress_ratio: int,
    block_topk: int | None = None,
) -> Tensor:
    """Eager-format TopK block expansion via pool_key_indexer.

    ``index_queries`` is ``[B, S, H, D]`` and ``pooled_keys`` is
    ``[B, blocks, D]`` — already normed and rotated by the shared modeling
    chain. Returns exactly what the eager loop builds as ``selected_local``
    BEFORE the tail concatenation: ``int32
    [B, S, min(block_topk, blocks) * compress_ratio]`` per-batch LOCAL token
    ids in score-ordered block lanes with ``-1`` padding. The shared modeling
    code appends the tail lanes, pads, and truncates to ``output_width``.
    """
    batch, query_length, query_heads, head_dim = index_queries.shape
    key_blocks = pooled_keys.shape[1]
    if block_topk is None:  # indexer attribute; the two differ when budget % cr != 0
        block_topk = token_budget // compress_ratio
    supported = (
        index_queries.device.type == "npu"
        and index_queries.dtype in (torch.float16, torch.bfloat16)
        and pooled_keys.dtype == index_queries.dtype
        and pooled_keys.shape == (batch, key_blocks, head_dim)
        and head_dim == 128
        and token_budget > 0
        and block_topk >= 1
    )
    if not supported:
        raise RuntimeError("QSA shape is unsupported by the CANN pool_key_indexer")

    device = index_queries.device
    # Token-level topk mirrors eager's TopK width: the op selects
    # topk // cr blocks and expands each into cr token lanes.
    topk_tokens = min(block_topk, key_blocks) * compress_ratio
    # Weights must share the query's dtype (fp32 is rejected); 1/sqrt(D)
    # matches eager's ReLU-sum / sqrt(head_dim). Cached by shape, shared
    # across per-layer indexer instances.
    weights_key = (str(device), batch, query_length, query_heads, index_queries.dtype)
    weights = _WEIGHT_CACHE.get(weights_key)
    if weights is None:
        weights = torch.full(
            (batch, query_length, query_heads),
            head_dim**-0.5,
            dtype=pooled_keys.dtype,
            device=device,
        )
        _WEIGHT_CACHE[weights_key] = weights
    # The honest residual pins the op's suffix alignment at zero shift:
    # kv_span = key_blocks * cr + pool_tail_k must equal query_length, else
    # every row shifts down by the difference.
    residual = query_length % compress_ratio
    tail_key = (str(device), batch, residual)
    tail_k = _TAIL_K_CACHE.get(tail_key)
    if tail_k is None:
        tail_k = torch.full((batch,), residual, dtype=torch.int64, device=device)
        _TAIL_K_CACHE[tail_key] = tail_k
    indices, _ = pool_key_indexer(
        index_queries.contiguous(),
        pooled_keys.unsqueeze(2),
        weights,
        tail_k,
        layout_q="BSND",
        layout_k="BSND",
        topk=topk_tokens,
        pool_size=compress_ratio,
        mask_mode=_SEGMENT_CAUSAL_MASK_MODE,
        return_value=False,
    )
    # The op's tail lanes (columns [topk_tokens, topk_tokens + cr)) follow a
    # different contract than eager's row-local tails — drop them; the
    # shared modeling tail construction appends the real ones.
    return indices[..., :topk_tokens]


def qwen4_exp_qsa_indexer_forward_lightning(
    self,
    q: Tensor,
    pooled_keys: Tensor,
    cu_seq_lens_q: Tensor | None = None,
    packed_seq_lens: list[int] | tuple[int, ...] | None = None,
) -> Tensor:
    """TopK block selection for ``Qwen4ExpTextQSAIndexer.forward`` (inline dispatch).

    Called with the rotated ``q`` ``[B, S, H, D]`` and pooled keys
    ``[1, blocks, D]`` after the shared projection/RoPE/pooling chain
    (single rank, no halo). Returns the eager loop's ``selected_local``
    PRE-tail: ``int32 [B, S, min(block_topk, blocks) * compress_ratio]``
    token ids with ``-1`` padding; the shared modeling code appends the
    tail lanes, pads, and truncates.

    The leading guards fire before any argument is touched (the rejection
    contracts call this entry with placeholder tensors). Sequence
    parallelism has no kernel support — keep
    ``qsa_indexer_implementation = "eager"`` there.
    """
    if not IS_NPU_AVAILABLE:
        raise ValueError(
            "qsa_indexer_implementation='npu_ascendc' requires an Ascend NPU with cann_ops_transformer; "
            "use qsa_indexer_implementation='eager' on other devices."
        )
    from veomni.distributed.parallel_state import get_parallel_state

    parallel_state = get_parallel_state()
    if bool(getattr(parallel_state, "ulysses_enabled", False)) or bool(getattr(parallel_state, "cp_enabled", False)):
        raise ValueError(
            "The CANN QSA indexer does not implement the Ulysses/CP halo/gather path; "
            "use qsa_indexer_implementation='eager' for sequence parallelism."
        )
    if self.index_head_dim != 128:
        raise ValueError(
            f"The CANN QSA indexer (pool_key_indexer) requires indexer_head_dim == 128; got {self.index_head_dim}. "
            "Use qsa_indexer_implementation='eager' otherwise."
        )

    batch_size, local_seq_len = q.shape[0], q.shape[1]

    if cu_seq_lens_q is None:
        # Equal-length batch: one BSND call over [B, S]. The op's ids are
        # already per-batch local — exactly the eager unpacked contract.
        return qsa_poolkey_selected_token_ids(
            q,
            pooled_keys.view(batch_size, -1, self.index_head_dim),
            token_budget=self.token_budget,
            compress_ratio=self.compress_ratio,
            block_topk=self.block_topk,
        )

    # Packed flat batch [1, T]. Host-side segment lengths in flat order: the
    # packing collator gives packed_seq_lens; a bare cu_seq_lens_q works too
    # (the eager metadata path already syncs it the same way).
    if packed_seq_lens is not None:
        segment_lens = [int(value) for value in packed_seq_lens]
    else:
        boundaries = cu_seq_lens_q.tolist()
        segment_lens = [end - start for start, end in zip(boundaries, boundaries[1:])]
    if sum(segment_lens) != batch_size * local_seq_len:
        raise ValueError(
            "Qwen4-Exp compact QSA CANN indexer needs segment lengths that cover the batch; "
            f"got {segment_lens} for batch_size={batch_size}, local_seq_len={local_seq_len}."
        )
    seg_block_counts = [length // self.compress_ratio for length in segment_lens]
    if sum(seg_block_counts) != pooled_keys.shape[1]:
        raise ValueError(
            "Qwen4-Exp compact QSA CANN indexer needs the flat block list to hold exactly "
            f"the per-segment blocks; got {pooled_keys.shape[1]} blocks for segment lengths "
            f"{segment_lens} with compress_ratio={self.compress_ratio}."
        )
    # One core call per segment on LOCAL rows (the op rejects actual_seq_q
    # under BSND, so there is no native varlen): the honest per-segment
    # pool_tail_k keeps the alignment exact. A segment's frame can be
    # narrower than eager's flat-batch TopK width, so the output is
    # pre-filled with -1 and each segment writes only its valid columns.
    columns = min(self.block_topk, pooled_keys.shape[1]) * self.compress_ratio
    out = torch.full((local_seq_len, columns), -1, dtype=torch.int32, device=q.device)
    token_start = 0
    block_start = 0
    for length, seg_blocks in zip(segment_lens, seg_block_counts):
        selected = qsa_poolkey_selected_token_ids(
            q[0, token_start : token_start + length].unsqueeze(0),
            pooled_keys[0, block_start : block_start + seg_blocks].unsqueeze(0),
            token_budget=self.token_budget,
            compress_ratio=self.compress_ratio,
            block_topk=self.block_topk,
        )[0]
        # Local -> flat ids; -1 padding slots must survive the offset.
        flat = torch.where(selected >= 0, selected + token_start, selected)
        out[token_start : token_start + length, : selected.shape[-1]] = flat
        token_start += length
        block_start += seg_blocks
    return out.view(1, local_seq_len, columns)

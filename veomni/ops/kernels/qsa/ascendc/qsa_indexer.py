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

"""CANN LightningIndexerV2 backend for the Qwen4-Exp QSA block selection.

Inline dispatch from the generated modeling forward: this replaces only the
eager TopK loop and returns its ``selected_local`` PRE-tail intermediate
(``int32`` token columns, ``-1`` padding); the shared eager
tail/concat/pad/truncation finishes the row for both implementations. The op
consumes the already-pooled blocks directly — one TND varlen call per batch.
"""

from __future__ import annotations

import torch
from torch import Tensor

from veomni.utils.device import IS_NPU_AVAILABLE

if IS_NPU_AVAILABLE:
    from cann_ops_transformer.ops import lightning_indexer, lightning_indexer_metadata

_WEIGHT_CACHE: dict[tuple, Tensor] = {}

# Segment-causal floor rule: block j is visible to query row t iff
# j * cr + cr - 1 <= t — the eager segment loop's rule, and what the
# op's mask_mode 3 implements.
_SEGMENT_CAUSAL_MASK_MODE = 3


def qsa_lightning_selected_token_ids(
    index_queries: Tensor,
    pooled_keys: Tensor,
    segment_lens: list[int] | tuple[int, ...],
    *,
    token_budget: int,
    compress_ratio: int,
    block_topk: int | None = None,
) -> Tensor:
    """Eager-format TopK block expansion via LightningIndexerV2 (TND varlen).

    ``index_queries`` is ``[B, S, H, D]`` and ``pooled_keys`` is
    ``[B, blocks, D]`` — already normed and rotated by the shared modeling
    chain, flattened here into one segment-major flat batch (segments = the
    ``segment_lens`` entries, in query-row order). Returns the eager loop's
    ``selected_local`` BEFORE the tail concatenation: ``int32
    [B * S, min(block_topk, blocks) * compress_ratio]`` per-segment LOCAL
    token ids in score-ordered block lanes with ``-1`` padding.
    """
    batch, query_length, query_heads, head_dim = index_queries.shape
    flat_keys = pooled_keys.reshape(-1, head_dim)
    total_blocks = flat_keys.shape[0]
    if block_topk is None:  # indexer attribute; the two differ when budget % cr != 0
        block_topk = token_budget // compress_ratio
    segment_block_counts = [length // compress_ratio for length in segment_lens]
    supported = (
        index_queries.device.type == "npu"
        and index_queries.dtype in (torch.float16, torch.bfloat16)
        and pooled_keys.dtype == index_queries.dtype
        and sum(segment_block_counts) == total_blocks
        and head_dim == 128
        and token_budget > 0
        and block_topk >= 1
    )
    if not supported:
        raise RuntimeError("QSA shape is unsupported by the CANN lightning indexer")

    total_tokens = batch * query_length
    width = min(block_topk, total_blocks)
    device = index_queries.device
    if width < 1:  # no complete block anywhere: the eager empty contract
        return torch.empty((total_tokens, 0), dtype=torch.int32, device=device)
    # One TND call covers every segment: the cu_seqlens boundaries carry the
    # per-segment candidate sets and the segment-causal rule (mask_mode 3).
    cu_q_host = [0]
    cu_k_host = [0]
    for length, count in zip(segment_lens, segment_block_counts):
        cu_q_host.append(cu_q_host[-1] + length)
        cu_k_host.append(cu_k_host[-1] + count)
    cu_seq_lens_q = torch.tensor(cu_q_host, dtype=torch.int32, device=device)
    cu_seq_lens_k = torch.tensor(cu_k_host, dtype=torch.int32, device=device)
    # LightningIndexerV2 rejects cmp_residual_k when cmp_ratio is 1; the
    # caller's all-zero entries are dropped for that case.
    if compress_ratio > 1:
        cmp_residual_k = torch.tensor(
            [length % compress_ratio for length in segment_lens], dtype=torch.int32, device=device
        )
    else:
        cmp_residual_k = None
    # Uniform fp32 per-head weights: 1/sqrt(head_dim) reproduces the eager
    # ReLU-sum score scaling. Cached by shape, shared across layers.
    weights_key = (str(device), total_tokens, query_heads)
    weights = _WEIGHT_CACHE.get(weights_key)
    if weights is None:
        weights = torch.full((total_tokens, query_heads), head_dim**-0.5, dtype=torch.float32, device=device)
        _WEIGHT_CACHE[weights_key] = weights
    with torch.no_grad():
        # The TND plan encodes the per-segment task split, so it is rebuilt
        # per call (the boundaries change with every batch).
        metadata = lightning_indexer_metadata(
            query_heads,
            1,
            head_dim,
            width,
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_k=cu_seq_lens_k,
            cmp_residual_k=cmp_residual_k,
            batch_size=len(segment_lens),
            max_seqlen_q=max(segment_lens),
            max_seqlen_k=max(segment_block_counts),
            layout_q="TND",
            layout_k="TND",
            mask_mode=_SEGMENT_CAUSAL_MASK_MODE,
            cmp_ratio=compress_ratio,
        )
        indices, _ = lightning_indexer(
            index_queries.to(torch.bfloat16).reshape(total_tokens, query_heads, head_dim).contiguous(),
            flat_keys.to(torch.bfloat16).reshape(total_blocks, 1, head_dim).contiguous(),
            weights,
            width,
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_k=cu_seq_lens_k,
            cmp_residual_k=cmp_residual_k,
            metadata=metadata,
            max_seqlen_q=-1,
            layout_q="TND",
            layout_k="TND",
            mask_mode=_SEGMENT_CAUSAL_MASK_MODE,
            cmp_ratio=compress_ratio,
            return_value=0,
        )
    # The kernel returns per-segment block ids ([total_tokens, width]);
    # expand each into its cr token lanes the way the eager loop does
    # (block-major lanes, -1 padding survives).
    indices = indices.reshape(total_tokens, width)
    valid = indices >= 0
    safe_ids = indices.clamp(min=0).to(torch.long)
    block_offsets = torch.arange(compress_ratio, device=device)
    lanes = safe_ids * compress_ratio + block_offsets
    return lanes.masked_fill(~valid[..., None], -1).flatten(-2).to(torch.int32)


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
            f"The CANN QSA indexer (LightningIndexerV2) requires indexer_head_dim == 128; got {self.index_head_dim}. "
            "Use qsa_indexer_implementation='eager' otherwise."
        )

    batch_size, local_seq_len = q.shape[0], q.shape[1]

    if cu_seq_lens_q is None:
        # Equal-length batch: one TND call over [B, S] as B segments. The
        # per-segment ids are already per-batch local — the eager unpacked
        # contract.
        return qsa_lightning_selected_token_ids(
            q,
            pooled_keys,
            [local_seq_len] * batch_size,
            token_budget=self.token_budget,
            compress_ratio=self.compress_ratio,
            block_topk=self.block_topk,
        ).view(batch_size, local_seq_len, -1)

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
    selected = qsa_lightning_selected_token_ids(
        q,
        pooled_keys,
        segment_lens,
        token_budget=self.token_budget,
        compress_ratio=self.compress_ratio,
        block_topk=self.block_topk,
    )
    # Per-segment LOCAL -> flat ids; -1 padding slots must survive the offset.
    cu_q_host = [0]
    for length in segment_lens:
        cu_q_host.append(cu_q_host[-1] + length)
    cu_seq_q = torch.tensor(cu_q_host, dtype=torch.int32, device=q.device)
    total_tokens = batch_size * local_seq_len
    token_segment = torch.searchsorted(
        cu_seq_q, torch.arange(total_tokens, device=q.device, dtype=cu_seq_q.dtype), right=True
    ) - 1
    segment_starts = cu_seq_q[:-1].index_select(0, token_segment).view(total_tokens, 1)
    selected = torch.where(selected >= 0, selected + segment_starts, selected)
    return selected.view(1, local_seq_len, -1)

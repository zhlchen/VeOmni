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

"""CANN LightningIndexerV2 backend for the Qwen4-Exp QSA block selection (TND varlen)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from veomni.utils.device import IS_NPU_AVAILABLE

if TYPE_CHECKING:
    from transformers.cache_utils import Cache

    from veomni.models.transformers.qwen4_exp.qsa_metadata import PreparedQSAMetadata

if IS_NPU_AVAILABLE:
    from cann_ops_transformer.ops import lightning_indexer, lightning_indexer_metadata

_SEGMENT_CAUSAL_MASK_MODE = 3


def qsa_lightning_selected_block_ids(
    index_q: Tensor,
    compressed_keys: Tensor,
    cu_seq_lens_q: Tensor,
    cu_seq_lens_k: Tensor,
    cmp_residual_k: Tensor,
    compress_ratio: int,
    top_k: int,
    batch_size: int | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
) -> Tensor:
    """Return selected compressed-block ids for a flat packed batch (TND).

    ``index_q`` is ``[B, S, H, D]`` (flattened internally to ``B * S`` tokens)
    and ``compressed_keys`` is ``[N, D]`` pooled keys in segment-major order —
    the flat order of the caller's block list. ``cu_seq_lens_q`` /
    ``cu_seq_lens_k`` are ``int32 [b+1]`` device tensors of cumulative token
    and block counts (first element 0, last element ``B * S`` / ``N``);
    ``cmp_residual_k`` is ``int32 [b]`` with ``segment_len % compress_ratio``.
    Each segment is selected with its own floor-causal rule, matching the
    eager segment-causal loop exactly.

    Returns ``int32 [B, S, min(top_k, N)]`` global block ids into the flat
    block list with ``-1`` padding, mirroring the eager loop's
    ``min(top_k, total_blocks)`` width.
    """
    if index_q.ndim != 4 or compressed_keys.ndim != 2:
        raise ValueError(
            "qsa_indexer cann varlen takes [B,S,H,D] queries and flat [N,D] compressed keys; "
            f"got {tuple(index_q.shape)} and {tuple(compressed_keys.shape)}"
        )
    batch, seq_len, num_heads, head_dim = index_q.shape
    total_tokens = batch * seq_len
    total_blocks = compressed_keys.shape[0]
    if compressed_keys.shape[1] != head_dim:
        raise ValueError("qsa_indexer cann varlen key head dimension must match the queries")
    if cu_seq_lens_q.ndim != 1 or cu_seq_lens_k.ndim != 1 or cmp_residual_k.ndim != 1:
        raise ValueError("qsa_indexer cann varlen boundary tensors must be 1-D")
    if cu_seq_lens_q.numel() != cu_seq_lens_k.numel() or cu_seq_lens_q.numel() < 2:
        raise ValueError(
            "qsa_indexer cann varlen needs matching [b+1] cu_seq_lens_q/cu_seq_lens_k; "
            f"got {cu_seq_lens_q.numel()} and {cu_seq_lens_k.numel()}"
        )
    if cmp_residual_k.numel() != cu_seq_lens_q.numel() - 1:
        raise ValueError(
            "qsa_indexer cann varlen needs one cmp_residual_k entry per segment; "
            f"got {cmp_residual_k.numel()} for {cu_seq_lens_q.numel() - 1} segments"
        )
    if top_k < 1:
        raise ValueError(f"qsa_indexer cann needs top_k >= 1; got {top_k}")
    if batch_size is None:
        batch_size = cu_seq_lens_q.numel() - 1
    if max_seqlen_q is None:
        max_seqlen_q = -1
    if max_seqlen_k is None:
        max_seqlen_k = -1
    width = min(top_k, total_blocks)
    if width < 1:
        return torch.full((batch, seq_len, top_k), -1, dtype=torch.int32, device=index_q.device)

    device = index_q.device
    with torch.no_grad():
        # LightningIndexerV2 rejects cmp_residual_k when cmp_ratio is 1; the
        # caller's all-zero entries are dropped for that case.
        kernel_residual = cmp_residual_k if compress_ratio > 1 else None
        # The TND plan encodes the per-segment task split, so it is rebuilt
        # per call (the boundaries change with every batch).
        metadata = lightning_indexer_metadata(
            num_heads,
            1,
            head_dim,
            width,
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_k=cu_seq_lens_k,
            cmp_residual_k=kernel_residual,
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            layout_q="TND",
            layout_k="TND",
            mask_mode=_SEGMENT_CAUSAL_MASK_MODE,
            cmp_ratio=compress_ratio,
        )
        # Uniform per-head weights: 1/sqrt(head_dim) reproduces the eager ReLU-sum score scaling.
        weights = torch.full((total_tokens, num_heads), head_dim**-0.5, dtype=torch.float32, device=device)
        indices, _ = lightning_indexer(
            index_q.to(torch.bfloat16).reshape(total_tokens, num_heads, head_dim).contiguous(),
            compressed_keys.to(torch.bfloat16).reshape(total_blocks, 1, head_dim).contiguous(),
            weights,
            width,
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_k=cu_seq_lens_k,
            cmp_residual_k=kernel_residual,
            metadata=metadata,
            max_seqlen_q=-1,
            layout_q="TND",
            layout_k="TND",
            mask_mode=_SEGMENT_CAUSAL_MASK_MODE,
            cmp_ratio=compress_ratio,
            return_value=0,
        )

    indices = indices.reshape(total_tokens, width).to(torch.int32)
    # The kernel returns per-segment block ids; offset them into the flat
    # segment-major block list (invalid -1 slots stay -1).
    token_segment = torch.searchsorted(
        cu_seq_lens_q, torch.arange(total_tokens, device=device, dtype=cu_seq_lens_q.dtype), right=True
    ) - 1
    segment_offsets = cu_seq_lens_k[:-1].index_select(0, token_segment).view(total_tokens, 1)
    indices = torch.where(indices >= 0, indices + segment_offsets, indices)
    return indices.reshape(batch, seq_len, width)


def qwen4_exp_qsa_indexer_forward_lightning(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None,
    cu_seq_lens_q: torch.Tensor | None = None,
    packed_seq_lens: list[int] | tuple[int, ...] | None = None,
    qsa_metadata: PreparedQSAMetadata | None = None,
) -> torch.Tensor:
    """CANN Lightning Indexer drop-in for ``Qwen4ExpTextQSAIndexer.forward``.

    Covers every layout the single-rank eager path covers — packed flat
    batches (``cu_seq_lens_q``/``packed_seq_lens``) and equal-length
    non-packed batches — and returns the same ``int32 [B, S, output_width]``
    token-id selection with ``-1`` padding, so the attention consumer cannot
    tell the implementations apart. Sequence parallelism (Ulysses/CP) and
    unsupported indexer shapes raise ``ValueError``; keep
    ``qsa_indexer_implementation = "eager"`` for those.
    """
    del attention_mask
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
    if past_key_values is not None:
        raise NotImplementedError("Qwen4-Exp compact QSA does not support a KV/indexer cache.")

    batch_size, local_seq_len, _ = hidden_states.shape
    if local_seq_len + self.compress_ratio > torch.iinfo(torch.int32).max:
        raise ValueError("Qwen4-Exp compact QSA token positions must fit int32, including compression padding.")
    if self.index_kv_heads != 1:
        raise ValueError(
            f"Qwen4-Exp compact QSA currently requires one indexer KV head; got {self.index_kv_heads}."
        )
    if self.compress_ratio > local_seq_len:
        raise ValueError(
            "Qwen4-Exp compact QSA needs each sequence-parallel shard to be at least one compression block wide; "
            f"got local_seq_len={local_seq_len}, compress_ratio={self.compress_ratio}."
        )
    if self.index_head_dim != 128:
        raise ValueError(
            f"The CANN Lightning Indexer requires indexer_head_dim == 128; got {self.index_head_dim}. "
            "Use qsa_indexer_implementation='eager' otherwise."
        )

    qk = self.index_qk_proj(hidden_states)
    q_width = self.index_n_heads * self.index_head_dim
    q, raw_keys = torch.split(qk, [q_width, self.index_head_dim], dim=-1)
    q = self.q_layernorm(q.reshape(batch_size, local_seq_len, self.index_n_heads, self.index_head_dim))
    local_cos, local_sin = position_embeddings
    if local_cos.shape[1] != local_seq_len or local_sin.shape[1] != local_seq_len:
        raise ValueError(
            "Qwen4-Exp compact QSA position embeddings must match the local sequence length; "
            f"got cos={local_cos.shape[1]}, sin={local_sin.shape[1]}, local_seq_len={local_seq_len}."
        )
    q, _ = self.veomni_rope(q, q, local_cos, local_sin, unsqueeze_dim=2)

    # Reuse tensors prepared once outside the decoder/checkpointed layer loop.
    # Direct component callers retain the same boundary validation and layout.
    if qsa_metadata is None:
        from veomni.models.transformers.qwen4_exp.qsa_metadata import prepare_qsa_metadata

        qsa_metadata = prepare_qsa_metadata(
            batch_size,
            local_seq_len,
            self.compress_ratio,
            0,
            1,
            hidden_states.device,
            cu_seq_lens_q,
            packed_seq_lens,
        )
    if qsa_metadata.layout != (batch_size, local_seq_len, self.compress_ratio, 0, 1):
        raise ValueError("Qwen4-Exp QSA metadata does not match the current sequence-parallel layout.")
    if qsa_metadata.local_segment_ids.device != hidden_states.device:
        raise ValueError("Qwen4-Exp QSA metadata must be on the hidden states' device.")
    local_batch_ids = qsa_metadata.local_batch_ids
    local_block_starts = qsa_metadata.local_block_starts
    block_starts = qsa_metadata.block_starts
    local_segment_starts = qsa_metadata.local_segment_starts

    # Single rank: no halo exchange, and the owned block list is the flat
    # segment-major block list the kernel indexes into.
    if local_batch_ids.numel():
        local_offsets = local_block_starts
        window_indices = (
            local_offsets[:, None] + torch.arange(self.compress_ratio, device=hidden_states.device)[None]
        )
        pooled_keys = (
            raw_keys[local_batch_ids[:, None], window_indices].float().mean(dim=1).to(raw_keys.dtype)
        )
        pooled_keys = self.k_layernorm(pooled_keys)
        # Rotate the owned blocks as one [1, 1, blocks, D] sequence, the rope op's
        # [B, H, S, D] layout, with per-block start positions.
        pooled_keys = pooled_keys[None, None]
        pooled_keys, _ = self.veomni_rope(
            pooled_keys,
            pooled_keys,
            local_cos[local_batch_ids, local_offsets][None],
            local_sin[local_batch_ids, local_offsets][None],
        )
        pooled_keys = pooled_keys[0]
    else:
        # Keep both the projection and RMSNorm parameters in the autograd graph
        # on batches that own no complete block.
        pooled_keys = self.k_layernorm(raw_keys[:, :0]).reshape(1, 0, self.index_head_dim)

    output_width = self.token_budget + self.compress_ratio - 1
    query_positions = torch.arange(local_seq_len, device=hidden_states.device)

    if block_starts.numel():
        # Host-side segment lengths in flat order: the packing collator gives
        # packed_seq_lens; unpacked batches are one equal-length segment per
        # row; a bare cu_seq_lens_q works too (the eager metadata path already
        # syncs it the same way).
        if cu_seq_lens_q is None:
            segment_lens = [local_seq_len] * batch_size
        elif packed_seq_lens is not None:
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
        if sum(seg_block_counts) != block_starts.numel():
            raise ValueError(
                "Qwen4-Exp compact QSA CANN indexer needs the flat block list to hold exactly "
                f"the per-segment blocks; got {block_starts.numel()} blocks for segment lengths "
                f"{segment_lens} with compress_ratio={self.compress_ratio}."
            )
        cu_q_host = [0]
        cu_k_host = [0]
        for length, count in zip(segment_lens, seg_block_counts):
            cu_q_host.append(cu_q_host[-1] + length)
            cu_k_host.append(cu_k_host[-1] + count)
        device = hidden_states.device
        block_ids = qsa_lightning_selected_block_ids(
            index_q=q,
            compressed_keys=pooled_keys[0],
            cu_seq_lens_q=torch.tensor(cu_q_host, dtype=torch.int32, device=device),
            cu_seq_lens_k=torch.tensor(cu_k_host, dtype=torch.int32, device=device),
            cmp_residual_k=torch.tensor(
                [length % self.compress_ratio for length in segment_lens], dtype=torch.int32, device=device
            ),
            compress_ratio=self.compress_ratio,
            top_k=self.block_topk,
            batch_size=len(segment_lens),
            max_seqlen_q=max(segment_lens),
            max_seqlen_k=max(seg_block_counts),
        )
        block_offsets = torch.arange(self.compress_ratio, dtype=torch.int32, device=device)
        valid_ids = block_ids >= 0
        safe_ids = block_ids.clamp(min=0).to(torch.long)
        selected_starts = block_starts[safe_ids].to(torch.int32)
        selected_local = (
            (selected_starts[..., None] + block_offsets).masked_fill(~valid_ids[..., None], -1).flatten(-2)
        )
    else:
        selected_local = torch.empty(
            batch_size,
            local_seq_len,
            0,
            dtype=torch.long,
            device=hidden_states.device,
        )

    visible_count = query_positions[None] - local_segment_starts + 1
    tail_count = torch.remainder(visible_count, self.compress_ratio)
    tail_start = (
        local_segment_starts
        + torch.div(visible_count, self.compress_ratio, rounding_mode="floor") * self.compress_ratio
    )
    tail_offsets = torch.arange(self.compress_ratio - 1, device=hidden_states.device)
    tails = tail_start[..., None] + tail_offsets
    tails = tails.to(torch.int32).masked_fill(tail_offsets[None, None] >= tail_count[..., None], -1)

    selected_local = torch.cat((selected_local, tails), dim=-1)
    if selected_local.shape[-1] < output_width:
        selected_local = torch.nn.functional.pad(
            selected_local,
            (0, output_width - selected_local.shape[-1]),
            value=-1,
        )
    selected_local = selected_local[..., :output_width].to(torch.int32).contiguous()
    return selected_local

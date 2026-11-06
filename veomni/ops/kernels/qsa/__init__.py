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

"""Qwen4-Exp QSA indexer (block selection) kernels.

The indexer projects the hidden states and selects the top-k compressed
key-value blocks visible to each query token. ``eager`` is the reference
``einsum + ReLU + topk`` loop in the generated modeling file.
``ascendc`` routes the selection through the CANN Lightning IndexerV2
operator on Ascend NPUs, mirroring the ``ops/qsa/`` layout of the
MindSpeed-MM repository.
"""

# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Patch configuration for the Qwen4-Exp NPU correctness path.

Regen command:
patchgen veomni.models.transformers.qwen4_exp.qwen4_exp_npu_patch_gen_config -o veomni/models/transformers/qwen4_exp/generated --diff

Qwen4-Exp uses partial interleaved mRoPE and experimental QSA/PLE code paths,
so this initial NPU build deliberately inherits the device-agnostic GPU patch
set without adding incompatible fused rotary or attention replacements.
"""

from copy import deepcopy

from veomni.models.transformers.qwen4_exp.qwen4_exp_gpu_patch_gen_config import config as gpu_config


config = deepcopy(gpu_config)
config.target_file = "patched_modeling_qwen4_exp_npu.py"
config.description = "Qwen4-Exp NPU VLM-SFT correctness integration with PLE sharding"

# Record the QSA indexer implementation knob next to the indexer's partial
# rope op, mirroring the MindSpeed-MM qsa_indexer_implementation format.
for _patch in config.patches:
    if _patch.target == "Qwen4ExpTextQSAIndexer.__init__":
        _patch.name_map = {
            "self.veomni_rope = VeomniOp(\"rope\", \"partial\", config.ops_config.rotary_pos_emb_implementation)": (
                "self.veomni_rope = VeomniOp(\"rope\", \"partial\", config.ops_config.rotary_pos_emb_implementation)\n"
                "        self.qsa_indexer_implementation = getattr(config.ops_config, "
                "\"qsa_indexer_implementation\", \"eager\")"
            ),
        }
        break
else:
    raise RuntimeError("Expected Qwen4ExpTextQSAIndexer.__init__ in qwen4_exp GPU patch config")

# Replace the GPU QSA indexer mask value in-place so NPU masks invisible block
# scores with -1.0 instead of -inf. ReLU block scores are non-negative, and
# CANN TopKV2 faults (AICore error 507015) on sparse rows dominated by -inf.
# Then dispatch the CANN Lightning Indexer (TND varlen, cann_ops_transformer)
# at the top of the forward: qsa_indexer_implementation == "npu_ascendc" routes
# the whole call to veomni.ops.kernels.qsa.ascendc.qsa_indexer, a standalone
# drop-in that reproduces the eager selection for the flat packed layout
# without Ulysses/CP (those raise there). The eager loop below — including
# the block_starts.numel() chunked einsum+ReLU+topk selection — stays
# byte-identical as the reference path.
for _patch in config.patches:
    if _patch.target == "Qwen4ExpTextQSAIndexer.forward":
        _patch.name_map = {
            'scores = scores.masked_fill(~visible, float("-inf"))': (
                "scores = scores.masked_fill(~visible, -1.0)  # NPU TopKV2 faults on sparse -inf rows (ReLU scores >= 0)"
            ),
            "del attention_mask": (
                "if getattr(self, \"qsa_indexer_implementation\", \"eager\") == \"npu_ascendc\":\n"
                "            from veomni.ops.kernels.qsa.ascendc.qsa_indexer import "
                "qwen4_exp_qsa_indexer_forward_lightning\n\n"
                "            return qwen4_exp_qsa_indexer_forward_lightning(\n"
                "                self,\n"
                "                hidden_states,\n"
                "                position_embeddings,\n"
                "                attention_mask,\n"
                "                past_key_values,\n"
                "                cu_seq_lens_q=cu_seq_lens_q,\n"
                "                packed_seq_lens=packed_seq_lens,\n"
                "                qsa_metadata=qsa_metadata,\n"
                "            )\n"
                "        del attention_mask"
            ),
        }
        break
else:
    raise RuntimeError("Expected Qwen4ExpTextQSAIndexer.forward in qwen4_exp GPU patch config")

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for Qwen3.8-Flash-Next configuration loading and validation."""

import os

from absl.testing import absltest
from maxtext.configs import pyconfig
from maxtext.configs import types
from maxtext.utils import globals as maxtext_globals
import pytest


@pytest.mark.cpu_only
class Qwen38FlashNextConfigTest(absltest.TestCase):
  """Tests full and mini YAML configurations for Qwen3.8-Flash-Next."""

  def setUp(self):
    super().setUp()
    self.base_yml = os.path.join(maxtext_globals.MAXTEXT_CONFIGS_DIR, "base.yml")
    self.entry_script = os.path.join(maxtext_globals.MAXTEXT_PKG_DIR, "train.py")

  def test_qwen38_flash_next_full_config(self):
    """Verifies full 180B Qwen3.8-Flash-Next config against HF reference and invariants."""
    config = pyconfig.initialize(
        [self.entry_script, self.base_yml],
        model_name="qwen3.8-flash-next",
        run_name="test_qwen38_flash_next_full",
    )
    # Core architecture (HF config.json)
    self.assertEqual(config.base_emb_dim, 2560)  # HF: hidden_size=2560
    self.assertEqual(config.base_num_decoder_layers, 48)  # HF: num_hidden_layers=48
    self.assertEqual(config.base_num_query_heads, 24)  # HF: num_attention_heads=24
    self.assertEqual(config.base_num_kv_heads, 2)  # HF: num_key_value_heads=2
    self.assertEqual(config.head_dim, 256)  # HF: head_dim=256
    self.assertEqual(config.vocab_size, 248320)  # HF: vocab_size=248320
    self.assertEqual(config.normalization_layer_epsilon, 1.0e-6)  # HF: rms_norm_eps=1.0e-6
    self.assertFalse(config.logits_via_embedding)

    # Hyper-Connections (HF config.json)
    self.assertEqual(config.hc_count, 4)  # HF: hc_count=4
    self.assertEqual(config.hc_lowrank, 320)  # HF: hc_lowrank=320

    # MoE (HF config.json)
    self.assertEqual(config.num_experts, 512)  # HF: num_experts=512
    self.assertEqual(config.num_experts_per_tok, 10)  # HF: num_experts_per_tok=10
    self.assertEqual(config.base_mlp_dim, 640)  # HF: moe_intermediate_size=640
    self.assertEqual(config.base_moe_mlp_dim, 640)  # HF: moe_intermediate_size=640
    self.assertEqual(config.shared_experts, 1)
    self.assertEqual(config.routed_scaling_factor, 1.0)
    self.assertTrue(config.norm_topk_prob)  # HF: norm_topk_prob=True
    self.assertEqual(config.load_balance_loss_weight, 0.001)  # HF: router_aux_loss_coef=0.001

    # GDN linear attention (HF config.json)
    self.assertEqual(config.inhomogeneous_layer_cycle_interval, 4)  # HF: full_attention_interval=4
    self.assertEqual(config.gdn_conv_kernel_dim, 4)  # HF: linear_conv_kernel_dim=4
    self.assertEqual(config.gdn_key_head_dim, 128)  # HF: linear_key_head_dim=128
    self.assertEqual(config.gdn_value_head_dim, 128)  # HF: linear_value_head_dim=128
    self.assertEqual(config.gdn_num_key_heads, 16)  # HF: linear_num_key_heads=16
    self.assertEqual(config.gdn_num_value_heads, 48)  # HF: linear_num_value_heads=48
    self.assertEqual(config.gdn_chunk_size, 64)
    self.assertEqual(config.gdn_output_gate_activation, "sigmoid")  # HF: output_gate_type="sigmoid"
    self.assertTrue(config.use_qk_l2norm_in_kernel)

    # RoPE and mRoPE (HF config.json)
    self.assertEqual(config.rope_max_timescale, 10000000.0)  # HF: rope_theta=10000000.0
    self.assertEqual(config.partial_rotary_factor, 0.25)  # HF: partial_rotary_factor=0.25
    self.assertEqual(list(config.mrope_section), [11, 11, 10])  # HF: mrope_section=[11, 11, 10]

    # QSA and Indexer (HF config.json)
    self.assertEqual(config.index_n_heads, 4)  # HF: indexer_n_heads=4
    self.assertEqual(config.index_kv_heads, 1)  # HF: indexer_kv_heads=1
    self.assertEqual(config.index_head_dim, 128)  # HF: indexer_head_dim=128
    self.assertEqual(config.index_select_block_size, 4)  # HF: indexer_compress_ratio=4
    self.assertEqual(config.index_select_budget, 2048)  # HF: indexer_budget=2048

    # PLE (HF config.json & configuration_qwen4_exp.py)
    self.assertTrue(config.use_ple_layer)
    self.assertEqual(config.ple_layer_id, 1)  # HF: ple_layer_ids=[2] (0-indexed 1 = 1-indexed 2)
    self.assertEqual(config.ple_embed_dim, 2560)  # HF: ple_embed_dim=2560
    self.assertEqual(config.ple_conv_kernel_size, 4)  # HF: ple_conv_kernel_size=4
    self.assertEqual(config.ngram_size, 3)  # HF: ngram_size=3
    self.assertEqual(config.heads_per_ngram, 8)  # HF: heads_per_ngram=8
    self.assertEqual(config.ngram_vocab_size_base, 20_000_000)  # HF: ngram_vocab_size_base=20000000
    self.assertEqual(config.make_ngram_vocab_size_divisible_by, 128)  # HF: make_ngram_vocab_size_divisible_by=128
    self.assertEqual(config.split_ngram_parts, 128)  # HF: split_ngram_parts=128
    self.assertEqual(config.ple_seed, 1234)  # HF: seed=1234
    self.assertEqual(config.ple_eos_token_id, 248044)  # HF: eos_token_id=248044

    # MTP
    self.assertEqual(config.mtp_num_layers, 1)
    self.assertEqual(config.mtp_loss_factor, 0.1)

    # HF-derived architectural invariants
    # 1. ple_embed_dim divisible by total n-gram heads: ((ngram_size - 1) * heads_per_ngram)
    ngram_heads = (config.ngram_size - 1) * config.heads_per_ngram
    ple_emb = config.ple_embed_dim or config.base_emb_dim
    self.assertEqual(ple_emb % ngram_heads, 0)
    # 2. QSA token budget divisible by compression block size
    self.assertEqual(config.index_select_budget % config.index_select_block_size, 0)
    # 3. RoPE rotary dim fits within QSA index head dimension
    rotary_dim = int(config.head_dim * config.partial_rotary_factor)
    self.assertLessEqual(rotary_dim, config.index_head_dim)
    # 4. MRoPE total sections match rotary dimension: sum(mrope_section) * 2 == rotary_dim
    self.assertEqual(sum(config.mrope_section) * 2, rotary_dim)
    # 5. PLE layer placed on a linear attention layer (not full/sparse attention)
    cycle = config.inhomogeneous_layer_cycle_interval
    self.assertNotEqual((config.ple_layer_id + 1) % cycle, 0)
    # 6. Decoder layers form complete cycles of inhomogeneous interval
    self.assertEqual(config.base_num_decoder_layers % cycle, 0)
    # 7. ple_eos_token_id is in-range for vocab_size
    self.assertTrue(0 <= config.ple_eos_token_id < config.vocab_size)

  def test_qwen38_flash_next_mini_config(self):
    """Verifies that mini-twin config loads and satisfies HF architectural invariants."""
    config = pyconfig.initialize(
        [self.entry_script, self.base_yml],
        model_name="qwen3.8-flash-next-mini",
        run_name="test_qwen38_flash_next_mini",
    )
    # Core architecture
    self.assertEqual(config.base_emb_dim, 256)
    self.assertEqual(config.base_num_decoder_layers, 4)
    self.assertEqual(config.base_num_query_heads, 4)
    self.assertEqual(config.base_num_kv_heads, 2)
    self.assertEqual(config.head_dim, 64)
    self.assertEqual(config.vocab_size, 1024)
    self.assertEqual(config.normalization_layer_epsilon, 1.0e-6)
    self.assertFalse(config.logits_via_embedding)

    # Hyper-Connections
    self.assertEqual(config.hc_count, 4)
    self.assertEqual(config.hc_lowrank, 32)

    # MoE
    self.assertEqual(config.num_experts, 8)
    self.assertEqual(config.num_experts_per_tok, 2)
    self.assertEqual(config.base_mlp_dim, 64)
    self.assertEqual(config.base_moe_mlp_dim, 64)
    self.assertEqual(config.shared_experts, 1)
    self.assertEqual(config.routed_scaling_factor, 1.0)
    self.assertTrue(config.norm_topk_prob)
    self.assertEqual(config.load_balance_loss_weight, 0.001)

    # GDN linear attention
    self.assertEqual(config.inhomogeneous_layer_cycle_interval, 4)
    self.assertEqual(config.gdn_conv_kernel_dim, 4)
    self.assertEqual(config.gdn_key_head_dim, 64)
    self.assertEqual(config.gdn_value_head_dim, 64)
    self.assertEqual(config.gdn_num_key_heads, 2)
    self.assertEqual(config.gdn_num_value_heads, 4)
    self.assertEqual(config.gdn_chunk_size, 16)
    self.assertEqual(config.gdn_output_gate_activation, "sigmoid")
    self.assertTrue(config.use_qk_l2norm_in_kernel)

    # RoPE and mRoPE
    self.assertEqual(config.rope_max_timescale, 10000000.0)
    self.assertEqual(config.partial_rotary_factor, 0.25)
    self.assertEqual(list(config.mrope_section), [3, 3, 2])

    # QSA and Indexer
    self.assertEqual(config.index_n_heads, 2)
    self.assertEqual(config.index_kv_heads, 1)
    self.assertEqual(config.index_head_dim, 32)
    self.assertEqual(config.index_select_block_size, 4)
    self.assertEqual(config.index_select_budget, 16)

    # PLE
    self.assertTrue(config.use_ple_layer)
    self.assertEqual(config.ple_layer_id, 1)
    self.assertEqual(config.ple_embed_dim, 256)
    self.assertEqual(config.ple_conv_kernel_size, 4)
    self.assertEqual(config.ngram_size, 3)
    self.assertEqual(config.heads_per_ngram, 2)
    self.assertEqual(config.ngram_vocab_size_base, 1000)
    self.assertEqual(config.make_ngram_vocab_size_divisible_by, 4)
    self.assertEqual(config.split_ngram_parts, 4)
    self.assertEqual(config.ple_seed, 1234)
    self.assertEqual(config.ple_eos_token_id, 3)

    # MTP
    self.assertEqual(config.mtp_num_layers, 1)
    self.assertEqual(config.mtp_loss_factor, 0.1)

    # HF-derived architectural invariants on mini-twin
    ngram_heads = (config.ngram_size - 1) * config.heads_per_ngram
    ple_emb = config.ple_embed_dim or config.base_emb_dim
    self.assertEqual(ple_emb % ngram_heads, 0)
    self.assertEqual(config.index_select_budget % config.index_select_block_size, 0)
    rotary_dim = int(config.head_dim * config.partial_rotary_factor)
    self.assertLessEqual(rotary_dim, config.index_head_dim)
    self.assertEqual(sum(config.mrope_section) * 2, rotary_dim)
    cycle = config.inhomogeneous_layer_cycle_interval
    self.assertNotEqual((config.ple_layer_id + 1) % cycle, 0)
    self.assertEqual(config.base_num_decoder_layers % cycle, 0)
    # 7. ple_eos_token_id is in-range for vocab_size
    self.assertTrue(0 <= config.ple_eos_token_id < config.vocab_size)

  def test_model_names_registered_in_globals(self):
    """Verifies both model names are registered in globals.HF_IDS."""
    self.assertIn("qwen3.8-flash-next", maxtext_globals.HF_IDS)
    self.assertIn("qwen3.8-flash-next-mini", maxtext_globals.HF_IDS)
    self.assertEqual(maxtext_globals.HF_IDS["qwen3.8-flash-next"], "Qwen/Qwen3.8-Flash-Next")
    self.assertEqual(maxtext_globals.HF_IDS["qwen3.8-flash-next-mini"], "Qwen/Qwen3.8-Flash-Next")

  def test_default_pydantic_config_fields(self):
    """Verifies default values in Qwen38FlashNext match HF reference defaults."""
    cfg = types.Qwen38FlashNext.model_validate({})
    self.assertEqual(cfg.hc_count, 4)
    self.assertEqual(cfg.hc_lowrank, 320)
    self.assertEqual(cfg.index_n_heads, 4)
    self.assertEqual(cfg.index_kv_heads, 1)
    self.assertEqual(cfg.index_head_dim, 128)
    self.assertEqual(cfg.index_select_block_size, 4)
    self.assertEqual(cfg.index_select_budget, 2048)
    self.assertEqual(cfg.gdn_output_gate_activation, "sigmoid")
    self.assertFalse(cfg.use_ple_layer)
    self.assertEqual(cfg.ple_layer_id, 1)
    self.assertIsNone(cfg.ple_embed_dim)
    self.assertEqual(cfg.ple_conv_kernel_size, 4)
    self.assertEqual(cfg.ngram_size, 3)
    self.assertEqual(cfg.heads_per_ngram, 8)
    self.assertEqual(cfg.ngram_vocab_size_base, 20_000_000)
    self.assertEqual(cfg.make_ngram_vocab_size_divisible_by, 128)
    self.assertEqual(cfg.split_ngram_parts, 128)
    self.assertEqual(cfg.ple_seed, 1234)
    self.assertEqual(cfg.ple_eos_token_id, 248044)
    self.assertIsNone(cfg.mtp_loss_factor)
    self.assertFalse(cfg.use_qk_l2norm_in_kernel)
    self.assertEqual(list(cfg.mrope_section), [11, 11, 10])


if __name__ == "__main__":
  absltest.main()

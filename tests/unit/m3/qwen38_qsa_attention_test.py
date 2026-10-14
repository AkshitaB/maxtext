# Copyright 2026 Google LLC
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

"""Unit tests for Qwen Sparse Attention (QSA), Indexer, and mRoPE."""

from typing import Any
from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3.models.qwen38_flash_next import qsa_attention  # pylint: disable=no-name-in-module
import numpy as np
import pytest
import torch
from transformers.models.qwen4_exp import configuration_qwen4_exp
from transformers.models.qwen4_exp import modeling_qwen4_exp


def _make_hf_config(**kwargs) -> configuration_qwen4_exp.Qwen4ExpTextConfig:
  """Constructs a test Qwen4ExpTextConfig."""
  rope = {
      "rope_type": "default",
      "rope_theta": 10000.0,
      "partial_rotary_factor": 0.25,
      "mrope_interleaved": True,
      "mrope_section": [2, 1, 1],
  }
  cfg = {
      "hidden_size": 64,
      "num_attention_heads": 4,
      "num_key_value_heads": 2,
      "head_dim": 32,
      "indexer_n_heads": 2,
      "indexer_kv_heads": 1,
      "indexer_head_dim": 32,
      "indexer_budget": 16,
      "indexer_compress_ratio": 4,
      "rms_norm_eps": 1e-6,
      "attention_bias": False,
      "attention_dropout": 0.0,
      "rope_parameters": rope,
  }
  cfg.update(kwargs)
  return configuration_qwen4_exp.Qwen4ExpTextConfig(**cfg)


def _copy_weights(pt: modeling_qwen4_exp.Qwen4ExpTextAttention, jx: qsa_attention.Qwen38SparseAttention):
  """Copies PyTorch attention weights to JAX QSA module."""

  def cv(t: torch.Tensor) -> jax.Array:
    return jnp.array(t.detach().cpu().numpy())

  jx.q_proj.kernel.value, jx.k_proj.kernel.value = cv(pt.q_proj.weight.T), cv(pt.k_proj.weight.T)
  jx.v_proj.kernel.value, jx.o_proj.kernel.value = cv(pt.v_proj.weight.T), cv(pt.o_proj.weight.T)
  jx.q_norm.scale.value, jx.k_norm.scale.value = cv(pt.q_norm.weight), cv(pt.k_norm.weight)
  jx.indexer.index_qk_proj.kernel.value = cv(pt.indexer.index_qk_proj.weight.T)
  jx.indexer.q_layernorm.scale.value = cv(pt.indexer.q_layernorm.weight)
  jx.indexer.k_layernorm.scale.value = cv(pt.indexer.k_layernorm.weight)


@pytest.mark.cpu_only
class Qwen38QSAAttentionTest(absltest.TestCase):
  """Tests QSA attention, indexer, and mRoPE against PyTorch reference."""

  def _make_jax_qsa(self, emb_dim: int = 64, rng_seed: int = 0) -> qsa_attention.Qwen38SparseAttention:
    """Constructs a test JAX QSA module."""
    args = (emb_dim, 4, 2, 32, 0.25, 10000.0, (2, 1, 1), 2, 1, 32, 16, 4, 1e-6, jnp.float32, jnp.float32)
    return qsa_attention.Qwen38SparseAttention(*args, rngs=nnx.Rngs(rng_seed))

  def test_qsa_parameter_tree_contract(self):
    """Verifies shapes and dtypes of all QSA parameters."""
    m = self._make_jax_qsa()
    self.assertEqual((m.q_proj.kernel.value.shape, m.k_proj.kernel.value.shape), ((64, 256), (64, 64)))
    self.assertEqual((m.v_proj.kernel.value.shape, m.o_proj.kernel.value.shape), ((64, 64), (128, 64)))
    self.assertEqual((m.q_norm.scale.value.shape, m.k_norm.scale.value.shape), ((32,), (32,)))
    self.assertEqual(
        (m.indexer.index_qk_proj.kernel.value.shape, m.indexer.q_layernorm.scale.value.shape), ((64, 96), (32,))
    )
    self.assertEqual((m.indexer.k_layernorm.scale.value.shape, m.q_proj.kernel.value.dtype), ((32,), jnp.float32))

  def _make_pt_and_jax(self, seed: int = 0) -> tuple[Any, Any, qsa_attention.Qwen38SparseAttention]:
    """Constructs parity-initialized PyTorch and JAX attention modules."""
    cfg = _make_hf_config()
    pt_r = modeling_qwen4_exp.Qwen4ExpTextRotaryEmbedding(cfg)
    pt_a = modeling_qwen4_exp.Qwen4ExpTextAttention(cfg, layer_idx=3)
    jax_a = self._make_jax_qsa(rng_seed=seed)
    _copy_weights(pt_a, jax_a)
    return pt_r, pt_a, jax_a

  def test_rotary_embedding_and_partial_mrope_pytorch_parity(self):
    """Verifies partial interleaved mRoPE matches PyTorch reference."""
    hf_cfg = _make_hf_config()
    pt_rope = modeling_qwen4_exp.Qwen4ExpTextRotaryEmbedding(hf_cfg)
    jax_rope = qsa_attention.Qwen38RotaryEmbedding(32, 0.25, 10000.0, (2, 1, 1))
    b, s = 2, 16
    pos_np = np.tile(np.arange(s), (b, 1))
    x_np = np.ones((b, s, 32), dtype=np.float32)

    pt_cos, pt_sin = pt_rope(torch.from_numpy(x_np), torch.from_numpy(pos_np))
    jax_cos, jax_sin = jax_rope(jnp.array(x_np), jnp.array(pos_np))

    np.testing.assert_allclose(np.array(jax_cos), pt_cos.detach().cpu().numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.array(jax_sin), pt_sin.detach().cpu().numpy(), rtol=1e-6, atol=1e-6)

  def test_qsa_attention_pytorch_weight_copy_parity_at_32(self):
    """Verifies output and selected-block parity vs HF attention at seq_len=32."""
    b, s, emb_dim = 1, 32, 64
    pt_rope, pt_attn, jax_attn = self._make_pt_and_jax(seed=0)

    x = np.random.default_rng(42).standard_normal((b, s, emb_dim)).astype(np.float32)
    pos = np.tile(np.arange(s), (b, 1))

    pt_cos, pt_sin = pt_rope(torch.from_numpy(x), torch.from_numpy(pos))
    mask = torch.tril(torch.ones((b, 1, s, s), dtype=torch.bool))
    pt_float_mask = torch.where(mask, torch.zeros((), dtype=torch.float32), torch.finfo(torch.float32).min)

    pt_out, _ = pt_attn(torch.from_numpy(x), (pt_cos, pt_sin), attention_mask=pt_float_mask)
    jax_out, jax_blocks = jax_attn(jnp.array(x), position_ids=jnp.array(pos), return_selected_blocks=True)

    pt_tok_mask = pt_attn.indexer(torch.from_numpy(x), (pt_cos, pt_sin), pt_float_mask, None)
    pt_tok_mask_np = pt_tok_mask.detach().cpu().numpy() == 0.0
    for q_idx in range(s):
      n_complete = (q_idx + 1) // 4
      if n_complete > 0:
        hf_sel = {m for m in range(n_complete) if np.all(pt_tok_mask_np[0, 0, q_idx, m * 4 : (m + 1) * 4])}
        jax_sel = {int(idx) for idx in np.array(jax_blocks[0, q_idx]) if idx >= 0}
        self.assertEqual(hf_sel, jax_sel)
    np.testing.assert_allclose(np.array(jax_out), pt_out.detach().cpu().numpy(), rtol=1e-5, atol=1e-5)

  def test_qsa_attention_packed_sequence_pytorch_parity(self):
    """Verifies packed-sequence parity and cross-document boundary isolation."""
    b, s, emb_dim = 1, 28, 64
    pt_rope, pt_attn, jax_attn = self._make_pt_and_jax(seed=1)

    # Document A = 1 on [0..9], Document B = 2 on [10..25], Padding = 0 on [26..27]
    seg_ids = np.zeros((b, s), dtype=np.int32)
    seg_ids[0, :10], seg_ids[0, 10:26] = 1, 2

    x = np.random.default_rng(123).standard_normal((b, s, emb_dim)).astype(np.float32)
    pos = np.tile(np.arange(s), (b, 1))

    pt_cos, pt_sin = pt_rope(torch.from_numpy(x), torch.from_numpy(pos))
    same = (seg_ids[:, :, None] == seg_ids[:, None, :]) & (seg_ids[:, :, None] != 0)
    pt_valid = same & np.tril(np.ones((s, s), dtype=bool))[None, :, :]
    pt_float_mask = torch.where(
        torch.from_numpy(pt_valid[:, None, :, :]), torch.zeros((), dtype=torch.float32), torch.finfo(torch.float32).min
    )

    pt_out, _ = pt_attn(torch.from_numpy(x), (pt_cos, pt_sin), attention_mask=pt_float_mask)
    jax_out, jax_blocks = jax_attn(
        jnp.array(x), position_ids=jnp.array(pos), segment_ids=jnp.array(seg_ids), return_selected_blocks=True
    )
    np.testing.assert_allclose(np.array(jax_out)[:, :26], pt_out[:, :26].detach().cpu().numpy(), rtol=1e-5, atol=1e-5)

    # Query t=20 in Document B: blocks [10..13] (0), [14..17] (1), tail [18..20]
    tok_mask, _ = jax_attn.indexer(
        jnp.array(x), (jnp.array(pt_cos.numpy()), jnp.array(pt_sin.numpy())), segment_ids=jnp.array(seg_ids)
    )
    tok_mask_np = np.array(tok_mask)[0, 0, 20]
    self.assertFalse(np.any(tok_mask_np[:10]))  # Document A tokens NEVER visible
    self.assertTrue(np.all(tok_mask_np[10:21]))  # Blocks 0, 1 + tail tokens 18..20 visible
    self.assertFalse(np.any(tok_mask_np[21:]))  # Future tokens not visible
    self.assertEqual(set(np.array(jax_blocks)[0, 20].tolist()), {0, 1, -1})


if __name__ == "__main__":
  absltest.main()

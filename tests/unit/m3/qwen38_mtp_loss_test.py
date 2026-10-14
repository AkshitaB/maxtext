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

"""Unit tests for Qwen3.8-Flash-Next MTP draft layer and integrated loss module.

Reference: SGLang qwen4_exp_mtp.py L78-85 & L104-116 and
transformers.models.qwen4_exp.modeling_qwen4_exp.
"""

# pylint: disable=no-name-in-module,missing-class-docstring
import types
from typing import Any
from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers.models.qwen4_exp import configuration_qwen4_exp
from transformers.models.qwen4_exp import modeling_qwen4_exp

from maxtext.m3.models.qwen38_flash_next import loss
from maxtext.m3.models.qwen38_flash_next import modeling_qwen38_flash_next
from maxtext.m3.models.qwen38_flash_next import mtp


def _create_mini_config(
    num_layers: int = 4, mtp_num_layers: int = 1, scan_layers: bool = False, dtype=jnp.float32, **kwargs: Any
) -> types.SimpleNamespace:
  """Creates a mini-twin config SimpleNamespace for fast MTP and loss unit testing."""
  base = {
      "emb_dim": 64, "num_decoder_layers": num_layers, "vocab_size": 128, "eos_token_id": 127, "ple_eos_token_id": 127,
      "hc_count": 4, "hc_lowrank": 16, "scan_layers": scan_layers, "dtype": dtype, "weight_dtype": dtype,
      "normalization_layer_epsilon": 1e-6, "inhomogeneous_layer_cycle_interval": 4, "gdn_num_key_heads": 2,
      "gdn_num_value_heads": 4, "gdn_key_head_dim": 16, "gdn_value_head_dim": 16, "gdn_conv_kernel_dim": 4,
      "gdn_chunk_size": 16, "num_query_heads": 4, "num_kv_heads": 1, "head_dim": 16, "partial_rotary_factor": 0.25,
      "rope_max_timescale": 1e4, "mrope_section": (1, 1, 0), "index_n_heads": 2, "index_kv_heads": 1,
      "index_head_dim": 16, "index_select_budget": 32, "index_select_block_size": 4, "moe_mlp_dim": 32, "mlp_dim": 32,
      "num_experts": 4, "num_experts_per_tok": 2, "norm_topk_prob": True, "routed_scaling_factor": 1.0,
      "use_ple_layer": False, "heads_per_ngram": 2, "ple_seed": 0, "ple_kernel_size": 3, "split_ngram_parts": 2,
      "ple_hash_seed": 0, "ngram_vocab_size_base": 500, "mtp_num_layers": mtp_num_layers, "mtp_num_streams": 4,
      "mtp_loss_factor": 0.1, "gdn_output_gate_activation": "silu", "ple_layer_ids": [],
  }
  base.update(kwargs)
  return types.SimpleNamespace(**base)


def _copy_layer_and_mixer_weights(pt_layer, pt_mixer, jx_layer, jx_mixer):
  """Copies decoder layer and mixer parameters from PyTorch to JAX modules."""
  def to_j(t, tr=False):
    return jnp.array(t.detach().cpu().numpy().T if tr else t.detach().cpu().numpy())

  def cp(dst, src, *keys, tr=True, attr="kernel"):
    for k in keys:
      getattr(getattr(dst, k), attr).value = to_j(getattr(src, k).weight, tr)

  for jh, ph in ((jx_layer.attn_hyper_connection, pt_layer.attn_hyper_connection),
                 (jx_layer.mlp_hyper_connection, pt_layer.mlp_hyper_connection)):
    cp(jh, ph, "input_mix_weight_down", "input_mix_weight_up", "block_inject_weight")
    cp(jh, ph, "hc_norm", tr=False, attr="scale")
  cp(jx_layer.self_attn, pt_layer.self_attn, "q_proj", "k_proj", "v_proj", "o_proj")
  cp(jx_layer.self_attn, pt_layer.self_attn, "q_norm", "k_norm", tr=False, attr="scale")
  cp(jx_layer.self_attn.indexer, pt_layer.self_attn.indexer, "q_layernorm", "k_layernorm", tr=False, attr="scale")
  cp(jx_layer.self_attn.indexer, pt_layer.self_attn.indexer, "index_qk_proj")
  for d_l, w in ((jx_layer.mlp.gate.linear, pt_layer.mlp.gate.weight),
                 (jx_layer.mlp.shared_expert_gate, pt_layer.mlp.shared_expert_gate.weight)):
    d_l.kernel.value = to_j(w, True)
  for k in ("gate_up_proj", "down_proj"):
    getattr(jx_layer.mlp.experts, k).value = jnp.swapaxes(to_j(getattr(pt_layer.mlp.experts, k)), -1, -2)
  cp(jx_layer.mlp.shared_expert, pt_layer.mlp.shared_expert, "gate_proj", "up_proj", "down_proj")
  cp(jx_mixer, pt_mixer, "input_mix_weight_down", "input_mix_weight_up")
  cp(jx_mixer, pt_mixer, "hc_norm", tr=False, attr="scale")


class Qwen38MTPLossTest(absltest.TestCase):

  def test_mtp_block_shapes_and_parity(self):
    cfg = _create_mini_config()
    block = mtp.Qwen38MTPBlock(cfg, rngs=nnx.Rngs(params=0))
    self.assertEqual((block.pre_fc_norm_embedding.scale.shape, block.pre_fc_norm_hidden.scale.shape), ((64,), (256,)))
    self.assertEqual((block.fc_embedding.kernel.shape, block.fc_hidden.kernel.shape), ((64, 64), (64, 64)))
    b, s, c, d = 2, 8, 4, 64
    hc_hidden = jax.random.normal(jax.random.PRNGKey(1), (b, s, c * d))
    next_embeds = jax.random.normal(jax.random.PRNGKey(2), (b, s, d))
    out = block(hc_hidden, next_embeds)
    assert isinstance(out, jax.Array)
    self.assertEqual(out.shape, (b, s, d))
    e_proj = block.fc_embedding(block.pre_fc_norm_embedding(next_embeds))
    h_proj = block.fc_hidden(block.pre_fc_norm_hidden(hc_hidden).reshape(b, s, c, d))
    h_mtp_in = (h_proj + jnp.expand_dims(e_proj, axis=-2)).reshape(b, s, c * d)
    ref_out = block.hyper_connection_mixer.mix_input(block.layers_0(h_mtp_in, return_aux=False))
    np.testing.assert_allclose(out, ref_out, rtol=1e-5, atol=1e-5)

  def test_pytorch_weight_copy_parity(self):
    cfg, (b, s, c, dim) = _create_mini_config(), (2, 4, 4, 64)
    rp = {"rope_theta": float(cfg.rope_max_timescale), "partial_rotary_factor": float(cfg.partial_rotary_factor),
          "mrope_interleaved": True, "mrope_section": list(cfg.mrope_section)}
    hf_cfg = configuration_qwen4_exp.Qwen4ExpTextConfig(
        vocab_size=cfg.vocab_size, hidden_size=dim, num_hidden_layers=4, head_dim=cfg.head_dim,
        num_attention_heads=cfg.num_query_heads, num_key_value_heads=cfg.num_kv_heads, hc_count=c,
        shared_expert_intermediate_size=cfg.mlp_dim, moe_intermediate_size=cfg.moe_mlp_dim, attention_bias=False,
        num_experts=cfg.num_experts, num_experts_per_tok=cfg.num_experts_per_tok, hc_lowrank=cfg.hc_lowrank,
        norm_topk_prob=cfg.norm_topk_prob, output_gate_type=cfg.gdn_output_gate_activation, attention_dropout=0.0,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"], rope_parameters=rp,
        ple_layer_ids=[], linear_num_key_heads=cfg.gdn_num_key_heads, linear_num_value_heads=cfg.gdn_num_value_heads,
        linear_key_head_dim=cfg.gdn_key_head_dim, linear_value_head_dim=cfg.gdn_value_head_dim,
        linear_conv_kernel_dim=cfg.gdn_conv_kernel_dim, indexer_compress_ratio=cfg.index_select_block_size,
        indexer_n_heads=cfg.index_n_heads, indexer_kv_heads=cfg.index_kv_heads, indexer_head_dim=cfg.index_head_dim,
        indexer_budget=cfg.index_select_budget, rms_norm_eps=cfg.normalization_layer_epsilon, seed=cfg.ple_seed,
    )
    pt_norm_emb, pt_norm_hid = (modeling_qwen4_exp.Qwen4ExpTextRMSNorm(d, group_size=None, eps=1e-6) for d in (dim, c * dim))
    pt_fc_emb, pt_fc_hid = (torch.nn.Linear(dim, dim, bias=False) for _ in range(2))
    pt_layer = modeling_qwen4_exp.Qwen4ExpTextDecoderLayer(hf_cfg, layer_idx=3)
    pt_mixer = modeling_qwen4_exp.Qwen4ExpTextGatedResidual(hf_cfg, use_combine=False)

    jx_block = mtp.Qwen38MTPBlock(cfg, rngs=nnx.Rngs(params=0))
    jx_block.pre_fc_norm_embedding.scale.value = jnp.array(pt_norm_emb.weight.detach().cpu().numpy())
    jx_block.pre_fc_norm_hidden.scale.value = jnp.array(pt_norm_hid.weight.detach().cpu().numpy())
    jx_block.fc_embedding.kernel.value = jnp.array(pt_fc_emb.weight.detach().cpu().numpy().T)
    jx_block.fc_hidden.kernel.value = jnp.array(pt_fc_hid.weight.detach().cpu().numpy().T)
    _copy_layer_and_mixer_weights(pt_layer, pt_mixer, jx_block.layers_0, jx_block.hyper_connection_mixer)

    torch.manual_seed(42)
    pt_hc, pt_nxt = torch.randn(b, s, c * dim), torch.randn(b, s, dim)
    pt_fused = pt_fc_hid(pt_norm_hid(pt_hc).view(b, s, c, dim)) + pt_fc_emb(pt_norm_emb(pt_nxt)).unsqueeze(-2)
    pt_mtp_in = pt_fused.view(b, s, c * dim)
    cos, sin = modeling_qwen4_exp.Qwen4ExpTextRotaryEmbedding(hf_cfg)(pt_mtp_in, torch.arange(s).unsqueeze(0).expand(b, -1))
    mask = torch.tril(torch.ones((b, 1, s, s), dtype=torch.bool))
    pt_mask = torch.where(mask, torch.zeros((), dtype=torch.float32), torch.finfo(torch.float32).min)
    pt_out = pt_mixer(pt_layer(pt_mtp_in, position_embeddings=(cos, sin), attention_mask=pt_mask)).detach().cpu().numpy()

    jx_out = np.array(jx_block(jnp.array(pt_hc.numpy()), jnp.array(pt_nxt.numpy()),
                               positions=jnp.broadcast_to(jnp.arange(s, dtype=jnp.int32), (b, s))))
    np.testing.assert_allclose(jx_out, pt_out, rtol=1e-5, atol=1e-5)

  def test_checkpoint_pinned_shape_contract(self):
    cfg = _create_mini_config(emb_dim=2560, hc_count=4, hc_lowrank=320, head_dim=256, index_head_dim=128,
                              num_query_heads=16, num_kv_heads=2, index_n_heads=4, index_kv_heads=2)
    shapes = nnx.eval_shape(lambda: mtp.Qwen38MTPBlock(cfg, rngs=nnx.Rngs(params=0)))
    self.assertEqual((shapes.fc_embedding.kernel.shape, shapes.fc_hidden.kernel.shape), ((2560, 2560), (2560, 2560)))
    self.assertEqual((shapes.pre_fc_norm_embedding.scale.shape, shapes.pre_fc_norm_hidden.scale.shape), ((2560,), (10240,)))
    self.assertEqual((shapes.hyper_connection_mixer.hc_norm.scale.shape,
                      shapes.hyper_connection_mixer.input_mix_weight_down.kernel.shape,
                      shapes.hyper_connection_mixer.input_mix_weight_up.kernel.shape),
                     ((10240,), (10240, 320), (320, 10240)))
    self.assertEqual((shapes.layers_0.self_attn.q_norm.scale.shape, shapes.layers_0.self_attn.k_norm.scale.shape),
                     ((256,), (256,)))
    self.assertEqual((shapes.layers_0.self_attn.indexer.q_layernorm.scale.shape,
                      shapes.layers_0.self_attn.indexer.k_layernorm.scale.shape), ((128,), (128,)))

  def test_phase_gating(self):
    cfg = _create_mini_config()
    model = modeling_qwen38_flash_next.Qwen38FlashNextModel(cfg, rngs=nnx.Rngs(params=0))
    self.assertIsNotNone(model.mtp)
    tokens = jax.random.randint(jax.random.PRNGKey(3), (2, 8), 0, 128)
    self.assertEqual(model(tokens, return_mtp=False).shape, (2, 8, 128))
    logits, mtp_logits = model(tokens, return_mtp=True)
    self.assertEqual((logits.shape, mtp_logits.shape), ((2, 8, 128), (2, 8, 128)))
    logits_aux, aux_dict = model(tokens, return_mtp=True, return_aux=True)
    self.assertEqual(logits_aux.shape, (2, 8, 128))
    self.assertTrue({"layer_aux", "mtp_logits", "mtp_aux"}.issubset(aux_dict))
    self.assertIn("router_probs", aux_dict["mtp_aux"])

  def test_cross_entropy_loss(self):
    logits, targets = jnp.zeros((2, 4, 8), dtype=jnp.float32), jnp.zeros((2, 4), dtype=jnp.int32)
    np.testing.assert_allclose(loss.cross_entropy_loss(logits, targets), np.log(8.0), rtol=1e-5)
    mask = jnp.array([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=jnp.float32)
    np.testing.assert_allclose(loss.cross_entropy_loss(logits, targets, mask=mask), np.log(8.0), rtol=1e-5)
    with self.assertRaises(ValueError):
      loss.cross_entropy_loss(jnp.array([]), jnp.array([]))

  def test_router_load_balancing_loss(self):
    b, s, e = 2, 4, 4
    r_probs, r_idx = jnp.full((b, s, e), 1.0 / e, dtype=jnp.float32), jnp.tile(jnp.array([0, 1], dtype=jnp.int32), (b, s, 1))
    aux = [{"router_probs": r_probs, "router_indices": r_idx}]
    lb_loss = loss.router_load_balancing_loss(aux, num_experts=e)
    self.assertTrue(jnp.isfinite(lb_loss) and lb_loss > 0.0)
    for bad_aux in (None, [], [{"bad": 1}]):
      with self.assertRaises(ValueError):
        loss.router_load_balancing_loss(bad_aux, 4)

  def test_compute_total_loss_and_gradients(self):
    cfg = _create_mini_config()
    model = modeling_qwen38_flash_next.Qwen38FlashNextModel(cfg, rngs=nnx.Rngs(params=0))
    inp, tgt = (jax.random.randint(jax.random.PRNGKey(k), (2, 16), 0, 128) for k in (4, 5))
    (total_l, metrics), grads = nnx.value_and_grad(
        lambda m: loss.compute_total_loss(m, inp, tgt, router_aux_coeff=0.001, mtp_loss_coeff=0.1, use_mtp=True),
        has_aux=True,
    )(model)
    self.assertTrue(jnp.isfinite(total_l))
    self.assertTrue({"total_loss", "primary_loss", "router_loss", "mtp_loss"}.issubset(metrics))
    self.assertGreater(jnp.sum(jnp.abs(grads.lm_head.kernel[...])), 0.0)
    self.assertGreater(jnp.sum(jnp.abs(grads.mtp.fc_embedding.kernel[...])), 0.0)

  def test_compute_total_loss_single_token(self):
    cfg = _create_mini_config(num_layers=4)
    model = modeling_qwen38_flash_next.Qwen38FlashNextModel(cfg, rngs=nnx.Rngs(params=0))
    toks = jnp.array([[5]], dtype=jnp.int32)
    total_l, mets = loss.compute_total_loss(model, toks, toks, use_mtp=True)
    self.assertTrue(jnp.isfinite(total_l))
    self.assertEqual(float(mets["mtp_loss"]), 0.0)

  def test_compute_total_loss_scanned(self):
    cfg = _create_mini_config(num_layers=4, scan_layers=True)
    model = modeling_qwen38_flash_next.Qwen38FlashNextModel(cfg, rngs=nnx.Rngs(params=0))
    toks = jax.random.randint(jax.random.PRNGKey(6), (2, 8), 0, 128)
    total_l, mets = loss.compute_total_loss(model, toks, toks, router_aux_coeff=0.001, use_mtp=True)
    self.assertTrue(jnp.isfinite(total_l))
    self.assertIn("router_loss", mets)
    self.assertGreater(float(mets["router_loss"]), 0.0)

  def test_packed_document_boundary_mtp_loss(self):
    cfg = _create_mini_config()
    model = modeling_qwen38_flash_next.Qwen38FlashNextModel(cfg, rngs=nnx.Rngs(params=0))
    toks, seg_ids = jnp.array([[10, 20, 30, 40]], dtype=jnp.int32), jnp.array([[1, 1, 2, 2]], dtype=jnp.int32)
    toks_modified = jnp.array([[10, 20, 99, 40]], dtype=jnp.int32)
    _, m1 = loss.compute_total_loss(model, toks, toks, segment_ids=seg_ids, use_mtp=True)
    _, m2 = loss.compute_total_loss(model, toks, toks_modified, segment_ids=seg_ids, use_mtp=True)
    np.testing.assert_allclose(float(m1["mtp_loss"]), float(m2["mtp_loss"]), rtol=1e-5)
    _, u1 = loss.compute_total_loss(model, toks, toks, use_mtp=True)
    _, u2 = loss.compute_total_loss(model, toks, toks_modified, use_mtp=True)
    self.assertNotEqual(float(u1["mtp_loss"]), float(u2["mtp_loss"]))


if __name__ == "__main__":
  absltest.main()

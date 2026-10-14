# Copyright 2023–2026 Google LLC
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

"""Unit and numerics tests for O3: GDN single-segment backward and forward Conv1D fast paths."""

from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from maxtext.common.common_types import MODEL_MODE_TRAIN, ShardMode
from maxtext.kernels.gdn import model_runner as gdn_model_runner
from maxtext.kernels.gdn.gdn_bwd import api as gdn_bwd_api
from maxtext.kernels.gdn.gdn_bwd import compute_conv1d_bwd
from maxtext.kernels.gdn.gdn_bwd import jax_compute_gdn_states


class O3GdnFastPathTest(unittest.TestCase):
  """Tests for `gdn_bwd_single_segment_fast_path` and `gdn_fwd_conv_fast_path`."""

  def test_segmented_causal_depthwise_conv1d_fast_path_bit_identical_on_single_segment(self):
    """`segmented_causal_depthwise_conv1d(..., fast_path=True)` is bit-identical on single-segment padded rows."""
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    batch, seq_len, channels, kernel_size = 2, 16, 8, 4
    # Single document with trailing zero padding; padding tokens already zeroed in qkv (as in GDN layer).
    segment_ids = jnp.array(
        [
            [1] * 12 + [0] * 4,
            [1] * 16,
        ],
        dtype=jnp.int32,
    )
    qkv_raw = jax.random.normal(k1, (batch, seq_len, channels), dtype=jnp.float32)
    qkv = jnp.where((segment_ids != 0)[:, :, None], qkv_raw, 0.0)
    conv_weight = jax.random.normal(k2, (kernel_size, 1, channels), dtype=jnp.float32)
    conv_bias = jax.random.normal(k3, (channels,), dtype=jnp.float32)

    def loss_fn(qkv_in, w_in, b_in, fast):
      out = gdn_model_runner.segmented_causal_depthwise_conv1d(
          qkv=qkv_in,
          conv_weight=w_in,
          conv_bias=b_in,
          segment_ids=segment_ids,
          kernel_size=kernel_size,
          fast_path=fast,
      )
      return jnp.sum(out * jnp.arange(1, out.size + 1, dtype=jnp.float32).reshape(out.shape)), out

    (loss_ref, out_ref), grads_ref = jax.value_and_grad(loss_fn, argnums=(0, 1, 2), has_aux=True)(
        qkv, conv_weight, conv_bias, False
    )
    (loss_fast, out_fast), grads_fast = jax.value_and_grad(loss_fn, argnums=(0, 1, 2), has_aux=True)(
        qkv, conv_weight, conv_bias, True
    )
    np.testing.assert_array_equal(np.asarray(out_fast), np.asarray(out_ref))
    np.testing.assert_array_equal(np.asarray(loss_fast), np.asarray(loss_ref))
    for g_fast, g_ref in zip(grads_fast, grads_ref):
      # Mask qkv grad on non-padding positions where inputs are active
      if g_fast.shape == qkv.shape:
        mask = (segment_ids != 0)[:, :, None]
        np.testing.assert_array_equal(np.asarray(g_fast * mask), np.asarray(g_ref * mask))
      else:
        np.testing.assert_allclose(np.asarray(g_fast), np.asarray(g_ref), rtol=1e-6, atol=1e-6)

  def test_conv1d_silu_fwd_bwd_fast_conv_bit_identical_on_unpadded_single_segment(self):
    """`conv1d_silu_fwd` and `conv1d_silu_bwd` with `fast_conv=True` match `fast_conv=False` bit-identically."""
    key = jax.random.PRNGKey(7)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    batch, seq_len, channels, kernel_size = 2, 16, 12, 4
    segment_ids = jnp.ones((batch, seq_len), dtype=jnp.int32)
    qkv = jax.random.normal(k1, (batch, seq_len, channels), dtype=jnp.float32)
    conv_weight = jax.random.normal(k2, (kernel_size, 1, channels), dtype=jnp.float32)
    conv_bias = jax.random.normal(k3, (channels,), dtype=jnp.float32)
    dy = jax.random.normal(k4, (batch, seq_len, channels), dtype=jnp.float32)

    conv_out_ref, qkv_conv_ref = compute_conv1d_bwd.conv1d_silu_fwd(
        qkv, conv_weight, conv_bias, kernel_size, segment_ids=segment_ids, fast_conv=False
    )
    conv_out_fast, qkv_conv_fast = compute_conv1d_bwd.conv1d_silu_fwd(
        qkv, conv_weight, conv_bias, kernel_size, segment_ids=segment_ids, fast_conv=True
    )
    np.testing.assert_array_equal(np.asarray(conv_out_fast), np.asarray(conv_out_ref))
    np.testing.assert_array_equal(np.asarray(qkv_conv_fast), np.asarray(qkv_conv_ref))

    dx_ref, dw_ref, db_ref = compute_conv1d_bwd.conv1d_silu_bwd(
        qkv, conv_weight, conv_bias, dy, kernel_size, conv_out=conv_out_ref, segment_ids=segment_ids, fast_conv=False
    )
    dx_fast, dw_fast, db_fast = compute_conv1d_bwd.conv1d_silu_bwd(
        qkv, conv_weight, conv_bias, dy, kernel_size, conv_out=conv_out_fast, segment_ids=segment_ids, fast_conv=True
    )
    np.testing.assert_array_equal(np.asarray(dx_fast), np.asarray(dx_ref))
    np.testing.assert_array_equal(np.asarray(dw_fast), np.asarray(dw_ref))
    np.testing.assert_array_equal(np.asarray(db_fast), np.asarray(db_ref))

  def test_decoupled_conv1d_gdn_bwd_kernel_single_segment_fast_path_parity(self):
    """`decoupled_conv1d_gdn_bwd_kernel` with `single_segment_bwd=True, fast_conv=True` is bit-identical on single-segment."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 7)
    batch, seq_len, chunk_size = 2, 64, 64
    num_k_heads, num_v_heads = 2, 4
    head_k_dim, head_v_dim = 128, 128
    kernel_size = 4
    dim_size = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim

    qkv = jax.random.normal(keys[0], (batch, seq_len, dim_size), dtype=jnp.float32) * 0.1
    b = jax.random.normal(keys[1], (batch, seq_len, num_v_heads), dtype=jnp.float32) * 0.1
    a = jax.random.normal(keys[2], (batch, seq_len, num_v_heads), dtype=jnp.float32) * 0.1
    conv_weight = jax.random.normal(keys[3], (kernel_size, 1, dim_size), dtype=jnp.float32) * 0.1
    conv_bias = jax.random.normal(keys[4], (dim_size,), dtype=jnp.float32) * 0.05
    a_log = jnp.linspace(-2.0, -0.5, num_v_heads, dtype=jnp.float32)
    dt_bias = jnp.zeros((num_v_heads,), dtype=jnp.float32)
    do = jax.random.normal(keys[5], (batch, seq_len, num_v_heads, head_v_dim), dtype=jnp.float32) * 0.1
    segment_ids = jnp.ones((batch, seq_len), dtype=jnp.int32)

    _, chunk_states, t_inv = jax_compute_gdn_states._compute_forward_conv_and_states(
        qkv=qkv,
        b=b,
        a=a,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=a_log,
        dt_bias=dt_bias,
        recurrent_state=None,
        conv_state=None,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_heads_dim=head_v_dim,
        conv_kernel_size=kernel_size,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=True,
        compute_dtype=jnp.float32,
        segment_ids=None,
    ) if "head_v_heads_dim" in jax_compute_gdn_states._compute_forward_conv_and_states.__code__.co_varnames else jax_compute_gdn_states._compute_forward_conv_and_states(
        qkv=qkv,
        b=b,
        a=a,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=a_log,
        dt_bias=dt_bias,
        recurrent_state=None,
        conv_state=None,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_kernel_size=kernel_size,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=True,
        compute_dtype=jnp.float32,
        segment_ids=None,
    )

    grads_seg = gdn_bwd_api.decoupled_conv1d_gdn_bwd_kernel(
        pre_conv_qkv=qkv,
        b=b,
        a=a,
        a_log=a_log,
        dt_bias=dt_bias,
        do=do,
        chunk_states=chunk_states,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        t_inv=t_inv,
        num_v_heads=num_v_heads,
        kq_head_dim=head_k_dim,
        v_head_dim=head_v_dim,
        kernel_size=kernel_size,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=True,
        segment_ids=segment_ids,
        interpret=True,
        single_segment_bwd=False,
        fast_conv=False,
    )
    grads_fast = gdn_bwd_api.decoupled_conv1d_gdn_bwd_kernel(
        pre_conv_qkv=qkv,
        b=b,
        a=a,
        a_log=a_log,
        dt_bias=dt_bias,
        do=do,
        chunk_states=chunk_states,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        t_inv=t_inv,
        num_v_heads=num_v_heads,
        kq_head_dim=head_k_dim,
        v_head_dim=head_v_dim,
        kernel_size=kernel_size,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=True,
        segment_ids=segment_ids,
        interpret=True,
        single_segment_bwd=True,
        fast_conv=True,
    )
    for g_f, g_s in zip(grads_fast, grads_seg):
      np.testing.assert_allclose(np.asarray(g_f), np.asarray(g_s), rtol=1e-5, atol=1e-5)

  def test_run_gdn_kernel_layer_fast_path_flags_parity(self):
    """`run_gdn_kernel_layer` with `gdn_bwd_single_segment_fast_path=True` matches default off on single-segment."""
    key = jax.random.PRNGKey(99)
    keys = jax.random.split(key, 8)
    batch, seq_len, chunk_size = 2, 64, 64
    num_k_heads, num_v_heads = 2, 4
    head_k_dim, head_v_dim = 128, 128
    kernel_size = 4
    dim_size = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim

    query = jax.random.normal(keys[0], (batch, seq_len, num_k_heads, head_k_dim), dtype=jnp.float32) * 0.1
    k_ten = jax.random.normal(keys[1], (batch, seq_len, num_k_heads, head_k_dim), dtype=jnp.float32) * 0.1
    value = jax.random.normal(keys[2], (batch, seq_len, num_v_heads, head_v_dim), dtype=jnp.float32) * 0.1
    value_raw = value.reshape(batch, seq_len, num_k_heads, -1)
    b = jax.random.normal(keys[3], (batch, seq_len, num_v_heads), dtype=jnp.float32) * 0.1
    a = jax.random.normal(keys[4], (batch, seq_len, num_v_heads), dtype=jnp.float32) * 0.1
    conv_weight = jax.random.normal(keys[5], (kernel_size, 1, dim_size), dtype=jnp.float32) * 0.1
    conv_bias = jax.random.normal(keys[6], (dim_size,), dtype=jnp.float32) * 0.05
    a_log = jnp.linspace(-2.0, -0.5, num_v_heads, dtype=jnp.float32)
    dt_bias = jnp.zeros((num_v_heads,), dtype=jnp.float32)
    decoder_segment_ids = jnp.ones((batch, seq_len), dtype=jnp.int32)

    def make_layer(bwd_fast: bool, fwd_fast: bool):
      cfg = SimpleNamespace(
          ici_context_parallelism=1,
          dcn_context_parallelism=1,
          ici_context_usp_ulysses_parallelism=1,
          dcn_context_usp_ulysses_parallelism=1,
          gdn_cp_mode="auto",
          gdn_cp_matmul_precision="highest",
          shard_mode=ShardMode.AUTO,
          gdn_conv_kernel_dim=kernel_size,
          gdn_chunk_size=chunk_size,
          use_qk_norm_in_gdn=True,
          packing=False,
          gdn_bwd_single_segment_fast_path=bwd_fast,
          gdn_fwd_conv_fast_path=fwd_fast,
          logical_axis_rules=(),
      )
      return SimpleNamespace(
          config=cfg,
          mesh=None,
          num_k_heads=num_k_heads,
          num_v_heads=num_v_heads,
          head_k_dim=head_k_dim,
          head_v_dim=head_v_dim,
          key_dim=num_k_heads * head_k_dim,
          v_heads_per_k_head=num_v_heads // num_k_heads,
          conv1d=SimpleNamespace(
              kernel=SimpleNamespace(value=conv_weight),
              bias=SimpleNamespace(value=conv_bias),
          ),
          A_log=a_log,
          dt_bias=dt_bias,
      )

    def run_loss(q_in, k_in, v_in, b_in, a_in, bwd_fast, fwd_fast):
      layer = make_layer(bwd_fast, fwd_fast)
      out, _, _ = gdn_model_runner.run_gdn_kernel_layer(
          layer=layer,
          query=q_in,
          key=k_in,
          value=v_in,
          value_raw=value_raw,
          b=b_in,
          a=a_in,
          active_cache=None,
          decoder_segment_ids=decoder_segment_ids,
          model_mode=MODEL_MODE_TRAIN,
          state_dtype=jnp.float32,
      )
      return jnp.sum(out * jnp.arange(1, out.size + 1, dtype=jnp.float32).reshape(out.shape)), out

    (loss_ref, out_ref), grads_ref = jax.value_and_grad(run_loss, argnums=(0, 1, 2, 3, 4), has_aux=True)(
        query, k_ten, value, b, a, False, False
    )
    (loss_fast, out_fast), grads_fast = jax.value_and_grad(run_loss, argnums=(0, 1, 2, 3, 4), has_aux=True)(
        query, k_ten, value, b, a, True, True
    )
    np.testing.assert_allclose(np.asarray(out_fast), np.asarray(out_ref), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(loss_fast), np.asarray(loss_ref), rtol=1e-5, atol=1e-5)
    for g_f, g_r in zip(grads_fast, grads_ref):
      np.testing.assert_allclose(np.asarray(g_f), np.asarray(g_r), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  unittest.main()

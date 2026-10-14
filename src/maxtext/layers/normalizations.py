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

"""Normalization Layers."""

from typing import Any

from flax import nnx
import jax
from jax import lax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from maxtext.common.common_types import Array, DType, ShardMode, is_fp8_dtype
from maxtext.layers import nnx_wrappers
from maxtext.layers.initializers import Initializer, variable_to_logically_partitioned
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils.sharding import truncate_out_sharding


def _align_scale_with_normalized_axis(scale: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  """Reshards a 1-D norm scale to match the sharding of the axis it normalizes.

  The scale is stored with the `kernel_axes` layout, which is picked for the
  usual case of normalizing over the embedding axis (`norm` -> `tensor`). A
  per-head QK norm instead normalizes over `head_dim`, which is unsharded while
  the neighbouring `heads` axis is the one on `tensor`. Multiplying those
  directly under `ShardMode.EXPLICIT` would place `tensor` on two axes of the
  result, which JAX rejects, so align the scale with the activation instead.

  This is a no-op whenever the two already agree, which includes every
  auto-sharding-equivalent layout for the ordinary layer norms.
  """
  # Read through `.partitions`: a scale carrying a reduced/unreduced tag (the deferred
  # data-parallel all-reduce under gradient accumulation) rejects direct indexing. The tags
  # are carried over to the new spec so the scale keeps its place on the deferred path.
  activation_axis = jax.typeof(y).sharding.spec.partitions[-1]
  scale_spec = jax.typeof(scale).sharding.spec
  if scale_spec.partitions[-1] == activation_axis:
    return scale
  return jax.sharding.reshard(
      scale,
      jax.sharding.PartitionSpec(activation_axis, unreduced=scale_spec.unreduced, reduced=scale_spec.reduced),
  )


def expand_replicated_scale_for_scan(
    scale: jnp.ndarray,
    mesh: jax.sharding.Mesh,
    batch_axes: tuple[str, ...],
    seq_axes: tuple[str, ...],
) -> jnp.ndarray:
  """Expands a replicated scanned RMSNorm scale along `(batch_axes, seq_axes)` to defer its backward all-reduce.

  In the forward pass, appends two size-1 local dimensions `(..., K, 1, 1)` sharded across
  `(batch_axes, seq_axes)` so the global shape is `(..., K, batch_shards, seq_shards)` with zero
  collectives. Inside `RMSNorm.__call__`, the sliced per-layer scale has shape `(K, batch_shards, seq_shards)`
  and contracts against the local `(batch, seq)` shard of `y` without cross-device reduction inside the
  scan loop. In the backward pass after the scan loop completes, `_expand_bwd` reduces the full stacked
  gradient across `batch_axes + seq_axes` in float32 in a single all-reduce.
  """
  active_axes = tuple(batch_axes) + tuple(seq_axes)
  if not active_axes:
    return scale
  b_spec = tuple(batch_axes) if len(batch_axes) > 1 else (batch_axes[0] if batch_axes else None)
  s_spec = tuple(seq_axes) if len(seq_axes) > 1 else (seq_axes[0] if seq_axes else None)
  in_spec = jax.sharding.PartitionSpec(*(None for _ in range(scale.ndim)))
  out_spec = jax.sharding.PartitionSpec(*((None,) * scale.ndim + (b_spec, s_spec)))
  s_dtype = scale.dtype

  @jax.custom_vjp
  def _expand(s):
    @jax.shard_map(mesh=mesh, in_specs=in_spec, out_specs=out_spec, check_vma=False)
    def _fwd(s_local):
      return jnp.expand_dims(jnp.asarray(s_local, jnp.float32), axis=(-2, -1))

    return _fwd(s)

  def _expand_fwd(s):
    return _expand(s), ()

  def _expand_bwd(_, g):
    @jax.shard_map(mesh=mesh, in_specs=out_spec, out_specs=in_spec, check_vma=False)
    def _bwd(g_local):
      g_sum = jax.lax.psum(jnp.squeeze(g_local, axis=(-2, -1)), active_axes)
      return jnp.asarray(g_sum, s_dtype)

    return (_bwd(g),)

  _expand.defvjp(_expand_fwd, _expand_bwd)
  return _expand(scale)


class RMSNorm(nnx.Module):
  """RMS normalization."""

  def __init__(
      self,
      num_features: int,
      epsilon: float = 1e-6,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      shard_mode: ShardMode = ShardMode.AUTO,
      kernel_axes: tuple[None | str, ...] = (),
      scale_init: Initializer = jax.nn.initializers.ones,
      parameter_memory_host_offload: bool = False,
      scale_offset: float = 0.0,
      with_scale: bool = True,
      *,
      rngs: nnx.Rngs,
  ):
    self.num_features = num_features
    self.epsilon = epsilon
    self.dtype = dtype
    self.weight_dtype = dtype if is_fp8_dtype(weight_dtype) else weight_dtype
    self.shard_mode = shard_mode
    self.kernel_axes = kernel_axes
    self.scale_init = scale_init
    self.parameter_memory_host_offload = parameter_memory_host_offload
    self.scale_offset = scale_offset
    self.with_scale = with_scale
    if self.with_scale:
      self.scale = nnx.Param(
          scale_init(rngs.params(), (num_features,), self.weight_dtype),
          out_sharding=kernel_axes,
      )
    else:
      self.scale = None

  def __call__(self, x: jnp.ndarray, out_sharding: NamedSharding | None = None) -> jnp.ndarray:
    """Applies layer normalization on the input."""
    x = jnp.asarray(x, jnp.float32)
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), jnp.float32)

    # out_sharding must be None in auto shard mode
    if self.shard_mode != ShardMode.EXPLICIT:
      out_sharding = None

    if out_sharding is not None:
      out_sharding = truncate_out_sharding(out_sharding, y.ndim)

    if not self.with_scale:
      y_out = jnp.asarray(y, self.dtype)
      if out_sharding is not None:
        y_out = jax.lax.with_sharding_constraint(y_out, out_sharding)
      return y_out

    scale = self.scale.get_value()
    # Move scale to device if parameter offloading is enabled
    if self.parameter_memory_host_offload:
      max_logging.log("normalizations.py: Moving scale parameter to device")
      scale = jax.device_put(scale, max_utils.device_space())

    if scale.ndim == 3 and y.ndim >= 3:
      _, b_sh, s_sh = scale.shape
      orig_shape = y.shape
      if orig_shape[0] % b_sh == 0 and orig_shape[1] % s_sh == 0:
        full_shape = (b_sh, orig_shape[0] // b_sh, s_sh, orig_shape[1] // s_sh) + orig_shape[2:]
        scale_fp32 = jnp.asarray(scale, jnp.float32) + jnp.float32(self.scale_offset)
        if self.shard_mode == ShardMode.EXPLICIT:
          y_sharding = jax.typeof(y).sharding
          s_partitions = jax.typeof(scale_fp32).sharding.spec.partitions
          full_sharding = NamedSharding(
              y_sharding.mesh,
              jax.sharding.PartitionSpec(s_partitions[1], None, s_partitions[2], None, *y_sharding.spec.partitions[2:]),
          )
          y_r = jnp.reshape(y, full_shape, out_sharding=full_sharding)
          out_r = jnp.einsum(
              "bmsl...k,kbs->bmsl...k",
              y_r,
              scale_fp32,
              precision=lax.Precision.HIGHEST,
              out_sharding=full_sharding,
          )
          out_fp32 = jnp.reshape(out_r, orig_shape, out_sharding=out_sharding if out_sharding is not None else y_sharding)
        else:
          y_r = jnp.reshape(y, full_shape)
          out_r = jnp.einsum(
              "bmsl...k,kbs->bmsl...k",
              y_r,
              scale_fp32,
              precision=lax.Precision.HIGHEST,
          )
          out_fp32 = jnp.reshape(out_r, orig_shape)
        return jnp.asarray(out_fp32, self.dtype)

    scale_fp32 = jnp.asarray(scale, jnp.float32)
    effective_scale = scale_fp32 + jnp.float32(self.scale_offset)
    if self.shard_mode == ShardMode.EXPLICIT:
      effective_scale = _align_scale_with_normalized_axis(effective_scale, y)

    out_fp32 = jnp.einsum(
        "...k,k->...k", y, effective_scale, precision=lax.Precision.HIGHEST, out_sharding=out_sharding
    )
    return jnp.asarray(out_fp32, self.dtype)


class GlobalRMSNorm(RMSNorm):
  """
  Applies RMSNorm over the last two dimensions (Heads * HeadDim).
  Used for Olmo3 which normalizes across all heads combined.
  """

  def __call__(self, x: jnp.ndarray, out_sharding: NamedSharding | None = None) -> jnp.ndarray:
    # x shape: [..., Heads, HeadDim]
    input_shape = x.shape

    # Flatten the last two dimensions: [..., Heads * HeadDim]
    # We use -2 and -1 to ensure we capture the last two dims regardless of rank
    flattened_shape = input_shape[:-2] + (input_shape[-2] * input_shape[-1],)
    x_flat = x.reshape(flattened_shape)

    # Apply standard RMSNorm (which normalizes over the last axis)
    y_flat = super().__call__(x_flat, out_sharding)

    # Reshape back to [..., Heads, HeadDim]
    return y_flat.reshape(input_shape)


def Qwen3NextRMSNorm(
    num_features: int,
    epsilon: float = 1e-6,
    dtype: DType = None,
    weight_dtype: DType = None,
    shard_mode=None,
    kernel_axes=None,
    parameter_memory_host_offload=None,
    *,
    rngs: nnx.Rngs,
):
  """Used for input and post attention layernorms in Qwen3NextDecoderLayer.

  This normalization layer is specific to Qwen3-Next. Key characteristics:
  1. The learnable scale parameter `scale` is initialized to ZEROS.
  2. The scale is applied as `(1.0 + self.scale)`, making the initial scale effectively 1.0.
     This matches the PyTorch implementation of Qwen3NextRMSNorm.
  """
  return nnx.data(
      RMSNorm(
          num_features=num_features,
          epsilon=epsilon,
          dtype=dtype,
          weight_dtype=weight_dtype,
          shard_mode=shard_mode if shard_mode is not None else ShardMode.AUTO,
          kernel_axes=kernel_axes if kernel_axes is not None else (),
          scale_init=jax.nn.initializers.zeros,
          scale_offset=1.0,
          parameter_memory_host_offload=bool(parameter_memory_host_offload),
          rngs=rngs,
      )
  )


class Qwen3NextRMSNormGated(nnx.Module):
  """
  This applies RMS Normalization and then a gated activation function (SiLU).
  This is used within the Qwen3NextGatedDeltaNet.

  The normalization is performed by an internal `RMSNorm` instance (`self.rms_norm`),
  which has its own learnable `scale` parameter, initialized to ONES.

  Attributes:
    num_features: The number of features in the input.
    eps: A small epsilon value to prevent division by zero in RMSNorm.
    dtype: The datatype of the computation.
    weight_dtype: The datatype of the internal RMSNorm scale.
  """

  def __init__(
      self,
      num_features: int,
      epsilon: float,
      dtype: DType,
      weight_dtype: DType,
      shard_mode: ShardMode = ShardMode.AUTO,
      *,
      rngs: nnx.Rngs,
  ):
    self.num_features = num_features
    self.epsilon = epsilon
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.rms_norm = nnx.data(
        RMSNorm(
            num_features=num_features,
            epsilon=self.epsilon,
            dtype=jnp.float32,
            weight_dtype=weight_dtype,
            shard_mode=shard_mode,
            scale_init=nnx.initializers.ones,
            rngs=rngs,
        )
    )

  def __call__(
      self,
      hidden_states: Array,
      gate: Array,
      out_sharding: NamedSharding | None = None,
  ) -> Array:
    """Applies RMSNorm and then a SiLU gate.

    Args:
      hidden_states: The input array to be normalized (o). Shape: (..., F)
      gate: The gating array for the activation (z). Shape: (..., F) where F is
        num_features.
      out_sharding: Optional layout for the normalized states, honoured only
        under `ShardMode.EXPLICIT`.

    Returns:
      The normalized and gated output array. Shape: (..., F)
    """
    normalized_states = self.rms_norm(hidden_states, out_sharding=out_sharding)

    # Gated Activation using SiLU (Sigmoid-weighted Linear Unit)
    gated_states = normalized_states * jax.nn.silu(gate.astype(jnp.float32))

    return gated_states.astype(self.dtype)


def rms_norm(
    num_features: int,
    epsilon: float = 1e-6,
    dtype: Any = jnp.float32,
    weight_dtype: Any = jnp.float32,
    shard_mode: ShardMode = ShardMode.AUTO,
    kernel_axes: tuple[None | str, ...] = (),
    scale_init: Initializer = jax.nn.initializers.ones,
    name: None | str = None,
    parameter_memory_host_offload: bool = False,
    with_scale: bool = True,
):
  """Creates a RMSNorm module."""
  module = nnx_wrappers.to_linen(
      RMSNorm,
      num_features=num_features,
      epsilon=epsilon,
      dtype=dtype,
      weight_dtype=weight_dtype,
      shard_mode=shard_mode,
      kernel_axes=kernel_axes,
      scale_init=scale_init,
      parameter_memory_host_offload=parameter_memory_host_offload,
      with_scale=with_scale,
      name=name,
      metadata_fn=variable_to_logically_partitioned,
  )
  return module


def l2norm(x: Array, dim: int = -1, eps: float = 1e-6) -> Array:
  """L2 normalization function. Normalizes a vector to have a length of 1.

  Args:
    x: Input array.
    dim: The axis or axes along which to normalize. Defaults to the last axis.
    eps: Small epsilon to prevent division by zero.

  Returns:
    L2 normalized array with the same shape as x.
  """

  x_f32 = jnp.asarray(x, jnp.float32)
  norm_sq = (x_f32 * x_f32).sum(axis=dim, keepdims=True)
  is_zero = norm_sq == 0.0
  safe_norm_sq = jnp.where(is_zero, 1.0, norm_sq)
  inv_norm = jnp.where(is_zero, 0.0, jax.lax.rsqrt(safe_norm_sq + jnp.float32(eps)))
  return jnp.asarray(x_f32 * inv_norm, x.dtype)


Qwen3NextRMSNormLinen = nnx_wrappers.to_linen_class(
    RMSNorm,
    base_metadata_fn=variable_to_logically_partitioned,
    scale_init=jax.nn.initializers.zeros,
    scale_offset=1.0,
)

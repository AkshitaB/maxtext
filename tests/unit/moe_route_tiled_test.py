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

"""Unit and CPU numerics tests for `moe_route_tiled`."""

import functools
import unittest
import jax
from jax.ad_checkpoint import checkpoint_policies
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from maxtext.kernels.ragged.ragged_sort import _checkpoint_moe_route, ring_ragged_sort


class MoeRouteTiledTest(unittest.TestCase):
  """Tests for `moe_route_tiled` in `ragged_sort.py`."""

  def test_checkpoint_moe_route_shapes_and_values(self):
    """`_checkpoint_moe_route` preserves 1D shape and exact values for aligned and unaligned lengths."""
    for length in (128, 256, 300, 512, 13):
      x = jnp.arange(length, dtype=jnp.int32)
      y_untiled = _checkpoint_moe_route(x, tiled=False)
      y_tiled = _checkpoint_moe_route(x, tiled=True)
      self.assertEqual(y_untiled.shape, (length,))
      self.assertEqual(y_tiled.shape, (length,))
      np.testing.assert_array_equal(np.asarray(y_untiled), np.asarray(x))
      np.testing.assert_array_equal(np.asarray(y_tiled), np.asarray(x))

  def test_checkpoint_moe_route_tiled_saves_2d_tiles_in_jaxpr(self):
    """When `tiled=True`, the `moe_route` checkpoint_name primitive receives a `(-1, 128)` 2D array."""

    def fn(x, tiled):
      return _checkpoint_moe_route(x, tiled=tiled)

    x = jnp.arange(300, dtype=jnp.int32)
    jaxpr_untiled = jax.make_jaxpr(lambda a: fn(a, False))(x)
    jaxpr_tiled = jax.make_jaxpr(lambda a: fn(a, True))(x)

    untiled_shapes = [
        eqn.invars[0].aval.shape
        for eqn in jaxpr_untiled.eqns
        if eqn.primitive.name == "name" and eqn.params.get("name") == "moe_route"
    ]
    tiled_shapes = [
        eqn.invars[0].aval.shape
        for eqn in jaxpr_tiled.eqns
        if eqn.primitive.name == "name" and eqn.params.get("name") == "moe_route"
    ]
    self.assertEqual(untiled_shapes, [(300,)])
    self.assertEqual(tiled_shapes, [(3, 128)])

  def test_ring_ragged_sort_moe_route_tiled_bit_identical_fwd_bwd(self):
    """`ring_ragged_sort(..., moe_route_tiled=True)` is bit-identical to `moe_route_tiled=False` in fwd+bwd."""
    devices = np.array(jax.devices()[:1])
    mesh = Mesh(devices, ("ep",))
    rng = jax.random.PRNGKey(0)
    hidden = jax.random.normal(rng, (64, 32), dtype=jnp.float32)
    topk_idx = jax.random.randint(rng, (64, 2), 0, 8, dtype=jnp.int32)

    def run_case(tiled: bool):
      @functools.partial(jax.remat, policy=checkpoint_policies.save_only_these_names("moe_route"))
      def loss_fn(h):
        @functools.partial(
            jax.shard_map,
            mesh=mesh,
            in_specs=(jax.sharding.PartitionSpec(), jax.sharding.PartitionSpec()),
            out_specs=(
                jax.sharding.PartitionSpec(),
                jax.sharding.PartitionSpec(),
                jax.sharding.PartitionSpec(),
            ),
            check_vma=False,
        )
        def _inner(h_local, idx_local):
          return ring_ragged_sort(
              h_local,
              idx_local,
              num_experts=8,
              topk=2,
              ep_name="ep",
              ep_size=1,
              buffer_size=96,
              enforce_gather_fallback=True,
              enforce_gather_reduce_fallback=True,
              moe_route_tiled=tiled,
          )

        out, gs, rev = _inner(h, topk_idx)
        return jnp.sum(out * jnp.arange(1, out.shape[0] + 1, dtype=jnp.float32)[:, None]), (gs, rev)

      (loss, (gs, rev)), grad_h = jax.value_and_grad(loss_fn, has_aux=True)(hidden)
      return loss, gs, rev, grad_h

    loss_f, gs_f, rev_f, grad_f = run_case(False)
    loss_t, gs_t, rev_t, grad_t = run_case(True)
    np.testing.assert_array_equal(np.asarray(loss_f), np.asarray(loss_t))
    np.testing.assert_array_equal(np.asarray(gs_f), np.asarray(gs_t))
    np.testing.assert_array_equal(np.asarray(rev_f), np.asarray(rev_t))
    np.testing.assert_array_equal(np.asarray(grad_f), np.asarray(grad_t))


if __name__ == "__main__":
  unittest.main()

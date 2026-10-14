# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU mesh verification for `defer_small_all_reduces` (W2d).

Verifies on an 8-device CPU mesh (`fsdp=1, context=4, expert=2` under `cp-as-ep`):
1. `defer_small_all_reduces` defaults to `False`.
2. Deferring per-layer MoE `has_overflow` and `log_moe_drops` (`moe_dropped_assignments`,
   `moe_max_load_ratio`) produces bit-identical forward outputs, overflow flags, per-layer
   audit vectors (`collect_moe_audit`), and gradients while eliminating the per-layer scalar
   `psum`/`pmax` collectives inside `RoutedMoE.permute`.
3. Expanding replicated `RMSNorm.scale` across the activation `(batch, sequence)` mesh axes
   via `normalizations.expand_replicated_scale_for_scan` produces bit-identical forward
   outputs and activation gradients, matching `scale` gradients (`<= 1e-6` relative diff),
   and eliminates all per-layer `RMSNorm.scale` `all-reduce`s inside the `jax.lax.scan`
   `while/body` loop in favor of a single post-scan `all-reduce`.
"""

import os
import subprocess
import sys

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import pytest

from maxtext.layers import moe
from maxtext.layers import normalizations
from maxtext.training_engine import maxtext_engine
from maxtext.utils import maxtext_utils
from tests.unit import moe_chunk_pipeline_test as layer_setup

_PASS_TOKEN = "DEFER_SMALL_ALL_REDUCES_CHECKS_PASSED"
_RBF = 2.5


@pytest.mark.cpu_only
def test_defer_small_all_reduces_on_cpu_mesh():
  """Runs `main` below in a child process with 8 CPU devices."""
  env = os.environ.copy()
  env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + f" {layer_setup._XLA_FLAGS}"  # pylint: disable=protected-access
  env["JAX_PLATFORMS"] = "cpu"
  repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
  src_root = os.path.join(repo_root, "src")
  env["PYTHONPATH"] = (
      os.pathsep.join([src_root, repo_root, env["PYTHONPATH"]])
      if env.get("PYTHONPATH")
      else os.pathsep.join([src_root, repo_root])
  )
  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)
  print(result.stdout)
  assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert _PASS_TOKEN in result.stdout, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def _run_moe_audit_and_overflow(cfg, mesh, params, x, forced):
  """Runs a 2-layer RoutedMoE forward+backward and returns outputs, grads, overflow, audit, and HLO."""
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
    layer0 = layer_setup._build(cfg, mesh)  # pylint: disable=protected-access
    layer1 = layer_setup._build(cfg, mesh)  # pylint: disable=protected-access

    class TwoMoELayers(nnx.Module):

      def __init__(self, l0, l1):
        self.layer_0 = l0
        self.layer_1 = l1

      def __call__(self, h, forced_experts):
        h0, _, _ = self.layer_0(h, forced_routed_experts=forced_experts)
        h1, _, _ = self.layer_1(h + h0, forced_routed_experts=forced_experts)
        return h + h0 + h1

    model = TwoMoELayers(layer0, layer1)
    graphdef, _, rest = nnx.split(model, nnx.Param, ...)
    combined_params = nnx.State({"layer_0": params, "layer_1": params})

    def loss_fn(p, inp):
      mdl = nnx.merge(graphdef, p, rest)
      out = mdl(inp, forced)
      sown = nnx.state(mdl, nnx.Intermediate).to_pure_dict()
      raw_flags = maxtext_utils.collect_intermediates_by_suffix(sown, "moe_has_overflow", ravel=False)
      has_overflow = moe.reduce_moe_overflow_flags(raw_flags)
      audit = maxtext_engine.collect_moe_audit(mdl)
      loss = jnp.mean(out.astype(jnp.float32) ** 2)
      return loss, (out, has_overflow, audit)

    jit_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True))
    hlo = jit_fn.lower(combined_params, x).compile().as_text()
    (loss, (out, has_overflow, audit)), (dparams, dx) = jit_fn(combined_params, x)
    return (
        float(loss),
        np.asarray(out, dtype=np.float32),
        bool(has_overflow),
        {k: np.asarray(v) for k, v in audit.items()},
        jax.tree.map(lambda g: np.asarray(g, dtype=np.float32), dparams),
        np.asarray(dx, dtype=np.float32),
        hlo,
    )


def _check_scanned_rmsnorm_deferral(mesh):
  """Verifies `expand_replicated_scale_for_scan` + `RMSNorm.__call__` inside a nested `jax.lax.scan`."""
  b, s, k = 4, 16, 64
  x_sharding = NamedSharding(mesh, P("fsdp", "context", None))
  scale_sharding_3d = NamedSharding(mesh, P(None, None, None))
  scale_sharding_2d = NamedSharding(mesh, P(None, None))
  x = jax.device_put(
      jnp.arange(b * s * k, dtype=jnp.bfloat16).reshape(b, s, k) * jnp.bfloat16(0.01),
      x_sharding,
  )
  s_local = jax.device_put(jnp.ones((k, 2, 3), dtype=jnp.float32) * 1.1, scale_sharding_3d)
  s_global = jax.device_put(jnp.ones((k, 2), dtype=jnp.float32) * 0.9, scale_sharding_2d)

  def _apply_norm(c, scale_val):
    norm = normalizations.RMSNorm(
        num_features=k,
        epsilon=1e-6,
        dtype=jnp.bfloat16,
        weight_dtype=jnp.float32,
        scale_offset=1.0,
        rngs=nnx.Rngs(params=0),
    )
    norm.scale.set_value(scale_val)
    return c + norm(c)

  def run_scan(inp, sl, sg, defer):
    sl_moved = jnp.moveaxis(sl, 1, 0)
    sg_moved = jnp.moveaxis(sg, 1, 0)
    if defer:
      sl_moved = normalizations.expand_replicated_scale_for_scan(sl_moved, mesh, ("fsdp", "expert"), ("context",))
      sg_moved = normalizations.expand_replicated_scale_for_scan(sg_moved, mesh, ("fsdp", "expert"), ("context",))

    def outer_body(carry, xs):
      sl_b, sg_b = xs
      sl_b_in = jnp.moveaxis(sl_b, 1, 0)

      def inner_body(c_in, sl_l):
        return jax.checkpoint(_apply_norm)(c_in, sl_l), None

      carry, _ = jax.lax.scan(inner_body, carry, sl_b_in)
      carry = jax.checkpoint(_apply_norm)(carry, sg_b)
      return carry, None

    out, _ = jax.lax.scan(outer_body, inp, (sl_moved, sg_moved))
    return jnp.sum(out.astype(jnp.float32)), out

  with jax.set_mesh(mesh):
    fn_base = jax.jit(
        jax.value_and_grad(lambda inp, sl, sg: run_scan(inp, sl, sg, False), argnums=(0, 1, 2), has_aux=True)
    )
    fn_def = jax.jit(jax.value_and_grad(lambda inp, sl, sg: run_scan(inp, sl, sg, True), argnums=(0, 1, 2), has_aux=True))
    hlo_base = fn_base.lower(x, s_local, s_global).compile().as_text()
    hlo_def = fn_def.lower(x, s_local, s_global).compile().as_text()
    (l_base, out_base), (dx_base, dsl_base, dsg_base) = fn_base(x, s_local, s_global)
    (l_def, out_def), (dx_def, dsl_def, dsg_def) = fn_def(x, s_local, s_global)

  assert float(jnp.max(jnp.abs(l_base - l_def))) == 0.0
  assert float(jnp.max(jnp.abs(out_base - out_def))) == 0.0
  assert float(jnp.max(jnp.abs(dx_base - dx_def))) == 0.0
  rel_dsl = float(jnp.max(jnp.abs(dsl_base - dsl_def)) / jnp.max(jnp.abs(dsl_base)))
  rel_dsg = float(jnp.max(jnp.abs(dsg_base - dsg_def)) / jnp.max(jnp.abs(dsg_base)))
  assert rel_dsl <= 1e-6, f"rel_dsl={rel_dsl}"
  assert rel_dsg <= 1e-6, f"rel_dsg={rel_dsg}"

  base_while_ars = [l for l in hlo_base.splitlines() if "all-reduce(" in l and "while" in l]
  def_while_ars = [l for l in hlo_def.splitlines() if "all-reduce(" in l and "while" in l]
  print(f"RMSNorm scan while/body all-reduces: base={len(base_while_ars)}, defer={len(def_while_ars)}", flush=True)
  assert len(base_while_ars) > 0, "baseline must have per-layer all-reduces inside while/body"
  assert len(def_while_ars) == 0, f"deferred scan must have 0 all-reduces inside while/body, got {def_while_ars}"


def main():
  cfg_default = layer_setup._cfg(True)  # pylint: disable=protected-access
  assert cfg_default.defer_small_all_reduces is False, "defer_small_all_reduces must default to False"

  cfg_base = layer_setup._cfg(  # pylint: disable=protected-access
      True, ragged_buffer_factor=_RBF, log_moe_drops=True, defer_small_all_reduces=False
  )
  cfg_defer = layer_setup._cfg(  # pylint: disable=protected-access
      True, ragged_buffer_factor=_RBF, log_moe_drops=True, defer_small_all_reduces=True
  )
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg_base), cfg_base.mesh_axes)
  x, _ = layer_setup._inputs(cfg_base, forced_mode=False)  # pylint: disable=protected-access
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_base.logical_axis_rules):
    _, params, _ = nnx.split(layer_setup._build(cfg_base, mesh), nnx.Param, ...)  # pylint: disable=protected-access
  top_k = cfg_base.num_experts_per_tok
  forced = jnp.broadcast_to(jnp.arange(top_k, dtype=jnp.int32), (x.shape[0], x.shape[1], top_k))

  l_b, out_b, ov_b, aud_b, dp_b, dx_b, hlo_b = _run_moe_audit_and_overflow(cfg_base, mesh, params, x, forced)
  l_d, out_d, ov_d, aud_d, dp_d, dx_d, hlo_d = _run_moe_audit_and_overflow(cfg_defer, mesh, params, x, forced)

  assert l_b == l_d, f"loss mismatch: {l_b} vs {l_d}"
  np.testing.assert_array_equal(out_b, out_d)
  assert ov_b is True and ov_d is True
  np.testing.assert_array_equal(aud_b["dropped"], aud_d["dropped"])
  np.testing.assert_allclose(aud_b["max_ratio"], aud_d["max_ratio"], rtol=0, atol=0)
  np.testing.assert_array_equal(dx_b, dx_d)
  for (path_b, g_b), (path_d, g_d) in zip(
      jax.tree_util.tree_leaves_with_path(dp_b), jax.tree_util.tree_leaves_with_path(dp_d)
  ):
    np.testing.assert_array_equal(g_b, g_d, err_msg=f"grad mismatch at {path_b} vs {path_d}")

  b_fwd_shmap_ars = [l for l in hlo_b.splitlines() if "all-reduce(" in l and "jvp()/shard_map/p" in l]
  d_fwd_shmap_ars = [l for l in hlo_d.splitlines() if "all-reduce(" in l and "jvp()/shard_map/p" in l]
  d_pred_ags = [l for l in hlo_d.splitlines() if "all-gather(" in l and "pred[" in l]
  print(
      f"2-layer MoE jvp/shard_map all-reduces: base={len(b_fwd_shmap_ars)}, defer={len(d_fwd_shmap_ars)}; "
      f"pred all-gathers={len(d_pred_ags)}",
      flush=True,
  )
  assert len(b_fwd_shmap_ars) >= 2, f"expected per-layer psum/pmax in baseline, got {b_fwd_shmap_ars}"
  assert (
      len(d_fwd_shmap_ars) == 0
  ), f"defer_small_all_reduces=True must eliminate all jvp()/shard_map all-reduces, got {d_fwd_shmap_ars}"
  assert len(d_pred_ags) == 0, f"defer_small_all_reduces=True must not emit pred all-gathers, got {d_pred_ags}"

  _check_scanned_rmsnorm_deferral(mesh)
  print(_PASS_TOKEN, flush=True)


if __name__ == "__main__":
  main()

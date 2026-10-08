# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the DeepSeek routed gate bias dtype."""

import unittest

from flax import nnx
import jax.numpy as jnp
from maxtext.layers import moe
import pytest

pytestmark = pytest.mark.cpu_only


class _DummyMoe(nnx.Module):

  def __init__(self, gate: moe.GateLogit):
    self.gate = gate


class _DummyLayer(nnx.Module):

  def __init__(self, gate: moe.GateLogit):
    self.moe = _DummyMoe(gate)


class _DummyScannedBlocks(nnx.Module):

  def __init__(self, gate0: moe.GateLogit, gate1: moe.GateLogit):
    self.layers_0 = _DummyLayer(gate0)
    self.layers_1 = _DummyLayer(gate1)


class _DummyDecoder(nnx.Module):

  def __init__(
      self,
      gate_unscanned: moe.GateLogit,
      gate_scanned0: moe.GateLogit,
      gate_scanned1: moe.GateLogit,
  ):
    self.layers_3 = _DummyLayer(gate_unscanned)
    self.scanned_blocks = _DummyScannedBlocks(gate_scanned0, gate_scanned1)


class _DummyModel(nnx.Module):

  def __init__(self, decoder: _DummyDecoder):
    self.decoder = decoder


class DeepSeekV4RoutedBiasTest(unittest.TestCase):
  """Tests for float32 routed gate bias precision."""

  def test_gate_logit_routed_bias_dtype(self):
    rngs = nnx.Rngs(0)
    gate_fp32 = moe.GateLogit(
        in_features_shape=16,
        out_features_shape=8,
        model_name="deepseek4-284b",
        mesh=None,
        rngs=rngs,
        use_bias=True,
        dtype=jnp.float32,
        weight_dtype=jnp.bfloat16,
    )
    self.assertIsInstance(gate_fp32.bias, moe.MoEBiasVar)
    self.assertEqual(gate_fp32.bias.get_value().dtype, jnp.float32)

    gate_bf16 = moe.GateLogit(
        in_features_shape=16,
        out_features_shape=8,
        model_name="deepseek4-284b",
        mesh=None,
        rngs=rngs,
        use_bias=True,
        dtype=jnp.bfloat16,
        weight_dtype=jnp.bfloat16,
    )
    self.assertIsInstance(gate_bf16.bias, moe.MoEBiasVar)
    self.assertEqual(gate_bf16.bias.get_value().dtype, jnp.bfloat16)


if __name__ == "__main__":
  unittest.main()

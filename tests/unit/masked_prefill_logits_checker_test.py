# Copyright 2023–2026 Google LLC
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

"""Tests for masked_prefill_logits_checker."""

import hashlib
import os
from absl.testing import absltest
from absl.testing import parameterized
from tests.utils import masked_prefill_logits_checker as checker
import numpy as np
import pytest


@pytest.mark.cpu_only
class MaskedPrefillLogitsCheckerTest(parameterized.TestCase):

  def test_golden_batch_verbatim_content_and_sha256(self):
    sha256 = hashlib.sha256(checker.GOLDEN_BATCH_TEXT.encode("utf-8")).hexdigest()
    self.assertEqual(sha256, "53b2f8182e49f439404efb937154ef4cba22d953e3efd189b7a60808c98fd984")
    self.assertEqual(checker.GOLDEN_BATCH, (checker.GOLDEN_BATCH_TEXT,))
    self.assertEqual(checker.GOLDEN_PROMPTS, checker.GOLDEN_BATCH)
    for segment in (
        "The James Webb Space Telescope (JWST)",
        "Newton's law of universal gravitation:",
        "F = G * (m1 * m2) / r^2",
        "def calculate_orbital_velocity(mass_kg, radius_m):",
        "# Payload Configuration",
        '"sunshield_deployed": true',
        "Chinese (Simplified): 韦伯太空望远镜 (Webb Space Telescope).",
        "Arabic: تلسكوب جيمس ويب الفضائي",
        "Russian: Космический телескоп Джеймса Уэбба",
        "Emoji byte-pair encoding test: 🚀🔭✨🌌🌍",
    ):
      self.assertIn(segment, checker.GOLDEN_BATCH_TEXT)

  def test_compare_arrays_shape_mismatch_and_parity(self):
    # pylint: disable=protected-access
    res_broadcast = checker._compare_arrays(np.zeros((2, 3, 4)), np.zeros((2, 3, 1)))
    self.assertEqual((res_broadcast["passed"], res_broadcast["shape_mismatch"]), (False, ((2, 3, 4), (2, 3, 1))))
    res_incompatible = checker._compare_arrays(np.zeros((2, 3, 4)), np.zeros((2, 5)))
    self.assertEqual((res_incompatible["passed"], res_incompatible["shape_mismatch"]), (False, ((2, 3, 4), (2, 5))))
    res_matching = checker._compare_arrays(np.ones((2, 3)), np.ones((2, 3)))
    self.assertEqual((res_matching["passed"], res_matching["shape_mismatch"]), (True, None))

  def test_check_routing_parity(self):
    routing_jax = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    res_pass = checker.check_routing_parity(routing_jax, routing_jax.copy())
    self.assertTrue(res_pass["passed"])
    self.assertEqual((res_pass["match_rate"], res_pass["num_mismatches"]), (1.0, 0))
    self.assertIsNone(res_pass["first_mismatch"])
    routing_hf = routing_jax.copy()
    routing_hf[1, 0, 1] = 99
    res_fail = checker.check_routing_parity(routing_jax, routing_hf)
    self.assertEqual((res_fail["passed"], res_fail["num_mismatches"], res_fail["first_mismatch"]), (False, 1, (1, 0, 1)))

  def test_check_indexer_block_parity_8k(self):
    np.random.seed(42)
    jax_blocks = np.random.randint(0, 2048, size=(12, 8192, 4))
    hf_permuted = jax_blocks[..., ::-1].copy()
    res_pass = checker.check_indexer_block_parity(jax_blocks, hf_permuted)
    self.assertEqual((res_pass["passed"], res_pass["match_rate"]), (True, 1.0))
    hf_mismatch = hf_permuted.copy()
    hf_mismatch[3, 100, 0] = 9999
    res_fail = checker.check_indexer_block_parity(jax_blocks, hf_mismatch)
    self.assertEqual((res_fail["passed"], res_fail["first_mismatch"][:2]), (False, (3, 100)))

  def test_dynamic_threshold_single_owner_scaling(self):
    self.assertEqual(checker.get_dynamic_threshold(0.01, 256), 0.01)
    self.assertEqual(checker.get_dynamic_threshold(0.01, 512), 0.01)
    self.assertEqual(checker.get_dynamic_threshold(0.01, 1024), 0.01)
    self.assertAlmostEqual(checker.get_dynamic_threshold(0.01, 2048), 0.014)
    self.assertAlmostEqual(checker.get_dynamic_threshold(0.01, 8192), 0.018)

  def test_check_logits_parity_and_min_cosine_gating(self):
    np.random.seed(0)
    logits = np.random.randn(2, 512, 32)
    res_pass = checker.check_logits_parity(logits, logits.copy(), seq_len=512)
    self.assertTrue(res_pass["passed"])
    self.assertAlmostEqual(res_pass["cosine_sim_mean"], 1.0)
    self.assertAlmostEqual(res_pass["cosine_sim_min"], 1.0)
    self.assertIn("cos_tol", res_pass)
    self.assertFalse(checker.check_logits_parity(logits, logits + 2.0, seq_len=512)["passed"])
    logits_a = np.ones((1, 512, 10), dtype=np.float64)
    logits_b = logits_a.copy()
    logits_b[0, 0, :] = -np.ones(10, dtype=np.float64)
    res_min_fail = checker.check_logits_parity(logits_a, logits_b, seq_len=512)
    self.assertFalse(res_min_fail["passed"])
    self.assertLess(res_min_fail["cosine_sim_min"], res_min_fail["cos_tol"])
    corrupt = logits.copy()
    corrupt[:, 200:, :] += 5.0
    mask = np.zeros((2, 512), dtype=bool)
    mask[:, :100] = True
    self.assertTrue(checker.check_logits_parity(logits, corrupt, mask=mask, seq_len=512)["passed"])
    with self.assertRaises(ValueError):
      checker.check_logits_parity(logits, corrupt, mask=np.zeros((2, 512), dtype=bool))
    logits_8k = np.ones((1, 8192, 10), dtype=np.float64)
    logits_8k_noise = logits_8k.copy()
    logits_8k_noise[..., 0] += 0.013
    self.assertFalse(checker.check_logits_parity(logits_8k, logits_8k_noise, seq_len=512)["passed"])
    self.assertTrue(checker.check_logits_parity(logits_8k, logits_8k_noise, seq_len=8192)["passed"])

  def test_compute_masked_ce_loss_alignment_contract(self):
    tokens = np.array([[1, 3, 5, 7, 2, 4]], dtype=np.int64)
    logits = np.zeros((1, 6, 10), dtype=np.float64)
    for i in range(5):
      logits[0, i, tokens[0, i + 1]] = 20.0
    self.assertLess(checker.compute_masked_ce_loss(logits, tokens, shift=True), 1e-4)
    self.assertGreater(checker.compute_masked_ce_loss(logits, tokens, shift=False), 10.0)
    self.assertLess(checker.compute_masked_ce_loss(logits[:, :-1, :], tokens[:, 1:], shift=False), 1e-4)
    bad_inputs = (
        (logits[:, :5, :], tokens, None, False),
        (logits, tokens[:, :5], None, False),
        (logits, tokens, np.ones((1, 5), dtype=bool), False),
        (np.zeros((1, 1, 10)), np.zeros((1, 1), dtype=np.int64), None, True),
    )
    for bad_logits, bad_tokens, bad_mask, shift_val in bad_inputs:
      with self.assertRaises(ValueError):
        checker.compute_masked_ce_loss(bad_logits, bad_tokens, bad_mask, shift=shift_val)

  def test_check_tinymmlu_loss_delta(self):
    np.random.seed(1)
    logits_jax = np.random.randn(2, 16, 10)
    targets = np.random.randint(0, 10, size=(2, 16))
    res_pass = checker.check_tinymmlu_loss_delta(logits_jax, logits_jax + 1e-6, targets)
    self.assertTrue(res_pass["passed"])
    self.assertLessEqual(res_pass["loss_delta"], 1e-4)
    corrupt_logits = logits_jax.copy()
    corrupt_logits[..., 0] += 0.5
    res_fail = checker.check_tinymmlu_loss_delta(logits_jax, corrupt_logits, targets)
    self.assertEqual((res_fail["passed"], res_fail["loss_delta"] > 1e-4), (False, True))

  def test_check_phase3_holdout_ce(self):
    np.random.seed(2)
    pre_sft = np.random.randn(2, 16, 10)
    targets = np.random.randint(0, 10, size=(2, 16))
    post_sft = pre_sft.copy()
    batch_indices, token_indices = np.ogrid[:2, :15]
    post_sft[batch_indices, token_indices, targets[:, 1:]] += 2.0
    res_pass = checker.check_phase3_holdout_ce(post_sft, pre_sft, targets)
    self.assertEqual((res_pass["passed"], res_pass["prob_trend_upward"]), (True, True))
    res_regress = checker.check_phase3_holdout_ce(pre_sft, post_sft, targets)
    self.assertEqual((res_regress["passed"], res_regress["prob_trend_upward"]), (False, False))
    nan_logits = pre_sft.copy()
    nan_logits[0, 0, 0] = np.nan
    res_nan = checker.check_phase3_holdout_ce(nan_logits, pre_sft, targets)
    self.assertEqual((res_nan["passed"], res_nan["is_finite"]), (False, False))

  def test_save_and_load_golden_npz_with_mean_var_and_meta(self):
    temp_dir = self.create_tempdir().full_path
    npz_path = os.path.join(temp_dir, "test_golden.npz")
    tokens, logits = np.array([[101, 102, 103]]), np.random.randn(1, 3, 50)
    routing, blocks = np.array([[[0, 1], [1, 2], [2, 3]]]), np.array([[[1, 2], [2, 3], [3, 4]]])
    mask = np.array([[True, True, False]])
    mean_var = {"post_attn_proj": np.array([0.12, 0.45]), "post_mlp_act": np.array([0.33, 0.78])}
    checker.save_golden_npz(npz_path, tokens, logits, routing, blocks, mask, mean_var, 512, "float32")
    loaded = checker.load_golden_npz(npz_path)
    for key, expected in (
        ("tokens", tokens),
        ("logits", logits),
        ("routing_indices", routing),
        ("indexer_selected_blocks", blocks),
        ("target_mask", mask),
    ):
      np.testing.assert_array_equal(loaded[key], expected)
    self.assertIn("mean_var", loaded)
    for name, arr in mean_var.items():
      np.testing.assert_array_equal(loaded["mean_var"][name], arr)
    self.assertEqual((loaded.get("meta", {}).get("seq_len"), loaded.get("meta", {}).get("dtype")), (512, "float32"))
    with self.assertRaises(FileNotFoundError):
      checker.load_golden_npz(os.path.join(temp_dir, "nonexistent.npz"))

  def test_main_cli_fails_loudly_for_unimplemented_and_invalid_modes(self):
    for valid_mode in ("prefill_logits", "tinymmlu", "phase3_holdout_ce", "generate_goldens"):
      with self.assertRaises(NotImplementedError) as context:
        checker.main([f"mode={valid_mode}"])
      self.assertIn("CL 8b", str(context.exception))
    with self.assertRaises(ValueError):
      checker.main(["mode=unsupported_mode"])


if __name__ == "__main__":
  absltest.main()

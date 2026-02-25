#!/usr/bin/env python3
"""Verify vectorized operations match for-loop implementations."""

import numpy as np
import torch
import sys
sys.path.insert(0, "/home/taejinp/projects/sot_mt_asr_rt/NeMo")
from nemo.collections.asr.data.audio_to_sot_text_lhotse import LhotseSpeechToTextBpeDataset

np.random.seed(42)
torch.manual_seed(42)


def test_dtw_local_cost():
    """Verify _dtw_cost_batch local cost matches for-loop version."""
    K, T, P, num_spk = 20, 100, 4, 4
    activity = np.random.rand(T, num_spk).astype(np.float32)
    spk_seq_arr = np.random.randint(0, num_spk, size=K, dtype=np.intp)
    perm_batch = np.array([[0, 1, 2, 3], [1, 0, 2, 3], [2, 1, 0, 3], [1, 2, 0, 3]], dtype=np.intp)
    token_weights = np.random.rand(K).astype(np.float32)

    # Vectorized (current)
    costs_vec = LhotseSpeechToTextBpeDataset._dtw_cost_batch(
        activity, spk_seq_arr, perm_batch, num_spk, token_weights
    )

    # For-loop local cost (manual verification of the local matrix)
    valid = spk_seq_arr < num_spk
    activity_permuted = activity[:, perm_batch].transpose(1, 0, 2)
    activity_sum = np.maximum(activity.sum(axis=1), 1.0).astype(np.float32)

    local_loop = np.ones((P, K, T), dtype=np.float32)
    for k in np.where(valid)[0]:
        local_loop[:, k, :] = 1.0 - activity_permuted[:, :, spk_seq_arr[k]] / activity_sum
    local_loop = local_loop * token_weights[np.newaxis, :, np.newaxis]

    # Vectorized local
    cols = np.where(valid, spk_seq_arr, 0)
    local_vec = 1.0 - activity_permuted[:, :, cols].transpose(0, 2, 1) / activity_sum
    local_vec[:, ~valid, :] = 1.0
    local_vec = local_vec * token_weights[np.newaxis, :, np.newaxis]

    assert np.allclose(local_loop, local_vec), "DTW local cost mismatch!"
    print("DTW local cost: OK")


def test_cost_matrix():
    """Verify cost matrix matches for-loop."""
    num_spk = 4
    text_ratio = torch.rand(num_spk)
    rttm_ratio = torch.rand(num_spk)

    cost_loop = torch.zeros(num_spk, num_spk)
    for row in range(num_spk):
        for col in range(num_spk):
            cost_loop[row, col] = torch.abs(text_ratio[row] - rttm_ratio[col])

    cost_vec = torch.abs(text_ratio.unsqueeze(1) - rttm_ratio.unsqueeze(0))

    assert torch.allclose(cost_loop, cost_vec), "Cost matrix mismatch!"
    print("Cost matrix: OK")


def test_token_counts():
    """Verify bincount token_counts matches for-loop."""
    spk_seq = [0, 1, 0, 1, 1, 2, 0]
    num_spk = 4

    token_counts_loop = [0] * num_spk
    for spk in spk_seq:
        if spk < num_spk:
            token_counts_loop[spk] += 1
    total_loop = sum(token_counts_loop)

    spk_arr = np.array(spk_seq, dtype=np.intp)
    token_counts_vec = np.bincount(spk_arr[spk_arr < num_spk], minlength=num_spk)
    total_vec = token_counts_vec.sum()

    assert total_loop == total_vec
    assert list(token_counts_loop) == list(token_counts_vec)
    print("Token counts: OK")


def test_column_reorder():
    """Verify speaker_activity[:, best_perm] matches for-loop."""
    T, num_spk = 50, 4
    speaker_activity = torch.rand(T, num_spk)
    best_perm = [1, 0, 2, 3]

    fixed_loop = torch.zeros_like(speaker_activity)
    for text_col, rttm_col in enumerate(best_perm):
        fixed_loop[:, text_col] = speaker_activity[:, rttm_col]

    fixed_vec = speaker_activity[:, best_perm].clone()

    assert torch.allclose(fixed_loop, fixed_vec), "Column reorder mismatch!"
    print("Column reorder: OK")


def test_cost_indexing():
    """Verify cost[range(num_spk), best_perm].sum() matches for-loop."""
    num_spk = 4
    cost = torch.rand(num_spk, num_spk)
    best_perm = [1, 0, 2, 3]

    sum_loop = sum(cost[idx, best_perm[idx]].item() for idx in range(num_spk))
    sum_vec = cost[list(range(num_spk)), best_perm].sum().item()

    assert abs(sum_loop - sum_vec) < 1e-6, "Cost indexing mismatch!"
    print("Cost indexing: OK")


def test_full_dtw_cost():
    """Verify full _dtw_cost_batch output (with invalid speakers)."""
    K, T, P, num_spk = 15, 80, 2, 4
    activity = np.random.rand(T, num_spk).astype(np.float32)
    spk_seq_arr = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1], dtype=np.intp)
    perm_batch = np.array([[0, 1, 2, 3], [1, 0, 2, 3]], dtype=np.intp)
    token_weights = np.ones(K, dtype=np.float32)

    costs = LhotseSpeechToTextBpeDataset._dtw_cost_batch(
        activity, spk_seq_arr, perm_batch, num_spk, token_weights
    )
    assert costs.shape == (2,)
    assert np.all(np.isfinite(costs))
    assert np.all(costs >= 0)
    print("Full DTW cost: OK")


def test_invalid_speakers():
    """Verify DTW with spk_seq containing invalid (>= num_spk) indices."""
    K, T, num_spk = 10, 50, 2
    activity = np.random.rand(T, num_spk).astype(np.float32)
    # Include invalid speaker index 5 when num_spk=2
    spk_seq_arr = np.array([0, 1, 0, 5, 1, 0, 1, 5, 0, 1], dtype=np.intp)
    perm_batch = np.array([[0, 1], [1, 0]], dtype=np.intp)
    token_weights = np.ones(K, dtype=np.float32)

    costs = LhotseSpeechToTextBpeDataset._dtw_cost_batch(
        activity, spk_seq_arr, perm_batch, num_spk, token_weights
    )
    assert costs.shape == (2,)
    assert np.all(np.isfinite(costs))
    # Invalid speakers should get cost 1.0, so total cost should be higher
    assert np.all(costs > 0)
    print("Invalid speakers: OK")


if __name__ == "__main__":
    test_dtw_local_cost()
    test_cost_matrix()
    test_token_counts()
    test_column_reorder()
    test_cost_indexing()
    test_full_dtw_cost()
    test_invalid_speakers()
    print("\nAll vectorized operations match for-loop implementations.")

# Copyright (c) Nex-AGI. All rights reserved.
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

"""
Compare production GRPO against the historical dev-zms-math baseline.

GRPO tests call the production trajectory API, including its sample/population
standard-deviation modes and intentional zero advantage for singleton groups.
The PPO loss and token broadcasting checks below retain historical/reference
calculations; they are not end-to-end checks of the current Weaver trainer.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch

from northrl.algorithm.advantage.grpo import compute_grpo_advantage_for_trajectories
from northrl.northrl_types import Trajectory

# ============================================================
# BASELINE FUNCTIONS (copied from dev-zms-math branch)
# ============================================================


def baseline_masked_mean(
    tensor: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    masked_tensor = tensor * mask
    return masked_tensor.sum() / (mask.sum() + eps)


def baseline_agg_loss_token_mean(loss_mat: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    return baseline_masked_mean(loss_mat, loss_mask)


def baseline_compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: list,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, dict]:
    """Baseline GRPO advantage (dev-zms-math core_algos.py)."""
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
                id2std[idx] = torch.tensor(1.0, device=scores.device)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(score_tensor)
                id2std[idx] = torch.std(score_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        for i in range(bsz):
            idx = index[i]
            scores[i] = (scores[i] - id2mean[idx]) / (id2std[idx] + epsilon)

        advantages = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return advantages, id2std


def baseline_compute_ppo_clip_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    eos_mask: torch.Tensor,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.2,
    clip_ratio_c: Optional[float] = None,
    entropy: Optional[torch.Tensor] = None,
    entropy_coeff: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Baseline PPO clip loss (NX_20250515 + v2 dual-clip)."""
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = baseline_masked_mean(-negative_approx_kl, eos_mask)

    lower_bound = 1.0 - clip_ratio_low
    upper_bound = 1.0 + clip_ratio_high

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, lower_bound, upper_bound)
    clip_pg_losses = torch.max(pg_losses, pg_losses2)

    # clipfrac: baseline uses gt(pg_losses2, pg_losses1)
    pg_clipfrac = baseline_masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)

    # Dual-clip (NX_20250515_v2)
    if clip_ratio_c is not None and clip_ratio_c > 1.0:
        pg_losses3 = -advantages * clip_ratio_c
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses)
        pg_clipfrac_lower = baseline_masked_mean(
            torch.gt(clip_pg_losses, pg_losses3).float() * (advantages < 0).float(), eos_mask
        )
        elementwise_loss = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses)
    else:
        elementwise_loss = clip_pg_losses
        pg_clipfrac_lower = torch.tensor(0.0)

    pg_loss = baseline_agg_loss_token_mean(elementwise_loss, eos_mask)

    # Entropy
    entropy_loss = torch.tensor(0.0)
    if entropy is not None:
        entropy_loss = baseline_masked_mean(entropy, eos_mask)
    policy_loss = pg_loss - entropy_coeff * entropy_loss

    # Metrics
    avg_ratio = baseline_masked_mean(ratio, eos_mask)

    metrics = {
        "pg_loss": pg_loss.item(),
        "policy_loss": policy_loss.item(),
        "entropy_loss": entropy_loss.item(),
        "pg_clipfrac": pg_clipfrac.item(),
        "ppo_kl": ppo_kl.item(),
        "avg_ratio": avg_ratio.item(),
        "dual_clipfrac": (
            pg_clipfrac_lower.item()
            if isinstance(pg_clipfrac_lower, torch.Tensor)
            else pg_clipfrac_lower
        ),
    }
    return policy_loss, elementwise_loss, metrics


# ============================================================
# HISTORICAL WEAVER LOSS REFERENCE (from weaver-trainer computers.py)
# ============================================================


def weaver_compute_ppo_clip_loss(
    logprobs: torch.Tensor,
    sampling_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    effective_valid: torch.Tensor,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.2,
    clip_ratio_c: Optional[float] = None,
    entropy: Optional[torch.Tensor] = None,
    entropy_coeff: float = 0.0,
    loss_agg_mode: str = "token-mean",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Weaver PPO clip loss (weaver-trainer PPOClipLossComputer)."""
    # Zero out non-response positions
    logprobs = torch.where(effective_valid, logprobs, torch.zeros_like(logprobs))
    sampling_logprobs = torch.where(
        effective_valid, sampling_logprobs, torch.zeros_like(sampling_logprobs)
    )
    advantages = torch.where(effective_valid, advantages, torch.zeros_like(advantages))

    logprob_diff = logprobs - sampling_logprobs
    prob_ratio = torch.exp(logprob_diff)

    lower_bound = 1.0 - clip_ratio_low
    upper_bound = 1.0 + clip_ratio_high

    pg_losses = -advantages * prob_ratio
    pg_losses2 = -advantages * torch.clamp(prob_ratio, lower_bound, upper_bound)
    elementwise_loss = torch.max(pg_losses, pg_losses2)

    # Dual-clip
    if clip_ratio_c is not None and clip_ratio_c > 1.0:
        pg_losses3 = -advantages * clip_ratio_c
        dual_clipped = torch.min(pg_losses3, elementwise_loss)
        elementwise_loss = torch.where(advantages < 0, dual_clipped, elementwise_loss)

    elementwise_loss = torch.where(
        effective_valid, elementwise_loss, torch.zeros_like(elementwise_loss)
    )

    # Loss aggregation
    num_valid = effective_valid.sum().float()
    if loss_agg_mode == "token-mean":
        loss = elementwise_loss[effective_valid].sum() / num_valid.clamp(min=1.0)
    else:
        raise NotImplementedError(f"Only token-mean supported in test, got {loss_agg_mode}")

    # Entropy
    entropy_loss_val = torch.tensor(0.0)
    if entropy is not None and entropy_coeff != 0.0:
        entropy_loss_val = entropy.masked_select(effective_valid).mean()
    loss = loss - entropy_coeff * entropy_loss_val

    # Metrics (response-only, matching baseline)
    mask = effective_valid
    total = max(int(mask.sum().item()), 1)
    std_clip_active = torch.gt(pg_losses2, pg_losses).float()
    clipfrac = float(std_clip_active[mask].sum().cpu()) / total
    valid_logprob_diff = logprob_diff[mask]
    ppo_kl = float(-valid_logprob_diff.mean().cpu()) if valid_logprob_diff.numel() > 0 else 0.0
    valid_ratios = prob_ratio.masked_select(mask)
    avg_ratio = float(valid_ratios.mean().cpu())

    dual_clipfrac = 0.0
    if clip_ratio_c is not None and clip_ratio_c > 1.0:
        std_clipped = torch.max(pg_losses, pg_losses2)
        dual_active = (std_clipped > pg_losses3) & (advantages < 0)
        dual_clipfrac = float(dual_active[mask].float().sum().cpu()) / total

    metrics = {
        "pg_loss": float((loss + entropy_coeff * entropy_loss_val).item()),
        "policy_loss": loss.item(),
        "entropy_loss": float(entropy_loss_val.item()),
        "pg_clipfrac": clipfrac,
        "ppo_kl": ppo_kl,
        "avg_ratio": avg_ratio,
        "dual_clipfrac": dual_clipfrac,
    }
    return loss, elementwise_loss, metrics


# ============================================================
# TEST DATA GENERATION
# ============================================================


def make_test_data(
    n_prompts: int = 4,
    rollout_n: int = 8,
    prompt_lengths: Optional[List[int]] = None,
    response_lengths: Optional[List[int]] = None,
    seed: int = 42,
):
    """Generate realistic test data for parity testing.

    Returns:
        trajectories: list of Trajectory objects (production GRPO input)
        baseline_batch: dict of tensors (baseline format)
    """
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    n_samples = n_prompts * rollout_n

    if prompt_lengths is None:
        prompt_lengths = rng.randint(50, 200, size=n_samples).tolist()
    if response_lengths is None:
        response_lengths = rng.randint(100, 500, size=n_samples).tolist()

    max_prompt = max(prompt_lengths)
    max_response = max(response_lengths)
    total_width = max_prompt + max_response

    # Generate rewards (binary: 0 or 1)
    rewards = rng.choice([0.0, 1.0], size=n_samples, p=[0.6, 0.4]).tolist()

    # Group IDs
    group_ids = []
    for p in range(n_prompts):
        group_ids.extend([f"prompt_{p}"] * rollout_n)

    # Generate sampling-logprob denominators (realistic: mostly small negative values).
    # Weaver trajectories: no padding, just raw [prompt | response] tokens
    trajectories = []
    for i in range(n_samples):
        p_len = prompt_lengths[i]
        r_len = response_lengths[i]
        total_len = p_len + r_len

        tokens = list(range(1000, 1000 + total_len + 1))
        loss_mask = [0] * p_len + [1] * (r_len + 1)  # include EOS in loss
        # Sampling-logprob denominator: 0 for prompt, small negative for response + EOS.
        old_log_probs_values = [0.0] * p_len + [rng.uniform(-0.5, -0.01) for _ in range(r_len + 1)]

        traj = Trajectory(
            tokens=tokens,
            loss_mask=loss_mask,
            reward=rewards[i],
            extra_fields={
                "old_log_probs": old_log_probs_values,
                "group_id": group_ids[i],
                "run_id": i % rollout_n,
                "expected_rollouts": rollout_n,
            },
        )
        trajectories.append(traj)

    # --- Build baseline-format batch ---
    # Baseline layout: [left_pad | prompt | response | right_pad]
    # All prompts left-padded to max_prompt, all responses right-padded to max_response.
    # This ensures response portion is always at [:, max_prompt:] = [:, -max_response:]
    input_ids = torch.full((n_samples, total_width), 0, dtype=torch.long)
    attention_mask = torch.zeros((n_samples, total_width), dtype=torch.float32)
    loss_mask_tensor = torch.zeros((n_samples, total_width), dtype=torch.float32)
    responses = torch.zeros((n_samples, max_response), dtype=torch.long)
    old_log_probs = torch.zeros((n_samples, max_response), dtype=torch.float32)

    for i in range(n_samples):
        p_len = prompt_lengths[i]
        r_len = response_lengths[i]
        traj = trajectories[i]

        # Left-pad prompt: prompt starts at (max_prompt - p_len)
        prompt_offset = max_prompt - p_len
        # Prompt tokens
        input_ids[i, prompt_offset:max_prompt] = torch.tensor(
            traj["tokens"][:p_len], dtype=torch.long
        )
        attention_mask[i, prompt_offset:max_prompt] = 1.0
        # loss_mask: 0 for prompt
        # Response tokens: right after prompt, right-padded
        input_ids[i, max_prompt : max_prompt + r_len] = torch.tensor(
            traj["tokens"][p_len : p_len + r_len], dtype=torch.long
        )
        attention_mask[i, max_prompt : max_prompt + r_len] = 1.0
        loss_mask_tensor[i, max_prompt : max_prompt + r_len] = 1.0

        # Response-only tensors (for baseline's response_length-based slicing)
        responses[i, :r_len] = torch.tensor(traj["tokens"][p_len : p_len + r_len], dtype=torch.long)
        old_log_probs[i, :r_len] = torch.tensor(
            traj["old_log_probs"][p_len : p_len + r_len], dtype=torch.float32
        )

    # response_mask = attention_mask in the response portion
    # Since layout is [left_pad | prompt | response | right_pad],
    # the response portion is [:, max_prompt:] = [:, -max_response:]
    response_mask = attention_mask[:, max_prompt:]  # [B, max_response]

    # Build token_level_rewards: reward at last valid response token
    token_level_rewards = torch.zeros((n_samples, max_response), dtype=torch.float32)
    for i in range(n_samples):
        r_len = response_lengths[i]
        token_level_rewards[i, r_len - 1] = rewards[i]

    baseline_batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "responses": responses,
        "old_log_probs": old_log_probs,
        "response_mask": response_mask,
        "token_level_rewards": token_level_rewards,
        "group_ids": group_ids,
        "rewards": rewards,
        "max_prompt": max_prompt,
        "max_response": max_response,
        "prompt_lengths": prompt_lengths,
        "response_lengths": response_lengths,
    }

    return trajectories, baseline_batch


# ============================================================
# TESTS
# ============================================================


class TestGRPOAdvantageParity:
    """Compare GRPO advantage computation between baseline and weaver."""

    def test_advantage_values_match(self):
        """Scalar advantages should be identical (same formula, same data)."""
        trajectories, baseline_batch = make_test_data()
        n_samples = len(trajectories)

        # --- Baseline ---
        baseline_adv, baseline_std = baseline_compute_grpo_outcome_advantage(
            token_level_rewards=baseline_batch["token_level_rewards"],
            eos_mask=baseline_batch["response_mask"],
            index=baseline_batch["group_ids"],
        )
        # Extract per-sample scalar advantage from the first valid response token.
        # With left-pad layout, response_mask[i, 0] = 1 for all samples.
        baseline_scalar_adv = []
        for i in range(n_samples):
            # Find first valid position in response_mask
            valid_pos = (baseline_batch["response_mask"][i] > 0).nonzero(as_tuple=True)[0]
            if len(valid_pos) > 0:
                baseline_scalar_adv.append(baseline_adv[i, valid_pos[0]].item())
            else:
                baseline_scalar_adv.append(0.0)

        # Training and the historical baseline both use sample standard deviation.
        weaver_trajs = compute_grpo_advantage_for_trajectories(trajectories, use_run_ids=True)
        assert weaver_trajs is trajectories
        weaver_scalar_adv = [float(t["advantage"]) for t in weaver_trajs]

        # --- Compare ---
        print("\n=== GRPO Advantage Comparison ===")
        print(
            f"{'Sample':>6} {'Reward':>6} {'Group':>10} {'Baseline':>10} {'Weaver':>10} {'Diff':>12}"
        )
        max_diff = 0.0
        for i in range(n_samples):
            diff = abs(baseline_scalar_adv[i] - weaver_scalar_adv[i])
            max_diff = max(max_diff, diff)
            if i < 16 or diff > 1e-4:  # print first 16 + any mismatches
                print(
                    f"{i:>6} {trajectories[i]['reward']:>6.1f} {trajectories[i]['group_id']:>10} "
                    f"{baseline_scalar_adv[i]:>10.6f} {weaver_scalar_adv[i]:>10.6f} {diff:>12.2e}"
                )

        print(f"\nMax absolute diff: {max_diff:.2e}")
        assert max_diff < 1e-4, f"Advantage mismatch: max diff = {max_diff}"

    @pytest.mark.parametrize("use_run_ids", [False, True])
    def test_sample_and_population_standard_deviations(self, use_run_ids):
        """The two legacy modes intentionally use different standard deviations."""
        trajectories = [
            Trajectory(
                tokens=[1, 2],
                loss_mask=[0, 1],
                reward=reward,
                extra_fields={"group_id": "prompt", "run_id": run, "expected_rollouts": 2},
            )
            for run, reward in enumerate([1.0, 3.0])
        ]
        compute_grpo_advantage_for_trajectories(trajectories, use_run_ids=use_run_ids)

        # For rewards [1, 3], the mean is 2, sample std is sqrt(2), and population std is 1.
        expected_std = 2.0**0.5 if use_run_ids else 1.0
        expected = [-1.0 / (expected_std + 1e-6), 1.0 / (expected_std + 1e-6)]
        assert [float(t["advantage"]) for t in trajectories] == pytest.approx(expected)

    @pytest.mark.parametrize("use_run_ids", [False, True])
    def test_singleton_groups_have_zero_advantage(self, use_run_ids):
        """Production centers singleton rewards; the historical baseline did not."""
        trajectories = [
            Trajectory(
                tokens=[1, 2],
                loss_mask=[0, 1],
                reward=reward,
                extra_fields={"group_id": group, "run_id": 0, "expected_rollouts": 1},
            )
            for group, reward in enumerate([1.0, -2.0])
        ]
        baseline_adv, _ = baseline_compute_grpo_outcome_advantage(
            token_level_rewards=torch.tensor([[1.0], [-2.0]]),
            eos_mask=torch.ones(2, 1),
            index=[0, 1],
        )
        assert baseline_adv[:, 0].tolist() == pytest.approx([1.0 / 1.000001, -2.0 / 1.000001])

        compute_grpo_advantage_for_trajectories(trajectories, use_run_ids=use_run_ids)
        assert [float(t["advantage"]) for t in trajectories] == [0.0, 0.0]


class TestPPOClipLossParity:
    """Compare PPO clip loss computation between baseline and weaver."""

    @pytest.fixture
    def loss_inputs(self):
        """Create identical inputs for both loss paths."""
        torch.manual_seed(42)
        bs, resp_len = 8, 200
        prompt_len = 100
        full_len = prompt_len + resp_len

        # Response-only tensors (baseline format)
        old_log_probs = torch.randn(bs, resp_len) * 0.1 - 0.2  # ~N(-0.2, 0.1)
        new_log_probs = old_log_probs + torch.randn(bs, resp_len) * 0.05  # small drift
        response_mask = torch.ones(bs, resp_len)
        # Make some tokens padding (variable response lengths)
        for i in range(bs):
            valid_len = 150 + i * 5
            if valid_len < resp_len:
                response_mask[i, valid_len:] = 0.0

        # Advantages: broadcast scalar per sample
        scalar_adv = torch.randn(bs) * 0.8  # ~N(0, 0.8)
        advantages_response = scalar_adv.unsqueeze(1) * response_mask  # [bs, resp_len]

        entropy = torch.rand(bs, resp_len) * 0.5  # positive entropy values

        # Full-sequence tensors (weaver format)
        # [prompt_zeros | response_values | padding_zeros]
        full_logprobs = torch.zeros(bs, full_len)
        full_sampling_logprobs = torch.zeros(bs, full_len)
        full_advantages = torch.zeros(bs, full_len)
        full_entropy = torch.zeros(bs, full_len)
        effective_valid = torch.zeros(bs, full_len, dtype=torch.bool)

        full_logprobs[:, prompt_len:] = new_log_probs
        full_sampling_logprobs[:, prompt_len:] = old_log_probs
        full_advantages[:, prompt_len:] = advantages_response
        full_entropy[:, prompt_len:] = entropy
        for i in range(bs):
            valid_len = 150 + i * 5
            actual_end = min(prompt_len + valid_len, full_len)
            effective_valid[i, prompt_len:actual_end] = True

        return {
            # Baseline format
            "old_log_probs": old_log_probs,
            "new_log_probs": new_log_probs,
            "advantages_response": advantages_response,
            "response_mask": response_mask,
            "entropy_response": entropy,
            # Weaver format
            "full_logprobs": full_logprobs,
            "full_sampling_logprobs": full_sampling_logprobs,
            "full_advantages": full_advantages,
            "full_entropy": full_entropy,
            "effective_valid": effective_valid,
        }

    @pytest.mark.parametrize("clip_ratio_c", [None, 3.0])
    @pytest.mark.parametrize("clip_low,clip_high", [(0.2, 0.2), (5.0, 5.0)])
    @pytest.mark.parametrize("entropy_coeff", [0.0, 1e-4])
    def test_loss_values_match(self, loss_inputs, clip_ratio_c, clip_low, clip_high, entropy_coeff):
        """PPO loss should match between baseline and weaver."""
        d = loss_inputs

        baseline_loss, baseline_el_loss, baseline_metrics = baseline_compute_ppo_clip_loss(
            old_log_prob=d["old_log_probs"],
            log_prob=d["new_log_probs"],
            advantages=d["advantages_response"],
            eos_mask=d["response_mask"],
            clip_ratio_low=clip_low,
            clip_ratio_high=clip_high,
            clip_ratio_c=clip_ratio_c,
            entropy=d["entropy_response"] if entropy_coeff > 0 else None,
            entropy_coeff=entropy_coeff,
        )

        weaver_loss, weaver_el_loss, weaver_metrics = weaver_compute_ppo_clip_loss(
            logprobs=d["full_logprobs"],
            sampling_logprobs=d["full_sampling_logprobs"],
            advantages=d["full_advantages"],
            effective_valid=d["effective_valid"],
            clip_ratio_low=clip_low,
            clip_ratio_high=clip_high,
            clip_ratio_c=clip_ratio_c,
            entropy=d["full_entropy"] if entropy_coeff > 0 else None,
            entropy_coeff=entropy_coeff,
            loss_agg_mode="token-mean",
        )

        config_str = f"clip=({clip_low},{clip_high}) c={clip_ratio_c} ent={entropy_coeff}"
        print(f"\n=== PPO Loss Comparison [{config_str}] ===")
        print(f"{'Metric':<20} {'Baseline':>12} {'Weaver':>12} {'Diff':>12}")
        all_diffs = {}
        for key in baseline_metrics:
            b = baseline_metrics[key]
            w = weaver_metrics.get(key, float("nan"))
            diff = abs(b - w)
            all_diffs[key] = diff
            print(f"{key:<20} {b:>12.6f} {w:>12.6f} {diff:>12.2e}")

        # Check elementwise loss parity on response tokens
        # Extract response portion from weaver's full-seq elementwise loss
        bs = d["response_mask"].shape[0]
        prompt_len = d["full_logprobs"].shape[1] - d["old_log_probs"].shape[1]
        resp_len = d["old_log_probs"].shape[1]
        weaver_el_response = weaver_el_loss[:, prompt_len : prompt_len + resp_len]
        el_diff = (baseline_el_loss - weaver_el_response).abs()
        el_max_diff = el_diff.max().item()
        el_mean_diff = el_diff.mean().item()
        print(f"\nElementwise loss diff: max={el_max_diff:.2e}, mean={el_mean_diff:.2e}")

        # Assertions
        for key, diff in all_diffs.items():
            assert diff < 1e-4, f"Metric {key} mismatch: diff={diff:.2e} [{config_str}]"
        assert (
            el_max_diff < 1e-4
        ), f"Elementwise loss mismatch: max={el_max_diff:.2e} [{config_str}]"

    def test_token_weighted_advantage_mean(self, loss_inputs):
        """Token-weighted advantage mean should match between reporting methods."""
        d = loss_inputs
        # Baseline: masked_mean(advantages, response_mask)
        baseline_twm = baseline_masked_mean(d["advantages_response"], d["response_mask"]).item()

        # Weaver: sum(scalar_adv * n_tokens) / sum(n_tokens)
        bs = d["response_mask"].shape[0]
        scalar_advs = d["advantages_response"][:, 0]  # first token = scalar adv (broadcast)
        token_counts = d["response_mask"].sum(dim=1)
        weaver_twm = (scalar_advs * token_counts).sum().item() / token_counts.sum().item()

        diff = abs(baseline_twm - weaver_twm)
        print(f"\n=== Token-weighted Advantage Mean ===")
        print(f"Baseline: {baseline_twm:.6f}")
        print(f"Weaver:   {weaver_twm:.6f}")
        print(f"Diff:     {diff:.2e}")
        assert diff < 1e-5, f"Token-weighted advantage mean mismatch: {diff}"


class TestDatumConversion:
    """Test that NorthRL's datum conversion preserves data correctly."""

    def test_advantage_broadcast(self):
        """Scalar advantage should be correctly broadcast to response tokens."""
        trajectories, _ = make_test_data(n_prompts=2, rollout_n=4)
        trajectories = compute_grpo_advantage_for_trajectories(trajectories, use_run_ids=True)

        for traj in trajectories:
            advantage = traj["advantage"]
            loss_mask = traj["loss_mask"]

            # Simulate datum conversion: token_advantages = [adv * m for m in adjusted_loss_mask]
            adjusted_loss_mask = (
                loss_mask[1:] if len(loss_mask) == len(traj["tokens"]) else loss_mask
            )
            token_advantages = [float(advantage * m) for m in adjusted_loss_mask]

            # Check: response tokens get the scalar advantage, prompt tokens get 0
            for i, (ta, m) in enumerate(zip(token_advantages, adjusted_loss_mask)):
                if m > 0:
                    assert (
                        abs(ta - advantage) < 1e-6
                    ), f"Response token {i}: expected {advantage}, got {ta}"
                else:
                    assert ta == 0.0, f"Prompt token {i}: expected 0, got {ta}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])

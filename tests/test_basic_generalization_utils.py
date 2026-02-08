#!/usr/bin/env python3
"""
Sanity checks for basic generalization helpers.
"""

from main_experiment import summarize_evaluation_results, compute_generalization_error


def test_summarize_and_generalization_error():
    evaluation_results = {
        "prob_opponent_0_p0.10": {"average_reward": 1.0, "cooperation_rate": 0.2},
        "prob_opponent_1_p0.30": {"average_reward": 3.0, "cooperation_rate": 0.6}
    }

    summary = summarize_evaluation_results(evaluation_results)
    assert abs(summary["mean_reward"] - 2.0) < 1e-6
    assert abs(summary["mean_cooperation_rate"] - 0.4) < 1e-6

    baseline = {"mean_reward": 2.0, "mean_cooperation_rate": 0.4}
    target = {"mean_reward": 1.5, "mean_cooperation_rate": 0.1}
    error = compute_generalization_error(baseline, target)

    assert abs(error["reward_delta"] - 0.5) < 1e-6
    assert abs(error["cooperation_delta"] - 0.3) < 1e-6


if __name__ == "__main__":
    test_summarize_and_generalization_error()
    print("✅ Basic generalization utility tests passed")

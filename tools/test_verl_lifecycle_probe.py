from copy import deepcopy

from tools.verl_lifecycle_probe import (
    ANSWER_REGEX, RESPONSE_REGEX, assess_groups, diagnostic_number,
    INTEGER_REPLY_REGEX, build_arithmetic_tasks, digest_json, score_arithmetic_reply,
    summarize_diagnostics, summarize_screening,
)


def group(rewards, *, finish="stop", parsed="7"):
    return {
        "sample_id": "train/0",
        "responses": [
            {"reward": reward, "finish_reason": finish, "parsed_answer": parsed}
            for reward in rewards
        ],
    }


def test_constant_rewards_do_not_admit_an_update():
    for rewards in ([0, 0, 0, 0], [1, 1, 1, 1]):
        report = assess_groups([group(rewards)])
        assert not report["passed"]
        assert not report["next_stage_allowed"]


def test_complete_correct_and_incorrect_answers_admit_the_next_stage():
    report = assess_groups([group([0, 1, 0, 1])])
    assert report["passed"]
    assert report["eligible_group_ids"] == ["train/0"]
    assert report["answer_parse_ratio"] == 1
    assert report["length_limit_ratio"] == 0


def test_truncation_or_parse_failure_cannot_create_an_eligible_group():
    assert not assess_groups([group([0, 1, 0, 1], finish="length")])["passed"]
    assert not assess_groups([group([0, 1, 0, 1], parsed=None)])["passed"]


def test_missing_or_nonfinite_candidates_fail_closed():
    assert not assess_groups([group([0, 1])])["passed"]
    assert not assess_groups([group([0, 1, 0, float("nan")])])["passed"]
    assert not assess_groups([])["passed"]


def test_numeric_diagnostics_do_not_extract_numbers_from_explanations():
    assert diagnostic_number("1,000.0") == diagnostic_number("1000")
    assert diagnostic_number("16.0") == diagnostic_number("16")
    for text in ("8 pounds", "12 / 3 = 4", "NaN", "Infinity", "12,34", None):
        assert diagnostic_number(text) is None


def test_format_and_arithmetic_axes_preserve_original_rewards_and_gate():
    rows = [{
        "sample_id": "train/0", "ground_truth": "16",
        "responses": [
            {"text": "16", "parsed_answer": None, "reward": 0.0,
             "finish_reason": "stop", "token_ids": [1]},
            {"text": "#### 16.0", "parsed_answer": "16.0", "reward": 0.0,
             "finish_reason": "stop", "token_ids": [1, 2]},
            {"text": "7", "parsed_answer": None, "reward": 0.0,
             "finish_reason": "stop", "token_ids": [1]},
            {"text": "8 x 2 = 16", "parsed_answer": None, "reward": 0.0,
             "finish_reason": "stop", "token_ids": [1, 2, 3]},
        ],
    }]
    original = deepcopy(rows)
    report = summarize_diagnostics(rows)
    counts = report["summary"]["counts"]
    assert counts["numeric_correct"] == 2
    assert counts["numeric_wrong"] == 1
    assert counts["numeric_unknown"] == 1
    assert counts["positive_reward"] == 0
    assert report["summary"]["primary_error_counts"]["numeric_representation_mismatch"] == 1
    assert report["groups"][0]["reward_std"] == 0
    assert not report["groups"][0]["eligible"]
    assert not assess_groups(rows)["passed"]
    assert rows == original


def test_group_variation_and_empty_reports_have_explicit_denominators():
    row = group([0, 1, 0, 1])
    row["ground_truth"] = "7"
    report = summarize_diagnostics([row])
    assert report["groups"][0]["reward_std"] == 0.5
    assert report["summary"]["mixed_reward_group_count"] == 1
    assert report["summary"]["eligible_group_ratio"] == 1
    empty = summarize_diagnostics([])
    assert empty["summary"]["numeric_correct_ratio"] is None
    assert empty["summary"]["eligible_group_ratio"] is None


def trace_group(rewards):
    row = group(rewards)
    row["ground_truth"] = "7"
    for reply in row["responses"]:
        reply.update(text="#### 7", token_ids=[1, 2],
                     behavior_log_probs=[-0.5, -0.1], action_mask=[1, 1])
    return row


def test_screening_partitions_groups_and_reports_usable_update_inputs():
    rows = [trace_group([0, 0, 0, 0]), trace_group([1, 1, 1, 1]),
            trace_group([0, 1, 0, 1])]
    original = deepcopy(rows)
    report = summarize_screening(rows)
    assert report["group_states"] == {
        "incomplete_or_invalid": 0, "all_wrong": 1, "all_correct": 1, "mixed_reward": 1,
    }
    assert report["usable_response_count"] == 4
    assert report["complete_group_positive_reward_ratio"] == 0.5
    assert report["valid_trace_token_count"] == 24
    assert report["groups"][-1]["centered_reward_nonzero_count"] == 4
    assert report["groups"][0]["centered_reward_nonzero_count"] == 0
    assert report["recommended_action"] == "prepare_bounded_update"
    assert not report["direct_training_reuse_allowed"]
    assert rows == original


def test_invalid_traces_block_the_recommendation_even_with_mixed_rewards():
    for mutation, reason in [
        ({"behavior_log_probs": None}, "missing_or_empty"),
        ({"behavior_log_probs": [-1]}, "length_mismatch"),
        ({"behavior_log_probs": [float("nan"), -1]}, "invalid_log_probability"),
        ({"behavior_log_probs": [0.1, -1]}, "invalid_log_probability"),
        ({"action_mask": [1, 0]}, "unexpected_action_mask"),
    ]:
        row = trace_group([0, 1, 0, 1])
        row["responses"][0].update(mutation)
        report = summarize_screening([row])
        assert report["trace_checks"][reason] == 1
        assert report["usable_mixed_group_count"] == 0
        assert report["recommended_action"] == "repair_trajectory_evidence"


def test_constraints_do_not_authorize_reusing_probabilities_for_ordinary_ppo():
    row = trace_group([0, 1, 0, 1])
    report = summarize_screening([row], constrained=True)
    assert report["recommended_action"] == "design_matching_policy_update"
    assert report["sampling_distribution"] == "grammar_constrained"
    assert not report["direct_training_reuse_allowed"]
    assert report["canonical_integer_answer_ratio"] == 1
    row["responses"][0]["text"] = "#### 7.0"
    report = summarize_screening([row], constrained=True)
    assert report["group_blocker_counts"]["constraint_not_satisfied"] == 1
    assert report["usable_mixed_group_count"] == 0


def test_all_constant_rewards_and_empty_screening_have_actionable_decisions():
    for rewards in ([0, 0, 0, 0], [1, 1, 1, 1]):
        report = summarize_screening([trace_group(rewards)])
        assert report["recommended_action"] == "review_question_difficulty_before_more_training"
    empty = summarize_screening([])
    assert empty["recommended_action"] == "collect_screening_data"
    assert empty["complete_group_positive_reward_ratio"] is None
    assert empty["canonical_integer_answer_ratio"] is None


def test_format_constraint_does_not_encode_the_expected_answer():
    import re

    for value in ("0", "7", "16", "-100", "987654321"):
        assert re.fullmatch(ANSWER_REGEX, f"#### {value}")
    for text in ("7", "#### 07", "#### 7.0", "#### 7\n", "answer: #### 7"):
        assert not re.fullmatch(ANSWER_REGEX, text)


def test_short_calculation_grammar_bounds_lines_and_keeps_one_final_answer():
    import re

    for text in ("#### 7", "12 / 3 = 4\n#### 4", "48 / 4 = 12\n12 + 15 = 27\n#### 27",
                 "x" * 96 + "\n#### -7"):
        assert re.fullmatch(RESPONSE_REGEX, text)
    for text in ("x" * 97 + "\n#### 7", "a\nb\nc\n#### 7", "#### 7\nmore",
                 "#### 1\n#### 7", "answer\n#### 7.0", "answer\n\n#### 7"):
        assert not re.fullmatch(RESPONSE_REGEX, text)


def test_calculation_diagnostics_use_the_recorded_grammar_not_latest_default():
    row = trace_group([0, 1, 0, 1])
    row["responses"][0]["text"] = "3 + 4 = 7\n#### 7"
    original = deepcopy(row)
    new = summarize_screening([row], constrained=True, response_regex=RESPONSE_REGEX)
    assert new["output_contract_match_count"] == 4
    assert new["calculation_response_count"] == 1
    assert new["canonical_integer_answer_count"] == 3  # Answer-only legacy diagnostic.
    assert new["usable_mixed_group_count"] == 1
    assert not new["direct_training_reuse_allowed"]
    old = summarize_screening([row], constrained=True, response_regex=ANSWER_REGEX)
    assert old["output_contract_match_count"] == 3
    assert old["usable_mixed_group_count"] == 0
    assert row == original


def test_arithmetic_pool_is_fixed_balanced_and_independently_scored():
    tasks = build_arithmetic_tasks()
    assert len(tasks) == 8
    assert len({t["sample_id"] for t in tasks}) == 8
    assert [t["ground_truth"] for t in tasks] == ["12", "15", "42", "8", "95", "34", "102", "12"]
    assert {op: sum(t["operation"] == op for t in tasks)
            for op in ("add", "subtract", "multiply", "divide")} == {
                "add": 2, "subtract": 2, "multiply": 2, "divide": 2,
            }
    for task in tasks:
        assert task["split"] == "mechanism_probe"
        assert score_arithmetic_reply(task["ground_truth"], task["ground_truth"])[1] == 1
        assert score_arithmetic_reply(str(int(task["ground_truth"]) + 1), task["ground_truth"])[1] == 0
    assert digest_json(tasks) == digest_json(build_arithmetic_tasks())


def test_arithmetic_verifier_scores_only_whole_integer_replies():
    assert score_arithmetic_reply(" 12\n", "12") == ("12", 1.0)
    assert score_arithmetic_reply("-12", "-12") == ("-12", 1.0)
    assert score_arithmetic_reply("13", "12") == ("13", 0.0)
    for text in ("#### 12", "7 + 5 = 12", "12 units", "12.0", "012", "", "NaN", "1\n2"):
        assert score_arithmetic_reply(text, "12") == (None, 0.0)


def test_unconstrained_arithmetic_trace_requires_the_same_complete_group_checks():
    row = trace_group([0, 1, 0, 1])
    for reply in row["responses"]:
        reply["text"] = "7"
    report = summarize_screening([row], response_regex=INTEGER_REPLY_REGEX)
    assert report["sampling_distribution"] == "unconstrained"
    assert report["output_contract_match_count"] == 4
    assert report["usable_mixed_group_count"] == 1
    assert report["recommended_action"] == "prepare_bounded_update"
    row["responses"][0]["finish_reason"] = "length"
    assert not assess_groups([row])["passed"]
    assert summarize_screening([row], response_regex=INTEGER_REPLY_REGEX)["usable_response_count"] == 0

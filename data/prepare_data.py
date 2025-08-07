#!/usr/bin/env python3
import argparse
import json
import logging
import os
from collections import defaultdict

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def convert_jsonl_to_parquet(input_file, output_file):
    """
    Convert toy filter JSONL file to Parquet format with structure matching openr1.jsonl
    Group samples by hash_problem and combine them into a single entry with multiple targets
    """
    print(f"Converting {input_file} to {output_file}...")

    # Read the JSONL file and group by hash_problem
    grouped_samples = defaultdict(list)
    with open(input_file, "r") as f:
        for line_idx, line in enumerate(f):
            item = json.loads(line)
            hash_problem = item.get("metadata", {}).get("hash_problem")

            if not hash_problem:
                logger.warning(
                    f"Sample at line {line_idx+1} doesn't have hash_problem, treating as unique"
                )
                hash_problem = f"unique_{line_idx}"

            grouped_samples[hash_problem].append((line_idx, item))

    logger.info(f"Found {len(grouped_samples)} unique problems")

    # Process each group
    data = []
    for hash_problem, samples in grouped_samples.items():
        # Check if all user prompts are the same
        user_prompts = []
        for _, item in samples:
            for conv in item.get("conversations", []):
                if conv.get("from") == "user":
                    user_prompts.append(conv.get("value"))
                    break

        if len(set(user_prompts)) > 1:
            logger.warning(
                f"Different user prompts found for hash_problem {hash_problem}"
            )
            for idx, (line_idx, _) in enumerate(samples):
                logger.warning(f"  Sample {idx+1} at line {line_idx+1}")

        # Get the first item's user prompt
        first_sample = samples[0][1]
        user_prompt = None
        for conv in first_sample.get("conversations", []):
            if conv.get("from") == "user":
                user_prompt = conv.get("value")
                break

        if not user_prompt:
            logger.warning(
                f"No user prompt found in first sample for hash_problem {hash_problem}"
            )
            continue

        # Collect all assistant responses and their correctness
        assistant_responses = []
        target_rewards = []
        ground_truth = first_sample.get("label", "")

        for _, item in samples:
            response = None
            for conv in item.get("conversations", []):
                if conv.get("from") == "assistant":
                    response = conv.get("value")
                    break

            if response:
                assistant_responses.append(response)

                # Check if this response is correct
                # You may need to adjust this based on your data format
                is_correct = item.get(
                    "is_majority", False
                )  # Try to get an explicit correctness flag

                # Store reward value (1 for correct, 0 for incorrect)
                target_rewards.append(1 if is_correct else 0)

        if not assistant_responses:
            logger.warning(
                f"No assistant responses found for hash_problem {hash_problem}"
            )
            continue

        # Manually construct JSON strings to avoid escaping issues
        prompt_json = [{"content": user_prompt, "role": "user"}]

        # Create a list of targets
        target_json_list = []
        for response in assistant_responses:
            target_json = [{"content": response, "role": "assistant"}]
            target_json_list.append(target_json)

        # Reorder target_json_list and target_rewards to alternate 1, 0, 1, 0...
        def reorder_by_alternating_rewards(target_rewards, target_json_list):
            """
            Reorder targets to alternate between positive(1) and negative(0) rewards.
            Pattern: 1, 0, 1, 0, 1, 0...
            If no more negatives available, fill remaining with positives.
            """
            # Separate positive and negative samples with their indices
            positive_samples = [
                (i, reward, target)
                for i, (reward, target) in enumerate(
                    zip(target_rewards, target_json_list)
                )
                if reward == 1
            ]
            negative_samples = [
                (i, reward, target)
                for i, (reward, target) in enumerate(
                    zip(target_rewards, target_json_list)
                )
                if reward == 0
            ]

            reordered_rewards = []
            reordered_targets = []

            pos_idx = 0
            neg_idx = 0

            # Alternate starting with positive (1)
            for position in range(len(target_rewards)):
                if position % 2 == 0:  # Even positions (0, 2, 4...) should be positive
                    if pos_idx < len(positive_samples):
                        _, reward, target = positive_samples[pos_idx]
                        reordered_rewards.append(reward)
                        reordered_targets.append(target)
                        pos_idx += 1
                    elif neg_idx < len(
                        negative_samples
                    ):  # No more positives, use negatives
                        _, reward, target = negative_samples[neg_idx]
                        reordered_rewards.append(reward)
                        reordered_targets.append(target)
                        neg_idx += 1
                else:  # Odd positions (1, 3, 5...) should be negative
                    if neg_idx < len(negative_samples):
                        _, reward, target = negative_samples[neg_idx]
                        reordered_rewards.append(reward)
                        reordered_targets.append(target)
                        neg_idx += 1
                    elif pos_idx < len(
                        positive_samples
                    ):  # No more negatives, use positives
                        _, reward, target = positive_samples[pos_idx]
                        reordered_rewards.append(reward)
                        reordered_targets.append(target)
                        pos_idx += 1

            return reordered_rewards, reordered_targets

        # Apply reordering
        target_rewards, target_json_list = reorder_by_alternating_rewards(
            target_rewards, target_json_list
        )

        # Create a new format similar to openr1.jsonl with multiple targets
        reformatted_item = {
            "data_source": "filter_all_voting",
            "prompt": prompt_json,
            "target": target_json_list[0],  # First target for backward compatibility
            "target_lst": target_json_list,  # List of all targets
            "target_rewards": target_rewards,  # List of rewards for each target (1=correct, 0=incorrect)
            "ability": "",
            "reward_model": {"ground_truth": ground_truth, "style": "rule"},
            "extra_info": {
                "index": -1,
                "split": "default",
                "hash_problem": hash_problem,
                "num_responses": len(assistant_responses),
            },
        }
        data.append(reformatted_item)

    # Convert to DataFrame and save as Parquet
    df = pd.DataFrame(data)
    df.to_parquet(output_file, index=False)
    print(
        f'Successfully converted {len(data)} problems with a total of {sum(len(item["target_lst"]) for item in data)} responses to parquet format'
    )

    # Print statistics about rewards
    total_targets = sum(len(item["target_lst"]) for item in data)
    correct_targets = sum(sum(item["target_rewards"]) for item in data)
    print(
        f"Reward statistics: {correct_targets}/{total_targets} targets are correct ({(correct_targets/total_targets)*100:.2f}%)"
    )

    # Print reordering statistics
    alternating_patterns = 0
    perfect_alternating = 0
    for item in data:
        rewards = item["target_rewards"]
        if len(rewards) > 1:
            # Check if it follows alternating pattern (starting with 1)
            expected_pattern = [1 if i % 2 == 0 else 0 for i in range(len(rewards))]
            # Allow filling with 1s if no more 0s available
            actual_follows_pattern = True
            for i, (actual, expected) in enumerate(zip(rewards, expected_pattern)):
                if expected == 0 and actual == 1:
                    # This is allowed (no more negatives, filled with positives)
                    continue
                elif actual != expected:
                    actual_follows_pattern = False
                    break

            if actual_follows_pattern:
                alternating_patterns += 1
                if rewards == expected_pattern:
                    perfect_alternating += 1

    print(
        f"Reordering statistics: {alternating_patterns}/{len(data)} problems follow alternating pattern"
    )
    print(
        f"Perfect alternating (1,0,1,0...): {perfect_alternating}/{len(data)} problems"
    )

    return len(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert JSONL to Parquet for training"
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Input JSONL file path"
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Output Parquet file path"
    )

    args = parser.parse_args()
    convert_jsonl_to_parquet(args.input_file, args.output_file)

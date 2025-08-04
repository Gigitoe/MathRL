# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import os

import datasets
from datasets import Dataset, DatasetDict

from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", default="./data/MATH-Train-Levels-3-To-5.parquet")
    parser.add_argument("--translator", action="store_true", 
                        help="Prepend solution to question and use translator dataset name")
    args = parser.parse_args()

    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = "DigitalLearningGmbH/MATH-lighteval"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # Add split information to each dataset
    train_dataset = train_dataset.map(lambda x: {"split": "train"})
    test_dataset = test_dataset.map(lambda x: {"split": "test"})

    # Combine train and test datasets
    combined_dataset = datasets.concatenate_datasets([train_dataset, test_dataset])

    # Filter for levels 3 to 5
    filtered_dataset = combined_dataset.filter(
        lambda x: x["level"] in ["Level 3", "Level 4", "Level 5"]
    )
    
    # Convert level strings to integers
    def convert_level(example):
        example["level"] = int(example["level"].split()[-1])
        return example
        
    filtered_dataset = filtered_dataset.map(convert_level)

    def process_fn(example, idx):
        question = example.pop("problem")
        solution = example.pop("solution")
        answer = extract_solution(solution)
        level = example.pop("level")
        subject = example.pop("type")
        split = example.pop("split")
        
        # Apply translator modifications if enabled
        if args.translator:
            modified_question = f"<solution>\n{solution}\n</solution>\n\n{question}"
            source_name = "MATH-Train-Levels-3-To-5-Translator"
        else:
            modified_question = question
            source_name = "MATH-Train-Levels-3-To-5"
        
        data = {
            "data_source": source_name,
            "prompt": [{"role": "user", "content": modified_question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "index": idx,
                "level": level,
                "subject": subject,
                "split": split,
                "problem": question,
                "solution": solution
            },
        }
        return data

    processed_dataset = filtered_dataset.map(function=process_fn, with_indices=True)

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save to specified file
    processed_dataset.to_parquet(args.output_file)
    print(f"Saved processed dataset to {args.output_file}")
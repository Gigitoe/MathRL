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
Preprocess the HuggingFaceH4/MATH-500 dataset to parquet format
"""

import argparse
import os

import datasets
from datasets import Dataset, DatasetDict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", default="./data/MATH500.parquet")
    args = parser.parse_args()

    # Load HuggingFaceH4/MATH-500 dataset
    data_source = "HuggingFaceH4/MATH-500"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)
    
    # This dataset only has a 'test' split
    test_dataset = dataset["test"]

    def process_fn(example, idx):
        question = example.pop("problem")
        solution = example.pop("solution")
        answer = example.pop("answer")
        level = example.pop("level")
        subject = example.pop("subject")
        
        # Remove unique_id if it exists
        example.pop("unique_id", None)
        
        data = {
            "data_source": "MATH500",
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "index": idx,
                "level": level,
                "subject": subject,
                "split": "test",
                "problem": question,
                "solution": solution
            },
        }
        return data

    processed_dataset = test_dataset.map(function=process_fn, with_indices=True)

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save to specified file
    processed_dataset.to_parquet(args.output_file)
    print(f"Saved processed dataset to {args.output_file}")
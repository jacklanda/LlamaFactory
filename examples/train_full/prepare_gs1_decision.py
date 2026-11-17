# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Validate a sampled GS1 train set and register it as one decision dataset."""

import argparse
import json
from pathlib import Path


def prepare(sample, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = questions = 0
    with Path(sample).open(encoding="utf-8") as source, (output / "train.jsonl").open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("partition", "train") != "train":
                raise ValueError(f"Only train records may be used (line {line_number}).")
            request = row.get("request", {})
            question_ids = set(request.get("questions", {}))
            if not question_ids or question_ids != set(row.get("targets", {})):
                raise ValueError(f"Question and target IDs differ or are empty (line {line_number}).")
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            records += 1
            questions += len(question_ids)
    if not records:
        raise ValueError("The sampled training file is empty.")
    registry = {"gs1_train": {"file_name": "train.jsonl", "formatting": "gs1"}}
    (output / "dataset_info.json").write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    return {"records": records, "questions": questions}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path)
    parser.add_argument("output", type=Path)
    print(prepare(**vars(parser.parse_args())))

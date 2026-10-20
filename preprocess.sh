#!/bin/bash

# Input JSONL file path for the English (en-US) MASSIVE dataset
INPUT_FILE="1.0/data/en-US.jsonl"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: $INPUT_FILE not found. Please ensure the dataset is extracted."
    exit 1
fi

# Run Python to parse the JSONL and split into train, dev, and test CSV files
python3 - << 'EOF'
import json
import csv

input_file = "1.0/data/en-US.jsonl"
splits = {"train": [], "dev": [], "test": []}

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        partition = data.get("partition")
        if partition in splits:
            splits[partition].append(data)

# Collect all unique keys across records to form consistent CSV headers
fieldnames = set()
for split_data in splits.values():
    for row in split_data:
        fieldnames.update(row.keys())
fieldnames = sorted(list(fieldnames))

# Write each split to its respective CSV file
for split_name, rows in splits.items():
    output_file = f"{split_name}.csv"
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Successfully wrote {len(rows)} records to {output_file}")
EOF

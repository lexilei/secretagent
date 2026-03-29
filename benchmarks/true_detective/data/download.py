#!/usr/bin/env python3
"""Download and prepare the True Detective dataset.

Source: https://github.com/MaksymDel/true-detective
Paper: "True Detective: A Deep Abductive Reasoning Benchmark" (*SEM 2023)

The dataset contains 191 detective mystery puzzles with multiple-choice answers.
"""

import csv
import io
import json
import re
import zipfile
from pathlib import Path
from urllib.request import urlopen

DATA_DIR = Path(__file__).parent
REPO_URL = "https://github.com/MaksymDel/true-detective/archive/refs/heads/main.zip"


def download_and_prepare():
    print("Downloading True Detective dataset...")
    response = urlopen(REPO_URL)
    zip_data = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        # Find the data.zip inside the repo
        data_zip_name = None
        for name in zf.namelist():
            if name.endswith('data.zip'):
                data_zip_name = name
                break

        if data_zip_name is None:
            raise FileNotFoundError("data.zip not found in repository")

        # Extract data.zip, then extract the CSV from it
        with zf.open(data_zip_name) as data_zip_file:
            inner_zip = zipfile.ZipFile(io.BytesIO(data_zip_file.read()))
            csv_name = [n for n in inner_zip.namelist() if n.endswith('.csv')][0]
            csv_text = inner_zip.read(csv_name).decode('utf-8')

    # Parse CSV and convert to secretagent format
    reader = csv.DictReader(io.StringIO(csv_text))
    examples = []

    for row in reader:
        # Parse answer options: "(a) Alice; (b) Bob; (c) Carol; (d) Dave"
        options_raw = row['answer_options']
        choices = re.findall(r'\([a-z]\)\s*([^;]+)', options_raw)
        choices = [c.strip() for c in choices]

        # Parse correct answer: "(b) Bob" -> index 1
        answer_raw = row['answer']
        answer_match = re.match(r'\(([a-z])\)', answer_raw)
        answer_index = ord(answer_match.group(1)) - ord('a')  # letter to 0-based index

        examples.append({
            'case_name': row['case_name'],
            'narrative': row['mystery_text'],
            'question': 'Who is guilty?',
            'choices': choices,
            'answer_index': answer_index,
            'answer_text': row['answer'],
            'golden_cot': row.get('outcome', ''),
        })

    # Save as JSON
    output = {'examples': examples}
    output_path = DATA_DIR / 'true_detective.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved {len(examples)} puzzles to {output_path}")
    from collections import Counter
    dist = Counter(len(e['choices']) for e in examples)
    print(f"  Choice distribution: {dict(sorted(dist.items()))}")

    # Also verify a few
    print(f"\nSample puzzle: {examples[0]['case_name']}")
    print(f"  Choices: {examples[0]['choices']}")
    print(f"  Answer index: {examples[0]['answer_index']}")


if __name__ == '__main__':
    download_and_prepare()

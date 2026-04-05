"""Pre-split MUSR murder mysteries into train/val/test.

Split: shuffle once with seed=42, then 75 train / 75 val / 100 test.
The split is fixed — all experiments use the same shuffled split.

Usage:
    uv run python experiments/induction/data_split.py
"""

import ast
import json
import random
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_FILE = _PROJECT_ROOT / 'benchmarks' / 'musr' / 'data' / 'murder_mysteries.json'
OUTPUT_DIR = Path(__file__).parent / 'data'


def main():
    with open(DATA_FILE) as f:
        data = json.load(f)

    cases = []
    for i, ex in enumerate(data['examples']):
        choices = ex['choices']
        if isinstance(choices, str):
            choices = ast.literal_eval(choices)
        cases.append({
            'name': f'ex{i:03d}',
            'narrative': ex['narrative'],
            'question': ex['question'],
            'choices': choices,
            'expected': ex['answer_index'],
        })

    random.Random(42).shuffle(cases)

    train = cases[:75]
    val = cases[75:150]
    test = cases[150:250]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in [('train', train), ('val', val), ('test', test)]:
        path = OUTPUT_DIR / f'{name}.json'
        with open(path, 'w') as f:
            json.dump(split, f, indent=2)
        print(f'{name}: {len(split)} cases -> {path}')


if __name__ == '__main__':
    main()

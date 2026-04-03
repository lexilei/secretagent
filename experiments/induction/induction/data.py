"""Data loading for induction experiments."""

import ast
import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'


def load_murder_cases(n: int = 75, seed: int = 42) -> list[dict]:
    """Load and shuffle murder mystery cases (legacy, no split)."""
    data_file = _PROJECT_ROOT / 'benchmarks' / 'musr' / 'data' / 'murder_mysteries.json'
    with open(data_file) as f:
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
    import random
    random.Random(seed).shuffle(cases)
    return cases[:n]


def load_split(split: str) -> list[dict]:
    """Load a pre-split data file (train.json, val.json, test.json)."""
    path = DATA_DIR / f'{split}.json'
    if not path.exists():
        raise FileNotFoundError(f'{path} not found. Run data_split.py first.')
    with open(path) as f:
        return json.load(f)


def load_cases(cfg) -> dict:
    """Load data based on config. Returns {'train': [...], 'val': [...]}."""
    mode = cfg.get('data', {}).get('mode', 'split')
    if mode == 'split':
        return {
            'train': load_split('train'),
            'val': load_split('val'),
            'test': load_split('test'),
        }
    else:
        cases = load_murder_cases(
            n=cfg.get('data', {}).get('n_cases', 75),
            seed=cfg.get('data', {}).get('seed', 42),
        )
        return {'train': cases, 'val': None, 'test': None}

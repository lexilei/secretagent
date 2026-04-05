"""Data loading for induction experiments."""

import ast
import json
import random
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'

# Benchmark data files relative to _PROJECT_ROOT
_BENCHMARK_FILES = {
    'murder': 'benchmarks/musr/data/murder_mysteries.json',
    'object': 'benchmarks/musr/data/object_placements.json',
    'team': 'benchmarks/musr/data/team_allocation.json',
    'true_detective': 'benchmarks/true_detective/data/true_detective.json',
}


def load_benchmark(benchmark: str, n: int = 0, seed: int = 42) -> list[dict]:
    """Load any supported benchmark. Returns list of case dicts.

    benchmark: 'murder', 'object', 'team', 'true_detective'
    n: max cases (0 = all)
    """
    data_file = _PROJECT_ROOT / _BENCHMARK_FILES[benchmark]
    with open(data_file) as f:
        data = json.load(f)
    examples = data['examples']
    cases = []
    for i, ex in enumerate(examples):
        choices = ex['choices']
        if isinstance(choices, str):
            choices = ast.literal_eval(choices)
        cases.append({
            'name': f'{benchmark}_{i:03d}',
            'narrative': ex['narrative'],
            'question': ex['question'],
            'choices': choices,
            'expected': ex['answer_index'],
        })
    random.Random(seed).shuffle(cases)
    return cases[:n] if n > 0 else cases


def load_murder_cases(n: int = 75, seed: int = 42) -> list[dict]:
    """Load and shuffle murder mystery cases (legacy, no split)."""
    return load_benchmark('murder', n=n, seed=seed)


def split_cases(cases: list[dict], train_frac: float = 0.3,
                val_frac: float = 0.3, seed: int = 42) -> dict:
    """Split cases into train/val/test."""
    rng = random.Random(seed)
    shuffled = list(cases)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return {
        'train': shuffled[:n_train],
        'val': shuffled[n_train:n_train + n_val],
        'test': shuffled[n_train + n_val:],
    }


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
    benchmark = cfg.get('data', {}).get('benchmark', 'murder')

    if mode == 'split' and benchmark == 'murder':
        # Use pre-existing splits for murder mysteries (backward compat)
        return {
            'train': load_split('train'),
            'val': load_split('val'),
            'test': load_split('test'),
        }
    elif mode == 'split':
        # Auto-split other benchmarks
        all_cases = load_benchmark(benchmark)
        return split_cases(all_cases, seed=cfg.get('data', {}).get('seed', 42))
    else:
        cases = load_benchmark(
            benchmark,
            n=cfg.get('data', {}).get('n_cases', 75),
            seed=cfg.get('data', {}).get('seed', 42),
        )
        return {'train': cases, 'val': None, 'test': None}

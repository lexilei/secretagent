"""True Detective benchmark experiment.

Paper: "True Detective: A Deep Abductive Reasoning Benchmark" (*SEM 2023)
191 detective mystery puzzles with multiple-choice answers.

Example CLI commands:

    # download data first
    uv run python benchmarks/true_detective/data/download.py

    # run with default config (ptool workflow)
    uv run python benchmarks/true_detective/expt.py run --config-file conf/workflow.yaml

    # run first 20 examples (smoke test / pilot)
    uv run python benchmarks/true_detective/expt.py run --config-file conf/workflow.yaml dataset.n=20

    # zero-shot baseline
    uv run python benchmarks/true_detective/expt.py run --config-file conf/zeroshot.yaml
"""

import ast
import json
import sys
import pandas as pd
from pathlib import Path
from typing import Any

_BENCHMARK_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BENCHMARK_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))
sys.path.insert(0, str(_BENCHMARK_DIR))

import typer

from secretagent import config
from secretagent.core import implement_via_config
from secretagent.dataset import Dataset, Case
from secretagent.evaluate import Evaluator
import secretagent.implement_pydantic  # noqa: F401
import secretagent.implement_ptp  # noqa: F401


class TrueDetectiveEvaluator(Evaluator):
    def compare_predictions(self, predicted_output, expected_output) -> dict[str, Any]:
        return dict(correct=float(predicted_output == expected_output))


def load_dataset(split: str) -> Dataset:
    json_file = _BENCHMARK_DIR / 'data' / f'{split}.json'
    with open(json_file) as f:
        data = json.load(f)

    cases = []
    for i, ex in enumerate(data['examples']):
        choices = ex['choices']
        if isinstance(choices, str):
            choices = ast.literal_eval(choices)
        cases.append(Case(
            name=ex.get('case_name', f'ex{i:03d}'),
            input_args=(ex['narrative'], ex.get('question', 'Who is guilty?'), choices),
            expected_output=ex['answer_index'],
            metadata={
                'answer_text': ex.get('answer_text', ''),
                'golden_cot': ex.get('golden_cot', ''),
            },
        ))

    return Dataset(name='true_detective', split=split, cases=cases)


app = typer.Typer()

@app.callback()
def callback():
    """True Detective benchmark."""

@app.command(context_settings={"allow_extra_args": True, "allow_interspersed_args": False})
def run(ctx: typer.Context,
        config_file: str = typer.Option(..., help="Config YAML file")):
    """Run True Detective evaluation. Extra args are config overrides."""

    cfg_path = Path(config_file)
    if not cfg_path.is_absolute():
        cfg_path = _BENCHMARK_DIR / cfg_path
    config.configure(yaml_file=str(cfg_path), dotlist=ctx.args)
    config.set_root(_BENCHMARK_DIR)

    import ptools
    implement_via_config(ptools, config.require('ptools'))

    split = config.require('dataset.split')
    dataset = load_dataset(split).configure(
        shuffle_seed=config.get('dataset.shuffle_seed'),
        n=config.get('dataset.n'))
    print('dataset:', dataset.summary())

    entry_point = config.require('evaluate.entry_point')
    interface = getattr(ptools, entry_point)

    evaluator = TrueDetectiveEvaluator()
    csv_path = evaluator.evaluate(dataset, interface)
    df = pd.read_csv(csv_path)
    print(f'\nAccuracy: {df["correct"].mean():.1%} ({df["correct"].sum():.0f}/{len(df)})')


if __name__ == '__main__':
    app()

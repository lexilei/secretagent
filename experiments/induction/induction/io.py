"""Saving ptools, traces, and formatting for prompts."""

import json
from pathlib import Path


def save_ptool(ptool: dict, ptools_dir: Path):
    ptools_dir.mkdir(parents=True, exist_ok=True)
    path = ptools_dir / f'{ptool["id"]}.json'
    with open(path, 'w') as f:
        json.dump(ptool, f, indent=2)
    print(f'  Saved ptool to {path}')


def save_ptools_as_python(ptools: list[dict], output_path: Path):
    """Write all induced ptools as a readable .py file."""
    lines = [
        '"""Auto-induced ptools from iterative induction pipeline."""',
        '', 'from secretagent.core import interface', '',
    ]
    for p in ptools:
        lines.append('@interface')
        lines.append(f'def {p["func_name"]}(narrative: str, focus: str) -> str:')
        doc_lines = p['doc'].strip().split('\n')
        lines.append(f'    """{doc_lines[0]}')
        for dl in doc_lines[1:]:
            lines.append(f'    {dl}')
        lines.append('    """')
        lines.append('')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def save_sample_traces(traces: list[dict], iter_path: Path, n: int = 10):
    successes = [t for t in traces if t.get('correct')]
    failures = [t for t in traces if not t.get('correct')]
    sample = {'successes': successes[:n], 'failures': failures[:n]}
    with open(iter_path / 'sample_traces.json', 'w') as f:
        json.dump(sample, f, indent=2, default=str)
    return sample


def format_ptool_actions_for_prompt(ptools: list[dict]) -> str:
    lines = []
    for i, p in enumerate(ptools, start=2):
        lines.append(
            f'({i}) {p["display_name"]}[focus], which calls the tool '
            f'`{p["func_name"]}(narrative, focus)` — {p["short_desc"]}'
        )
    return '\n'.join(lines)

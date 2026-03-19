import json
import pytest
from pathlib import Path

from omegaconf import OmegaConf

from secretagent import config
from secretagent.dataset import Dataset
from secretagent.learn.utils import (
    collect_interface_data,
    _extract_cases_from_record,
    _extract_cases_from_dirs,
)


@pytest.fixture(autouse=True)
def clean_config():
    """Reset config before and after each test."""
    saved = config.GLOBAL_CONFIG.copy()
    yield
    config.GLOBAL_CONFIG = saved


def _make_recording_dir(base, name, cfg_dict, records):
    """Create a fake recording directory with config.yaml and results.jsonl."""
    d = base / name
    d.mkdir()
    with open(d / 'config.yaml', 'w') as f:
        f.write(OmegaConf.to_yaml(OmegaConf.create(cfg_dict)))
    with open(d / 'results.jsonl', 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')
    return d


SAMPLE_RECORDS = [
    {
        'predicted_output': True,
        'expected_output': True,
        'rollout': [
            {'func': 'analyze', 'args': ['hello'], 'kw': {}, 'output': ['hello']},
            {'func': 'classify', 'args': ['hello'], 'kw': {}, 'output': True},
        ],
    },
    {
        'predicted_output': False,
        'expected_output': False,
        'rollout': [
            {'func': 'analyze', 'args': ['goodbye'], 'kw': {}, 'output': ['goodbye']},
            {'func': 'classify', 'args': ['goodbye'], 'kw': {}, 'output': False},
        ],
    },
]


# --- _extract_cases_from_record tests ---

def test_extract_cases_from_record_filters_by_interface():
    cases = list(_extract_cases_from_record(0, 0, 'classify', SAMPLE_RECORDS[0]))
    assert len(cases) == 1
    assert cases[0].input_args == ['hello']
    assert cases[0].expected_output is True


def test_extract_cases_from_record_skips_other_interfaces():
    cases = list(_extract_cases_from_record(0, 0, 'nonexistent', SAMPLE_RECORDS[0]))
    assert len(cases) == 0


def test_extract_cases_from_record_names_include_indices():
    cases = list(_extract_cases_from_record(2, 3, 'analyze', SAMPLE_RECORDS[0]))
    assert cases[0].name == 'analyze_2.3.0'


def test_extract_cases_from_record_empty_rollout():
    record = {'rollout': []}
    cases = list(_extract_cases_from_record(0, 0, 'classify', record))
    assert cases == []


def test_extract_cases_from_record_missing_rollout():
    record = {}
    cases = list(_extract_cases_from_record(0, 0, 'classify', record))
    assert cases == []


def test_extract_cases_from_record_kw_none_when_empty():
    cases = list(_extract_cases_from_record(0, 0, 'classify', SAMPLE_RECORDS[0]))
    assert cases[0].input_kw is None


def test_extract_cases_from_record_preserves_kw():
    record = {
        'rollout': [
            {'func': 'f', 'args': [], 'kw': {'x': 1}, 'output': 'ok'},
        ],
    }
    cases = list(_extract_cases_from_record(0, 0, 'f', record))
    assert cases[0].input_kw == {'x': 1}


# --- _extract_cases_from_dirs tests ---

def test_extract_cases_from_dirs_single_dir(tmp_path):
    d = _make_recording_dir(tmp_path, '20260101.120000.expt', {'llm': {'model': 'a'}}, SAMPLE_RECORDS)
    cases = _extract_cases_from_dirs([d], 'classify')
    assert len(cases) == 2
    assert cases[0].expected_output is True
    assert cases[1].expected_output is False


def test_extract_cases_from_dirs_multiple_dirs(tmp_path):
    d1 = _make_recording_dir(tmp_path, '20260101.120000.expt', {'llm': {'model': 'a'}}, SAMPLE_RECORDS)
    d2 = _make_recording_dir(tmp_path, '20260102.120000.expt', {'llm': {'model': 'b'}}, SAMPLE_RECORDS)
    cases = _extract_cases_from_dirs([d1, d2], 'classify')
    assert len(cases) == 4


def test_extract_cases_from_dirs_missing_jsonl(tmp_path):
    d = tmp_path / '20260101.120000.expt'
    d.mkdir()
    with pytest.raises(ValueError, match='missing jsonl file'):
        _extract_cases_from_dirs([d], 'classify')


# --- collect_interface_data tests ---

def test_collect_creates_data_json(tmp_path):
    src = _make_recording_dir(tmp_path, '20260101.120000.expt', {'llm': {'model': 'a'}}, SAMPLE_RECORDS)
    train_dir = tmp_path / 'train'
    train_dir.mkdir()
    config.configure(cfg={'train_dir': str(train_dir)})

    out_dir, dataset = collect_interface_data([src], 'classify', file_under='test')

    assert (out_dir / 'data.json').exists()
    loaded = Dataset.model_validate_json((out_dir / 'data.json').read_text())
    assert loaded.name == 'classify'
    assert len(loaded.cases) == 2


def test_collect_creates_sources_txt(tmp_path):
    src = _make_recording_dir(tmp_path, '20260101.120000.expt', {'llm': {'model': 'a'}}, SAMPLE_RECORDS)
    train_dir = tmp_path / 'train'
    train_dir.mkdir()
    config.configure(cfg={'train_dir': str(train_dir)})

    out_dir, _ = collect_interface_data([src], 'classify', file_under='test')

    sources = (out_dir / 'sources.txt').read_text().strip().split('\n')
    assert len(sources) == 1
    assert sources[0] == str(src)


def test_collect_copies_source_configs(tmp_path):
    src = _make_recording_dir(tmp_path, '20260101.120000.expt', {'llm': {'model': 'test-model'}}, SAMPLE_RECORDS)
    train_dir = tmp_path / 'train'
    train_dir.mkdir()
    config.configure(cfg={'train_dir': str(train_dir)})

    out_dir, _ = collect_interface_data([src], 'classify', file_under='test')

    cfg_dir = out_dir / 'source_configs'
    assert cfg_dir.exists()
    copied = cfg_dir / '20260101.120000.expt.yaml'
    assert copied.exists()
    loaded_cfg = OmegaConf.load(copied)
    assert OmegaConf.select(loaded_cfg, 'llm.model') == 'test-model'


def test_collect_saves_config_yaml(tmp_path):
    src = _make_recording_dir(tmp_path, '20260101.120000.expt', {'llm': {'model': 'a'}}, SAMPLE_RECORDS)
    train_dir = tmp_path / 'train'
    train_dir.mkdir()
    config.configure(cfg={'train_dir': str(train_dir)})

    out_dir, _ = collect_interface_data([src], 'classify', file_under='test')

    assert (out_dir / 'config.yaml').exists()


def test_collect_returns_dataset(tmp_path):
    src = _make_recording_dir(tmp_path, '20260101.120000.expt', {'llm': {'model': 'a'}}, SAMPLE_RECORDS)
    train_dir = tmp_path / 'train'
    train_dir.mkdir()
    config.configure(cfg={'train_dir': str(train_dir)})

    _, dataset = collect_interface_data([src], 'classify', file_under='test')

    assert isinstance(dataset, Dataset)
    assert dataset.name == 'classify'
    assert len(dataset.cases) == 2


def test_collect_multiple_sources(tmp_path):
    src1 = _make_recording_dir(tmp_path, '20260101.120000.expt', {'llm': {'model': 'a'}}, SAMPLE_RECORDS)
    src2 = _make_recording_dir(tmp_path, '20260102.120000.expt', {'llm': {'model': 'b'}}, SAMPLE_RECORDS)
    train_dir = tmp_path / 'train'
    train_dir.mkdir()
    config.configure(cfg={'train_dir': str(train_dir)})

    out_dir, dataset = collect_interface_data([src1, src2], 'classify', file_under='test')

    assert len(dataset.cases) == 4
    sources = (out_dir / 'sources.txt').read_text().strip().split('\n')
    assert len(sources) == 2
    cfg_dir = out_dir / 'source_configs'
    assert len(list(cfg_dir.iterdir())) == 2


def test_collect_missing_config_raises(tmp_path):
    d = tmp_path / '20260101.120000.expt'
    d.mkdir()
    with open(d / 'results.jsonl', 'w') as f:
        f.write(json.dumps(SAMPLE_RECORDS[0]) + '\n')
    train_dir = tmp_path / 'train'
    train_dir.mkdir()
    config.configure(cfg={'train_dir': str(train_dir)})

    with pytest.raises(ValueError, match='missing config file'):
        collect_interface_data([d], 'classify', file_under='test')

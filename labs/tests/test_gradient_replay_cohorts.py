"""Exercise sample identity and isolation without loading any MNIST split."""
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import pytest
import torch


@pytest.fixture
def cohort_functions():
    path = Path(__file__).resolve().parents[2] / 'experiments/analyze_conv_eqprop_bptt_checkpoint_gradients.py'
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name in {'_checked_replay_indices', '_build_validation_cohort'}]
    dataset = torch.utils.data.TensorDataset(torch.tensor([[10.], [20.], [30.], [40.]]),
                                            torch.tensor([0, 1, 2, 3]))
    loader = SimpleNamespace(validation_indices=[100, 200, 300, 400], split_seed=0,
                             validation_indices_hash='split',
                             validation_loader=torch.utils.data.DataLoader(dataset, batch_size=2))
    scope = dict(torch=torch, np=np, Path=Path, Mapping=Mapping, Sequence=Sequence, Any=Any,
                 COHORT_SCHEMA='test', _dataset_signature=lambda *a, **k: {'params': {}, 'factory': 'fake'},
                 _resolve_callable=lambda _: lambda **k: SimpleNamespace(build=lambda: loader),
                 stable_index_sequence_hash=lambda v: repr(tuple(v)),
                 stable_batch_order_hash=lambda v: repr(tuple(v)),
                 _batch_payload_sha256=lambda x,y,i: repr((x.tolist(), y.tolist(), tuple(i))))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), scope)
    return scope


def test_explicit_prefix_is_identical_to_historical_path(cohort_functions):
    build = cohort_functions['_build_validation_cohort']
    old, _ = build({}, data_root=Path('/unused'), batch_size=2, example_count=4)
    new, _ = build({}, data_root=Path('/unused'), batch_size=2, example_count=4,
                   source_indices=[100, 200, 300, 400])
    assert [r['payload_sha256'] for r in old] == [r['payload_sha256'] for r in new]


def test_explicit_reordering_moves_images_labels_and_indices_together(cohort_functions):
    batches, cohort = cohort_functions['_build_validation_cohort'](
        {}, data_root=Path('/unused'), batch_size=2, example_count=4,
        source_indices=[400, 100, 300, 200])
    assert batches[0]['images'].tolist() == [[40.], [10.]]
    assert batches[0]['labels'].tolist() == [3, 0]
    assert batches[0]['source_indices'] == (400, 100)
    assert cohort['class_counts'][:4] == [1, 1, 1, 1]


@pytest.mark.parametrize('indices,count,excluded', [
    ([100, 100], 2, []), ([100, 999], 2, []), ([100], 2, []),
    ([100., 200], 2, []), ([True, 200], 2, []), ([100, 200], 2, [200]),
])
def test_invalid_or_overlapping_cohorts_fail(cohort_functions, indices, count, excluded):
    with pytest.raises(ValueError):
        cohort_functions['_checked_replay_indices']([100, 200, 300, 400], indices, count, excluded)

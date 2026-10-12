from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from afriscience_mt.core.schema import (
    ExperimentResult, Metrics, ModelInfo, PredictionFile,
    load_result, save_result, save_predictions,
)
from afriscience_mt.core.validation import validate_parallel_inputs
from afriscience_mt.commands import evaluate
from afriscience_mt.evaluation import metrics


@pytest.mark.parametrize('columns', [
    {'hypotheses': ['x'], 'references': []},
    {'sources': ['x', 'y'], 'hypotheses': ['x'], 'references': ['x']},
    {'hypotheses': [], 'references': []},
    {'hypotheses': [None], 'references': ['x']},
    {'hypotheses': [42], 'references': ['x']},
])
def test_parallel_validation(columns):
    with pytest.raises(ValueError):
        validate_parallel_inputs(**columns)


@pytest.mark.parametrize('function', [metrics.compute_bleu, metrics.compute_chrf])
def test_text_metrics_reject_mismatch(function):
    with pytest.raises(ValueError):
        function(['translation'], [])


@pytest.mark.parametrize('function', [metrics.compute_comet, metrics.compute_ssa_comet,
                                     metrics.compute_all_metrics])
def test_neural_validation_before_loading(function):
    with pytest.raises(ValueError):
        function(['source'], ['hypothesis'], [])


def test_cached_neural_validation_before_loading(monkeypatch):
    scorer = metrics.SSACOMETScorer(device='cpu')
    load = Mock(side_effect=AssertionError('must not load'))
    monkeypatch.setattr(scorer, '_load_model', load)
    with pytest.raises(ValueError):
        scorer.score(['source'], [], ['reference'])
    load.assert_not_called()


def test_actual_text_metrics_keep_empty_hypotheses():
    refs = ['the quick brown fox jumps', 'a second sentence for testing']
    assert metrics.compute_bleu(refs, refs)['bleu'] == pytest.approx(100)
    assert metrics.compute_chrf(refs, refs)['chrf'] == pytest.approx(100)
    assert metrics.compute_chrf([refs[0], ''], refs)['chrf'] < 100


@pytest.fixture
def artifact(tmp_path):
    result = ExperimentResult('demo', 'zero_shot', '2026-10-12',
        ModelInfo('test-model', 'seq2seq'), 'eng', 'yor', 'test', 'dcs', {},
        Metrics(bleu=5, num_samples=999))
    predictions = PredictionFile('demo', 'eng', 'yor', [
        {'src': 'first source', 'ref': 'first reference sentence', 'hyp': ''},
        {'src': 'second source', 'ref': 'second reference sentence', 'hyp': 'second reference sentence'},
    ])
    save_result(result, tmp_path / 'results.json')
    save_predictions(predictions, tmp_path / 'predictions.json')
    return tmp_path, result, predictions


def test_rescore_keeps_empty_and_counts_all(artifact):
    path, _, _ = artifact
    assert evaluate._rescore(['--src', str(path), '--metric', 'chrf']) == 0
    result = load_result(path / 'results.json')
    assert result.metrics.num_samples == 2
    assert 0 < result.metrics.chrf < 100


@pytest.mark.parametrize('corruption', ['missing_result', 'bad_json', 'null_hyp',
                                      'empty_ref', 'identity', 'empty_samples'])
def test_preflight_failure_leaves_result_unchanged(artifact, corruption, monkeypatch):
    path, result, predictions = artifact
    if corruption == 'missing_result':
        (path / 'results.json').unlink()
    elif corruption == 'bad_json':
        (path / 'predictions.json').write_text('{')
    else:
        if corruption == 'null_hyp': predictions.samples[0]['hyp'] = None
        elif corruption == 'empty_ref': predictions.samples[0]['ref'] = ''
        elif corruption == 'identity': predictions.experiment_id = 'different'
        elif corruption == 'empty_samples': predictions.samples = []
        save_predictions(predictions, path / 'predictions.json')
    original = (path / 'results.json').read_bytes() if (path / 'results.json').exists() else None
    compute = Mock(side_effect=AssertionError('preflight must prevent scoring'))
    monkeypatch.setattr(metrics, 'compute_chrf', compute)
    assert evaluate._rescore(['--src', str(path), '--metric', 'chrf']) == 1
    compute.assert_not_called()
    if original is not None: assert (path / 'results.json').read_bytes() == original


@pytest.mark.parametrize('count', [0, 1, 3])
def test_legacy_fallback_exact_length(artifact, monkeypatch, count):
    _, result, predictions = artifact
    predictions.samples = [{'hyp': 'first'}, {'hyp': 'second'}]
    monkeypatch.setattr(evaluate, '_test_text_column', lambda lang: ('text',) * count)
    with pytest.raises(ValueError, match='length'):
        evaluate._rescore_inputs(predictions, result)


def test_legacy_fallback_success(artifact, monkeypatch):
    _, result, predictions = artifact
    predictions.samples = [{'hyp': ''}, {'hyp': 'second'}]
    monkeypatch.setattr(evaluate, '_test_text_column', lambda lang: ('first', 'second'))
    sources, hyps, refs = evaluate._rescore_inputs(predictions, result)
    assert sources == refs == ['first', 'second']
    assert hyps == ['', 'second']


@pytest.mark.parametrize('dataset,split', [('mafand', 'test'), ('dcs', 'dev')])
def test_no_wrong_dataset_fallback(artifact, dataset, split):
    _, result, predictions = artifact
    predictions.samples = [{'hyp': 'text'}]
    result.dataset, result.split = dataset, split
    with pytest.raises(ValueError, match='dataset=dcs'):
        evaluate._rescore_inputs(predictions, result)


@pytest.mark.parametrize('failure', ['scoring', 'serialization', 'replace', 'nan'])
def test_failed_rescore_preserves_file(artifact, monkeypatch, failure):
    path, _, _ = artifact
    original = (path / 'results.json').read_bytes()
    if failure == 'scoring':
        monkeypatch.setattr(metrics, 'compute_chrf', Mock(side_effect=RuntimeError('model failed')))
    elif failure == 'nan':
        monkeypatch.setattr(metrics, 'compute_chrf', lambda *args: {'chrf': float('nan')})
    else:
        from afriscience_mt.core import schema
        if failure == 'serialization':
            monkeypatch.setattr(schema.json, 'dump', Mock(side_effect=TypeError('serialization failed')))
        else:
            monkeypatch.setattr(schema.os, 'replace', Mock(side_effect=OSError('disk failed')))
    assert evaluate._rescore(['--src', str(path), '--metric', 'chrf']) == 1
    assert (path / 'results.json').read_bytes() == original
    assert not list(path.glob('.results.json.*'))


def test_no_artifacts_is_failure(tmp_path):
    assert evaluate._rescore(['--src', str(tmp_path), '--metric', 'chrf']) == 1


def test_skip_existing_preserves_file(artifact):
    path, _, _ = artifact
    original = (path / 'results.json').read_bytes()
    assert evaluate._rescore(['--src', str(path), '--metric', 'bleu', '--skip-existing']) == 0
    assert (path / 'results.json').read_bytes() == original


def test_all_artifacts_preflighted_before_write(artifact):
    path, _, _ = artifact
    bad = path / 'invalid'
    bad.mkdir()
    (bad / 'predictions.json').write_text('{')
    original = (path / 'results.json').read_bytes()
    assert evaluate._rescore(['--src', str(path), '--metric', 'chrf']) == 1
    assert (path / 'results.json').read_bytes() == original


def test_cached_neural_scoring_keeps_each_sample(monkeypatch):
    scorer = metrics.SSACOMETScorer(device='cpu')
    model = Mock()
    model.predict.return_value = SimpleNamespace(system_score=0.5, scores=[0.2, 0.8])
    monkeypatch.setattr(scorer, '_load_model', lambda: model)
    result = scorer.score(['one', 'two'], ['', 'translation'], ['ref one', 'ref two'])
    assert result['ssa_comet_scores'] == [0.2, 0.8]
    assert len(model.predict.call_args.args[0]) == 2
    assert model.predict.call_args.args[0][0]['mt'] == ''

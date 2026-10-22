import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from afriscience_mt.core.alignment import (
    aligned_inputs, manifest_fingerprint, translate_manifest, validate_manifest,
)
from afriscience_mt.core.schema import (
    ExperimentResult, Metrics, ModelInfo, load_result, save_predictions, save_result,
)
from afriscience_mt.commands.evaluate import _rescore


@pytest.fixture
def manifest():
    return dict(dataset='demo', split='dev', source_lang='eng', target_lang='yor', samples=[
        dict(sample_id='doc:1', src='source one', ref='reference one'),
        dict(sample_id='doc:2', src='source two', ref='reference two'),
    ])


def infer(manifest):
    async def translate(texts, source_lang, target_lang):
        assert texts == ['source one', 'source two']
        assert (source_lang, target_lang) == ('eng', 'yor')
        return ['reference one', '']
    return asyncio.run(translate_manifest(manifest, 'experiment', translate))


def result():
    return ExperimentResult('experiment', 'zero_shot', '2026-10-22',
        ModelInfo('test', 'seq2seq'), 'eng', 'yor', 'dev', 'demo', {}, Metrics())


def test_reorder_and_round_trip(manifest, tmp_path):
    predictions = infer(manifest)
    predictions.samples.reverse()
    assert aligned_inputs(predictions, result(), manifest) == (
        ['source one', 'source two'], ['reference one', ''], ['reference one', 'reference two'])
    save_predictions(predictions, tmp_path / 'predictions.json')
    save_result(result(), tmp_path / 'results.json')
    import json
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    assert _rescore(['--src', str(tmp_path), '--metric', 'chrf', '--reference-manifest', str(path)]) == 0
    saved = load_result(tmp_path / 'results.json')
    assert saved.metrics.num_samples == 2
    assert 0 < saved.metrics.chrf < 100
    assert saved.metadata['rescored'][0]['alignment'] == {
        'mode': 'sample_id', 'reference_fingerprint': manifest_fingerprint(manifest)}


def test_fingerprint_order_independent_but_text_sensitive(manifest):
    changed = deepcopy(manifest)
    changed['samples'].reverse()
    assert manifest_fingerprint(changed) == manifest_fingerprint(manifest)
    changed['samples'][0]['ref'] += 'é'
    assert manifest_fingerprint(changed) != manifest_fingerprint(manifest)


@pytest.mark.parametrize('corruption', ['duplicate', 'missing', 'extra', 'id', 'fingerprint', 'ref', 'hyp'])
def test_reject_corrupt_predictions(manifest, corruption):
    predictions = infer(manifest)
    if corruption == 'duplicate':
        predictions.samples[1]['sample_id'] = 'doc:1'
    elif corruption == 'missing':
        predictions.samples.pop()
    elif corruption == 'extra':
        predictions.samples.append(dict(sample_id='extra', hyp='x'))
    elif corruption == 'id':
        del predictions.samples[0]['sample_id']
    elif corruption == 'fingerprint':
        predictions.reference_fingerprint = None
    elif corruption == 'ref':
        predictions.samples[0]['ref'] = 'stale'
    else:
        predictions.samples[0]['hyp'] = None
    with pytest.raises(ValueError):
        aligned_inputs(predictions, result(), manifest)


@pytest.mark.parametrize('field', ['dataset', 'split', 'source_lang', 'target_lang'])
def test_result_identity(manifest, field):
    artifact = result()
    setattr(artifact, field, 'wrong')
    with pytest.raises(ValueError, match='mismatch'):
        aligned_inputs(infer(manifest), artifact, manifest)


@pytest.mark.parametrize('corruption', ['duplicate', 'blank_id', 'empty_ref', 'nonstring', 'empty_samples'])
def test_invalid_manifest(manifest, corruption):
    if corruption == 'duplicate':
        manifest['samples'][1]['sample_id'] = 'doc:1'
    elif corruption == 'blank_id':
        manifest['samples'][0]['sample_id'] = ''
    elif corruption == 'empty_ref':
        manifest['samples'][0]['ref'] = ' '
    elif corruption == 'nonstring':
        manifest['samples'][0]['src'] = 123
    else:
        manifest['samples'] = []
    with pytest.raises(ValueError):
        validate_manifest(manifest)


@pytest.mark.parametrize('returned', [[], ['only one'], ['x', None]])
def test_inference_cardinality_and_types(manifest, returned):
    async def translate(*args):
        return returned
    with pytest.raises(ValueError):
        asyncio.run(translate_manifest(manifest, 'experiment', translate))


def test_snapshot_before_inference(manifest):
    fingerprint = manifest_fingerprint(manifest)
    async def translate(*args):
        manifest['samples'][0]['ref'] = 'mutated while awaiting'
        return ['one', 'two']
    predictions = asyncio.run(translate_manifest(manifest, 'experiment', translate))
    assert predictions.reference_fingerprint == fingerprint
    assert predictions.samples[0]['ref'] == 'reference one'


def test_ids_without_manifest_leave_results_unchanged(manifest, tmp_path):
    save_predictions(infer(manifest), tmp_path / 'predictions.json')
    save_result(result(), tmp_path / 'results.json')
    before = (tmp_path / 'results.json').read_bytes()
    assert _rescore(['--src', str(tmp_path), '--metric', 'chrf']) == 1
    assert (tmp_path / 'results.json').read_bytes() == before

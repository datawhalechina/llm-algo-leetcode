"""CPU tests of evidence selection and decisions; never start Ray or CUDA."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.verl_mechanism_closure import (
    SCHEMA, SOURCE, ResourceTrace, compare_scores, decide_candidate, encode_prompt,
    load_frozen_source, read_summary, system_decision, write_once,
)


def candidate():
    return {
        'status': 'measured', 'trajectories': [{'sample_id': 'probe'}],
        'scores': {'behavior': [[-0.4, -0.5]], 'actor_before': [[-0.4, -0.5]],
                   'rollout_before': [[-0.4, -0.5]], 'actor_after': [[-0.42, -0.52]],
                   'rollout_after': [[-0.42, -0.52]]},
        'update': {'optimizer_steps': 1, 'same_names_shapes_dtype': True, 'loss': 1.0,
                   'grad_norm': 0.5, 'delta_l2': 0.1, 'changed_tensors': 1,
                   'before_sha256': 'before', 'after_sha256': 'after'},
        'initial_sync': {'rpc_completed': True, 'sender_adapter_sha256': 'before', 'version': 0},
        'sync': {'rpc_completed': True, 'sender_adapter_sha256': 'after', 'version': 1},
        'rollout_versions_before': [0], 'rollout_versions_after': [1],
        'generation_before': [{'version': 0, 'token_ids': [7], 'log_probs': [-0.4]}],
        'generation_after': [{'version': 1, 'token_ids': [8], 'log_probs': [-0.42]}],
    }


def test_changes_and_sync_pass_without_reward_or_rejection():
    result = decide_candidate(candidate())
    assert result['passed']
    assert result['admission'][0]['admitted']


@pytest.mark.parametrize('key,value', [
    ('grad_norm', 0), ('grad_norm', float('nan')), ('loss', float('inf')),
    ('delta_l2', 0), ('changed_tensors', 0), ('optimizer_steps', 2),
    ('same_names_shapes_dtype', False), ('after_sha256', 'before'),
])
def test_invalid_updates_do_not_pass(key, value):
    record = candidate()
    record['update'][key] = value
    assert not decide_candidate(record)['passed']


def test_absolute_tolerance_cannot_hide_unsynchronized_adapter():
    record = candidate()
    record['scores']['rollout_after'] = record['scores']['rollout_before']
    result = decide_candidate(record)
    assert result['checks']['same_version_probabilities']
    assert not result['checks']['synchronized_change_visible']
    assert not result['passed']


def test_wrong_version_and_missing_generation_do_not_pass():
    record = candidate()
    record['rollout_versions_after'] = [0]
    record['generation_after'] = []
    result = decide_candidate(record)
    assert not result['checks']['sync_rpc_and_version']
    assert not result['checks']['real_generation_records']


@pytest.mark.parametrize('left,right', [([], []), ([[]], [[]]), ([[-1]], [[-1, -2]]),
                                          ([[-1]], [[float('nan')]]), ([[-1]], [[0.1]])])
def test_invalid_scores_raise(left, right):
    with pytest.raises(ValueError):
        compare_scores(left, right)


def test_identity_count_checked():
    record = candidate()
    record['trajectories'] = []
    with pytest.raises(ValueError):
        decide_candidate(record)


def test_predeclared_source_selection(tmp_path):
    if not SOURCE.is_file():
        pytest.skip('Local historical evidence is not distributed with this checkout')
    source, rows = load_frozen_source(SOURCE)
    assert len(rows) == 8 and all(r['source_response_index'] == 0 for r in rows)
    assert rows[0]['response_ids'] == source['groups'][0]['responses'][0]['token_ids']
    source['groups'][0]['responses'][0]['finish_reason'] = 'length'
    path = tmp_path / 'source.json'
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match='First frozen'):
        load_frozen_source(path)  # Must not select a later successful response.


def test_source_prompt_tampering_rejected(tmp_path):
    if not SOURCE.is_file():
        pytest.skip('Local historical evidence is not distributed with this checkout')
    source = json.loads(SOURCE.read_text())
    source['groups'][0]['messages'][0]['content'] += 'changed'
    path = tmp_path / 'source.json'
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match='prompt hash'):
        load_frozen_source(path)


def test_no_overwrite(tmp_path):
    path = tmp_path / 'result.json'
    write_once(path, {'old': True})
    with pytest.raises(FileExistsError):
        write_once(path, {'old': False})
    assert json.loads(path.read_text()) == {'old': True}


def test_resource_phase_alignment_and_unknown_peaks():
    trace = ResourceTrace()
    trace.events = [{'name': 'empty', 'begin_ns': 1, 'end_ns': 2},
                    {'name': 'sampled', 'begin_ns': 3, 'end_ns': 5}]
    trace.samples = [{'monotonic_ns': 4, 'gpu_used_mb': 7, 'host_used_mb': 20},
                     {'monotonic_ns': 6, 'gpu_used_mb': 100, 'host_used_mb': 30}]
    events = trace.report()['events']
    assert events[0]['sampled_gpu_peak_mb'] is None
    assert events[1]['sampled_gpu_peak_mb'] == 7


def test_tokenizer_uses_explicit_text_encoding():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {'tokenize': False, 'add_generation_prompt': True}
            return 'rendered'

        def encode(self, text, **kwargs):
            assert text == 'rendered' and kwargs == {'add_special_tokens': False}
            return [1, 2]

    assert encode_prompt(Tokenizer(), []) == [1, 2]


def paired_evidence():
    row = candidate()
    row['decision'] = decide_candidate(row)
    row['source_sha256'] = 'source'
    row['worker_events'] = [{'name': name, 'edge': edge} for name in
                            ('verl_weight_sync', 'rollout_weight_transfer')
                            for edge in ('begin', 'end')]
    row['resolved_config'] = {'actor_rollout_ref': {'rollout': {'free_cache_engine': True}}}
    row['initialization'] = {'seed': 42, 'adapter_sha256': 'before', 'expected_sha256': None,
                             'paired_hash_matched': None}
    candidates = {'release': row, 'retain': deepcopy(row)}
    candidates['retain']['initialization'].update(expected_sha256='before', paired_hash_matched=True)
    row['worker_events'].extend({'name': 'rollout_resume', 'edge': edge, 'tags': [tag]}
                               for tag in ('weights', 'kv_cache') for edge in ('begin', 'end'))
    candidates['retain']['resolved_config']['actor_rollout_ref']['rollout']['free_cache_engine'] = False
    events = [{'name': name, 'begin_ns': 1, 'end_ns': 2, 'duration_s': 1e-9,
               'sample_count': 0, 'sampled_gpu_peak_mb': None} for name in
              ('generation_before', 'actor_score_before', 'supervised_update',
               'actor_score_after', 'weight_sync', 'generation_after', 'sleep_before_actor')]
    trace = {'events': events, 'samples': [{'gpu_used_mb': 100, 'host_used_mb': 200}]}
    return candidates, {'release': trace, 'retain': trace}


def test_resource_pair_needs_real_events_and_same_settings():
    candidates, resources = paired_evidence()
    result = system_decision(candidates, resources)
    assert result['passed'] and not result['steady_state_speedup_proven']
    candidates['retain']['resolved_config']['other'] = 1
    assert not system_decision(candidates, resources)['checks']['paired_config']
    candidates['retain']['worker_events'] = []
    assert not system_decision(candidates, resources)['checks']['retain_actual_sync_events']


def test_skipped_pair_is_partial_not_accept():
    candidates, resources = paired_evidence()
    candidates['retain'] = {'status': 'skipped', 'reason': 'release_not_validated'}
    resources.pop('retain')
    result = system_decision(candidates, resources)
    assert not result['passed']
    assert result['measurements']['retain']['sampled_gpu_peak_mb'] is None


@pytest.mark.parametrize('field,value', [('adapter_sha256', 'different'), ('seed', 7),
                                       ('expected_sha256', None), ('paired_hash_matched', False)])
def test_pair_requires_verified_identical_initial_parameters(field, value):
    candidates, resources = paired_evidence()
    candidates['retain']['initialization'][field] = value
    result = system_decision(candidates, resources)
    assert not result['checks']['paired_initial_parameters']
    assert not result['passed']


def test_v1_without_initialization_evidence_stays_unconfirmed():
    candidates, resources = paired_evidence()
    for row in candidates.values():
        row.pop('initialization')
    assert not system_decision(candidates, resources)['checks']['paired_initial_parameters']


def test_independent_page_readers(tmp_path):
    candidates, resources = paired_evidence()
    path = tmp_path / 'result.json'
    write_once(path, {'schema_version': SCHEMA, 'candidates': candidates,
                      'system_decision': system_decision(candidates, resources)})
    assert read_summary(path, 'lifecycle')['release']['decision']['passed']
    assert read_summary(path, 'system')['passed']
    with pytest.raises(ValueError):
        read_summary(path, 'other')


@pytest.mark.parametrize('returncode', [0, 4])
@pytest.mark.parametrize('filename,ids', [
    ('PT-ROLLOUT-LIFECYCLE_Rollout_and_Policy_Freshness.ipynb', ('b9441446', '73c04def', 'fca9afca')),
    ('PT-RL-SYSTEM_Rollout_Training_and_Reward_System.ipynb', ('77b51cbc', '1606ce15', '2a7939dc')),
])
def test_notebook_preflight_and_saved_read_are_independent(tmp_path, monkeypatch, filename, ids, returncode):
    import subprocess
    from tools import project_runtime
    monkeypatch.setattr(project_runtime, 'resolve_project_root', lambda: tmp_path)
    calls = []
    monkeypatch.setattr(subprocess, 'run', lambda command, **kwargs:
                        (calls.append(command) or SimpleNamespace(returncode=returncode)))
    path = Path(__file__).resolve().parents[1] / '02_PyTorch_Algorithms' / filename
    cells = {c['id']: ''.join(c['source']) for c in json.loads(path.read_text())['cells']}
    namespace = {}
    exec(cells[ids[0]], namespace)
    # Test the preflight branch explicitly; saved learner notebooks may be in real_gpu mode.
    namespace['RUN_MODE'] = 'dry_run'
    exec(cells[ids[1]], namespace)
    exec(cells[ids[2]], namespace)  # dry_run has no result, and must not invent one.
    assert calls[0][calls[0].index('--mode') + 1] == 'preflight'
    assert not namespace['RESULT_PATH'].exists()
    candidates, resources = paired_evidence()
    write_once(namespace['RESULT_PATH'], {'schema_version': SCHEMA,
               'run_label': namespace['RUN_LABEL'], 'candidates': candidates,
               'system_decision': system_decision(candidates, resources)})
    calls.clear()
    namespace = {}
    exec(cells[ids[0]], namespace)  # Simulate a fresh kernel and only 5.1 + 5.3.
    exec(cells[ids[2]], namespace)
    namespace['RUN_MODE'] = 'real_gpu'
    exec(cells[ids[1]], namespace)  # Shared result reuse must not start another worker.
    assert not calls


@pytest.mark.parametrize('mode,environment_ready,expected_download,expected_code', [
    ('preflight', True, False, 4), ('run', False, False, 2), ('run', True, True, 2),
])
def test_model_download_gate_before_gpu(tmp_path, monkeypatch, mode, environment_ready, expected_download, expected_code):
    import sys
    from tools import verl_mechanism_closure as closure
    monkeypatch.setattr(closure, 'load_frozen_source', lambda path: ({}, []))
    monkeypatch.setattr(closure, 'inspect_api', lambda path: {'inspection_passed': True})
    monkeypatch.setattr(closure, 'inspect_runtime', lambda *a, **k: {'ready': environment_ready})
    calls = []
    def cpu(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1, stdout='', stderr='MODEL_CACHE_MISSING: fixture')
    monkeypatch.setattr(closure.subprocess, 'run', cpu)
    monkeypatch.setattr(sys, 'argv', ['closure', '--mode', mode, '--verl-root', str(tmp_path),
                                     '--run-label', 'test', '--output', str(tmp_path/'unused.json')])
    assert closure.main() == expected_code
    assert len(calls) == 1  # Only the mocked CPU child; no GPU worker.
    assert ('--prepare-model' in calls[0]) == expected_download
    assert not (tmp_path/'unused.json').exists()

"""CPU-only tests for projects 77/78's saved-evidence review and registration."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def project_cell(tag: str, chapter: str = '77') -> str:
    """Read the actual notebook cell so tests exercise the published workflow."""
    path = next((ROOT / '02_PyTorch_Algorithms').glob(f'{chapter}_*.ipynb'))
    notebook = json.loads(path.read_text(encoding='utf-8'))
    return next(''.join(cell['source']) for cell in notebook['cells']
                if tag in cell.get('metadata', {}).get('tags', []))


class MemoryProjectClosureTests(unittest.TestCase):
    """Use synthetic files in a temporary directory; never load a model or CUDA."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        manifest_path = root / 'benchmarks/project_inputs/input.json'
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text('{"fixture": true}\n', encoding='utf-8')
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        models = [
            {'name': 'baseline', 'model_id': 'cpu-kv', 'architecture_type': 'explicit_kv', 'revision': 'a' * 40},
            {'name': 'candidate', 'model_id': 'cpu-recurrent', 'architecture_type': 'recurrent_state', 'revision': 'a' * 40},
        ]
        workload = {'seq_lens': [256, 512], 'batch_size': 1, 'dtype': 'float16',
                    'max_new_tokens': 16, 'warmup': 2, 'repeats': 3, 'needle_phrase': 'CPU_NEEDLE',
                    'measurement_protocol': 'prefill-single-decode/v2'}
        self.record = {
            'schema_version': 'architecture-evaluation/v1',
            'semantic_id': 'ARCH-LONG-SEQUENCE-MEMORY', 'run_label': 'cpu_fixture',
            'input_manifest_sha256': manifest_hash, 'workload': workload,
            'hardware': {'gpu': 'synthetic fixture, not measured'},
            'models': [
                {'name': model['name'], 'model': model['model_id'],
                 'architecture_type': model['architecture_type'], 'resolved_revision': 'a' * 40,
                 'architecture': {'model_type': model['architecture_type']},
                 'rows': [
                     {'seq_len': length, 'state_interface': 'synthetic_state',
                      'quality': {'eval_length': length, 'metric': 1.0, 'failure': None},
                      'cost': {'prefill_ms': 2.0, 'decode_step_ms': 1.0, 'peak_memory_mb': 100.0,
                               'incremental_peak_memory_mb': 10.0, 'completed_decode_trials': 3,
                               'measurement_boundary': 'prefill_and_one_cached_decode_excludes_quality_generation',
                               'trials': [{'prefill_ms': 2.0, 'decode_step_ms': 1.0,
                                           'before_memory_mb': 90.0, 'peak_memory_mb': 100.0}
                                          for _ in range(3)]}}
                     for length in workload['seq_lens']]}
                for model in models],
        }
        self.raw_path = root / 'benchmarks/results/77_memory_architecture/gpu_memory_arch_cpu_fixture.json'
        self.raw_path.parent.mkdir(parents=True)
        self.write_raw()
        self.namespace = {
            'json': json, 'hashlib': hashlib, 'math': math, 're': re, 'sys': sys,
            'RUN_STAGE': 'close', 'RUN_LABEL': 'cpu_fixture',
            'RESULT_SCHEMA': 'architecture-evaluation/v1', 'SEMANTIC_ID': 'ARCH-LONG-SEQUENCE-MEMORY',
            'MEMORY_ARCH_REPOSITORY_ROOT': root, 'MEMORY_ARCH_RESULT_PATH': self.raw_path,
            'MEMORY_ARCH_MANIFEST_SHA256': manifest_hash, 'manifest_path': manifest_path,
            'memory_expected_models': models, 'memory_fixed_workload': workload,
            'MEMORY_ARCH_GPU_CONFIG': {**workload, 'models': models, 'revision': 'main',
                                       'comparison_design': 'cross_model_observational',
                                       'comparison_notes': 'CPU closure fixture'},
        }
        self.namespace['List'] = list
        self.run_cell('project-contract')

    def write_raw(self):
        """Replace only a temporary test input, never a repository result."""
        self.raw_path.write_text(json.dumps(self.record), encoding='utf-8')

    def run_cell(self, tag):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(project_cell(tag), self.namespace)

    def test_prompt_preserves_answer_distance_and_exact_length(self):
        """Keep the answer at the beginning and the question after the filler."""
        class Tokenizer:
            def __call__(self, text, **kwargs):
                return type('Tokens', (), {'input_ids': list(text.encode('ascii'))})()

        namespace = {**self.namespace, 'List': list}
        namespace['MEMORY_ARCH_GPU_CONFIG']['models'] = self.namespace['memory_expected_models']
        with contextlib.redirect_stdout(io.StringIO()):
            exec(project_cell('project-contract'), namespace)
        for length in [256, 512]:
            tokens = namespace['build_memory_arch_prompt'](Tokenizer(), length, 'CPU_NEEDLE')
            text = bytes(tokens).decode('ascii')
            self.assertEqual(len(tokens), length)
            self.assertEqual(text.count('CPU_NEEDLE'), 1)
            self.assertLess(text.index('CPU_NEEDLE'), 40)
            self.assertGreater(text.index('Question:'), length - 80)
        with self.assertRaises(ValueError):
            namespace['build_memory_arch_prompt'](Tokenizer(), 10, 'CPU_NEEDLE')

    def test_review_rejects_mismatched_identity_and_missing_baseline(self):
        """A wrong label, changed manifest or absent baseline length blocks close."""
        for field, value in [('run_label', 'other'), ('input_manifest_sha256', 'wrong')]:
            original = self.record[field]
            self.record[field] = value
            self.write_raw()
            self.run_cell('project-evidence')
            self.assertIn(field, self.namespace['memory_review']['issues'])
            with self.assertRaises(ValueError):
                self.run_cell('project-decision')
            self.record[field] = original
        self.record['models'][0]['rows'].pop()
        self.write_raw()
        self.run_cell('project-evidence')
        self.assertIn('models.0.length_coverage', self.namespace['memory_review']['issues'])

    def test_repeat_close_and_recover_missing_receipt(self):
        """Identical close is idempotent; a missing receipt can be recreated."""
        self.run_cell('project-evidence')
        self.run_cell('project-decision')
        report = self.namespace['companion_path']
        receipt = self.namespace['receipt_path']
        report_bytes, stamp = report.read_bytes(), report.stat().st_mtime_ns
        self.run_cell('project-decision')
        self.assertTrue(self.namespace['already_complete'])
        receipt.unlink()
        self.run_cell('project-decision')
        self.assertFalse(self.namespace['already_complete'])
        self.assertTrue(receipt.is_file())
        self.assertEqual(report.read_bytes(), report_bytes)
        self.assertEqual(report.stat().st_mtime_ns, stamp)
        self.assertIsNone(self.namespace['registration_receipt']['handoff'])

    def test_snapshot_workload_and_numeric_mismatches_block_registration(self):
        """Reject a different snapshot, changed protocol and invalid measurements."""
        original = json.loads(json.dumps(self.record))
        mutations = [
            lambda record: record['models'][0].update(resolved_revision='b' * 40),
            lambda record: record['workload'].update(repeats=99),
            lambda record: record['models'][0]['rows'][0]['cost'].update(prefill_ms=float('nan')),
            lambda record: record['models'][1]['rows'][0]['cost'].update(decode_step_ms=None),
            lambda record: record['models'][1]['rows'][0]['cost'].update(peak_memory_mb=99),
            lambda record: record['models'][1]['rows'].append(record['models'][1]['rows'][0]),
        ]
        for mutate in mutations:
            self.record = json.loads(json.dumps(original))
            mutate(self.record)
            self.write_raw()
            self.run_cell('project-evidence')
            self.assertFalse(self.namespace['memory_review']['ready_for_registration'])
            with self.assertRaises(ValueError):
                self.run_cell('project-decision')

    def test_conflicting_receipt_is_not_overwritten(self):
        """Precheck all existing outputs before writing another asset."""
        self.run_cell('project-evidence')
        self.run_cell('project-decision')
        path = self.namespace['receipt_path']
        path.write_text('{"conflict": true}', encoding='utf-8')
        self.namespace['companion_path'].unlink()
        with self.assertRaises(FileExistsError):
            self.run_cell('project-decision')
        self.assertFalse(self.namespace['companion_path'].exists())
        self.assertEqual(json.loads(path.read_text()), {'conflict': True})

    def test_changed_raw_after_review_blocks_close(self):
        """Close must still use the exact bytes reviewed in 5.5."""
        self.run_cell('project-evidence')
        self.raw_path.write_text(self.raw_path.read_text() + '\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.run_cell('project-decision')

    def test_failed_recall_is_registered_as_reject(self):
        """Preserve valid failure evidence without issuing an adoption handoff."""
        self.record['models'][1]['rows'][0]['quality'].update(metric=0.0, failure='needle_not_recalled')
        self.write_raw()
        self.run_cell('project-evidence')
        self.run_cell('project-decision')
        self.assertEqual(self.namespace['registration_receipt']['decision'], 'reject')
        self.assertEqual(self.namespace['companion']['status'], 'failed')
        self.assertIsNone(self.namespace['registration_receipt']['handoff'])

    def test_trial_summary_uses_all_peaks_and_matched_start_points(self):
        """Aggregate all windows, not only the last peak or its starting usage."""
        summarize = self.namespace['summarize_memory_trials']
        result = summarize([
            {'prefill_ms': 3.0, 'decode_step_ms': 2.0, 'before_memory_mb': 100.0, 'peak_memory_mb': 150.0},
            {'prefill_ms': 1.0, 'decode_step_ms': 1.0, 'before_memory_mb': 20.0, 'peak_memory_mb': 100.0},
            {'prefill_ms': 2.0, 'decode_step_ms': 3.0, 'before_memory_mb': 50.0, 'peak_memory_mb': 90.0},
        ])
        self.assertEqual(result['prefill_ms'], 2.0)
        self.assertEqual(result['decode_step_ms'], 2.0)
        self.assertEqual(result['peak_memory_mb'], 150.0)
        self.assertEqual(result['incremental_peak_memory_mb'], 80.0)
        with self.assertRaises(ValueError):
            summarize([])


class AccessProjectClosureTests(unittest.TestCase):
    """Validate the native-adapter protocol without implementing a fake backend."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = self.root / 'inputs/manifest.json'
        self.manifest.parent.mkdir()
        self.manifest.write_text('{"fixture": true}', encoding='utf-8')
        self.workload = self.root / 'inputs/requests.jsonl'
        self.workload.write_text('{"request_id": "cpu-request"}\n', encoding='utf-8')
        self.adapter = self.root / 'inputs/adapter.py'
        self.adapter.write_text('# CPU fixture only, never executed\n', encoding='utf-8')
        self.contract = {
            'protocol': 'attention-access-native/v1', 'semantic_id': 'ARCH-ATTENTION-ACCESS',
            'run_label': 'cpu_access', 'model_id': 'cpu/model', 'model_revision': 'a' * 40,
            'manifest_sha256': self.digest(self.manifest),
            'adapter_path': 'inputs/adapter.py', 'adapter_sha256': self.digest(self.adapter),
            'workload_path': 'inputs/requests.jsonl', 'workload_sha256': self.digest(self.workload),
            'request_ids': ['cpu-request'], 'raw_result_schema': 'attention-access-native-run/v1',
        }
        self.result_dir = self.root / 'results'
        self.result_dir.mkdir()
        self.contract_path = self.result_dir / 'contract.json'
        self.contract_path.write_text(json.dumps(self.contract), encoding='utf-8')
        self.raw = self.result_dir / 'raw.json'
        model = {'model_id': 'cpu/model', 'model_revision': 'a' * 40}
        self.record = {
            'schema_version': 'attention-access-native-run/v1',
            'semantic_id': 'ARCH-ATTENTION-ACCESS', 'run_label': 'cpu_access', 'contract': self.contract,
            'native_sparse_executed': True, 'fallback': False, 'failure': None,
            'backend': 'CPU fixture, not measured', 'kernel': 'CPU fixture, not measured',
            'baseline': {**model, 'candidate_id': 'global'},
            'candidates': [{**model, 'candidate_id': 'window'}],
            'workload': {'request_ids': ['cpu-request'], 'sha256': self.digest(self.workload),
                         'actual_input_tokens': {'cpu-request': 12}},
            'quality': {'global': {'quality_score': 1.0}, 'window': {'quality_score': 1.0}},
            'metrics': {'global': {'latency_ms': 2.0, 'peak_memory_mb': 10.0},
                        'window': {'latency_ms': 1.0, 'peak_memory_mb': 8.0}},
        }
        self.write_raw()
        self.namespace = {
            'json': json, 'hashlib': hashlib, 'math': math, 're': re, 'sys': sys,
            'RUN_STAGE': 'close', 'RUN_LABEL': 'cpu_access', 'REPO_ROOT': self.root,
            'SEMANTIC_ID': 'ARCH-ATTENTION-ACCESS', 'RESULT_SCHEMA': 'architecture-evaluation/v1',
            'NATIVE_RUN_SCHEMA': 'attention-access-native-run/v1', 'candidate_contract': self.contract,
            'raw_result_path': self.raw, 'contract_path': self.contract_path,
            'capability_path': self.result_dir / 'capability.json',
            'companion_path': self.result_dir / 'report.json', 'receipt_path': self.result_dir / 'receipt.json',
            'input_manifest_path': self.manifest, 'manifest_sha256': self.digest(self.manifest),
            'workload_path': self.workload, 'workload_sha256': self.digest(self.workload),
            'adapter_path': self.adapter, 'adapter_sha256': self.digest(self.adapter),
        }

    @staticmethod
    def digest(path):
        """Hash one temporary source file used in the protocol fixture."""
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write_raw(self):
        """Save synthetic protocol input only inside the temporary test root."""
        self.raw.write_text(json.dumps(self.record), encoding='utf-8')

    def run_cell(self, tag):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(project_cell(tag, '78'), self.namespace)

    def test_registration_repeat_and_missing_receipt_recovery(self):
        """Native-path records remain tune; identical close preserves the report."""
        self.run_cell('project-evidence')
        self.run_cell('project-decision')
        report = self.namespace['companion_path']
        stamp = report.stat().st_mtime_ns
        self.run_cell('project-decision')
        self.assertTrue(self.namespace['already_complete'])
        self.namespace['receipt_path'].unlink()
        self.run_cell('project-decision')
        self.assertEqual(report.stat().st_mtime_ns, stamp)
        self.assertEqual(self.namespace['registration_receipt']['decision'], 'tune')
        self.assertIsNone(self.namespace['registration_receipt']['handoff'])

    def test_identity_request_coverage_and_metrics(self):
        """Reject mismatched identity, omitted requests and non-finite numbers."""
        original = json.loads(json.dumps(self.record))
        changes = [
            lambda record: record.update(run_label='other'),
            lambda record: record['contract'].update(adapter_sha256='wrong'),
            lambda record: record['baseline'].update(model_revision='b' * 40),
            lambda record: record['workload'].update(request_ids=[]),
            lambda record: record['workload'].update(actual_input_tokens={'cpu-request': 0}),
            lambda record: record['metrics']['window'].update(latency_ms=float('nan')),
            lambda record: record['quality']['window'].update(quality_score=None),
            lambda record: record.update(native_sparse_executed='true'),
        ]
        for change in changes:
            self.record = json.loads(json.dumps(original))
            change(self.record)
            self.write_raw()
            self.run_cell('project-evidence')
            self.assertFalse(self.namespace['ready_for_decision'])
            with self.assertRaises(ValueError):
                self.run_cell('project-decision')

    def test_probe_does_not_create_performance_report(self):
        """Input readiness does not count as native execution evidence."""
        self.raw.unlink()
        probe = {'schema_version': 'architecture-capability-probe/v1',
                 'semantic_id': 'ARCH-ATTENTION-ACCESS', 'run_label': 'cpu_access',
                 'contract': self.contract, 'status': 'input_not_ready',
                 'native_sparse_executed': False, 'issues': ['adapter missing']}
        self.namespace['capability_path'].write_text(json.dumps(probe), encoding='utf-8')
        self.run_cell('project-evidence')
        self.run_cell('project-decision')
        self.assertEqual(self.namespace['registration_receipt']['decision'], 'pending')
        self.assertIsNone(self.namespace['registration_receipt']['report_path'])
        self.assertFalse(self.namespace['companion_path'].exists())

    def test_failure_and_fallback_never_become_accept(self):
        """Record failures as reject, and fallback as tune, without a handoff."""
        for failure, fallback, expected in [(None, True, 'tune'), ('request_failed', False, 'reject')]:
            self.record.update(failure=failure, fallback=fallback)
            self.write_raw()
            self.run_cell('project-evidence')
            for key in ('companion_path', 'receipt_path'):
                self.namespace[key].unlink(missing_ok=True)
            self.run_cell('project-decision')
            self.assertEqual(self.namespace['registration_receipt']['decision'], expected)
            self.assertIsNone(self.namespace['registration_receipt']['handoff'])

    def test_conflict_prevents_partial_new_output(self):
        """An existing incompatible receipt stops report creation."""
        self.namespace['receipt_path'].write_text('{"conflict": true}', encoding='utf-8')
        self.run_cell('project-evidence')
        with self.assertRaises(FileExistsError):
            self.run_cell('project-decision')
        self.assertFalse(self.namespace['companion_path'].exists())

    def test_source_mutation_after_review(self):
        """Any changed source bytes require a new read before close."""
        for path in (self.raw, self.workload, self.adapter, self.manifest, self.contract_path):
            before = path.read_bytes()
            self.run_cell('project-evidence')
            path.write_bytes(before + b'\n')
            with self.assertRaises(ValueError):
                self.run_cell('project-decision')
            path.write_bytes(before)

    def test_measurement_refuses_to_replace_raw_result(self):
        """A second measure cannot call the adapter over an existing result."""
        self.namespace['RUN_STAGE'] = 'measure'
        before = self.raw.read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_cell('project-experiment')
        self.assertEqual(self.raw.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()

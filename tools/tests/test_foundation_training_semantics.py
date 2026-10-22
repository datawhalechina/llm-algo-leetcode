"""Check Part 00 reference code against independent CPU expectations."""

import ast
import copy
import json
import math
from pathlib import Path
import random
import unittest

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[2]


def notebook_definitions(number):
    path = next((ROOT / "00_Prerequisites").glob(f"{number:02d}_*.ipynb"))
    notebook = json.loads(path.read_text(encoding="utf-8"))
    definitions = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            definitions.extend(node for node in ast.parse("".join(cell["source"])).body
                               if isinstance(node, (ast.FunctionDef, ast.ClassDef)))
    namespace = dict(torch=torch, nn=nn, F=F, math=math, copy=copy,
                     random=random, DataLoader=DataLoader, Dataset=Dataset)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@unittest.skipIf(torch is None, "requires the CPU PyTorch foundation environment")
class FoundationTrainingSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pages = {number: notebook_definitions(number) for number in (9, 10, 11, 12, 14, 16)}

    def setUp(self):
        self.python_rng = random.getstate()
        self.torch_rng = torch.get_rng_state()

    def tearDown(self):
        random.setstate(self.python_rng)
        torch.set_rng_state(self.torch_rng)

    def test_relu_matches_reference_gradient_including_zero(self):
        page = self.pages[14]
        points = torch.tensor([-2.0, 0.0, 2.0])
        actual = page['value_and_gradient'](page['relu'], points)
        expected = page['value_and_gradient'](F.relu, points)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference)

    def test_nonpersistent_buffer_remains_registered_but_is_not_saved(self):
        module = self.pages[9]['ToyModule']()
        module.register_buffer('temporary', torch.ones(1), persistent=False)
        self.assertIn('temporary', dict(module.named_buffers()))
        self.assertNotIn('temporary', module.state_dict())

    def test_early_stopping_preserves_best_snapshot_and_resets_on_improvement(self):
        page = self.pages[10]
        model = page['TinyNet']()
        initial = dict(best_val_loss=float('inf'), bad_epochs=0, patience=2,
                       should_stop=False, best_model_state=None)
        best = page['update_validation_state'](1.0, initial, model)
        saved = best['best_model_state']['linear.weight'].clone()
        with torch.no_grad():
            model.linear.weight.add_(1)
        failed_once = page['update_validation_state'](1.1, best, model)
        stopped = page['update_validation_state'](1.2, failed_once, model)
        self.assertFalse(failed_once['should_stop'])
        self.assertTrue(stopped['should_stop'])
        self.assertEqual(best['bad_epochs'], 0)
        torch.testing.assert_close(stopped['best_model_state']['linear.weight'], saved)
        recovered = page['update_validation_state'](0.5, stopped, model)
        self.assertFalse(recovered['should_stop'])
        self.assertEqual(recovered['bad_epochs'], 0)

    def test_checkpoint_restores_same_next_update_and_keeps_independent_snapshot(self):
        page = self.pages[10]
        model = page['TinyNet']()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5)
        page['run_random_training_step'](model, optimizer, scheduler)
        checkpoint = page['build_training_checkpoint'](model, optimizer, scheduler,
                                                       {'optimizer_step': 1}, {})
        saved = checkpoint['model']['linear.weight'].clone()
        expected_loss, expected_input = page['run_random_training_step'](model, optimizer, scheduler)
        self.assertTrue(torch.equal(checkpoint['model']['linear.weight'], saved))
        restored = page['TinyNet']()
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
        restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, 1, gamma=0.5)
        page['restore_training_checkpoint'](checkpoint, restored, restored_optimizer, restored_scheduler)
        loss, inputs = page['run_random_training_step'](restored, restored_optimizer, restored_scheduler)
        self.assertEqual(loss, expected_loss)
        self.assertTrue(torch.equal(inputs, expected_input))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(restored.state_dict()[key], value))
        self.assertEqual(restored_scheduler.state_dict(), scheduler.state_dict())

    def test_skipped_accumulation_window_cannot_leak_into_next_update(self):
        model = nn.Linear(1, 1, bias=False)
        nn.init.zeros_(model.weight)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5)
        batches = [(torch.ones(1, 1), torch.tensor([[target]]))
                   for target in (2.0, 2.0, 100.0, 100.0, 2.0)]
        timeline = self.pages[11]['run_accumulated_updates'](
            model, batches, optimizer, scheduler, 2, [True, False, True])
        # w=0 -> 0.4 at lr=0.1 -> 0.56 at lr=0.05; the skipped targets contribute nothing.
        torch.testing.assert_close(model.weight, torch.tensor([[0.56]]))
        self.assertEqual([row['optimizer_step'] for row in timeline], [0, 1, 1, 1, 2])
        self.assertIsNone(model.weight.grad)

    def test_validation_aggregates_samples_not_batch_means(self):
        page = self.pages[12]
        train = page['ToyDataset']()
        validation = page['ToyDataset']((4, 5, 6, 7))
        self.assertTrue(set(train.sample_ids).isdisjoint(validation.sample_ids))
        model = page['MaskedMeanRegressor']()
        loader = DataLoader(validation, batch_size=3, collate_fn=page['collate_pair'])
        metrics = page['evaluate_regressor'](model, loader)
        # Errors are [4, 4, 6, 6]; two batches contain 3 and 1 samples.
        self.assertEqual(metrics['sample_count'], 4)
        self.assertEqual(metrics['mse'], 26.0)
        self.assertEqual(metrics['mae'], 5.0)

    def test_attention_rejects_unknown_modes_and_malformed_dimensions(self):
        check = self.pages[16]['attention_contract_ok']
        valid = (2, 4, 8, 16)
        for shape, mode in ((valid, 'typo'), ((2, 4, 8), 'self'),
                            ((2, 4, 0, 16), 'self'), ((2, 4, 8, True), 'self')):
            with self.subTest(shape=shape, mode=mode):
                self.assertFalse(check(shape, valid, valid, mode))

    def test_fully_masked_attention_has_zero_output_and_finite_gradients(self):
        scores = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        mask = torch.tensor([[True, False], [False, False]])
        probabilities = self.pages[16]['masked_softmax'](scores, mask)
        torch.testing.assert_close(probabilities, torch.tensor([[1.0, 0.0], [0.0, 0.0]]))
        probabilities.square().sum().backward()
        self.assertTrue(torch.isfinite(scores.grad).all())


if __name__ == '__main__':
    unittest.main()

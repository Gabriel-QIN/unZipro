"""Regression tests for the local differentiable MAML implementation."""

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "script"))
from learn2learn.algorithms.maml import MAML, clone_module


class ScalarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.4, dtype=torch.float64))

    def forward(self):
        return self.weight


class MAMLTests(unittest.TestCase):
    def test_multiple_adaptations_match_analytic_meta_gradient(self):
        for first_order in (False, True):
            with self.subTest(first_order=first_order):
                model = ScalarModel()
                meta_model = MAML(model, lr=0.1, first_order=first_order)
                learner = meta_model.clone()
                for _ in range(2):
                    learner.adapt(0.5 * (learner() - 2.0).square())

                self.assertAlmostEqual(model().item(), 0.4)
                self.assertAlmostEqual(learner().item(), 0.704)
                self.assertIsNone(model.weight.grad)
                query_loss = 0.5 * (learner() + 1.0).square()
                query_loss.backward()
                expected_gradient = 1.704 if first_order else 1.704 * 0.9**2
                self.assertAlmostEqual(model.weight.grad.item(), expected_gradient)

    def test_batchnorm_state_is_private_to_each_learner(self):
        model = nn.BatchNorm1d(3)
        meta_model = MAML(model, lr=0.1)
        learner = meta_model.clone()
        other_learner = meta_model.clone()
        initial_state = {name: value.clone() for name, value in model.state_dict().items()}

        learner(torch.arange(12, dtype=torch.float32).reshape(4, 3))

        for name, initial in initial_state.items():
            self.assertTrue(torch.equal(model.state_dict()[name], initial), name)
            self.assertTrue(torch.equal(other_learner.module.state_dict()[name], initial), name)
        self.assertEqual(learner.module.num_batches_tracked.item(), 1)
        self.assertFalse(torch.equal(learner.module.running_mean, model.running_mean))
        for name, base_buffer in model.named_buffers():
            cloned_buffer = dict(learner.module.named_buffers())[name]
            self.assertNotEqual(base_buffer.data_ptr(), cloned_buffer.data_ptr())

    def test_buffer_aliases_and_differentiable_buffers_are_preserved(self):
        module = nn.Module()
        shared_buffer = torch.tensor([1.0, 2.0])
        module.register_buffer("first", shared_buffer)
        module.register_buffer("second", shared_buffer)
        module.register_buffer("differentiable", torch.tensor(3.0, requires_grad=True))
        module.register_buffer("optional", None)

        cloned = clone_module(module)

        self.assertIs(cloned.first, cloned.second)
        self.assertIsNot(cloned.first, module.first)
        self.assertIsNone(cloned.optional)
        cloned.first.add_(1)
        self.assertTrue(torch.equal(module.first, shared_buffer))
        self.assertTrue(torch.equal(module.first, torch.tensor([1.0, 2.0])))
        cloned.differentiable.square().backward()
        self.assertEqual(module.differentiable.grad.item(), 6.0)

    def test_adapt_preserves_original_autograd_error(self):
        learner = MAML(ScalarModel(), lr=0.1).clone()
        unrelated_loss = torch.tensor(1.0, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "not have been used in the graph"):
            learner.adapt(unrelated_loss)


if __name__ == "__main__":
    unittest.main()

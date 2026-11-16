"""Regression checks for episode adaptation and the meta-training boundary."""

import copy
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "script"))

import learn2learn as l2l
from unZipro_finetuning import fast_adapt, train_test_split


def graph(nodes, labels, mask=None, name="graph"):
    nodes = torch.tensor(nodes, dtype=torch.float64)
    count = len(nodes)
    return (
        nodes,
        torch.zeros(count, count, 1, dtype=torch.float64),
        torch.eye(count, dtype=torch.bool),
        torch.tensor(labels, dtype=torch.long),
        torch.ones(count, dtype=torch.bool)
        if mask is None else torch.tensor(mask, dtype=torch.bool),
        name,
    )


class LinearGraphModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(
            [[0.2, -0.4], [0.5, 0.1]], dtype=torch.float64
        ))
        self.bias = nn.Parameter(torch.tensor([0.1, -0.2], dtype=torch.float64))

    def forward(self, node, edge, adjacency):
        return node @ self.weight.T + self.bias


class BatchNormGraphModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(2, 3, dtype=torch.float64)
        self.norm = nn.BatchNorm1d(3, dtype=torch.float64)
        self.output = nn.Linear(3, 2, dtype=torch.float64)

    def forward(self, node, edge, adjacency):
        return self.output(torch.tanh(self.norm(self.input(node))))


class CountingSGD(torch.optim.SGD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps = 0

    def step(self, closure=None):
        self.steps += 1
        return super().step(closure)


class FinetuningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(7)
        self.criterion = nn.CrossEntropyLoss()
        # One support graph makes repeated inner steps deterministic.
        self.support = [graph(
            [[1.0, 0.5], [-0.5, 1.0], [0.2, -0.8]],
            [0, 1, 0], [True, True, False], "support"
        )]
        # Unequal lengths and masks distinguish residue weighting from a mean
        # of batch losses. batchsize=2 also exercises a partial final batch.
        self.query = [
            graph([[0.3, 1.2], [0.2, -0.4]], [0, 1],
                  [True, False], "query_a"),
            graph([[1.2, -0.7], [0.6, 0.2], [-0.8, 0.9], [0.1, 0.3]],
                  [1, 0, 1, 0], [True, True, True, False], "query_b"),
            graph([[-0.2, 0.7], [0.8, -0.1], [0.9, 0.5]],
                  [0, 1, 0], [True, False, True], "query_c"),
        ]

    def query_tensors(self):
        nodes = torch.cat([sample[0][sample[4]] for sample in self.query])
        targets = torch.cat([sample[3][sample[4]] for sample in self.query])
        return nodes, targets

    def test_meta_update_matches_differentiated_sgd_for_both_orders(self):
        for first_order in (False, True):
            with self.subTest(first_order=first_order):
                model = LinearGraphModel()
                initial = [p.detach().clone() for p in model.parameters()]
                weight, bias = [p.clone().requires_grad_() for p in initial]
                initial_weight, initial_bias = weight, bias
                support = self.support[0]
                for _ in range(3):
                    logits = support[0] @ weight.T + bias
                    support_loss = self.criterion(
                        logits[support[4]], support[3][support[4]]
                    )
                    grads = torch.autograd.grad(
                        support_loss, (weight, bias),
                        create_graph=not first_order
                    )
                    weight = weight - 0.3 * grads[0]
                    bias = bias - 0.3 * grads[1]

                nodes, targets = self.query_tensors()
                expected_logits = nodes @ weight.T + bias
                expected_loss = self.criterion(expected_logits, targets)
                expected_accuracy = 100 * (
                    expected_logits.argmax(1) == targets
                ).double().mean().item()
                meta_grads = torch.autograd.grad(
                    expected_loss, (initial_weight, initial_bias)
                )
                expected_parameters = [
                    p - 0.2 * g for p, g in zip(initial, meta_grads)
                ]

                meta_model = l2l.algorithms.MAML(
                    model, lr=0.3, first_order=first_order
                )
                optimizer = CountingSGD(meta_model.parameters(), lr=0.2)
                loss, accuracy = fast_adapt(
                    model, meta_model, self.support, self.query,
                    torch.device("cpu"), optimizer, self.criterion,
                    adapt_step=3, batchsize=2, cpu=0
                )

                self.assertAlmostEqual(loss, expected_loss.item(), places=11)
                self.assertAlmostEqual(accuracy, expected_accuracy, places=11)
                self.assertEqual(optimizer.steps, 1)
                for actual, expected in zip(
                    model.parameters(), expected_parameters
                ):
                    torch.testing.assert_close(
                        actual, expected, rtol=1e-10, atol=1e-11
                    )

    def assert_nested_equal(self, actual, expected):
        if isinstance(actual, torch.Tensor):
            self.assertTrue(torch.equal(actual, expected))
        elif isinstance(actual, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in actual:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(actual, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for actual_item, expected_item in zip(actual, expected):
                self.assert_nested_equal(actual_item, expected_item)
        else:
            self.assertEqual(actual, expected)

    def test_validation_adapts_clone_without_mutating_training_state(self):
        model = BatchNormGraphModel()
        meta_model = l2l.algorithms.MAML(model, lr=0.2, first_order=False)
        optimizer = torch.optim.Adam(meta_model.parameters(), lr=0.01)
        # Populate both Adam state and BatchNorm running statistics before the
        # validation episode, then retain nonzero parameter gradients.
        sample = self.support[0]
        model(*sample[:3]).square().mean().backward()
        optimizer.step()
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 0.125)

        reference = copy.deepcopy(model)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.2)
        reference.train()
        for _ in range(2):
            reference_optimizer.zero_grad()
            loss = self.criterion(
                reference(*sample[:3])[sample[4]], sample[3][sample[4]]
            )
            loss.backward()
            reference_optimizer.step()
        reference.eval()
        nodes, targets = self.query_tensors()
        with torch.no_grad():
            logits = reference(nodes, None, None)
            expected_loss = self.criterion(logits, targets).item()
            expected_accuracy = 100 * (
                logits.argmax(1) == targets
            ).double().mean().item()

        state_before = copy.deepcopy(model.state_dict())
        optimizer_before = copy.deepcopy(optimizer.state_dict())
        gradients_before = [p.grad.clone() for p in model.parameters()]
        modes_before = [module.training for module in model.modules()]
        # Validation callers may already have disabled gradient recording;
        # support adaptation still needs its own enabled gradient context.
        with torch.no_grad():
            loss, accuracy = fast_adapt(
                model, meta_model, self.support, self.query,
                torch.device("cpu"), optimizer, self.criterion,
                adapt_step=2, do_backward=False, batchsize=2, cpu=0
            )

        self.assertAlmostEqual(loss, expected_loss, places=11)
        self.assertAlmostEqual(accuracy, expected_accuracy, places=11)
        self.assert_nested_equal(model.state_dict(), state_before)
        self.assert_nested_equal(optimizer.state_dict(), optimizer_before)
        for parameter, gradient in zip(model.parameters(), gradients_before):
            self.assertTrue(torch.equal(parameter.grad, gradient))
        self.assertEqual(
            [module.training for module in model.modules()], modes_before
        )

    def test_empty_and_singleton_datasets_cannot_form_an_episode(self):
        for dataset in ([], ["one"]):
            with self.subTest(size=len(dataset)):
                with self.assertRaises(ValueError):
                    train_test_split(dataset)

    def test_valid_split_ratios_keep_both_sets_nonempty_and_disjoint(self):
        dataset = list(range(7))
        for ratio in (0.001, 0.5, 0.999):
            with self.subTest(ratio=ratio):
                support, query = train_test_split(dataset, ratio=ratio)
                self.assertGreater(len(support), 0)
                self.assertGreater(len(query), 0)
                self.assertFalse(set(support.indices) & set(query.indices))
                self.assertEqual(
                    set(support.indices) | set(query.indices),
                    set(range(len(dataset)))
                )

    def test_invalid_split_ratios_are_rejected(self):
        for ratio in (-0.1, 0, 1, 1.1, float("nan")):
            with self.subTest(ratio=ratio):
                with self.assertRaises(ValueError):
                    train_test_split(list(range(4)), ratio=ratio)

    def test_empty_support_or_query_is_rejected_before_optimization(self):
        for support, query in (([], self.query), (self.support, [])):
            with self.subTest(support_size=len(support), query_size=len(query)):
                model = LinearGraphModel()
                meta_model = l2l.algorithms.MAML(model, lr=0.2)
                optimizer = CountingSGD(meta_model.parameters(), lr=0.1)
                with self.assertRaises(ValueError):
                    fast_adapt(
                        model, meta_model, support, query,
                        torch.device("cpu"), optimizer, self.criterion,
                        adapt_step=2, batchsize=2, cpu=0
                    )
                self.assertEqual(optimizer.steps, 0)

    def test_entirely_masked_episode_is_rejected(self):
        masked = [graph([[0.1, 0.2], [0.3, 0.4]], [0, 1],
                        [False, False])]
        for support, query in ((masked, self.query), (self.support, masked)):
            with self.subTest(masked_support=support is masked):
                model = LinearGraphModel()
                meta_model = l2l.algorithms.MAML(model, lr=0.2)
                optimizer = CountingSGD(meta_model.parameters(), lr=0.1)
                with self.assertRaises(ValueError):
                    fast_adapt(
                        model, meta_model, support, query,
                        torch.device("cpu"), optimizer, self.criterion,
                        adapt_step=2, batchsize=2, cpu=0
                    )
                self.assertEqual(optimizer.steps, 0)


if __name__ == "__main__":
    unittest.main()

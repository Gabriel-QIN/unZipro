"""Regression checks for worker-free structure checking and graph loading."""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "script"))

import main as pipeline
from utils import dataset, featurizer


class WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_zero_workers_checks_and_loads_real_structure_in_current_process(self):
        path = str(REPO / "data/example/6vpcE.pdb")
        with mock.patch.object(featurizer, "ProcessPoolExecutor", side_effect=AssertionError("unexpected process pool")):
            usable = featurizer.check_pdb([path], num_workers=0)
        self.assertEqual(usable, [path])
        graphs = dataset.GraphDataset(usable, noise=0, cache_dir=None)
        loader = dataset.get_loader(graphs, batchsize=1, num_workers=0)
        self.assertIsNone(loader.prefetch_factor)
        batch = list(loader)
        self.assertEqual(len(batch), 1)
        node, edge, adjacency, labels, mask, name, count = batch[0]
        self.assertGreater(mask.sum().item(), 0)
        self.assertEqual(node.shape[0], labels.shape[0])
        self.assertEqual(edge.shape[:2], adjacency.shape[:2])
        self.assertEqual(name, path)
        self.assertEqual(count, 1)

    def test_positive_workers_keep_prefetch_enabled(self):
        loader = dataset.get_loader([], batchsize=1, num_workers=1)
        self.assertEqual(loader.num_workers, 1)
        self.assertEqual(loader.prefetch_factor, 2)

    def test_negative_worker_counts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            featurizer.check_pdb([], num_workers=-1)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            dataset.get_loader([], batchsize=1, num_workers=-1)

    def test_pretrained_pipeline_honors_cpu_zero_and_writes_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            argv = [
                "main.py", "--pdb", "6vpcE", "--pdb_dir", str(REPO / "data/example"),
                "--config_path", str(REPO / "config/unZipro_pretrain.json"),
                "--param", str(REPO / "Models/unZipro_params.pt"),
                "--outdir", str(root / "output"), "--work_dir", str(root / "work"),
                "--pretrained", "--cpu_only", "--cpu", "0", "--rank_by_prob",
            ]

            def load_graphs(*args, **kwargs):
                # Fail before any iterator can launch unwanted workers.
                self.assertEqual(kwargs.get("num_workers"), 0)
                return dataset.get_loader(*args, **kwargs)

            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(pipeline, "get_loader", side_effect=load_graphs),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                pipeline.main()
            ranked = root / "output/6vpcE.info_rank_by_prob.csv"
            self.assertTrue(ranked.is_file())
            self.assertGreater(ranked.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

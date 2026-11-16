"""Check input filtering and checkpoint selection around MAML episodes."""

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "script"))

import unZipro_finetuning as finetuning


class TinyStateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.zeros(()))
        self.register_buffer("counter", torch.zeros((), dtype=torch.long))


class FinetuningIOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_list_resolves_ids_and_extensions_and_filters_real_pdbs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copyfile(REPO / "data/example/OsPHR2.pdb", root / "alpha.pdb")
            shutil.copyfile(REPO / "data/example/6vpcE.pdb", root / "beta.pdb")
            (root / "broken.pdb").write_text("HEADER invalid structure\nEND\n")
            ids = root / "ids.csv"
            ids.write_text(
                "  alpha  \n\n beta.pdb \nalpha\nalpha.pdb\nbroken\nmissing\n"
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                paths = finetuning._load_pdb_list(str(ids), str(root), 20, 0)
            self.assertEqual(paths, [str(root / "alpha.pdb"), str(root / "beta.pdb")])
            self.assertIn("2 usable structures", output.getvalue())

    def test_insufficient_usable_structures_explains_missing_and_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copyfile(REPO / "data/example/OsPHR2.pdb", root / "alpha.pdb")
            (root / "broken.pdb").write_text("HEADER invalid structure\nEND\n")
            ids = root / "ids.csv"
            ids.write_text("alpha\nbroken\nmissing\n")
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(
                    ValueError, r"only 1 remain \(1 missing, 1 rejected\)"
                ):
                    finetuning._load_pdb_list(str(ids), str(root), 20, 0)

    def test_structures_without_valid_targets_are_filtered(self):
        source = (REPO / "data/example/OsPHR2.pdb").read_text().splitlines()
        atoms = [line for line in source if line.startswith("ATOM")]
        ca_only = [line for line in atoms if line[12:16].strip() == "CA"]
        unknown = [line[:17] + "UNK" + line[20:] for line in atoms]
        # Move whole residues alternately by 100 A: their internal N-CA-C
        # geometry stays intact, but every peptide connection is broken.
        residues = dict.fromkeys(line[21:27] for line in atoms)
        offsets = {residue: 100 * (index % 2)
                   for index, residue in enumerate(residues)}
        disconnected = [
            line[:30]
            + f"{float(line[30:38]) + offsets[line[21:27]]:8.3f}"
            + line[38:]
            for line in atoms
        ]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for name, lines in (
                ("complete", atoms), ("ca_only", ca_only),
                ("disconnected", disconnected), ("unknown", unknown),
            ):
                path = root / f"{name}.pdb"
                path.write_text("\n".join(lines) + "\nEND\n")
                paths.append(str(path))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                usable = finetuning.check_pdb(paths, nneighbor=20, num_workers=0)

            self.assertEqual(usable, paths[:1])
            for path in paths[1:]:
                self.assertIn(f"skip {path}", output.getvalue())
            self.assertEqual(output.getvalue().count("no valid target residues"), 3)

    def test_ca_only_structure_is_rejected_before_dataset_split(self):
        source = (REPO / "data/example/OsPHR2.pdb").read_text().splitlines()
        ca_only = [line for line in source
                   if line.startswith("ATOM") and line[12:16].strip() == "CA"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copyfile(REPO / "data/example/OsPHR2.pdb", root / "alpha.pdb")
            shutil.copyfile(REPO / "data/example/6vpcE.pdb", root / "beta.pdb")
            (root / "ca_only.pdb").write_text("\n".join(ca_only) + "\nEND\n")
            ids = root / "ids.csv"
            ids.write_text("alpha\nca_only\nbeta\nmissing\n")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                usable = finetuning._load_pdb_list(str(ids), str(root), 20, 0)
            self.assertEqual(usable, [str(root / "alpha.pdb"), str(root / "beta.pdb")])
            self.assertIn("4 listed, 3 found, 2 usable structures", output.getvalue())

            ids.write_text("alpha\nca_only\nmissing\n")
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(
                    ValueError, r"only 1 remain \(1 missing, 1 rejected\)"
                ):
                    finetuning._load_pdb_list(str(ids), str(root), 20, 0)

    def test_id_and_extension_aliases_cannot_fake_a_second_structure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copyfile(REPO / "data/example/OsPHR2.pdb", root / "alpha.pdb")
            ids = root / "ids.csv"
            ids.write_text("alpha\nalpha.pdb\n./alpha.pdb\n")
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "only 1 remain"):
                    finetuning._load_pdb_list(str(ids), str(root), 20, 0)

    def test_best_checkpoint_is_frozen_and_validation_never_updates_base(self):
        # Controlled episode results isolate checkpoint selection and the
        # caller's validation flag from graph featurization and optimization.
        for first_order in (False, True):
            with self.subTest(first_order=first_order):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    train_file, valid_file = root / "train.csv", root / "valid.csv"
                    for path in (train_file, valid_file):
                        path.write_text("one\ntwo\n")
                    for name in ("one.pdb", "two.pdb"):
                        (root / name).touch()
                    config_path = root / "config.json"
                    config_path.write_text(json.dumps({"nneighbor": 4}))
                    model = TinyStateModel()
                    param_file = root / "initial.pt"
                    torch.save(model.state_dict(), param_file)
                    calls = []
                    validation_splits = []
                    validation_losses = iter((0.2, 0.5))

                    def episode(base, meta, support, query, *args, **kwargs):
                        do_backward = kwargs.get("do_backward", True)
                        calls.append((do_backward, meta.first_order))
                        if len(calls) % 2:
                            with torch.no_grad():
                                base.value.add_(1)
                                base.counter.add_(1)
                            return 1.0, 50.0
                        validation_splits.append((support, query))
                        return next(validation_losses), 60.0

                    with (
                        mock.patch.object(finetuning, "check_pdb", side_effect=lambda paths, **kwargs: paths),
                        mock.patch.object(finetuning, "GraphDataset", side_effect=lambda datalist, **kwargs: list(datalist)) as datasets,
                        mock.patch.object(finetuning, "unZipro", return_value=model) as model_factory,
                        mock.patch.object(finetuning, "fast_adapt", side_effect=episode),
                        contextlib.redirect_stdout(io.StringIO()),
                        contextlib.redirect_stderr(io.StringIO()),
                    ):
                        checkpoint_path = finetuning.unZipro_finetune(
                            str(train_file), str(valid_file),
                            config_path=str(config_path), pdb_dir=str(root),
                            model_store_dir=str(root / "models"),
                            param_file=str(param_file), project_name="tiny",
                            epochs=2, adapt_step=1, cpu=0, cpu_only=True,
                            noise=0.1, nneighbor=2, first_order=first_order
                        )

                    self.assertEqual(calls, [
                        (True, first_order), (False, first_order),
                        (True, first_order), (False, first_order)
                    ])
                    saved = torch.load(checkpoint_path, weights_only=True)
                    self.assertEqual(saved["value"].item(), 1)
                    self.assertEqual(saved["counter"].item(), 1)
                    self.assertEqual(model.value.item(), 2)
                    self.assertEqual(model.counter.item(), 2)
                    self.assertIs(validation_splits[0][0], validation_splits[1][0])
                    self.assertIs(validation_splits[0][1], validation_splits[1][1])
                    self.assertEqual(datasets.call_args_list[0].kwargs["noise"], 0.1)
                    self.assertEqual(datasets.call_args_list[1].kwargs["noise"], 0.0)
                    self.assertEqual(model_factory.call_args.args[0].nneighbor, 2)


if __name__ == "__main__":
    unittest.main()

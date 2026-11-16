"""Regression checks for Foldseek hit selection and independent AFDB deficits."""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'script'))

from parse_foldseek_results import _split_afdb_hits, parse_mmseqs


def write_hits(path, hits):
    """Write headerless Foldseek web results with distinct confidence scores."""
    rows = []
    for target, probability in hits:
        row = ['query', target, 0.5, 100, 0, 0, 1, 100, 1, 100,
               probability, 1e-10, 100, 100, 100, 'A', 'A', '0,0,0',
               'A', 9606, 'Homo sapiens']
        rows.append('\t'.join(map(str, row)))
    path.write_text('\n'.join(rows) + '\n')


def pdb_target(identifier):
    return f'{identifier[:4]}-assembly1.cif.gz_{identifier[4:]}'


class FoldseekSplitTests(unittest.TestCase):
    def test_afdb_handles_each_deficit_independently(self):
        hits = pd.DataFrame({'PDB_ID': ['Q12345', 'R12345', 'P12345']})
        cases = (
            (0, 1, [], ['Q12345']),
            (2, 0, ['R12345', 'P12345'], []),
            (2, 1, ['R12345', 'P12345'], ['Q12345']),
            (0, 0, [], []),
        )
        for train_gap, valid_gap, expected_train, expected_valid in cases:
            with self.subTest(train_gap=train_gap, valid_gap=valid_gap):
                train, valid = _split_afdb_hits(hits, train_gap, valid_gap)
                self.assertEqual(train['PDB_ID'].tolist(), expected_train)
                self.assertEqual(valid['PDB_ID'].tolist(), expected_valid)
                self.assertFalse(set(train['PDB_ID']) & set(valid['PDB_ID']))

    def test_pdb_split_keeps_chains_together_and_preserves_first_hit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'alis_pdb100.m8'
            write_hits(source, [(pdb_target(name), score) for name, score in
                                [('7bv5A', 0.99), ('7bv5B', 0.98),
                                 ('1abcA', 0.8), ('2abcA', 0.7)]])
            train, valid, ids, pdb_count, af_count = parse_mmseqs(
                str(source), str(root / 'hits.csv'), str(root / 'train.csv'),
                str(root / 'test.csv'), train_size=2, max_train_size=2,
                test_ratio=0.5, include_af2=False)
            self.assertEqual(train['PDB_ID'].tolist(), ['1abcA', '2abcA'])
            self.assertEqual(valid['PDB_ID'].tolist(), ['7bv5A'])
            self.assertEqual((pdb_count, af_count), (4, 0))
            self.assertFalse({name[:4] for name in train['PDB_ID']} &
                             {name[:4] for name in valid['PDB_ID']})
            self.assertEqual(ids, ['1abcA', '2abcA', '7bv5A'])

    def test_afdb_scores_stay_with_ids_and_final_sets_are_disjoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'alis_pdb100.m8'
            write_hits(source, [(pdb_target('1abcA'), 0.9)])
            write_hits(root / 'alis_afdb50.m8', [
                ('AF-P12345-F1-model_v4', 0.01),
                ('AF-R12345-F1-model_v4', 0.5),
                ('AF-Q12345-F1-model_v4', 0.99),
            ])
            train_path, valid_path = root / 'train.csv', root / 'test.csv'
            train, valid, ids, pdb_count, af_count = parse_mmseqs(
                str(source), str(root / 'hits.csv'), str(train_path),
                str(valid_path), train_size=2, max_train_size=3,
                test_ratio=1 / 3)
            self.assertEqual(train['PDB_ID'].tolist(), ['1abcA', 'R12345', 'P12345'])
            self.assertEqual(valid['PDB_ID'].tolist(), ['Q12345'])
            self.assertEqual((pdb_count, af_count), (1, 3))
            self.assertEqual(len(set(ids)), 4)
            self.assertEqual(train_path.read_text().splitlines(), train['PDB_ID'].tolist())
            self.assertEqual(valid_path.read_text().splitlines(), valid['PDB_ID'].tolist())

    def test_afdb_can_fill_only_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'alis_pdb100.m8'
            write_hits(source, [(pdb_target('1abcA'), 0.99),
                                (pdb_target('2abcA'), 0.8)])
            write_hits(root / 'alis_afdb50.m8', [
                ('AF-Q12345-F1-model_v4', 0.9),
                ('AF-P12345-F1-model_v4', 0.1),
            ])
            train, valid, _, _, _ = parse_mmseqs(
                str(source), str(root / 'hits.csv'), str(root / 'train.csv'),
                str(root / 'test.csv'), train_size=3, max_train_size=3,
                test_ratio=0.5)
            self.assertEqual(train['PDB_ID'].tolist(), ['2abcA', 'Q12345', 'P12345'])
            self.assertEqual(valid['PDB_ID'].tolist(), ['1abcA'])

    def test_disabling_afdb_keeps_short_splits_without_reading_afdb(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'alis_pdb100.m8'
            write_hits(source, [(pdb_target('1abcA'), 0.9)])
            # An invalid AFDB file must be ignored when supplementation is disabled.
            (root / 'alis_afdb50.m8').write_text('invalid\n')
            train, valid, ids, _, af_count = parse_mmseqs(
                str(source), str(root / 'hits.csv'), str(root / 'train.csv'),
                str(root / 'test.csv'), train_size=2, max_train_size=3,
                test_ratio=1 / 3, include_af2=False)
            self.assertEqual(train['PDB_ID'].tolist(), ['1abcA'])
            self.assertTrue(valid.empty)
            self.assertEqual(ids, ['1abcA'])
            self.assertEqual(af_count, 0)

    def test_afdb_can_supply_all_hits_when_pdb_has_no_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'alis_pdb100.m8'
            source.write_text('')
            write_hits(root / 'alis_afdb50.m8', [
                ('AF-Q12345-F1-model_v4', 0.99),
                ('AF-R12345-F1-model_v4', 0.8),
                ('AF-S12345-F1-model_v4', 0.7),
                ('AF-P12345-F1-model_v4', 0.1),
            ])
            train, valid, ids, pdb_count, af_count = parse_mmseqs(
                str(source), str(root / 'hits.csv'), str(root / 'train.csv'),
                str(root / 'test.csv'), train_size=3, max_train_size=3,
                test_ratio=1 / 3)
            self.assertEqual(train['PDB_ID'].tolist(), ['R12345', 'S12345', 'P12345'])
            self.assertEqual(valid['PDB_ID'].tolist(), ['Q12345'])
            self.assertEqual((pdb_count, af_count), (0, 4))
            self.assertEqual(len(set(ids)), 4)


if __name__ == '__main__':
    unittest.main()

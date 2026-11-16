"""Offline regression tests for downloading the ABE structure lists."""

import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import numpy as np
import biotite.structure as struc
import biotite.structure.io as strucio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'script'))
import fetch_PDB_parallel as fetch


def protein(chains=('A', 'B')):
    array = struc.AtomArray(2 * len(chains))
    array.chain_id = np.repeat(chains, 2)
    array.res_id[:] = 1
    array.res_name[:] = 'ALA'
    array.atom_name = np.tile(['N', 'CA'], len(chains))
    array.element = np.tile(['N', 'C'], len(chains))
    array.coord = np.arange(6 * len(chains), dtype=np.float32).reshape(-1, 3)
    return array


class FetchTests(unittest.TestCase):
    def test_identifiers_preserve_digit_and_multicharacter_chains(self):
        self.assertEqual(fetch._parse_identifier('8aw33'), ('8aw33', '8aw3', '3', 'RCSB'))
        self.assertEqual(fetch._parse_identifier('4v48BK.pdb'), ('4v48BK', '4v48', 'BK', 'RCSB'))
        self.assertEqual(fetch._parse_identifier('P12345'), ('P12345', 'P12345', 'A', 'AFDB'))
        self.assertEqual(fetch._parse_identifier('AF-P12345-F1-model_v6')[1:],
                         ('P12345', 'A', 'AFDB'))
        with self.assertRaises(ValueError):
            fetch._parse_identifier('../1abcA')

    def test_chain_extraction_uses_first_nmr_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = protein()
            second = first.copy()
            second.coord += 100
            source, output = root / 'source.pdb', root / 'chain.pdb'
            strucio.save_structure(str(source), struc.stack([first, second]))
            fetch._extract_chain(source, output, 'B')
            result = strucio.load_structure(str(output), model=1)
            self.assertEqual(set(result.chain_id), {'B'})
            self.assertEqual(len(result), 2)
            np.testing.assert_allclose(result.coord, first.coord[2:])

    def test_multicharacter_chain_uses_cif_and_keeps_identifier_filename(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'fixture.cif'
            strucio.save_structure(str(source), protein(('BK',)))
            urls = []

            def download(url, destination, **kwargs):
                urls.append(url)
                shutil.copyfile(source, destination)

            with mock.patch.object(fetch, '_download', side_effect=download):
                output = fetch.fetch_and_save('4v48BK', root / 'output')
            self.assertEqual(Path(output).name, '4v48BK.pdb')
            self.assertEqual(urls, ['https://files.rcsb.org/download/4v48.cif'])
            result = strucio.load_structure(output, model=1)
            self.assertEqual(set(result.chain_id), {'A'})
            self.assertTrue(fetch._has_protein_atoms(output))

    def test_pdb_chain_failure_falls_back_to_author_chain_in_cif(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdb, cif = root / 'fixture.pdb', root / 'fixture.cif'
            strucio.save_structure(str(pdb), protein(('A',)))
            strucio.save_structure(str(cif), protein(('B',)))
            calls = []

            def download(url, destination, **kwargs):
                calls.append(url)
                shutil.copyfile(cif if url.endswith('.cif') else pdb, destination)

            with mock.patch.object(fetch, '_download', side_effect=download):
                output = fetch.fetch_and_save('1abcB', root / 'output')
            self.assertEqual(len(calls), 2)
            result = strucio.load_structure(output, model=1)
            self.assertEqual(set(result.chain_id), {'B'})

    def test_parallel_chains_do_not_share_download_temporary_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'fixture.pdb'
            strucio.save_structure(str(source), protein())
            barrier = threading.Barrier(2)
            destinations = []

            def download(url, destination, **kwargs):
                destinations.append(destination)
                barrier.wait(timeout=10)
                shutil.copyfile(source, destination)

            with mock.patch.object(fetch, '_download', side_effect=download):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    jobs = [executor.submit(fetch.fetch_and_save, name, root / 'output')
                            for name in ('1abcA', '1abcB')]
                    outputs = [job.result() for job in jobs]
            self.assertNotEqual(destinations[0], destinations[1])
            for output, chain in zip(outputs, ('A', 'B')):
                result = strucio.load_structure(output, model=1)
                self.assertEqual(set(result.chain_id), {chain})
            self.assertFalse(list((root / 'output').glob('.fetch_*')))

    def test_invalid_existing_file_is_retried_and_failures_are_not_saved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'fixture.pdb'
            strucio.save_structure(str(source), protein())
            output = root / '1abcA.pdb'
            output.write_text('HEADER partial download\n')
            with mock.patch.object(fetch, '_download', side_effect=lambda url, path, **kw: shutil.copyfile(source, path)):
                fetch.fetch_and_save('1abcA', root)
            self.assertTrue(fetch._has_protein_atoms(output))
            with mock.patch.object(fetch, '_download') as download:
                self.assertEqual(fetch.fetch_and_save('1abcA', root), str(output))
                download.assert_not_called()
            with mock.patch.object(fetch, '_download', side_effect=RuntimeError('offline')):
                with self.assertRaisesRegex(RuntimeError, 'offline'):
                    fetch.fetch_and_save('1abcC', root)
            self.assertFalse((root / '1abcC.pdb').exists())
            self.assertFalse(list(root.glob('.fetch_*')))

    def test_cli_reports_partial_failure_and_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ids, report = root / 'ids.txt', root / 'report.json'
            ids.write_text('1abcA\n\n1abcB\n1abcA\n')

            def download(name, *args):
                if name == '1abcB':
                    raise RuntimeError('missing chain')
                return str(root / f'{name}.pdb')

            with mock.patch.object(fetch, 'fetch_and_save', side_effect=download):
                with contextlib.redirect_stdout(io.StringIO()):
                    code = fetch.main(['-i', str(ids), '-o', str(root), '--cpu', '2',
                                       '--report', str(report)])
            self.assertEqual(code, 1)
            results = json.loads(report.read_text())
            self.assertEqual((results['requested'], results['available'], results['failed']), (2, 1, 1))
            self.assertEqual(results['failures'], {'1abcB': 'missing chain'})


if __name__ == '__main__':
    unittest.main()

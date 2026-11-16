"""Download single-chain structures for unZipro input lists."""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


_UNIPROT = re.compile(
    r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})"
)


def _parse_identifier(identifier):
    identifier = identifier.strip()
    if identifier.lower().endswith(".pdb"):
        identifier = identifier[:-4]
    if not identifier or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
        raise ValueError(f"Invalid structure identifier: {identifier!r}")
    af_match = re.fullmatch(r"AF[-_]([A-Z0-9]+)(?:-F[0-9]+-model_v[0-9]+)?", identifier)
    if af_match:
        return identifier, af_match.group(1), "A", "AFDB"
    if _UNIPROT.fullmatch(identifier):
        return identifier, identifier, "A", "AFDB"
    rcsb_match = re.fullmatch(r"([0-9][A-Za-z0-9]{3})([A-Za-z0-9]*)", identifier)
    if rcsb_match:
        return identifier, rcsb_match.group(1).lower(), rcsb_match.group(2) or "A", "RCSB"
    raise ValueError(f"Expected a PDB ID with chain or a UniProt accession, got {identifier!r}")


def _has_protein_atoms(path):
    """Do not accept empty files or error pages left by interrupted downloads."""
    try:
        with open(path) as handle:
            return any(line.startswith("ATOM  ") and line[12:16].strip() == "CA"
                       for line in handle)
    except (OSError, UnicodeError):
        return False


def _download(url, destination, timeout=60, retries=3):
    retry = Retry(total=retries, backoff_factor=0.5,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET",))
    # requests inherits HTTP(S)_PROXY from the caller's environment.
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=retry))
        with session.get(url, stream=True, timeout=(min(10, timeout), timeout)) as response:
            response.raise_for_status()
            with open(destination, "wb") as handle:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        handle.write(chunk)


def _extract_chain(source, destination, chain_id):
    # Delay importing Biotite until a structure actually needs processing.
    import biotite.structure.io as strucio

    structure = strucio.load_structure(str(source), model=1)
    selected = structure[structure.chain_id == chain_id]
    if len(selected) == 0:
        available = ", ".join(sorted(set(structure.chain_id)))
        raise ValueError(f"Chain {chain_id!r} is absent (available: {available})")
    # Legacy PDB supports only one character; retain the original ID in the
    # filename and use chain A for mmCIF author IDs such as 4v48/BK.
    if len(chain_id) > 1:
        selected.chain_id[:] = "A"
    strucio.save_structure(str(destination), selected)
    if not _has_protein_atoms(destination):
        raise ValueError(f"Chain {chain_id!r} contains no protein CA atoms")


def fetch_and_save(pdb, outdir="PDB_structures", verbose=False, timeout=60, retries=3):
    """Fetch one structure, extract its chain/first model, and save atomically.

    Identifiers can be PDB ID + author chain (``6vpcF``, ``8aw33``,
    ``4v48BK``) or a UniProt accession for AlphaFold DB. The filename always
    matches the supplied identifier so it can be resolved by GraphDataset.
    """
    identifier, pdb_name, chain_id, mode = _parse_identifier(pdb)
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{identifier}.pdb"
    if _has_protein_atoms(out_file):
        if verbose:
            print(f"[OK] {identifier}: already exists", flush=True)
        return str(out_file)

    if mode == "AFDB":
        urls = [("pdb", f"https://alphafold.ebi.ac.uk/files/AF-{pdb_name}-F1-model_v6.pdb")]
    else:
        # Multi-character author chain IDs require mmCIF. Use it as a fallback
        # for other entries unavailable in legacy PDB format as well.
        formats = ("cif",) if len(chain_id) > 1 else ("pdb", "cif")
        urls = [(suffix, f"https://files.rcsb.org/download/{pdb_name}.{suffix}")
                for suffix in formats]

    errors = []
    # Per-job paths avoid races when different chains share a PDB accession.
    # Keeping the temporary output on the destination filesystem allows an
    # atomic rename and never exposes partially written .pdb files.
    with TemporaryDirectory(prefix=f".fetch_{identifier}_", dir=output_dir) as temporary:
        for suffix, url in urls:
            source = Path(temporary) / f"source.{suffix}"
            output = Path(temporary) / "chain.pdb"
            try:
                _download(url, source, timeout=timeout, retries=retries)
                _extract_chain(source, output, chain_id)
                os.replace(output, out_file)
                if verbose:
                    remap = f" (chain {chain_id} saved as A)" if len(chain_id) > 1 else ""
                    print(f"[OK] {identifier}: {out_file}{remap}", flush=True)
                return str(out_file)
            except Exception as error:
                errors.append(f"{suffix}: {error}")
    raise RuntimeError(f"Could not fetch {identifier}: {'; '.join(errors)}")


def safe_fetch(pdb_id, pdb_dir):
    """Fetch one item for callers that prefer a missing result to an exception."""
    try:
        return fetch_and_save(pdb_id, pdb_dir, verbose=True)
    except Exception as error:
        print(f"[FAIL] {pdb_id}: {error}", flush=True)
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdblist', '-i', type=str, required=True)
    parser.add_argument('--outdir', '-o', type=str, required=True)
    parser.add_argument('--cpu', '-cpu', type=int, default=8, help='Concurrent downloads.')
    parser.add_argument('--timeout', type=float, default=60, help='Read timeout in seconds.')
    parser.add_argument('--retries', type=int, default=3, help='Retries for temporary HTTP failures.')
    parser.add_argument('--report', type=str, help='Optional JSON download report.')
    args = parser.parse_args(argv)
    if args.cpu < 1 or args.timeout <= 0 or args.retries < 0:
        parser.error('--cpu and --timeout must be positive; --retries must be non-negative.')

    with open(args.pdblist) as handle:
        pdblist = list(dict.fromkeys(line.strip() for line in handle if line.strip()))
    if not pdblist:
        parser.error('The structure list is empty.')

    available, failures = {}, {}
    with ThreadPoolExecutor(max_workers=args.cpu) as executor:
        jobs = {executor.submit(fetch_and_save, pdb, args.outdir, False,
                                args.timeout, args.retries): pdb for pdb in pdblist}
        for job in as_completed(jobs):
            pdb = jobs[job]
            try:
                available[pdb] = job.result()
                print(f"[OK] {pdb}: {available[pdb]}", flush=True)
            except Exception as error:
                failures[pdb] = str(error)
                print(f"[FAIL] {pdb}: {error}", flush=True)

    report = {"requested": len(pdblist), "available": len(available),
              "failed": len(failures), "files": dict(sorted(available.items())),
              "failures": dict(sorted(failures.items()))}
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Download summary: {len(available)}/{len(pdblist)} available, "
          f"{len(failures)} failed.", flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())

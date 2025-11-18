import os
import re
import argparse
import requests
from tempfile import gettempdir
from concurrent.futures import ThreadPoolExecutor
import biotite.structure.io as strucio
import biotite.structure as struc


def fetch_and_save(pdb, outdir="PDB_structures", verbose=False):
    """
    Download a PDB structure from AlphaFold DB or RCSB,
    extract the specified chain (default 'A' for AFDB),
    and save as a single-chain PDB.
    
    pdb: str, either 'UniProtID' for AFDB or 'PDBIDChain' for RCSB (e.g., '6vpcA')
    """
    os.makedirs(outdir, exist_ok=True)
    # Determine pdb_name and chain_id
    if pdb.startswith("AF_") or 'model' in pdb:
        pdb_name = pdb.split('-')[0]
        chain_id = 'A'
        mode = "AFDB"
    elif re.match(r'^[A-NR-Z][0-9A-Z]{5,9}$', pdb):
        pdb_name = pdb_name
        chain_id = 'A'
        mode = "AFDB"
    elif pdb.isalnum():
        pdb_name = pdb[:4]
        chain_id = pdb[4:]
        mode = "RCSB"
    else:
        raise ValueError("Invalid pdb identifier or mode")
    out_file = os.path.join(outdir, f"{pdb_name}{chain_id}.pdb")
    if os.path.exists(out_file):
        if verbose:
            print(f"✅ {out_file} already exists, skipping.")
        return out_file
    try:
        file_path = None
        # if afdb or rcsb:
        try:
            if mode.upper() == "AFDB":
                # import biotite.database.afdb as afdb
                # file_path = afdb.fetch(pdb_name, "pdb", gettempdir(), overwrite=True)
                file_path=None
            elif mode.upper() == "RCSB":
                import biotite.database.rcsb as rcsb
                file_path = rcsb.fetch(pdb_name, "pdb", gettempdir(), overwrite=True)
        except Exception as e:
            pass
            # print(f"⚠️ Biotite fetch failed ({e}), fallback to requests")

        # Fallback to requests if Biotite failed or unavailable
        if file_path is None:
            if mode.upper() == "AFDB":
                url = f"https://alphafold.ebi.ac.uk/files/AF-{pdb_name}-F1-model_v6.pdb"
            elif mode.upper() == "RCSB":
                url = f"https://files.rcsb.org/download/{pdb_name}.pdb"
            file_path = os.path.join(gettempdir(), f"{pdb_name}.pdb")
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        # Extract specified chain using Biotite if possible
        if strucio:
            struct = strucio.load_structure(file_path)
            struct_chain = struct[struct.chain_id == chain_id]
            strucio.save_structure(out_file, struct_chain)
            if verbose:
                print(f"✅ Saved single chain to {out_file}")
            return out_file
        else:
            # If Biotite unavailable, just move the file
            final_path = os.path.join(outdir, os.path.basename(file_path))
            if not os.path.exists(final_path):
                os.rename(file_path, final_path)
            if verbose:
                print(f"⚠️ Biotite not available, saved full structure to {final_path}")
            return final_path
    except Exception as e:
        if verbose:
            print(f"⚠️ Download failed ({e}). Skip {pdb}!")

def safe_fetch(pdb_id, pdb_dir):
    try:
        fetch_and_save(pdb_id, pdb_dir)
        print(f"✅ Downloaded {pdb_id}")
    except Exception as e:
        print(f"❌ Failed to download {pdb_id}: {e}")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdblist', '-i', type=str, required=True)
    parser.add_argument('--outdir', '-o', type=str, required=True)
    parser.add_argument('--cpu', '-cpu', type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    with open(args.pdblist, 'r') as f:
        pdblist = [line.strip() for line in f]

    with ThreadPoolExecutor(max_workers=args.cpu) as executor:
        futures = [executor.submit(fetch_and_save, pdb, args.outdir) for pdb in pdblist]
        for f in futures:
            f.result()

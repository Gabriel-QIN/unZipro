#!/usr/bin/env python3
import requests
import os
import time
import tarfile
import argparse
import subprocess

def parse_pdb_input(pdb_input):
    """
    Parse pdb_input which can be either:
    - A filename (ending with .txt) containing PDB names (one per line)
    - A comma-separated list of PDB names
    """
    if os.path.isfile(pdb_input):
        with open(pdb_input) as f:
            pdb_list = [line.strip() for line in f if line.strip()]
    else:
        # Split by comma and strip spaces
        pdb_list = [p.strip() for p in pdb_input.split(",") if p.strip()]
    return pdb_list

def submit_pdb_to_foldseek(pdb_list, outdir, pdb_dir="data/benchmark/Mutation/ProteinGym_AF2_structures",
                            use_wget=False, use_aria2c=True, wait_time=60, only_download=False):
    """
    Submit PDB files to Foldseek Web API and generate download scripts.
    
    Args:
        pdb_list (list): pdb_list (list): List of PDB IDs (without .pdb)
        outdir (str): Directory to save download scripts.
        pdb_dir (str): Directory containing PDB structures.
        use_wget (bool): Whether to generate wget commands.
        use_aria2c (bool): Whether to generate aria2c commands.
        wait_time (int): Seconds to wait between submissions.
        only_download (bool): Only perform download if set to `True`.
    """
    os.makedirs(outdir, exist_ok=True)
    url = "https://search.foldseek.com/api/ticket"
    if not only_download:
        for i, pdb in enumerate(pdb_list, start=1):
            file_path = os.path.join(pdb_dir, f"{pdb}.pdb")
            pdb_name = os.path.basename(file_path).replace('.pdb', '')

            with open(file_path, "rb") as f:
                files = {"q": f}
                data = [
                    ("mode", "3diaa"),
                    ("database[]", "afdb50"),
                    ("database[]", "pdb100"),
                ]
                response = requests.post(url, files=files, data=data)

            print(f"Status: {response.status_code} | Response: {response.text}")
            if response.status_code != 200:
                raise RuntimeError(f"Foldseek error: {response.status_code}")

            resp_json = response.json()
            if "id" not in resp_json:
                raise RuntimeError(f"No id found, full response: {resp_json}")
            
            ticket = resp_json["id"]
            print(f'Query {pdb_name} | Ticket: {ticket}')

            download_url = f"https://search.foldseek.com/api/result/download/{ticket}"
            save_path = f"{outdir}/{pdb_name}.tar.gz"
            os.makedirs(f'{outdir}/cmd/', exist_ok=True)
            # Write download commands to script
            with open(os.path.join(f'{outdir}/cmd/', f"{pdb_name}.txt"), 'w') as f:
                if use_wget:
                    f.write(f'wget -O {save_path} {download_url}\n')
                elif use_aria2c:
                    f.write(f'aria2c -x 16 -s 16 -o {save_path} {download_url}\n')
                f.write(f'echo {pdb_name} && mkdir -p {outdir}/{pdb_name} && tar -zxvf {save_path} -C {outdir}/{pdb_name}\n')

            print(f"Processed {i}/{len(pdb_list)}: {pdb_name}. Waiting {wait_time}s...")
            # Foldseek server has rate limiting, so we add a delay (60-120s) to avoid overloading the server.
            time.sleep(wait_time) # Wait before submitting the next request
    print("All tasks submitted. Now waiting for results...")
    for i, pdb_name in enumerate(pdb_list, start=1):
        cmd = os.path.join(f'{outdir}/cmd/', f"{pdb_name}.txt")
        print(f"Executing download script for {pdb_name}...")
        download_successful = False
        retries = 5  # Number of retry attempts
        while not download_successful and retries > 0:
            # Use os.system() to execute the shell script
            os.system(f"bash {cmd}")
            # Check if the file exists after running the script
            if os.path.exists(f"{outdir}/{pdb_name}.tar.gz"):
            # if os.path.exists(f"{outdir}/{pdb_name}/alis_pdb100.m8") or os.path.exists(f"{outdir}/{pdb_name}/alis_afdb50.m8"):
                print(f"Download completed for {pdb_name}.")
                download_successful = True
            else:
                retries -= 1
                if retries > 0:
                    print(f"Retrying download for {pdb_name}... {retries} retries left.")
                    time.sleep(wait_time)  # Wait before retrying
                else:
                    print(f"Failed to download {pdb_name} after multiple attempts.")
    print("All tasks processed and results downloaded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit PDB files to Foldseek API and generate download scripts")
    parser.add_argument('--pdb', type=str, required=True, help="PDB names (comma-separated) or a file containing PDB IDs")
    parser.add_argument('--outdir', type=str, required=True, help="Directory to save download scripts")
    parser.add_argument('--pdb_dir', type=str, default="data/benchmark/Mutation/ProteinGym_AF2_structures/", help="Directory of PDB files")
    parser.add_argument('--use_wget', action='store_true', help="Generate wget commands instead of aria2c")
    parser.add_argument('--wait_time', type=int, default=120, help="Seconds to wait between submissions")
    parser.add_argument('--only_download', action="store_true", help="Whether execute the download commands")
    args = parser.parse_args()
    
    pdb_list = parse_pdb_input(args.pdb)
    submit_pdb_to_foldseek(
        pdb_list=pdb_list,
        outdir=args.outdir,
        pdb_dir=args.pdb_dir,
        use_wget=args.use_wget,
        use_aria2c=not args.use_wget,
        wait_time=args.wait_time,
        only_download=args.only_download
    )

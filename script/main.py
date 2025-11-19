#!/usr/bin/env python
# ============================================================
#  Script: main.py
#  Description: Streamlined pipeline for unZipro mutation prioritization
# ============================================================

import os
import sys
import time
import json
import argparse
from tempfile import gettempdir
from concurrent.futures import ThreadPoolExecutor
dir_script = osp.dirname(osp.realpath(__file__))
sys.path.append(dir_script+'/../')
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import learn2learn as l2l
from tqdm import tqdm
import requests
import biotite.structure as struc
import biotite.structure.io as strucio

# ------------------------------------------------------------
#  Setup project paths
# ------------------------------------------------------------
dir_script = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_script, '..'))

# ------------------------------------------------------------
#  Local imports
# ------------------------------------------------------------
from utils import *
from model import unZipro, weights_init
from foldseek_api import submit_pdb_to_foldseek, parse_pdb_input
from parse_foldseek_results import parse_mmseqs, write_ids
from fetch_PDB_parallel import fetch_and_save, safe_fetch
from unZipro_finetuning import unZipro_finetune
from unZipro_mutation import infer_single_protein

# ============================================================
#  Main pipeline (below)
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="A streamlized pipeline for unZipro mutation prioritization")
    parser.add_argument('--pdb', type=str, required=True, help="PDB names (comma-separated) or a file containing PDB IDs")
    parser.add_argument('--pdb_dir', type=str, default="data/example/", help="Directory of PDB files")
    parser.add_argument('--work_dir', type=str, default="data/tmp/", help="Working directory")
    parser.add_argument('--pretrained', '-pre', action='store_true', help='Use pretrained inverse folding model for mutation prioritization (skip finetuning).')
    parser.add_argument('--use_wget', action='store_true', help="Use wget instead of aria2c for data download")
    parser.add_argument('--wait_time', type=int, default=120, help="Seconds to wait between submissions")
    parser.add_argument('--train_size', type=float, default=100, help="Train size for meta-transfer learning")
    parser.add_argument('--cpu', '-cpu', type=int, default=8, metavar='[Int]', help='CPU processors for data loading.(default:8).')
    parser.add_argument('--gpu', '-gpu', type=int, default=0, metavar='[Int]', help='GPU id.(default:0).')
    parser.add_argument('--cpu_only', action='store_true', help='Use CPU for fine-tuning instead of GPU.')
    parser.add_argument('--config_path', type=str, default='config/unZipro_pretrain.json', help='Model config.')
    parser.add_argument('--param', type=str, default='Models/unZipro_params.pt', metavar='[File]', help='Pre-trained parameter file.')
    parser.add_argument('--epochs', '-e', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--adapt_lr', '-lr', type=float, default=1e-6, help='Adapt learning rate. ')
    parser.add_argument('--meta_lr', '-mlr', type=float, default=1e-6, help='Meta learning rate. ')
    parser.add_argument('--adapt_step', '-as', type=int, default=10, help='Adaptation steps. ')
    parser.add_argument('--batchsize', '-bs', type=int, default=5, metavar='[Int]', help='Batch size.')
    parser.add_argument('--patience', '-pt', type=int, default=5, metavar='[Int]', help='Early stopping patience.')
    parser.add_argument('--noise', '-ni', type=float, default=0.01, metavar='float', help='Training noise.')
    parser.add_argument('--nneighbor', '-nb', type=int, default=20, metavar='int', help='Number of node neighbors.')
    parser.add_argument("--save_model_ckp", action='store_true', help="Save model checkpoints for each epochs. ")
    parser.add_argument("--skip_foldseek", action='store_true', help="Skip Foldseek search if you already have them. ")
    parser.add_argument('--outdir', '-o', type=str, default='data/example/', help='output directory.')
    parser.add_argument('--probs', '-p', action='store_true', help='Output sequence probabilities')
    parser.add_argument('--logits', '-l', action='store_true', help='Output sequence logits')
    parser.add_argument('--rank_by_prob', '-rp', action='store_true', help='Rank by mutation prob')
    parser.add_argument('--res', '-res', type=str, default=None, help='Define residues to output prob, separated by `,` (e.g., 83,123)')

    args = parser.parse_args()

    # parse input PDB
    pdb_list = parse_pdb_input(args.pdb)

    for pdb_name in pdb_list:
        start = time.perf_counter()
        if not args.pretrained:
            # Step 1: foldseek search through web server API
            if not args.skip_foldseek:
                print(f"INFO | Step 1: start to retrieve similar structures using Foldseek!")
                _ = submit_pdb_to_foldseek(
                    pdb_list=pdb_list,
                    outdir=args.work_dir,
                    pdb_dir=args.pdb_dir,
                    use_wget=args.use_wget,
                    use_aria2c=not args.use_wget,
                    wait_time=args.wait_time,
                    only_download=False
                )

            # Step 2: dataset split
            print(f"INFO | Step 2: split data into meta-training and testing!")
            try:
                m8_path = os.path.join(args.work_dir, f'{pdb_name}/alis_pdb100.m8')
                savepath = os.path.join(args.work_dir, f'{pdb_name}_pdb100.csv')
                data_dir = os.path.join(args.work_dir, f'{pdb_name}/')
                pdb_id_path = os.path.join(data_dir, 'PDB_IDs.txt')
                os.makedirs(data_dir, exist_ok=True)
                train_path = os.path.join(data_dir, f'train.csv')
                test_path = os.path.join(data_dir, f'test.csv')
                train, test, download_pdblist, num_pdbs, num_af_pdbs = parse_mmseqs(m8_path, savepath, train_path, test_path, train_size=args.train_size, include_af2=True)
                if len(train) < 10 or len(test) < 10:
                    raise ValueError(f"Too little homologs for {pdb_name} Train size {len(train)} | Test size {len(test)}!")
                if len(train) < 100 or len(test) < 20:
                    print(f'Warning! [{pdb_name}]: Train size {len(train)} | Test size {len(test)}')
            except Exception as e:
                print(f'Error in {pdb_name}: {e}')

            # Step 3: data download
            print(f"INFO | Step 3: start to download structural data!")
            pdb_dir = os.path.join(args.work_dir, 'pdb')
            os.makedirs(pdb_dir, exist_ok=True)
            print(f"INFO | Found {len(train)} meta-training structures and {len(test)} meta-testing structures!")
            with ThreadPoolExecutor(max_workers=args.cpu) as executor:
                # futures = [executor.submit(fetch_and_save, pdb, pdb_dir) for pdb in download_pdblist]
                futures = [executor.submit(safe_fetch, pdb, pdb_dir) for pdb in download_pdblist]
                for f in futures:
                    f.result()

            # Step 4: Finetuning
            print(f"INFO | Step 4: start unZipro finetuning!")
            model_store_dir = os.path.join(args.work_dir, 'model/')
            cache_dir = os.path.join(args.work_dir, 'tmp/')
            # unZipro finetuning
            model_param = unZipro_finetune(train_path, test_path, config_path=args.config_path, pdb_dir=pdb_dir, model_store_dir=model_store_dir, 
                            param_file=args.param, project_name=pdb_name, epochs=args.epochs, 
                            adapt_lr=args.adapt_lr, meta_lr=args.meta_lr, adapt_step=args.adapt_step, 
                            batchsize=args.batchsize, cpu=args.cpu, gpu=args.gpu,cpu_only=args.cpu_only,
                            noise=args.noise, nneighbor=args.nneighbor, patience=args.patience, cache_dir=cache_dir, save_model_ckp=args.save_model_ckp)
            os.system(f'rm -rf {cache_dir}/*')
        # Step 5: Mutation prioritation
        if args.pretrained:
            model_param = args.param
            print(f"INFO | start mutation prioritation!")
            cache_dir = os.path.join(args.work_dir, 'tmp/')
        else:
            model_param = f'{model_store_dir}/{pdb_name}.pt'
            print(f"INFO | Step 5: start mutation prioritation!")
        os.makedirs(f'{args.outdir}', exist_ok=True)
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and not args.cpu_only else 'cpu')
        with open(args.config_path, "r") as f:
            data = json.load(f)
        model_config = Config(**data)
        model = unZipro(model_config).to(device)
        state_dict = torch.load(model_param, map_location=torch.device(device), weights_only=True)
        try:
            model.load_state_dict(state_dict)
        except:
            new_state_dict = {k.replace("module.", ""):v for k,v in state_dict.items()}
            model.load_state_dict(new_state_dict)
        # dataloader setup
        dataset = GraphDataset(datalist=[osp.join(args.pdb_dir, f'{pdb_name}.pdb')], nneighbor=args.nneighbor, noise=0, cache_dir=cache_dir)
        loader = get_loader(dataset=dataset, batchsize=1)
        criterion = nn.CrossEntropyLoss().to(device)
        model.eval()
        df_list= infer_single_protein(model, criterion, loader, args.pdb_dir, args.outdir, temperature=1.0, device=device, output_prob=args.probs, output_logits=args.logits, rank_by_prob=args.rank_by_prob, res=args.res)
        end = time.perf_counter()
        print(f"{pdb_name} | {(end - start):.4f}s.")

if __name__=='__main__':
    main()
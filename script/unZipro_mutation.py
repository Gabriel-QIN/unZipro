#! /usr/bin/env python

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import time
import json
import os.path as osp
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from utils import *
from model import unZipro, weights_init

def infer_single_protein(model, criterion, loader, pdbdir, outdir, temperature=1.0, device=torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu'), output_prob=True, output_logits=False, rank_by_prob=True, res=None):
    """
    Infer on a batch of proteins, output the in silico mutation scores
    Args:
        model: torch model
        criterion: loss function
        loader: DataLoader
        pdbdir: output directory
        temperature: softmax T
        device: 'cuda' or 'cpu'
        
    Returns:
        df: a list of pandas DataFrame for residue-level mutation scanning
        avg_loss
        recovery
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_count = 0
    df_list = []
    time_list = []
    with torch.no_grad():
        for step, batch in enumerate(loader):
            t0 = time.time()
            dat1, dat2, dat3, target, mask, name, num = batch
            name = osp.basename(name).split('.')[0]
            dat1 = dat1.squeeze(0).to(device, non_blocking=True)
            dat2 = dat2.squeeze(0).to(device, non_blocking=True)
            dat3 = dat3.squeeze(0).to(device, non_blocking=True)
            target = target.squeeze(0).to(device, non_blocking=True)
            mask = mask.squeeze(0).to(device, non_blocking=True)
            with autocast("cuda"):
                outputs = model(dat1, dat2, dat3)
                loss = criterion(outputs[mask], target[mask])
            
            predicted = torch.argmax(outputs, dim=1)
            count = target[mask].size(0)
            correct = (target[mask] == predicted[mask]).sum().item()
            
            total_count += count
            total_correct += correct
            total_loss += loss.item() * count
            # print(dat1.shape, dat2.shape, dat3.shape, target, mask, name ,num)
            print(f"[INFO] {name} | Recovery: {correct / count * 100:.2f}% | Loss: {loss.item():.4f}")
            res_list = get_pdb_info(name, pdbdir)
            alllist = []
            aa_ind = 0
            prev_res_ind = 0
            break_start = 0
            for ind, o in enumerate(outputs):
                if ind == 0 or ind == len(mask)-1:
                    continue
                target_idx = target[ind].item()
                ori_list_item = target_idx
                if mask[ind]:
                    aa_type = torch.argmax(o).detach().cpu()
                    aa_logits = torch.max(o).detach().cpu()
                    aa_prob = torch.nn.functional.softmax(o/temperature, dim=0).detach().cpu()
                    
                    mutant = f"{res_map[target_idx]}{ind}{res_map[aa_type.item()]}"
                    mutant_auth = f"{res_map[target_idx]}{res_list[aa_ind][2]}{res_map[aa_type.item()]}"
                    pdbinfo = res_list[aa_ind][:3]
                    prev_res_ind = pdbinfo[-1]

                    mutate_prob = 0 if target_idx == aa_type.item() else aa_prob.max().item()
                    wt_prob = aa_prob[target_idx]
                    wt_logit = o[target_idx]
                    logit_log_ratio = torch.log2(aa_prob.max() / wt_prob)

                    residue_info = [*pdbinfo, ind, one2three[res_map[target_idx]], res_map[target_idx],
                                    one2three[res_map[aa_type.item()]], res_map[aa_type.item()],
                                    mutant, mutant_auth, mutate_prob, aa_prob.max().item(), 
                                    wt_prob.item(), aa_logits.item(), wt_logit.item(), 
                                    abs(logit_log_ratio.item()), *aa_prob.tolist(), *o.tolist()]
                    alllist.append(residue_info)
                    aa_ind += 1
                else:
                    pdbinfo = [*res_list[0][:2], res_list[aa_ind][2]]
                    ppp = [0.]*20
                    residue_info = [*pdbinfo, ind, one2three[res_map[target_idx]], res_map[target_idx],
                                    one2three[res_map[target_idx]], res_map[target_idx],
                                    f"{res_map[target_idx]}{ind}{res_map[target_idx]}", 
                                    f"{res_map[target_idx]}{res_list[aa_ind][2]}{res_map[target_idx]}",
                                    0, 0, 0, 0, 0, *ppp, *ppp]
                    alllist.append(residue_info)
                    aa_ind += 1

            header = ['pdb', 'chain', 'auth_idx', 'index', 'target_3', 'target_1', 
                      'predict_3', 'predict_1', 'mutation_', 'mutation', 'mut_prob', 
                      'model_prob', 'wt_prob', 'mut_logit', 'wt_logit', 'logit_ratio', 
                      *[f'prob_{a}' for a in mapped_20.keys()], 
                      *[f'logit_{a}' for a in mapped_20.keys()]]
            df = pd.DataFrame(alllist, columns=header)
            selected_header = ['pdb', 'chain', 'auth_idx', 'mutation', 'mut_prob', 'model_prob', 'wt_prob', 'mut_logit', 'wt_logit', 'logit_ratio']
            selected_df = df[selected_header]
            savepath = osp.join(outdir, f'{name}.info.csv')
            selected_df.to_csv(savepath,index=False)
            print(f"[INFO] Saved in silico mutation scores to {savepath}!")
            if rank_by_prob:
                selected_df = selected_df.sort_values(by='mut_prob', ascending=False)
                savepath = osp.join(outdir, f'{name}.info_rank_by_prob.csv')
                selected_df.to_csv(savepath,index=False)
                print(f"[INFO] Saved ranked scores to {savepath}!")
            df_list.append(selected_df)
            if output_prob:
                selected_header = ['pdb', 'chain', 'auth_idx',  'mutation', 'mut_prob', 'model_prob', 'wt_prob', 'mut_logit', 'wt_logit',] + [f'prob_{a}' for a in mapped_20.keys()]
                prob_df = df[selected_header]
                savepath = osp.join(outdir, f'{name}.info_probs.csv')
                prob_df.to_csv(savepath,index=False)
                print(f"[INFO] Saved per-residue probability matrix  to {savepath}!")
            if output_logits:
                selected_header = ['pdb', 'chain', 'auth_idx',  'mutation', 'mut_prob', 'model_prob', 'wt_prob', 'mut_logit', 'wt_logit',] + [f'logit_{a}' for a in mapped_20.keys()]
                logits_df = df[selected_header]
                savepath = osp.join(outdir, f'{name}.info_logits.csv')
                logits_df.to_csv(savepath,index=False)
                print(f"[INFO] Saved per-residue logits to {savepath}!")
            if res is not None:
                poslist = [int(r.strip()) for r in res.split(',')]
                columns = ['pdb', 'chain', 'auth_idx', 'mutation', 'mut_prob', 'model_prob', 'wt_prob'] + [f'prob_{a}' for a in mapped_20.keys()]
                res_df = df[columns].copy()
                res_df['auth_idx'] = res_df['auth_idx'].astype(str).str.strip().astype(int)
                existing_pos = [p for p in poslist if p in res_df['auth_idx'].values]
                res_df = res_df[res_df['auth_idx'].isin(existing_pos)].copy()
                res_df['sort_order'] = res_df['auth_idx'].apply(lambda x: existing_pos.index(x))
                res_df = res_df.sort_values('sort_order').drop(columns='sort_order')
                savepath = osp.join(outdir, f'{name}.res_prob.csv')
                res_df.to_csv(savepath, index=False)
                print(f"[INFO] Saved defined residue probs to {savepath}!")
            dt = time.time() - t0
            time_list.append([name, dt])
    time_df = pd.DataFrame(time_list)
    time_df.columns = ['Name', 'Time']
    print(time_df)
    time_df.to_csv(f'unZipro_time.csv', sep='\t')

    total_time = time_df['Time'].sum()
    print(f"Total time: {total_time:.3f} s")

    avg_time = time_df['Time'].mean()
    print(f"Average per item: {avg_time:.3f} s")
    avg_loss = total_loss / total_count
    acc = total_correct / total_count
    return df_list
    
if __name__=='__main__':
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', '-i', type=str, metavar='File',
                        help='Input PDB list.')
    parser.add_argument('--pdbdir', '-dir' , type=str, default='data/example/',
                        help='PDB directory.')
    parser.add_argument('--gpu', '-gpu', type=int, default=0, metavar='[Int]',
                        help='GPU id.(default:0).')
    parser.add_argument('--param', '-m', type=str, default="Models/unZipro_params.pt",
                        help='Parameter file.')
    parser.add_argument('--config_path', type=str, default='config/unZipro_pretrain.json', 
                        help='Model config.')
    parser.add_argument('--outdir', '-o', type=str, default='Design',
                        help='output directory.')
    parser.add_argument('--name', '-n', type=str, default='example_crispr',
                        help='Name for saving designed file.')
    parser.add_argument('--nneighbor', '-nb', type=int, default=20, metavar='[Int]',
                        help='Number of node neighbors. (default:20)')
    parser.add_argument('--cachedir', '-cd', type=str, default=None, metavar='[DIR]',
                        help='Cache feature directory.')
    parser.add_argument('--probs', '-p', action='store_true', help='Output sequence probabilities')
    parser.add_argument('--logits', '-l', action='store_true', help='Output sequence logits')
    parser.add_argument('--rank_by_prob', '-rp', action='store_true', help='Rank by mutation prob')
    parser.add_argument('--res', '-res', type=str, default=None, help='Define residues to output prob, separated by `,` (e.g., 83,123)')
    ##  arguments  ##
    args = parser.parse_args()
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    name = args.name
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    #  check input
    pdb_file = args.pdb
    pdbdir = args.pdbdir
    param = args.param
    nneighbor = args.nneighbor
    probs = args.probs
    logits = args.logits
    output_prob = args.probs
    output_logits = args.logits
    rank_by_prob = args.rank_by_prob
    res = args.res
    assert osp.exists(param), f"Model parameters {param} not found."
    assert osp.isdir(pdbdir), f"Please check your PDB directory {pdbdir}!"
    ## Model Setup ##
    with open(args.config_path, "r") as f:
        data = json.load(f)
    model_config = Config(**data)
    model = unZipro(model_config).to(device)
    state_dict = torch.load(param, map_location=torch.device(device), weights_only=False)
    try:
        model.load_state_dict(state_dict)
    except:
        new_state_dict = {k.replace("module.", ""):v for k,v in state_dict.items()}
        model.load_state_dict(new_state_dict)

    # --- parse pdb input ---
    if osp.isfile(pdb_file):
        # if it's a text file, read all lines
        with open(pdb_file, "r") as f:
            pdblist = [line.strip() for line in f if line.strip()]
    else:
        # otherwise, treat it as a comma-separated list
        pdblist = [x.strip() for x in pdb_file.split(",") if x.strip()]
    datalist = [osp.join(pdbdir, f'{pdb}.pdb') for pdb in pdblist]
    # dataloader setup
    dataset = GraphDataset(datalist=datalist, nneighbor=nneighbor, noise=0, cache_dir=args.cachedir)
    loader = get_loader(dataset=dataset, batchsize=1)

    criterion = nn.CrossEntropyLoss().to(device)
    # test
    start = time.perf_counter()
    model.eval()
    df_list= infer_single_protein(model, criterion, loader, pdbdir, outdir, temperature=1.0, device=device, output_prob=output_prob, output_logits=output_logits, rank_by_prob=rank_by_prob, res=res)
    end = time.perf_counter()
    print(f"{name} | {(end - start):.4f}s for {len(loader)} proteins.")

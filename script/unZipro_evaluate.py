#! /usr/bin/env python

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import time
import json
from os import path
import os.path as osp
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from utils import *
from parser import parse_args
from model import unZipro, weights_init

def beam_search(prob_matrix, beam_width=1):
    """
    Beam search algorithm for sequence prediction.

    Args:
    - prob_matrix (numpy.ndarray): The probability matrix of shape [L, K].
    - beam_width (int): The number of top sequences to keep (beam size).

    Returns:
    - sequences (list of lists): The top-k sequences.
    - sequence_scores (list of floats): The corresponding scores for the top-k sequences.
    """
    L, K = prob_matrix.shape  # L: sequence length, K: number of classes
    
    # Initialize the beam with an empty sequence and zero log probability
    sequences = [[[], 0.0]]  # Each element is a tuple of (sequence, log_prob)
    
    for t in range(L):
        all_candidates = []
        
        # Expand each sequence in the current beam
        for seq, score in sequences:
            for i in range(K):
                # Get the probability of the i-th class at time step t
                prob = prob_matrix[t, i]
                
                # Calculate new log probability (additive in log space)
                new_score = score + np.log(prob + 1e-10)  # Avoid log(0) by adding a small epsilon
                
                # Create a new candidate sequence
                candidate = (seq + [i], new_score)
                
                # Append to all candidates
                all_candidates.append(candidate)
        
        # Sort candidates by score (log probability) and select top beam_width sequences
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        sequences = ordered[:beam_width]
    
    # Extract the sequences and their scores
    final_sequences = [seq for seq, score in sequences]
    final_scores = [score for seq, score in sequences]
    
    return final_sequences, final_scores

def compute_pairwise_diversity(seq1, seq2):
    """
    Compute pairwise diversity D_ij between two sequences.

    LaTeX Formula:
        D_{ij} = \frac{1}{n} \sum_{l=1}^{n} \mathbb{1}(r_{i,l} \ne r_{j,l})

    where:
        - n is the sequence length
        - r_{i,l} is the l-th residue of the i-th sequence
        - \mathbb{1} is the indicator function (1 if different, else 0)

    Args:
        seq1 (str): First amino acid sequence
        seq2 (str): Second amino acid sequence

    Returns:
        float: Pairwise diversity score
    """
    assert len(seq1) == len(seq2), "Sequences must be of the same length"
    n = len(seq1)
    diff_count = sum(res1 != res2 for res1, res2 in zip(seq1, seq2))
    return diff_count / n

def compute_overall_diversity(sequences):
    """
    Compute overall diversity among a set of sequences.

    LaTeX Formula:
        \text{Div} = \frac{1}{m^2} \sum_{i=1}^{m} \sum_{j=1}^{m} D_{ij}

    where:
        - m is the total number of sequences
        - D_{ij} is the pairwise diversity between sequences i and j

    Args:
        sequences (List[str]): List of designed sequences

    Returns:
        float: Overall diversity score
    """
    m = len(sequences)
    total_div = 0.0
    for i in range(m):
        for j in range(m):
            total_div += compute_pairwise_diversity(sequences[i], sequences[j])
    return total_div / (m * m)

def calculate_metrics(true_seq, pred_seq, pred_probs):

    true_seq_encoded = [ord(c) for c in true_seq]
    pred_seq_encoded = [ord(c) for c in pred_seq]


    true_seq_tensor = torch.tensor(true_seq_encoded, dtype=torch.long)
    pred_seq_tensor = torch.tensor(pred_seq_encoded, dtype=torch.long)


    recovery = (true_seq_tensor == pred_seq_tensor).float().mean().item()

    confidence = pred_probs.max(dim=-1)[0].mean().item()

    return {
        "Recovery": recovery,
        "Confidence": confidence
    }

def inference(model, criterion, device, valid_loader, sampling_strategy='beam_search', num_samples=1, temperature=1):
    model.eval()
    metrics = []
    with torch.no_grad():
        for batch_idx, (node, edge, adjmax, target, mask, name, num) in enumerate(valid_loader):
            t0 = time.time()
            node = node.squeeze(0).to(device)
            edge = edge.squeeze(0).to(device)
            adjmax = adjmax.squeeze(0).to(device)
            target = target.squeeze(0).to(device)
            mask = mask.squeeze(0).to(device)
            outputs = model(node, edge, adjmax)
            loss = criterion(outputs[mask], target[mask])
            # === Sample  sequence ===
            probs = torch.nn.functional.softmax(outputs/temperature, dim=-1).detach().cpu()
            native_sequence = ''.join([res_map[i] for i in target[1:-1].cpu().tolist()])
            true_seq_encoded = [ord(c) for c in native_sequence]
            if sampling_strategy == 'argmax':
                mask = mask.detach().cpu()
                target = target.detach().cpu()
                probs, predict_indices = torch.max(probs, 1)
                confidence = probs.max(dim=-1)[0].mean().item()
                design_sequence = ''.join([res_map[i] for i in predict_indices[1:-1].tolist()])
                count = target[mask].size()[0]
                correct = (target[mask]==predict_indices[mask]).sum().item()
                recovery = correct / count
                t1 = time.time()
                time_elapsed = round(float(t1-t0), 4)
                metrics.append([osp.basename(name), 0, recovery, confidence, time_elapsed, native_sequence, design_sequence])
            elif sampling_strategy == 'beam_search':
                sequences, scores = beam_search(probs.detach().cpu(), num_samples)
                for ix, (sampled_seq, score) in enumerate(zip(sequences, scores)):
                    design_sequence = ''.join([res_map[i] for i in sampled_seq[1:-1]])
                    # print(design_sequence)
                    design_log_probs = probs[torch.arange(outputs.size(0)), sampled_seq]
                    pred_probs = probs[sampled_seq]
                    confidence = pred_probs.max(dim=-1)[0].mean().item()
                    true_seq_encoded = [ord(c) for c in native_sequence]
                    pred_seq_encoded = [ord(c) for c in design_sequence]
                    true_seq_tensor = torch.tensor(true_seq_encoded, dtype=torch.long)
                    pred_seq_tensor = torch.tensor(pred_seq_encoded, dtype=torch.long)
                    recovery = (true_seq_tensor == pred_seq_tensor).float().mean().item()
                    t1 = time.time()
                    time_elapsed = round(float(t1-t0), 4)
                    metrics.append([osp.basename(name), ix, recovery, confidence, time_elapsed, native_sequence, design_sequence])
            elif sampling_strategy == 'multinomial':
                # Multinomial sampling based on probability distribution
                sampled_indices = torch.multinomial(probs, num_samples=num_samples)
                for ix in range(sampled_indices.size(-1)):
                    sampled_seq = sampled_indices[:,ix]
                    design_log_probs = probs[torch.arange(outputs.size(0)), sampled_seq]
                    pred_probs = probs[sampled_seq]
                    confidence = pred_probs.max(dim=-1)[0].mean().item()
                    design_sequence = ''.join([res_map[i] for i in sampled_seq[1:-1].cpu().tolist()])
                    pred_seq_encoded = [ord(c) for c in design_sequence]
                    true_seq_tensor = torch.tensor(true_seq_encoded, dtype=torch.long)
                    pred_seq_tensor = torch.tensor(pred_seq_encoded, dtype=torch.long)
                    recovery = (true_seq_tensor == pred_seq_tensor).float().mean().item()
                    t1 = time.time()
                    time_elapsed = round(float(t1-t0), 4)
                    metrics.append([osp.basename(name), ix, recovery, confidence, time_elapsed, native_sequence, design_sequence])
    df = pd.DataFrame(metrics)
    df.columns = ['PDB_code', 'Design_idx', 'Recovery', 'Confidence', 'Time', 'Nat_seq', 'Design_seq']
    return df

if __name__=='__main__':
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='unZipro_seq_design',
                        help='Name for saving designed file.')
    parser.add_argument('--sampling_strategy', type=str, default='argmax',choices=['argmax', 'beam_search', 'multinomial'],
                        help='Sampling strategy. Choose from: argmax, beam_search, multinomial.')
    parser.add_argument('--input', '-i', type=str, metavar='File', required=True,
                        help='List of testing data.')
    parser.add_argument('--pdbdir', '-pk', type=str, default='data/pretrained/PDB', metavar='[DIR]',
                            help='PDB directory.')
    parser.add_argument('--outdir', type=str, default='outputs/seq_design',
                        help='output directory.')

    parser.add_argument('--cache_dir', type=str, default=None, metavar='[Directory]',
                        help='Directory where pkl files are be stored.')
    parser.add_argument('--config_path', type=str, default='config/unZipro_pretrain.json', metavar='[Directory]',
                        help='Directory where pkl files are be stored.')
    
    parser.add_argument('--param', type=str, default="Models/unZipro_params.pt",
                        help='Parameter file.')
    parser.add_argument('--cpu', '-cpu', type=int, default=16, metavar='int',
                             help='CPU cores for data loading.')
    parser.add_argument('--gpu', '-gpu', type=int, default=0, metavar='[Int]',
                        help='GPU id.(default:0).')
    parser.add_argument('--noise', '-ni', type=float, default=0, metavar='float',
                             help='Training noise.')
    parser.add_argument('--nneighbor', '-nb', type=int, default=20, metavar='int',
                             help='Number of node neighbors.')
    parser.add_argument('--temprature', '-t', type=float, default=1.0, metavar='[Float]',
                        help='Temprature for predicting amino acid probability by model.')
    parser.add_argument('--cutoff', '-c', type=float, default=0, metavar='[Float]',
                        help='Cut-off temprature for amino acid substitution by model.')

    ##  arguments  ##
    args = parser.parse_args()
    config_path = osp.join(args.config_path)
    model_config = ModelConfig(**json.load(open(config_path)))
    model_config = update_config_from_args(model_config, args)
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    cache_dir = args.cache_dir
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
    pdbdir = args.pdbdir
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    #  check input
    input_file = args.input
    project_name = args.project_name
    assert osp.isfile(input_file), f"Test data file {input_file} was not found."
    datalist = open(input_file, 'r').read().splitlines()
    datalist = [osp.join(pdbdir, f'{pdb}.pdb') for pdb in datalist]
    result_csv = osp.join(outdir, f"{args.project_name}.csv")
    ## Model Setup ##
    model = unZipro(model_config).to(device)
    try:
        state_dict = torch.load(args.param, map_location=torch.device(device), weights_only=True)
        model.load_state_dict(state_dict)
    except Exception as e:
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "")] = v
        model.load_state_dict(new_state_dict)

    test_dataset = GraphDataset(datalist=datalist, nneighbor=args.nneighbor, noise=0, cache_dir=args.cache_dir)
    print(f'Loading datasets for testing: {len(test_dataset)}')
    num_workers = args.cpu if args.cpu > 0 else min(8, os.cpu_count())
    test_loader = get_loader(test_dataset, 1, num_workers=num_workers, shuffle=False)

    # loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # test
    start = time.perf_counter()
    df = inference(model, criterion, device, test_loader, sampling_strategy=args.sampling_strategy, num_samples=1, temperature=0.1)
    sorted_df = df.sort_values(by="Recovery", ascending=True)
    sorted_df.to_csv(result_csv, index=False, sep='\t')
    end = time.perf_counter()
    avg_recovery = sorted_df['Recovery'].mean()
    print(f"Test for {project_name} | {(end - start):.4f}s")
    print(f'Overall recovery {avg_recovery}')
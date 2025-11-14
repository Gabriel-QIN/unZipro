#! /usr/bin/env python

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import json
import os.path as osp
import time
import numpy as np
import torch
import torch.nn as nn
from dataclasses import asdict
from utils import *
from parser import parse_args
from model import unZipro, weights_init

def main(args):
    PROJECT_NAME = args.project_name
    model_config = ModelConfig()
    model_config = update_config_from_args(model_config, args)
    config_dir = args.config_dir
    config_path = osp.join(config_dir, f'{PROJECT_NAME}.json')
    os.makedirs(args.config_dir, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(asdict(model_config), f, indent=2)
    MODEL_STORE_DIR = args.model
    TENSORBOARD_LOG_DIR = osp.join(args.logdir, f"{args.project_name}/")
    NUM_TRAINING_EPOCHS = args.epochs
    pdbdir = args.pdbdir
    os.makedirs(MODEL_STORE_DIR, exist_ok=True)
    os.makedirs(f"{MODEL_STORE_DIR}/{PROJECT_NAME}", exist_ok=True)
    
    device = torch.device(f'cuda:{args.gpu}')
    print(f"[INFO] Using device: {args.gpu}")
    #  check input
    assert osp.isfile(args.train_list), f"Training data file {args.train_list} was not found."
    assert osp.isfile(args.valid_list), f"Validation data file {args.valid_list} was not found."

    # ---------------- Model ----------------
    model = unZipro(model_config).to(device)
    model.apply(weights_init)
    params = model.size()
    model = model.to(device)
    epoch_init = 1

    # ---------------- optimizer & scheduler ----------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 2), gamma=0.1)
    optimizer.zero_grad()

    if args.logging:
        from torch.utils.tensorboard import SummaryWriter
        os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
        writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)
    else:
        writer = None
    # ---------------- Datasets ----------------
    with open(args.train_list, 'r') as f:
        trainlist = f.read().splitlines()
    with open(args.valid_list, 'r') as f:
        validlist = f.read().splitlines()
    if args.pdbdir is not None:
        trainlist = [osp.join(pdbdir, f'{pdb[:4].lower()}{pdb[4:]}.pdb') for pdb in trainlist if osp.exists(osp.join(pdbdir, f'{pdb[:4].lower()}{pdb[4:]}.pdb'))]
        validlist = [osp.join(pdbdir, f'{pdb[:4].lower()}{pdb[4:]}.pdb') for pdb in validlist if osp.exists(osp.join(pdbdir, f'{pdb[:4].lower()}{pdb[4:]}.pdb'))]
    assert len(trainlist) != 0, f"Please check your training list: valid PDB file={len(trainlist)}"
    assert len(validlist) != 0, f"Please check your valid list: valid PDB file={len(validlist)}"
    trainlist = check_pdb(trainlist, nneighbor=args.nneighbor, num_workers=args.cpu)
    validlist = check_pdb(validlist, nneighbor=args.nneighbor, num_workers=args.cpu)
    train_dataset = GraphDataset(datalist=trainlist, nneighbor=args.nneighbor, noise=args.noise, cache_dir=args.cachedir)
    valid_dataset = GraphDataset(datalist=validlist, nneighbor=args.nneighbor, noise=args.noise, cache_dir=args.cachedir)
    print(f'Loading datasets for training: Train {len(train_dataset)}; Valid {len(valid_dataset)}')
    num_workers = args.cpu if args.cpu > 0 else min(8, os.cpu_count())
    train_loader = get_loader(train_dataset, args.batchsize, num_workers=num_workers, shuffle=True)
    valid_loader = get_loader(valid_dataset, args.batchsize, num_workers=num_workers, shuffle=False)

    # loss function
    criterion = nn.CrossEntropyLoss().to(device)

    global_step = 0
    minimum_loss = np.inf
    best_valid_acc = 0.0
    best_epoch = NUM_TRAINING_EPOCHS - 1

    # training routine
    print(f"# Total Parameters: {params/1e6:.2f}M")
    start_training_clock = time.perf_counter()
    for iepoch in range(epoch_init, NUM_TRAINING_EPOCHS+1):
        start = time.perf_counter()
        loss_train, acc_train, loss_valid, acc_valid = float('inf'), 0, float('inf'), 0
        # ----- Training -----
        loss_train, acc_train, global_step, total_count_train = train_one_epoch(model, criterion, device, train_loader, optimizer, global_step, writer, iepoch)
        end = time.perf_counter()
        print(f"Train | Epoch {iepoch} | {(end - start):.4f}s | Loss:{loss_train:.4f} | Train_acc {acc_train:.4f} % | {len(train_loader.dataset) / (end - start):.0f} samples/s | {total_count_train/(end-start):.0f} residues/s for {total_count_train} residues in {len(train_loader)} proteins.")
        # ----- Validation -----
        start = time.perf_counter()
        loss_valid, acc_valid, total_count_valid = valid_one_epoch(model, criterion, device, valid_loader)
        end = time.perf_counter()
        scheduler.step()
        print(f"Valid | Epoch {iepoch} | {(end - start):.4f}s | Loss: {loss_valid:.4f} | Acc: {acc_valid:.4f} %. | {total_count_valid/(end-start):.0f} residues/s  for {total_count_valid} residues in {len(valid_loader)} proteins.")
        torch.save(model.state_dict(), f'{MODEL_STORE_DIR}/{PROJECT_NAME}/unZipro_acc{int(best_valid_acc*100)}_epoch{iepoch}.pkl')
        if acc_valid > best_valid_acc:
            best_valid_acc = acc_valid
            best_epoch = iepoch
            print(f"Best valid acc in epoch {best_epoch}: {best_valid_acc:.4f}%.")
    end_training_clock = time.perf_counter()
    print(f"Training finished. Total elapsed time: {(end_training_clock- start_training_clock):.0f}s. Time cost per epoch: {(end_training_clock - start_training_clock)//NUM_TRAINING_EPOCHS:.0f}s.\nBest valid accuracy: {best_valid_acc:.4f}% in {best_epoch}th epoch.")

if __name__ == "__main__":
    args = parse_args()
    main(args)
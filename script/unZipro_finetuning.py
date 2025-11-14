#! /usr/bin/env python

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import os.path as osp
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import learn2learn as l2l
from torch.utils.data import DataLoader,random_split
from tqdm import tqdm
dir_script = osp.dirname(osp.realpath(__file__))
sys.path.append(dir_script+'/../')
from utils import *
from model import unZipro, weights_init

def fast_adapt(model, meta_model, support_dataset, query_dataset, device, optimizer, criterion, adapt_step, do_backward=True, batchsize=5, cpu=8):
    # ----------------- Inner loop (support set) -----------------
    # pbar = tqdm(range(adapt_step), desc="Inner loop", unit="Step")
    for step in range(adapt_step):
        # Adaptation: Instanciate a copy of model
        learner = meta_model.clone()
        random_index = int(np.random.random()*len(support_dataset))
        adapt_data = support_dataset[random_index]
        node, edge, adjmax, target, mask, name = adapt_data
        node = node.squeeze(0).to(device)
        edge = edge.squeeze(0).to(device)
        adjmax = adjmax.squeeze(0).to(device)
        target = target.squeeze(0).to(device)
        mask = mask.squeeze(0).to(device)
        optimizer.zero_grad()
        output = model(node, edge, adjmax)
        train_error = criterion(output[mask], target[mask])
        learner.adapt(train_error)
        torch.cuda.empty_cache()
        del node, edge, adjmax, target, mask, output, train_error
    
    # ----------------- Outer loop (query set) -----------------
    total_loss, total_count, total_correct = 0, 0, 0
    idx = 1
    query_loader = get_loader(query_dataset, batchsize, num_workers=cpu, shuffle=False)
    for batch_idx, (node, edge, adjmax, target, mask, name, num) in enumerate(query_loader):
    # for batch_idx, (node, edge, adjmax, target, mask, name, num) in enumerate(tqdm(query_loader, desc="Outer loop")):
        node = node.squeeze(0).to(device)
        edge = edge.squeeze(0).to(device)
        adjmax = adjmax.squeeze(0).to(device)
        target = target.squeeze(0).to(device)
        mask = mask.squeeze(0).to(device)
        outputs = model(node, edge, adjmax)
        pdbname = osp.basename(name[0]).split('.')[0]
        loss = criterion(outputs[mask], target[mask])
        if do_backward:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        predicted = torch.max(outputs, 1)
        count = target[mask].size()[0]
        correct = (target[mask]==predicted[1][mask]).sum().item()
        total_count += count
        total_correct += correct
        total_loss += loss.item()*count
        idx += 1
        del node, edge, adjmax, target, mask, outputs, loss, predicted
        torch.cuda.empty_cache()
    
    avg_loss = total_loss/total_count
    avg_acc = 100*total_correct/total_count
    return avg_loss, avg_acc

def train_test_split(dataset, ratio=0.5):
    dataset_size = len(dataset)
    train_size = int(ratio * dataset_size)
    test_size = dataset_size - train_size
    support_dataset, query_dataset = random_split(dataset, [train_size, test_size])
    return support_dataset, query_dataset

def unZipro_finetune(train_file, valid_file, config_path="contig/unZipro_pretrain.json", pdb_dir='data/PDB', model_store_dir='Models/finetuned/', 
                     param_file='Models/unZipro_params.pt', project_name='unZipro_finetuned', 
                     epochs=20, adapt_lr=1e-6, meta_lr=1e-6, adapt_step=10, batchsize=5, cpu=8, gpu=0,
                     cpu_only=False, noise=0.01, nneighbor=20, patience=5, cache_dir='data/tmp/', save_model_ckp=False, args=None):
    """
    UnZipro finetuning API
    :param train_file: training file path (without .pdb)
    :param valid_list: Valid file path (without .pdb)
    :param pdb_dir: directory of pdb files
    :param model_store_dir: directory to save model
    :param param_file: pretrained model params
    :param project_name: checkpoint filename prefix
    :param epochs: training epochs
    :param adapt_lr: inner loop learning rate
    :param meta_lr: outer loop learning rate
    :param adapt_step: inner loop steps
    :param batchsize: batch size
    :param cpu: num workers
    :param gpu: GPU index
    :param cpu_only: force CPU
    :param noise: training noise
    :param nneighbor: node neighbors
    :param patience: early stopping patience
    :param cache_dir: temporary cache directory
    :param save_model_ckp: whether save checkpoints
    """
    os.makedirs(model_store_dir, exist_ok=True)
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() and not cpu_only else 'cpu')

    #  check input
    assert osp.isfile(train_file), f"Training data file {train_file} was not found."
    assert osp.isfile(valid_file), f"Validation data file {valid_file} was not found."
    # ---------------- Datasets ----------------
    with open(train_file, 'r') as f:
        trainlist = f.read().splitlines()
    with open(valid_file, 'r') as f:
        validlist = f.read().splitlines()
    # Prepare datasets
    trainlist = [osp.join(pdb_dir, f'{pdb}.pdb') for pdb in trainlist if osp.exists(osp.join(pdb_dir, f'{pdb}.pdb'))]
    validlist = [osp.join(pdb_dir, f'{pdb}.pdb') for pdb in validlist if osp.exists(osp.join(pdb_dir, f'{pdb}.pdb'))]
    trainlist = check_pdb(trainlist, nneighbor=nneighbor, num_workers=cpu)
    validlist = check_pdb(validlist, nneighbor=nneighbor, num_workers=cpu)
    train_dataset = GraphDataset(datalist=trainlist, nneighbor=nneighbor, noise=noise, cache_dir=cache_dir)
    valid_dataset = GraphDataset(datalist=validlist, nneighbor=nneighbor, noise=noise, cache_dir=cache_dir)

    # Model init
    with open(config_path, "r") as f:
        data = json.load(f)
    model_config = Config(**data)
    if args is not None:
        model_config = update_config_from_args(model_config, args)
    model = unZipro(model_config).to(device)
    state_dict = torch.load(param_file, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except:
        new_state_dict = {k.replace("module.", ""):v for k,v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    meta_model = l2l.algorithms.MAML(model, lr=adapt_lr, first_order=True, allow_unused=True)
    optimizer = torch.optim.Adam(meta_model.parameters(), lr=meta_lr)
    criterion = nn.CrossEntropyLoss().to(device)

    best_valid_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    start_training_clock = time.perf_counter()
    # for iepoch in range(1, epochs+1):
    for iepoch in tqdm(range(1, epochs + 1), desc="Finetuning", unit="epoch"):
        # ----------------Meta-training------------------
        support_dataset, query_dataset = train_test_split(train_dataset)
        loss_train, acc_train = fast_adapt(model, meta_model, support_dataset, query_dataset, device, optimizer, criterion, adapt_step, batchsize=batchsize, cpu=cpu)
        # print(f"Epoch {iepoch} | Train Loss: {loss_train:.4f} | Train Acc: {acc_train:.4f}%")
        
        # ---------Meta-tesing (no backward step)---------
        support_dataset, query_dataset = train_test_split(valid_dataset)
        loss_valid, acc_valid = fast_adapt(model, meta_model, support_dataset, query_dataset, device, optimizer, criterion, adapt_step, batchsize=batchsize, cpu=cpu)
        # print(f"Epoch {iepoch} | Valid Loss: {loss_valid:.4f} | Valid Acc: {acc_valid:.4f}%")
        tqdm.write(
            f"Epoch {iepoch} | "
            f"Train Loss: {loss_train:.4f} | Train Acc: {acc_train:.2f}% | "
            f"Valid Loss: {loss_valid:.4f} | Valid Acc: {acc_valid:.2f}%"
        )
        if save_model_ckp:
            torch.save(model.state_dict(), f'{model_store_dir}/{project_name}_epoch_{iepoch}.pt')
        # ---------------- Early stopping ---------------
        # Save model if validation loss improves
        if loss_valid < best_valid_loss:
            best_valid_loss = loss_valid
            best_epoch = iepoch
            best_model = model
            # torch.save(model.state_dict(), f'{model_store_dir}/{project_name}.pt')
            patience_counter = 0
        else:
            patience_counter += 1
        # Stop training if validation loss did not improve for `patience` epochs
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
    end_training_clock = time.perf_counter()
    print(f"Finished training. Total elapsed time: {(end_training_clock- start_training_clock):.0f}s. Best valid loss: {best_valid_loss:.4f} at epoch {best_epoch}")
    torch.save(best_model.state_dict(), f'{model_store_dir}/{project_name}.pt')
    return f'{model_store_dir}/{project_name}.pt'

if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', '-t', type=str, metavar='str',
                        help='List of training data.', required=True)
    parser.add_argument('--valid', '-v', type=str, metavar='str',
                        help='List of validation data.', required=True)
    parser.add_argument('--pdbdir', type=str, default='data/PDB', 
                        help='PDB directory.')
    parser.add_argument('--config_path', type=str, default='config/unZipro_pretrain.json', 
                        help='Model config.')
    parser.add_argument('--epochs', '-e', type=int, default=20,
                        help='Number of training epochs.')
    parser.add_argument('--adapt_lr', '-lr', type=float, default=1e-6,
                        help='Adapt learning rate. ')
    parser.add_argument('--meta_lr', '-mlr', type=float, default=1e-6,
                        help='Meta learning rate. ')
    parser.add_argument('--adapt_step', '-as', type=int, default=10,
                        help='Adaptation steps. ')
    parser.add_argument('--param', type=str, default='Models/unZipro_params.pt', metavar='[File]',
                        help='Pre-trained parameter file.')
    parser.add_argument('--batchsize', '-bs', type=int, default=5, metavar='[Int]',
                        help='Batch size.')
    parser.add_argument('--cpu', '-cpu', type=int, default=8, metavar='[Int]',
                        help='CPU processors for data loading.(default:8).')
    parser.add_argument('--gpu', '-gpu', type=int, default=0, metavar='[Int]',
                        help='GPU id.(default:0).')
    parser.add_argument('--cpu_only', action='store_true',
                        help='Use CPU for fine-tuning instead of GPU.')
    parser.add_argument('--noise', '-ni', type=float, default=0.01, metavar='float',
                             help='Training noise.')
    parser.add_argument('--nneighbor', '-nb', type=int, default=20, metavar='int',
                             help='Number of node neighbors.')
    parser.add_argument('--patience', '-pt', type=int, default=5, metavar='[Int]',
                        help='Early stopping patience.')
    parser.add_argument("--project_name", default='unZipro_finetuned', type=str,
                            help="Project name for saving model checkpoints and best model.")
    parser.add_argument("--model", default='Models/finetuned/', type=str,
                        help="Directory for model storage and logits. ")
    parser.add_argument("--cache_dir", default='data/tmp/', type=str,
                        help="Temprorary directory. ")
    parser.add_argument("--save_model_ckp", action='store_true',
                        help="Save model checkpoints for each epochs. ")
    ##  arguments  ##
    args = parser.parse_args()
    _ = unZipro_finetune(args.train, args.valid, config_path=args.config_path, pdb_dir=args.pdbdir, model_store_dir=args.model, 
                     param_file=args.param, project_name=args.project_name, epochs=args.epochs, 
                     adapt_lr=args.adapt_lr, meta_lr=args.meta_lr, adapt_step=args.adapt_step, 
                     batchsize=args.batchsize, cpu=args.cpu, gpu=args.gpu,cpu_only=args.cpu_only,
                     noise=args.noise, nneighbor=args.nneighbor, patience=args.patience, save_model_ckp=args.save_model_ckp, cache_dir=args.cache_dir)

# if __name__=='__main__':
#     # argument parser
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--train', '-t', type=str, metavar='str',
#                         help='List of training data.', required=True)
#     parser.add_argument('--valid', '-v', type=str, metavar='str',
#                         help='List of validation data.', required=True)
#     parser.add_argument('--pdbdir', type=str, default='data/PDB', 
#                         help='PDB directory.')
#     parser.add_argument('--epochs', '-e', type=int, default=20,
#                         help='Number of training epochs.')
#     parser.add_argument('--adapt_lr', '-lr', type=float, default=1e-6,
#                         help='Adapt learning rate. ')
#     parser.add_argument('--meta_lr', '-mlr', type=float, default=1e-6,
#                         help='Meta learning rate. ')
#     parser.add_argument('--adapt_step', '-as', type=int, default=10,
#                         help='Adaptation steps. ')
#     parser.add_argument('--param', type=str, default='Models/unZipro_params.pt', metavar='[File]',
#                         help='Pre-trained parameter file.')
#     parser.add_argument('--batchsize', '-bs', type=int, default=5, metavar='[Int]',
#                         help='Batch size.')
#     parser.add_argument('--cpu', '-cpu', type=int, default=8, metavar='[Int]',
#                         help='CPU processors for data loading.(default:8).')
#     parser.add_argument('--gpu', '-gpu', type=int, default=0, metavar='[Int]',
#                         help='GPU id.(default:0).')
#     parser.add_argument('--cpu_only', action='store_true',
#                         help='Use CPU for fine-tuning instead of GPU.')
#     parser.add_argument('--noise', '-ni', type=float, default=0.01, metavar='float',
#                              help='Training noise.')
#     parser.add_argument('--nneighbor', '-nb', type=int, default=20, metavar='int',
#                              help='Number of node neighbors.')
#     parser.add_argument('--patience', '-pt', type=int, default=5, metavar='[Int]',
#                         help='Early stopping patience.')
#     parser.add_argument("--project_name", default='unZipro_finetuned', type=str,
#                             help="Project name for saving model checkpoints and best model.")
#     parser.add_argument("--model", default='Models/finetuned/', type=str,
#                         help="Directory for model storage and logits. ")
#     parser.add_argument("--cache_dir", default='data/tmp/', type=str,
#                         help="Temprorary directory. ")
#     ##  arguments  ##
#     args = parser.parse_args()
#     model_config = Config()
#     model_config = update_config_from_args(model_config, args)
#     PROJECT_NAME = args.project_name
#     MODEL_STORE_DIR = args.model
#     NUM_TRAINING_EPOCHS = args.epochs
#     adapt_lr = args.adapt_lr
#     meta_lr = args.meta_lr
#     adapt_step = args.adapt_step
#     os.makedirs(MODEL_STORE_DIR, exist_ok=True)

#     #  check input
#     assert osp.isfile(args.train), "Training data file {:s} was not found.".format(args.train)
#     assert osp.isfile(args.valid), "Validation data file {:s} was not found.".format(args.valid)

#     # ---------------- Datasets ----------------
#     pdbdir = args.pdbdir
#     with open(args.train, 'r') as f:
#         trainlist = f.read().splitlines()
#     with open(args.valid, 'r') as f:
#         validlist = f.read().splitlines()
#     trainlist = [osp.join(pdbdir, f'{pdb}.pdb') for pdb in trainlist if osp.exists(osp.join(pdbdir, f'{pdb}.pdb'))]
#     validlist = [osp.join(pdbdir, f'{pdb}.pdb') for pdb in validlist if osp.exists(osp.join(pdbdir, f'{pdb}.pdb'))]
#     assert len(trainlist) != 0, f"Please check your training list: valid PDB file={len(trainlist)}"
#     assert len(validlist) != 0, f"Please check your valid list: valid PDB file={len(validlist)}"
#     trainlist = check_pdb(trainlist, nneighbor=args.nneighbor, num_workers=args.cpu)
#     validlist = check_pdb(validlist, nneighbor=args.nneighbor, num_workers=args.cpu)

#     device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
#     if args.cpu_only:
#         device = torch.device('cpu') 
#     ## Model Setup ##
#     model = unZipro(model_config).to(device)
#     try:
#         state_dict = torch.load(args.param, map_location=torch.device(device), weights_only=True)
#         model.load_state_dict(state_dict)
#     except Exception as e:
#         new_state_dict = {}
#         for k, v in state_dict.items():
#             new_state_dict[k.replace("module.", "")] = v
#         model.load_state_dict(new_state_dict)

#     params = model.size()
#     meta_model = l2l.algorithms.MAML(model, lr=adapt_lr, first_order=True, allow_unused=True)

#     # optimizer & scheduler
#     optimizer = torch.optim.Adam(meta_model.parameters(), lr=meta_lr)
#     print(args)

#     train_dataset = GraphDataset(datalist=trainlist, nneighbor=args.nneighbor, noise=args.noise, cache_dir=args.cache_dir)
#     valid_dataset = GraphDataset(datalist=validlist, nneighbor=args.nneighbor, noise=args.noise, cache_dir=args.cache_dir)
#     print(f'Loading datasets for training: Train {len(train_dataset)}; Valid {len(valid_dataset)}')
    
#     # loss function
#     criterion = nn.CrossEntropyLoss().to(device)

#     global_step = 0
#     minimum_loss = np.inf
#     best_valid_acc = 0.0
#     best_epoch = NUM_TRAINING_EPOCHS - 1

#     print("# Total Parameters : {:.2f}M\n".format(params/1000000))
#     start_training_clock = time.perf_counter()
#     epoch_init = 1
#     max_patience = args.patience
#     patience = 0
#     for iepoch in range(epoch_init, NUM_TRAINING_EPOCHS+1):
#         start = time.perf_counter()
#         # meta training
#         support_dataset, query_dataset = train_test_split(train_dataset)
#         loss_train, acc_train = fast_adapt(meta_model, support_dataset, query_dataset, device, adapt_step, args=args)
#         end = time.perf_counter()
#         print(f"Train | Epoch {iepoch} | {(end - start):.4f}s | Loss:{loss_train:.4f} | Train_acc {acc_train:.4f} %.")
#         optimizer.step()
#         # meta test
#         start = time.perf_counter()
#         support_dataset, query_dataset = train_test_split(valid_dataset)
#         loss_valid, acc_valid = fast_adapt(meta_model, support_dataset, query_dataset, device, adapt_step, do_backward=False, args=args)
#         end = time.perf_counter()
#         print(f"Test | Epoch {iepoch} | {(end - start):.4f}s | Loss: {loss_valid:.4f} | Acc: {acc_valid:.4f} %.")
#         if acc_valid > best_valid_acc:
#             best_valid_acc = acc_valid
#             best_epoch = iepoch
#             best_model = model
#             torch.save(model.state_dict(), f'{MODEL_STORE_DIR}/{PROJECT_NAME}.pt')
#             print(f"Best valid acc in epoch {best_epoch}: {best_valid_acc:.4f}%.")
#         else:
#             patience += 1
#         if patience > max_patience:
#             break  
#     end_training_clock = time.perf_counter()
#     print(f"Training finished. Total elapsed time: {(end_training_clock- start_training_clock):.0f}s. Time cost per epoch: {(end_training_clock - start_training_clock)//NUM_TRAINING_EPOCHS:.0f}s.\nBest valid accuracy: {best_valid_acc:.4f}% in {best_epoch}th epoch.")

#! /usr/bin/env python

import os
import sys
import os.path as osp
sys.path.insert(0, os.path.dirname(__file__))
dir_script = osp.dirname(osp.realpath(__file__))
sys.path.append(dir_script+'/../')
import time
import json
import argparse
import torch
import torch.nn as nn
import learn2learn as l2l
from torch.utils.data import random_split
from tqdm import tqdm
from utils import *
from model import unZipro

def fast_adapt(model, meta_model, support_dataset, query_dataset, device, optimizer, criterion, adapt_step, do_backward=True, batchsize=5, cpu=8):
    """Adapt one task, then update the initialization using its mean query loss.
    """
    if len(support_dataset) == 0 or len(query_dataset) == 0:
        raise ValueError("MAML requires non-empty, disjoint support and query datasets.")
    if adapt_step < 1:
        raise ValueError("adapt_step must be at least 1.")
    if do_backward:
        optimizer.zero_grad()

    # Clone once so all inner steps build on the preceding adaptation.
    with torch.enable_grad():
        learner = meta_model.clone() if do_backward else meta_model.clone(first_order=True)
        learner.train()
        for _ in range(adapt_step):
            random_index = torch.randint(len(support_dataset), ()).item()
            node, edge, adjmax, target, mask, name = support_dataset[random_index]
            node, edge, adjmax = (tensor.to(device) for tensor in (node, edge, adjmax))
            target, mask = target.to(device), mask.to(device)
            if not mask.any().item():
                raise ValueError(f"Support structure {name} has no valid target residues.")
            output = learner(node, edge, adjmax)
            train_error = criterion(output[mask], target[mask])
            if not torch.isfinite(train_error).item():
                raise ValueError(f"Non-finite support loss for structure {name}.")
            learner.adapt(train_error)
            del node, edge, adjmax, target, mask, output, train_error

    # Every query batch uses the same adapted parameters. Backpropagate each
    learner.train(do_backward)
    total_loss, total_count, total_correct = 0, 0, 0
    query_loader = get_loader(query_dataset, batchsize, num_workers=cpu, shuffle=False)
    with torch.set_grad_enabled(do_backward):
        for batch_idx, (node, edge, adjmax, target, mask, name, num) in enumerate(query_loader):
            node, edge, adjmax = (tensor.to(device) for tensor in (node, edge, adjmax))
            target, mask = target.to(device), mask.to(device)
            count = target[mask].numel()
            if count == 0:
                continue
            outputs = learner(node, edge, adjmax)
            loss = criterion(outputs[mask], target[mask])
            if not torch.isfinite(loss).item():
                raise ValueError(f"Non-finite query loss for structures {name}.")
            if do_backward:
                # Later query batches still traverse the shared inner updates.
                (loss * count).backward(retain_graph=batch_idx + 1 < len(query_loader))
            predicted = outputs.detach().argmax(dim=1)
            total_correct += (target[mask] == predicted[mask]).sum().item()
            total_count += count
            total_loss += loss.item() * count
            del node, edge, adjmax, target, mask, outputs, loss, predicted

    if total_count == 0:
        raise ValueError("Query dataset has no valid target residues.")
    if do_backward:
        for param in meta_model.parameters():
            if param.grad is not None:
                param.grad.div_(total_count)
        optimizer.step()

    avg_loss = total_loss / total_count
    avg_acc = 100 * total_correct / total_count
    return avg_loss, avg_acc

def train_test_split(dataset, ratio=0.5):
    dataset_size = len(dataset)
    if dataset_size < 2:
        raise ValueError(
            f"MAML requires at least 2 usable structures for disjoint support/query sets; got {dataset_size}. "
            "Check the input list, --pdbdir, and PDB filtering."
        )
    if not 0 < ratio < 1:
        raise ValueError("The support split ratio must be between 0 and 1 (exclusive).")
    train_size = max(1, min(dataset_size - 1, int(ratio * dataset_size)))
    test_size = dataset_size - train_size
    support_dataset, query_dataset = random_split(dataset, [train_size, test_size])
    return support_dataset, query_dataset

def _load_pdb_list(list_file, pdb_dir, nneighbor, cpu):
    """Resolve the headerless PDB ID list and explain filtering before splitting."""
    with open(list_file, 'r') as f:
        pdb_ids = list(dict.fromkeys(line.strip() for line in f if line.strip()))
    paths = list(dict.fromkeys(
        osp.normpath(osp.join(pdb_dir, pdb if pdb.endswith('.pdb') else f'{pdb}.pdb'))
        for pdb in pdb_ids
    ))
    existing = [path for path in paths if osp.isfile(path)]
    usable = check_pdb(existing, nneighbor=nneighbor, num_workers=cpu) if existing else []
    print(f"{list_file}: {len(paths)} listed, {len(existing)} found, {len(usable)} usable structures.")
    if len(usable) < 2:
        raise ValueError(
            f"{list_file}: MAML requires at least 2 usable structures, but only {len(usable)} remain "
            f"({len(paths) - len(existing)} missing, {len(existing) - len(usable)} rejected). "
            f"Check the PDB IDs, --pdbdir ({pdb_dir}), and --nneighbor ({nneighbor})."
        )
    return usable


def unZipro_finetune(train_file, valid_file, config_path="config/unZipro_pretrain.json", pdb_dir='data/PDB', model_store_dir='Models/finetuned/',
                     param_file='Models/unZipro_params.pt', project_name='unZipro_finetuned', 
                     epochs=20, adapt_lr=1e-6, meta_lr=1e-6, adapt_step=10, batchsize=5, cpu=8, gpu=0,
                     cpu_only=False, noise=0.01, nneighbor=20, patience=5, cache_dir='data/tmp/', save_model_ckp=False, args=None,
                     first_order=False):
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
    :param first_order: use FOMAML instead of full second-order MAML
    """
    if epochs < 1 or adapt_step < 1 or batchsize < 1 or patience < 1:
        raise ValueError("epochs, adapt_step, batchsize and patience must all be at least 1.")
    if cpu < 0:
        raise ValueError("cpu must be non-negative.")
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() and not cpu_only else 'cpu')

    #  check input
    assert osp.isfile(train_file), f"Training data file {train_file} was not found."
    assert osp.isfile(valid_file), f"Validation data file {valid_file} was not found."
    if not osp.isdir(pdb_dir):
        raise FileNotFoundError(f"PDB directory {pdb_dir} was not found. Check --pdbdir and download the listed structures.")
    # ---------------- Datasets ----------------
    trainlist = _load_pdb_list(train_file, pdb_dir, nneighbor, cpu)
    validlist = _load_pdb_list(valid_file, pdb_dir, nneighbor, cpu)
    train_dataset = GraphDataset(datalist=trainlist, nneighbor=nneighbor, noise=noise, cache_dir=cache_dir)
    valid_dataset = GraphDataset(datalist=validlist, nneighbor=nneighbor, noise=0.0, cache_dir=cache_dir)
    valid_support, valid_query = train_test_split(valid_dataset)
    os.makedirs(model_store_dir, exist_ok=True)

    # Model init
    with open(config_path, "r") as f:
        data = json.load(f)
    model_config = Config(**data)
    if args is not None:
        model_config = update_config_from_args(model_config, args)
    model_config.nneighbor = nneighbor
    model = unZipro(model_config).to(device)
    state_dict = torch.load(param_file, map_location=device, weights_only=True)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        new_state_dict = {k.replace("module.", ""):v for k,v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    meta_model = l2l.algorithms.MAML(model, lr=adapt_lr, first_order=first_order, allow_unused=True)
    optimizer = torch.optim.Adam(meta_model.parameters(), lr=meta_lr)
    criterion = nn.CrossEntropyLoss().to(device)

    best_valid_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    start_training_clock = time.perf_counter()
    for iepoch in tqdm(range(1, epochs + 1), desc="Finetuning", unit="epoch"):
        # ----------------Meta-training------------------
        support_dataset, query_dataset = train_test_split(train_dataset)
        loss_train, acc_train = fast_adapt(model, meta_model, support_dataset, query_dataset, device, optimizer, criterion, adapt_step, batchsize=batchsize, cpu=cpu)

        # Meta-validation adapts a fresh clone and never updates the initialization.
        loss_valid, acc_valid = fast_adapt(model, meta_model, valid_support, valid_query, device, optimizer, criterion, adapt_step, do_backward=False, batchsize=batchsize, cpu=cpu)
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
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        # Stop training if validation loss did not improve for `patience` epochs
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
    end_training_clock = time.perf_counter()
    print(f"Finished training. Total elapsed time: {(end_training_clock- start_training_clock):.0f}s. Best valid loss: {best_valid_loss:.4f} at epoch {best_epoch}")
    torch.save(best_state, f'{model_store_dir}/{project_name}.pt')
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
    parser.add_argument('--epochs', '--epoch', '-e', type=int, default=20,
                        help='Number of training epochs.')
    parser.add_argument('--adapt_lr', '-lr', type=float, default=1e-6,
                        help='Adapt learning rate. ')
    parser.add_argument('--meta_lr', '-mlr', type=float, default=1e-6,
                        help='Meta learning rate. ')
    parser.add_argument('--adapt_step', '-as', type=int, default=10,
                        help='Adaptation steps. ')
    parser.add_argument('--first_order', action='store_true',
                        help='Use the FOMAML approximation to reduce memory; default is full second-order MAML.')
    parser.add_argument('--param', type=str, default='Models/unZipro_params.pt', metavar='[File]',
                        help='Pre-trained parameter file.')
    parser.add_argument('--batchsize', '-bs', type=int, default=5, metavar='[Int]',
                        help='Batch size.')
    parser.add_argument('--cpu', '-cpu', type=int, default=8, metavar='[Int]',
                        help='Data-loading workers; 0 runs in the current process.')
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
                     noise=args.noise, nneighbor=args.nneighbor, patience=args.patience, save_model_ckp=args.save_model_ckp, cache_dir=args.cache_dir,
                     first_order=args.first_order)

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

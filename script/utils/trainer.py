import os
import os.path as osp
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.amp import GradScaler, autocast
from .featurizer import protein_to_features
from .commons import *

def train_one_epoch(model, criterion, device, train_loader, optimizer, global_step, writer, iepoch):
    """Train the model for one epoch with mixed precision and gradient clipping."""
    model.train()
    scaler = GradScaler("cuda")

    total_loss, total_count, total_correct, total_sample_count = 0, 0, 0, 0

    for step, batch in enumerate(train_loader):
        node, edge, adjmax, target, mask, name, num = batch
        # GPU async copy
        node = node.squeeze(0).to(device, non_blocking=True)
        edge = edge.squeeze(0).to(device, non_blocking=True)
        adjmax = adjmax.squeeze(0).to(device, non_blocking=True)
        target = target.squeeze(0).to(device, non_blocking=True)
        mask = mask.squeeze(0).to(device, non_blocking=True)

        total_sample_count += num
        optimizer.zero_grad()

        with autocast("cuda"):
            outputs = model(node, edge, adjmax)
            loss = criterion(outputs[mask], target[mask])

        # backward with AMP
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2.0)
        scaler.step(optimizer)
        scaler.update()

        predicted = torch.max(outputs, 1)[1]
        count = target[mask].size(0)
        correct = (target[mask] == predicted[mask]).sum().item()

        total_count += count
        total_correct += correct
        total_loss += loss.item() * count
        
        if writer is not None:
            writer.add_scalar("Training_loss_step", total_loss / total_count, global_step)
            writer.add_scalar("Step_lr", optimizer.param_groups[0]['lr'], global_step)
        else:
            print(f"Epoch {iepoch} | Step {step} | Loss = {total_loss/total_count:.4f} "
                f"| Batch acc {correct/count:.2f} | Total acc {total_correct/total_count:.2f}")

        global_step += 1

    avg_loss = total_loss / total_count
    avg_acc = 100 * total_correct / total_count
    return avg_loss, avg_acc, global_step, total_count

def valid_one_epoch(model, criterion, device, valid_loader, iepoch=0, writer=None, global_step=0):
    """Evaluate the model for one epoch on the validation set."""
    model.eval()
    total_loss, total_count, total_correct = 0, 0, 0

    with torch.no_grad():
        for step, batch in enumerate(valid_loader):
            node, edge, adjmax, target, mask, name, num = batch
            # GPU async copy
            node = node.squeeze(0).to(device, non_blocking=True)
            edge = edge.squeeze(0).to(device, non_blocking=True)
            adjmax = adjmax.squeeze(0).to(device, non_blocking=True)
            target = target.squeeze(0).to(device, non_blocking=True)
            mask = mask.squeeze(0).to(device, non_blocking=True)

            with autocast("cuda"):
                outputs = model(node, edge, adjmax)
                loss = criterion(outputs[mask], target[mask])

            predicted = torch.max(outputs, 1)[1]
            count = target[mask].size(0)
            correct = (target[mask] == predicted[mask]).sum().item()

            total_count += count
            total_correct += correct
            total_loss += loss.item() * count
            if writer is not None:
                writer.add_scalar("Valid_loss_step", total_loss / total_count, global_step)

            global_step += 1

    avg_loss = total_loss / total_count
    avg_acc = 100 * total_correct / total_count
    return avg_loss, avg_acc, total_count

def get_pdb_info(pdbname, pdbdir='pdb'):
    """Extract residue-level information from a PDB file."""
    resnum_list = []
    resname_list = []
    res_list = []
    with open(osp.join(pdbdir, pdbname+'.pdb'), 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                resname = line[17:20].lstrip(' ').rstrip(' ')
                resnum = line[22:26].lstrip(' ').rstrip(' ')
                chainID = line[21]
                if resnum not in resnum_list:
                    resnum_list.append(resnum)
                    resname_list.append(resname)
                    res_list.append([pdbname, chainID, resnum, resname, three2one[resname]])
    return res_list

def test(model, criterion, device, test_loader, temperature, cutoff, outdir, pdbdir):
    """Perform inference on the test dataset."""
    model.eval()
    total_loss, total_count, total_correct = 0, 0, 0
    designs = []
    df_list = []
    with torch.no_grad():
        for batch_idx, (node, edge, adjmax, target, mask, name) in enumerate(test_loader):
            start = time.perf_counter()
            node = node.squeeze(0).to(device)
            edge = edge.squeeze(0).to(device)
            adjmax = adjmax.squeeze(0).to(device)
            target = target.squeeze(0).to(device)
            mask = mask.squeeze(0).to(device)
            outputs = model(node, edge, adjmax)
            loss = criterion(outputs[mask], target[mask]).item()
            predicted = torch.max(outputs, 1)
            count = target[mask].size()[0]
            correct = (target[mask]==predicted[1][mask]).sum().item()
            total_count += count
            total_correct += correct
            total_loss += loss*count
            end = time.perf_counter()
            
            # Target    index   logits  prob(<  cutoff)
            # Predict   index   logits  prob(>= cutoff)
            target_indices = target[mask]
            logits = torch.max(outputs[mask], 1)[0]
            probs_indices = torch.nn.functional.softmax(outputs[mask], dim=1)
            probs, predict_indices = torch.max(probs_indices, 1)
            mutations = [1 if p.item() >= cutoff and predict_indices[i] != target_indices[i] else 0 for i, p in enumerate(probs)]
            # print(f"Original sequence   vs  Designed sequence ({(torch.tensor(mutations) == 1).sum().item()} mutations; {len(target_indices)} len; {100*correct/count:.2f} recovery | ID: {osp.basename(name[0]).split('.')[0]} :")
            
            res_list = get_pdb_info(name[0].split('/')[-1].split('.')[0], pdbdir)
            savepath = osp.join(outdir, name[0].split('/')[-1].split('.')[0]+'.info.csv')
            ori_list = []
            design_list = []
            design_logits_list = []
            design_prob_list = []
            design_all_chain_probs_list = []
            design_all_chain_logits_list = []
            aa_ind = 0
            alllist = []
            prev_res_ind = 0
            break_start = 0
            for ind, o in enumerate(outputs):
                if ind == 0 or ind == len(mask)-1:
                    continue
                ori_list.append(target[ind].item())
                if mask[ind] == True:
                    # print(torch.max(o), torch.argmax(o)[-1].item(), target[ind])
                    aa_type = torch.argmax(o).detach().cpu()
                    aa_logits = torch.max(o).detach().cpu()
                    aa_prob = torch.nn.functional.softmax(o/temperature, dim=0).detach().cpu()
                    mutant = f"{res_map[target[ind].item()]}{ind}{res_map[aa_type.item()]}"
                    mutant_auth = f"{res_map[target[ind].item()]}{res_list[aa_ind][2]}{res_map[aa_type.item()]}"
                    pdbinfo = res_list[aa_ind][:3]
                    prev_res_ind = pdbinfo[-1]
                    mutate_prob = 0 if target[ind].item() == aa_type.item() else aa_prob.max().item()
                    wt_prob = aa_prob[target[ind].item()]
                    wt_logit = o[target[ind].item()]
                    logit_log_ratio = torch.log2(aa_logits) / torch.log2(wt_logit)
                    residue_info = [*pdbinfo, ind, one2three[res_map[target[ind].item()]], res_map[target[ind].item()], one2three[res_map[aa_type.item()]], res_map[aa_type.item()],  mutant, mutant_auth, mutate_prob, aa_prob.max().item(), wt_prob.item(), aa_logits.item(), wt_logit.item(), abs(logit_log_ratio.item()), *[ppp for ppp in aa_prob.tolist()], *[iii for iii in o.tolist()]]
                    alllist.append(residue_info)
                    design_list.append(aa_type.item())
                    design_prob_list.append(aa_prob.max())
                    design_all_chain_probs_list.append([p.item() for p in aa_prob])
                    design_all_chain_logits_list.append([p.item() for p in o])
                    aa_ind += 1
                else:
                    mask_aa_type = target[ind].item()
                    if break_start != aa_ind:
                        break_start = aa_ind
                        break_ind = aa_ind
                    else:
                        break_ind = aa_ind+1
                    if res_list[break_ind][3] == one2three[res_map[mask_aa_type]]:
                        aa_ind += 1
                    mutant_auth = f"{res_map[target[ind].item()]}{res_list[break_ind][2]}{res_map[mask_aa_type]}"
                    pdbinfo = [*res_list[0][:2], res_list[break_ind][2]]
                    # print([0. for i in range(20)])
                    ppp = [*[0. for i in range(20)]]
                    mutant = f"{res_map[target[ind].item()]}{ind}{res_map[mask_aa_type]}"
                    mutate_prob = 0
                    wt_prob = 0
                    gnn_logit = 0
                    wt_logit = 0
                    logit_ratio = 0
                    residue_info = [*pdbinfo, ind, one2three[res_map[mask_aa_type]], res_map[mask_aa_type], one2three[res_map[mask_aa_type]], res_map[mask_aa_type], mutant, mutant_auth, mutate_prob, wt_prob, 0., gnn_logit, wt_logit, logit_ratio, *ppp, *ppp]
                    alllist.append(residue_info)
                    design_prob_list.append(0.)
                    design_list.append(mask_aa_type)
                    design_all_chain_probs_list.append([0. for i in range(20)])
                    design_all_chain_logits_list.append([0. for i in range(20)])
            d = np.array(alllist, dtype=object)
            data = pd.DataFrame(d)
            header = ['pdb', 'chain', 'auth_res_index', 'index', 'target_3', 'target_1', 'predict_3', 'predict_1', 'mutation', 'mutation_auth', 'mut_prob', 'model_prob', 'wt_prob', 'gnn_logit', 'wt_logit', 'logit_ratio', *[f'prob_{a}' for a in mapped_20.keys()], *[f'logit_{a}' for a in mapped_20.keys()]]
            data.columns = header
            df_list.append(data)
            data.to_csv(savepath, sep='\t', index=False, header=header)
            original_seq = ''.join([res_map[x] for x in ori_list])
            bool_seq = ''.join(["1" if p >= cutoff and design_list[i] != ori_list[i] else "0" for i, p in enumerate(design_prob_list)])
            designed_seq = ''.join([res_map[design_list[i]] if p >= cutoff else res_map[ori_list[i]] for i, p in enumerate(design_prob_list)])
            tmp = [osp.basename(name[0]).split(".")[0], len(original_seq) , 100*correct/count, mutations.count(1), original_seq, bool_seq, designed_seq]
            designs.append(tmp)
    return total_loss/total_count, 100*total_correct/total_count, designs, df_list

#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os

import scipy.io
import urllib.request
import dgl
import math
import numpy as np
from model import *
import argparse
from load_dataset import score, load_datasets
import random

parser = argparse.ArgumentParser(description='Training GNN on ogbn-products benchmark')

parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--n_hid', type=int, default=400)
parser.add_argument('--clip', type=int, default=1.0)
parser.add_argument('--max_lr', type=float, default=1e-3)
parser.add_argument('--seed', type=int, default=4)
parser.add_argument('-l', '--label', type=str, default="../data/ACM/labels_5_fold_cross_validation_4.pkl",
                    help='label path')
parser.add_argument('--result', type=str, default="./")
parser.add_argument('--dataset', type=str, default="ACM")
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device("cuda:{}".format(args.device))

all_edges, node_type_features, labels, train_idx, val_idx, test_idx = load_datasets("../data/ACM",
                                                                                    labels_path=args.label)

Micro = []
Macro = []


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def train(model, G):
    best_val_acc = torch.tensor(0)
    best_val_loss = torch.tensor(10000)
    best_test_acc = torch.tensor(0)
    train_step = torch.tensor(0)
    for epoch in np.arange(args.n_epoch) + 1:
        model.train()
        logits = model(G, 'type0')
        # The loss is computed only for labeled nodes.
        loss = F.cross_entropy(logits[train_idx], labels[train_idx].to(device))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_step += 1
        scheduler.step(train_step)
        model.eval()
        logits = model(G, 'type0')
        acc_train, micro_train, macro_train = score(logits[train_idx], labels[train_idx])
        acc_val, micro_val, macro_val = score(logits[val_idx], labels[val_idx])
        loss_val = F.cross_entropy(logits[val_idx], labels[val_idx].to(device))
        acc_test, micro_test, macro_test = score(logits[test_idx], labels[test_idx])
        pred = logits.argmax(1).cpu()
        train_acc = (pred[train_idx] == labels[train_idx]).float().mean()
        val_acc = (pred[val_idx] == labels[val_idx]).float().mean()
        test_acc = (pred[test_idx] == labels[test_idx]).float().mean()
        if best_val_loss > loss_val.cpu():
            best_val_loss = loss_val.cpu().detach().item()
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_train_micro = micro_train
            best_test_micro = micro_test
            best_val_micro = micro_val
            best_train_macro = macro_train
            best_test_macro = macro_test
            best_val_macro = macro_val
        print(
            'Epoch: %d LR: %.5f Train Loss %.4f, Val Loss %.4f, Train Macro %.4f, Val Macro %.4f (Best %.4f), Best Test Macro %.4f Micro %.4f' % (
                epoch,
                optimizer.param_groups[0]['lr'],
                loss.item(),
                loss_val.item(),
                macro_train,
                macro_val,
                best_val_macro,
                best_test_macro,
                best_test_micro,
            ))
    Micro.append(best_test_micro)
    Macro.append(best_test_macro)


G = dgl.heterograph(all_edges)
print(G)

node_dict = {}
edge_dict = {}
for ntype in G.ntypes:
    node_dict[ntype] = len(node_dict)
for etype in G.etypes:
    edge_dict[etype] = len(edge_dict)
#     G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]

#     Random initialize input feature
for ntype in G.ntypes:
    G.nodes[ntype].data['inp'] = node_type_features[ntype]

G = G.to(device)

model = HGT(G,
            node_dict, edge_dict,
            n_inp=node_type_features[G.ntypes[0]].shape[1],
            n_hid=args.n_hid,
            n_out=labels.max().item() + 1,
            n_layers=4,
            n_heads=8,
            use_norm=True).to(device)
optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-6)
print('Training HGT with #param: %d' % (get_n_params(model)))
train(model, G)

all_edges, node_type_features, labels, train_idx, val_idx, test_idx = load_datasets("../data/ACM", labels_path=args.label)
G = dgl.heterograph(all_edges)
print(G)

node_dict = {}
edge_dict = {}
for ntype in G.ntypes:
    node_dict[ntype] = len(node_dict)
for etype in G.etypes:
    edge_dict[etype] = len(edge_dict)
#     G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]

#     Random initialize input feature
for ntype in G.ntypes:
    G.nodes[ntype].data['inp'] = node_type_features[ntype]

G = G.to(device)

model = HeteroRGCN(G,
                   in_size=node_type_features[G.ntypes[0]].shape[1],
                   hidden_size=args.n_hid,
                   out_size=labels.max().item() + 1).to(device)
optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-6)
print('Training RGCN with #param: %d' % (get_n_params(model)))
train(model, G)

model = HGT(G,
            node_dict, edge_dict,
            n_inp=node_type_features[G.ntypes[0]].shape[1],
            n_hid=args.n_hid,
            n_out=labels.max().item() + 1,
            n_layers=0,
            n_heads=4).to(device)
optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=300, max_lr=1e-3, pct_start=0.05)
print('Training MLP with #param: %d' % (get_n_params(model)))
train(model, G)
print("HGT results Micro {} Macro {}".format(Micro[0], Macro[0]))
print("RGCN results Micro {} Macro {}".format(Micro[1], Macro[1]))
print("MLP results Micro {} Macro {}".format(Micro[2], Macro[2]))
results = np.array([Micro, Macro]).transpose()
labels = os.path.split(args.label)[1][-5]
name = "hgt.csv"
result_path = os.path.join(args.result, name)
import csv

with open(result_path, 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([args.dataset, labels, args.seed, Micro[0], Macro[0]])

name = "rgcn.csv"
result_path = os.path.join(args.result, name)

with open(result_path, 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([args.dataset, labels, args.seed, Micro[1], Macro[1]])

name = "mlp.csv"
result_path = os.path.join(args.result, name)

with open(result_path, 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([args.dataset, labels, args.seed, Micro[2], Macro[2]])

import os
import pathlib
import pickle
import pickle as pkl

import networkx as nx

import utils.preprocess
from scipy import sparse
import numpy as np
import scipy as sp
import torch

DBLP_meta_paths = [[(0, 1, 0), (0, 1, 2, 1, 0)]]


def process_data_mata_path_dblp(adj: sparse.csr_matrix, type_mask: np.ndarray, save_prefix: str, label_length: int):
    for i in range(1):
        pathlib.Path(os.path.join(save_prefix, '{}'.format(i))).mkdir(parents=True, exist_ok=True)

    adjM = adj.toarray()
    neighbor_pairs = utils.preprocess.get_metapath_neighbor_pairs(adjM, type_mask, DBLP_meta_paths[0])
    G_list = utils.preprocess.get_networkx_graph(neighbor_pairs, type_mask, 0)

    for G, metapath in zip(G_list, DBLP_meta_paths[0]):
        nx.write_adjlist(G, os.path.join(save_prefix, '{}'.format(0), '-'.join(map(str, metapath)) + '.adjlist'))
    all_edge_metapath_idx_array = utils.preprocess.get_edge_metapath_idx_array(neighbor_pairs)

    edge_metapath_idx_array_list = {}
    for metapath, edge_metapath_idx_array in zip(DBLP_meta_paths[0], all_edge_metapath_idx_array):
        edge_metapath_idx_array_list[metapath] = edge_metapath_idx_array
        np.save(os.path.join(save_prefix, '{}'.format(0), '-'.join(map(str, metapath)) + '_idx.npy'),
                edge_metapath_idx_array)

    target_idx_list = np.arange(label_length)
    for metapath in DBLP_meta_paths[0]:
        edge_metapath_idx_array = np.load(
            os.path.join(save_prefix, '{}'.format(0), '-'.join(map(str, metapath)) + '_idx.npy'))
        target_metapaths_mapping = {}
        for target_idx in target_idx_list:
            target_metapaths_mapping[target_idx] = edge_metapath_idx_array[
                                                       edge_metapath_idx_array[:, 0] == target_idx][
                                                   :, ::-1]
        out_file = open(os.path.join(save_prefix, '{}'.format(0), '-'.join(map(str, metapath)) + '_idx.pickle'), 'wb')
        pickle.dump(target_metapaths_mapping, out_file)
        out_file.close()


def process_dblp(path, labels_path):
    print('processing {} dataset...'.format("DBLP"))

    prefix = path
    with open(os.path.join(prefix, "edges.pkl"), "rb") as f:
        edges = pkl.load(f)
        f.close()

    adjs_pt = None
    i = 0
    for mx in edges:
        if i == 0:
            adjs_pt = mx.astype(np.float32)
            i = 1
        else:
            adjs_pt += mx.astype(np.float32)

    with open(os.path.join(prefix, "node_features.pkl"), "rb") as f:
        node_feats = pickle.load(f)
        f.close()
    with open(labels_path, "rb") as f:
        labels = pickle.load(f)
        f.close()

    node_types = np.load(os.path.join(prefix, "node_types.npy"))
    node_types_dblp = np.zeros_like(node_types)
    node_types_dblp[node_types == 0] = 0
    node_types_dblp[node_types == 1] = 1
    node_types_dblp[node_types == 2] = 2

    prefix_magnn = os.path.join(prefix, "preprocessed_magnn")

    train_idx = np.array(labels[0])[:, 0]
    valid_idx = np.array(labels[1])[:, 0]
    test_idx = np.array(labels[2])[:, 0]
    label_old = labels
    labels = np.array([-1 for _ in range(max(max(train_idx), max(valid_idx), max(test_idx)) + 1)])
    labels[train_idx] = np.array(label_old[0])[:, 1]
    labels[valid_idx] = np.array(label_old[1])[:, 1]
    labels[test_idx] = np.array(label_old[2])[:, 1]

    if not os.path.isdir(os.path.join(prefix, "preprocessed_magnn")) or len(
            os.listdir(os.path.join(prefix, "preprocessed_magnn"))) == 0:
        process_data_mata_path_dblp(adjs_pt, node_types_dblp, prefix_magnn, len(labels))
    
    train_val_test_idx = {"val_idx": valid_idx, "train_idx": train_idx, "test_idx": test_idx}

    print("load dataset")

    in_file = open(os.path.join(prefix_magnn, '0/0-1-0.adjlist'), 'r')
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00[3:]
    in_file.close()
    in_file = open(os.path.join(prefix_magnn, '0/0-1-2-1-0.adjlist'), 'r')
    adjlist01 = [line.strip() for line in in_file]
    adjlist01 = adjlist01[3:]
    in_file.close()

    in_file = open(os.path.join(prefix_magnn, '0/0-1-0_idx.pickle'), 'rb')
    idx00 = pickle.load(in_file)
    in_file.close()
    in_file = open(os.path.join(prefix_magnn, '0/0-1-2-1-0_idx.pickle'), 'rb')
    idx01 = pickle.load(in_file)
    in_file.close()

    adjM = adjs_pt

    for i in range(max(node_types) + 1):
        if i == 0:
            features_0 = node_feats[node_types_dblp == i]
        elif i == 1:
            features_1 = node_feats[node_types_dblp == i]
        elif i == 2:
            features_2 = node_feats[node_types_dblp == i]

    return [adjlist00, adjlist01], \
           [idx00, idx01], \
           [features_0, features_1, features_2], \
           adjM, \
           node_types_dblp, \
           labels, \
           train_val_test_idx


IMDB_meta_paths = [
    [(0, 1, 0), (0, 2, 0)],
    [(1, 0, 1), (1, 0, 2, 0, 1)],
    [(2, 0, 2), (2, 0, 1, 0, 2)]
]


def process_data_mata_path_imdb(adj: sparse.csr_matrix, type_mask: np.ndarray, save_prefix: str,
                                num_type: int):
    for i in range(num_type):
        pathlib.Path(os.path.join(save_prefix, '{}'.format(i))).mkdir(parents=True, exist_ok=True)

    adjM = adj.toarray()
    for i in range(num_type):
        neighbor_pairs = utils.preprocess.get_metapath_neighbor_pairs(adjM, type_mask, IMDB_meta_paths[i])
        G_list = utils.preprocess.get_networkx_graph(neighbor_pairs, type_mask, i)

        for G, metapath in zip(G_list, IMDB_meta_paths[i]):
            nx.write_adjlist(G, os.path.join(save_prefix, '{}'.format(i), '-'.join(map(str, metapath)) + '.adjlist'))
        all_edge_metapath_idx_array = utils.preprocess.get_edge_metapath_idx_array(neighbor_pairs)

        edge_metapath_idx_array_list = {}
        for metapath, edge_metapath_idx_array in zip(IMDB_meta_paths[i], all_edge_metapath_idx_array):
            edge_metapath_idx_array_list[metapath] = edge_metapath_idx_array
            np.save(os.path.join(save_prefix, '{}'.format(i), '-'.join(map(str, metapath)) + '_idx.npy'),
                    edge_metapath_idx_array)


def process_imdb(path, labels_path):
    print('processing {} dataset...'.format("IMDB"))

    prefix = path
    with open(os.path.join(prefix, "edges.pkl"), "rb") as f:
        edges = pkl.load(f)
        f.close()

    adjs_pt = None
    i = 0
    for mx in edges:
        if i == 0:
            adjs_pt = mx.astype(np.float32)
            i = 1
        else:
            adjs_pt += mx.astype(np.float32)

    with open(os.path.join(prefix, "node_features.pkl"), "rb") as f:
        node_feats = pickle.load(f)
        f.close()
    with open(labels_path, "rb") as f:
        labels = pickle.load(f)
        f.close()

    node_types_imdb = np.load(os.path.join(prefix, "node_types.npy"))

    prefix_magnn = os.path.join(prefix, "preprocessed_magnn")

    train_idx = np.array(labels[0])[:, 0]
    valid_idx = np.array(labels[1])[:, 0]
    test_idx = np.array(labels[2])[:, 0]
    label_old = labels
    labels = np.array([-1 for _ in range(max(max(train_idx), max(valid_idx), max(test_idx)) + 1)])
    labels[train_idx] = np.array(label_old[0])[:, 1]
    labels[valid_idx] = np.array(label_old[1])[:, 1]
    labels[test_idx] = np.array(label_old[2])[:, 1]

    if not os.path.isdir(os.path.join(prefix, "preprocessed_magnn")) or len(
            os.listdir(os.path.join(prefix, "preprocessed_magnn"))) == 0:
        process_data_mata_path_imdb(adjs_pt, node_types_imdb, prefix_magnn, max(node_types_imdb) + 1)

    train_val_test_idx = {"val_idx": valid_idx, "train_idx": train_idx, "test_idx": test_idx}

    print("load dataset")

    G00 = nx.read_adjlist(prefix_magnn + '/0/0-1-0.adjlist', create_using=nx.MultiDiGraph)
    G01 = nx.read_adjlist(prefix_magnn + '/0/0-2-0.adjlist', create_using=nx.MultiDiGraph)
    G10 = nx.read_adjlist(prefix_magnn + '/1/1-0-1.adjlist', create_using=nx.MultiDiGraph)
    G11 = nx.read_adjlist(prefix_magnn + '/1/1-0-2-0-1.adjlist', create_using=nx.MultiDiGraph)
    G20 = nx.read_adjlist(prefix_magnn + '/2/2-0-2.adjlist', create_using=nx.MultiDiGraph)
    G21 = nx.read_adjlist(prefix_magnn + '/2/2-0-1-0-2.adjlist', create_using=nx.MultiDiGraph)
    idx00 = np.load(prefix_magnn + '/0/0-1-0_idx.npy')
    idx01 = np.load(prefix_magnn + '/0/0-2-0_idx.npy')
    idx10 = np.load(prefix_magnn + '/1/1-0-1_idx.npy')
    idx11 = np.load(prefix_magnn + '/1/1-0-2-0-1_idx.npy')
    idx20 = np.load(prefix_magnn + '/2/2-0-2_idx.npy')
    idx21 = np.load(prefix_magnn + '/2/2-0-1-0-2_idx.npy')

    adjM = adjs_pt

    for i in range(max(node_types_imdb) + 1):
        if i == 0:
            features_0 = node_feats[node_types_imdb == i]
        elif i == 1:
            features_1 = node_feats[node_types_imdb == i]
        elif i == 2:
            features_2 = node_feats[node_types_imdb == i]

    return [[G00, G01], [G10, G11], [G20, G21]], \
           [[idx00, idx01], [idx10, idx11], [idx20, idx21]], \
           [features_0, features_1, features_2], \
           adjM, \
           node_types_imdb, \
           labels, \
           train_val_test_idx


ACM_meta_paths = [
    [(0, 1, 0), (0, 2, 0)],
]


def process_data_mata_path_acm(adj: sparse.csr_matrix, type_mask: np.ndarray, save_prefix: str,
                               num_type: int):
    for i in range(num_type):
        pathlib.Path(os.path.join(save_prefix, '{}'.format(i))).mkdir(parents=True, exist_ok=True)

    adjM = adj.toarray()
    for i in range(num_type):
        neighbor_pairs = utils.preprocess.get_metapath_neighbor_pairs(adjM, type_mask, IMDB_meta_paths[i])
        G_list = utils.preprocess.get_networkx_graph(neighbor_pairs, type_mask, i)

        for G, metapath in zip(G_list, ACM_meta_paths[i]):
            nx.write_adjlist(G, os.path.join(save_prefix, '{}'.format(i), '-'.join(map(str, metapath)) + '.adjlist'))
        all_edge_metapath_idx_array = utils.preprocess.get_edge_metapath_idx_array(neighbor_pairs)

        edge_metapath_idx_array_list = {}
        for metapath, edge_metapath_idx_array in zip(ACM_meta_paths[i], all_edge_metapath_idx_array):
            edge_metapath_idx_array_list[metapath] = edge_metapath_idx_array
            np.save(os.path.join(save_prefix, '{}'.format(i), '-'.join(map(str, metapath)) + '_idx.npy'),
                    edge_metapath_idx_array)


def process_acm(path, labels_path, dblp=False):
    print('processing {} dataset...'.format("ACM" if not dblp else "IMDB"))

    prefix = path
    with open(os.path.join(prefix, "edges.pkl"), "rb") as f:
        edges = pkl.load(f)
        f.close()

    adjs_pt = None
    i = 0
    for mx in edges:
        if i == 0:
            adjs_pt = mx.astype(np.float32)
            i = 1
        else:
            adjs_pt += mx.astype(np.float32)

    with open(os.path.join(prefix, "node_features.pkl"), "rb") as f:
        node_feats = pickle.load(f)
        f.close()
    with open(labels_path, "rb") as f:
        labels = pickle.load(f)
        f.close()

    node_types_acm = np.load(os.path.join(prefix, "node_types.npy"))

    prefix_magnn = os.path.join(prefix, "preprocessed_magnn")

    train_idx = np.array(labels[0])[:, 0]
    valid_idx = np.array(labels[1])[:, 0]
    test_idx = np.array(labels[2])[:, 0]
    label_old = labels
    labels = np.array([-1 for _ in range(max(max(train_idx), max(valid_idx), max(test_idx)) + 1)])
    labels[train_idx] = np.array(label_old[0])[:, 1]
    labels[valid_idx] = np.array(label_old[1])[:, 1]
    labels[test_idx] = np.array(label_old[2])[:, 1]

    if not os.path.isdir(os.path.join(prefix, "preprocessed_magnn")) or len(
            os.listdir(os.path.join(prefix, "preprocessed_magnn"))) == 0:
        process_data_mata_path_acm(adjs_pt, node_types_acm, prefix_magnn, 1)

    train_val_test_idx = {"val_idx": valid_idx, "train_idx": train_idx, "test_idx": test_idx}

    print("load dataset")

    if dblp:
        in_file = open(os.path.join(prefix_magnn, '0/0-1-0.adjlist'), 'r')
        adjlist00 = [line.strip() for line in in_file]
        G00 = adjlist00[3:]
        in_file.close()
        in_file = open(os.path.join(prefix_magnn, '0/0-2-0.adjlist'), 'r')
        adjlist01 = [line.strip() for line in in_file]
        G01 = adjlist01[3:]
        in_file.close()
    else:
        G00 = nx.read_adjlist(prefix_magnn + '/0/0-1-0.adjlist', create_using=nx.MultiDiGraph)
        G01 = nx.read_adjlist(prefix_magnn + '/0/0-2-0.adjlist', create_using=nx.MultiDiGraph)
    idx00 = np.load(prefix_magnn + '/0/0-1-0_idx.npy')
    idx01 = np.load(prefix_magnn + '/0/0-2-0_idx.npy')

    adjM = adjs_pt

    for i in range(max(node_types_acm) + 1):
        if i == 0:
            features_0 = node_feats[node_types_acm == i]
        elif i == 1:
            features_1 = node_feats[node_types_acm == i]
        elif i == 2:
            features_2 = node_feats[node_types_acm == i]

    return [[G00, G01]], \
           [[idx00, idx01]], \
           [features_0, features_1, features_2], \
           adjM, \
           node_types_acm, \
           labels, \
           train_val_test_idx


def process_acm_2_dblp_like(path, labels_path):
    output = process_acm(path, labels_path, dblp=True)
    output = list(output)
    output[0] = output[0][0]
    output[1] = output[1][0]
    target_idx_list = np.arange(len(output[-2]))
    adjlist = []
    for edge_metapath_idx_array in output[1]:
        target_metapaths_mapping = {}
        for target_idx in target_idx_list:
            target_metapaths_mapping[target_idx] = edge_metapath_idx_array[
                                                       edge_metapath_idx_array[:, 0] == target_idx][
                                                   :, ::-1]
        adjlist.append(target_metapaths_mapping)

    output[1] = adjlist
    return output


if __name__ == "__main__":
    process_dblp(r"../data/DBLP")

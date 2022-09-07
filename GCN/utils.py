import numpy as np
import scipy.sparse as sp
import torch
import os
import pickle
import pickle as pkl


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def sparse_mx_to_torch_sparse_tensor_others(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize_row(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx.tocoo()


def load_data_others(path="../data/DBLP/", dataset="DBLP", labels_path=None):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    prefix = path
    with open(os.path.join(prefix, "edges.pkl"), "rb") as f:
        edges = pkl.load(f)
        f.close()

    i = 0
    for mx in edges:
        if i == 0:
            # adjs_pt = sparse_mx_to_torch_sparse_tensor_others(
            #     normalize_row(mx.astype(np.float32) + sp.eye(mx.shape[0], dtype=np.float32)))
            adjs_pt = sparse_mx_to_torch_sparse_tensor_others(
                # mx.astype(np.float32).tocoo())
                normalize_row(mx.astype(np.float32)))

            i = 1
        else:
            # adjs_pt += sparse_mx_to_torch_sparse_tensor_others(
            #     normalize_row(mx.astype(np.float32) + sp.eye(mx.shape[0], dtype=np.float32)))
            adjs_pt += sparse_mx_to_torch_sparse_tensor_others(
                # mx.astype(np.float32).tocoo())
                normalize_row(mx.astype(np.float32)))
    # adjs_pt += sparse_mx_to_torch_sparse_tensor_others(sp.eye(edges[0].shape[0], dtype=np.float32).tocoo())

    with open(os.path.join(prefix, "node_features.pkl"), "rb") as f:
        node_feats = pickle.load(f)
        f.close()
    node_feats = torch.from_numpy(node_feats.astype(np.float32))
    if not labels_path:
        with open(os.path.join(prefix, "labels.pkl"), "rb") as f:
            labels = pickle.load(f)
            f.close()
    else:
        with open(os.path.join(labels_path), "rb") as f:
            labels = pickle.load(f)
            f.close()

    train_idx = torch.from_numpy(np.array(labels[0])[:, 0]).type(torch.long)
    valid_idx = torch.from_numpy(np.array(labels[1])[:, 0]).type(torch.long)
    test_idx = torch.from_numpy(np.array(labels[2])[:, 0]).type(torch.long)
    label_old = labels
    labels = torch.from_numpy(
        np.array([-1 for _ in range(max(max(train_idx), max(valid_idx), max(test_idx)) + 1)])).type(torch.long)
    labels[train_idx] = torch.from_numpy(np.array(label_old[0])[:, 1]).type(torch.long)
    labels[valid_idx] = torch.from_numpy(np.array(label_old[1])[:, 1]).type(torch.long)
    labels[test_idx] = torch.from_numpy(np.array(label_old[2])[:, 1]).type(torch.long)

    return adjs_pt, node_feats, labels, train_idx, valid_idx, test_idx


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

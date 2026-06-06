'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from code.base_class.dataset import dataset
import torch
import numpy as np
import scipy.sparse as sp

class Dataset_Loader(dataset):
    data = None
    dataset_name = None

    def __init__(self, seed=None, dName=None, dDescription=None):
        super(Dataset_Loader, self).__init__(dName, dDescription)

    def adj_normalize(self, mx):
        # normalize adjacency matrix before passing to the model
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        # convert scipy sparse matrix into PyTorch format
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def encode_onehot(self, labels):
        # convert class labels into one-hot vectors
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        onehot_labels = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return onehot_labels

    def load(self):
        # Load citation network dataset
        print('Loading {} dataset...'.format(self.dataset_name))

        # read node information from the dataset
        idx_features_labels = np.genfromtxt("{}/node".format(self.dataset_source_folder_path), dtype=np.dtype(str))
        # extract feature matrix and labels
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        onehot_labels = self.encode_onehot(idx_features_labels[:, -1])

        # load link data from file and build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        # create node id mapping for easier indexing
        idx_map = {j: i for i, j in enumerate(idx)}
        reverse_idx_map = {i: j for i, j in enumerate(idx)}
        # load edge list and build adjacency matrix
        edges_unordered = np.genfromtxt("{}/link".format(self.dataset_source_folder_path), dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(onehot_labels.shape[0], onehot_labels.shape[0]), dtype=np.float32)
        # make graph undirected
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # add self-loops and normalize adjacency matrix
        norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0]))

        # convert everything into tensors
        features = torch.FloatTensor(np.array(features.todense()))
        # normalize node features so each row sums to 1
        rowsum = features.sum(1, keepdim=True).clamp(min=1e-8)
        features = features / rowsum
        labels = torch.LongTensor(np.where(onehot_labels)[1])
        adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj)

        if self.dataset_name == 'cora':
            train_per_class = 30
            val_per_class = 50
            test_per_class = 150

        elif self.dataset_name == 'citeseer':
            train_per_class = 20
            val_per_class = 30
            test_per_class = 200

        elif self.dataset_name == 'pubmed':
            train_per_class = 20
            val_per_class = 30
            test_per_class = 200

        # use fixed seeds so results can be reproduced
        np.random.seed(42)
        torch.manual_seed(42)

        idx_train = []
        idx_test = []
        idx_val = []

        num_classes = len(np.unique(labels.numpy()))

        # create train, validation, and test splits for each class
        for c in range(num_classes):
            class_idx = np.where(labels.numpy() == c)[0]
            # randomly shuffle nodes within the current class
            np.random.shuffle(class_idx)

            idx_train.extend(
                class_idx[:train_per_class]
            )

            idx_test.extend(
                class_idx[train_per_class:train_per_class + test_per_class]
            )

            idx_val.extend(
                class_idx[
                    train_per_class + test_per_class:
                    train_per_class + test_per_class + val_per_class
                ]
            )

        # save split indices as tensors
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        train_test_val = {'idx_train': idx_train, 'idx_test': idx_test, 'idx_val': idx_val}
        graph = {'node': idx_map, 'edge': edges, 'X': features, 'y': labels, 'utility': {'A': adj, 'reverse_idx': reverse_idx_map}}
        return {'graph': graph, 'train_test_val': train_test_val}

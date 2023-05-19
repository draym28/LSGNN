import os
import os.path as osp
import random
from math import ceil

import numpy as np
import scipy.sparse as sp

import torch
import torch_geometric.datasets as pygds
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import (add_remaining_self_loops, homophily,
                                   remove_self_loops)
from ogb.nodeproppred import PygNodePropPredDataset

from load_large_graph.dataset import load_nc_dataset
import config as c


ds_root = './data'


# -------------------------------------------------------------------
class DS:
    def __init__(self, x, y, edge_index):
        self.x = x
        self.y = y
        self.edge_index = edge_index

def get_dataset(name:str):
    name = name.lower()
    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = pygds.Planetoid(ds_root, name=name)
    elif name in ['chameleon', 'squirrel']:
        dataset = pygds.WikipediaNetwork(ds_root, name=name)
    elif name in ['texas', 'cornell', 'wisconsin']:
        dataset = pygds.WebKB(ds_root, name=name)
    elif name == 'actor':
        dataset = pygds.Actor(osp.join(ds_root, 'Actor'))
    elif name == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(root=osp.join(ds_root, 'ogb'), name=name)
        dataset = [ToUndirected()(dataset[0])]
    elif name in ['arxiv-year']:
        dataset = load_nc_dataset(dataname=name)
        data = dataset[0]
        ds = DS(
            x=data[0]['node_feat'], 
            y=data[1].squeeze().type(torch.long), 
            edge_index=data[0]['edge_index'], 
        )
        dataset = [ds]
    else:
        raise NameError('dataset not found, name error.')

    return dataset

def dataset_split(
    y, 
    num_nodes, 
    num_classes,  
    train_r=0.48, 
    val_r=0.32, 
    test_r=0.20, 
    seed=c.GLOBAL_SEED):

    assert train_r + val_r + test_r == 1

    np.random.seed(seed)
    idx = torch.LongTensor(list(range(num_nodes)))
    train_mask = torch.zeros([num_nodes], dtype=torch.bool)
    val_mask = torch.zeros([num_nodes], dtype=torch.bool)
    test_mask = torch.zeros([num_nodes], dtype=torch.bool)
    rest_mask = torch.ones([num_nodes], dtype=torch.bool)

    # ignore unlabeled data (label = -1)
    unlabeled_mask = (y == -1)
    num_labeled = (num_nodes - unlabeled_mask.sum()).item()
    if unlabeled_mask.sum() > 0:
        num_classes -= 1

    for k in range(num_classes):
        k_mask = (y == k)
        num_nodes_k = k_mask.sum().int().item()
        num_train_k = ceil(num_nodes_k * train_r)

        idx_k = np.random.permutation(idx[k_mask])
        train_mask[idx_k[:num_train_k]] = True
    
    num_val = round(val_r * num_labeled)

    rest_mask[train_mask] = False
    rest_mask[unlabeled_mask] = False
    rest_idx = np.random.permutation(idx[rest_mask])
    val_idx = idx[rest_idx[:num_val]]
    test_idx = idx[rest_idx[num_val:]]

    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # print(train_mask.sum() + val_mask.sum() + test_mask.sum())

    return train_mask, val_mask, test_mask


def get_dataset_split(ds, data, config, idx_repeat=0):
    if ds == 'ogbn-arxiv':
        train_mask, val_mask, test_mask = dataset_split_ogbn_arxiv(data)

    elif config['train_val_test'][0] == 0.48 and \
        ds in ['chameleon', 'squirrel', 'actor', 'cornell', 'texas', 'wisconsin']:
            train_mask = data.train_mask[:,idx_repeat]
            val_mask = data.val_mask[:,idx_repeat]
            test_mask = data.test_mask[:,idx_repeat]
    else:
        y = data.y
        N = data.x.shape[0]
        num_classes = y.unique().shape[0]
        train_mask, val_mask, test_mask = \
            dataset_split(y, N, num_classes, *config['train_val_test'], seed=c.GLOBAL_SEED+idx_repeat)
        set_seed(c.GLOBAL_SEED)
    
    return train_mask, val_mask, test_mask


def dataset_split_ogbn_arxiv(data):
    # node_year = torch.LongTensor(data.node_year).squeeze()
    node_year = data.node_year.squeeze()
    train_mask = (node_year <= 2017)
    val_mask = (node_year == 2018)
    test_mask = (node_year >= 2019)
    return train_mask, val_mask, test_mask


def scipy_coo_matrix_to_torch_sparse_tensor(sparse_mx):
    indices1 = torch.from_numpy(np.stack([sparse_mx.row, sparse_mx.col]).astype(np.int64))
    values1 = torch.from_numpy(sparse_mx.data)
    shape1 = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices=indices1, values=values1, size=shape1)


def cal_filter(edge_index, num_nodes):
    edge_index = edge_index.cpu()
    N = num_nodes

    # A
    edge_index, _ = remove_self_loops(edge_index=edge_index)
    edge_index_sl, _ = add_remaining_self_loops(edge_index=edge_index)
    
    # D
    adj_data = np.ones([edge_index.shape[1]], dtype=np.float32)
    adj_sp = sp.csr_matrix((adj_data, (edge_index[0], edge_index[1])), shape=[N, N])

    adj_sl_data = np.ones([edge_index_sl.shape[1]], dtype=np.float32)
    adj_sl_sp = sp.csr_matrix((adj_sl_data, (edge_index_sl[0], edge_index_sl[1])), shape=[N, N])

    # D-1/2
    deg = np.array(adj_sl_sp.sum(axis=1)).flatten()
    deg_sqrt_inv = np.power(deg, -0.5)
    deg_sqrt_inv[deg_sqrt_inv == float('inf')] = 0.0
    deg_sqrt_inv = sp.diags(deg_sqrt_inv)

    # filters
    DAD = sp.coo_matrix(deg_sqrt_inv * adj_sp * deg_sqrt_inv)
    DAD = scipy_coo_matrix_to_torch_sparse_tensor(DAD)

    return DAD


def set_seed(seed=28):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    ds_list = [
        'cora', 'citeseer', 'pubmed', 
        'chameleon', 'squirrel', 'actor', 
        'cornell', 'texas', 'wisconsin', 
        'ogbn-arxiv', 'arxiv-year'
    ]

    import pandas as pd
    df = pd.DataFrame(index=['H_node', 'H_edge'], columns=ds_list)

    for i, ds_name in enumerate(ds_list):
        dataset = get_dataset(ds_name)
        data = dataset[0]
        node_homo = homophily(data.edge_index, data.y, method='node')
        edge_homo = homophily(data.edge_index, data.y, method='edge')
        df[ds_name] = [node_homo, edge_homo]
    print(df)
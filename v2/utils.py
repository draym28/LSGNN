import os
import os.path as osp
import random
from math import ceil

import numpy as np
import scipy.sparse as sp
import torch

import torch_geometric
import torch_geometric.datasets as pygds
from torch_geometric.data import Data as pygData
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, degree, to_undirected
from torch_geometric.utils import homophily as pyg_homo
from torch_geometric.transforms import ToUndirected
from torch_scatter import scatter_mean
from torch_sparse import SparseTensor

import config as c
from load_large_graph.dataset import load_nc_dataset


class DS:
    def __init__(self, x, y, edge_index):
        self.x = x
        self.y = y
        self.edge_index = edge_index


def get_dataset(name:str, undirected=False):
    if 'DS_OLD' in os.environ and int(os.environ['DS_OLD']):
        ds_dir = c.ds_dir_old
    else:
        ds_dir = c.ds_dir
    name = name.lower()
    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = pygds.Planetoid(ds_dir, name=name)
    elif name in ['chameleon', 'squirrel']:
        dataset = pygds.WikipediaNetwork(ds_dir, name=name)
    elif name in ['texas', 'cornell', 'wisconsin']:
        dataset = pygds.WebKB(ds_dir, name=name)
    elif name == 'actor':
        dataset = pygds.Actor(osp.join(ds_dir, 'Actor'))
    elif name == 'ogbn-arxiv':
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(root=osp.join(ds_dir, 'ogb'), name=name)
    elif 'ogb' in name:
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(root=osp.join(ds_dir, 'ogb'), name=name)
    elif name in ["AmazonProducts", "Reddit2", "Reddit", "Flickr", "Yelp"]:
        path = osp.join(ds_dir, name)
        dataset_class = getattr(torch_geometric.datasets, name)
        dataset = dataset_class(path)
    elif name in ["Photo", "Computers"]:
        from torch_geometric.datasets  import Amazon
        dataset = Amazon(root=ds_dir, name = name)
    elif name in ["CS", "Physics"]:
        from torch_geometric.datasets import Coauthor
        dataset = Coauthor(root=ds_dir, name = name)
    elif name in ['penn94']:
        dataset = load_nc_dataset(dataname='fb100', sub_dataname='Penn94')
    # elif name in ['pokec', 'arxiv-year', 'snap-patents', 'genius', 'twitch-gamer', 'ogbn-arxiv']:
    elif name in ['pokec', 'arxiv-year', 'snap-patents', 'genius', 'twitch-gamer']:
        dataset = load_nc_dataset(dataname=name)
    elif name in ["Photo", "Computers"]:
        from torch_geometric.datasets  import Amazon
        dataset = Amazon(root=ds_dir, name = name)
    elif name in ["CS", "Physics"]:
        from torch_geometric.datasets import Coauthor
        dataset = Coauthor(root=ds_dir, name = name)
    else:
        raise NameError('dataset not found, name error.')
    if name in c.large_graph and name != 'ogbn-arxiv':
        data = dataset[0]
        x = data[0]['node_feat']
        y = data[1].squeeze().type(torch.long)
        edge_index = data[0]['edge_index']
        if undirected:
            edge_index = to_undirected(edge_index, num_nodes=x.shape[0])
        data = DS(
            x=x, y=y, 
            edge_index=edge_index, 
        )
    else:
        if undirected:
            data = ToUndirected()(dataset[0])
        else:
            data = dataset[0]
    return data


def class_rand_splits(label, tr_num_per_class=20, val_num_per_class=30):
    train_idx, valid_idx, test_idx = [], [], []
    idx = torch.arange(label.shape[0])
    class_list = label.squeeze().unique()
    for i in range(class_list.shape[0]):
        c_i = class_list[i]
        idx_i = idx[label.squeeze() == c_i]
        n_i = idx_i.shape[0]
        rand_idx = idx_i[torch.randperm(n_i)]
        train_idx += rand_idx[:tr_num_per_class].tolist()
        valid_idx += rand_idx[tr_num_per_class:tr_num_per_class+val_num_per_class].tolist()
        test_idx += rand_idx[tr_num_per_class+val_num_per_class:].tolist()
    train_idx = torch.as_tensor(train_idx)
    valid_idx = torch.as_tensor(valid_idx)
    test_idx = torch.as_tensor(test_idx)
    test_idx = test_idx[torch.randperm(test_idx.shape[0])]

    return train_idx, valid_idx, test_idx


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


def dataset_split_ogbn_arxiv(data):
    # node_year = torch.LongTensor(data.node_year).squeeze()
    node_year = data.node_year.squeeze()
    train_mask = (node_year <= 2017)
    val_mask = (node_year == 2018)
    test_mask = (node_year >= 2019)
    return train_mask, val_mask, test_mask


def edge_index_to_adj(edge_index:torch.Tensor, num_nodes:int):
    adj = SparseTensor(
        row=edge_index[1], col=edge_index[0], 
        value=torch.ones(edge_index.shape[1]).to(edge_index.device), 
        sparse_sizes=[num_nodes, num_nodes]
    )
    return adj


# def cal_filter(edge_index, num_nodes, transposed=True):
#     edge_index, _ = remove_self_loops(edge_index)
#     row, col = edge_index

#     edge_index_sl, _ = add_remaining_self_loops(edge_index)
#     deg = degree(edge_index_sl[1], num_nodes=num_nodes)
#     deg_norm = torch.pow(deg.to(edge_index.device), -0.5)
#     deg_norm = torch.nan_to_num(deg_norm, nan=0.0, posinf=0.0, neginf=0.0)

#     value = deg_norm[col] * deg_norm[row]
#     adj = SparseTensor(
#         row=col, col=row, 
#         value=value, sparse_sizes=[num_nodes, num_nodes]).to(edge_index.device)

#     if transposed:
#         return adj.t()
#     else:
#         return adj


def scipy_coo_matrix_to_torch_sparse_tensor(sparse_mx):
    indices1 = torch.from_numpy(np.stack([sparse_mx.row, sparse_mx.col]).astype(np.int64))
    values1 = torch.from_numpy(sparse_mx.data)
    shape1 = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices=indices1, values=values1, size=shape1)

def cal_filter(edge_index, num_nodes, beta=0.5, transposed=True):
    dev = edge_index.device
    edge_index = edge_index.cpu()
    N = num_nodes

    edge_index, _ = remove_self_loops(edge_index=edge_index)
    adj_data = np.ones([edge_index.shape[1]], dtype=np.float32)
    if transposed:
        adj_sp = sp.csr_matrix((adj_data, (edge_index[0], edge_index[1])), shape=[N, N])
    else:
        adj_sp = sp.csr_matrix((adj_data, (edge_index[1], edge_index[0])), shape=[N, N])

    edge_index_sl, _ = add_remaining_self_loops(edge_index=edge_index)
    adj_sl_data = np.ones([edge_index_sl.shape[1]], dtype=np.float32)
    if transposed:
        adj_sl_sp = sp.csr_matrix((adj_sl_data, (edge_index_sl[0], edge_index_sl[1])), shape=[N, N])
    else:
        adj_sl_sp = sp.csr_matrix((adj_sl_data, (edge_index_sl[1], edge_index_sl[0])), shape=[N, N])

    # D-1/2
    deg = np.array(adj_sl_sp.sum(axis=1)).flatten()
    deg_sqrt_inv = np.power(deg, -0.5)
    deg_sqrt_inv[deg_sqrt_inv == float('inf')] = 0.0
    deg_sqrt_inv = sp.diags(deg_sqrt_inv)

    # filters
    I = sp.eye(num_nodes)
    DAD = deg_sqrt_inv * adj_sp * deg_sqrt_inv
    filter_l = sp.coo_matrix(beta * I + DAD)
    filter_h = sp.coo_matrix((1. - beta) * I - DAD)
    filter_l = scipy_coo_matrix_to_torch_sparse_tensor(filter_l)
    filter_h = scipy_coo_matrix_to_torch_sparse_tensor(filter_h)
    filter_l = SparseTensor.from_torch_sparse_coo_tensor(filter_l)
    filter_h = SparseTensor.from_torch_sparse_coo_tensor(filter_h)

    return filter_l.to(dev), filter_h.to(dev)

def cal_filter_sl(edge_index, num_nodes, transposed=True):
    edge_index = edge_index.cpu()
    N = num_nodes

    # A
    edge_index_sl, _ = add_remaining_self_loops(edge_index=edge_index)

    # D
    adj_sl_data = np.ones([edge_index_sl.shape[1]], dtype=np.float32)
    if transposed:
        adj_sl_sp = sp.csr_matrix((adj_sl_data, (edge_index_sl[0], edge_index_sl[1])), shape=[N, N])
    else:
        adj_sl_sp = sp.csr_matrix((adj_sl_data, (edge_index_sl[1], edge_index_sl[0])), shape=[N, N])

    # D-1/2
    deg = np.array(adj_sl_sp.sum(axis=1)).flatten()
    deg_sqrt_inv = np.power(deg, -0.5)
    deg_sqrt_inv[deg_sqrt_inv == float('inf')] = 0.0
    deg_sqrt_inv = sp.diags(deg_sqrt_inv)

    # filters
    DAD = sp.coo_matrix(deg_sqrt_inv * adj_sl_sp * deg_sqrt_inv)
    DAD = scipy_coo_matrix_to_torch_sparse_tensor(DAD)
    # DAD = SparseTensor.from_torch_sparse_coo_tensor(DAD)

    return DAD


def edge_homophily(data:pygData, mean_reduce=True):
    num_edges = data.num_edges
    row, col = data.edge_index
    ans = torch.zeros([num_edges], device=data.x.device)
    ans[data.y[row] == data.y[col]] = 1.

    if mean_reduce:
        return ans.float().mean().item()
    else:
        return ans.float()


def node_homophily(data:pygData, mean_reduce=True):
    row, col = data.edge_index
    src = torch.zeros([row.shape[0]], device=data.x.device)
    src[data.y[row] == data.y[col]] = 1.
    ans = scatter_mean(src, col, dim_size=data.y.shape[0])

    if mean_reduce:
        return ans.float().mean().item()
    else:
        return ans.float()


def node_homophily2(edge_index, y, mean_reduce=True):
    row, col = edge_index
    src = torch.zeros([col.shape[0]])
    src[y[row] == y[col]] = 1.
    ans = scatter_mean(src, col.cpu(), dim_size=y.cpu().shape[0])

    if mean_reduce:
        return ans.float().mean().item()
    else:
        return ans.float()


def homophily(data:pygData, method='edge'):
    edge_index = data.edge_index
    y = data.y
    return pyg_homo(edge_index, y, method=method)


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
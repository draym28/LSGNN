import json
import numpy as np
import argparse

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

import config as c
import model as m
import utils as u

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


DIST = None
DAD = None
X_OUT_L_OUT_H = None
SHOW_CONFIG = 0

def train(ds:str, model_name='lsgnn', config=None, n=0):
    # get dataset
    data = u.get_dataset(ds)[0]

    if config == None:
        config = c.lsgnn(ds)

    x = F.normalize(data.x.to(device), p=2, dim=-1)
    N = x.shape[0]
    y = data.y.to(device).squeeze()
    edge_index = data.edge_index.to(device)
    num_classes = y.unique().shape[0]

    # dataset split
    train_mask, val_mask, test_mask = u.get_dataset_split(ds, data, config, idx_repeat=n)
    u.set_seed(c.GLOBAL_SEED)

    # ignore unlabeled data (label = -1)
    if (y == -1).sum() > 0:
        num_classes -= 1

    # get config and model
    if model_name == 'lsgnn':
        model = m.LSGNN(
            in_channels=x.shape[1], 
            out_channels=num_classes, 
            config=config, 
            num_nodes=N, 
            ds=ds, 
        ).to(device)
    else:
        raise NotImplementedError

    # pre-computed
    global DIST, DAD, X_OUT_L_OUT_H, SHOW_CONFIG
    if DIST == None:
        print('pre-computing distances...')
        DIST = model.dist(x, edge_index)
    if DAD == None:
        print('pre-computing DAD...')
        N = x.shape[0]
        DAD = u.cal_filter(edge_index, N)
        DAD = DAD.to(device)
    if X_OUT_L_OUT_H == None:
        print('pre-computing intermediate representations...')
        X_OUT_L_OUT_H = model.prop(x, edge_index, DAD)

    # show config
    if not SHOW_CONFIG:
        config_dict = json.dumps(config, indent=4)
        print(config_dict)
        SHOW_CONFIG = 1

    # optim and loss
    if config['adamw']:
        optim = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['wd'])
    else:
        optim = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['wd'])
    loss_fn = nn.CrossEntropyLoss()

    # train val
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_test_acc = 0.0
    train_acc_history = []
    val_acc_history = []
    test_acc_history = []
    for _ in tqdm(range(config['num_epochs']), ncols=70, desc='iter %d' % (n)):
        # train
        model.train()
        optim.zero_grad()
        logits = model(x, edge_index, DIST, X_OUT_L_OUT_H)
        pred = logits.argmax(dim=1)

        loss = loss_fn(logits[train_mask], y[train_mask])
        loss.backward()
        optim.step()

        train_acc = (pred[train_mask].eq(y[train_mask]).float().sum() / train_mask.sum()).item()
        train_acc_history.append(train_acc)

        # val and test
        model.eval()
        logits = model(x, edge_index, DIST, X_OUT_L_OUT_H)
        pred = logits.argmax(dim=1)

        val_loss = loss_fn(logits[val_mask], y[val_mask])
        val_acc = (pred[val_mask].eq(y[val_mask]).sum() / val_mask.sum()).item()
        val_acc_history.append(val_acc)

        test_acc = (pred[test_mask].eq(y[test_mask]).sum() / test_mask.sum()).item()
        test_acc_history.append(test_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_test_acc = test_acc

    print('model: %s, dataset: %s, val acc: %.4f, test acc: %.4f' % \
        (model_name, ds, best_val_acc, best_test_acc))

    return best_val_acc, best_test_acc

def train_ntime(ds:str, model_name:str='lsgnn', config=None, num_train=None):
    print('model: %s, dataset: %s' % (model_name, ds))

    global DIST, DAD, X_OUT_L_OUT_H, SHOW_CONFIG
    DIST = None
    DAD = None
    X_OUT_L_OUT_H = None
    SHOW_CONFIG = 0

    if num_train is None:
        num_train = 10
        if ds == 'ogbn-arxiv':
            num_train = 1
        elif ds in c.large_graph:
            num_train = 5

    result = []
    for n in range(num_train):
        best_val_acc, best_test_acc = train(ds, model_name=model_name, config=config, n=n)
        result.append([best_val_acc, best_test_acc])
    
    val_acc_mean, test_acc_mean = np.mean(result, axis=0) * 100
    val_acc_std, test_acc_std = np.sqrt(np.var(result, axis=0)) * 100

    print('model: %s, dataset: %s, val_acc: %.2f±%.2f, test acc: %.2f±%.2f' % \
        (model_name, ds, val_acc_mean, val_acc_std, test_acc_mean, test_acc_std))

    return test_acc_mean, test_acc_std

if __name__ == '__main__':
    u.set_seed(c.GLOBAL_SEED)

    ds_list = [
        'cora', 'citeseer', 'pubmed', 
        'chameleon', 'squirrel', 
        'actor', 
        'cornell', 'texas', 'wisconsin', 
        'ogbn-arxiv', 'arxiv-year'
    ][-5:-4]

    parser = argparse.ArgumentParser(description='LSGNN')
    parser.add_argument("--dataset", default='cora', dest='dataset', type=str, choices=ds_list, help='Dataset')
    args = parser.parse_args()

    acc_mean, acc_std = train_ntime(args.dataset, model_name='lsgnn')
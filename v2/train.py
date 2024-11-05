import os
import json
from time import time
from copy import deepcopy
from termcolor import cprint
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from tqdm import tqdm

import config as c
import model as m
import utils as u

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
folder_name = os.path.dirname(__file__).split(os.sep)[-1]


SAVE_ACC = 0
SAVE_MODEL = 0

DIST = None
X_OUT_L_OUT_H = None


def train(ds:str, model_name:str='lsgnn', config=None, n=0):
    u.set_seed(c.GLOBAL_SEED)

    # get dataset
    data = u.get_dataset(ds, undirected=config['undirected'])

    N = data.x.shape[0]
    y = data.y.squeeze()  # .type(torch.float32)
    if ds in ["AmazonProducts", "Yelp"]:
        y = torch.argmax(y, -1).squeeze()
        # num_classes = y.shape[-1]
    num_classes = y.unique().shape[0]
    if ds in ["AmazonProducts", "Yelp"]:
        dic_num_classes = {"AmazonProducts": 107, "Yelp": 100}
        num_classes = dic_num_classes[ds]

    # dataset split
    if ds in ['chameleon', 'squirrel', 'actor', 'cornell', 'texas', 'wisconsin'] and \
            config['train_val_test'] == [0.48, 0.32, 0.20]:
        train_mask = data.train_mask[:,n]
        val_mask = data.val_mask[:,n]
        test_mask = data.test_mask[:,n]
    elif ds == 'ogbn-arxiv':
        train_mask, val_mask, test_mask = u.dataset_split_ogbn_arxiv(data)
    elif 'ogb' in ds:
        split_idx = dataset.get_idx_split()
        train_mask, val_mask, test_mask = split_idx["train"], split_idx["valid"], split_idx["test"]
    elif ds in ["AmazonProducts", "Reddit2", "Reddit", "Flickr", "Yelp"]:
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
    else:
        train_mask, val_mask, test_mask = \
            u.dataset_split(y, N, num_classes, *config['train_val_test'], seed=c.GLOBAL_SEED+n)
        u.set_seed(c.GLOBAL_SEED)
    # ignore unlabeled data (label = -1)
    if (y == -1).sum() > 0:
        num_classes -= 1

    x = F.normalize(data.x.to(device), p=2, dim=-1)
    y = y.to(device)
    edge_index = data.edge_index.to(device)

    # get config and model
    if model_name == 'lsgnn':
        model = m.LSGNN(
            in_channels=x.shape[1], 
            out_channels=num_classes, 
            config=config, 
            num_nodes=N, 
            ds=ds)
    else:
        raise NameError('model not found, model name error.')
    model = model.to(device)

    global DIST, X_OUT_L_OUT_H
    if DIST is None:
        print('pre-computing dist...')
        DIST = model.dist(x, edge_index)
        print('pre-computing filters...')
        filters = u.cal_filter(edge_index, N, beta=config['beta'], transposed=config['transposed'])
        print('pre-computing intermediate representations...')
        X_OUT_L_OUT_H = model.prop(x, filters)

    # optim
    if config['adamw']:
        base_optim = torch.optim.AdamW
    else:
        base_optim = torch.optim.Adam
    optim = base_optim(model.parameters(), lr=config['lr'], weight_decay=config['wd'])

    # train val
    best_model_sd = None
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

        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        optim.step()

        train_acc = (logits.argmax(dim=1)[train_mask].eq(y[train_mask]).float().sum() / train_mask.sum()).item()
        train_acc_history.append(train_acc)

        # val and test
        with torch.no_grad():
            model.eval()
            logits = model(x, edge_index, DIST, X_OUT_L_OUT_H)

            val_loss = F.cross_entropy(logits[val_mask], y[val_mask])
            val_acc = (logits.argmax(dim=1)[val_mask].eq(y[val_mask]).sum() / val_mask.sum()).item()
            val_acc_history.append(val_acc)

            test_acc = (logits.argmax(dim=1)[test_mask].eq(y[test_mask]).sum() / test_mask.sum()).item()
            test_acc_history.append(test_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_model_sd = deepcopy(model.state_dict())

    if SAVE_ACC:
        plt.cla()
        x = list(range(1, config['num_epochs'] + 1))
        plt.plot(x, train_acc_history, label='train')
        plt.plot(x, val_acc_history, label='val')
        plt.plot(x, test_acc_history, label='test')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.title('%s %s' % (ds, model_name))
        plt.legend()
        plt.savefig('./img/%s_%s_%d_%.2f_%.2f_%.2f_acc.png' % (model_name, ds, n, *config['train_val_test']))

    print('%s, model: %s, dataset: %s, val acc: %.4f, test acc: %.4f' % \
          (folder_name, model_name, ds, best_val_acc, best_test_acc))

    return best_val_acc, best_test_acc, best_model_sd


def train_ntime(ds:str, model_name:str='lsgnn', config=None, num_train=None, log_root='log'):
    print(f'{folder_name}, model: {model_name}, dataset: {ds}')
    os.makedirs(log_root, exist_ok=True)

    if config is None:
        if model_name == 'lsgnn':
            config = c.lsgnn(ds)

    global DIST, X_OUT_L_OUT_H
    DIST = None
    X_OUT_L_OUT_H = None

    if num_train is None:
        if ds == 'ogbn-arxiv':
            num_train = 1
        elif ds in c.large_graph:
            num_train = 5
        else:
            num_train = 10

    val_accs = []
    test_accs = []
    t = []
    for n in range(num_train):
        t0 = time()
        best_val_acc, best_test_acc, best_model_sd = train(ds, model_name, config=config, n=n)
        t.append(time() - t0)
        val_accs.append(best_val_acc)
        test_accs.append(best_test_acc)

        if SAVE_MODEL:
            os.makedirs('./best_model', exist_ok=True)
            torch.save(best_model_sd, f'./best_model/{model_name}_{ds}_{n}.ckpt')
        # break

    val_acc_mean = np.mean(val_accs) * 100
    val_acc_std = np.std(val_accs) * 100
    test_acc_mean = np.mean(test_accs) * 100
    test_acc_std = np.std(test_accs) * 100

    mean_time = np.mean(t)

    results = {
        # 'val_acc_mean': val_acc_mean, 
        # 'val_acc_std': val_acc_std, 
        'test_acc_mean': test_acc_mean, 
        'test_acc_std': test_acc_std, 
        # 'val_accs': val_accs, 
        'test_accs': test_accs, 
    }

    save_results = {'results': results, 'config': config}
    with open(f'{log_root}/results_{model_name}_{ds}.json', 'w') as f:
        json.dump(save_results, f, indent=4)

    cprint('%s, model: %s, dataset: %s, val acc: %.2f±%.2f, test acc: %.2f±%.2f, mean_time: %.2fs' % \
           (folder_name, model_name, ds,
            val_acc_mean, val_acc_std, test_acc_mean, test_acc_std,
            mean_time), 'green')

    return test_acc_mean, test_acc_std, results


if __name__ == '__main__':
    ds_list = [
        'cora', 'citeseer', 'pubmed',
        'chameleon', 'squirrel',
        'actor',
        'cornell', 'texas', 'wisconsin',
        'ogbn-arxiv', 'arxiv-year', 
    ]
    model_list = ['lsgnn']

    acc_dict = {}
    for ds in ds_list:
        acc_dict['%s_mean' % (ds)] = []
        acc_dict['%s_std' % (ds)] = []

    acc_dict = {}
    for ds in ds_list:
        acc_dict['%s_mean' % (ds)] = []
        acc_dict['%s_std' % (ds)] = []

    for model_name in model_list:
        for dataset in ds_list:
            # config = None
            config_file = glob.glob(f"log/config/log*{dataset}*.json")[-1]
            config = json.load(open(config_file))['config']
            if config['ds_old']:
                os.environ['DS_OLD'] = str(1)
            acc_mean, acc_std, results = train_ntime(dataset, model_name, config, log_root='log/')
            acc_dict['%s_mean' % (dataset)].append(acc_mean)
            acc_dict['%s_std' % (dataset)].append(acc_std)

    df = pd.DataFrame(acc_dict, index=model_list)
    print(df)
    # df.to_csv('./acc.csv')
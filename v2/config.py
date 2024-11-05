import os
ds_dir = os.path.abspath(os.path.dirname(__file__) + '/../_data/')
ds_dir_old = os.path.abspath(os.path.dirname(__file__) + '/../_data/_old/')

GLOBAL_SEED = 28


large_graph = ['ogbn-arxiv', 'arxiv-year']


def default_config(ds='cora'):
    config = {
        'num_epochs': 200, 
        'train_val_test': [0.48, 0.32, 0.20], 
        'undirected': False, 
        'transposed': True, 

        'num_layers': 2,  # for SGC, GCN, etc.
        'hidden_channels': 16, 

        'lr': 0.01, 
        'wd': 5e-4, 
        'dropout': 0.5, 
    }

    if ds == 'ogbn-arxiv':
        config['num_epochs'] = 500
        config['train_val_test'] = None
        config['undirected'] = True
        config['transposed'] = False

    elif ds in large_graph:
        config['train_val_test'] = [0.50, 0.25, 0.25]

    return config


def lsgnn(ds='cora'):
    config = default_config(ds)

    config['adamw'] = False

    config['num_reduce_layers'] = 1  # num of reduce heads
    config['A_embed'] = False        # whether use A embedding
    config['out_norm'] = True        # whether use normalization before prediction
    config['out_mlp'] = False        # whether use mlp for prediction head, otherwise linear
    config['irdc'] = True, 

    config['K'] = 5
    config['beta'] = 1.
    config['gamma'] = 0.5
    config['method'] = 'norm2'

    config['ds_old'] = False

    if ds == 'ogbn-arxiv':
        config['hidden_channels'] = 64
        config['out_norm'] = False
        config['irdc'] = False
        config['wd'] = 0.001
        config['adamw'] = True

    elif ds in large_graph:
        config['hidden_channels'] = 32
        config['num_reduce_layers'] = 2
        config['A_embed'] = True
        config['out_norm'] = False
        config['out_mlp'] = True

    elif ds == 'cornell':
        config['ds_old'] = True

    return config
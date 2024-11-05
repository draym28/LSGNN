GLOBAL_SEED = 42

large_graph = [ 'arxiv-year', 'ogbn-arxiv']

def default_config():
    config = {
        # Generally fixed
        'num_epochs': 200, 
        'hidden_channels': 16, 
        'num_reduce_layers': 1,  # num of reduce heads
        'A_embed': False,        # whether use A embedding
        'out_norm': True,        # whether use normalization before prediction
        'out_mlp': False,        # whether use mlp for prediction head, otherwise linear
        'adamw': False, 
        'K': 5, 
        'train_val_test': [0.48, 0.32, 0.2], 

        # params need to search
        'lr': 0.01, 
        'wd': 5e-4, 
        'dropout': 0.5, 
        'beta': 1., 
        'gamma': 0.5, 
        'method': 'norm2', 
    }
    return config


def lsgnn(ds='chameleon'):
    config = default_config()

    if ds == 'ogbn-arxiv':
        config['num_epochs'] = 500
        config['hidden_channels'] = 64
        config['out_norm'] = False
        config['wd'] = 0.001
        config['adamw'] = True
        config['train_val_test'] = None

    elif ds in large_graph:
        config['hidden_channels'] = 32
        config['num_reduce_layers'] = 2
        config['A_embed'] = True
        config['out_norm'] = False
        config['out_mlp'] = True
        config['train_val_test'] = [0.50, 0.25, 0.25]

    return config
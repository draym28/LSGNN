import os


for fname in os.listdir('.'):
    if fname.endswith('.json'):
        new_name = fname.replace('_K5', '').replace('_dir', '').replace('_undir', '').replace('_adj', '').replace('_adjt', '')
        os.remove(fname)
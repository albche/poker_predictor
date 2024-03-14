import glob
import numpy as np
import torch
from tqdm.auto import tqdm

def load_data(mode='training'):
    input_glob, target_glob = f'data/{mode}/input*.pt', f'data/{mode}/target*.pt'
    input_fns = np.sort(glob.glob(input_glob))
    target_fns = np.sort(glob.glob(target_glob))
    all_inputs = []
    all_targets = []
    print(f'Loading {mode} Inputs...')
    for fn in tqdm(input_fns):
        all_inputs += torch.load(fn)
    print(f'Loading {mode} Targets...')
    for fn in tqdm(target_fns):
        all_targets += torch.load(fn)
    return all_inputs, all_targets
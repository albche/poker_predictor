import glob
import numpy as np
import torch
from tqdm.auto import tqdm

def load_data(encodings='fully_encoded', mode='training', p=1.0):
    input_glob, target_glob = f'data/{encodings}/{mode}/input*.pt', f'data/{encodings}/{mode}/target*.pt'

    assert(p >= 0.05 and p <= 1.0) #smallest value allowed for p is 0.05 since validation doesn't have enough data to support less

    input_fns = np.sort(glob.glob(input_glob))
    target_fns = np.sort(glob.glob(target_glob))
    indices = torch.randperm(len(input_fns))[:round(len(input_fns)*p)]
    input_fns = input_fns[:round(len(input_fns)*p)]
    target_fns = target_fns[:round(len(target_fns)*p)]

    #in case theres only one element and it becomes a string
    if type(input_fns) == np.str_: 
        input_fns = np.array([input_fns])
    if type(target_fns) == np.str_: 
        target_fns = np.array([target_fns])

    all_inputs = []
    all_targets = []

    print(f'Loading {mode} Inputs...')
    for fn in tqdm(input_fns):
        all_inputs += torch.load(fn)
    print(f'Loading {mode} Targets...')
    for fn in tqdm(target_fns):
        all_targets += torch.load(fn)
    
    return all_inputs, all_targets
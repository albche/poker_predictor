import glob
import numpy as np
import torch
from tqdm.auto import tqdm
import os

def load_batches(batch_size):
    print('Batching...')
    return_tuple = []
    for data_group in ('training', 'validation', 'testing'):
        print(f'Loading data for {data_group} dataset')
        glob_path = os.path.join('data', data_group, '*.pt')
        all_data = []
        print('Reading in raw .pt data...')
        for fn in tqdm(glob.glob(glob_path)[:1]):
            all_data += (torch.load(fn))
        random_indices = np.random.permutation(range(len(all_data)))

        input_batches = []
        target_batches = []
        print('Batching...')
        for i in tqdm(range(0, len(all_data), batch_size)):
            end_index = min((i+batch_size), len(all_data))
            batched_inputs = [torch.transpose(all_data[random_indices[j]][0], 0, 1) for j in range(i, end_index)]
            batched_targets = [all_data[random_indices[j]][1] for j in range(i, end_index)]
            input_batches.append(torch.stack(batched_inputs, dim=0))
            target_batches.append(torch.stack(batched_targets, dim=0))
        return_tuple.append(input_batches)
        return_tuple.append(target_batches)
    print('\n')
    return tuple(return_tuple)
import json

import torch
from torch.utils.data import Dataset

from typing import Dict, List, Tuple, Any, Set, List

from src.token import SpecialToken
from src.utils.data_augment import augment_compact_grids
from src.utils.helper import print_thread_info
from src.utils.grid_data_process import shuffle_all_but_last_pair, tokenize_compact_task, end_of_examples_mark, preprocess_for_fast_access, preprocess_array_data, pad_collate, shuffle_all_pairs
class GridDataset(Dataset):
    def __init__(self):
        self.data = []
        self.augment_seed = -1

    def __len__(self):
        return len(self.data)

    @staticmethod
    def convert_to_int_lists(np_data):
        return [int(x) for x in np_data]

    def __getitem__(self, idx):
        if self.augment_seed >= 0:                
            augmented_task = augment_compact_grids(self.convert_to_int_lists(self.data[idx]))
            shuffled_task = shuffle_all_but_last_pair(augmented_task)
            task = tokenize_compact_task(shuffled_task)
        else:
            task = tokenize_compact_task(self.convert_to_int_lists(self.data[idx]))
        
        end_of_examples_index = end_of_examples_mark(task)
        assert end_of_examples_index > 0
        
        return {
            'task': task,
            'idx': idx,
            'end_of_examples_index': end_of_examples_index
        }

    def set_augment_seed(self, augment_seed):
        self.augment_seed = augment_seed
    
    def set_max_length(self, max_length):
        self.max_length = int(max_length)

    def set_source_ranges(self, source_ranges: Dict[str, Tuple[int, int]]):
        self.source_ranges = source_ranges

    def cut_long_sequence(self, threshold_length):
        # iterate over self.data remove all elements longer than threshold_length
        self.data = [seq for seq in self.data if len(seq) <= threshold_length]

    def sort_by_length(self, *, reverse:bool):
        self.data.sort(key=len, reverse=reverse)

    @classmethod
    def load_from_paired_file(cls, challenges: Dict[str, Any], solutions: Dict[str, Any], source_ranges: Dict[str, Tuple[int, int]] = {'ignore': (-1, -1)}, second_only: bool = False) -> 'GridDataset':
        instance = cls()
        instance.source_ranges = source_ranges
        instance.second_only = second_only
        preprocess_for_fast_access(challenges, solutions, instance.second_only, instance.data)
        return instance
            
    def pad_collate(self, batch):
        return pad_collate(batch, self.max_length)

class DynamicGridDataset(Dataset):
    def __init__(self, compact_grid, sample_size, max_seq_length):
        self.compact_grid = compact_grid
        self.sample_size = sample_size
        self.max_seq_length = max_seq_length

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        augmented_task = augment_compact_grids(self.compact_grid)
        shuffled_task = shuffle_all_pairs(augmented_task)
        task = tokenize_compact_task(shuffled_task)
        
        end_of_examples_index = end_of_examples_mark(task)
        assert end_of_examples_index > 0
        
        return {
            'task': task,
            'idx': idx,
            'end_of_examples_index': end_of_examples_index
        }
        
    def pad_collate(self, batch):
        return pad_collate(batch, self.max_seq_length)

# Loading JSON data
def load_json(file_path: str) -> dict:
    with open(file_path) as f:
        data = json.load(f)
    return data

def load_from_json(case: str, base_path: str) -> tuple:
    with open(base_path + case + '_challenges.json') as f:
        challenges = json.load(f)

    try:
        with open(base_path + case + '_solutions.json') as f:
            solutions = json.load(f)
    except FileNotFoundError:
        print(f"Skipping {case}_solutions: File not found.")
        solutions = None

    return challenges, solutions

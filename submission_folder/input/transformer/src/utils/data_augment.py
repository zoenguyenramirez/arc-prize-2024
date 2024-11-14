# src/utils/data_augment.py

import random
import copy
import itertools
from src.token import SpecialToken

def rotate_grid(compact_grid: list[int], rotation):
    height, width = compact_grid[:2]
    grid = [compact_grid[i:i+width] for i in range(2, len(compact_grid), width)]
    
    if rotation == 0:
        return compact_grid
    elif rotation == 1:
        rotated = list(map(list, zip(*grid[::-1])))
    elif rotation == 2:
        rotated = [row[::-1] for row in grid[::-1]]
    elif rotation == 3:
        rotated = list(map(list, zip(*grid)))[::-1]
    else:
        raise ValueError("Rotation must be 0, 1, 2, or 3")
    
    return [len(rotated), len(rotated[0])] + [cell for row in rotated for cell in row]

def flip_grid_vertical(compact_grid: list[int]):
    height, width = compact_grid[:2]
    grid = [compact_grid[i:i+width] for i in range(2, len(compact_grid), width)]
    flipped = grid[::-1]
    return [height, width] + [cell for row in flipped for cell in row]
    
def permute_mapping():
    perm = list(range(1, 10))
    random.shuffle(perm)

    mapping = dict(zip(range(1, 10), perm))
    mapping[0] = 0  # Ensure 0 always maps to 0

    return mapping

def permute(compact_grid: list[int], mapping):
    height, width = compact_grid[:2]
    permuted = [height, width] + [mapping.get(cell, cell) for cell in compact_grid[2:]]
    return permuted

def augment_compact_grids(compact_grids: list[int], *, return_config = False):
    augmented_task = []
    offset = 0

    rotation = random.randint(0, 3)
    flip = random.random() < 0.5
    mapping = permute_mapping()

    # print('\nrotation, flip, mapping', rotation, flip, mapping)

    while offset < len(compact_grids):
        if compact_grids[offset] in [SpecialToken.START_INPUT.value, SpecialToken.START_OUTPUT.value]:
            augmented_task.append(compact_grids[offset])
            offset += 1
        
        height, width = compact_grids[offset:offset+2]
        grid_size = height * width

        grid = compact_grids[offset:offset+grid_size+2]
        
        # Apply augmentations
        grid = rotate_grid(grid, rotation)

        # print('post rotate_grid', grid)
        
        if flip:
            grid = flip_grid_vertical(grid)

        # print('post flip_grid_vertical', grid)
        
        grid = permute(grid, mapping)

        # print('post permute', grid)
        
        augmented_task.extend(grid)
        offset += grid_size + 2

    if return_config:
        return augmented_task, rotation, flip, mapping
    else:
        return augmented_task

def reverse_augment_compact_grids(augmented_grids: list[int], rotation: int, flip: bool, mapping: dict):
    reversed_task = []
    offset = 0

    # Create reverse mapping
    reverse_mapping = {v: k for k, v in mapping.items()}

    while offset < len(augmented_grids):
        if augmented_grids[offset] in [SpecialToken.START_INPUT.value, SpecialToken.START_OUTPUT.value]:
            reversed_task.append(augmented_grids[offset])
            offset += 1
        else:
            assert False

        if offset + 2 > len(augmented_grids):
            assert False
        
        height, width = augmented_grids[offset:offset+2]
        grid_size = height * width
        grid = augmented_grids[offset:offset+grid_size+2]
        
        # Reverse the augmentations in opposite order
        # 1. Reverse permutation
        grid = permute(grid, reverse_mapping)
        
        # 2. Reverse flip
        if flip:
            grid = flip_grid_vertical(grid)  # flip is its own inverse
        
        # 3. Reverse rotation
        reverse_rotation = (4 - rotation) % 4  # Calculate inverse rotation
        grid = rotate_grid(grid, reverse_rotation)
        
        reversed_task.extend(grid)
        offset += grid_size + 2

    return reversed_task
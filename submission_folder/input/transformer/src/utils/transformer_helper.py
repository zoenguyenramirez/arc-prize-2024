import numpy as np
import random

import torch
import os

from typing import List

from src.token import SpecialToken

def dump_input(input_ids: torch.Tensor, indices, prefix="", dump_dir='./temp'):
    """
    Dump input tensors to files for debugging or analysis.
    
    Args:
    input_ids (torch.Tensor): The input tensor containing token IDs.
    prefix (str): A prefix for the filename.
    dump_dir (str): Directory to save the dumps.
    """
    
    # Save input_ids
    torch.save({
            'input_ids': input_ids,
            'indices': indices
        }, os.path.join(dump_dir, f"{prefix}_input_ids.pt"))

def format_batch(batch, max_print_length=150):
    def token_to_str(token):
        if token < SpecialToken.CELL_TOKEN_SIZE.value:
            return str(token)
        return SpecialToken(token).name

    formatted_sequences = []
    for sequence in batch:
        tokens = [token_to_str(t.item()) for t in sequence[:max_print_length]]
        if len(sequence) > max_print_length:
            tokens.append('...')
        formatted_sequences.append(' '.join(tokens))
    
    return '\n\n'.join(formatted_sequences)

def create_mask(input_ids, device, end_of_example_indices:List[int], mask_hack=False):
    assert input_ids.dim() == 3
    batch_size, seq_length, _ = input_ids.shape
    assert batch_size == len(end_of_example_indices)

    mask = torch.triu(torch.ones((seq_length, seq_length), device=device), diagonal=1).bool()

    # mask = mask.unsqueeze(0).expand(batch_size, -1, -1) # NOT PROVEN YET: was a bug, causing unpredictable logits outputs!
    mask = mask.unsqueeze(0).repeat(batch_size, 1, 1)  # it will later be expand again by number of heads in the transformer, this is also required for DataParallel

    if mask_hack:
        for batch_index, end_index in enumerate(end_of_example_indices):
            assert end_index > 0
            # assert end_index <= seq_length
            # print('batch_index, end_index', batch_index, end_index)
            mask[batch_index, :, :end_index] = 0

    return mask

def prefix_mask(target, seq_length, end_of_examples:List[int]):
    prefix_length = torch.tensor(end_of_examples, device = target.device)    
    mask = torch.arange(seq_length, device=target.device).unsqueeze(0) < prefix_length.unsqueeze(1) - 1
    target[mask] = SpecialToken.PAD.value
    
    return target

def mask_expansion(mask, num_heads):
    assert mask.dim() == 3
    batch_size, sequence_length, _ = mask.shape
    mask = mask.unsqueeze(1)  # [batch_size, 1, sequence_length, sequence_length]
    mask = mask.expand(-1, num_heads, -1, -1)  # [batch_size, heads, sequence_length, sequence_length]
    mask = mask.reshape(num_heads * batch_size, sequence_length, sequence_length)  # e.g. Final shape: [2800, 203, 203]

    assert mask.dtype is torch.bool, f"mask.dtype is not bool, {mask.dtype}"
    return mask

def combine_encoding(x: torch.Tensor, batch_size: int, use_grid_encoder: bool, seq_length: int, max_grid_size: int, grid_scale: torch.Tensor, grid_encoding: torch.Tensor, positional_encoding: torch.Tensor):
    # The optimized grid encoding
    if use_grid_encoder:
        # -1 will access the last element so, it is OK
        # Assert that all values are within the expected range
        assert torch.all((x[:, :, 1:] >= -1) & (x[:, :, 1:] < max_grid_size)), "Indices out of range"

        row_encodings = grid_encoding[:, x[:, :, 1], :]
        col_encodings = grid_encoding[:, x[:, :, 2], :]
        top_encodings = grid_encoding[:, x[:, :, 3], :]
        sample_encoding = grid_encoding[:, x[:, :, 4], :]
        grid_encodings = torch.cat([
            row_encodings * grid_scale[0],
            col_encodings * grid_scale[1],
            top_encodings * grid_scale[2],
            sample_encoding
        ], dim=-1)  # (N, seq_length, embed_size // 2)

        return grid_encodings.squeeze(0)

    return positional_encoding[:, :seq_length, :]

def count_parameters(model):
    print("Layer_name\t\tNumber of Parameters")
    print("="*50)
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        print(f"{name}\t\t{param}")
        total_params += param
    print("="*50)
    print(f"Total Trainable Params: {total_params/1024/1024}M")
    return total_params

def dump_model_operation(outputs, input_ids, mask, file_name):
    torch.save({'outputs': outputs, 'input_ids': input_ids, 'mask': mask}, f"{file_name}.pt")
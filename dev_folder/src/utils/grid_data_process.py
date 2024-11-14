from typing import Dict, List, Tuple, Any, Set, List, TypedDict
import random
import numpy as np
import torch
import logging

from src.token import SpecialToken
from src.utils.topology_helper import color_charts

"""
A few terms we used in the related code

Where                     Name                Representation
-----------------------   ------------------  ----------------------------------------------------
in json (*.json):         json list           List[List[int]]
in dataset file (*.pt):   compact_grids       np.array(sequence, dtype=np.int8) = compact_grid(s)
in batch:                 Tokenized Tasks     {
                                                'data': torch.stack(padded_batch),
                                                'indices': ...
                                              }
                                              padded_batch are the tokenized tasks here
                                              (another variant of compact grid)
"""

def preprocess_array_data(task_array: List[Any], data):
    index = 0
    while True:
        group_size = random.randint(4, 6)
        if index + group_size > len(task_array):
            break

        group = task_array[index:index + group_size]

        try:
            sequence = []

            for train_example in group:
                sequence.append(SpecialToken.START_INPUT.value)
                sequence.extend(compact_grid(train_example['input']))
                sequence.append(SpecialToken.START_OUTPUT.value)
                sequence.extend(compact_grid(train_example['output']))

            data.append(np.array(sequence, dtype=np.int8))

        except AssertionError as e:
            logging.error(f'Failed to process group at index {index}', exc_info=True)

        index += group_size

def json_list_to_compact_grids(challenge, solution, test_index):
    sequence = []                

    # Add training examples
    for train_example in challenge['train']:
        sequence.append(SpecialToken.START_INPUT.value)
        sequence.extend(compact_grid(train_example['input']))
        sequence.append(SpecialToken.START_OUTPUT.value)
        sequence.extend(compact_grid(train_example['output']))

    sequence.append(SpecialToken.START_INPUT.value)
    # Add test input
    sequence.extend(compact_grid(challenge['test'][test_index]['input']))

    if solution:
        sequence.append(SpecialToken.START_OUTPUT.value)
        sequence.extend(compact_grid(solution[test_index]))

    return sequence


def preprocess_for_fast_access(challenges: Dict[str, Any], solutions: Dict[str, Any], second_only, data) -> None:
    for _, (challenge_id, challenge) in enumerate(challenges.items()):
        for test_index in range(len(challenge['test'])):
            if second_only and test_index == 0:
                continue
            elif not second_only and test_index > 0:
                continue
            
            # create a test task utility function
            sequence = json_list_to_compact_grids(challenge, solutions[challenge_id] if solutions else None, test_index)

            data.append(np.array(sequence, dtype=np.int8))


def extend_and_tokenize_compact_grid(compact_task:list[int], offset: int, working_sequence: list[list[int]], grid_index):
    height = compact_task[offset]
    width = compact_task[offset + 1]

    if height == 0 or width == 0:
        return offset + 2
    segment = compact_task[offset:offset + (height * width + 2)]
    charts = color_charts(segment)

    # print('segment, charts', segment, charts)

    # Add assertion to check the range of values in charts
    assert all(0 <= color < 64 for color in charts), f"Charts values must be non-negative and less than 64, but got {max(charts)}"

    offset += 2
    for y in range(height):
        numbers_to_extend = compact_task[offset:offset + width]
        assert all(0 <= num < 10 for num in numbers_to_extend), "All numbers must be >= 0 and < 10"

        for x in range(width):
            working_sequence.append([numbers_to_extend[x], y, x, charts[y * width + x], grid_index])

        working_sequence.append([SpecialToken.ROW_SEPARATOR.value, y, width, -1, grid_index])

        offset += width

    return offset

@staticmethod
def create_special_token_sequence(token: SpecialToken, grid_index) -> List[int]:
    dummy_coord = -1
    return [token.value, dummy_coord, dummy_coord, dummy_coord, grid_index]


def tokenize_compact_task(compact_task:list[int], *, single_output:bool = False) -> List[List[int]]:
    offset = 0
    grid_index = 0
    if not single_output:
        sequence = [create_special_token_sequence(SpecialToken.START, grid_index)]
    else:
        sequence = []

    while True:
        grid_index += 1
        if offset >= len(compact_task):
            assert False
        if compact_task[offset] == SpecialToken.START_INPUT.value:
            sequence.append(create_special_token_sequence(SpecialToken.START_INPUT, grid_index))
            offset = extend_and_tokenize_compact_grid(compact_task, offset + 1, sequence, grid_index)
            sequence.append(create_special_token_sequence(SpecialToken.END_INPUT, grid_index))
        else:
            assert single_output

        if offset >= len(compact_task):
            break

        assert compact_task[offset] == SpecialToken.START_OUTPUT.value

        grid_index += 1
        sequence.append(create_special_token_sequence(SpecialToken.START_OUTPUT, grid_index))
        offset = extend_and_tokenize_compact_grid(compact_task, offset + 1, sequence, grid_index)
        sequence.append(create_special_token_sequence(SpecialToken.END_OUTPUT, grid_index))

        if offset >= len(compact_task):
            if offset == len(compact_task):
                sequence.append(create_special_token_sequence(SpecialToken.END, grid_index))
            else:
                assert False
            break

    return sequence

def build_pairs(compact_task: list[int]) -> List[List[int]]:
    offset = 0
    pairs = []

    # Identify all input-output pairs
    while offset < len(compact_task):
        assert compact_task[offset] == SpecialToken.START_INPUT.value
        input_start = offset
        offset = find_next_special_token(compact_task, offset + 1)

        if offset < len(compact_task):            
            assert compact_task[offset] == SpecialToken.START_OUTPUT.value
            output_start = offset
            offset = find_next_special_token(compact_task, offset + 1)
        else:
            assert offset == len(compact_task)
        
        pair = compact_task[input_start:offset]
        pairs.append(pair)

    return pairs

def shuffle_all_but_last_pair(compact_task: list[int]) -> list[int]:
    pairs = build_pairs(compact_task)

    # Shuffle all but the last pair
    if len(pairs) > 1:
        sliced = pairs[:-1]
        random.shuffle(sliced)
        sliced.append(pairs[-1])

        # Reconstruct the shuffled compact_task
        shuffled_task = []
        for pair in sliced:
            shuffled_task.extend(pair)
            
        return shuffled_task

    return compact_task

def shuffle_all_pairs(compact_task: list[int]) -> list[int]:
    pairs = build_pairs(compact_task)
    random.shuffle(pairs)

    shuffled_task = []
    for pair in pairs:
        shuffled_task.extend(pair)
        
    return shuffled_task

def find_next_special_token(compact_task: list[int], start_offset: int) -> int:
    for i in range(start_offset + 2, len(compact_task)):
        if compact_task[i] in [SpecialToken.START_INPUT.value, SpecialToken.START_OUTPUT.value]:
            return i
    return len(compact_task)

def compact_grid(raw_grid: List[List[int]]):
    # Calculate width and height
    height = len(raw_grid)
    width = max(len(row) for row in raw_grid)

    assert height <= 30 and width <= 30

    # Flatten the grid
    result = []
    for row in raw_grid:
        result.extend(row)

    # Create a list with width, height, and flattened grid
    full_result = [height, width] + result

    assert len(result) == height * width

    return full_result
    
def find_last_occurrence(lst, value):
    for index, item in reversed(list(enumerate(lst))):
        if item[0] == value:
            return index
    return -1  # Return -1 if the value is not found

def end_of_examples_mark(task):
    last_index = find_last_occurrence(task, SpecialToken.START_OUTPUT.value)
    return last_index

def to_json_list(tokenized_grid):
    row_list = []
    column_list = []
    for elem in tokenized_grid:
        token = elem[0] if isinstance(elem, list) else elem

        if token < SpecialToken.CELL_TOKEN_SIZE.value:
            row_list.append(token)
        elif len(row_list) > 0:
            column_list.append(row_list)
            row_list = []

    if len(row_list) > 0:
        column_list.append(row_list)
        row_list = []

    return column_list

def detokenize_to_compact_grid(current_data: List[int]): 
    number_of_rows = 0
    compact_grid = []
    for index, element in enumerate(current_data):
        if element == SpecialToken.ROW_SEPARATOR.value:
            number_of_rows += 1
            continue
        assert element < 10, f'{element}, {index}'
        compact_grid.append(element)

    if number_of_rows:
        number_of_col = len(compact_grid) // number_of_rows
    elif len(compact_grid) > 0:
        number_of_col = len(compact_grid)
        number_of_rows = 1
    else:
        number_of_col = 0
    
    assert len(compact_grid) >= number_of_rows * number_of_col
    return [number_of_rows, number_of_col] + compact_grid[:number_of_rows * number_of_col]


def detokenize_to_compact_grids(tokenized_sequence: List[List[int]]|List[int]) -> List[int]:
    current_data = []
    output_stream = []
    can_receive_cell = False
    
    for index, elem in enumerate(tokenized_sequence):
        token = elem[0] if isinstance(elem, list) else elem
        if token == SpecialToken.START_INPUT.value:
            current_data = []
            can_receive_cell = True
        elif token == SpecialToken.END_INPUT.value and can_receive_cell:
            output_stream.append(SpecialToken.START_INPUT.value)
            output_stream.extend(detokenize_to_compact_grid(current_data))
            can_receive_cell = False
        elif token == SpecialToken.START_OUTPUT.value:
            current_data = []      
            can_receive_cell = True      
        elif token == SpecialToken.END_OUTPUT.value and can_receive_cell:
            output_stream.append(SpecialToken.START_OUTPUT.value)
            output_stream.extend(detokenize_to_compact_grid(current_data))
            can_receive_cell = False
        elif can_receive_cell:
            if token < 10 or token == SpecialToken.ROW_SEPARATOR.value:
                current_data.append(token)
        else:
            if token < 10 or token == SpecialToken.ROW_SEPARATOR.value:
                assert True # this does happen :(
    
    return output_stream

def count_grids_of_tokened_task(tokend_task: List[int]) -> int:
    number_of_start_output = tokend_task.count(SpecialToken.START_OUTPUT.value)
    return number_of_start_output

def count_grids_of_compact_grids(compact_task: List[int]) -> int:
    offset = 0
    grid_index = 0
    sequence_placeholder = []

    while offset < len(compact_task):
        grid_index += 1
        assert compact_task[offset] == SpecialToken.START_INPUT.value
        offset = extend_and_tokenize_compact_grid(compact_task, offset + 1, sequence_placeholder, grid_index)

        if offset >= len(compact_task):
            break

        assert compact_task[offset] == SpecialToken.START_OUTPUT.value
        grid_index += 1
        offset = extend_and_tokenize_compact_grid(compact_task, offset + 1, sequence_placeholder, grid_index)

    return (grid_index + 1) // 2

def estimate_output_grid_token_length(compact_task: List[int]) -> int:
    offset = 0
    grid_index = 0
    sequence_placeholder = []

    max_size = last_input_size = 0

    while offset < len(compact_task):
        grid_index += 1
        assert compact_task[offset] == SpecialToken.START_INPUT.value
        last_input_size = compact_task[offset + 1] * compact_task[offset + 2]
        offset = extend_and_tokenize_compact_grid(compact_task, offset + 1, sequence_placeholder, grid_index)

        if offset >= len(compact_task):
            break

        assert compact_task[offset] == SpecialToken.START_OUTPUT.value
        grid_index += 1
        max_size = max(max_size, compact_task[offset + 1] * compact_task[offset + 2])
        offset = extend_and_tokenize_compact_grid(compact_task, offset + 1, sequence_placeholder, grid_index)

    return max(max_size, last_input_size)

def remove_last_grid(compact_task: List[int]) -> List[int]:
    offset = 0
    grid_index = 0
    last_start_index = -1
    sequence_placeholder = []

    while offset < len(compact_task):
        grid_index += 1
        last_start_index = offset
        assert compact_task[offset] == SpecialToken.START_INPUT.value
        offset = extend_and_tokenize_compact_grid(compact_task, offset + 1, sequence_placeholder, grid_index)

        if offset >= len(compact_task):
            break

        assert compact_task[offset] == SpecialToken.START_OUTPUT.value
        grid_index += 1
        offset = extend_and_tokenize_compact_grid(compact_task, offset + 1, sequence_placeholder, grid_index)

    return compact_task[:last_start_index]

class PaddedSequence(TypedDict):
    data: torch.Tensor
    seq_length: List[int]
    end_of_examples: List[int]
    indices: List[int]

def pad_collate(batch, max_length) -> PaddedSequence:
    """
    Pad and collate a batch of sequences.

    Parameters
    ----------
    batch : list
        Each element is expected to be in the format (seq, index, end_of_encoder_mark).

    """
    # Find maximum length
    seq_lengths = [len(seq['task']) for seq in batch]
    max_len = max(seq_lengths)
    max_len = min(max_len, max_length)
    assert max_len > 8, f"{max_len}"
    assert not isinstance(batch, torch.Tensor), "Batch should not be a PyTorch tensor"

    element_size = len(batch[0]['task'][0])

    assert element_size == 5
    
    # Pad sequences dynamically
    padded_batch = []
    for sample in batch:
        seq = torch.tensor(sample['task'])
        assert seq .device.type == 'cpu'
        pad_len = max(0, max_len - seq.shape[0])
        if pad_len > 0:
            padding = torch.zeros((pad_len, element_size), dtype=seq.dtype, device=seq.device)
            padding[:, 0] = SpecialToken.PAD.value
            padding[:, 1:] = -1
            padded_seq = torch.cat([seq[:max_len], padding], dim=0)
        else:
            padded_seq = seq[:max_len]

        # chart id is -1 after the examples
        if sample['end_of_examples_index'] >= 0:
            padded_seq[sample['end_of_examples_index']:, -2:] = -1  # fill the last 2 elements -1
        
        padded_batch.append(padded_seq)

    data = torch.stack(padded_batch) # Shape: (batch_size, max_len, 3)
    end_of_examples = [elem['end_of_examples_index'] for elem in batch]
    indices = [elem['idx'] for elem in batch]

    # assert all(index < data.shape[1] for index in end_of_examples), f"{end_of_examples}, {data.shape}"

    return {
            'data': data,
            'end_of_examples': end_of_examples,
            'indices': indices,
            'seq_length': seq_lengths
        }

def to_ascii_board(predicted, target):
    # print('target', target[:-3])
    target_grid = detokenize_to_compact_grid(target[:-3])
    # print('predicted', predicted[:-3])
    predicted_grid = detokenize_to_compact_grid(predicted[:-3])

    # Print the two boards side by side on terminal
    pred_height, pred_width = predicted_grid[0], predicted_grid[1]
    target_height, target_width = target_grid[0], target_grid[1]

    # Get the grid values (skip the height and width)
    pred_values = predicted_grid[2:]
    target_values = target_grid[2:]

    # Print header
    print("Predicted" + " " * (pred_width * 2 + 2) + "Target")
    print("-" * (pred_width * 2 + 2) + "  " + "-" * (target_width * 2))

    # Print rows side by side
    max_height = max(pred_height, target_height)
    for y in range(max_height):
        # Print predicted grid row
        if y < pred_height:
            for x in range(pred_width):
                val = pred_values[y * pred_width + x]
                print(f"{val:1d} ", end="")
        else:
            print(" " * (pred_width * 2), end="")
        
        print("  ", end="")  # Spacing between grids

        # Print target grid row
        if y < target_height:
            for x in range(target_width):
                val = target_values[y * target_width + x]
                print(f"{val:1d} ", end="")
        print()  # New line after each row

    print()  # Extra line for spacing
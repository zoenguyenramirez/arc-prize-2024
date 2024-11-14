import argparse
import re

import json
import numpy as np
import hashlib
import argparse

MAX_INPUT_VALUE = 10

def generate_tasks(mapping_function, num_tasks, num_train_examples, global_seed_offset):
    function_name = mapping_function.__name__
    challenges = {}
    solutions = {}
    
    for task_index in range(num_tasks):
        task_id = f"{function_name}_{task_index + 1}"
        
        # Generate train data
        train_data = []
        for i in range(num_train_examples):
            input_value = generate_input(i + global_seed_offset, task_id)
            output_value = mapping_function(input_value)
            
            assert max(max(row) for row in input_value) < MAX_INPUT_VALUE
            assert max(max(row) for row in output_value) < MAX_INPUT_VALUE, f'{task_id} has >= {MAX_INPUT_VALUE} cell(s)'
            
            train_data.append({
                "input": input_value,
                "output": output_value
            })
        
        # Generate test data
        test_input = generate_input(num_train_examples, task_id)
        test_output = mapping_function(test_input)
        
        challenges[task_id] = {
            "train": train_data,
            "test": [{"input": test_input}]
        }
        
        solutions[task_id] = [test_output]
    
    return challenges, solutions

def generate_input(seed, task_id):
    # Hash the task_id
    hash_object = hashlib.md5(task_id.encode())
    task_hash = int(hash_object.hexdigest(), 16)
    
    # Combine the original seed with the task hash and take modulo 2**32
    combined_seed = (seed + task_hash) % (2**32)
    
    np.random.seed(combined_seed)
    rows = np.random.randint(1, 5)
    cols = np.random.randint(1, 5)
    return np.random.randint(0, MAX_INPUT_VALUE, size=(rows, cols)).tolist()

# Mapping functions

def conditional_logic(input_array):
    return [[1 if x % 2 == 0 else 0 for x in row] for row in input_array]

def array_indexing(input_array):
    if len(input_array) > 1 and len(input_array[0]) > 1:
        return [[input_array[-1][-1]]]
    return [[0]]

def arithmetic_operations(input_array):
    total_sum = sum(sum(row) for row in input_array)
    return [[total_sum % 10]]

def modular_arithmetic(input_array):
    return [[x % 3 for x in row] for row in input_array]

def find_min_max(input_array):
    flat = [x for row in input_array for x in row]
    return [[min(flat), max(flat)]]

def count_occurrences(input_array):
    flat = [x for row in input_array for x in row]
    return [[flat.count(i) for i in range(10)]]

def element_wise_operations(input_array):
    return [[(x * 2) % 10 for x in row] for row in input_array]

def array_manipulation(input_array):
    return np.array(input_array).T.tolist()  # Transpose the array

# List of all mapping functions
mapping_functions = [
    conditional_logic,
    array_indexing,
    arithmetic_operations,
    modular_arithmetic,
    find_min_max,
    count_occurrences,
    element_wise_operations,
    array_manipulation
]

def parse_selected_functions(selected):
    """Parse the selected functions string."""
    selected_funcs = []
    for func in mapping_functions:
        if re.match(selected, func.__name__):
            selected_funcs.append(func)
    return selected_funcs

def main(args):
    # Generate output filenames
    challenges_file = f"input_data/synth_{args.output_prefix}_challenges.json"
    solutions_file = f"input_data/synth_{args.output_prefix}_solutions.json"

    # Parse selected functions
    selected_functions = parse_selected_functions(args.selected_functions)

    # Generate tasks
    challenges = {}
    solutions = {}
    
    for func in selected_functions:
        func_challenges, func_solutions = generate_tasks(func, args.num_tasks, args.num_train_examples, args.global_seed_offset)
        challenges.update(func_challenges)
        solutions.update(func_solutions)
    
    # Save to JSON files
    with open(challenges_file, 'w') as f:
        json.dump(challenges, f, indent=2)
    
    with open(solutions_file, 'w') as f:
        json.dump(solutions, f, indent=2)
    
    print(f"Synthetic dataset with {len(challenges)} tasks generated successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate synthetic dataset for pattern matching tasks.')
    parser.add_argument('--num-tasks', type=int, default=300, help='Number of tasks per mapping function')
    parser.add_argument('--num-train-examples', type=int, default=5, help='Number of training examples per task')
    parser.add_argument('--output-prefix', type=str, default="pattern", help='Prefix for output files')
    parser.add_argument('--selected-functions', type=str, default=".*", help='Regular expression to select functions (default: all)')
    parser.add_argument('--global-seed-offset', type=int, default=0, help='Global random seed (default: None)')
    
    
    args = parser.parse_args()
    main(args)

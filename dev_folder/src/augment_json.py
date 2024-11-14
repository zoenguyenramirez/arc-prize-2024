import argparse
import json
import copy
import itertools
import random
import sys
import time
from src.load_data import load_from_json

def rotate_grid(grid, rotation):
    """
    Rotate the grid clockwise by the specified rotation.
    rotation: 0, 1, 2, or 3 (corresponding to 0, 90, 180, and 270 degrees)
    """
    if rotation == 0:
        return grid
    elif rotation == 1:
        return list(map(list, zip(*grid[::-1])))
    elif rotation == 2:
        return [row[::-1] for row in grid[::-1]]
    elif rotation == 3:
        return list(map(list, zip(*grid)))[::-1]
    else:
        raise ValueError("Rotation must be 0, 1, 2, or 3")

def flip_grid_vertical(grid):
    """Flip the grid vertically"""
    return grid[::-1]

def augment_task_rot(task_id, challenge, solution):
    augmented_challenges = {}
    augmented_solutions = {}
    for rotation in range(4):  # 90, 180, 270 degrees
        new_task_id = f"{task_id}_rot{rotation*90}"

        if rotation:
            # rotated
            new_challenge = augmented_challenges[new_task_id] = copy.deepcopy(challenge)
            new_solution = augmented_solutions[new_task_id] = copy.deepcopy(solution)

            for i in range(len(new_challenge['train'])):
                new_challenge['train'][i]['input'] = rotate_grid(new_challenge['train'][i]['input'], rotation)
                new_challenge['train'][i]['output'] = rotate_grid(new_challenge['train'][i]['output'], rotation)

            for i in range(len(new_challenge['test'])):
                new_challenge['test'][i]['input'] = rotate_grid(new_challenge['test'][i]['input'], rotation)

            new_solution[0] = rotate_grid(new_solution[0], rotation)

        # flipped
        flipped_new_task_id = f"{task_id}_rot{rotation*90}_flipped"
        new_challenge = augmented_challenges[flipped_new_task_id] = copy.deepcopy(new_challenge if rotation else challenge)
        new_solution = augmented_solutions[flipped_new_task_id] = copy.deepcopy(new_solution if rotation else solution)

        for i in range(len(new_challenge['train'])):
            new_challenge['train'][i]['input'] = flip_grid_vertical(new_challenge['train'][i]['input'])
            new_challenge['train'][i]['output'] = flip_grid_vertical(new_challenge['train'][i]['output'])

        for i in range(len(new_challenge['test'])):
            new_challenge['test'][i]['input'] = flip_grid_vertical(new_challenge['test'][i]['input'])

        new_solution[0] = flip_grid_vertical(new_solution[0])

    return augmented_challenges, augmented_solutions

def count(matrix, frequency):
    for row in matrix:
        for num in row:
            frequency[num] += 1

def transform(matrix, mapping):
    for row in matrix:
        for i, num in enumerate(row):
            row[i] = mapping[num] if num > 0 else 0
    return matrix

def data_augment_preprocess(challenge, task_solution):
    # Initialize frequency dictionary
    frequency = {i: 0 for i in range(10)}  # 0 to 9
    
    for input_matrix in challenge['train']:
        count(input_matrix['input'], frequency)
        count(input_matrix['output'], frequency)
    
    count(challenge['test'][0]['input'], frequency)
    count(task_solution[0], frequency)
    
    # Filter out keys with count 0 and return only the keys
    non_zero_keys = [key for key, count in frequency.items() if count > 0]
    
    return set(non_zero_keys)

def generate_deterministic_permutations(input_set, N, seed=43):
    # Set the seed for reproducibility
    random.seed(seed)
    
    if 0 in input_set:
        input_set.remove(0)
    
    # Generate all possible permutations
    all_permutations = list(itertools.permutations(range(1, 10), len(input_set)))
    
    # Shuffle the permutations deterministically
    random.shuffle(all_permutations)
    
    # Select the first N unique permutations (or all if N is greater)
    unique_permutations = all_permutations[:min(N, len(all_permutations))]
    
    return unique_permutations  


def data_augment_process(challenge, task_solution, keys, perm):
    mapping = dict(zip(keys, perm))
    
    challenge = copy.deepcopy(challenge)
    task_solution = copy.deepcopy(task_solution)
    
#     print('deep copied')

#     print(task['train'], type(task['train']))
    for i, input_matrix  in enumerate(challenge['train']):
        # print('to transform train input ', i)
        challenge['train'][i]['input'] = transform(input_matrix['input'], mapping)
        # print('to transform train input ', i)
        challenge['train'][i]['output'] = transform(input_matrix['output'], mapping)
    
    # print('to transform test input ')
    challenge['test'][0]['input'] = transform(challenge['test'][0]['input'], mapping)
    # print('to transform solution ')
    task_solution = transform(task_solution, mapping)
    
    return challenge, task_solution

def augment_task_perm(task_id, challenge, solution, perm_count):
    augmented_challenges = {}
    augmented_solutions = {}

    keys = data_augment_preprocess(challenge, solution)
    # print('keys', keys)
    
    perms = generate_deterministic_permutations(keys, perm_count, task_id)
    
    for perm_index, perm in enumerate(perms):
        new_challenge, new_task_solution = data_augment_process(challenge, solution[0], keys, perm)
        new_task_id = f"{task_id}_p{perm_index}"
        # continue the random seed from the permutation
        random.shuffle(new_challenge['train'])
        augmented_challenges[new_task_id] = new_challenge 
        augmented_solutions[new_task_id] = [new_task_solution] 

    return augmented_challenges, augmented_solutions

def augment_data(challenges, solutions, perm_count = 40, show_progress=False):
    """Augment the data by rotating and flipping the grids"""
    rot_augmented_challenges = copy.deepcopy(challenges)
    rot_augmented_solutions = copy.deepcopy(solutions)

    total_tasks = len(challenges)
    if show_progress:
        print("Rotation augmentation progress:")
    for i, (task_id, task) in enumerate(challenges.items(), 1):
        results = augment_task_rot(task_id, task, solutions[task_id])
        rot_augmented_challenges.update(results[0])
        rot_augmented_solutions.update(results[1])

        if show_progress:
            progress = (i / total_tasks) * 100
            sys.stdout.write(f"\r[{'=' * int(progress / 2)}{' ' * (50 - int(progress / 2))}] {progress:.1f}% {len(rot_augmented_challenges)}")
            sys.stdout.flush()
    
    if show_progress:
        print("\n\nPermutation augmentation progress:")
    perm_augmented_challenges = copy.deepcopy(rot_augmented_challenges)
    perm_augmented_solutions = copy.deepcopy(rot_augmented_solutions)

    total_tasks = len(rot_augmented_challenges)
    for i, (task_id, task) in enumerate(rot_augmented_challenges.items(), 1):
        results = augment_task_perm(task_id, task, rot_augmented_solutions[task_id], perm_count)
        perm_augmented_challenges.update(results[0])
        perm_augmented_solutions.update(results[1])
        
        if show_progress:
            progress = (i / total_tasks) * 100
            sys.stdout.write(f"\r[{'=' * int(progress / 2)}{' ' * (50 - int(progress / 2))}] {progress:.1f}% {len(perm_augmented_challenges)}")
            sys.stdout.flush()

    if show_progress:
        print("\n")  # Add a newline after the progress bar
    return perm_augmented_challenges, perm_augmented_solutions

def main(source, base_path):
    challenges, solutions = load_from_json(source, base_path)

    start_time = time.perf_counter()
    # Augment the data
    augmented_challenges, augmented_solutions = augment_data(challenges, solutions, show_progress=True)

    elapsed_time = time.perf_counter() - start_time
    minutes, seconds = divmod(int(elapsed_time), 60)

    print(f"\nTotal generation time: {minutes:02d}:{seconds:02d}\nSaving....")

    with open(base_path + 'augmented_' + source + '_challenges.json', 'w') as f:
        json.dump(augmented_challenges, f, indent=2)

    with open(base_path + 'augmented_' + source + '_solutions.json', 'w') as f:
        json.dump(augmented_solutions, f, indent=2)

    print(f"Augmented data saved to {base_path}augmented_{source}_challenges.json and {base_path}augmented_{source}_solutions.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment ARC Prize 2024 data by rotating and flipping grids.")
    parser.add_argument("--source", type=str, default="arc-agi_training", help="Source of the data (default: arc-agi_training)")
    parser.add_argument("--base-path", type=str, default="./input_data/", help="Base path for input and output files (default: ./input_data/)")
    parser.add_argument("--skip-retirement", action="store_true", help="Skip the retirement message and proceed with the old functionality")
    
    args = parser.parse_args()
    
    if not args.skip_retirement:
        print("This tool is retired. We augment on the fly now.")
        sys.exit(0)
    
    main(args.source, args.base_path)

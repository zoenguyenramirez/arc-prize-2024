# src/prepare_data.py
import argparse
import torch
import os
import json
import jsonlines
from tqdm import tqdm

from typing import Dict, List, Tuple
from src.load_data import load_from_json, GridDataset
from src.utils.helper import gc_collect
from src.utils.grid_data_process import preprocess_array_data

def prepare_dataset(data_sources: List[str], input_dir: str, second_only: bool) -> Tuple[GridDataset, Dict[str, Tuple[int, int]]]:
    all_challenges: Dict[str, Dict] = {}
    all_solutions: Dict[str, Dict] = {}
    source_ranges: Dict[str, Tuple[int, int]] = {}
    current_index = 0

    gc_collect(report_memory = True)

    for source in data_sources:
        try:
            challenges, solutions = load_from_json(source, input_dir)
            start_index: int = current_index
            all_challenges.update(challenges)
            all_solutions.update(solutions)
            end_index: int = current_index + len(challenges) - 1
            source_ranges[source] = (start_index, end_index)
            current_index = end_index + 1
            print(f"Initial index range for {source}: {start_index} to {end_index}")
            gc_collect(report_memory = True)
        except FileNotFoundError as e:
            print(f"Error loading {source}: {e}. Skipping this data source.")

    dataset = GridDataset.load_from_paired_file(all_challenges, all_solutions, source_ranges=source_ranges, second_only=second_only)
    
    # Print updated source ranges
    for source, (start, end) in dataset.source_ranges.items():
        print(f"Final index range for {source}: {start} to {end}")
    
    return dataset, dataset.source_ranges

def prepare_dataset_from_jsonl(jsonl_file: str) -> Tuple[GridDataset, Dict[str, Tuple[int, int]]]:
    source_ranges: Dict[str, Tuple[int, int]] = {}
    dataset = GridDataset()
    
    print(f"Loading data from JSONL file: {jsonl_file}")
    
    try:
        with jsonlines.open(jsonl_file) as reader:
            for sample in tqdm(reader, desc="Processing samples"):
                original_tasks = sample['examples']
                tasks = [{'input': s[0], 'output': s[1]} for s in original_tasks if len(s[0]) < 30 and len(s[1]) < 30 and len(s[0][0]) < 30 and len(s[1][0]) < 30]
                preprocess_array_data(tasks, dataset.data)

            gc_collect(report_memory=True)
    
    except Exception as e:
        print(f"Error loading JSONL file: {e}")
        raise
    
    print('Finished loading JSONL file')
    
    source_ranges[jsonl_file] = (0, len(dataset.data) - 1)
    dataset.set_source_ranges(source_ranges)
    return dataset, dataset.source_ranges

def prepare_dataset_from_array(data_array_folder: str) -> Tuple[GridDataset, Dict[str, Tuple[int, int]]]:
    source_ranges: Dict[str, Tuple[int, int]] = {}
    file_index = 0
    dataset = GridDataset()

    for filename in os.listdir(data_array_folder):
        if filename.endswith('.json') and not filename.startswith('metadata'):
            file_path = os.path.join(data_array_folder, filename)

            with open(file_path, 'r') as f:
                file_tasks = json.load(f)

            assert isinstance(file_tasks, list), f"file_tasks is not a list for {filename}"

            start_index = len(dataset)
            preprocess_array_data(file_tasks, dataset.data)
            end_index = len(dataset) - 1

            source_ranges[filename] = (start_index, end_index)
            file_index = file_index + 1
            
            print(f"Processed file: {filename}, {file_index}, sample index: {start_index} to {end_index}")
            gc_collect(report_memory=True)

    print('Finished loading all JSON files')
    
    dataset.set_source_ranges(source_ranges)
    return dataset, dataset.source_ranges

def save_dataset(dataset: GridDataset, output_file: str, data_sources: List[str], source_ranges: Dict[str, Tuple[int, int]]) -> None:
    data_to_save: Dict[str, object] = {
        'dataset': dataset,
        'data_sources': data_sources,
        'source_ranges': source_ranges
    }
    torch.save(data_to_save, output_file)

def load_dataset(input_file: str) -> Tuple[GridDataset, List[str], Dict[str, Tuple[int, int]]]:
    loaded_data: Dict[str, object] = torch.load(input_file)
    return loaded_data['dataset'], loaded_data['data_sources'], loaded_data['source_ranges']

def load_datasets(input_files: list[str]) -> Tuple[GridDataset, List[str], Dict[str, Tuple[int, int]]]:
    """
    Load and combine multiple datasets from a list of input files.
    
    Args:
        input_files: List of paths to dataset files
        
    Returns:
        Tuple containing:
        - Combined GridDataset
        - List of all data sources
        - Dictionary of source ranges for the combined dataset
    """
    combined_dataset = GridDataset()
    all_data_sources = []
    combined_source_ranges = {}
    current_index = 0

    for input_file in input_files:
        # Load individual dataset
        dataset, data_sources, source_ranges = load_dataset(input_file)
        
        # Update the starting index for all ranges in this dataset
        updated_ranges = {}
        for source, (start, end) in source_ranges.items():
            length = end - start + 1
            updated_ranges[source] = (current_index, current_index + length - 1)
            current_index += length

        # Extend the combined dataset with the current dataset's data
        combined_dataset.data.extend(dataset.data)
        
        # Update the tracking variables
        all_data_sources.extend(data_sources)
        combined_source_ranges.update(updated_ranges)

    # Set the source ranges for the combined dataset
    combined_dataset.set_source_ranges(combined_source_ranges)

    return combined_dataset, all_data_sources, combined_source_ranges

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for AGI training.")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--data-sources', nargs='+',
                      help='List of data sources to include in the dataset.')
    group.add_argument('--data-array', type=str,
                      help='String representation of data sources array.')
    group.add_argument('--jsonl-file', type=str,
                      help='Path to JSONL file containing the dataset.')
    
    parser.add_argument('--input-dir', type=str, default='./input_data/',
                      help='Directory containing input data files.')
    
    parser.add_argument('--output-file', type=str, default='./intermediate_data/prepared_dataset.pth',
                      help='Path where the prepared dataset will be saved.')
    
    parser.add_argument('--second-only', action='store_true', default=False,
                      help='Use second test only')
    
    args = parser.parse_args()
    
    print(f"Loading and preparing")
    if args.data_sources:
        dataset, source_ranges = prepare_dataset(args.data_sources, args.input_dir, args.second_only)
    elif args.data_array:
        dataset, source_ranges = prepare_dataset_from_array(args.data_array)
    else:
        dataset, source_ranges = prepare_dataset_from_jsonl(args.jsonl_file)

    print(f"Dataset length: {len(dataset)}")
    
    if args.data_sources:
        data_source = args.data_sources
    elif args.data_array:
        data_source = [args.data_array]
    elif args.jsonl_file:
        data_source = [args.jsonl_file]
    else:
        raise ValueError("No valid data source provided")

    save_dataset(dataset, args.output_file, data_source, source_ranges)
    
    print(f"Dataset saved to {args.output_file}")

if __name__ == "__main__":
    main()

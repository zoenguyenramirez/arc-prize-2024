import argparse
import itertools
import logging
import subprocess
import time
from typing import Dict, List, Callable
from datetime import datetime
from src.utils.helper import generate_run_name, HyperParameters
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def run_training(params: HyperParameters, runs_name, index, total_combinations, load_checkpoint=None) -> None:
    command = [
        "python",
        "-m",
        "src.train",
        "--epochs", "180",
        "--max-seq-length", "2048",
        "--samples-per-epoch", "128",
        "--runs-name", runs_name,
        "--seed", "53",
        "--sampling-strategy", "gaussian",
        "--dataset-file", "./intermediate_data/prepared_dataset_re_arc_100_5000.pth",
        "--log-hyperparameters"
    ]

    command.extend(params.to_command_args())
    if load_checkpoint:
        command.extend(["--load-checkpoint", load_checkpoint])

    command.append("--check-existing-hyperparameters")

    try:
        # Log start
        logging.info(f"Starting training run with {command}. **{index}/{total_combinations}**")
        start_time = time.time()

        # Run the command
        result = subprocess.run(command)

        if result.returncode == 0:
            logging.info("the hyperparameter job hasn't run before, start it now...")
            command.pop()
            result = subprocess.run(command)

        # Log end
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"Training run completed in {duration:.2f} seconds. Result: {result.returncode}")
    except subprocess.CalledProcessError as e:
        logging.error(f"An error occurred while running the command: {e}")
        logging.error(f"Command output:\n{e.output}")
        if e.stderr:
            logging.error(f"Command stderr:\n{e.stderr}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")

def generate_hyperparameter_combinations(param_ranges: Dict[str, List], filter_func: Callable[[Dict], bool]) -> List[HyperParameters]:
    keys, values = zip(*param_ranges.items())
    combinations = [
        dict(zip(keys, v)) 
        for v in itertools.product(*values) 
        if filter_func(dict(zip(keys, v)))
    ]
    return [HyperParameters(**combo) for combo in combinations]

def get_param_ranges(scenario: str) -> Dict[str, List]:
    def default_filter(combo: Dict) -> bool:
        if combo['sampling_strategy'] == 'gaussian':
            if combo['sampling_sigma'] == None:
                        return False
        if combo['sampling_strategy'] == 'none':
            if combo['sampling_sigma'] != None:
                        return False    
        return combo['learning_rate'] >= combo['base_lr'] 

    def gradiant_filter(combo: Dict) -> bool:
        if default_filter(combo) == False:
            return False
        gradient_batch_size = combo['batch_size'] * combo['accumulation_steps']
        if not gradient_batch_size in [16]:
            return False
        learning_rate_factor = combo['learning_rate'] / gradient_batch_size
        if learning_rate_factor < 5e-5 / 275:
            # print ('learning_rate_factor too small:', learning_rate_factor, gradient_batch_size)
            return False
        if learning_rate_factor > 2e-3 / 275:
            # print ('learning_rate_factor too big:', learning_rate_factor, gradient_batch_size)
            return False
        
        return True    

    scenarios = {
        "default": {
            'params': {
                'auto_cast': [True, False],
                'warmup_epochs': [10],
                'augment_data': [True],
                'heads': [6],
                'base_lr': [1e-8],
                'num_layers': [7, 10],
                'batch_size': [26],
                'schedular': ['cosine'],
                'step_size_up': [0],
                'embed_size': [780],
                'learning_rate': [4e-4],
                'grid_encoder': [True]
            },
            'filter': default_filter
        },
        "gradiant": {
            'params': {
                'auto_cast': [True],
                'warmup_epochs': [10],
                'augment_data': [True],
                'heads': [6],
                'base_lr': [1e-8],
                'num_layers': [7],
                'learning_rate': [1e-4, 3e-4, 3e-5],
                'batch_size': [1, 2, 4, 8, 16],
                'accumulation_steps': [1, 2, 4, 8, 16, 32, 64],
                'schedular': ['cosine'],
                'step_size_up': [0],
                'embed_size': [780],
                'grid_encoder': [True]
            },
            'filter': gradiant_filter
        },
        "rotational": {
            'params': {
                'auto_cast': [True],
                'warmup_epochs': [0],
                'augment_data': [True],
                'heads': [6],
                'base_lr': [7e-5],
                'num_layers': [7],
                'batch_size': [11],
                'accumulation_steps': [5, 2],
                'learning_rate': [7e-5, 7e-4, 7e-3, 7e-6],
                'schedular': ['cosine'],
                'step_size_up': [0],
                'embed_size': [780],
                'grid_encoder': [True]
            },
            'filter': default_filter
        },
        "new_mask": {
            'params': {
                'auto_cast': [True],
                'warmup_epochs': [10],
                'augment_data': [True],
                'heads': [8],
                'base_lr': [1e-4],
                'num_layers': [7],
                'batch_size': [5],
                'accumulation_steps': [2],
                'learning_rate': [2e-4],
                'schedular': ['cosine'],
                'step_size_up': [0],
                'embed_size': [784],
                'progressive_head': [2],
                'grid_encoder': [True]
            },
            'filter': default_filter
        },
        "optimizer": {
            'params': {
                'auto_cast': [True],
                'warmup_epochs': [4],
                'augment_data': [True],
                'heads': [4],
                'base_lr': [1e-4],
                'num_layers': [7],
                'batch_size': [5],
                'accumulation_steps': [2],
                'learning_rate': [2e-4],
                'schedular': ['cosine'],
                'step_size_up': [0],
                'embed_size': [784],
                'progressive_head': [2],
                'grid_encoder': [True],
                'optimizer': ['sgd', 'adam']
            },
            'filter': default_filter
        },
        "sampling": {
            'params': {
                'auto_cast': [True],
                'warmup_epochs': [4],
                'augment_data': [True],
                'heads': [4],
                'base_lr': [8e-6],
                'num_layers': [18],
                'batch_size': [3],
                'accumulation_steps': [4],
                'learning_rate': [1e-5],
                'schedular': ['cosine'],
                'step_size_up': [0],
                'embed_size': [784],
                'progressive_head': [3],
                'grid_encoder': [True],
                'optimizer': ['adam'],
                'mask_hack': [False],
                'sampling_strategy': ['gaussian', 'none'],
                'sampling_sigma': [None, 4/400., 1/400.]
            },
            'filter': default_filter
        },
        "local": {
            'params': {
                'auto_cast': [True, False],
                'warmup_epochs': [10],
                'augment_data': [True],
                'heads': [2],
                'base_lr': [1e-5],
                'num_layers': [7],
                'batch_size': [512],
                'schedular': ['cosine'],
                'step_size_up': [0],
                'embed_size': [36],
                'learning_rate': [1e-3],
                'grid_encoder': [True]
            },
            'filter': default_filter
        },
    }
    
    if scenario is None:
        return scenarios["default"]
    
    if scenario not in scenarios:
        raise ValueError(f"Invalid scenario: '{scenario}'. Available scenarios are: {', '.join(scenarios.keys())}")
    
    return scenarios[scenario]

def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning with optional timestamp and checkpoint.")
    parser.add_argument("--runs-name", type=str, help="Specify a runs_name to reuse an existing runs folder")
    parser.add_argument("--load-checkpoint", type=str, help="Specify a checkpoint to load")
    parser.add_argument("--scenario", type=str, default="default", help="Specify the hyperparameter scenario to use")
    args = parser.parse_args()

    scenario_data = get_param_ranges(args.scenario)
    param_ranges = scenario_data['params']
    filter_func = scenario_data['filter']
    hyperparameter_combinations = generate_hyperparameter_combinations(param_ranges, filter_func)

    # Use the provided timestamp or generate a new one
    runs_name = args.runs_name if args.runs_name else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    total_combinations = len(hyperparameter_combinations)

    print(f'to run {total_combinations} jobs')

    for i, params in enumerate(hyperparameter_combinations, 1):
        run_training(params, runs_name, i, total_combinations, args.load_checkpoint)

if __name__ == "__main__":
    main()

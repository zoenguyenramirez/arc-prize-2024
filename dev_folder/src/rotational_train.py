import argparse
import logging
import subprocess
import time
import os
from datetime import datetime
from typing import List, Tuple
import glob
from typing import Optional
from src.utils.tar_file_helper import extract_tar_file, release_extracted_tar_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

decay_per_segment = 0.94
decay_per_session = 0.94
decay_after_first_checkpoint = 0.3

def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    pattern = os.path.join(checkpoint_dir, "**", "Transformer_*.pt")
    checkpoints = glob.glob(pattern, recursive=True)
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getctime)

def update_learning_rate(lr: float, session_index:int) -> float:
    if session_index == 1:
        return lr * decay_after_first_checkpoint
    return lr * decay_per_segment

def get_interleave_indices(dataset_index, interleave_ratio):
    assert len(interleave_ratio) == 2

    group_size = sum(interleave_ratio)
    group_index = dataset_index // group_size
    index_in_group = dataset_index % group_size

    if index_in_group < interleave_ratio[0]:
        return (0, group_index * interleave_ratio[0] + index_in_group)
    else:
        return (1, group_index * interleave_ratio[1] + index_in_group - interleave_ratio[0])

def get_dataset_file(dataset_files, dataset_index, interleave_ratio):
    current_dataset_file_context = get_interleave_indices(dataset_index, interleave_ratio)
    dataset_file = dataset_files[current_dataset_file_context[0]]

    return dataset_file[current_dataset_file_context[1] % len(dataset_file)]

def prepare_next_dataset_file(dataset_files, dataset_index):
    '''
    we interleave the dataset files at a certain frequency at a ratio of 7:5
    every 5 + 7 would be a group, so we have a group index.
    '''
    interleave_ratio = (3, 1)
    
    current_dataset_file = get_dataset_file(dataset_files, dataset_index, interleave_ratio)
    next_dataset_file = get_dataset_file(dataset_files, dataset_index + 1, interleave_ratio)

    extracted_file_name = extract_tar_file(current_dataset_file, blocking=True)
    extract_tar_file(next_dataset_file, blocking=False)

    assert extracted_file_name

    return extracted_file_name

def run_training(runs_name: str, session_index: int, initial_lr: float, total_epochs: int, epochs_per_rotation: int, dataset_files: Tuple[List[str], List[str]]) -> None: 
    checkpoint_dir = os.path.join("runs", runs_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    current_epoch = 0
    current_seed = session_index
    current_lr = initial_lr
    dataset_index = session_index

    while current_epoch < total_epochs:
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        
        if not latest_checkpoint:
            assert current_epoch == 0
            
        epochs_to_run = min(epochs_per_rotation, total_epochs - current_epoch)

        # Get the current dataset file
        current_dataset_file = prepare_next_dataset_file(dataset_files, dataset_index)
        assert(current_dataset_file)

        command = [
            "python",
            "-m",
            "src.train",
            "--epochs", str(epochs_to_run),
            "--max-seq-length", "2920",
            "--samples-per-epoch", "800",
            "--runs-name", runs_name,
            "--seed", str(current_seed),
            "--learning-rate", str(current_lr),
            '--warmup-epochs', '10',
            '--heads', '8',
            '--base-lr', str(current_lr * decay_per_session),
            '--num-layers', '9',
            '--batch-size', '24',
            '--schedular', 'cosine',
            '--embed-size', '1024',
            '--accumulation-steps', '1',
            '--progressive-head', '1',
            '--minimize-checkpoints',
            '--mask-hack', 'False',
            '--dataset-file', current_dataset_file
        ]

        if latest_checkpoint:
            command.extend(["--load-checkpoint", latest_checkpoint])

        try:
            logging.info(f"Starting training run with seed {current_seed} and learning rate {current_lr:.6f} latest_checkpoint: {latest_checkpoint}, command: {command}")
            start_time = time.time()

            result = subprocess.run(command, check=True)

            end_time = time.time()
            duration = end_time - start_time
            logging.info(f"Training run completed in {duration:.2f} seconds. Result: {result.returncode}")

        except subprocess.CalledProcessError as e:
            logging.error(f"An error occurred while running the command: {e}")
            logging.error(f"Command output:\n{e.output}")
            if e.stderr:
                logging.error(f"Command stderr:\n{e.stderr}")
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred: {str(e)}")
            break

        current_epoch += epochs_to_run
        current_seed += 1
        session_index += 1
        current_lr = update_learning_rate(current_lr, session_index)
        dataset_index += 1
        release_extracted_tar_file(current_dataset_file)

def main():
    parser = argparse.ArgumentParser(description="Run rotational training with changing seeds and learning rates.")
    parser.add_argument("--runs-name", type=str, help="Specify a runs_name for the training session")
    parser.add_argument("--initial-index", type=int, default=0, help="Initial session index for the first rotation")
    parser.add_argument("--initial-lr", type=float, default=2e-4, help="Initial learning rate")
    parser.add_argument("--total-epochs", type=int, default=50000, help="Total number of epochs to train")
    parser.add_argument("--epochs-per-rotation", type=int, default=100, help="Number of epochs per rotation")
    parser.add_argument("--dataset-files1", nargs='+', required=True, help="First list of dataset files to use for training")        
    parser.add_argument("--dataset-files2", nargs='+', required=True, help="Second list of dataset files to use for training")        
    
    args = parser.parse_args()

    runs_name = args.runs_name if args.runs_name else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    logging.info(f"dataset_files, {args.dataset_files1}, /&/ {args.dataset_files2}")

    run_training(runs_name, args.initial_index, args.initial_lr, args.total_epochs, args.epochs_per_rotation, (args.dataset_files1, args.dataset_files2))

if __name__ == "__main__":
    main()

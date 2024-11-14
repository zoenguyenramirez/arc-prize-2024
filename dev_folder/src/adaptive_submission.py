import torch
import json
import logging
import os
from os import system
import argparse
from collections import defaultdict
import math
from filelock import FileLock
import json
import time
from typing import Dict, List, Tuple, Union, Optional
import gc

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from src.load_data import load_from_json, DynamicGridDataset
from src.checkpoint_handler import CheckpointHandler
from src.utils.grid_data_process import json_list_to_compact_grids, tokenize_compact_task, to_json_list, shuffle_all_but_last_pair, tokenize_compact_task, detokenize_to_compact_grids, count_grids_of_compact_grids, remove_last_grid, PaddedSequence, estimate_output_grid_token_length
from src.meta_adaptive_train import train_single_epoch
from src.sample import generate_sample
from src.token import SpecialToken
from src.utils.helper import set_deterministic
from src.utils.data_augment import augment_compact_grids, reverse_augment_compact_grids
from src.utils.grid_data_process import pad_collate
from src.utils.transformer_helper import create_mask

def update_input_ids(coord_tracking: List[int], next_token_id: int, max_grid_size: int, token_in_focus: torch.Tensor) -> None:
    if next_token_id < SpecialToken.CELL_TOKEN_SIZE.value:
        coord = coord_tracking.copy()
        coord_tracking[1] = coord_tracking[1] + 1
        coord_tracking[1] = min(coord_tracking[1], max_grid_size - 1)
    elif next_token_id == SpecialToken.ROW_SEPARATOR.value:
        coord = coord_tracking.copy()
        coord_tracking[1] = 0
        coord_tracking[0] = coord_tracking[0] + 1
        coord_tracking[0] = min(coord_tracking[0], max_grid_size - 1)
    else:
        coord_tracking[0] = 0
        coord_tracking[1] = 0
        coord = (-1, -1)

    token_in_focus[0:5] = torch.tensor([next_token_id, coord[0], coord[1], -1, -1])

def auto_regressive_generate(
    model: nn.Module,
    input_sequence: PaddedSequence,
    max_length: int,
    device: torch.device,
    estimated_output_grid_token_length: int,
    scaler: Optional[GradScaler]
) -> Tuple[List[List[int]], List[int]]:
    model.eval()
    with torch.no_grad():
        batch_size = input_sequence['data'].shape[0]
        input_ids = input_sequence['data'].to(device)  # (batch_size, seq_length)

        min_seq_length = max(input_sequence['seq_length'])
        max_tokens_to_generate = min(1000, max_length - min_seq_length, int(estimated_output_grid_token_length * 1.3 + 30))

        batch_finished = [False] * batch_size
        end_token_indices = [-1] * batch_size
        coord_tracking = [[0, 0] for _ in range(batch_size)]

        pad_token = torch.full((batch_size, 1, 5), -1, dtype=torch.long, device=device)
        pad_token[:, :, 0] = SpecialToken.PAD.value

        for generated_token_index in range(max_tokens_to_generate):
            mask = create_mask(input_ids, device, input_sequence['seq_length'])

            # Add autocast context
            with autocast(enabled=scaler is not None):
                outputs = model(input_ids, mask)  # (B, seq_length, vocab_size)

            input_ids = torch.cat([input_ids, pad_token], dim=1)

            for index_in_batch in range(batch_size):
                token_index_in_focus = input_sequence['seq_length'][index_in_batch] + generated_token_index
                next_token_logits = outputs[index_in_batch, token_index_in_focus - 1, :]  # (vocab_size)
                next_token_id = torch.argmax(next_token_logits).item()

                print(f'\r{generated_token_index} {next_token_id}@[{index_in_batch}][{token_index_in_focus}] ', end="", flush=True)

                update_input_ids(coord_tracking[index_in_batch], next_token_id, model.max_grid_size, input_ids[index_in_batch][token_index_in_focus])

                if next_token_id == SpecialToken.END.value:
                    batch_finished[index_in_batch] = True
                    end_token_indices[index_in_batch] = min(end_token_indices[index_in_batch], token_index_in_focus) if end_token_indices[index_in_batch] >= 0 else token_index_in_focus

                if all(batch_finished):
                    return input_ids.tolist(), end_token_indices

            torch.cuda.empty_cache()
            
    return input_ids.tolist(), end_token_indices

def aug_generate_sample(
    model: nn.Module,
    max_seq_length: int,
    device: torch.device,
    *,
    batch_size: int,
    vote: Dict[Tuple[int, ...], float],
    train_loss: float,
    input_compact_grids: List[int],
    estimated_output_grid_token_length: int,
    scaler: Optional[GradScaler]
) -> None:
    aug_batch = []

    for aug_index in range(batch_size):
        augmented_task, rotation, flip, mapping = augment_compact_grids(input_compact_grids, return_config=True)
        shuffled_task = shuffle_all_but_last_pair(augmented_task)
        augmented_task = tokenize_compact_task(shuffled_task)

        # assert len(augmented_task) == ? col/row switch will change the token sequence length

        item = {
                'task': augmented_task,
                'idx': 0,
                'end_of_examples_index': len(augmented_task), # col/row switch will change the token sequence length
                'aux': (rotation, flip, mapping)
            }
        aug_batch.append(item)

    collated_batch = pad_collate(aug_batch, max_seq_length)
    predicted, end_token_indices = auto_regressive_generate(model, collated_batch, max_seq_length, device=device, estimated_output_grid_token_length=estimated_output_grid_token_length, scaler=scaler)
    print()

    for aug_index in range(batch_size):
        try:
            if end_token_indices[aug_index] <= 0:
                logging.info(f'{aug_index} Did not have complete answer')
                continue
            aug_end_of_example = aug_batch[aug_index]['end_of_examples_index']
            predicted_seq = predicted[aug_index][aug_end_of_example:end_token_indices[aug_index]]

            predicted_compact_grids = detokenize_to_compact_grids(predicted_seq)
            if len(predicted_compact_grids) <= 1:
                continue
            predicted_compact_grids = reverse_augment_compact_grids(predicted_compact_grids, *aug_batch[aug_index]['aux'])
            inverse_augmented_seq = tokenize_compact_task(predicted_compact_grids, single_output=True)

            inverse_augmented_seq = [elem[0] for elem in inverse_augmented_seq]

            # Get the sequence prediction for the answer part
            pred_sequence = tuple(inverse_augmented_seq)
            vote[pred_sequence] = vote.get(pred_sequence, 0) + math.exp(-train_loss)
            # logging.info(f'{aug_index}, pred_sequence, {pred_sequence}')
        except Exception as e:
            logging.info(f'{aug_index} Cannot recover the answer: {type(e).__name__}: {str(e)}')
    
    return

def should_exit_early(start_time: float, time_budget: float) -> bool:
    elapsed_time = time.time() - start_time
    if elapsed_time > time_budget:
        logging.info(f"Time budget of {time_budget}s exceeded ({elapsed_time}). Stopping training.")
        return True
    
    return False

def adaptive_generate_sample(
    model: nn.Module,
    compact_grids: List[int],
    batch_size: int,
    device: torch.device,
    *,
    max_seq_length: int,
    scaler: Optional[GradScaler],
    criterion: nn.Module,
    accumulation_steps: int,
    batch_multiplier: int,
    num_epochs: int,
    learning_rate: float,
    base_lr: float,
    time_budget: float = 3600
) -> List[Tuple[Tuple[int, ...], float]]:
    set_deterministic()
    start_time = time.time()  # Add this line at the beginning of the function

    number_of_grids = count_grids_of_compact_grids(compact_grids)
    estimated_output_grid_token_length = estimate_output_grid_token_length(compact_grids)

    assert number_of_grids > 2

    compact_training_data = remove_last_grid(compact_grids)
    number_of_training_grids = count_grids_of_compact_grids(compact_training_data)
    assert number_of_grids == number_of_training_grids + 1 

    total_dataset_size = min(batch_size * 100, batch_size * accumulation_steps * batch_multiplier)
    active_training_dataset = DynamicGridDataset(compact_training_data, total_dataset_size, max_seq_length)
    val_loader = DataLoader(active_training_dataset, batch_size=batch_size, collate_fn=active_training_dataset.pad_collate, prefetch_factor = 4, num_workers = 1, persistent_workers=True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

    logging.info(f'To Active_train: datset:{total_dataset_size}, batch count:{total_dataset_size // batch_size} batch_size:{batch_size}')

    # Add scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=base_lr
    )

    vote = {}

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        train_loss = train_single_epoch(
            model, device, epoch,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            accumulation_steps=accumulation_steps,
            val_loader=val_loader,
            mask_hack=False,
        )

        epoch_duration = time.time() - epoch_start_time
        current_lr = scheduler.get_last_lr()[0]
        if scheduler is not None:
            scheduler.step()

        logging.info(f'\r___Epoch {epoch} [{epoch_duration:.1f}s], loss: {train_loss}, lr: {current_lr:.2e}, b{batch_size}, a{accumulation_steps}, bc{len(val_loader)}, x{max_seq_length}')

        target_vote_count = min(19, 19 * batch_size // 5) # Total number of votes we want scaled by training batch size, no more than 19
        aug_vote_size = target_vote_count
        
        while aug_vote_size > 0:
            set_deterministic() # so we can reproduce the following code
            try:
                iterations = max(1, target_vote_count // aug_vote_size)
                
                for itr_index in range(iterations):
                    logging.info(f'Generate and vote {itr_index}/{iterations} iterations')
                    aug_generate_sample(
                        model, max_seq_length, device, 
                        batch_size=aug_vote_size, 
                        scaler=scaler, 
                        vote=vote, 
                        train_loss=train_loss, 
                        input_compact_grids=compact_grids, 
                        estimated_output_grid_token_length=estimated_output_grid_token_length
                    )

                    if should_exit_early(start_time, time_budget):
                        break

                break  # If successful, exit the loop
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                logging.info(f"OOM in aug_generate_sample with batch_size {aug_vote_size}, retrying with half batch size")
                torch.cuda.empty_cache()
                gc.collect()
                if aug_vote_size == 1:
                    logging.warning("Failed even with minimum batch size, skipping")
                    break                
                aug_vote_size = max(1, aug_vote_size // 2)  # Reduce batch size, but keep minimum of 1
                        
        if train_loss < 1e-2:
            break

        if should_exit_early(start_time, time_budget):
            break

    print()

    # Get top 2 voted sequences
    top_sequences = sorted(vote.items(), key=lambda x: x[1], reverse=True)
    
    return top_sequences

def get_available_gpu(max_retries=10, sleep_time=1):
    """Get an available GPU for exclusive use."""
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return None

    lock_file = "/tmp/gpu_processes.json"
    lock = FileLock(f"{lock_file}.lock")

    for attempt in range(max_retries):
        with lock:
            try:
                # Initialize or load GPU allocations
                if os.path.exists(lock_file):
                    with open(lock_file, 'r') as f:
                        gpu_data = json.load(f)
                else:
                    gpu_data = {str(i): None for i in range(num_gpus)}

                # Clean up finished processes
                for gpu_id in gpu_data:
                    if gpu_data[gpu_id] is not None:
                        if not os.path.exists(f"/proc/{gpu_data[gpu_id]}"):
                            gpu_data[gpu_id] = None

                # Find first available GPU
                current_pid = os.getpid()
                for gpu_id in gpu_data:
                    if gpu_data[gpu_id] is None:
                        gpu_data[gpu_id] = current_pid
                        # Save updated data
                        with open(lock_file, 'w') as f:
                            json.dump(gpu_data, f)
                        return int(gpu_id)

                # No GPU available
                return None

            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logging.error("Failed to get GPU")
                    return None
                time.sleep(sleep_time)

    return None

def setup_device():
    if torch.cuda.is_available():
        try:
            device_id = get_available_gpu()
            torch.cuda.set_device(device_id)
            device = torch.device(f"cuda:{device_id}")
            free_memory = torch.cuda.mem_get_info(device_id)[0]/1024**3
            logging.info(f"Process {os.getpid()} using GPU {device_id} with {free_memory:.1f}GB free memory")
        except Exception as e:
            logging.error(f"GPU setup failed: {e}, falling back to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    return device

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process ARC challenges and solutions.")
    parser.add_argument("--source", type=str, default="arc-agi_evaluation", help="the challenges.json file")
    parser.add_argument("--checkpoint-path", type=str, default="cloud_runs/69.55.141.236/2500/runs/2500/20241029_172457_nogit_nobranch_lr1e-05_bl1e-05_ssu0_bs21_h4_es784_nl18_we10_as1_ph3_ac1_ad1_scosine_oadam_ge1_mh0_ssnone_ss1e-02_c19/Transformer_best.pt", help="path to the checkpoint file")
    parser.add_argument("--input-path", type=str, default="./input_data/", help="path to the input data directory")
    parser.add_argument("--task-id", type=str, default='191', help="taks id")
    parser.add_argument("--time-budget", type=float, default=120, help="time budget in seconds for adaptive training (default: 3600)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f'args{args}')

    system("mkdir -p submission")

    challenges, _ = load_from_json(args.source, args.input_path)
    
    device = setup_device()
        
    training_criterion = nn.CrossEntropyLoss(ignore_index=SpecialToken.PAD.value)
    scaler = GradScaler()

    submission = defaultdict(list)

    if len(args.task_id) <= 4:
        challenge_id, challenge = list(challenges.items())[int(args.task_id)]
    else:
        challenge_id = args.task_id
        challenge = challenges[args.task_id]

    for test_index in range(len(challenge['test'])):
        submission[challenge_id].append({})

        sequence = json_list_to_compact_grids(challenge, None, test_index)

        for batch_size in [6, 5, 4, 3, 2, 1, 0]:
            try:
                model, max_seq_length, checkpoint_args = CheckpointHandler.load_checkpoint_in_production(args.checkpoint_path, device, adjust_max_length=9999)
                mask_hack = checkpoint_args.get('mask_hack', True)
                assert mask_hack == False

                if batch_size > 0:                
                    accumulation_steps = 20 // batch_size
                    top_sequences = adaptive_generate_sample(
                        model, 
                        sequence, 
                        batch_size, 
                        device, 
                        max_seq_length = max_seq_length, 
                        scaler=scaler, 
                        criterion=training_criterion, 
                        accumulation_steps=accumulation_steps, 
                        batch_multiplier=15, 
                        num_epochs=16, 
                        learning_rate=1e-5, 
                        base_lr=1e-8,
                        time_budget=args.time_budget)

                    attempt = []
                    for rank, (seq, weight) in enumerate(top_sequences):
                        logging.info(f"Rank: {rank}, Vote weight: {weight}")
                        attempt.append(seq)

                    if len(attempt) >= 2:
                        submission[challenge_id][test_index] = {'attempt_1': to_json_list(attempt[0]), 'attempt_2': to_json_list(attempt[1])}
                    elif len(attempt) == 1:
                        submission[challenge_id][test_index] = {'attempt_1': to_json_list(attempt[0])}

                    break
                else:
                    assert batch_size == 0

                    tokenized_seq = tokenize_compact_task(sequence)
                    input_length = len(tokenized_seq)
                    output, _ = generate_sample(model, tokenized_seq, max_seq_length, device, mask_hack=mask_hack)
                    json_list = to_json_list(output[input_length:])
                    submission[challenge_id][test_index] = {'attempt_1': json_list}
                    break
            
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                logging.info(f"OutOfMemoryError, batch_size:{batch_size}, {len(sequence)}, {e}")
                # After each batch or when handling exceptions
                torch.cuda.empty_cache()
                gc.collect()  # Force Python garbage collection
                pass

            torch.cuda.empty_cache()

    # Save dictionary to JSON file
    with open(f'submission/{challenge_id}.json', 'w') as f:
        json.dump(submission, f, indent=4)

    logging.info(f"Done, submission.json saved")
        
if __name__ == "__main__":
    main()
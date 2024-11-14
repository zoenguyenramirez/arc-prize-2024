import logging
import argparse
from dataclasses import dataclass
from typing import Optional, Union, Literal
import torch.optim as optim
from pathlib import Path
import re
import time

import torch
import math
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from src.model import Transformer
from src.token import VOCAB_SIZE, SpecialToken
from src.load_data import load_from_json, GridDataset, DynamicGridDataset
from src.utils.helper import set_deterministic
from src.utils.display_diff import compare_sequences, colorize, split_into_chunks
from src.checkpoint_handler import CheckpointHandler
from src.prepare_data import load_dataset
from src.utils.grid_data_process import shuffle_all_but_last_pair, tokenize_compact_task, end_of_examples_mark, preprocess_for_fast_access, preprocess_array_data, detokenize_to_compact_grids, count_grids_of_compact_grids, remove_last_grid, shuffle_all_pairs, to_ascii_board, pad_collate
from src.utils.data_augment import augment_compact_grids, reverse_augment_compact_grids
from src.utils.transformer_helper import dump_input, prefix_mask, create_mask, dump_model_operation
from src.utils.logger_helper import setup_logging


@dataclass
class ModelConfig:
    """Configuration for the Transformer model architecture"""
    max_seq_length: int = 2960
    
    def validate(self):
        assert self.max_seq_length > 0, "Sequence length must be positive"

@dataclass
class TrainingConfig:
    """Configuration for training parameters and optimization"""
    # Device settings
    debug_on_cpu: bool = False
    
    # Optimization parameters
    optimizer: Literal["Adam", "AdamW", "SGD"] = "Adam"
    learning_rate: float = 1e-5
    base_lr: float = 1e-8
    weight_decay: float = 0.0
    
    # Training process
    batch_size: int = 2
    
    # Data loading
    num_workers: int = 1
    prefetch_factor: int = 1
    sequence_cutoff_factor: float = 0.8
    
    def validate(self):
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.batch_size > 0, "Batch size must be positive"
        assert 0 < self.sequence_cutoff_factor < 1, "Sequence cutoff factor must be between 0 and 1"
            
    def get_optimizer(self, model_params) -> optim.Optimizer:
        if self.optimizer == "Adam":
            return optim.Adam(model_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == "AdamW":
            return optim.AdamW(model_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == "SGD":
            return optim.SGD(model_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        raise ValueError(f"Unsupported optimizer: {self.optimizer}")

@dataclass
class ActiveTrainingConfig:
    """Configuration specific to active training phase"""
    batch_size: int = 5
    batch_multiplier: int = 15
    accumulation_steps: int = 4
    num_epochs: int = 16
    augment_vote_count: int = 19
    
    def validate(self):
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.batch_multiplier > 0, "Batch multiplier must be positive"
        assert self.accumulation_steps > 0, "Accumulation steps must be positive"

class Config:
    """Main configuration class that combines all config components"""
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.active_training = ActiveTrainingConfig()
    
    def validate(self):
        """Validate all configurations"""
        self.model.validate()
        self.training.validate()
        self.active_training.validate()
    
    @classmethod
    def from_args(cls, args):
        """Create config from command line arguments"""
        config = cls()
        return config

def diff_seq(predicted, end_of_example, sample_seq_length, target, incorrect_count):
    pred_slice = predicted
    target_slice = target
    generated_pairs = list(zip(pred_slice, target_slice))
    diff_seq = [(pair_index, pair[0], pair[1]) for pair_index, pair in enumerate(generated_pairs) if pair[0] != pair[1]]
    
    if not incorrect_count == len(diff_seq):
        len_pred = len(pred_slice)
        len_target = len(target_slice)
        # Add length difference information to the diff sequence
        diff_seq.append(('length_mismatch', f'predicted:{len_pred}', f'target:{len_target}'))
        
    return diff_seq

def diff_pair(predicted, target):
    generated_pairs = list(zip(predicted, target))
    diff_pairs = [(pair_index - 1, pair[0], pair[1]) for pair_index, pair in enumerate(generated_pairs) if pair[0] != pair[1]]
    
    if len(predicted) != len(target):
        # Add length difference information to the diff sequence
        diff_pairs.append(('length_mismatch', f'predicted:{len(predicted)}', f'target:{len(target)}'))
    
    return diff_pairs

def validate_batch(model, batch, criterion, device, mask_hack):
    model.eval()
    with torch.no_grad():
        input_ids = batch['data'].to(device)
        batch_size, seq_length, _ = input_ids.shape
        target = input_ids.clone()
        target = target[:, 1:, 0]  # (batch_size, seq_length-1)
        input_ids = input_ids[:, :-1, :]  # (batch_size, seq_length-1)

        mask = create_mask(input_ids, device, batch['end_of_examples'], mask_hack)

        outputs = model(input_ids, mask)  # (batch_size, seq_length-1, vocab_size)

        # print('outputs', outputs.shape)

        # Get the predicted class
        _, predicted = torch.max(outputs, 2)
        # print('predicted', predicted.shape)

        assert seq_length == input_ids[:, :, 0].shape[1] + 1, f"{seq_length} != {input_ids[:, :, 0].shape[1]} + 1"
        targets = prefix_mask(target, seq_length - 1, batch['end_of_examples'])
        loss = criterion(outputs.view(-1, outputs.shape[-1]), target.flatten())
        loss = loss.view(batch_size, -1)
        # print('loss', loss.shape)

        # Compare predictions with targets
        correct = (predicted == targets)
        # print('correct', correct.shape)

    # we should return a list of loss and true/false answer
    return loss, correct, predicted.flatten().tolist(), targets.flatten().tolist()

def train_single_epoch(model, device, epoch, *, optimizer, scaler, criterion, accumulation_steps, val_loader, mask_hack):
    torch.cuda.empty_cache()  # Clear unused memory cached by PyTorch
    need_to_zero_grad = True
    model.train()
    total_loss = 0

    for batch_index, batch in enumerate(val_loader):
        batch_start_time = time.time()
        # if epoch % 2 == 0 and batch_index % 2 == 0:
        #     dump_input(batch, batch['indices'], f"{epoch:04d}_{batch_index:04d}_adaptive_train")
        if need_to_zero_grad:
            optimizer.zero_grad()  # Zero gradients at the beginning of accumulation
            need_to_zero_grad = False

        input_ids = batch['data'].to(device)  # (batch_size, seq_length)
        batch_size, seq_length, _ = input_ids.shape
        target = input_ids.clone()
        target = target[:, 1:, 0]  # (batch_size, seq_length-1)
        input_ids = input_ids[:, :-1, :]  # (batch_size, seq_length-1)

        mask = create_mask(input_ids, device, batch['end_of_examples'], mask_hack)

        with autocast(enabled=(scaler is not None)):
            outputs = model(input_ids, mask)  # (batch_size, seq_length-1, vocab_size)

            assert seq_length == input_ids[:, :, 0].shape[1] + 1, f"{seq_length} != {input_ids[:, :, 0].shape[1]} + 1"
            target = prefix_mask(target, seq_length - 1, batch['end_of_examples'])
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), target.reshape(-1))  # (batch_size * (seq_length-1), vocab_size)
            loss = loss / accumulation_steps  # Normalize the loss

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_index + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            need_to_zero_grad = True

        total_loss += loss.item()
        batch_time = time.time() - batch_start_time
        print(f'\rB{batch_index} ({batch_time:.2f}s) {list(input_ids.shape)}', end="", flush=True)

    return total_loss * accumulation_steps  # Undo the normalization for logging

def prepare_model_for_fine_tuning(model, start_layer=10):
    """Prepare model for fine-tuning by freezing early layers and enabling gradients for later layers.
    
    Args:
        model: The transformer model
        start_layer: The starting layer number for fine-tuning (inclusive)
    
    Returns:
        list: Parameters that require gradients for optimization
    """
    trainable_count = 0
    frozen_count = 0
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            requires_grad = False
            if 'fc_out' in name:
                requires_grad = True
                trainable_count += param.numel()
            elif name.startswith('layers.'):
                try:
                    layer_num = int(name.split('.')[1])
                    requires_grad = (layer_num >= start_layer)
                    if requires_grad:
                        trainable_count += param.numel()
                    else:
                        frozen_count += param.numel()
                except (IndexError, ValueError):
                    continue
            
            param.requires_grad = requires_grad
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logging.info(f'Fine-tuning model: {len(trainable_params)} parameter groups, Trainable parameters: {trainable_count:,}, Frozen parameters: {frozen_count:,}')
    
    return trainable_params

def fine_tune_on_task(model, task_sequence, end_of_example, expected_answer_length, device, *, scaler, criterion, validation_criterion, config:Config, mask_hack):
    set_deterministic()

    sample_seq_length = end_of_example + expected_answer_length

    # Tokenized Tasks back to 
    input_compact_grids = detokenize_to_compact_grids(task_sequence.tolist())
    training_target = [elem[0].item() for elem in task_sequence[1:sample_seq_length + 1]][end_of_example - 1:sample_seq_length]

    number_of_grids = count_grids_of_compact_grids(input_compact_grids)

    assert number_of_grids > 2

    compact_training_data = remove_last_grid(input_compact_grids)
    number_of_training_grids = count_grids_of_compact_grids(compact_training_data)
    assert number_of_grids == number_of_training_grids + 1 

    total_dataset_size = config.active_training.batch_size * config.active_training.accumulation_steps * config.active_training.batch_multiplier
    active_training_dataset = DynamicGridDataset(compact_training_data, total_dataset_size)
    val_loader = DataLoader(active_training_dataset, batch_size=config.active_training.batch_size, collate_fn=active_training_dataset.pad_collate, prefetch_factor = 4, num_workers = 1, persistent_workers=True)

    trainable_params = model.parameters() # prepare_model_for_fine_tuning(model)
    optimizer = config.training.get_optimizer(trainable_params)

    logging.info(f'To Active_train: [{end_of_example}:{end_of_example + expected_answer_length}] len:{expected_answer_length}, datset:{total_dataset_size}, batch count:{total_dataset_size // config.active_training.batch_size} batch_size:{config.active_training.batch_size}')

    # model.set_dropout_rate(0) # no significance

    # Add scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.active_training.num_epochs,
        eta_min=config.training.base_lr
    )

    vote = {}

    for epoch in range(config.active_training.num_epochs):
        train_loss = train_single_epoch(
            model, device, epoch,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            accumulation_steps=config.active_training.accumulation_steps,
            val_loader=val_loader,
            mask_hack=mask_hack,
        )
        current_lr = scheduler.get_last_lr()[0]
        if scheduler is not None:
            scheduler.step()

            
        set_deterministic()
        aug_batch = []

        for aug_index in range(config.active_training.augment_vote_count):
            augmented_task, rotation, flip, mapping = augment_compact_grids(input_compact_grids, return_config=True)
            shuffled_task = shuffle_all_but_last_pair(augmented_task)
            augmented_task = tokenize_compact_task(shuffled_task)

            # assert len(shuffled_task) == len(input_compact_grids)
            # assert len(augmented_task) == end_of_example + expected_answer_length, col/row switch will change the token sequence length

            aug_batch.append({
                    'task': augmented_task,
                    'idx': 0,
                    'end_of_examples_index': end_of_examples_mark(augmented_task), # col/row switch will change the token sequence length
                    'aux': (rotation, flip, mapping)
                })

        collated_batch = pad_collate(aug_batch, config.model.max_seq_length)
        assert task_sequence.shape >= collated_batch['data'].shape[:2]
        seq_length_per_batch = collated_batch['data'].shape[1]
        new_loss, new_correct, predicted, aug_target = validate_batch(model, collated_batch, validation_criterion, device, mask_hack)

        for aug_index in range(config.active_training.augment_vote_count):
            try:
                aug_end_of_example = aug_batch[aug_index]['end_of_examples_index']
                aug_seq_length = len(aug_batch[aug_index]['task'])
                predicted_seq = predicted[(seq_length_per_batch - 1) * aug_index + aug_end_of_example - 1 : (seq_length_per_batch - 1) * aug_index + aug_seq_length - 1]

                predicted_compact_grids = detokenize_to_compact_grids(predicted_seq)
                if len(predicted_compact_grids) <= 1:
                    continue
                predicted_compact_grids = reverse_augment_compact_grids(predicted_compact_grids, *aug_batch[aug_index]['aux'])
                inverse_augmented_seq = tokenize_compact_task(predicted_compact_grids, single_output=True)

                inverse_augmented_seq = [elem[0] for elem in inverse_augmented_seq]

                correct_per_sample = new_correct[aug_index][aug_end_of_example:aug_seq_length - 1]
                loss_per_sample = new_loss[aug_index][aug_end_of_example:aug_seq_length - 1]
                assert aug_seq_length > aug_end_of_example
                incorrect_count = (~correct_per_sample).sum().item()
                logging.info(f'Epoch {epoch}/{aug_index}, loss: {train_loss}, lr: {current_lr:.2e}, Post active train Correct {correct_per_sample.sum().item()}, incorrect {incorrect_count}, loss: {loss_per_sample.sum().item() / loss_per_sample.shape[0]}, {diff_seq(inverse_augmented_seq, end_of_example, sample_seq_length, training_target, incorrect_count)}')

                # Get the sequence prediction for the answer part
                pred_sequence = tuple(inverse_augmented_seq)
                vote[pred_sequence] = vote.get(pred_sequence, 0) + math.exp(-train_loss)
                # logging.info(f'{aug_index}, pred_sequence, {pred_sequence}')
            except Exception as e:
                logging.info(f'Epoch {epoch}/{aug_index} Cannot recover the answer: {type(e).__name__}: {str(e)}')

        # print(to_ascii_board(predicted[end_of_example:sample_seq_length], target[end_of_example:sample_seq_length]))

        if train_loss < 1e-2:
            break

    # Get top 2 voted sequences
    top_sequences = sorted(vote.items(), key=lambda x: x[1], reverse=True)
    
    logging.info(f"Top voted sequences:")
    for rank, (seq, weight) in enumerate(top_sequences):
        logging.info(f"Rank: {rank}, Vote weight: {weight}, {len(seq)} vs {len(training_target)} Diff: {diff_pair(seq, training_target)}") #, {seq}
        if rank > 5:
            break

    return model

def main():
    parser = argparse.ArgumentParser(description='Active train for Active Inference (Test-Time Fine-Tuning)')
    
    parser.add_argument('--checkpoint-path', type=str, default='cloud_runs/69.55.141.236/2500/runs/2500/20241029_172457_nogit_nobranch_lr1e-05_bl1e-05_ssu0_bs21_h4_es784_nl18_we10_as1_ph3_ac1_ad1_scosine_oadam_ge1_mh0_ssnone_ss1e-02_c19/Transformer_best.pt',
                        help='Path to the checkpoint file')
    
    parser.add_argument("--dataset-file", type=str, default="./intermediate_data/prepared_dataset_using_arc_evaluation.pth", # 
                        help="Path to the prepared dataset file")

    parser.add_argument('--logger-file', type=str, default='meta_training.log',
                    help='Log file name (default: meta_training.log)')

    args = parser.parse_args()

    # Initialize and validate configuration
    config = Config.from_args(args)
    config.validate()
    
    set_deterministic()

    setup_logging(args.logger_file)  # Set up logging

    device = torch.device("cuda" if (torch.cuda.is_available() and not config.training.debug_on_cpu) else "cpu")
    # load a pre-trained model
    model, max_seq_length, checkpoint_args = CheckpointHandler.load_checkpoint_in_production(args.checkpoint_path, device, adjust_max_length=config.model.max_seq_length)
    mask_hack = checkpoint_args.get('mask_hack', True)

    # to active train on this data
    dataset, data_sources, _ = load_dataset(args.dataset_file) 
    # dataset.sort_by_length(reverse=False)
    dataset.set_augment_seed(-1)
    dataset.set_max_length(max_seq_length) # the task data will be expanded
    dataset.cut_long_sequence(max_seq_length * config.training.sequence_cutoff_factor)

    val_loader = DataLoader(dataset, batch_size=config.training.batch_size, collate_fn=dataset.pad_collate, prefetch_factor = config.training.prefetch_factor, num_workers = config.training.num_workers)

    validation_criterion = nn.CrossEntropyLoss(ignore_index=SpecialToken.PAD.value, reduction='none')
    training_criterion = nn.CrossEntropyLoss(ignore_index=SpecialToken.PAD.value)
    scaler = GradScaler()

    logging.info(f'Processing: {args.checkpoint_path}')
    logging.info(f'dataset: {args.dataset_file} %d', len(dataset))
    logging.info('max_seq_length: %d', max_seq_length)

    # Generate samples
    for batch_index, batch in enumerate(val_loader):
        logging.info(f"BATCH NO.{batch_index}, shape: {list(batch['data'].shape)}")
        loss, correct, _, _ = validate_batch(model, batch, validation_criterion, device, mask_hack)

        for index_in_batch, task_sequence in enumerate(batch['data']):
            end_of_example = batch['end_of_examples'][index_in_batch]

            if end_of_example > task_sequence.shape[0]:
                assert(end_of_example > dataset.max_length)
                continue

            end_of_seq_indices = torch.where(task_sequence[:, 0] == SpecialToken.END.value)

            sample_seq_length = task_sequence.shape[0]
            if end_of_seq_indices[0].numel():
                sample_seq_length = min(sample_seq_length, end_of_seq_indices[0].item())

            logging.info(f"Batch[{index_in_batch}]dataset[{batch['indices'][index_in_batch]}], shape: {list(task_sequence.shape)}, [{end_of_example}:{sample_seq_length}]")

            expected_answer_length = sample_seq_length - end_of_example
            correct_per_sample = correct[index_in_batch][end_of_example:sample_seq_length]
            loss_per_sample = loss[index_in_batch][end_of_example:sample_seq_length]
            assert expected_answer_length == loss_per_sample.shape[0], f"{expected_answer_length} != {loss_per_sample.shape[0]}"

            incorrect_total = (~correct_per_sample).sum().item()

            logging.info(f'Correct {correct_per_sample.sum().item()}, incorrect {incorrect_total}, loss: {loss_per_sample.sum().item() / loss_per_sample.shape[0]}')

            # estimation to save us some time
            if max_seq_length > sample_seq_length and incorrect_total > 0:
                try:
                    set_deterministic()
                    torch.cuda.empty_cache()
                    fine_tune_on_task(model, task_sequence, end_of_example, expected_answer_length, device, criterion = training_criterion, scaler=scaler, validation_criterion=validation_criterion, config=config, mask_hack=mask_hack)
                except torch.cuda.OutOfMemoryError:
                    logging.info('OutOfMemoryError!')
                finally:
                    torch.cuda.empty_cache()
                    CheckpointHandler.restore_model_state(model, args.checkpoint_path, device, adjust_max_length=config.model.max_seq_length)
            else:
                torch.cuda.empty_cache()
                logging.info('skip adaptive training')


if __name__ == "__main__":
    main()

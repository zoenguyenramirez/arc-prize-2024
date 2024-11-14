import logging
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from src.token import VOCAB_SIZE, SpecialToken
from src.load_data import load_from_json, GridDataset
from src.utils.helper import set_deterministic
from src.checkpoint_handler import CheckpointHandler
from src.meta_adaptive_train import validate_batch
from src.utils.logger_helper import setup_logging

def main():
    parser = argparse.ArgumentParser(description='Active train for Active Inference (Test-Time Fine-Tuning)')
    
    parser.add_argument('--checkpoint-path', type=str, default='cloud_runs/69.55.141.236/2500/runs/2500/20241029_043432_nogit_nobranch_lr3e-05_bl2e-05_ssu0_bs21_h4_es784_nl18_we10_as1_ph3_ac1_ad1_scosine_oadam_ge1_mh0_ssnone_ss1e-02_c11/Transformer_latest.pt',
                        help='Path to the checkpoint file')
    
    parser.add_argument('--data-source', type=str, default='arc-agi_evaluation',
                        help='Data source to use (default: arc-agi_evaluation)')

    parser.add_argument('--logger-file', type=str, default='validation.log',
                    help='Log file name (default: sample_generation.log)')
    
    args = parser.parse_args()
    
    set_deterministic()

    setup_logging(args.logger_file)  # Set up logging

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load a pre-trained model
    model, max_seq_length, checkpoint_args = CheckpointHandler.load_checkpoint_in_production(args.checkpoint_path, device, adjust_max_length=10100)
    mask_hack = checkpoint_args.get('mask_hack', True)

    # Load data
    data_sources = [args.data_source]
    all_challenges = {}
    all_solutions = {}

    for source in data_sources:
        try:
            challenges, solutions = load_from_json(source, './input_data/')
            all_challenges.update(challenges)
            all_solutions.update(solutions)
        except FileNotFoundError as e:
            logging.error("Error loading %s: %s. Skipping this data source.", source, e)

    if not all_challenges:
        logging.error("No data could be loaded. Please check the file paths and data sources.")
        return

    # to active train on this data
    dataset = GridDataset.load_from_paired_file(all_challenges, all_solutions)
    # dataset, data_sources, _ = load_dataset(args.dataset_file) 
    dataset.set_augment_seed(1)
    dataset.set_max_length(max_seq_length) # the task data will be expanded
    dataset.cut_long_sequence(max_seq_length * 0.92)

    val_loader = DataLoader(dataset, batch_size=1, collate_fn=dataset.pad_collate, prefetch_factor = 1, num_workers = 1)

    validation_criterion = nn.CrossEntropyLoss(ignore_index=SpecialToken.PAD.value, reduction='none')

    logging.info(f'Processing: {args.checkpoint_path}')
    logging.info('dataset: %s %d', args.data_source, len(dataset))
    logging.info('max_seq_length: %d', max_seq_length)

    corect_answer_count = 0
    tested_count = 0
    total_correct_token_length = 0

    # Generate samples
    for batch_index, batch in enumerate(val_loader):
        loss, correct, _, _ = validate_batch(model, batch, validation_criterion, device, mask_hack)
        torch.cuda.empty_cache()

        for index_in_batch, task_sequence in enumerate(batch['data']):
            end_of_example = batch['end_of_examples'][index_in_batch]

            if end_of_example > task_sequence.shape[0]:
                assert(end_of_example > dataset.max_length)
                continue

            tested_count += 1

            end_of_seq_indices = torch.where(task_sequence[:, 0] == SpecialToken.END.value)

            sample_seq_length = task_sequence.shape[0]
            if end_of_seq_indices[0].numel():
                sample_seq_length = min(sample_seq_length, end_of_seq_indices[0].item())

            expected_answer_length = sample_seq_length - end_of_example
            correct_per_sample = correct[index_in_batch][end_of_example:sample_seq_length]
            loss_per_sample = loss[index_in_batch][end_of_example:sample_seq_length]
            assert expected_answer_length == loss_per_sample.shape[0], f"{expected_answer_length} != {loss_per_sample.shape[0]}"

            incorrect_total = (~correct_per_sample).sum().item()
            
            total_correct_token_length += correct_per_sample.sum().item()

            # estimation to save us some time
            if max_seq_length > sample_seq_length and incorrect_total <= 0:
                corect_answer_count += 1

        print(f'\rcorect_answer_count {corect_answer_count} batch {batch_index}/{tested_count}\t', end="", flush=True)

    logging.info('\nCorrect cases = %d/%d, success rate %.2f%%, total correct tokens: %d', corect_answer_count, tested_count, (corect_answer_count) * 100. / tested_count, total_correct_token_length)
    
    logging.info('-' * 20)

if __name__ == "__main__":#
    main()

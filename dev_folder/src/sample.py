import logging
import argparse
import torch
from src.model import Transformer
from src.token import VOCAB_SIZE, SpecialToken
from src.load_data import load_from_json, GridDataset
from src.utils.helper import set_deterministic
from src.utils.display_diff import compare_sequences, colorize, split_into_chunks
from src.utils.transformer_helper import create_mask
from src.checkpoint_handler import CheckpointHandler
from src.utils.logger_helper import setup_logging

debug_on_cpu = False

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

def generate_sample(model, input_sequence, max_length, device, *, mask_hack:bool = False, early_stop = None):
    model.eval()
    y = 0
    x = 0
    coord = (-1, -1)
    with torch.no_grad():
        input_ids = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0).to(device)  # (1, seq_length)

        seq_length = len(input_sequence)
        for generated_token_index in range(max_length - seq_length):
            mask = create_mask(input_ids, device, [seq_length], mask_hack)
            outputs = model(input_ids, mask)  # (1, seq_length, vocab_size)
            next_token_logits = outputs[0, -1, :]  # (vocab_size)
            next_token_id = torch.argmax(next_token_logits).item()

            if early_stop:
                expected_token = early_stop[seq_length + generated_token_index]
                if expected_token != next_token_id:
                    # print(f'early stop @ [{generated_token_index}] expected:{expected_token} vs actual:{next_token_id}')
                    return [], generated_token_index    # Do not put + 1 here, because we are counting the number of correct tokens
                
            print(f'\r{generated_token_index} ', end="", flush=True)

            if next_token_id < SpecialToken.CELL_TOKEN_SIZE.value:
                coord = (y, x)
                x = x + 1
                x = min(x, model.max_grid_size - 1)
            elif next_token_id == SpecialToken.ROW_SEPARATOR.value:
                coord = (y, x)
                x = 0
                y = y + 1
                y = min(y, model.max_grid_size - 1)
            else:
                y = 0
                x = 0
                coord = (-1, -1)

            input_ids = torch.cat([input_ids, torch.tensor([[[next_token_id, coord[0], coord[1], -1, -1]]], dtype=torch.long, device=device)], dim=1)  # (1, seq_length + 1)
            if next_token_id == SpecialToken.END.value or \
                (generated_token_index > 1000 and not early_stop): # we don't know the right answer and this has generated more than 30x30 cells
                return input_ids.squeeze(0).tolist(), generated_token_index + 1
            torch.cuda.empty_cache()
            
    return input_ids.squeeze(0).tolist(), max_length - seq_length

def main():
    parser = argparse.ArgumentParser(description='Generate samples using the trained model')
    
    parser.add_argument('--data-source', type=str, default='arc-agi_evaluation',
                        help='Data source to use (default: arc-agi_evaluation)')
    
    parser.add_argument('--checkpoint-path', type=str, default='cloud_runs/69.55.141.236/2500/runs/2500/20241029_043432_nogit_nobranch_lr3e-05_bl2e-05_ssu0_bs21_h4_es784_nl18_we10_as1_ph3_ac1_ad1_scosine_oadam_ge1_mh0_ssnone_ss1e-02_c11/Transformer_latest.pt',
                        help='Path to the checkpoint file')

    parser.add_argument('--start-testing-index', type=int, default=0,
                        help='Index of the sample to generate (default: 0)')

    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Enable verbose output')
    
    parser.add_argument('--second-only', action='store_true', default=False,
                        help='Use second test only')
    
    parser.add_argument('--unlimited-sequence', action='store_true', default=True,
                        help='Index of the max sequence to be inf')
    
    parser.add_argument('--logger-file', type=str, default='sample_generation.log',
                    help='Log file name (default: sample_generation.log)')

    args = parser.parse_args()

    setup_logging(args.logger_file)  # Set up logging

    set_deterministic()

    # Load data
    data_sources = [args.data_source]
    all_challenges = {}

    for source in data_sources:
        try:
            challenges, solutions = load_from_json(source, './input_data/')
            all_challenges.update(challenges)
        except FileNotFoundError as e:
            logging.error("Error loading %s: %s. Skipping this data source.", source, e)

    if not all_challenges:
        logging.error("No data could be loaded. Please check the file paths and data sources.")
        return

    device = torch.device("cuda" if (torch.cuda.is_available() and not debug_on_cpu) else "cpu")
    model, max_seq_length, checkpoint_args = CheckpointHandler.load_checkpoint_in_production(args.checkpoint_path, device, adjust_max_length=12000 if args.unlimited_sequence else 0)

    mask_hack = checkpoint_args.get('mask_hack', True)

    # Create the dataset
    dataset_ref = GridDataset.load_from_paired_file(all_challenges, solutions, second_only=args.second_only)
    dataset = GridDataset.load_from_paired_file(all_challenges, None, second_only=args.second_only)

    logging.info(f'Processing: {args.checkpoint_path}')

    logging.info('dataset: %s %d', args.data_source, len(dataset))
    logging.info('dataset_ref: %d', len(dataset_ref))
    logging.info('max_seq_length: %d', max_seq_length)

    mismatch_count = 0    
    oom_count = 0
    tested_count = 0
    total_correct_token_length = 0
    # Generate samples
    start_index = args.start_testing_index
    for i in range(start_index, len(dataset)):
        input_sequence = dataset[i]
        # print('input_sequence a', input_sequence)
        expected_sequence = [s[0] for s in dataset_ref[i]['task']]
    
        if max_seq_length > len(input_sequence) and max_seq_length > len(expected_sequence):
            tested_count += 1
            try:
                sample, generated_length = generate_sample(model, input_sequence['task'], max_seq_length, device, early_stop = expected_sequence, mask_hack=mask_hack)
                total_correct_token_length += generated_length

                sample = [s[0] for s in sample]
                if sample != expected_sequence[:max_seq_length]:
                    mismatch_count += 1

                    if args.verbose:
                        input_length = len(input_sequence)
                        logging.info("\nInput sequence (%d, %d):", i, input_length)
                        logging.info(format_batch([torch.tensor(input_sequence['task'])], max_print_length=99999))

                        compare_sequences(
                            format_batch([torch.tensor(expected_sequence[input_length + 1:max_seq_length])]), 
                                        format_batch([torch.tensor(sample[input_length + 1:])]))                
                        continue
            except torch.cuda.OutOfMemoryError:
                oom_count += 1
            except Exception as e:
                logging.error(f"Unexpected error during sample generation at index {i}: {str(e)}")

            print(f'\r____. Tested {tested_count}@{i + 1}, generated {generated_length or 0}. Failed cases = {mismatch_count}/{tested_count}, oom_count: {oom_count}\t', end="", flush=True)
            
    logging.info('\nFailed cases = %d/%d, success rate %.2f%%, total correct tokens: %d', mismatch_count, tested_count, (tested_count - mismatch_count) * 100. / tested_count, total_correct_token_length)
    
    logging.info('-' * 20)

if __name__ == "__main__":
    main()

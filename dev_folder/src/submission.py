import torch
import json
from os import system

import argparse
from collections import defaultdict

from src.checkpoint_handler import CheckpointHandler
from src.load_data import load_from_json, GridDataset
from src.utils.grid_data_process import json_list_to_compact_grids, tokenize_compact_task, to_json_list
from src.sample import generate_sample

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process ARC challenges and solutions.")
    parser.add_argument("--source", type=str, default="arc-agi_test", help="the challenges.json file")
    parser.add_argument("--checkpoint-path", type=str, default="cloud_runs/69.55.141.119/genesis/runs/genesis/20241026_150711_nogit_nobranch_lr2e-06_bl2e-06_ssu0_bs22_h4_es784_nl18_we10_as1_ph3_ac1_ad1_scosine_oadam_ge1_c43/Transformer_latest.pt", help="path to the checkpoint file")
    parser.add_argument("--input-path", type=str, default="./input_data/", help="path to the input data directory")    
    parser.add_argument("--task-id", type=str, default='00576224', help="taks id")
    args = parser.parse_args()

    system("mkdir -p submission")

    challenges, _ = load_from_json(args.source, args.input_path)
    
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    model, max_seq_length, checkpoint_args = CheckpointHandler.load_checkpoint_in_production(args.checkpoint_path, device, adjust_max_length=10000)

    mask_hack = checkpoint_args.get('mask_hack', True)

    submission = defaultdict(list)

    if len(args.task_id) <= 4:
        challenge_id, challenge = list(challenges.items())[int(args.task_id)]
    else:
        challenge_id = args.task_id
        challenge = challenges[args.task_id]

    for test_index in range(len(challenge['test'])):
        sequence = json_list_to_compact_grids(challenge, None, test_index)

        tokenized_seq = tokenize_compact_task(sequence)

        input_length = len(tokenized_seq)

        try:
            output, _ = generate_sample(model, tokenized_seq, max_seq_length, device, mask_hack=mask_hack)

            json_list = to_json_list(output[input_length:])

            submission[challenge_id].append({'attempt_1': json_list, 'attempt_2': [[0]]})
        except Exception as e:
            print('Error', e)

    # Save dictionary to JSON file
    with open(f'submission/{challenge_id}.json', 'w') as f:
        json.dump(submission, f, indent=4)
        
if __name__ == "__main__":
    main()
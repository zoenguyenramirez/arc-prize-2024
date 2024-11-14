import logging
import argparse
import torch
from src.model import Transformer
from src.token import VOCAB_SIZE, SpecialToken
from src.load_data import load_from_json, GridDataset
from src.utils.helper import set_deterministic
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
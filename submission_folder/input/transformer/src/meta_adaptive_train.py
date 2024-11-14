import logging
import argparse
from dataclasses import dataclass
from typing import Optional, Union, Literal
import torch.optim as optim
from pathlib import Path
import re
import time

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from src.utils.transformer_helper import dump_input, prefix_mask, create_mask, dump_model_operation

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


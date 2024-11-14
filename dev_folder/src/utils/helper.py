"""
    Any utility functions that don't directly affect AI writing our code
    can be placed in this module to keep the main code files clean and focused.
"""

import platform
import sys
import os
import torch
import random
import numpy as np
import torch
import os
import math
import tensorflow as tf
from dataclasses import dataclass, fields, asdict
from datetime import datetime
import subprocess
from tensorflow.core.util import event_pb2
from tensorflow.python.framework import tensor_util
import argparse
import logging
from tensorflow.python.framework.errors_impl import DataLossError
import sys
import gc
import psutil
from typing import List, Any, Dict
import threading
from src.token import SpecialToken
from src.checkpoint_handler import CheckpointHandler

def shorten(name: str) -> str:
    return ''.join(word[0] for word in name.split('_'))

@dataclass
class HyperParameters:
    learning_rate: float
    base_lr:       float
    step_size_up:  int

    batch_size:    int
    heads:         int
    embed_size:    int
    num_layers:    int
    warmup_epochs: int
    accumulation_steps:  int
    progressive_head: int

    auto_cast:     bool
    augment_data:  bool
    schedular:     str
    optimizer:     str
    grid_encoder:  bool  # New parameter for grid encoder
    mask_hack:     bool

    sampling_strategy: str
    sampling_sigma:     float
    
    def to_components(self) -> list[str]:
        components = []
        for field in fields(self):
            short_field_name = shorten(field.name)
            value = getattr(self, field.name)
            if isinstance(value, float):
                components.append(f"{short_field_name}{value:.0e}")
            elif isinstance(value, bool):
                components.append(f"{short_field_name}{1 if value else 0}")
            else:
                components.append(f"{short_field_name}{value}")
        return components
    
    def to_command_args(self) -> List[str]:
        command = []
        for field in fields(self):
            arg_name = f"--{field.name.replace('_', '-')}"
            arg_value = str(getattr(self, field.name))
            command.extend([arg_name, arg_value])
        return command    
    
    def to_log_dict(self) -> Dict[str, Any]:
        return asdict(self)
    

    def assert_checkpoint(self, other: dict):
        for field in fields(self):
            if field.name in ['learning_rate', 'base_lr', 'warmup_epochs', 'accumulation_steps', 'batch_size', 'step_size_up', 'schedular', 'sampling_strategy', 'sampling_sigma']:
                continue

            value_a = getattr(self, field.name)
            value_b = other.get(field.name, True if field.name == 'mask_hack' else None)
            assert value_a == value_b, f'{value_a} != {value_b} for {field.name}'

def get_git_hash() -> str:
    """Get the current Git commit hash."""
    try:
        command = ["git", "rev-parse", "--short", "HEAD"]
        return subprocess.check_output(command, stderr=subprocess.DEVNULL).decode("utf-8").strip()
    except subprocess.CalledProcessError:
        return "nogit"

def get_git_branch() -> str:
    """Get the current Git branch name."""
    try:
        command = ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        return subprocess.check_output(command, stderr=subprocess.DEVNULL).decode("utf-8").strip()
    except subprocess.CalledProcessError:
        return "nobranch"
    
def generate_run_name(params: HyperParameters, hyperparameter_tuning_mode: bool = True) -> str:
    """
    Generate a unique run name based on hyperparameters and current state.
    
    Args:
        params (HyperParameters): Object containing hyperparameters.
        hyperparameter_tuning_mode (bool): Whether to include timestamp and git hash.
    
    Returns:
        str: A unique run name.
    """
    components = []
    
    if hyperparameter_tuning_mode:
        components.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
        components.append(get_git_hash())

    components.append(get_git_branch())
    
    components.extend(params.to_components())

    return "_".join(components)

def set_deterministic(seed = 0):
    # Set seed for Python's random module
    random.seed(seed)
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for PyTorch
    torch.manual_seed(seed)
    
    # Set seed for CUDA if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Use deterministic algorithms
    torch.use_deterministic_algorithms(True)
    
    # Set environment variable for CUDA
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def read_tensorboard_events(path):
    dataset = tf.data.TFRecordDataset(path)
    try:
        for serialized_example in dataset:
            try:
                event = event_pb2.Event.FromString(serialized_example.numpy())
                for value in event.summary.value:
                    yield value.tag, event.step, value.metadata.plugin_data.content
            except Exception as e:
                logging.warning(f"Error processing event in {path}: {str(e)}")
                continue
    except DataLossError as e:
        logging.warning(f"DataLossError in file {path}: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error reading {path}: {str(e)}")

def check_existing_hyperparameters(log_dir):
    """
    Check if hyperparameters have been logged to TensorBoard in a previous session.

    Args:
    log_dir (str): Path to the TensorBoard log directory.

    Returns:
    dict or None: A dictionary of hyperparameters if found, None otherwise.
    """
    if os.path.exists(log_dir):        
        for event_file in os.listdir(log_dir):
            if event_file.startswith("events.out.tfevents"):
                path = os.path.join(log_dir, event_file)
                for tag, step, content in read_tensorboard_events(path):
                    if tag == '_hparams_/session_start_info':
                        logging.info("Existing hyperparameter event was discovered and this run is being skipped now.")
                        sys.exit(1)  # Error, found existing hyperparameter event

    sys.exit(0) # Success

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def log_hyperparameters(hyperparams: HyperParameters, train_loss, val_loss, gpu_memory_allocated, writer):
    metric_dict = {"train_loss": train_loss, "val_loss": val_loss, "gpu_memory_allocated": gpu_memory_allocated}
    writer.add_hparams(hyperparams.to_log_dict(), metric_dict, run_name=".")

def validate_parameters(hyperparams: HyperParameters, args):
    assert hyperparams.embed_size % 4 == 0
    if hyperparams.schedular == 'cyclic':
        assert hyperparams.warmup_epochs == 0
        assert hyperparams.step_size_up > 0
    else:
        assert hyperparams.schedular == 'cosine'
        assert hyperparams.step_size_up == 0
        assert hyperparams.warmup_epochs > 0

    CheckpointHandler.validate_total_epoch(args.epochs)

def get_env():
    # Get environment info
    env_info = {
        "OS": platform.system(),
        "OS Version": platform.version(),
        "Python Version": sys.version,
        "Python Implementation": platform.python_implementation(),
        "CPU": platform.processor(),
        "Machine": platform.machine(),
        "CUDA Available": torch.cuda.is_available(),
        "CUDA Version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "PyTorch Version": torch.__version__,
        "NumPy Version": np.__version__,
        "Launch Command": " ".join(sys.argv),
    }

    # Get CUDA device info if available
    if torch.cuda.is_available():
        env_info["CUDA Device Count"] = torch.cuda.device_count()
        env_info["CUDA Device Name"] = torch.cuda.get_device_name(0)  # Get name of the first CUDA device

    # Get relevant environment variables
    relevant_env_vars = ["PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "CUDA_HOME"]
    for var in relevant_env_vars:
        env_info[f"Env: {var}"] = os.environ.get(var, "Not set")

    # Format the information as a string
    env_str = "\n\n".join([f"{key}: {value}" for key, value in env_info.items()])

    return env_str


def gc_collect(report_memory: bool = False):
    gc.collect()
    if report_memory:

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        # Get system-wide memory information
        system_memory = psutil.virtual_memory()
        
        used_memory = memory_info.rss / 1024 / 1024  # Convert to MB
        free_memory = system_memory.available / 1024 / 1024  # Convert to MB
        
        print(f"Used memory: {used_memory:.2f} MB, Free memory: {free_memory:.2f} MB")

def print_thread_info():
    current_thread = threading.current_thread()
    thread_id = threading.get_ident()
    is_main_thread = current_thread == threading.main_thread()
    
    print(f"Thread name: {current_thread.name} Is main thread: {is_main_thread} Current thread ID: {thread_id}")

def round_up_to_multiple(value, multiple):
    return math.ceil(value / multiple) * multiple

def detokenize_grid(tokenized_sequence):
    # print('detokenize_grid', tokenized_sequence)
    grid = []
    current_row = []
    for cell in tokenized_sequence:
        if cell == SpecialToken.ROW_SEPARATOR.value:
            if current_row:
                grid.append(current_row)
                current_row = []
        else:
            current_row.append(cell)
    if current_row:
        grid.append(current_row)
    return grid

def is_gpu_memory_low(threshold_mb=1000):
    if torch.cuda.is_available():
        free_memory, total_memory = torch.cuda.mem_get_info()
        free_memory_mb = free_memory / (1024 * 1024)  # Convert to MB
        return free_memory_mb < threshold_mb
    return False
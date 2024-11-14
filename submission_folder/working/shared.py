from subprocess import Popen, PIPE, STDOUT

import torch
import os
import json
import glob
from pathlib import Path

from datetime import datetime
import pytz



import torch
import os
import time

# Set timezone
os.environ['TZ'] = 'America/Los_Angeles'
time.tzset()

# Get current time in new timezone
current_time = time.strftime('%Y-%m-%d %H:%M:%S')
print(current_time)

sample_path = '../input/arc-prize-2024/sample_submission.json'

def print_cuda_devices():
    if not torch.cuda.is_available():
        print("No CUDA devices available")
        return
        
    device_count = torch.cuda.device_count()
    print(f"Found {device_count} CUDA device(s):")
    
    for i in range(device_count):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")

def is_in_slices_safe(number: int, slice_string: str) -> bool:
    """
    Safe version with error handling.
    """
    try:
        parts = slice_string.split(',')
        
        for part in parts:
            part = part.strip()  # Handle potential whitespace
            if ':' in part:
                start, end = map(int, part.split(':'))
                if start <= number <= end:
                    return True
            else:
                if number == int(part):
                    return True
        return False
    except ValueError:
        raise ValueError("Invalid slice format. Use format like '5:9,12,15:99'")
    except Exception as e:
        raise Exception(f"Error processing slices: {str(e)}")

def merge_with_sample(data_path, sample_path, sub_solver): 
    with open(sample_path,'r') as f:        
        sample = json.load(f) 

    # ...............................................................................
    with open(data_path,'r') as f:
        data = json.load(f)
        tasks_name = list(data.keys())
        tasks_file = list(data.values())

    for n in range(len(tasks_name)):
        task = tasks_file[n]
        t = tasks_name[n]
            
        for i in range(len(task['test'])): 
            # First check if task id exists
            if t not in sub_solver:
                sub_solver[t] = []
            
            # Ensure we have enough elements in the list
            while len(sub_solver[t]) <= i:
                sub_solver[t].append({})
            
            # Now check if attempt_1 exists or is empty
            if 'attempt_1' not in sub_solver[t][i] or not sub_solver[t][i]['attempt_1']:
                sub_solver[t][i]['attempt_1'] = sample[t][i]['attempt_1']
                
            # Same for attempt_2
            if 'attempt_2' not in sub_solver[t][i] or not sub_solver[t][i]['attempt_2']:
                sub_solver[t][i]['attempt_2'] = sample[t][i]['attempt_2']

    return sub_solver

def get_task_count():
    # Create the target datetime in PST
    pacific_tz = pytz.timezone('US/Pacific')
    target_date = pacific_tz.localize(datetime(2024, 11, 8, 14, 57))
    
    # Get current time in PST
    current_time = datetime.now(pacific_tz)
    
    # Compare and return appropriate value
    task_count = 9 if current_time < target_date else 2000
    return task_count

def mySystem(cmd):
    print(cmd)
    process = Popen(cmd, shell=True) # stdout=PIPE, stderr=STDOUT, 
    return process.wait() == 0 # do not assert here, to keep the program going

# Run the function
print_cuda_devices()

def find_transformer_model(directory):
    """
    Find the single Transformer*.pt file in the given directory and its subdirectories.
    
    Args:
        directory (str): Directory path to search in
    
    Returns:
        str: Full path to the found Transformer*.pt file
    
    Raises:
        AssertionError: If directory doesn't exist, no file found, or multiple files found
    """
    # Check if directory exists
    assert os.path.exists(directory), f"Directory '{directory}' does not exist"
    
    # Search for Transformer*.pt files
    pattern = os.path.join(directory, "**", "Transformer*.pt")
    matching_files = glob.glob(pattern, recursive=True)
    
    # Assert we found exactly one file
    assert len(matching_files) > 0, f"No Transformer*.pt file found in '{directory}'"
    assert len(matching_files) == 1, (
        f"Multiple Transformer*.pt files found in '{directory}': {matching_files}"
    )
    
    return matching_files[0]

def print_version_files():
    input_dir = "../input/"
    pattern = os.path.join(input_dir, "**", "__version__.txt")
    version_files = glob.glob(pattern, recursive=True)
    
    if not version_files:
        print("No __version__.txt files found")
        return
        
    for file_path in version_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read().strip()
            relative_path = os.path.relpath(file_path, input_dir)
            print(f"\n{relative_path}:")
            print(content)
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")

def is_tmp_writable():
    try:
        testfile = os.path.join('/tmp', 'write_test')
        with open(testfile, 'w') as f:
            f.write('test')
        os.remove(testfile)
        return True
    except (IOError, OSError):
        return False

assert is_tmp_writable()

# Display detailed GPU information
# !nvidia-smi
# !ls -R ../input

model_path = find_transformer_model("../input/transformer_model/")

# model_path = '../input/transformer_model/Transformer_best.pt'

print('\n\nmodel_path:', model_path)

# Add this line after line 164
print("\nVersion files found:")
print_version_files()
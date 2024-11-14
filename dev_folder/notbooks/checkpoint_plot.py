import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from enum import Enum

from plot_funcs import plot_one
from src.model import Transformer
from src.token import VOCAB_SIZE, SpecialToken

from plot_funcs import plot_one
from src.utils.helper import detokenize_grid

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


def plot_task(task, i, t):
    """    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app    """    
    
    num_train = len(task['train'])
    num_test  = len(task['test'])

    w = num_train + num_test
    fig, axs = plt.subplots(2, w, figsize=(3*w, 3*2))
    plt.suptitle(f'Set #{i}, {t}:', fontsize=20, fontweight='bold', y=1)


    for j in range(num_train):     
        plot_one(axs[0, j], task, j, 'train', 'input')
        plot_one(axs[1, j], task, j, 'train', 'output')        
    for j in range(num_test):     
        plot_one(axs[0, num_train + j], task, j, 'test', 'input')
        plot_one(axs[1, num_train + j], task, j, 'test', 'output')        
       
    fig.patch.set_linewidth(5)
    fig.patch.set_edgecolor('black') 
    fig.patch.set_facecolor('#dddddd')
   
    plt.tight_layout()
    
    plt.show()  

def plot_answer(input_sequence, answer, task_index):
    print (f'display{task_index}, {len(input_sequence)}, {len(answer)}, {len(answer) - len(input_sequence)}')
    # Extract task data
    task = {'train': [], 'test': []}
    
    current_section = None
    current_data = []
    input_length = len(input_sequence)
    
    for index, element in enumerate(answer):
        token = element[0]
        if token == SpecialToken.START_INPUT.value:
            current_section = 'input'
            current_data = []
        elif token == SpecialToken.END_INPUT.value:
            if current_section == 'input':
                task['train'].append({'input': detokenize_grid(current_data)})
        elif token == SpecialToken.START_OUTPUT.value:
            current_section = 'output'
            current_data = []
        elif token == SpecialToken.END_OUTPUT.value:
            if current_section == 'output':
                output_data = {'output': detokenize_grid(current_data)}
                if index < input_length:
                    task['train'][-1].update(output_data)
                else:
                    print ('new answer', output_data)
                    last_train = task['train'].pop()
                    last_train.update(output_data)
                    task['test'].append(last_train)                    
        else:
            if token < 10 or token == SpecialToken.ROW_SEPARATOR.value:
                current_data.append(token)
                
    # Plot task
    plot_task(task, task_index, f"{task_index}")
    
    # print('fetched', sequence)
    # print('detokenized', task)    
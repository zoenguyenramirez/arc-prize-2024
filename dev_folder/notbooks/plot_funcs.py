from   matplotlib import colors
import matplotlib.pyplot as plt

# 0:black, 1:blue, 2:red, 3:green, 4:yellow, # 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown

cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)

def plot_legnd():

    plt.figure(figsize=(3, 1), dpi=150)
    plt.imshow([list(range(10))], cmap=cmap, norm=norm)
    plt.xticks(list(range(10)))
    plt.yticks([])
    plt.show()


def plot_task(task, task_solutions, i, t, save_prefix=""):
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
    
    # Modify this part to use plot_one for 'Test output'
    for j in range(num_test):
        plot_one(axs[0, num_train + j], task, j, 'test', 'input')
    
        test_output = {'test': [{'output': task_solutions[j]}]}
        plot_one(axs[1, num_train + j], test_output, 0, 'test', 'output')

    fig.patch.set_linewidth(5)
    fig.patch.set_edgecolor('black') 
    fig.patch.set_facecolor('#dddddd')
   
    plt.tight_layout()
    
    if save_prefix != "":
        # Save the figure as a JPG file
        filename = f"{save_prefix}_task_{i}_{t}.jpg"
        plt.savefig(filename, format='jpg', dpi=300, bbox_inches='tight')
        
    plt.show()  

def plot_one(ax, task_data, i, train_or_test, input_or_output):
    try:
        input_matrix = task_data[train_or_test][i][input_or_output]
    except:
        return
        
    im = ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True, which='both', color='lightgrey', linewidth=0.5)
    
    plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
    ax.set_xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])     
    ax.set_yticks([x-0.5 for x in range(1 + len(input_matrix))])
    
    # Calculate font size based on grid dimensions
    grid_size = max(len(input_matrix), len(input_matrix[0]))
    base_font_size = 10  # Base font size for small grids
    min_font_size = 4    # Minimum font size to ensure readability
    font_size = max(base_font_size - (grid_size - 5) * 0.5, min_font_size)
    
    # Add text annotations with adjusted font size
    for y in range(len(input_matrix)):
        for x in range(len(input_matrix[0])):
            value = input_matrix[y][x]
            text_color = 'white' if value > 5 or value == 0 else 'black'
            ax.text(x, y, str(value), ha='center', va='center', color=text_color, fontsize=font_size)
    
    ax.set_title(f'{train_or_test} {input_or_output}', color='black' if train_or_test == 'train' else 'red')
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.manifold import TSNE

def aspect_ratio_greater_than(tensor_np, ratio):
    if len(tensor_np.shape) == 1:
      return tensor_np.shape[0] > ratio
    height, width = sorted(tensor_np.shape, reverse=True)[:2]

    aspect_ratio = height / width

    return aspect_ratio > ratio

def analyze_tensor(tensor, name="Tensor"):
    """
    Analyze a tensor by calculating mean, max, std, and avg for each row.
    Supports mouse click to show row and column vectors.
    
    Args:
    tensor (torch.Tensor): The input tensor to analyze
    name (str): Name of the tensor for display purposes
    
    Returns:
    dict: A dictionary containing the statistics for each row
    """
    print(f"\n--- Analyzing {name} ---")
    print(f"Shape: {tensor.shape}")
    
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    
    # Ensure the tensor is on CPU and convert to numpy
    if isinstance(tensor, torch.Tensor):
        tensor_np = tensor.detach().cpu().numpy()
    else:
        tensor_np = tensor
    
    results = []
    
    for i, row in enumerate(tensor_np):
        min_val = np.min(row)
        max_val = np.max(row)
        std = np.std(row)
        avg = np.average(row)
        sum = np.sum(row)
        
        results.append({
            "Row": i,
            "Avg": avg,
            "Std": std,
            "Min": min_val,
            "Max": max_val,
            "Sum": sum
        })

    display(pd.DataFrame(results))

    aspect='auto' if aspect_ratio_greater_than(tensor_np, 5) else 'equal'
    
    # Visualization using matplotlib
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if tensor_np.ndim == 2:
        im = ax.imshow(tensor_np, interpolation='nearest', cmap='viridis', aspect=aspect)
    elif tensor_np.ndim == 3:
        if tensor_np.shape[0] == 1:
            im = ax.imshow(tensor_np[0], interpolation='nearest', cmap='viridis', aspect=aspect)
        else:  # Multiple 2D slices
            raise ValueError("Cannot display 3D tensor with multiple slices")
    else:
        raise ValueError("Unsupported tensor dimensions")
    
    ax.set_title(name, fontsize=16, color='blue')
    fig.colorbar(im)
    
    # Add text box for displaying vector information
    text_box = ax.text(0.02, 0.98, '', transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def on_click(event):
        if event.inaxes == ax:
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= x < tensor_np.shape[1] and 0 <= y < tensor_np.shape[0]:
                row_vector = tensor_np[y, :]
                col_vector = tensor_np[:, x]
                text = f"Clicked: ({x}, {y}, {tensor_np[y, x]})\n"
                text += f"Row vector: {np.mean(row_vector)}/{np.std(row_vector)}\n"
                text += f"Column vector: {np.mean(col_vector)}/{np.std(col_vector)}"
                text_box.set_text(text)
                fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    plt.tight_layout()
    plt.show()

def visualize_tsne(tensor, name="Tensor", perplexity=30, n_iter=1000):
    """
    Visualize a tensor using t-SNE dimensionality reduction.
    
    Args:
    tensor (torch.Tensor or numpy.ndarray): The input tensor to visualize
    name (str): Name of the tensor for display purposes
    perplexity (float): The perplexity parameter for t-SNE (default: 30)
    n_iter (int): The number of iterations for t-SNE optimization (default: 1000)
    
    Returns:
    None
    """
    print(f"\n--- t-SNE Visualization for {name} ---")
    
    # Ensure the tensor is on CPU and convert to numpy
    if isinstance(tensor, torch.Tensor):
        tensor_np = tensor.detach().cpu().numpy()
    else:
        tensor_np = tensor
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    tsne_results = tsne.fit_transform(tensor_np)
    
    # Visualize the results
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=range(len(tsne_results)), 
                         cmap='viridis', alpha=0.7)
    
    ax.set_title(f't-SNE Visualization of {name}', fontsize=16, color='blue')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Data point index')
    
    # Add text box for displaying point information
    text_box = ax.text(0.02, 0.98, '', transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def on_click(event):
        if event.inaxes == ax:
            cont, ind = scatter.contains(event)
            if cont:
                index = ind['ind'][0]
                original_data = tensor_np[index]
                text = f"Point {index}\n"
                text += f"Original data: mean={np.mean(original_data):.4f}, std={np.std(original_data):.4f}\n"
                text += f"t-SNE coords: ({tsne_results[index, 0]:.4f}, {tsne_results[index, 1]:.4f})"
                text_box.set_text(text)
                fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    plt.tight_layout()
    plt.show()
    
    print(f"t-SNE visualization complete for {name}")

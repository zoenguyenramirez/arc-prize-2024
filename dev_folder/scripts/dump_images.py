import arckit
import arckit.vis as vis
import os

# Create a directory to store the images
os.makedirs("arc_images", exist_ok=True)

# Load the dataset
train_set, eval_set = arckit.load_data()

# Function to save task as image
def save_task_image(task, filename):
    task_vis = vis.draw_task(task, width=800, height=600)
    vis.output_drawing(task_vis, filename)

# Save training tasks
for i, task in enumerate(train_set):
    save_task_image(task, f"arc_images/train_{task.id}.png")
    print(f"Saved train task {i+1}/{len(train_set)}")

# Save evaluation tasks
for i, task in enumerate(eval_set):
    save_task_image(task, f"arc_images/eval_{task.id}.png")
    print(f"Saved eval task {i+1}/{len(eval_set)}")

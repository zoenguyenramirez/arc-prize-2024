import tensorflow as tf
import sys
import os
import tempfile
import subprocess
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def get_user_input():
    """
    Open the user's default text editor to get input.
    Returns the text entered by the user.
    """
    editor = os.environ.get('EDITOR', 'vi')  # Default to 'vi' if EDITOR is not set
    with tempfile.NamedTemporaryFile(suffix=".md", mode='w+') as tf:
        tf.write("# Enter your text here (Markdown format supported)\n\n")
        tf.flush()
        subprocess.call([editor, tf.name])
        tf.seek(0)
        return tf.read().strip()

def add_text_summary(log_dir, tag, text, step):
    """
    Add a text summary to an existing TensorBoard log directory.
    
    Args:
        log_dir (str): Path to the TensorBoard log directory.
        tag (str): Tag for the summary.
        text (str): Text content of the summary.
        step (int): Step number for the summary.
    """
    # Create a file writer
    writer = tf.summary.create_file_writer(log_dir)

    # Write the summary
    with writer.as_default():
        tf.summary.text(tag, text, step=step)
    
    # Close the writer to flush the summary
    writer.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python edit_summary.py <log_directory>")
        sys.exit(1)

    log_dir = sys.argv[1]
    
    # Get user input for the tag
    tag = 'Post Train'
    
    # Get user input for the text content
    print("Your default text editor will open. Enter your text in Markdown format.")
    text = get_user_input()
    
    # Get user input for the step number
    step = int(time.time())
    
    add_text_summary(log_dir, tag, text, step)
    print(f"Added text summary '{tag}' to the log directory.")

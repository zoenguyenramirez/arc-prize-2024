import difflib
import sys

def colorize(text, color):
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'reset': '\033[0m'
    }
    return f"{colors[color]}{text}{colors['reset']}"

def split_into_chunks(text):
    return text.replace(' ROW_SEPARATOR ', '\n').split('\n')

def compare_sequences(expected, generated):
    expected_chunks = split_into_chunks(expected)
    generated_chunks = split_into_chunks(generated)

    differ = difflib.Differ()
    diff = list(differ.compare(expected_chunks, generated_chunks))

    for line in diff:
        if line.startswith('  '):
            print(line[2:])
        elif line.startswith('- '):
            print(colorize(line[2:], 'red'))
        elif line.startswith('+ '):
            print(colorize(line[2:], 'green'))
        elif line.startswith('? '):
            print(colorize(line[2:].replace('\n', ''), 'yellow'))
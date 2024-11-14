import time
import os
import json

from shared import mySystem, get_task_count, merge_with_sample, sample_path

def soma_main(test_path):
    print(f'soma_main Start @ {time.strftime("%Y-%m-%d %H:%M:%S")}')    

    task_count = get_task_count()
    print('task_count', task_count)

    mySystem("mkdir -p soma; cp -r ../input/python/* ./soma")
    mySystem(f"cd soma; python soma.py --test_path {test_path} --range 0:{task_count - 1}")
    print(f'soma_main Done. @ {time.strftime("%Y-%m-%d %H:%M:%S")}')

if __name__ == "__main__":
    soma_main(os.path.abspath('../input/arc-prize-2024/arc-agi_test_challenges.json'))

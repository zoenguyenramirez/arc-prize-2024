import torch
import os
import json
import time

from shared import mySystem, merge_with_sample, get_task_count, model_path

def transformer_main(source):
    print(f'transformer_main Start @ {time.strftime("%Y-%m-%d %H:%M:%S")}')    

    task_count = get_task_count()
    print('task_count', task_count)

    device_count = max(1, torch.cuda.device_count())

    mySystem("rm -rf transformer; cp -r ../input/transformer ./transformer")

    mySystem(f"cd transformer; python3 safe_run.py --checkpoint-path {os.path.abspath(model_path)} --source {source} --maximum-task-count {task_count} --process-count {device_count}")
        
    print(f'transformer_main Done. @ {time.strftime("%Y-%m-%d %H:%M:%S")}')    

if __name__ == "__main__":
    
    source = 'arc-agi_test'
    test_path = os.path.abspath(f'../input/arc-prize-2024/{source}_challenges.json')

    transformer_main(source)

    with open('transformer/submission.json','r') as f:        
        transformer_result = json.load(f) 

    sample_path = '../input/arc-prize-2024/sample_submission.json'
    sub_dict = merge_with_sample(test_path, sample_path, transformer_result)

    with open('submission.json', 'w') as file:
        json.dump(sub_dict, file, indent=4)
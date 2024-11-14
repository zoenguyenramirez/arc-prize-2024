import json
import multiprocessing
import time
import os
import sys
import subprocess

from transformer import transformer_main
from soma import soma_main  # Assuming there's a soma_main function
from icecuber import ice_main  # Assuming there's an icecuber_main function
from shared import mySystem

def append_if_not_exist(attempts_dict, new_attempts, score):
    for idx, attempt in enumerate(new_attempts):
        # Convert attempt to tuple of tuples for hashability
        attempt_key = tuple(map(tuple, attempt))
        # Calculate weighted score based on position
        weighted_score = score * ((1 / 8) ** idx)
        # Add or accumulate the score
        attempts_dict[attempt_key] = attempts_dict.get(attempt_key, 0) + weighted_score

        print(f'+{weighted_score} ({score}[{idx}]) -> {attempts_dict[attempt_key]} for {attempt_key}')
    
    return attempts_dict

def build_top_2_attempts(t, i, *, soma, ice, transformer):
    attempts_dict = {}
    try:
        append_if_not_exist(attempts_dict, soma[t][i]['attempts'], 0.3)
    except:
        pass

    try:
        append_if_not_exist(attempts_dict, ice[t][i]['attempts'], 0.18)
    except:
        pass

    try:
        append_if_not_exist(attempts_dict, transformer[t][i]['attempts'], 0.22)
    except:
        pass

    sorted_items = sorted(attempts_dict.items(), key=lambda x: x[1], reverse=True)

    for index, item in enumerate(sorted_items):
        print(f'{t}[{i}][{index}]', item)

    # Convert back to list and sort by votes
    sorted_attempts = [list(map(list, k)) for k, v in 
                        sorted_items]

    if len(sorted_attempts) >= 2:
        answer = {'attempt_1': sorted_attempts[0], 'attempt_2': sorted_attempts[1]}
    elif len(sorted_attempts) >= 1:
        answer = {'attempt_1': sorted_attempts[0], 'attempt_2': [[0]]}
    else:
        answer = {'attempt_1': [[0]], 'attempt_2': [[0]]}

    return answer

def merge_all_with_sample(data_path, transformer_submission, soma_submission, ice_submission): 
    with open(data_path,'r') as f:
        data = json.load(f)
        sorted_items = sorted(data.items(), key=lambda x: str(x[1]))
        tasks_name = [item[0] for item in sorted_items]
        tasks_file = [item[1] for item in sorted_items]

    sub_solver = {}

    for n in range(len(tasks_name)):
        task = tasks_file[n]
        t = tasks_name[n]
        sub_solver[t] = []
            
        for i in range(len(task['test'])):
            answer = build_top_2_attempts(t, i, soma = soma_submission, ice = ice_submission, transformer = transformer_submission)

            sub_solver[t].append(answer)

    return sub_solver

def run_with_output_redirect(target, args, output_file):
    # Create a copy of the original stdout and stderr
    old_stdout = os.dup(sys.stdout.fileno())
    old_stderr = os.dup(sys.stderr.fileno())
    
    # Fork tee process
    tee_stdout = subprocess.Popen(
        ['tee', output_file],
        stdin=subprocess.PIPE,
        stdout=sys.stdout,
        text=True
    )
    
    try:
        # Redirect stdout and stderr to the tee process
        os.dup2(tee_stdout.stdin.fileno(), sys.stdout.fileno())
        os.dup2(tee_stdout.stdin.fileno(), sys.stderr.fileno())
        
        # Run the target function
        target(*args)
    finally:
        # Restore original stdout and stderr
        os.dup2(old_stdout, sys.stdout.fileno())
        os.dup2(old_stderr, sys.stderr.fileno())
        
        # Close duplicated file descriptors
        os.close(old_stdout)
        os.close(old_stderr)
        
        # Ensure tee process is terminated
        tee_stdout.stdin.close()
        tee_stdout.wait()

def ensemble_main():
    source = 'arc-agi_test'
    test_path = os.path.abspath(f'../input/arc-prize-2024/{source}_challenges.json')

    print(f'Start @ {time.strftime("%Y-%m-%d %H:%M:%S")}')
    
    # Create processes for each main function with tee output
    # processes = [
    #     multiprocessing.Process(
    #         target=run_with_output_redirect,
    #         args=(transformer_main, (test_path, source), 'transformer_output.log')
    #     ),
    #     multiprocessing.Process(
    #         target=run_with_output_redirect,
    #         args=(soma_main, (test_path,), 'soma_output.log')
    #     ),
    #     multiprocessing.Process(
    #         target=run_with_output_redirect,
    #         args=(ice_main, (test_path,), 'ice_output.log')
    #     )
    # ]
    
    processes = [
        multiprocessing.Process(
            target=transformer_main,
            args=(source, )
        ),
        multiprocessing.Process(
            target=soma_main,
            args=(test_path,)
        ),
        multiprocessing.Process(
            target=ice_main,
            args=(test_path,)
        )
    ]    
    # Start all processes
    for p in processes:
        p.start()
    
    print('All process started.')
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print(f'All process ended. @ {time.strftime("%Y-%m-%d %H:%M:%S")} All parallel processes completed, running ensemble...')

    with open('transformer/submission_candidates.json','r') as f:
        transformer_result = json.load(f) 

    with open('soma/submission_candidates.json', 'r') as file:
        soma_submission = json.load(file)

    with open('ice_submission_candidates.json', 'r') as file:
        ice_submission = json.load(file)

    submission = merge_all_with_sample(test_path, transformer_result, soma_submission, ice_submission)
    
    with open('submission.json', 'w') as file:
        json.dump(submission, file, indent=4)

    # otherwise, kaggle cannot find our answer!
    mySystem("tar -czf abs_store.tar.gz absres-c-files/store")
    mySystem("cd absres-c-files; find . -maxdepth 1 -type d -not -path . -exec rm -r {} +")

    mySystem("rm -r abstraction-and-reasoning-challenge")

    mySystem("tar -czf transformer_store.tar.gz transformer/store")
    mySystem("tar -czf transformer_submission.tar.gz transformer/submission")

    mySystem("cd transformer; find . -maxdepth 1 -type d -not -path . -exec rm -r {} +")

if __name__ == "__main__":
    ensemble_main()
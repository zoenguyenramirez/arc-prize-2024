import os
import json
import time
    
from shared import mySystem, merge_with_sample, get_task_count, is_in_slices_safe, sample_path

#######################################################################################
# Adapt ARC Prize 2024 files to work with Abstraction and Resoning Corpus 2020 rules ##
#######################################################################################

def adapt_2024_to_2020_rules(json_file_path, task_list_slices):
    # Load the JSON content
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Create the 'test' directory
    output_dir = '../working/abstraction-and-reasoning-challenge/test'  
    os.makedirs(output_dir, exist_ok=True)

    # Split the JSON content into individual files
    for task_index, (task_id, task_data) in enumerate(data.items()):

        if is_in_slices_safe(task_index, task_list_slices):
            output_file_path = os.path.join(output_dir, f'{task_id}.json')
            with open(output_file_path, 'w') as output_file:
                json.dump(task_data, output_file, indent=4)

############################################
# Beginning of icecuber's original solution#
##########################################

def icecuber_solution():
    if open("../input/arc-solution-source-files-by-icecuber/version.txt").read().strip() == "671838222":
        print("Dataset has correct version")
    else:
        print("Dataset version not matching!")
        assert(0)
        
    mySystem("cp -r ../input/arc-solution-source-files-by-icecuber ./absres-c-files")
    mySystem("cd absres-c-files; make -j")
    mySystem("cd absres-c-files; python3 safe_run.py")
    mySystem("cp absres-c-files/submission_part.csv old_submission.csv")

# Function to translate from old submission format (csv) to new one (json)
def translate_submission(file_path):
    # Read the original submission file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    submission_dict = {}

    for line in lines[1:]:  # Skip the header line
        output_id, output = line.strip().split(',')
        task_id, output_idx = output_id.split('_')
        predictions = output.split(' ')  # Split predictions based on ' '
        
        # # Take only the first two predictions
        # if len(predictions) > 2:
        #     predictions = predictions[:2]

        processed_predictions = []
        for pred in predictions:
            if pred:  # Check if pred is not an empty string
                pred_lines = pred.split('|')[1:-1]  # Remove empty strings from split
                pred_matrix = [list(map(int, line)) for line in pred_lines]
                processed_predictions.append(pred_matrix)

        attempt_dict = {
            "attempts": processed_predictions,
        }

        if task_id not in submission_dict:
            submission_dict[task_id] = []

        if output_idx == '0':
            submission_dict[task_id].insert(0, attempt_dict)
        else:
            submission_dict[task_id].append(attempt_dict)
    
    return submission_dict

def ice_main(test_path):
    print(f'ice_main Start @ {time.strftime("%Y-%m-%d %H:%M:%S")}')    

    task_count = get_task_count()
    print('task_count', task_count)

    adapt_2024_to_2020_rules(test_path, f'0:{task_count - 1}')
    icecuber_solution()
    sub_dict = translate_submission('./old_submission.csv')
    # sub_dict = merge_with_sample(test_path, sample_path, sub_dict)

    with open('ice_submission_candidates.json', 'w') as file:
        json.dump(sub_dict, file, indent=4)

    print(f'ice_main Done @ {time.strftime("%Y-%m-%d %H:%M:%S")}')    

if __name__ == "__main__":
    # TODO do not use "evaluation" at submission time
    ice_main(os.path.abspath('../input/arc-prize-2024/arc-agi_test_challenges.json'))

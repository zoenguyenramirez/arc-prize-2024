import json

with open('input_data/arc-agi_training_challenges.json') as f:
    training_challenges = json.load(f)

with open('input_data/arc-agi_evaluation_challenges.json') as f:
    evaluation_challenges = json.load(f)

with open('input_data/arc-agi_training_solutions.json') as f:
    training_solutions = json.load(f)

with open('input_data/arc-agi_evaluation_solutions.json') as f:
    evaluation_solutions = json.load(f)

with open('input_data/arc-agi_test_challenges.json') as f:
    test_challenges = json.load(f)

test_solutions = {}

for index, test_key in enumerate(test_challenges):
    if test_key in training_challenges:
        if training_challenges[test_key] == test_challenges[test_key]:
            print(f'[{index}] test_key matched in training')
            test_solutions[test_key] = training_solutions[test_key]
    elif test_key in evaluation_challenges:
        if evaluation_challenges[test_key] == test_challenges[test_key]:
            print(f'[{index}] test_key matched in evaluation')            

with open('arc-agi_test_solutions.json', 'w') as f:
    json.dump(test_solutions, f)
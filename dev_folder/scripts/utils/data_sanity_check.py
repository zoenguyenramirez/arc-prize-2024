import json

with open('input_data/arc-agi_training_challenges.json') as f:
    training_challenges = json.load(f)
    training_keys = training_challenges.keys()

with open('input_data/arc-agi_evaluation_challenges.json') as f:
    evaluation_challenges = json.load(f)
    evaluation_keys = evaluation_challenges.keys()

with open('input_data/arc-agi_test_challenges.json') as f:
    test_challenges = json.load(f)
    test_keys = test_challenges.keys()

# Convert keys to sets for easier comparison
training_set = set(training_keys)
evaluation_set = set(evaluation_keys)
test_set = set(test_keys)

# Get sizes
print(f"Number of training challenges: {len(training_set)}")
print(f"Number of evaluation challenges: {len(evaluation_set)}")
print(f"Number of test challenges: {len(test_set)}")

# Check for overlaps
train_eval_overlap = training_set.intersection(evaluation_set)
train_test_overlap = training_set.intersection(test_set)
eval_test_overlap = evaluation_set.intersection(test_set)

print("\nOverlaps between sets:")
print(f"Training & Evaluation overlap: {len(train_eval_overlap)} tasks")
print(f"Training & Test overlap: {len(train_test_overlap)} tasks")
print(f"Evaluation & Test overlap: {len(eval_test_overlap)} tasks")

# Check for unique entries
only_in_training = training_set - (evaluation_set | test_set)
only_in_evaluation = evaluation_set - (training_set | test_set)
only_in_test = test_set - (training_set | evaluation_set)

print("\nUnique entries:")
print(f"Only in Training: {len(only_in_training)} tasks")
print(f"Only in Evaluation: {len(only_in_evaluation)} tasks")
print(f"Only in Test: {len(only_in_test)} tasks")

# Check for tasks present in all three sets
common_to_all = training_set.intersection(evaluation_set, test_set)
print(f"\nTasks present in all three sets: {len(common_to_all)} tasks")

# Total unique tasks across all sets
all_tasks = training_set | evaluation_set | test_set
print(f"\nTotal unique tasks across all sets: {len(all_tasks)} tasks")
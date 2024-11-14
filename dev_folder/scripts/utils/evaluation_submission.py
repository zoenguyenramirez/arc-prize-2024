import json

test_case = 'test' # 'training' # 

# submission_file = '/home/nikola/Code/GenII/3-arc24/backup/evaluation/transformer/liweichen88_dataking_scriptVersionId=204715704(20241029_172457_c19_best).json'
submission_file = f'/home/nikola/Code/GenII/3-arc24/backup/{test_case}/transformer/kaggle_with_bug_submission.json'
# submission_file = '/home/nikola/Downloads/submission.json'
# submission_file = f'/home/nikola/Code/GenII/3-arc24/backup/{test_case}/soma/submission.json'
# submission_file = f'/home/nikola/Code/GenII/3-arc24/backup/{test_case}/ice/submission.json'

with open(submission_file) as f:
    submission_dict = json.load(f)

with open(f'input_data/arc-agi_{test_case}_solutions.json') as f:
    solutions_dict = json.load(f)

test_solutions = {}

tests_count = 0
has_answer_key = 0
score_in_attempt1 = 0
potential_attempt1_score = 0
score_in_attempt2 = 0
potential_attempt2_score = 0
score = 0
for index, test_key in enumerate(solutions_dict):
    tests_count += len(solutions_dict[test_key])
    if test_key in submission_dict:
        has_answer_key += 1
        for test_index, solution in enumerate(solutions_dict[test_key]):
            has_correct_anser = False
            possible_score = 1 / len(solutions_dict[test_key])
            if 'attempt_1' in submission_dict[test_key][test_index]:
                potential_attempt1_score += possible_score
                if submission_dict[test_key][test_index]['attempt_1'] == solution:
                    has_correct_anser = True
                    score_in_attempt1 += possible_score
            
            if 'attempt_2' in submission_dict[test_key][test_index]:
                potential_attempt2_score += possible_score
                if submission_dict[test_key][test_index]['attempt_2'] == solution:
                    assert has_correct_anser == False
                    has_correct_anser = True
                    score_in_attempt2 += possible_score

            if has_correct_anser:
                # print(f'correct: {test_key}[{test_index}]')
                score += possible_score

assert len(submission_dict) == has_answer_key

print(f'tests_count: {tests_count}, has_answer_key: {has_answer_key} {has_answer_key/len(solutions_dict)*100:.2f}% score: {score/len(solutions_dict)*100:.1f}% ({score}) score_in_attempt1:{score_in_attempt1/potential_attempt1_score*100:.1f}%, score_in_attempt2:{score_in_attempt2/potential_attempt2_score*100 if potential_attempt2_score > 0 else 0:.1f}%') # {score/len(submission_dict)*100:.1f}%, 
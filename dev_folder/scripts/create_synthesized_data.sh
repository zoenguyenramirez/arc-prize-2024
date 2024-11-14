#!/bin/bash

functions=("conditional_logic" 
    "array_indexing" 
    "arithmetic_operations"
    "modular_arithmetic"
    "find_min_max"
    "count_occurrences"
    "element_wise_operations"
    "array_manipulation")

num_tasks=3000

for func in "${functions[@]}"; do
	python -m src.synthesize_data --selected-functions "$func" --output-prefix "$func" --num-tasks $num_tasks
	
	python -m src.synthesize_data --selected-functions "$func" --output-prefix "${func}_test" --global-seed-offset 1
done

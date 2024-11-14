#!/bin/bash
source "$(dirname "$0")/utils/suppress_warnings.sh"

# Function to run the original loop
run_synthetic_loop() {
    # Define an array of data sources
    data_sources=(
        "conditional_logic" 
        "array_indexing" 
        "arithmetic_operations"
        "modular_arithmetic"
        "find_min_max"
        "count_occurrences"
        "element_wise_operations"
        "array_manipulation"
    )

    # Define the checkpoint path
    checkpoint_path="special_checkpoints/run_e32_l7_h4_b64_lr0.002_20240923_134643_6c15a8f/run_e32_l7_h4_b64_lr0.002_20240923_134643_6c15a8f_best.pth"

    # Loop through each data source
    for data_source in "${data_sources[@]}"; do
        echo "Processing data source: $data_source"
        python -m src.sample --checkpoint-path "$checkpoint_path" --data-source "synth_${data_source}_test"
        echo "Finished processing $data_source"
        echo "------------------------"
    done
}

# Function to run the conditional logic case
run_conditional_logic() {
    python -m src.sample --checkpoint-path special_checkpoints/run_e32_l7_h4_b64_lr0.002_20240922_194647/run_e32_l7_h4_b64_lr0.002_20240922_194647_best.pth --data-source synth_conditional_logic_test
}

# Function to run the array indexing case
run_array_indexing() {
    python -m src.sample --checkpoint-path special_checkpoints/run_e32_l7_h4_b64_lr0.002_20240922_210835_96e3060/run_e32_l7_h4_b64_lr0.002_20240922_210835_96e3060_best.pth --data-source synth_array_indexing_test
}

# Function to run the conditional logic case
run_arc() {
    checkpoint_path='cloud_runs/69.55.141.119/genesis/runs/genesis/20241026_150711_nogit_nobranch_lr2e-06_bl2e-06_ssu0_bs22_h4_es784_nl18_we10_as1_ph3_ac1_ad1_scosine_oadam_ge1_c43/Transformer_latest.pt'

    # python -m src.sample --checkpoint-path "$checkpoint_path" --data-source arc-agi_training --second-only # --verbose
    python -m src.sample --checkpoint-path "$checkpoint_path" --data-source arc-agi_evaluation # --second-only # --verbose
    # python -m src.sample --checkpoint-path $checkpoint_path --data-source arc-agi_test --second-only # --verbose
}

# Main execution logic
case "$1" in
    "synthetic")
        run_synthetic_loop
        ;;
    "conditional_logic")
        run_conditional_logic
        ;;
    "array_indexing")
        run_array_indexing
        ;;
    "arc")
        run_arc
        ;;
    *)
        echo "Usage: $0 {synthetic|conditional_logic|array_indexing|arc}"
        exit 1
        ;;
esac

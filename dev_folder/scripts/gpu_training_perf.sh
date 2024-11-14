#!/bin/bash

# Check if batch size is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <batch_size>"
    exit 1
fi

BATCH_SIZE=$1

echo "Preparing data..."
python -m src.prepare_data --data-sources arc-agi_training arc-agi_evaluation


echo "Starting training with batch size: $BATCH_SIZE"

time (
python -m src.train --epochs 5 --max-seq-length 2048 --samples-per-epoch 275 --runs-name perf --warmup-epochs 2 --heads 6 --num-layers 7 --batch-size $BATCH_SIZE --embed-size 780 --accumulation-steps 5 --minimize-checkpoints 
)

echo "Training completed and it was timed."
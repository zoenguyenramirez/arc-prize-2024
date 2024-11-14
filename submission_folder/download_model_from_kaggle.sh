#!/bin/bash

kaggle models instances versions download zoenguyenramirez/transformer_model/pyTorch/transformer_best_7738/1

mkdir -p input/transformer_model 

tar -xzf transformer_model.tar.gz -C input/transformer_model --transform='s/.*/Transformer_best.pt/'

# Ask for confirmation
read -p "Do you want to delete the tar.gz file? (y/N): " response

# Convert response to lowercase for case-insensitive comparison
response_lower=$(echo "$response" | tr '[:upper:]' '[:lower:]')

# Check the response
if [[ "$response_lower" == "y" || "$response_lower" == "yes" ]]; then
    rm transformer_model.tar.gz
    echo "tar.gz file deleted."
else
    echo "tar.gz file kept."
fi



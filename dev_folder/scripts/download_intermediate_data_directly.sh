#!/bin/bash

kaggle datasets download -d zoenguyenramirez/prepared-dataset-used-in-nikolas-training -p intermediate_data

unzip intermediate_data/prepared-dataset-used-in-nikolas-training.zip -d intermediate_data

# Ask for confirmation
read -p "Do you want to delete the zip file? (y/N): " response

# Convert response to lowercase for case-insensitive comparison
response_lower=$(echo "$response" | tr '[:upper:]' '[:lower:]')

# Check the response
if [[ "$response_lower" == "y" || "$response_lower" == "yes" ]]; then
    rm intermediate_data/prepared-dataset-used-in-nikolas-training.zip
    echo "Zip file deleted."
else
    echo "Zip file kept."
fi



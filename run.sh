#!/bin/bash

# Function to validate the script name, extract the part after "main_", and remove the .py extension
get_script_suffix() {
    if [[ $1 == main_* ]]; then
        local script_name="${1#main_}"  # Extract part after "main_"
        echo "${script_name%.py}"  # Remove the .py extension if present
    else
        echo "The script name must start with 'main_'."
        exit 1
    fi
}

# Check if the script name is provided as a parameter
if [ -z "$1" ]; then
    echo "Usage: $0 <python_script_name>"
    exit 1
fi

# Validate and set the script name, extract suffix for output file
python_script="$1"
script_suffix=$(get_script_suffix "$1")

# Generate a timestamped output file name using the script suffix
current_date=$(date +"%Y-%m-%d_%H-%M-%S")
output_file="output_${script_suffix}_${current_date}.txt"

# Run the Python script with nohup and direct output to a file
nohup stdbuf -oL python3 -u "$python_script" --cuda > "outputs/$output_file" 2>&1 &

echo "Script $python_script is running in background with output redirected to outputs/$output_file"


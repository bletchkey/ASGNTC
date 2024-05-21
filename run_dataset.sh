#!/bin/bash

current_date=$(date +"%Y-%m-%d_%H-%M-%S")
output_file="output_dataset_$current_date.txt"
nohup stdbuf -oL python3 -u main_dataset.py --cuda > "outputs/$output_file" 2>&1 &


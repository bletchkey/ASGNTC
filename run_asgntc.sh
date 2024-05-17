#!/bin/bash

current_date=$(date +"%Y-%m-%d_%H-%M-%S")
output_file="output_asgntc_$current_date.txt"

nohup stdbuf -oL python3 -u main_asgntc.py --cuda > "$output_file" 2>&1 &


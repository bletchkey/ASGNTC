#!/bin/bash

nohup stdbuf -oL python3 -u main_asgntc.py --cuda > output.txt 2>&1 &


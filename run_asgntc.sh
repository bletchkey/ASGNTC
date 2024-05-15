#!/bin/bash

nohup stdbuf -oL python3 -u main.py --cuda > output.txt 2>&1 &


#!/bin/bash

nohup stdbuf -oL python3 -u main_predictors.py --cuda > output.txt 2>&1 &


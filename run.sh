#!/bin/bash

# Concatenates all args to a single variable
pyargs="$*"

module load cuda/8.0

cd src/
python -u captionSentences.py $pyargs

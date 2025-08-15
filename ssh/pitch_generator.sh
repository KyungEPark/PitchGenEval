#!/bin/bash

# Example arguments
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
DEVICE="mps"
BATCH_SIZE=1
MAX_NEW_TOKENS=128
TEMPERATURE=0.0
# DO_SAMPLE="--do_sample"  # Set to "" to disable sampling

python codes/pitch_generation.py \
    --model_name "$MODEL_NAME" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE"
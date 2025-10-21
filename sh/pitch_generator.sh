#!/bin/bash
#SBATCH --partition=topml
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --gres=gpu:3
#SBATCH --time=06:00:00
#SBATCH --mem=50gb
#SBATCH --output=/lustre/project/ki-topml/minbui/repos/PitchGenEval/sh/output/test_%j.log


BATCH_SIZE=128
MAX_NEW_TOKENS=512
TEMPERATURE=0.0
# DO_SAMPLE="--do_sample"  # Set to "" to disable sampling

source /lustre/project/ki-topml/minbui/.bashrc
conda_initialize
micromamba activate hatespeech

cd /lustre/project/ki-topml/minbui/repos/PitchGenEval

MODELS=(
"/lustre/project/ki-topml/minbui/projects/models/sync/models--Qwen--Qwen2.5-7B-Instruct/snapshots/Qwen2.5-7B-Instruct"
"/lustre/project/ki-topml/minbui/projects/models/sync/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/Llama-3.1-8B-Instruct"
"/lustre/project/ki-topml/minbui/projects/models/Llama-3.3-70B-Instruct"
"/lustre/project/ki-topml/minbui/projects/models/qwen_2.5_72b_chat"
)

for MODEL_NAME in "${MODELS[@]}"; do
  echo "Running model: $MODEL_NAME"
  python codes/pitch_generation.py \
      --model_name "$MODEL_NAME" \
      --batch_size "$BATCH_SIZE" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --temperature "$TEMPERATURE"
done

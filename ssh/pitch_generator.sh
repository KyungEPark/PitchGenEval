#!/bin/bash
#SBATCH --account=westai0091           # Account details
#SBATCH --nodes=1                        # Number of compute nodes required
#SBATCH --ntasks-per-node=1              # Number of tasks per node
#SBATCH --gres=gpu:1                  
#SBATCH --time=05:00:00                  # Maximum runtime
#SBATCH --mem=50gb
#SBATCH --partition=dc-hwai
#SBATCH --output /p/project/westai0091/venturebias/logs/slurm-%j.out


now=$(date +"%T")
echo "Program starts:  $now"


source /p/project/westai0091/.bashrc
eval "$(micromamba shell hook --shell bash)"
micromamba activate stat
which python
python -c "import transformers; print('Transformers version:', transformers.__version__)"
cd /p/project/westai0091/venturebias

MODEL_NAME="gpt-oss-20b"
BATCH_SIZE=10
MAX_NEW_TOKENS=5000
TEMPERATURE=0.0
# DO_SAMPLE="--do_sample"  # Set to "" to disable sampling

echo "Model: ${MODEL_NAME}"

python codes/pitch_generation.py \
    --model_name "$MODEL_NAME" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE"


end=$(date +"%T")
echo "Completed: $end"

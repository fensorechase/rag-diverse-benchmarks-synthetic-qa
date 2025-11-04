#!/bin/bash
#SBATCH --job-name=ana_gen_baseline
#SBATCH --output=ana_gen_baseline
#SBATCH --gres=gpu:1
#SBATCH --mem=80GB

# Load any required modules
# module load python/3.8 cuda/11.7  # Adjust versions as needed

export TRANSFORMERS_CACHE="/local/scratch/.../huggingface"
#export HF_HOME="/local/scratch/.../hf_cache"
export HF_HOME="/local/scratch/.../huggingface"
export HF_DATASETS_CACHE="/local/scratch/.../hf_datasets"
export PIP_CACHE_DIR="/local/scratch/.../pip_cache"
export TORCH_HOME="/local/scratch/.../torch_cache"
export PYTORCH_HOME="/local/scratch/.../torch"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set OMP_NUM_THREADS to avoid CPU oversubscription
export OMP_NUM_THREADS=8

#CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Set up environment
export PYTHONPATH=$(pwd):$PYTHONPATH


source ./venv/bin/activate
pwd

# Source the .env file to load environment variables
source .env

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "HF_TOKEN is not set in .env file. Please add your Hugging Face token."
    exit 1
fi


# Check if the index exists by reading from config
INDEX_PATH=$(python3 -c "from config import BM25_INDEX_PATH; print(BM25_INDEX_PATH)")
echo "Using index at: $INDEX_PATH"

if [ ! -d "$INDEX_PATH" ]; then
    echo "Warning: Index directory not found at $INDEX_PATH"
    echo "Make sure your index path is correct in config.py"
fi

# Run the script
# Compute all metrics and generate tables
python compute_rag_metrics.py

python analyze_rq3_interactions.py


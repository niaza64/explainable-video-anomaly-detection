#!/bin/bash
# Run ONCE on the TCSC login node before submitting the SLURM job.
# Usage: bash /scratch/svc_td_ppml/qrx527/niaz_research/code/setup_env.sh

set -e

RESEARCH=/scratch/svc_td_ppml/qrx527/niaz_research
export HF_HOME=$RESEARCH/.hf_cache

echo "=== Setting up Python environment ==="
module load Python/3.11.3-GCCcore-12.3.0

python -m venv ~/venvs/vad_env
source ~/venvs/vad_env/bin/activate

pip install --upgrade pip --quiet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
pip install "transformers>=4.45.0" "accelerate>=0.30.0" peft qwen-vl-utils openai tqdm numpy --quiet

echo "=== Done. Activate with: source ~/venvs/vad_env/bin/activate ==="
python -c "import torch; print('PyTorch:', torch.__version__)"

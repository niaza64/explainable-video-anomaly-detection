#!/bin/bash
#SBATCH --job-name=qwen_finetune_vad
#SBATCH --partition=large
#SBATCH --gres=gpu:nvidia_h200_4g.71gb:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/svc_td_ppml/qrx527/niaz_research/logs/qwen_%j.log
#SBATCH --error=/scratch/svc_td_ppml/qrx527/niaz_research/logs/qwen_%j.err

set -e

RESEARCH=/scratch/svc_td_ppml/qrx527/niaz_research
export HF_HOME=$RESEARCH/.hf_cache
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY in your environment}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== Job started: $(date) ==="
echo "Node: $SLURMD_NODENAME"
nvidia-smi | head -15

module load Python/3.11.3-GCCcore-12.3.0
source ~/venvs/vad_env/bin/activate

mkdir -p $RESEARCH/{logs,tcsc_finetune_results}

echo "=== Running Qwen3-VL-32B + LoRA inference ==="
python $RESEARCH/code/run_qwen_inference.py \
    --rtfm-outputs  $RESEARCH/rtfm_outputs \
    --lora-dir      $RESEARCH/models/qwen3vl32_rtfm_lora \
    --annotations   $RESEARCH/annotations.json \
    --results-dir   $RESEARCH/tcsc_finetune_results

echo "=== Job finished: $(date) ==="

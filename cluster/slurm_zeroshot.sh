#!/bin/bash
#SBATCH --job-name=qwen_zeroshot_vad
#SBATCH --partition=large
#SBATCH --gres=gpu:nvidia_h200_4g.71gb:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/svc_td_ppml/qrx527/niaz_research/logs/qwen_zeroshot_%j.log
#SBATCH --error=/scratch/svc_td_ppml/qrx527/niaz_research/logs/qwen_zeroshot_%j.err

set -e

RESEARCH=/scratch/svc_td_ppml/qrx527/niaz_research
export HF_HOME=$RESEARCH/.hf_cache
# Reuse the OpenAI key already present in slurm_qwen.sh (judge step) without
# re-pasting the secret. NOTE: that key is hardcoded in plaintext there —
# rotate it and move to a secret file when you get a chance.
export OPENAI_API_KEY=$(grep -m1 'OPENAI_API_KEY=' $RESEARCH/code/slurm_qwen.sh | sed -E 's/.*OPENAI_API_KEY="?([^"]+)"?.*/\1/')
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== Job started: $(date) ==="
echo "Node: $SLURMD_NODENAME"
nvidia-smi | head -15

module load Python/3.11.3-GCCcore-12.3.0
source ~/venvs/vad_env/bin/activate

mkdir -p $RESEARCH/{logs,tcsc_zeroshot_results}

echo "=== Running Qwen3-VL-32B ZERO-SHOT (no LoRA) inference on the full rtfm_outputs set ==="
python $RESEARCH/code/run_qwen_zeroshot_inference.py \
    --rtfm-outputs  $RESEARCH/rtfm_outputs \
    --annotations   $RESEARCH/annotations.json \
    --results-dir   $RESEARCH/tcsc_zeroshot_results

echo "=== Job finished: $(date) ==="
echo "Summary: $RESEARCH/tcsc_zeroshot_results/qwen_zeroshot_judge_summary.json"

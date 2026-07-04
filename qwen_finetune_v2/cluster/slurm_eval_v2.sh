#!/bin/bash
#SBATCH --job-name=qwen_eval_v2
#SBATCH --partition=large
#SBATCH --gres=gpu:nvidia_h200_4g.71gb:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/svc_td_ppml/qrx527/niaz_research_v2/logs/eval_v2_%j.log
#SBATCH --error=/scratch/svc_td_ppml/qrx527/niaz_research_v2/logs/eval_v2_%j.err

set -e

V1ROOT=/scratch/svc_td_ppml/qrx527/niaz_research
V2ROOT=/scratch/svc_td_ppml/qrx527/niaz_research_v2
export HF_HOME=$V1ROOT/.hf_cache
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY in your environment}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== Job started: $(date) ==="
nvidia-smi | head -15

module load Python/3.11.3-GCCcore-12.3.0
source ~/venvs/vad_env/bin/activate     # reuse the v1 inference venv

mkdir -p $V2ROOT/{logs,tcsc_v2_results}

echo "=== Running Qwen3-VL-32B + LoRA v2 inference ==="
python $V2ROOT/code/run_qwen_inference_v2.py \
    --rtfm-outputs  $V1ROOT/rtfm_outputs \
    --lora-dir      $V2ROOT/models/qwen3vl32_rtfm_lora_v2 \
    --annotations   $V1ROOT/annotations.json \
    --results-dir   $V2ROOT/tcsc_v2_results

echo "=== Job finished: $(date) ==="

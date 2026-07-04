#!/bin/bash
#SBATCH --job-name=qwen_eval_v3
#SBATCH --partition=large
#SBATCH --gres=gpu:nvidia_h200_4g.71gb:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/svc_td_ppml/qrx527/niaz_research_v3_enriched_labels_single_strategy_with_normal_negatives/logs/eval_v3_%j.log
#SBATCH --error=/scratch/svc_td_ppml/qrx527/niaz_research_v3_enriched_labels_single_strategy_with_normal_negatives/logs/eval_v3_%j.err

set -e
V1ROOT=/scratch/svc_td_ppml/qrx527/niaz_research
V3ROOT=/scratch/svc_td_ppml/qrx527/niaz_research_v3_enriched_labels_single_strategy_with_normal_negatives
export HF_HOME=$V1ROOT/.hf_cache
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY in your environment}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== Job started: $(date) ==="
nvidia-smi | head -15

module load Python/3.11.3-GCCcore-12.3.0
source ~/venvs/vad_env/bin/activate    # reuse v1 inference venv (transformers, peft, openai)

mkdir -p $V3ROOT/{logs,tcsc_v3_results}

echo "=== Running Qwen3-VL-32B + LoRA v3 inference ==="
python $V3ROOT/code/run_qwen_inference_v3.py \
    --rtfm-outputs  $V1ROOT/rtfm_outputs \
    --lora-dir      $V3ROOT/models/qwen3vl32_rtfm_lora_v3 \
    --annotations   $V1ROOT/annotations.json \
    --results-dir   $V3ROOT/tcsc_v3_results

echo "=== Job finished: $(date) ==="

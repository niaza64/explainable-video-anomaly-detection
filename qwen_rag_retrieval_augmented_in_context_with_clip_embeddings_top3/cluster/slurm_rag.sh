#!/bin/bash
#SBATCH --job-name=qwen_rag_clip
#SBATCH --partition=large
#SBATCH --gres=gpu:nvidia_h200_4g.71gb:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/svc_td_ppml/qrx527/niaz_research_rag_in_context_with_clip_embeddings/logs/rag_%j.log
#SBATCH --error=/scratch/svc_td_ppml/qrx527/niaz_research_rag_in_context_with_clip_embeddings/logs/rag_%j.err

set -e

V1ROOT=/scratch/svc_td_ppml/qrx527/niaz_research
RAGROOT=/scratch/svc_td_ppml/qrx527/niaz_research_rag_in_context_with_clip_embeddings
export HF_HOME=$V1ROOT/.hf_cache
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY in your environment}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== Job started: $(date) ==="
nvidia-smi | head -15

module load Python/3.11.3-GCCcore-12.3.0
source ~/venvs/vad_env/bin/activate    # reuse v1 inference venv

# Ensure open_clip is installed
python -c "import open_clip" 2>/dev/null || pip install open_clip_torch --quiet

mkdir -p $RAGROOT/{logs,tcsc_rag_results}

echo "=== Running Qwen3-VL-32B + retrieval-augmented in-context inference (CLIP top-3) ==="
python $RAGROOT/code/run_qwen_rag_inference.py \
    --rtfm-outputs  $V1ROOT/rtfm_outputs \
    --annotations   $V1ROOT/annotations.json \
    --results-dir   $RAGROOT/tcsc_rag_results

echo "=== Job finished: $(date) ==="
